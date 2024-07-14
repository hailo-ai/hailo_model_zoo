#!/usr/bin/env python

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_file, download_from_drive, download_to_file

CALIB_SET_OFFSET = 999
CALIB_SET_LENGTH = 1024
DOWNLOAD_URL = {
    "calib": "https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download",
    "val": "https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download",
    "test": "https://drive.google.com/uc?id=1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T&export=download",
    "annotations": "https://drive.google.com/uc?export=download&id=1sAl2oml7hK6aZRdgRjqQJsjV5CEr7nl4",
    "hard_gt": "https://github.com/biubug6/Pytorch_Retinaface/raw/master/widerface_evaluate/ground_truth/wider_hard_val.mat",
}

FILENAME = {
    "calib": "WIDER_train.zip",
    "val": "WIDER_val.zip",
    "annotations": "wider_face_split.zip",
}

IMAGE_DIRECTORY_NAME = {
    "calib": "WIDER_train/images",
    "val": "WIDER_val/images",
}

GT_MAT_NAME = {
    "calib": "wider_face_train.mat",
    "val": "wider_face_val.mat",
}

CALIB_LOCATION = "models_files/widerface/2022-06-14/widerfacecalibration_set.tfrecord"
VAL_LOCATION = "models_files/widerface/2020-03-23/widerfaceval.tfrecord"
TFRECORD_LOCATION = {"calib": CALIB_LOCATION, "val": VAL_LOCATION}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_gt_boxes(gt_mat_path, hard_mat_path):
    gt_mat = loadmat(gt_mat_path)

    facebox_list = gt_mat["face_bbx_list"]
    event_list = gt_mat["event_list"]
    file_list = gt_mat["file_list"]

    if hard_mat_path is not None:
        hard_mat = loadmat(hard_mat_path)
        hard_gt_list = hard_mat["gt_list"]
    else:
        hard_gt_list = None

    return facebox_list, event_list, file_list, hard_gt_list


def _create_tfrecord(gt_mat_path, hard_mat_path, dataset_dir, name, is_calibration):
    """Loop over all the images in filenames and create the TFRecord"""
    hard_mat_path = str(hard_mat_path)
    gt_mat_path = str(gt_mat_path)

    tfrecords_filename = dataset_dir / f"widerface{name}.tfrecord"
    all_file_paths = sorted(Path(dataset_dir).glob("**/*.jpg"))

    # NOTE: In case we deal with the training set, the hard `.mat` used for validation is ignored.
    # This turns evaluation on the training set into not meanignful
    if "train" in gt_mat_path:
        hard_mat_path = None
    facebox_list, event_list, file_list, hard_gt_list = get_gt_boxes(gt_mat_path, hard_mat_path)

    if is_calibration:
        all_file_paths = all_file_paths[CALIB_SET_OFFSET : CALIB_SET_OFFSET + CALIB_SET_LENGTH]
    progress_bar = tqdm(all_file_paths)
    file_paths_iterator = enumerate(progress_bar)
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in file_paths_iterator:
            xmin, xmax, ymin, ymax, category_id = [], [], [], [], []
            with img_path.open("rb") as image_file:
                img_jpeg = image_file.read()
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]

            img_name = img_path.with_suffix("").name
            wider_category_name = img_path.parent.name
            event_id = None
            for event_index in range(event_list.shape[0]):
                if event_list[:, 0][event_index][0] == wider_category_name:
                    event_id = event_index

            assert event_id is not None

            image_id = None

            for image_index in range(file_list[event_id][0].shape[0]):
                if file_list[event_id][0][image_index][0][0] == img_name:
                    image_id = image_index

            assert image_id is not None

            gt_boxes = facebox_list[event_id][0][image_id][0]
            wider_hard_keep_index = [0] * 1024
            if hard_gt_list is not None:
                if hard_gt_list[event_id][0][image_id][0].shape[0] != 0:
                    wider_hard_keep_index = list(hard_gt_list[event_id][0][image_id][0].squeeze(axis=1))
                    wider_hard_keep_index += [0] * (1024 - len(wider_hard_keep_index))

            for object_annotations in gt_boxes:
                (x, y, width, height) = object_annotations
                if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                    continue
                xmin.append(float(x) / image_width)
                xmax.append(float(x + width) / image_width)
                ymin.append(float(y) / image_height)
                ymax.append(float(y + height) / image_height)
                category_id.append(1)  # All objects are faces

            progress_bar.set_description(f"#{i}: {img_name}")
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "num_boxes": _int64_feature(len(gt_boxes)),
                        "wider_hard_keep_index": _int64_feature(wider_hard_keep_index),
                        "image_id": _int64_feature(i),
                        "xmin": _float_list_feature(xmin),
                        "xmax": _float_list_feature(xmax),
                        "ymin": _float_list_feature(ymin),
                        "ymax": _float_list_feature(ymax),
                        "category_id": _int64_feature(category_id),
                        "image_name": _bytes_feature(str.encode(f"{wider_category_name}/{img_name}")),
                        "image_jpeg": _bytes_feature(img_jpeg),
                    }
                )
            )
            writer.write(example.SerializeToString())
    images_num = i + 1
    print("Done converting {} images".format(images_num))

    return tfrecords_filename


def download_dataset(type, images_dir, gt_mat_path, hard_mat_path):
    dataset_directory = Path(images_dir)
    dataset_root = gt_mat_path.parent.parent

    if not dataset_directory.is_dir():
        print(f"Image directory not found in {dataset_directory}. Downloading...")
        with tempfile.NamedTemporaryFile() as outfile:
            download_from_drive(DOWNLOAD_URL[type], outfile, desc=FILENAME[type])

            with zipfile.ZipFile(outfile, "r") as zip_ref:
                zip_ref.extractall(str(dataset_root))

    if not gt_mat_path.is_file():
        print(f"Ground truth not found in {gt_mat_path}. Downloading...")
        with tempfile.NamedTemporaryFile() as outfile:
            download_to_file(DOWNLOAD_URL["annotations"], outfile, desc=FILENAME["annotations"])

            with zipfile.ZipFile(outfile, "r") as zip_ref:
                zip_ref.extractall(str(dataset_root))

    if not hard_mat_path.is_file():
        print(f"Ground truth (hard set) not found in {hard_mat_path}. Downloading...")
        download_file(DOWNLOAD_URL["hard_gt"], hard_mat_path)


def run(type, dataset_dir, gt_mat_path, hard_mat_path):
    tfrecord_path = path_resolver.resolve_data_path(TFRECORD_LOCATION[type])
    hard_mat_path = hard_mat_path / "wider_hard_val.mat"
    if tfrecord_path.exists():
        print(f"tfrecord already exists at {tfrecord_path}. Skipping...")
        return

    print(f"Creating {type} set...")
    images_directory = dataset_dir / IMAGE_DIRECTORY_NAME[type]
    train_gt_mat_path = gt_mat_path / GT_MAT_NAME[type]
    download_dataset(type, images_directory, train_gt_mat_path, hard_mat_path)
    is_calibration = True if type == "calib" else False
    result_tfrecord_path = _create_tfrecord(
        train_gt_mat_path, hard_mat_path, images_directory, name="calibration_set", is_calibration=is_calibration
    )
    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result_tfrecord_path, tfrecord_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="TFRecord of which dataset to create", type=str, choices=["calib", "val"])
    parser.add_argument("--img", "-img", help="images directory", type=Path, default="./widerface")
    parser.add_argument(
        "--gt_mat_path",
        "-gt",
        type=Path,
        default="./widerface/wider_face_split/",
        help="Path of gt `.mat` file. "
        "See https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate",
    )
    parser.add_argument(
        "--hard_mat_path",
        "-hard",
        type=Path,
        default="./widerface/wider_face_split/",
        help="Path of HARD gt `.mat` file. "
        "See https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate",
    )
    args = parser.parse_args()
    run(args.type, args.img, args.gt_mat_path, args.hard_mat_path)
