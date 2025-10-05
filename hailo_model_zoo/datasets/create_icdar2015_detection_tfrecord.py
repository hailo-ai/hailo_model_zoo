#!/usr/bin/env python

import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import downloader, path_resolver

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "models_files/icdar2015/2025-07-14/icdar15_val_set.tfrecord",
    "calib": "models_files/icdar2015/2025-07-14/icdar15_calib_set.tfrecord",
}

ICDAR2015_PATH = "https://www.kaggle.com/api/v1/datasets/download/bestofbests9/icdar2015"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_anns(gt_path, h, w):
    xmin, xmax, ymin, ymax, tags, area = [], [], [], [], [], []
    with open(gt_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        gt = line.strip().split(",")
        if gt[-1] == "###":
            continue  # skip ignore boxes

        box = [(int(gt[i]), int(gt[i + 1])) for i in range(0, 8, 2)]
        x_min = float(min([p[0] for p in box])) / w
        x_max = float(max([p[0] for p in box])) / w
        y_min = float(min([p[1] for p in box])) / h
        y_max = float(max([p[1] for p in box])) / h

        xmin.append(x_min)
        xmax.append(x_max)
        ymin.append(y_min)
        ymax.append(y_max)
        tags.append(gt[-1])
        area.append((x_max - x_min) * (y_max - y_min))

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "tags": tags,
        "area": area,
        "is_crowd": [0] * len(xmin),
        "num_boxes": len(xmin),
    }


def get_img_anns_dirs(dataset_dir, dataset_type):
    key = "training" if dataset_type == "calib" else "test"
    images_dir = dataset_dir / f"ch4_{key}_images"
    annotations_dir = dataset_dir / f"ch4_{key}_localization_transcription_gt"
    return images_dir, annotations_dir


def download_dataset(dataset_dir):
    if dataset_dir is None or dataset_dir == "":
        dataset_dir = Path.cwd()
        dataset_dir = dataset_dir / "icdar2015_raw"
    else:
        dataset_dir = Path(dataset_dir)

    if dataset_dir.is_dir():
        print("ICDAR2015 dataset already exists at {}".format(dataset_dir))
    else:
        print("Downloading ICDAR2015 dataset to {}".format(dataset_dir))
        with tempfile.NamedTemporaryFile() as outfile:
            downloader.download_to_file(ICDAR2015_PATH, outfile)
            with zipfile.ZipFile(outfile, "r") as zip_ref:
                zip_ref.extractall(str(dataset_dir))

    return dataset_dir


def create_tfrecord(dataset_dir, dataset_type, num_images):
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[dataset_type])
    if tfrecords_filename.exists():
        print(f"tfrecord already exists at {tfrecords_filename}. Skipping...")
        return 0

    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    images_dir, annotations_dir = get_img_anns_dirs(dataset_dir, dataset_type)
    image_files = list(images_dir.glob("*.jpg"))

    progress_bar = tqdm(image_files[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in enumerate(progress_bar):
            progress_bar.set_description(f"{dataset_type} #{i+1}: {img_path}")

            ann_name = "gt_" + img_path.stem.split(".")[0] + ".txt"
            ann_path = annotations_dir / ann_name

            img_jpeg = open(img_path, "rb").read()

            img = cv2.imread(str(img_path))
            height, width, _ = img.shape
            annotations = get_anns(ann_path, height, width)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "num_boxes": _int64_feature(annotations["num_boxes"]),
                        "xmin": _float_list_feature(annotations["xmin"]),
                        "xmax": _float_list_feature(annotations["xmax"]),
                        "ymin": _float_list_feature(annotations["ymin"]),
                        "ymax": _float_list_feature(annotations["ymax"]),
                        "area": _float_list_feature(annotations["area"]),
                        "image_jpeg": _bytes_feature(img_jpeg),
                        "image_name": _bytes_feature(str.encode(img_path.name)),
                        "image_id": _int64_feature(int(img_path.stem.split("_")[1])),
                        "is_crowd": _float_list_feature(annotations["is_crowd"]),
                        "category_id": _float_list_feature([0] * annotations["num_boxes"]),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def run(dataset_dir, dataset_type, num_images):
    dataset_dir = download_dataset(dataset_dir)
    images_num = create_tfrecord(dataset_dir, dataset_type, num_images)
    if images_num > 0:
        print(f"\nDone converting {images_num} images")
        print(f"Dataset saved at {TF_RECORD_LOC[dataset_type]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE), choices=["calib", "val"])
    parser.add_argument(
        "--dataset-dir", type=str, default="icdar2015_raw", help="Path to ICDAR2015 dataset base directory"
    )
    parser.add_argument("--num-images", type=int, default=8192, help="Limit num images")
    args = parser.parse_args()
    run(args.dataset_dir, args.type, args.num_images)
