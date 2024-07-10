#!/usr/bin/env python

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import transform as trans
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_file

CALIB_LOCATION = "models_files/arcface_lfw/2022-12-12/arcface_lfw_pairs_calib.tfrecord"
VAL_LOCATION = "models_files/arcface_lfw/2022-12-12/arcface_lfw_pairs_val.tfrecord"
TFRECORD_LOCATION = {"calib": CALIB_LOCATION, "val": VAL_LOCATION}
CALIB_PAIR_COUNT = 128

TGZ_PATH = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
PAIRS_PATH = "http://vis-www.cs.umass.edu/lfw/pairs.txt"

DEFAULT_KPS_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/FaceRecognition/data/lfw/2022-08-30/lfw.json"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _get_jpg_path(lfw_dir, person_name, image_num):
    return os.path.join(lfw_dir, person_name, "{}_{}.jpg".format(person_name, image_num.zfill(4)))


def _get_paths(lfw_dir, pairs):
    path_list, issame_list = [], []
    for pair_num, line in enumerate(pairs):
        values = line.strip().split("\t")
        if len(values) == 3:
            path0 = _get_jpg_path(lfw_dir, values[0], values[1])
            path1 = _get_jpg_path(lfw_dir, values[0], values[2])
            is_same = True
        else:
            path0 = _get_jpg_path(lfw_dir, values[0], values[1])
            path1 = _get_jpg_path(lfw_dir, values[2], values[3])
            is_same = False
        print(pair_num, is_same)

        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append((path0, path1))
            issame_list.append(is_same)
        else:
            raise FileNotFoundError("One of the files:\n{}\n{}\n not found".format(path0, path1))
    return path_list, issame_list


def get_example(img_path, gt, pair_idx):
    img = np.array(Image.open(img_path), np.uint8)
    filename = str.encode("{:<20}".format(os.path.basename(img_path)))
    height = img.shape[0]
    width = img.shape[1]
    img = img.tostring()
    gt = int(gt)
    pair_idx = int(pair_idx)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "height": _int64_feature(height),
                "width": _int64_feature(width),
                "image_name": _bytes_feature(filename),
                "image": _bytes_feature(img),
                "pair_index": _int64_feature(pair_idx),
                "is_same": _int64_feature(gt),
            }
        )
    )
    return example


def _convert_dataset(img_list, gt_list, dataset_dir, is_calibration=False):
    tfrecords_filename = os.path.join(
        dataset_dir, "arcface_lfw_pairs_{}.tfrecord".format("calib" if is_calibration else "val")
    )
    image_label_pair = list(zip(img_list, gt_list))
    if is_calibration:
        image_label_pair = image_label_pair[:CALIB_PAIR_COUNT]
    progress_bar = tqdm(image_label_pair)
    with tf.io.TFRecordWriter(tfrecords_filename) as writer:
        for i, (img_path, gt) in enumerate(progress_bar):
            img_path0, img_path1 = img_path
            example0 = get_example(img_path0, gt, i)
            example1 = get_example(img_path1, gt, i)
            progress_bar.set_description(f"{i}: {img_path0}")
            writer.write(example0.SerializeToString())
            writer.write(example1.SerializeToString())
    pairs_processed = i + 1
    print("Done converting {} images".format(pairs_processed * 2))
    return tfrecords_filename


def download_annotatons(pairs_path):
    file_path = Path(pairs_path)
    if file_path.exists():
        return

    download_file(PAIRS_PATH, file_path)


def filter_lfw(members):
    length = len("lfw/")
    for member in members:
        if member.path.startswith("lfw/"):
            member.path = member.path[length:]
            yield member


def download_dataset(images_dir, tgz_file):
    images_dir = Path(images_dir)
    if images_dir.is_dir():
        return

    images_dir.mkdir(parents=True)

    tgz_path = Path(tgz_file)
    if not tgz_path.is_file():
        download_file(TGZ_PATH, tgz_path)

    with tarfile.open(tgz_path, "r:gz") as tar_ref:
        members = filter_lfw(tar_ref.getmembers())
        tar_ref.extractall(images_dir, members=members)


ARCFACE_SOURCE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
    else:
        ratio = float(image_size) / 128.0
    dst = ARCFACE_SOURCE * ratio
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return Image.fromarray(warped)


def align_dataset(args):
    if args.no_align:
        return args.img

    keypoints_path = Path(args.kps)
    if not keypoints_path.exists():
        download_file(args.kps_url, keypoints_path)

    images_path = Path(args.img)
    aligned_path = Path(args.img + "_aligned")
    aligned_path.mkdir(parents=True, exist_ok=True)
    with open(keypoints_path) as f:
        keypoints = json.load(f)

    keypoints = {k: np.array(v["keypoints"]) for k, v in keypoints.items()}

    for person_dir in images_path.iterdir():
        if not person_dir.is_dir():
            continue

        for image_path in person_dir.iterdir():
            if not image_path.is_file():
                continue

            image = Image.open(image_path)
            person_name = person_dir.name
            key = str(Path(person_name) / image_path.name)
            landmarks = keypoints.get(key)
            # lfw contains images not used in pairs.txt
            if landmarks is None:
                continue

            image_aligned = norm_crop(np.array(image), landmarks)
            Path(aligned_path / person_name).mkdir(exist_ok=True)
            out_path = aligned_path / key
            image_aligned.save(out_path)

    return aligned_path


def run(args):
    dataset_type = args.type
    tfrecord_path = path_resolver.resolve_data_path(TFRECORD_LOCATION[dataset_type])

    if not args.force and tfrecord_path.exists():
        print(f"tfrecord already exists at {tfrecord_path}. Skipping...")
        return

    download_dataset(args.img, args.tgz)
    download_annotatons(args.pairs)
    aligned_path = align_dataset(args)
    with open(args.pairs) as f:
        pairs = f.readlines()[1:]
    img_list, gt_list = _get_paths(aligned_path, pairs)
    result_tfrecord_path = _convert_dataset(img_list, gt_list, args.img, dataset_type == "calib")
    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result_tfrecord_path, tfrecord_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python create_arcface_lfw_tfrecord.py calib\n"
            "  python create_arcface_lfw_tfrecord.py val\n"
        ),
    )
    parser.add_argument("type", help="TFRecord of which dataset to create", type=str, choices=["calib", "val"])

    parser.add_argument("--img", "-i", help="images directory", type=str, default="lfw")
    parser.add_argument("--tgz", help="path to lfw.tgz", type=str, default="lfw.tgz")

    parser.add_argument("--pairs", "-p", help="pairs.txt file path", type=str, default="pairs.txt")

    alignment_parser = parser.add_mutually_exclusive_group()
    alignment_parser.add_argument("--kps-url", help="download url for keypoints file", default=DEFAULT_KPS_URL)
    alignment_parser.add_argument("--kps", "-k", help="path to keypoints file", default="lfw.json")
    alignment_parser.add_argument(
        "--no-align",
        action="store_true",
        help=("Assume images are already aligned," " does not use keypoints for alignment"),
        default=False,
    )
    parser.add_argument("--force", "-f", help="override existing tfrecord", action="store_true", default=False)

    args = parser.parse_args()
    run(args)
