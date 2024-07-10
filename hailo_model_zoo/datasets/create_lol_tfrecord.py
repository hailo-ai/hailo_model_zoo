#!/usr/bin/env python

import argparse
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import PIL
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive

LOL_DATASET_PAPER = "https://arxiv.org/abs/1808.04560v1"
DESCRIPTION = (
    "Create tfrecord for Low Light Enhancement task from LOL dataset (low and high light images).\n"
    + f"More info in the paper: {LOL_DATASET_PAPER}\n\n"
    + "Example cmd to create a val_set_lol.tfrecord:\n\n"
    + "\tpython create_lol_tfrecord.py val --ll your/lowlight/images/directory/ "
    + "--lle your/highlight/images/directory/"
)

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "models_files/LowLightEnhancement/LOL/{}/hailo_val_set_lol.tfrecord",
    "calib": "models_files/LowLightEnhancement/LOL/{}/hailo_calib_set_lol.tfrecord",
}

DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB&authuser=0"
LOL_ZIP_NAME = "LOLdataset.zip"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def create_tf_example(img_path, ll_enhanced_dir):
    ll_image = PIL.Image.open(img_path)
    ll_width, ll_height = ll_image.size
    ll_filename = os.path.basename(img_path)
    ll_enhanced_image = PIL.Image.open(os.path.join(ll_enhanced_dir, ll_filename))

    feature_dict = {
        "height": _int64_feature(ll_height),
        "width": _int64_feature(ll_width),
        "image_name": _bytes_feature(ll_filename.encode("utf8")),
        "ll_img": _bytes_feature(np.array(ll_image, np.uint8).tobytes()),
        "ll_enhanced_img": _bytes_feature(np.array(ll_enhanced_image, np.uint8).tobytes()),
        "format": _bytes_feature("png".encode("utf8")),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _download_dataset(dataset_dir):
    if dataset_dir.is_dir():
        print(f"{dataset_dir} already exists, skipping download.")
        return
    dataset_dir.mkdir()
    with tempfile.NamedTemporaryFile() as outfile:
        download_from_drive(DOWNLOAD_URL, outfile, desc=LOL_ZIP_NAME)
        with zipfile.ZipFile(outfile, "r") as zip_ref:
            zip_ref.extractall(str(dataset_dir))
    print(f"Downloaded dataset to {dataset_dir}")


def _create_tf_record(ll_dir, ll_enhanced_dir, name, num_images=None):
    current_date = datetime.now().strftime("%Y-%m-%d")
    tfrecord_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name].format(current_date))
    (tfrecord_filename.parent).mkdir(parents=True, exist_ok=True)
    num_images = num_images if num_images is not None else (128 if args.type == "calib" else 1024)

    if ll_dir is None or ll_enhanced_dir is None:
        if ll_dir is not None or ll_enhanced_dir is not None:
            raise ValueError("Please use ll and lle arguments together.")
        dataset_path = Path("lol_dataset")
        _download_dataset(dataset_path)

        if name == "val":
            ll_dir = dataset_path.joinpath("eval15", "low")
            ll_enhanced_dir = dataset_path.joinpath("eval15", "high")
        else:  # 'calib'
            ll_dir = dataset_path.joinpath("our485", "low")
            ll_enhanced_dir = dataset_path.joinpath("our485", "high")

    writer = tf.io.TFRecordWriter(str(tfrecord_filename))
    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(ll_dir)) for f in fn if f[-3:] == "png"]
    images = images[:num_images]
    for i, img_path in enumerate(tqdm(images, desc="Converting images")):
        try:
            tf_example = create_tf_example(img_path, ll_enhanced_dir)
        except (FileNotFoundError, TypeError):
            continue
        writer.write(tf_example.SerializeToString())
        if i == num_images:
            break
    writer.close()
    i = i if i == num_images else i + 1
    print(f"Done converting {i} images.\nSaved to: {tfrecord_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--ll", help="low light images directory", type=str, default=None)
    parser.add_argument("--lle", help="low light enhanced images directory", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=None, help="Limit num images")

    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "Please provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    tot_images = _create_tf_record(args.ll, args.lle, args.type, num_images=args.num_images)
