#!/usr/bin/env python

import argparse
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_file

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "models_files/oxford_pet/2022-02-01/oxford_pet_val.tfrecord",
    "calib": "models_files/oxford_pet/2022-02-01/oxford_pet_calib.tfrecord",
}
DATASET_DOWNLOAD_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_DOWNLOAD_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(train_list, test_list, num_images, type):
    """Loop over all the images in dataset_dir and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[type])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    images_list = train_list if type == "calib" else test_list
    random.shuffle(images_list)
    progress_bar = tqdm(images_list[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in enumerate(progress_bar):
            img_jpeg = open(img_path, "rb").read()
            mask = Image.open(str(img_path).replace("images", "annotations/trimaps").replace(".jpg", ".png"))
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]

            progress_bar.set_description(f"TFRECORD #{i}")
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                        "mask": _bytes_feature(np.array(mask, np.uint8).tostring()),
                        "image_jpeg": _bytes_feature(img_jpeg),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def _download_dataset():
    dataset_dir = path_resolver.resolve_data_path("oxford_pet")

    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # download images
    filename = download_file(DATASET_DOWNLOAD_URL)
    with tarfile.open(filename, "r:gz") as tar_ref:
        tar_ref.extractall(dataset_dir)
    Path(filename).unlink()

    # download annotations
    filename = download_file(ANNOTATIONS_DOWNLOAD_URL)
    with tarfile.open(filename, "r:gz") as tar_ref:
        tar_ref.extractall(dataset_dir)
    Path(filename).unlink()
    return dataset_dir


def _get_train_test_datasets(dataset_dir):
    test_images, train_images = [], []
    test_file = "annotations/test.txt"
    images_dir = "images"
    with open(dataset_dir / test_file, "r") as f:
        data = f.read()
    test_filenames = [x.split(" ")[0] + ".jpg" for x in data.split("\n") if x != ""]
    for img in (dataset_dir / images_dir).iterdir():
        if ".jpg" in img.name:
            if img.name in test_filenames:
                test_images.append(dataset_dir / "images" / img.name)
            else:
                train_images.append(dataset_dir / "images" / img.name)
    return train_images, test_images


def run(dataset_dir, num_images, type):
    if dataset_dir == "":
        dataset_dir = _download_dataset()
    train_list, test_list = _get_train_test_datasets(Path(dataset_dir))
    images_num = _create_tfrecord(train_list, test_list, num_images, type)
    print("\nDone converting {} images".format(images_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--data", "-data", help="dataset directory", type=str, default="")
    parser.add_argument("--num-images", type=int, default=4096, help="Limit num images")
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    run(args.data, args.num_images, args.type)
