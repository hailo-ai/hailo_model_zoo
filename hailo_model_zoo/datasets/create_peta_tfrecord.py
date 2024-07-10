#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

LABEL_FILE = "PETA.mat"
TF_RECORD_TYPE = "val", "train"
TF_RECORD_LOC = {
    "val": "models_files/peta/2022-06-09/peta_val.tfrecord",
    "train": "models_files/peta/2022-06-09/peta_train.tfrecord",
}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(filenames, name, num_images):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[: int(num_images)])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, labels) in enumerate(progress_bar):
            img_jpeg = open(img_path, "rb").read()
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            progress_bar.set_description(f"{name} #{i}: {img_path}")
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "attributes": _int64_feature(labels),
                        "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                        "image_jpeg": _bytes_feature(img_jpeg),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def _get_files_and_labels_list(dataset_dir, type):
    """Get a list of filenames and labels from the dataset directory"""
    data = loadmat(os.path.join(dataset_dir, LABEL_FILE))
    file_list = []
    trainval_split = (data["peta"][0][0][3][0][0][0][0][0][:, 0] - 1).tolist() + (
        data["peta"][0][0][3][0][0][0][0][1][:, 0] - 1
    ).tolist()
    trainval_split = [str(x + 1).zfill(5) + ".png" for x in trainval_split]
    test_split = (data["peta"][0][0][3][0][0][0][0][2][:, 0] - 1).tolist()
    test_split = [str(x + 1).zfill(5) + ".png" for x in test_split]
    for image in os.listdir(os.path.join(dataset_dir, "images")):
        idx = int(os.path.splitext(image)[0])
        label = data["peta"][0][0][0][idx - 1, 4:].tolist()
        if type == "train" and image in trainval_split:
            file_list.append([os.path.join(dataset_dir, "images", image), label])
        elif type == "val" and image in test_split:
            file_list.append([os.path.join(dataset_dir, "images", image), label])
    if type == "train":
        random.seed(0)
        random.shuffle(file_list)
    return file_list


def run(type, dataset_dir, num_images):
    img_labels_list = _get_files_and_labels_list(dataset_dir, type)
    images_num = _create_tfrecord(img_labels_list, type, num_images)
    print("Done converting {} images".format(images_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--data", help="dataset directory", type=str, default="")
    parser.add_argument("--num-images", type=int, default=1e9, help="Limit num images")
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    assert args.data != "", "please provide dataset directory"
    run(args.type, args.data, args.num_images)
