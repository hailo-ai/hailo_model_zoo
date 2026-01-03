#!/usr/bin/env python

import argparse
import os

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

TF_RECORD_TYPE = "val", "calib"
TF_RECORD_LOC = {
    "val": "./dpm_val.tfrecord",
    "calib": "./dpm_calib.tfrecord",
}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(labels, images, split, limit):
    """Loop over all the images in filenames and create the TFRecord"""
    tfrecords_filename = os.path.join("./", TF_RECORD_LOC["val" if split == "val" else "calib"])
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    i = 0
    for img_path, label in tqdm.tqdm(zip(images, labels), desc="Image:"):
        img = np.array(Image.open(img_path), np.uint8)
        image_height = img.shape[0]
        image_width = img.shape[1]
        mask = np.array(Image.open(label), np.uint8)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "height": _int64_feature(image_height),
                    "width": _int64_feature(image_width),
                    "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                    "mask": _bytes_feature(np.array(mask, np.uint8).tostring()),
                    "image": _bytes_feature(np.array(img, np.uint8).tostring()),
                }
            )
        )
        writer.write(example.SerializeToString())
        i += 1
        if i >= limit:
            break
    writer.close()
    return i


def get_labels_images(dataset_dir, split, limit):
    labels, images_path = [], []
    image_dir = "val" if split == "val" else "train"
    for img in os.listdir(os.path.join(dataset_dir, "images", image_dir)):
        image_full_path = os.path.join(dataset_dir, "images", image_dir, img)
        pil_img = Image.open(image_full_path)
        # filter large frames
        if pil_img.width >= 1024 or pil_img.height >= 1024:
            continue
        images_path.append(image_full_path)
        labels.append(os.path.join(dataset_dir, "labels", image_dir, img))
        if len(images_path) >= limit:
            break
    return labels, images_path


def run(dataset_dir, split, limit):
    assert os.path.isdir(dataset_dir), "no dataset directory"
    assert split in TF_RECORD_TYPE, "split must be 'calib' or 'val'"
    labels, images = get_labels_images(dataset_dir, split, limit)
    images_num = _create_tfrecord(labels, images, split, limit)
    print("Done converting {} images".format(images_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset directory", type=str, default="/data/data/datasets/semseg_dataset")
    parser.add_argument("--split", help="split (calib/val)", type=str, default="calib")
    parser.add_argument("--limit", help="limit number of images to convert", type=int, default=int(1e9))
    args = parser.parse_args()
    run(args.dataset, args.split, args.limit)

"""
-----------------------------------------------------------------
CMD used to create a DPM Semantic segmentation TFRecord
python create_ss_dpm_tfrecord.py
-----------------------------------------------------------------
"""
