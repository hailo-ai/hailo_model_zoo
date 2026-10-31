#!/usr/bin/env python

import argparse
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

TF_RECORD_TYPE = "calib", "val", "test"
TF_RECORD_LOC = {
    "calib": "models_files/uavid/2026-09-03/uavid_calib.tfrecord",
    "val": "models_files/uavid/2026-09-03/uavid_val.tfrecord",
    "test": "models_files/uavid/2026-09-03/uavid_test.tfrecord",
}
SPLIT_DIR = {"calib": "train", "val": "val", "test": "test"}
NUM_CLASSES = 8
IGNORE_LABEL = 255


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _load_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"{mask_path}: expected a single-channel class-id mask, got shape {mask.shape}")
    valid = (mask < NUM_CLASSES) | (mask == IGNORE_LABEL)
    if not valid.all():
        raise ValueError(f"{mask_path}: unexpected label values {np.unique(mask[~valid])}")
    return mask.astype(np.uint8)


def _create_tfrecord(filenames, name, num_images):
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)
    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (mask_path, img_path) in enumerate(progress_bar):
            img = np.array(Image.open(img_path).convert("RGB"), np.uint8)
            image_height, image_width = img.shape[:2]
            mask = _load_mask(mask_path)
            if mask.shape != (image_height, image_width):
                raise ValueError(f"{img_path}: image {img.shape[:2]} and mask {mask.shape} differ")
            img_jpeg = tf.image.encode_jpeg(img, quality=95)
            progress_bar.set_description(f"{name} #{i + 1}: {img_path}")
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(image_height),
                        "width": _int64_feature(image_width),
                        "image_name": _bytes_feature(str.encode(os.path.basename(img_path))),
                        "mask": _bytes_feature(mask.tobytes()),
                        "image_jpeg": _bytes_feature(img_jpeg.numpy()),
                    }
                )
            )
            writer.write(example.SerializeToString())
    return i + 1


def get_img_labels_list(data_dir, name):
    split = SPLIT_DIR[name]
    img_dir = Path(data_dir) / "images" / split
    mask_dir = Path(data_dir) / "masks" / split
    files = []
    for img in sorted(img_dir.glob("*.png")):
        mask = mask_dir / img.name
        if not mask.exists():
            raise FileNotFoundError(mask)
        files.append([str(mask), str(img)])
    if name == "calib":
        random.seed(0)
        random.shuffle(files)
    return files


def run(data_dir, name, num_images):
    img_labels_list = get_img_labels_list(data_dir, name)
    images_num = _create_tfrecord(img_labels_list, name, num_images)
    print("Done converting {} images".format(images_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", help="which tf-record to create {}".format(TF_RECORD_TYPE))
    parser.add_argument("--data", help="UAVid root with images/<split> and masks/<split>", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=None, help="Limit num images")
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, "need to provide which kind of tfrecord to create {}".format(TF_RECORD_TYPE)
    num_images = args.num_images if args.num_images is not None else 100000
    run(args.data, args.type, num_images)

"""
-----------------------------------------------------------------
Expects the converted UAVid layout (single-channel class-id masks, 0..7, 255 = ignore) as
produced by https://github.com/dronefreak/CABiNet src/scripts/convert_uavid_to_yolo.py or the
HuggingFace mirror dronefreak/UAVid-2020:
python create_uavid_tfrecord.py calib --data /fastdata/users/shacharg/cabinet/data/uavid
python create_uavid_tfrecord.py test  --data /fastdata/users/shacharg/cabinet/data/uavid
-----------------------------------------------------------------
"""
