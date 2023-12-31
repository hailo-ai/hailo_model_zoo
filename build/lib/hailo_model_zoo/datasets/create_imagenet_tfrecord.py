#!/usr/bin/env python

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver


TF_RECORD_TYPE = 'val', 'calib'
TF_RECORD_LOC = {'val': 'models_files/imagenet/2021-06-20/imagenet_val.tfrecord',
                 'calib': 'models_files/imagenet/2021-06-20/imagenet_calib.tfrecord'}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(filenames, name, num_images):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, label_index) in enumerate(progress_bar):
            img_jpeg = open(img_path, 'rb').read()
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            progress_bar.set_description(f"{name} #{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'label_index': _int64_feature(label_index),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return i + 1


def _get_files_and_labels_list(dataset_dir):
    """Get a list of filenames and labels from the dataset directory
    """
    file_list = []
    lib_names = sorted(f.name for f in Path(dataset_dir).iterdir() if re.match(r"^n[0-9]+$", f.name))
    class_id = {v: i for i, v in enumerate(lib_names)}
    for lib in Path(dataset_dir).iterdir():
        for img in lib.iterdir():
            file_list.append([str(img), class_id[lib.name]])
    random.seed(0)
    random.shuffle(file_list)
    return file_list


def run(type, dataset_dir, num_images):
    img_labels_list = _get_files_and_labels_list(dataset_dir)
    images_num = _create_tfrecord(img_labels_list, type, num_images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=None, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    assert args.img != '', 'please provide dataset directory'
    num_images = args.num_images if args.num_images is not None else (50000 if args.type == 'val' else 8192)
    run(args.type, args.img, num_images)
