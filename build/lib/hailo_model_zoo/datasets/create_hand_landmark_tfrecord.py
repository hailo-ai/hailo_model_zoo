#!/usr/bin/env python

import argparse
import os
import tempfile
import zipfile
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive


TF_RECORD_LOC = 'models_files/hands/2022-01-23/hands_calib.tfrecord'
DATASET_DOWNLOAD_URL = 'https://drive.google.com/u/0/uc?id=1KcMYcNJgtK1zZvfl_9sTqnyBUTri2aP2&export=download'


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(dataset_dir, num_images):
    """Loop over all the images in dataset_dir and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC)
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    filenames = [x for x in os.listdir(dataset_dir)]
    random.shuffle(filenames)
    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, img_path in enumerate(progress_bar):
            img_jpeg = open(os.path.join(dataset_dir, img_path), 'rb').read()
            img = np.array(Image.open(os.path.join(dataset_dir, img_path)))
            image_height = img.shape[0]
            image_width = img.shape[1]

            progress_bar.set_description(f"TFRECORD #{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return i + 1


def download_dataset():
    dataset_dir = path_resolver.resolve_data_path('hands')

    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # download images if needed
    with tempfile.NamedTemporaryFile() as outfile:
        download_from_drive(DATASET_DOWNLOAD_URL, outfile, desc='hands.zip')
        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))

    return dataset_dir / 'Hands'


def run(dataset_dir, num_images):
    if dataset_dir == '':
        dataset_dir = download_dataset()
    images_num = _create_tfrecord(Path(dataset_dir), num_images)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=128, help='Limit num images')
    args = parser.parse_args()
    run(args.img, args.num_images)
