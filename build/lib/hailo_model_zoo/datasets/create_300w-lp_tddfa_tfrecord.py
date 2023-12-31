#!/usr/bin/env python

import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive

DATASET_LOCATION = 'models_files/300w-lp_tddfa/2021-11-28/300w-lp_tddfa.tfrecord'
DATASET_DOWNLOAD_URL = 'https://drive.google.com/uc?id=17LfvBZFAeXt0ACPnVckfdrLTMHUpIQqE&export=download'

CALIB_SET_SIZE = 2048


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def run(dataset_dir):
    tfrecord_path = path_resolver.resolve_data_path(DATASET_LOCATION)
    if tfrecord_path.exists():
        print(f'tfrecord already exists at {tfrecord_path}. Skipping...')
        return

    download_dataset(dataset_dir)

    random.seed(0)
    images = random.sample(sorted(dataset_dir.glob('*.jpg')), CALIB_SET_SIZE)

    created_tfrecord_path = "aflw2k3d_tddfa.tfrecord"
    with tf.io.TFRecordWriter(str(created_tfrecord_path)) as writer:
        for file_name in tqdm(images):
            with open(file_name, 'rb') as f:
                image_cropped = f.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_name': _bytes_feature(file_name.name.encode('utf-8')),
                'image': _bytes_feature(image_cropped),
            }))
            writer.write(example.SerializeToString())

    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(created_tfrecord_path, tfrecord_path)


def download_dataset(dataset_dir):
    if not dataset_dir.exists():
        print('Downloading train_aug_120x120.zip')
        with tempfile.NamedTemporaryFile() as outfile:
            download_from_drive(DATASET_DOWNLOAD_URL, outfile, desc='train_aug_120x120.zip')

            with zipfile.ZipFile(outfile, 'r') as zip_ref:
                zip_ref.extractall(str(dataset_dir.parent))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Path to dataset root", type=Path, default='./train_aug_120x120')
    args = parser.parse_args()
    run(args.dir)
