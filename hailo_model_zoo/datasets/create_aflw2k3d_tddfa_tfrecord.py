#!/usr/bin/env python

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_to_file, download_from_drive, download_file

AFLW2000 = Path('AFLW2000')
AFLW2000_CROP = Path('AFLW2000-3D_crop')

DATASET_LOCATION = 'models_files/aflw2k3d_tddfa/2021-03-16/aflw2k3d_tddfa.tfrecord'
DATASET_DOWNLOAD_URL = 'https://drive.google.com/uc?id=1r_ciJ1M0BSRTwndIBt42GlPFRv6CvvEP&export=download'
UNCROPPED_DATASET_URL = 'http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip'


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def run(dataset_dir):
    tfrecord_path = path_resolver.resolve_data_path(DATASET_LOCATION)
    if tfrecord_path.exists():
        print(f'tfrecord already exists at {tfrecord_path}. Skipping...')
        return

    download_dataset(dataset_dir)

    yaws_list = np.load(dataset_dir / 'AFLW2000-3D.pose.npy')
    pts68_all_ori = np.load(dataset_dir / 'AFLW2000-3D.pts68.npy')
    pts68_all_re = np.load(dataset_dir / 'AFLW2000-3D-Reannotated.pts68.npy')
    roi_boxes = np.load(dataset_dir / 'AFLW2000-3D_crop.roi_box.npy')
    images_dir = dataset_dir / 'test.data'
    with open(images_dir / 'AFLW2000-3D_crop.list') as list_file:
        file_list = [line.strip() for line in list_file]

    created_tfrecord_path = dataset_dir / "aflw2k3d_tddfa.tfrecord"
    with tf.io.TFRecordWriter(str(created_tfrecord_path)) as writer:
        for file_name, yaw, labels, labels_re, boxes in tqdm(
                zip(file_list, yaws_list, pts68_all_ori, pts68_all_re, roi_boxes)):
            with open(images_dir / AFLW2000 / file_name, 'rb') as f:
                image_original = f.read()
            with open(images_dir / AFLW2000_CROP / file_name, 'rb') as f:
                image_cropped = f.read()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_name': _bytes_feature(file_name.encode('utf-8')),
                'image': _bytes_feature(image_cropped),
                'uncropped_image': _bytes_feature(image_original),
                'landmarks': _float_list_feature(labels.transpose().flatten()),
                'landmarks_reannotated': _float_list_feature(labels_re.transpose().flatten()),
                'yaw': _float_feature(yaw),
                'roi_box': _float_list_feature(boxes),
            }))
            writer.write(example.SerializeToString())

    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(created_tfrecord_path, tfrecord_path)


def download_dataset(dataset_dir):
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for f in [
        'https://github.com/cleardusk/3DDFA/raw/master/test.configs/AFLW2000-3D.pose.npy',
        'https://github.com/cleardusk/3DDFA/raw/master/test.configs/AFLW2000-3D.pts68.npy',
        'https://github.com/cleardusk/3DDFA/raw/master/test.configs/AFLW2000-3D-Reannotated.pts68.npy',
        'https://github.com/cleardusk/3DDFA/raw/master/test.configs/AFLW2000-3D_crop.roi_box.npy'
    ]:
        filename = f.split('/')[-1]
        if not dataset_dir.joinpath(filename).exists():
            print(f'Downloading {f} to {dataset_dir}/{filename}')
            download_file(f, dataset_dir)

    images_dir = dataset_dir.joinpath('test.data')
    if not images_dir.exists():
        print('Downloading test.data.zip')
        with tempfile.NamedTemporaryFile() as outfile:
            download_from_drive(DATASET_DOWNLOAD_URL, outfile, desc='test.data.zip')

            with zipfile.ZipFile(outfile, 'r') as zip_ref:
                zip_ref.extractall(str(dataset_dir))

    if not images_dir.joinpath(AFLW2000).exists():
        print(f'Downloading {UNCROPPED_DATASET_URL}')
        with tempfile.NamedTemporaryFile() as outfile:
            download_to_file(UNCROPPED_DATASET_URL, outfile)

            with zipfile.ZipFile(outfile, 'r') as zip_ref:
                zip_ref.extractall(str(images_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Path to dataset root", type=Path, default='./aflw2k3d_tddfa')
    args = parser.parse_args()
    run(args.dir)
