#!/usr/bin/env python

import random
import argparse
import tempfile
import os
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import h5py

from hailo_model_zoo.utils import path_resolver, downloader

TF_RECORD_TYPE = 'val', 'calib'

TF_RECORD_LOC = {'val': 'models_files/nyu_depth_v2/2020-11-01/nyu_depth_v2_val.tfrecord',
                 'calib': 'models_files/nyu_depth_v2/2020-11-01/nyu_depth_v2_calib.tfrecord'}

DOWNLOAD_URL = {'dataset': ("http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz")}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def h5_loader(path):
    with h5py.File(path, "r") as h5f:
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
    return rgb, depth


def _create_tfrecord(filenames, name, num_images=8192, shuffle=False):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    num_images = min(len(filenames), num_images)
    if shuffle:
        random.seed(0)
        random.shuffle(filenames)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, file in enumerate(progress_bar):
            rgb, depth = h5_loader(file)
            name = os.path.basename(file)
            image_width = rgb.shape[1]
            image_height = rgb.shape[0]

            progress_bar.set_description(f"#{i}: {name}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'image_name': _bytes_feature(str.encode(os.path.basename(file))),
                'depth': _bytes_feature(np.array(depth, np.float32).tobytes()),
                'rgb': _bytes_feature(np.array(rgb, np.uint8).tobytes()),
            }))
            writer.write(example.SerializeToString())
    print('\nDone converting {} images'.format(i + 1))
    return tfrecords_filename


def get_image_files(data_dir):
    return [str(f) for f in data_dir.glob('**/*.h5')]


def download_dataset(path=None):
    if path is None:
        dataset_dir = Path.cwd()
        dataset_dir = dataset_dir / 'nyu_depth_v2'
        # download images if needed
        if not dataset_dir.exists():
            # create the libraries if needed
            dataset_dir.mkdir(parents=True, exist_ok=True)
            print(f'Downloading dataset to {dataset_dir}')
            with tempfile.NamedTemporaryFile('wb', suffix='.tar.gz') as outfile:
                downloader.download_to_file(DOWNLOAD_URL['dataset'], outfile)
                outfile.seek(0)  # rewind to beginning of file
                with tarfile.open(outfile.name, 'r') as tar:
                    tar.extractall(str(dataset_dir))
    else:
        dataset_dir = Path(path)
    dataset_train, dataset_val = validate_dataset_dirs(dataset_dir)
    return dataset_train, dataset_val


def validate_dataset_dirs(dataset_dir):
    dataset_train = dataset_dir / 'nyudepthv2' / 'train'
    dataset_val = dataset_dir / 'nyudepthv2' / 'val'
    err_msg = f'Could not find expected data libraries, delete {str(dataset_dir)} to re-download dataset'
    if not (dataset_train.exists() and dataset_val.exists()):
        raise ValueError(err_msg)
    return dataset_train, dataset_val


def run(data_dir, num_images, name, shuffle=False):
    images = get_image_files(data_dir)
    output_file = _create_tfrecord(images, name, num_images, shuffle)
    print(f'Done saved tfrecord to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--data', help="optional: path to manual downloaded dataset directory", type=str, default=None)
    parser.add_argument('--num-images', help="optional: limit the number of images", type=int, default=1024)
    args = parser.parse_args()

    dataset_train, dataset_val = download_dataset(args.data)

    if args.type == 'val':
        print('Creating validation tfrecord')
        run(dataset_val, args.num_images, name='val', shuffle=False)
    else:
        if args.type == 'calib':
            print('Creating calibration tfrecord')
            run(dataset_train, args.num_images, name='calib', shuffle=True)
        else:
            print('ERROR No dataset type selected')

"""
----------------------------------------------------------------------------
CMD used to create a nyu_depth_v2_val.tfrecord validation dataset:
----------------------------------------------------------------------------
python create_nyu_depth_v2_tfrecord.py val

----------------------------------------------------------------------------
CMD used to create a nyu_depth_v2_calib.tfrecord calibration dataset:
----------------------------------------------------------------------------
python create_nyu_depth_v2_tfrecord.py calib

"""
