#!/usr/bin/env python

import os
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
import PIL
import numpy as np
from pathlib import Path
import tempfile
import zipfile
import tarfile

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive, download_to_file


DESCRIPTION = "Create tfrecord for image denoising from BSD68/CBSD68 benchmark images.\n"
TF_RECORD_TYPE = 'val', 'calib'
TF_RECORD_LOC = {'BSD68': {'val': 'models_files/BSD68/2023-06-14/bsd68_val.tfrecord',
                           'calib': 'models_files/BSD68/2023-06-14/bsd68_calib.tfrecord'},
                 'CBSD68': {'val': 'models_files/CBSD68/2023-06-25/cbsd68_val.tfrecord',
                            'calib': 'models_files/CBSD68/2023-06-25/cbsd68_calib.tfrecord'}}
VAL_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1mwMLt-niNqcQpfN_ZduG9j4k6P_ZkOl0"
CALIB_DOWNLOAD_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"

DATASET_IMAGE_DEPTH = {'BSD68': 1, 'CBSD68': 3}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def create_tf_example(dataset, img_path, dataset_type, img_index):
    image = PIL.Image.open(img_path)
    if dataset_type == 'calib' and dataset == 'BSD68':
        image = image.convert("L")
    width, height = image.size
    feature_dict = {
        'index': _int64_feature(img_index),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'depth': _int64_feature(DATASET_IMAGE_DEPTH[dataset]),
        'image_name': _bytes_feature(img_path.encode('utf8')),
        'img': _bytes_feature(np.array(image, np.uint8).tobytes()),
        'format': _bytes_feature('png'.encode('utf8') if dataset_type == 'val' else 'jpg'.encode('utf-8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(dataset, name, images_dir):
    tfrecord_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[dataset][name])
    (tfrecord_filename.parent).mkdir(parents=True, exist_ok=True)
    writer = tf.io.TFRecordWriter(str(tfrecord_filename))
    images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    i = 0
    for i, img_path in enumerate(images):
        try:
            tf_example = create_tf_example(dataset, img_path, name, i)
        except (FileNotFoundError, TypeError):
            print('error - %s' % img_path)
            continue
        writer.write(tf_example.SerializeToString())
        i += 1
    writer.close()
    return i


def _download_val_dataset(dataset_dir):
    if dataset_dir.is_dir():
        return
    dataset_dir.mkdir()
    with tempfile.NamedTemporaryFile() as outfile:
        out_filename = 'denoising_datasets.zip'
        download_from_drive(VAL_DOWNLOAD_URL, outfile, desc=out_filename)
        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))


def _download_calib_dataset(dataset_dir):
    if dataset_dir.is_dir():
        return
    dataset_dir.mkdir()
    with tempfile.NamedTemporaryFile() as outfile:
        out_filename = 'BDS300-images.tgz'
        download_to_file(CALIB_DOWNLOAD_URL, outfile, desc=out_filename)
        with tarfile.open(outfile.name, 'r') as tar_ref:
            tar_ref.extractall(str(dataset_dir))


def run(args):
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    if args.data_path:
        dataset_dir = args.data_path
    elif args.type == 'calib':
        dataset_path = path_resolver.resolve_data_path(Path('bsd300'))
        _download_calib_dataset(dataset_path)
        dataset_dir = dataset_path / 'BSDS300/images/train'
    else:
        dataset_path = path_resolver.resolve_data_path(Path('denoising_datasets'))
        _download_val_dataset(dataset_path)
        dataset_dir = dataset_path / 'test' / args.dataset
    tot_images = _create_tf_record(args.dataset, args.type, dataset_dir)
    print('Done converting {} images'.format(tot_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('dataset', help="dataset name", choices=['BSD68', 'CBSD68'])
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE),
                        choices=['val', 'calib'])
    parser.add_argument('--data-path', help='external data-path')
    args = parser.parse_args()
    run(args)

"""
-----------------------------------------------------------------
CMD used to create a bsd68_val.tfrecord dataset:
python create_bsd68_tfrecord.py BSD68 val
"""
