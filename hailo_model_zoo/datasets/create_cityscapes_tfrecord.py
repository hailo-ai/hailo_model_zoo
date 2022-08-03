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


TF_RECORD_TYPE = 'val', 'calib'
TF_RECORD_LOC = {'val': 'models_files/cityscapes/2022-05-15/cityscapes_val.tfrecord',
                 'calib': 'models_files/cityscapes/2022-05-15/cityscapes_calib.tfrecord'}
classMap = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
            8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255,
            15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8,
            22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18}


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
        for i, (mask_path, img_path) in enumerate(progress_bar):
            img = np.array(Image.open(img_path), np.uint8)
            image_height = img.shape[0]
            image_width = img.shape[1]
            mask = np.array(Image.open(mask_path))
            mask = np.array(np.vectorize(classMap.get)(mask), np.uint8)
            img_jpeg = tf.image.encode_jpeg(img)
            progress_bar.set_description(f"{name} #{i+1}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'mask': _bytes_feature(mask.tostring()),
                'image_jpeg': _bytes_feature(img_jpeg.numpy())}))
            writer.write(example.SerializeToString())
    return i + 1


def get_img_labels_list(data_dir, name):
    dataset_name = 'val' if 'val' in name else 'train'
    gtFine = Path(data_dir) / 'gtFine' / dataset_name
    files = []
    for city in gtFine.iterdir():
        for mask in city.iterdir():
            if 'labelIds' not in str(mask):
                continue
            mask_file_name = str(mask)
            image_file_name = mask_file_name.replace('gtFine_labelIds', 'leftImg8bit')
            image_file_name = image_file_name.replace('gtFine', 'leftImg8bit')
            files.append([mask_file_name, image_file_name])
    random.seed(0)
    random.shuffle(files)
    return files


def run(data_dir, name, num_images):
    img_labels_list = get_img_labels_list(data_dir, name)
    images_num = _create_tfrecord(img_labels_list, name, num_images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--data', help="Cityscapes data directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=None, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    num_images = args.num_images if args.num_images is not None else (500 if args.type == 'val' else 1024)
    run(args.data, args.type, num_images)

"""
-----------------------------------------------------------------
CMD used to create a cityscapes.tfrecord dataset:
python create_cityscapes_tfrecord.py val --num-images 500 --data /data/data/Cityscapes/
-----------------------------------------------------------------
"""
