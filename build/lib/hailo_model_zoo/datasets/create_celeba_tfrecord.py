#!/usr/bin/env python

import argparse
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver


TF_RECORD_TYPE = 'val', 'train'
TF_RECORD_LOC = {'val': 'models_files/celeba/2022-07-03/celeba_val.tfrecord',
                 'train': 'models_files/celeba/2022-07-03/celeba_train.tfrecord'}
PARTITION_FILE = "list_eval_partition.txt"
ANNOTATIONS_FILE = "list_attr_celeba.txt"


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
    tfrecords_filename.parent.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:int(num_images)])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, labels) in enumerate(progress_bar):
            img_jpeg = open(img_path, 'rb').read()
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            progress_bar.set_description(f"{name} #{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'attributes': _int64_feature(labels),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return i + 1


def _get_image_names(dataset_dir, type):
    img_list = []
    with open(os.path.join(dataset_dir, PARTITION_FILE)) as f:
        for line in f.readlines():
            row = line.strip().split(" ")
            img_name = row[0]
            label = row[-1]
            if label == "0" and type == 'train':
                img_list.append(img_name)
            elif label == "2" and type == 'val':
                img_list.append(img_name)
    return img_list


def _get_annnotations(dataset_dir):
    ann = {}
    with open(os.path.join(dataset_dir, ANNOTATIONS_FILE), encoding="utf-8") as f:
        for line in f.readlines():
            row = line.strip().split(" ")
            row = [i.replace("-1", "0") for i in row if i != ""]
            if len(row) == 41:
                img_name = row[0]
                ann[img_name] = " ".join(row)
    return ann


def _get_files_and_labels_list(dataset_dir, type):
    """Get a list of filenames and labels from the dataset directory
    """
    file_list = []
    img_names = _get_image_names(dataset_dir, type)
    annotations = _get_annnotations(dataset_dir)
    for image in img_names:
        labels = annotations[image].split(' ')
        labels = [np.int64(x) for x in labels[1:]]
        full_image_path = os.path.join(dataset_dir, 'img_align_celeba_png', image)
        file_list.append([full_image_path, labels])
    if type == 'train':
        random.seed(0)
        random.shuffle(file_list)
    return file_list


def run(type, dataset_dir, num_images):
    img_labels_list = _get_files_and_labels_list(dataset_dir, type)
    images_num = _create_tfrecord(img_labels_list, type, num_images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--data', help="dataset directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=1e9, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    assert args.data != '', 'please provide dataset directory'
    run(args.type, args.data, args.num_images)
