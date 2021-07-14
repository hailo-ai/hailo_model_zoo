#!/usr/bin/env python

from __future__ import print_function
from builtins import str
import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

CALIB_PAIR_COUNT = 128


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _get_jpg_path(lfw_dir, person_name, image_num):
    return os.path.join(lfw_dir, person_name, '{}_{}.jpg'.format(person_name, image_num.zfill(4)))


def _get_paths(lfw_dir, pairs):
    path_list, issame_list = [], []
    for pair_num, line in enumerate(pairs):
        values = line.strip().split('\t')
        if len(values) == 3:
            path0 = _get_jpg_path(lfw_dir, values[0], values[1])
            path1 = _get_jpg_path(lfw_dir, values[0], values[2])
            is_same = True
        else:
            path0 = _get_jpg_path(lfw_dir, values[0], values[1])
            path1 = _get_jpg_path(lfw_dir, values[2], values[3])
            is_same = False
        print(pair_num, is_same)

        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list.append((path0, path1))
            issame_list.append(is_same)
        else:
            raise FileNotFoundError("One of the files:\n{}\n{}\n not found".format(path0, path1))
    return path_list, issame_list


def get_example(img_path, gt, pair_idx):
    img = np.array(Image.open(img_path), np.uint8)
    filename = str.encode("{:<20}".format(os.path.basename(img_path)))
    height = img.shape[0]
    width = img.shape[1]
    img = img.tostring()
    gt = int(gt)
    pair_idx = int(pair_idx)
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_name': _bytes_feature(filename),
        'image': _bytes_feature(img),
        'pair_index': _int64_feature(pair_idx),
        'is_same': _int64_feature(gt)}))
    return example


def _convert_dataset(img_list, gt_list, dataset_dir, is_calibration=False):
    tfrecords_filename = os.path.join(dataset_dir,
                                      'arcface_lfw_pairs_{}.tfrecord'.format('calib' if is_calibration else 'val'))
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    i = 0
    for img_path, gt in zip(img_list, gt_list):
        if is_calibration and i == CALIB_PAIR_COUNT:
            break
        i += 1
        img_path0, img_path1 = img_path
        example0 = get_example(img_path0, gt, i)
        example1 = get_example(img_path1, gt, i)
        print("converting image number " + str(i) + " :" + img_path[0])
        writer.write(example0.SerializeToString())
        writer.write(example1.SerializeToString())
    writer.close()
    return i


def run(lfw_dir, pairs_file, is_calibration):
    assert lfw_dir != '', 'no dataset directory'
    assert pairs_file != '', 'no label file (pairs)'
    with open(pairs_file) as f:
        pairs = f.readlines()[1:]
    img_list, gt_list = _get_paths(lfw_dir, pairs)
    num_of_pairs = _convert_dataset(img_list, gt_list, lfw_dir, is_calibration)
    print("Done converting {} images".format(num_of_pairs * 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', help="images directory", type=str, default='')
    parser.add_argument('--pairs', '-p', help="pairs.txt file path", type=str, default='')
    parser.add_argument('--calibration', '-c',
                        help=("Should create calibration set of only {} pairs".format(CALIB_PAIR_COUNT)),
                        action='store_true', default=False)
    args = parser.parse_args()
    run(args.img, args.pairs, args.calibration)
