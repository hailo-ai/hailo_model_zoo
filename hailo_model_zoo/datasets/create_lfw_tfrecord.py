#!/usr/bin/env python

from __future__ import print_function
from builtins import str
import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image


def _add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def _get_paths(lfw_dir, pairs):
    path_list, issame_list = [], []
    for pair in pairs:
        if len(pair) == 3:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = _add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = _add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list.append([path0, path1])
            issame_list.append(issame)
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


def _convert_dataset(img_list, gt_list, dataset_dir):
    tfrecords_filename = os.path.join(dataset_dir, 'lfw_pairs.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    i = 0
    for img_path, gt in zip(img_list, gt_list):
        i += 1
        img_path0, img_path1 = img_path
        example0 = get_example(img_path0, gt, i)
        example1 = get_example(img_path1, gt, i)
        print("converting image number " + str(i) + " :" + img_path[0])
        writer.write(example0.SerializeToString())
        writer.write(example1.SerializeToString())
    writer.close()
    return i


def run(lfw_dir, pairs_file):
    assert lfw_dir != '', 'no dataset directory'
    assert pairs_file != '', 'no label file (pairs)'
    pairs = _read_pairs(pairs_file)
    img_list, gt_list = _get_paths(lfw_dir, pairs)
    num_of_pairs = _convert_dataset(img_list, gt_list, lfw_dir)
    print("Done converting {} images".format(num_of_pairs * 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--pairs', '-pairs', help="pairs text file", type=str, default='')
    args = parser.parse_args()
    run(args.img, args.pairs)
