#!/usr/bin/env python

import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


AFLW2000 = Path('AFLW2000')
AFLW2000_CROP = Path('AFLW2000-3D_crop')


def run(dataset_dir):
    yaws_list = np.load(dataset_dir / 'AFLW2000-3D.pose.npy')
    pts68_all_ori = np.load(dataset_dir / 'AFLW2000-3D.pts68.npy')
    pts68_all_re = np.load(dataset_dir / 'AFLW2000-3D-Reannotated.pts68.npy')
    roi_boxes = np.load(dataset_dir / 'AFLW2000-3D_crop.roi_box.npy')
    with open(dataset_dir / 'AFLW2000-3D_crop.list') as list_file:
        file_list = [line.strip() for line in list_file]

    with tf.io.TFRecordWriter("aflw2k3d_tddfa.tfrecord") as writer:
        for file_name, yaw, labels, labels_re, boxes in tqdm(
                zip(file_list, yaws_list, pts68_all_ori, pts68_all_re, roi_boxes)):
            with open(dataset_dir / AFLW2000 / file_name, 'rb') as f:
                image_original = f.read()
            with open(dataset_dir / AFLW2000_CROP / file_name, 'rb') as f:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=Path, default='')
    args = parser.parse_args()
    run(args.img)
