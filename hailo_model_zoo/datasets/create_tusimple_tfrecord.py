#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int_list_feature(value):
    return tf.train.Feature(int_list=tf.train.Int64List(value=value))


def _create_tfrecord(image_names_list, tfrecord_filename):
    """Loop over all the images in filenames and create the TFRecord
    """
    writer = tf.io.TFRecordWriter(tfrecord_filename)
    i = 0
    for img_path in image_names_list:
        img_jpeg = open(img_path, 'rb').read()
        img = np.array(Image.open(img_path))
        image_height = img.shape[0]
        image_width = img.shape[1]
        print("converting image number " + str(i) + " from " + tfrecord_filename + " : " + img_path, end="\r")
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_height),
            'width': _int64_feature(image_width),
            'image_name': _bytes_feature(str.encode(img_path.split('/tusimple/')[1])),
            'image_jpeg': _bytes_feature(img_jpeg)}))
        writer.write(example.SerializeToString())
        i += 1
    writer.close()
    return i


def get_img_labels_list(dataset_dir, label_file):
    with tf.io.gfile.GFile(label_file, 'r') as fid:
        lane_annotations = fid.readlines()
    lane_annotation_dicts = [eval(lane_annotations[line]) for line in range(len(lane_annotations))]
    orig_file_names = [os.path.join(dataset_dir, image_label['raw_file']) for image_label in lane_annotation_dicts]
    return orig_file_names


def run(dataset_dir, label_file, file_name):
    assert dataset_dir != '', 'no dataset directory was provided'
    assert label_file != '', 'no label file was provided'
    assert file_name != '', 'please provide a name for the tfrecord output'
    if label_file != 'wild':
        img_names_list = get_img_labels_list(dataset_dir, label_file)
        images_num = _create_tfrecord(img_names_list, file_name)
    else:
        img_names_list = [os.path.join(dataset_dir, im_name) for im_name in os.listdir(dataset_dir)]
        images_num = _create_tfrecord(img_names_list, file_name)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str,
                        default='/local/data/datasets/tusimple/')
    parser.add_argument('--labels', '-labels', help="lanes ground truth", type=str,
                        default='/local/data/datasets/tusimple/test_label.json')
    parser.add_argument('--fname', '-fname', help="name of tfrecord file", type=str,
                        default='tusimple_test_with_json_for_labels.tfrecord')
    args = parser.parse_args()
    run(args.img, args.labels, args.fname)

"""
----------------------------------------------------------------------------
CMD used to create a tusimple_train.tfrecord for the TuSimple training set:
----------------------------------------------------------------------------
python create_tusimple_tfrecord.py
--img /local/data/datasets/tusimple_polylane/tusimple/train/
--labels /local/data/datasets/tusimple_polylane/tusimple/train/label_data_0313.json
--fname tusimple_train.tfrecord

----------------------------------------------------------------------------
CMD used to create tusimple_test_with_json_for_labels.tfrecord for the TuSimple validation set:
----------------------------------------------------------------------------
python create_tusimple_tfrecord.py
--img /local/data/datasets/tusimple/
--labels /local/data/datasets/tusimple/test_label.json
--fname tusimple_test_with_json_for_labels.tfrecord
"""
