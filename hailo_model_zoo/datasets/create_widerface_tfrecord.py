#!/usr/bin/env python

import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from PIL import Image


CALIB_SET_OFFSET = 999
CALIB_SET_LENGTH = 128


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_gt_boxes(gt_mat_path, hard_mat_path):
    gt_mat = loadmat(gt_mat_path)

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    if hard_mat_path is not None:
        hard_mat = loadmat(hard_mat_path)
        hard_gt_list = hard_mat['gt_list']
    else:
        hard_gt_list = None

    return facebox_list, event_list, file_list, hard_gt_list


def _create_tfrecord(gt_mat_path, hard_mat_path, dataset_dir, name, is_calibration):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join(dataset_dir, 'widerface' + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    all_file_paths = []
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1] in ['.jpg', '.png']:
                all_file_paths.append(os.path.join(dataset_dir, root, name))
    all_file_paths.sort()

    # NOTE: In case we deal with the training set, the hard `.mat` used for validation is ignored.
    # This turns evaluation on the training set into not meanignful
    if 'train' in gt_mat_path:
        hard_mat_path = None
    facebox_list, event_list, file_list, hard_gt_list = get_gt_boxes(gt_mat_path, hard_mat_path)

    i = 0
    for i, img_path in enumerate(all_file_paths):
        if is_calibration:
            if i < CALIB_SET_OFFSET:
                continue
            if i > CALIB_SET_OFFSET + CALIB_SET_LENGTH:
                break
        xmin, xmax, ymin, ymax, category_id = [], [], [], [], []
        img_jpeg = open(img_path, 'rb').read()
        img = np.array(Image.open(img_path))
        image_height = img.shape[0]
        image_width = img.shape[1]

        img_name = os.path.basename(img_path).replace('.jpg', '')
        wider_category_name = os.path.basename(os.path.dirname(img_path))
        event_id = None
        for event_index in range(event_list.shape[0]):
            if event_list[:, 0][event_index][0] == wider_category_name:
                event_id = event_index

        assert event_id is not None

        image_id = None

        for image_index in range(file_list[event_id][0].shape[0]):
            if file_list[event_id][0][image_index][0][0] == img_name:
                image_id = image_index

        assert image_id is not None

        gt_boxes = facebox_list[event_id][0][image_id][0]
        wider_hard_keep_index = [0] * 1024
        if hard_gt_list is not None:
            if hard_gt_list[event_id][0][image_id][0].shape[0] != 0:
                wider_hard_keep_index = list(hard_gt_list[event_id][0][image_id][0].squeeze(axis=1))
                wider_hard_keep_index += [0] * (1024 - len(wider_hard_keep_index))

        for object_annotations in gt_boxes:
            (x, y, width, height) = object_annotations
            if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                continue
            xmin.append(float(x) / image_width)
            xmax.append(float(x + width) / image_width)
            ymin.append(float(y) / image_height)
            ymax.append(float(y + height) / image_height)
            category_id.append(1)  # All objects are faces

        print("converting image number {} from {}\r".format(str(i), img_name), end='')
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_height),
            'width': _int64_feature(image_width),
            'num_boxes': _int64_feature(len(gt_boxes)),
            'wider_hard_keep_index': _int64_feature(wider_hard_keep_index),
            'image_id': _int64_feature(i),
            'xmin': _float_list_feature(xmin),
            'xmax': _float_list_feature(xmax),
            'ymin': _float_list_feature(ymin),
            'ymax': _float_list_feature(ymax),
            'category_id': _int64_feature(category_id),
            'image_name': _bytes_feature(str.encode(os.path.join(wider_category_name, img_name))),
            'image_jpeg': _bytes_feature(img_jpeg)}))
        writer.write(example.SerializeToString())
    writer.close()

    return i


def run(dataset_dir, gt_mat_path, hard_mat_path, is_calibration, name='val'):
    assert dataset_dir != '', 'no dataset directory'
    assert gt_mat_path != '', 'no ground truth path'
    if is_calibration and name != 'calibration_set':
        print("Calibration set creation is chosen but file name suffix is {}... "
              "Setting it to 'calibration_set'...".format(name))
        name = 'calibration_set'
    images_num = _create_tfrecord(gt_mat_path, hard_mat_path, dataset_dir, name,
                                  is_calibration)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str,
                        default='/local/data/datasets/face/WIDER/WIDER_val/')
    parser.add_argument('--gt_mat_path', '-gt',
                        help="Path of gt `.mat` file. "
                             "See https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate",
                        type=str, default='/local/data/datasets/face/WIDER/wider_face_split/wider_face_val.mat')
    parser.add_argument('--hard_mat_path', '-hard',
                        help="Path of HARD gt `.mat` file. "
                             "See https://github.com/biubug6/Pytorch_Retinaface/tree/master/widerface_evaluate",
                        type=str, default='/local/data/datasets/face/WIDER/wider_face_split/wider_hard_val.mat')
    parser.add_argument('--calibration_set', '-c',
                        help="Create a calibration set of 128 images using the {}th to the {}th files".format(
                            CALIB_SET_OFFSET + 1, CALIB_SET_OFFSET + CALIB_SET_LENGTH),
                        action='store_true', default=False)
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='val')
    args = parser.parse_args()
    run(args.img, args.gt_mat_path, args.hard_mat_path, args.calibration_set, args.name)
"""
----------------------------------------------------------------------------
CMD used to create a widerfaceval.tfrecord of the WIDER FACE validation dataset:
----------------------------------------------------------------------------
python create_widerface_tfrecord.py
--img /local/data/datasets/face/WIDER/WIDER_val
--gt_mat_path /local/data/datasets/face/WIDER/wider_face_split/wider_face_val.mat
--hard_mat_path /local/data/datasets/face/WIDER/wider_face_split/wider_hard_val.mat
--name val

----------------------------------------------------------------------------
CMD used to create a widerfacetrain.tfrecord of the WIDER FACE training dataset:
Note: In this case we don't pass the hard `.mat` file,
      and evaluation on the training set becomes not meaningful.
----------------------------------------------------------------------------
python create_widerface_tfrecord.py
--img /local/data/datasets/face/WIDER/WIDER_train
--gt_mat_path /local/data/datasets/face/WIDER/wider_face_split/wider_face_train.mat
--name train
"""
