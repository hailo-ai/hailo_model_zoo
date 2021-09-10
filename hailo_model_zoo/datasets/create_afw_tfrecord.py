#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.io as sio


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _create_tfrecord(labels, images):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join('./', 'afw_val.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    with tf.Graph().as_default():
        image_placeholder = tf.compat.v1.placeholder(dtype=tf.uint8, name='image_placeholder')
        encoded_image = tf.image.encode_jpeg(image_placeholder)
        i = 0
        with tf.compat.v1.Session('') as sess:
            for img_path, label in zip(images, labels):
                img = np.array(Image.open(img_path).crop(label[0]), np.uint8)
                image_height = img.shape[0]
                image_width = img.shape[1]
                img_jpeg = sess.run(encoded_image, feed_dict={image_placeholder: img})

                print("converting image number " + str(i) + ": " + img_path)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_height),
                    'width': _int64_feature(image_width),
                    'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                    'angles': _float_list_feature(label[1]),
                    'image_jpeg': _bytes_feature(img_jpeg)}))
                writer.write(example.SerializeToString())
                i += 1
            writer.close()
            return i


def get_labels_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    x_min = min(pt2d[0, :])
    y_min = min(pt2d[1, :])
    x_max = max(pt2d[0, :])
    y_max = max(pt2d[1, :])
    x_min -= 2 * 0.2 * abs(x_max - x_min)
    y_min -= 2 * 0.2 * abs(y_max - y_min)
    x_max += 2 * 0.2 * abs(x_max - x_min)
    y_max += 0.6 * 0.2 * abs(y_max - y_min)

    pre_pose_params = mat['Pose_Para'][0]
    pose = pre_pose_params[:3]
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    return [int(x_min), int(y_min), int(x_max), int(y_max)], [pitch, yaw, roll]


def get_labels_images(dataset_dir):
    labels, images_path = [], []
    images = [x for x in os.listdir(dataset_dir) if 'jpg' in x]
    for img in images:
        images_path.append(os.path.join(dataset_dir, img))
        crop_vals, angle_vals = get_labels_from_mat(os.path.join(dataset_dir, img.replace('jpg', 'mat')))
        labels.append([crop_vals, angle_vals])
    return labels, images_path


def run(dataset_dir):
    assert dataset_dir != '', 'no dataset directory'
    labels, images = get_labels_images(dataset_dir)
    images_num = _create_tfrecord(labels, images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    args = parser.parse_args()
    run(args.img)
