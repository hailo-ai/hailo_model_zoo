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


def _create_tfrecord(labels, images, num_images):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join('./', 'kitti_val.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    with tf.Graph().as_default():
        image_placeholder = tf.compat.v1.placeholder(dtype=tf.uint8, name='image_placeholder')
        encoded_image = tf.image.encode_jpeg(image_placeholder)
        i = 0
        with tf.compat.v1.Session('') as sess:
            for img_path, label in zip(images, labels):
                img = np.array(Image.open(img_path), np.uint8)
                image_height = img.shape[0]
                image_width = img.shape[1]
                img_jpeg = sess.run(encoded_image, feed_dict={image_placeholder: img})
                depth = labels[i]

                print("converting image number {}: {}".format(i, img_path))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_height),
                    'width': _int64_feature(image_width),
                    'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                    'depth': _bytes_feature(np.array(depth, np.float32).tobytes()),
                    'image_jpeg': _bytes_feature(img_jpeg)}))
                writer.write(example.SerializeToString())
                i += 1
                if i > num_images:
                    break
            writer.close()
            return i


def get_label(gt_file):
    gt_depths = np.load(gt_file, fix_imports=True, encoding='latin1',
                        allow_pickle=True)["data"]
    return gt_depths


def get_image_files(data_dir, split_file):
    files = []
    with open(split_file, 'r') as f:
        for line in f:
            fdir = line[:line.find(' ')]
            cam = '2' if line[-2] == 'l' else '3'
            fdir = os.path.join(fdir, 'image_0' + cam, 'data')
            img = line[(line.find(' ') + 1):][:-3] + '.png'
            files.append(os.path.join(data_dir, fdir, img))
    return files


def run(data_dir, split_file, gt_file, num_images):
    assert data_dir != '', 'no data directory'
    assert split_file != '', 'no split file'
    assert gt_file != '', 'no gt file'
    images = get_image_files(data_dir, split_file)
    labels = get_label(gt_file)
    images_num = _create_tfrecord(labels, images, num_images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="data directory", type=str, default='')
    parser.add_argument('--split', help="split file", type=str, default='')
    parser.add_argument('--gt', help="gt npz file", type=str, default='')
    parser.add_argument('--num-images', help="limit the number of images", type=int, default=127)
    args = parser.parse_args()
    run(args.data, args.split, args.gt, args.num_images)
