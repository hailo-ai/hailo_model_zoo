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


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _create_tfrecord(images, num_images, out_name):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join('./', out_name)
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    with tf.Graph().as_default():
        image_placeholder = tf.compat.v1.placeholder(dtype=tf.uint8, name='image_placeholder')
        encoded_image = tf.image.encode_jpeg(image_placeholder)
        i = 0
        with tf.compat.v1.Session('') as sess:
            for img_path in images:
                print(img_path)
                img = np.array(Image.open(img_path), np.uint8)
                image_height = img.shape[0]
                image_width = img.shape[1]
                img_jpeg = sess.run(encoded_image, feed_dict={image_placeholder: img})

                print("converting image number {}: {}".format(i, img_path))
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(image_height),
                    'width': _int64_feature(image_width),
                    'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                    'image_jpeg': _bytes_feature(img_jpeg),
                }))
                writer.write(example.SerializeToString())
                i += 1
                if i > num_images:
                    break
            writer.close()
            return i


def get_image_files(data_dir, split_file):
    print('data dir: {}'.format(data_dir))
    print('split file: {}'.format(split_file))
    files = []
    if split_file == 'all':
        file_list = os.listdir(data_dir)
        for f in file_list:
            if os.path.isfile(os.path.join(data_dir, f)):
                files.append(os.path.join(data_dir, f))
        print('file list: {}'.format(files))
    else:
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                img = line + '.png'
                files.append(os.path.join(data_dir, img))
    return files


def run(data_dir, split_file, num_images, out_name):
    assert data_dir != '', 'no data directory'
    assert split_file != '', 'no split file'
    images = get_image_files(data_dir, split_file)
    images_num = _create_tfrecord(images, num_images, out_name)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="data directory", type=str, default='')
    parser.add_argument('--split', help="split file", type=str, default='')
    parser.add_argument('--num-images', help="limit the number of images", type=int, default=127)
    parser.add_argument('--out-name', help="name of output file", type=str, default='output.tfrecord')
    args = parser.parse_args()
    run(args.data, args.split, args.num_images, args.out_name)
