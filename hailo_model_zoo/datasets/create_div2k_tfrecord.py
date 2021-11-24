#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
import PIL
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def create_tf_example(img_path, hr_dir):
    lr_image = PIL.Image.open(img_path)
    lr_width, lr_height = lr_image.size
    lr_filename = os.path.basename(img_path)

    hr_image = PIL.Image.open(os.path.join(hr_dir, lr_filename.replace('x4', '')))

    feature_dict = {
        'height': _int64_feature(lr_height),
        'width': _int64_feature(lr_width),
        'image_name': _bytes_feature(lr_filename.encode('utf8')),
        'lr_img': _bytes_feature(np.array(lr_image, np.uint8).tostring()),
        'hr_img': _bytes_feature(np.array(hr_image, np.uint8).tostring()),
        'format': _bytes_feature('png'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(lr_dir, hr_dir, output_path, max_imgs=100):
    writer = tf.io.TFRecordWriter(output_path)
    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(lr_dir)) for f in fn if f[-3:] == 'png']
    tot_images = 0
    for idx, image_path in enumerate(images):
        try:
            tf_example = create_tf_example(image_path, hr_dir)
            tf.compat.v1.logging.info('On image %d of %d', idx, len(images))
        except (FileNotFoundError, TypeError):
            continue
        writer.write(tf_example.SerializeToString())
        if tot_images > max_imgs:
            break
        tot_images += 1
    writer.close()
    return tot_images + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help="low res images directory", type=str,
                        default='/local/data/datasets/Div2k/DIV2K_valid_LR_bicubic/X4')
    parser.add_argument('--hr', help="high res images directory", type=str,
                        default='/local/data/datasets/Div2k/DIV2K_valid_HR')
    parser.add_argument('--out', help="output directory", type=str, default="./")
    parser.add_argument('--calib', help="create calibration set", action='store_true')
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='validation')
    args = parser.parse_args()
    max_imgs = 100
    if args.calib and args.name != 'calib':
        print("Calibration set creation is chosen but file name suffix is {}... "
              "Setting it to 'calib'...".format(args.name))
        args.name = 'calib'
        max_imgs = 30
    elif 'train' in args.lr:
        max_imgs = 400
    tfrecord_fname = 'hailo_' + args.name + '_set_div2k_super_resolution.tfrecord'
    tot_images = _create_tf_record(args.lr, args.hr, os.path.join(args.out, tfrecord_fname), max_imgs=max_imgs)
    print('Done converting {} images'.format(tot_images))


"""
Downloading data for validation set:
---------------
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip -q DIV2K_valid_LR_bicubic_X4.zip
unzip -q DIV2K_valid_HR.zip

Downloading data for training set:
-------------
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip -q DIV2K_valid_LR_bicubic_X4.zip
unzip -q DIV2K_valid_HR.zip

----------------------------------------------------------------------------
CMD used to create a hailo_validation_set_div2k_super_resolution.tfrecord of the DIV2K validation dataset:
----------------------------------------------------------------------------
python create_widerface_tfrecord.py
--lr /local/data/datasets/Div2k/DIV2K_valid_LR_bicubic/X4/
--hr /local/data/datasets/Div2k/DIV2K_valid_HR/
--name validation

----------------------------------------------------------------------------
CMD used to create a hailo_train_set_div2k_super_resolution.tfrecord of the DIV2K training dataset:
----------------------------------------------------------------------------
python create_widerface_tfrecord.py
--lr /local/data/datasets/Div2k/DIV2K_train_LR_bicubic/X4/
--hr /local/data/datasets/Div2k/DIV2K_train_HR/
--name train
"""
