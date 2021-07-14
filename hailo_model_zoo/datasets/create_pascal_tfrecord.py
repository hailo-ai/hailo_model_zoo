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


def _create_tfrecord(filenames, calib, name='val'):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = 'pascal_' + name + '.tfrecord'
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    i = 0
    for img_path, mask_path in filenames:
        img = open(img_path, "rb").read()
        img_pil = np.array(Image.open(img_path))
        image_height = img_pil.shape[0]
        image_width = img_pil.shape[1]
        mask = np.array(Image.open(mask_path), np.uint8)

        print("converting image number " + str(i) + ": " + img_path, end='\r')
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_height),
            'width': _int64_feature(image_width),
            'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
            'mask': _bytes_feature(mask.tostring()),
            'image_jpeg': _bytes_feature(img)}))
        writer.write(example.SerializeToString())
        i += 1
        if calib and i > 127:
            break
    writer.close()
    return i


def get_images_list(val_images):
    with open(val_images, 'r') as f:
        data = f.read()
    return data.split('\n')[:-1]


def get_img_labels_list(dataset_dir, seg_dir, val_images):
    image_file_names, mask_file_names = [], []
    img_val_list = get_images_list(val_images)
    for img in os.listdir(dataset_dir):
        if os.path.splitext(img)[0] in img_val_list:
            image_file_names.append(os.path.join(dataset_dir, img))
            mask_file_names.append(os.path.join(seg_dir, img.replace('.jpg', '.png')))
    return zip(image_file_names, mask_file_names)


def run(dataset_dir, seg_dir, val_images, calib, name='val'):
    assert dataset_dir != '', 'no dataset directory'
    assert seg_dir != '', 'no segmentation directory'
    assert val_images != '', 'no validation file'
    if calib and name != 'calib':
        print("Calibration set creation is chosen but file name suffix is {}... "
              "Setting it to 'calib'...".format(name))
        name = 'calib'
    img_labels_list = get_img_labels_list(dataset_dir, seg_dir, val_images)
    images_num = _create_tfrecord(img_labels_list, calib, name)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', help="images directory", type=str,
                        default='/local/data/pascal_seg/benchmark_RELEASE/dataset/img')
    parser.add_argument('--seg', help="segmentation directory", type=str,
                        default='/local/data/pascal_seg/benchmark_RELEASE/dataset/cls_png')
    parser.add_argument('--val', help="validation text file", type=str,
                        default='/local/data/pascal_seg/benchmark_RELEASE/dataset/val.txt')
    parser.add_argument('--calib', help="create calibration set", action='store_true')
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='val')
    args = parser.parse_args()
    run(args.img, args.seg, args.val, args.calib, args.name)
"""
----------------------------------------------------------------------------
CMD used to create a pascal_train.tfrecord of the Pascal VOC training dataset:
----------------------------------------------------------------------------
python create_pascal_tfrecord.py
--img /local/data/pascal_seg/benchmark_RELEASE/dataset/img
--seg /local/data/pascal_seg/benchmark_RELEASE/dataset/cls_png
--val /local/data/pascal_seg/benchmark_RELEASE/dataset/train.txt
--name train

----------------------------------------------------------------------------
CMD used to create a pascal_val.tfrecord of the Pascal VOC validation dataset:
----------------------------------------------------------------------------
python create_pascal_tfrecord.py
--img /local/data/pascal_seg/benchmark_RELEASE/dataset/img
--seg /local/data/pascal_seg/benchmark_RELEASE/dataset/cls_png
--val /local/data/pascal_seg/benchmark_RELEASE/dataset/val.txt
--name val
"""
