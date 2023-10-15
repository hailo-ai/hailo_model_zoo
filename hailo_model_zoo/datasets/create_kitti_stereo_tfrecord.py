#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
from hailo_model_zoo.utils import path_resolver
from PIL import Image
import numpy as np
from tqdm import tqdm

TF_RECORD_TYPE = 'calib', 'val'
TF_RECORD_LOC = {'calib': 'models_files/kitti_stereo/kitti_stereo_calib.tfrecord',
                 'val': 'models_files/kitti_stereo/kitti_stereo_val.tfrecord'}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(labels_left, labels_right, images_left, images_right, num_images, name):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)
    writer = tf.io.TFRecordWriter(str(tfrecords_filename))

    progress_bar = tqdm(zip(images_left[:num_images], images_right[:num_images],
                        labels_left[:num_images], labels_right[:num_images]))
    for i, (img_l_path, img_r_path, label_l_path, label_r_path) in enumerate(progress_bar):
        img_l = np.array(Image.open(img_l_path), dtype=np.uint8)
        img_l_png = tf.image.encode_png(img_l)
        img_r = np.array(Image.open(img_r_path), dtype=np.uint8)
        img_r_png = tf.image.encode_png(img_r)
        label_l = np.array(Image.open(label_l_path), dtype=float)
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'height_l': _int64_feature(img_l.shape[0]),
                                   'width_l': _int64_feature(img_l.shape[1]),
                                   'height_r': _int64_feature(img_r.shape[0]),
                                   'width_r': _int64_feature(img_r.shape[1]),
                                   'image_l_name': _bytes_feature(str.encode(os.path.basename(img_l_path))),
                                   'image_r_name': _bytes_feature(str.encode(os.path.basename(img_r_path))),
                                   'image_l_png': _bytes_feature(img_l_png.numpy()),
                                   'image_r_png': _bytes_feature(img_r_png.numpy()),
                                   'label_l_name': _bytes_feature(str.encode(label_l_path)),
                                   'label_r_name': _bytes_feature(str.encode(label_r_path)),
                                   'label_l': _bytes_feature(np.array(label_l, np.float32).tobytes())
                                   }))
        writer.write(example.SerializeToString())
    writer.close()
    return i + 1


def get_label(data_dir, cam, name):
    gt_dir = os.path.join(data_dir, 'training', 'disp_occ_0' if cam == "left" else 'disp_occ_1')
    gt_list = os.listdir(gt_dir)
    gt_list.sort()
    labels = [os.path.join(gt_dir, img) for img in gt_list if img.find('_10') > -1]
    if name == "calib":
        return labels[:160]
    return labels[160:]


def get_image_files(data_dir, cam, name):
    image_dir = os.path.join(data_dir, 'training', 'image_2' if cam == "left" else 'image_3')
    img_list = os.listdir(image_dir)
    img_list.sort()
    image = [os.path.join(image_dir, img) for img in img_list if img.find('_10') > -1]
    if name == "calib":
        return image[:160]
    return image[160:]


def run(data_dir, num_images, name):
    assert data_dir != '', 'no data directory'
    images_left = get_image_files(data_dir, cam='left', name=name)
    images_right = get_image_files(data_dir, cam='right', name=name)
    labels_left = get_label(data_dir, cam='left', name=name)
    labels_right = get_label(data_dir, cam='right', name=name)
    images_num = _create_tfrecord(labels_left, labels_right, images_left, images_right, num_images, name)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=('examples:\n'
                                             'python create_kitti_stereo_tfrecord.py calib --data'
                                             + '<TRAIN_DIR>\n'
                                             'python create_kitti_stereo_tfrecord.py val '
                                             + '--data <VALIDATION_DIR>\n'))
    parser.add_argument('--data', help="data directory", type=str, default='')
    parser.add_argument('--num-images', help="limit the number of images", type=int, default=160)
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    args = parser.parse_args()
    run(args.data, args.num_images, args.type)
