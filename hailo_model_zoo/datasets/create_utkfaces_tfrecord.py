import argparse
import os

import numpy as np
import tensorflow as tf

from PIL import Image


CALIB_SET_LENGTH = 128


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _create_tfrecord(dataset_dir, name, is_calibration):
    """
    Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join(dataset_dir, 'utkfaces_' + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)

    all_file_paths = []
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1] in ['.jpg', '.png']:
                all_file_paths.append(os.path.join(dataset_dir, root, name))
    all_file_paths.sort()

    if is_calibration:
        jump_size = len(all_file_paths) // (CALIB_SET_LENGTH - 1)
        all_file_paths = all_file_paths[::jump_size]

    i = 0
    for i, img_path in enumerate(all_file_paths):
        img_jpeg = open(img_path, 'rb').read()
        img = np.array(Image.open(img_path))
        image_height = img.shape[0]
        image_width = img.shape[1]

        img_name = os.path.basename(img_path).replace('.jpg', '')
        image_metadata = img_name.split('_')
        age = int(image_metadata[0])
        is_female_int = int(image_metadata[1])

        print("converting image number {} from {}\r".format(str(i), img_name), end='')
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_height),
            'width': _int64_feature(image_width),
            'age': _int64_feature(age),
            'is_female_int': _int64_feature(is_female_int),
            'image_id': _int64_feature(i),
            'image_name': _bytes_feature(str.encode(img_name)),
            'image_jpeg': _bytes_feature(img_jpeg)}))
        writer.write(example.SerializeToString())
    writer.close()

    return i


def run(dataset_dir, is_calibration):
    assert dataset_dir != '', 'no dataset directory'
    images_num = _create_tfrecord(dataset_dir, 'calibration_set' if is_calibration else 'val',
                                  is_calibration)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--calibration_set', '-c',
                        help="Create a calibration set of 128 images using every DATASET_LEN/128 image",
                        action='store_true', default=False)
    args = parser.parse_args()
    run(args.img, args.calibration_set)
