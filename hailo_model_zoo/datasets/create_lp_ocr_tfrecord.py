import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver


# DEF_IMG_DIR = '/data/data/lpr/plate_detection/plates_2021Dec01/val/images'
# DEF_LABEL_DIR = '/data/data/lpr/plate_detection/plates_2021Dec01/val/labels'

TF_RECORD_TYPE = 'train', 'val', 'calib'
TF_RECORD_LOC = {'train': Path(os.path.join(os.getcwd(), 'lp_ocr_train.tfrecord')),
                 'val': Path(os.path.join(os.getcwd(), 'lp_ocr_val.tfrecord')),
                 'calib': Path(os.path.join(os.getcwd(), 'lp_ocr_calib.tfrecord'))}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tf_record(filenames, name, num_images):
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path) in enumerate(progress_bar):
            img_jpeg = open(img_path, 'rb').read()
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]
            plate, _ = os.path.splitext(os.path.basename(img_path))

            progress_bar.set_description(f"{name} #{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'plate': _bytes_feature(str.encode(plate)),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return i + 1, tfrecords_filename


def run(img_dir, name, num_images, seed=0):
    # for root, dirs, files in os.walk(img_dir):
    img_list = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    images_num, tfrecord_name = _create_tf_record(img_list, name, num_images)
    print('\nDone converting {} images'.format(images_num))
    print(F'Dataset saved at {tfrecord_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='Which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="Images directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=5000, help='Limit num images')
    parser.add_argument('--seed', type=int, default=0, help='Seed for dataset order shuffling')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    run(args.img, args.type, args.num_images, seed=args.seed)


""" Usage example:

python create_lp_ocr_tfrecord.py val
       --img /data/data/lpr/plate_recognition/val/images/
"""
