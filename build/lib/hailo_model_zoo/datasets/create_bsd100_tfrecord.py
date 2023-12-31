#!/usr/bin/env python

import os
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf
import PIL
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tempfile
import zipfile

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive


DESCRIPTION = "Create tfrecord for super resolution from High-Res and Low-Res images.\n" +\
              "Example cmd to create a bsd100_x4_val.tfrecord from High-Res and Low-Res images scaled by x4:\n\n" +\
              "\tpython create_bsd100_tfrecord.py val --lr BSD100/LRbicx4/ --hr BSD100/GTmod12/"

TF_RECORD_TYPE = 'val', 'calib'
TF_RECORD_LOC = {'val': 'models_files/espcn/2022-08-02/bsd100_x{}_val.tfrecord',
                 'calib': 'models_files/espcn/2022-08-02/bsd100_x{}_calib.tfrecord'}

DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id=1oOqJHTu2JIUz0qyEmVuSI_Nye36nioYX"


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def create_tf_example(img_path, hr_dir, upscale_factor):
    lr_image = PIL.Image.open(img_path)
    lr_width, lr_height = lr_image.size
    lr_filename = os.path.basename(img_path)

    hr_image = PIL.Image.open(os.path.join(hr_dir, lr_filename))

    if (lr_width * upscale_factor != hr_image.size[0]
       or lr_height * upscale_factor != hr_image.size[1]):
        raise ValueError('High and low resolution images do not agree with upscale factor')

    feature_dict = {
        'upscale_factor': _int64_feature(upscale_factor),
        'height': _int64_feature(lr_height),
        'width': _int64_feature(lr_width),
        'image_name': _bytes_feature(lr_filename.encode('utf8')),
        'lr_img': _bytes_feature(np.array(lr_image, np.uint8).tobytes()),
        'hr_img': _bytes_feature(np.array(hr_image, np.uint8).tobytes()),
        'format': _bytes_feature('png'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(lr_dir, hr_dir, name, max_imgs=100):
    upscale_factor = int(os.path.basename(lr_dir)[-1])
    tfrecord_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name].format(upscale_factor))
    (tfrecord_filename.parent).mkdir(parents=True, exist_ok=True)
    writer = tf.io.TFRecordWriter(str(tfrecord_filename))
    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(lr_dir)) for f in fn if f[-3:] == 'png']
    i = 0
    pbar = tqdm(images, total=max_imgs - 1)
    for img_path in pbar:
        try:
            tf_example = create_tf_example(img_path, hr_dir, upscale_factor)
            i += 1
            pbar.set_description(f"Converting image number {i}/{len(images)}: {img_path}")
        except (FileNotFoundError, TypeError):
            continue
        writer.write(tf_example.SerializeToString())
        if i >= max_imgs:
            break
    writer.close()
    return i


def _download_dataset(dataset_dir):
    if dataset_dir.is_dir():
        return
    dataset_dir.mkdir()
    with tempfile.NamedTemporaryFile() as outfile:
        out_filename = 'BSD100_dataset.zip'
        download_from_drive(DOWNLOAD_URL, outfile, desc=out_filename)
        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))


def run(args):
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    num_images = args.num_images if args.num_images is not None else (32 if args.type == 'calib' else 100)

    if args.lr is None or args.hr is None:
        if args.lr is not None or args.hr is not None:
            raise ValueError('Please use lr and hr arguments together.')
        dataset_path = Path('BSD100_dataset')
        _download_dataset(dataset_path)

        hr_dir = dataset_path / 'GTmod12'
        lr_dirs = [dataset_path / f'LRbicx{x}' for x in [2, 3, 4]]

    else:
        lr_dirs = [Path(args.lr)]
        hr_dir = Path(args.hr)

    for lr_dir in lr_dirs:
        tot_images = _create_tf_record(lr_dir, hr_dir, args.type, max_imgs=num_images)
        print('Done converting {} images'.format(tot_images))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--lr', help="low resolution images directory", type=str)
    parser.add_argument('--hr', help="high resolution images directory", type=str)

    parser.add_argument('--num-images', type=int, default=None, help='Limit num images')
    args = parser.parse_args()
    run(args)

"""
-----------------------------------------------------------------
CMD used to create a bsd100_x4.tfrecord dataset (assuming provided High-Res and Low-Res images are scaled by x4):
python create_bsd100_tfrecord.py val --lr BSD100/LRbicx4/ --hr BSD100/GTmod12/
"""
