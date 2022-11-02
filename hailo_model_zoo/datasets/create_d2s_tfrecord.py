#!/usr/bin/env python


import random
import argparse
import collections
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver, downloader


TF_RECORD_TYPE = 'val', 'calib'

TF_RECORD_LOC = {'val': 'models_files/d2s/2021-10-24/D2S_val.tfrecord',
                 'calib': 'models_files/d2s/2021-10-24/D2S_calib.tfrecord'}

DOWNLOAD_URL = {'dataset': ("https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/"
                            "download/420938639-1629953496/d2s_images_v1.tar.xz"),
                'annotations': ("https://www.mydrive.ch/shares/39000/993e79a47832a8ea7208a14d8b277c35/"
                                "download/420938386-1629953481/d2s_annotations_v1.1.tar.xz")}

JSON_NAME = {'val': 'D2S_validation.json',
             'calib': 'D2S_augmented.json'}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _create_tfrecord(filenames, name, num_images):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for idx, (img_path, bbox_annotations, img_shape) in enumerate(progress_bar):
            xmin, xmax, ymin, ymax, category_id, is_crowd, area = [], [], [], [], [], [], []
            img_jpeg = open(img_path, 'rb').read()
            image_height = img_shape[0]
            image_width = img_shape[1]
            for object_annotations in bbox_annotations:
                (x, y, width, height) = tuple(object_annotations['bbox'])
                if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                    continue
                xmin.append(float(x) / image_width)
                xmax.append(float(x + width) / image_width)
                ymin.append(float(y) / image_height)
                ymax.append(float(y + height) / image_height)
                is_crowd.append(object_annotations['iscrowd'])
                area.append(object_annotations['area'])
                category_id.append(int(object_annotations['category_id']))
                progress_bar.set_description(f"{name} #{idx}: {img_path}")

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'num_boxes': _int64_feature(len(bbox_annotations)),
                'image_id': _int64_feature(object_annotations['image_id']),
                'xmin': _float_list_feature(xmin),
                'xmax': _float_list_feature(xmax),
                'ymin': _float_list_feature(ymin),
                'ymax': _float_list_feature(ymax),
                'area': _float_list_feature(area),
                'category_id': _int64_feature(category_id),
                'is_crowd': _int64_feature(is_crowd),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'mask': _bytes_feature(np.array(0).tostring()),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return idx + 1


def _get_img_labels_list(dataset_dir, det_file):
    with tf.io.gfile.GFile(det_file, 'r') as fid:
        obj_annotations = json.load(fid)

    img_to_obj_annotation = collections.defaultdict(list)
    for annotation in obj_annotations['annotations']:
        image_name = "D2S_{0}.jpg".format(str(annotation['image_id']).zfill(6))
        img_to_obj_annotation[image_name].append(annotation)

    orig_file_names, det_annotations, imgs_shape = [], [], []
    for img in obj_annotations['images']:
        img_filename = img['file_name']
        det_annotations.append(img_to_obj_annotation[img_filename])
        orig_file_names.append(os.path.join(dataset_dir, img_filename))
        imgs_shape.append((img['height'], img['width']))
    files = list(zip(orig_file_names, det_annotations, imgs_shape))
    random.seed(0)
    random.shuffle(files)
    return files


def download_dataset(name):
    dataset_dir = path_resolver.resolve_data_path('d2s')
    dataset_annotations = dataset_dir / 'annotations'
    dataset_images = dataset_dir / 'images'

    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # download images if needed
    if not dataset_images.is_dir():
        filename = downloader.download_file(DOWNLOAD_URL['dataset'])  # 'd2s_images_v1.tar.xz'  #
        with tarfile.open(filename, 'r') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, str(dataset_dir))
        Path(filename).unlink()

    # download annotations if needed
    if not dataset_annotations.is_dir():
        filename = downloader.download_file(DOWNLOAD_URL['annotations'])
        with tarfile.open(filename, 'r') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, str(dataset_dir))

    anno_filename = JSON_NAME[name]
    anno_dest_file = dataset_annotations / anno_filename
    return dataset_images, anno_dest_file


def _copy_annotations_file(anno_source_file):
    # for instance segmentation we copy the json file
    anno_dest_folder = path_resolver.resolve_data_path(TF_RECORD_LOC['val']).parent
    anno_src_file = Path(anno_source_file).parent / JSON_NAME['val']
    shutil.copy(anno_src_file, anno_dest_folder)


def run(dataset_dir, det_file, name='val', num_images=8192):
    if dataset_dir == '' or det_file == '':
        if dataset_dir != '' or det_file != '':
            raise ValueError('Please use img and det arguments together.')
        dataset_dir, det_file = download_dataset(name)
    img_labels_list = _get_img_labels_list(Path(dataset_dir), Path(det_file))
    images_num = _create_tfrecord(img_labels_list, name, num_images)
    _copy_annotations_file(det_file)
    print(f"Done converting {images_num} images")
    print(f"Output saved to: {path_resolver.resolve_data_path(TF_RECORD_LOC[name]).parent}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--det', '-det', help="detection ground truth", type=str, default='')
    parser.add_argument('--num-images', type=int, default=8192, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'Need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    run(args.img, args.det, args.type, args.num_images)
"""
----------------------------------------------------------------------------
CMD used to create a D2S_train.tfrecord for D2S training dataset:
----------------------------------------------------------------------------
python create_d2s_tfrecord.py
--img /local/data/MVTec/D2S/images
--det /local/data/MVTec/D2S/annotations/D2S_training.json
--name train

----------------------------------------------------------------------------
CMD used to create a D2S_test.tfrecord for D2S validation dataset:
----------------------------------------------------------------------------
python create_d2s_tfrecord.py
--img /local/data/MVTec/D2S/images
--det /local/data/MVTec/D2S/annotations/D2S_validation.json
--name test
"""
