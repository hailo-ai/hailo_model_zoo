#!/usr/bin/env python

import random
import argparse
import collections
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import wget
import tensorflow as tf
from PIL import Image

from hailo_model_zoo.utils import path_resolver


TF_RECORD_TYPE = 'val2017', 'calib2017'
TF_RECORD_LOC = {'val2017': 'models_files/coco/2021-06-18/coco_val2017.tfrecord',
                 'calib2017': 'models_files/coco/2021-06-18/coco_calib2017.tfrecord'}


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
    writer = tf.io.TFRecordWriter(str(tfrecords_filename))

    i = 0
    for img_path, bbox_annotations in filenames:
        xmin, xmax, ymin, ymax, category_id, is_crowd, area = [], [], [], [], [], [], []
        img_jpeg = open(img_path, 'rb').read()
        img = np.array(Image.open(img_path))
        image_height = img.shape[0]
        image_width = img.shape[1]
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

        print("converting image number " + str(i) + " from " + name + " : " + img_path, end='\r')
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
            'image_jpeg': _bytes_feature(img_jpeg)}))
        writer.write(example.SerializeToString())
        i += 1
        if i >= num_images:
            break
    writer.close()
    return i


def get_img_labels_list(dataset_dir, det_file):
    with tf.io.gfile.GFile(str(det_file), 'r') as fid:
        obj_annotations = json.load(fid)

    img_to_obj_annotation = collections.defaultdict(list)
    for annotation in obj_annotations['annotations']:
        image_name = str(annotation['image_id']).zfill(12) + '.jpg'
        img_to_obj_annotation[image_name].append(annotation)

    orig_file_names, det_annotations = [], []
    for img in sorted(dataset_dir.iterdir()):
        det_annotations.append(img_to_obj_annotation[img.name])
        orig_file_names.append(str(img))
    files = list(zip(orig_file_names, det_annotations))
    random.seed(0)
    random.shuffle(files)
    return files


def download_dataset(name):
    dataset_name = 'val2017' if 'val' in name else 'train2017'
    dataset_dir = path_resolver.resolve_data_path('coco')
    dataset_annotations = dataset_dir / 'annotations'

    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # download images if needed
    if not (dataset_dir / dataset_name).is_dir():
        filename = wget.download('http://images.cocodataset.org/zips/' + dataset_name + '.zip')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))
        Path(filename).unlink()

    # download annotations if needed
    if not dataset_annotations.is_dir():
        filename = wget.download('http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))

    # for pose estimation / instance segmentation we copy the json file
    if 'val' in name:
        anno_folder = path_resolver.resolve_data_path(TF_RECORD_LOC[name]).parent
        anno_filename = anno_folder / ('person_keypoints_' + dataset_name + '.json')
        (Path(anno_filename).parent).mkdir(parents=True, exist_ok=True)
        shutil.copy(dataset_annotations / ('person_keypoints_' + dataset_name + '.json'), anno_filename)
        anno_filename = anno_folder / ('instances_' + dataset_name + '.json')
        shutil.copy(dataset_annotations / ('instances_' + dataset_name + '.json'), anno_filename)

    return dataset_dir / dataset_name, dataset_annotations / ('instances_' + dataset_name + '.json')


def run(dataset_dir, det_file, name, num_images):
    if dataset_dir == '' or det_file == '':
        if dataset_dir != '' or det_file != '':
            raise ValueError('Please use img and det arguments together.')
        dataset_dir, det_file = download_dataset(name)
    img_labels_list = get_img_labels_list(Path(dataset_dir), Path(det_file))
    images_num = _create_tfrecord(img_labels_list, name, num_images)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--det', '-det', help="detection ground truth", type=str, default='')
    parser.add_argument('--num-images', type=int, default=8192, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    run(args.img, args.det, args.type, args.num_images)
