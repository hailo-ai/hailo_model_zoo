#!/usr/bin/env python

import random
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver

""" Usage example:
python create_personface_detector_tfrecord.py val
       --img /fastdata/users/amitk/people_detector/golden_val_person/images/val
       --det /fastdata/users/amitk/people_detector/golden_val_person/custom_val_dataset.json
       --num-images 128
       --seed 1111
"""


TF_RECORD_TYPE = 'val', 'calib'
TF_RECORD_LOC = {'val': Path(os.path.join(os.getcwd(), 'personface_detector_val.tfrecord')),
                 'calib': Path(os.path.join(os.getcwd(), 'personface_detector_calib.tfrecord'))}


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

    oobs = []
    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, bbox_annotations) in enumerate(progress_bar):
            if not bbox_annotations:
                raise ValueError(f"annotations is empty in {img_path}. Please check how did this happen")
            xmin, xmax, ymin, ymax, category_id, is_crowd, area = [], [], [], [], [], [], []
            img_jpeg = open(img_path, 'rb').read()
            img = np.array(Image.open(img_path))
            image_height = img.shape[0]
            image_width = img.shape[1]
            for object_annotations in bbox_annotations:
                (x, y, width, height) = tuple(object_annotations['bbox'])
                if width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
                    continue
                xmin_n = float(x) / image_width
                xmax_n = float(x + width) / image_width
                ymin_n = float(y) / image_width
                ymax_n = float(y + image_height) / image_width
                if np.array([xmin_n, xmax_n, ymin_n, ymax_n]).any() > 1.0:
                    oobs.append(img_path)
                xmin.append(np.clip(float(x) / image_width, 0.0, 1.0))
                xmax.append(np.clip(float(x + width) / image_width, 0.0, 1.0))
                ymin.append(np.clip(float(y) / image_height, 0.0, 1.0))
                ymax.append(np.clip(float(y + height) / image_height, 0.0, 1.0))
                is_crowd.append(object_annotations['iscrowd'])
                area.append(object_annotations['area'])
                category_id.append(int(object_annotations['category_id']))

                progress_bar.set_description(f"{name} #{i}: {img_path}")

            img_id_clip = object_annotations['image_id'] % (10 ** 16)
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'num_boxes': _int64_feature(len(bbox_annotations)),
                'image_id': _int64_feature(img_id_clip),
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
    if oobs:
        print(f"There are {len(oobs)} records")
        for ii, oob_path in enumerate(oobs):
            print(f"{ii+1}: {oob_path}")
    else:
        print("No Out-Of-Bound annotations!")
    return i + 1


def get_img_labels_list(dataset_dir, det_file, seed):
    with tf.io.gfile.GFile(str(det_file), 'r') as fid:
        obj_annotations = json.load(fid)

    img_to_obj_annotation = collections.defaultdict(list)
    for annotation in obj_annotations['annotations']:
        if str(annotation['image_id']).zfill(12) == str(annotation['image_id']):  # Not original COCO image
            img_id_int_bytes = annotation['image_id']
            image_name = img_id_int_bytes.to_bytes((img_id_int_bytes.bit_length() + 7) // 8, 'little').decode() + '.jpg'
        else:
            image_name = str(annotation['image_id']).zfill(12) + '.jpg'
        img_to_obj_annotation[image_name].append(annotation)

    orig_file_names, gt_annotations = [], []
    for img in sorted(dataset_dir.iterdir()):
        gt_annotations.append(img_to_obj_annotation[img.name])
        orig_file_names.append(str(img))
    files = list(zip(orig_file_names, gt_annotations))
    random.seed(seed)
    random.shuffle(files)
    return files


def run(dataset_dir, det_file, name, num_images, seed=42):
    if dataset_dir == '' or det_file == '':
        raise ValueError('Please specify images and detections directory.')
    img_labels_list = get_img_labels_list(Path(dataset_dir), Path(det_file), seed)
    images_num = _create_tfrecord(img_labels_list, name, num_images)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--det', '-det', help="detection ground truth", type=str, default='')
    parser.add_argument('--num-images', type=int, default=8192, help='Limit num images')
    parser.add_argument('--seed', type=int, default=42, help='Seed for dataset order shuffling')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    run(args.img, args.det, args.type, args.num_images, seed=args.seed)
