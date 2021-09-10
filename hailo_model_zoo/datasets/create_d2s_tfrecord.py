#!/usr/bin/env python

import os
import argparse
import tensorflow as tf
import numpy as np
import json
import collections


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _create_tfrecord(filenames, dataset_dir, name):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = os.path.join(dataset_dir, 'D2S_' + name + '.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecords_filename)
    for idx, (img_path, bbox_annotations, img_shape) in enumerate(list(filenames)):
        if idx % 100 == 0:
            print("Finished {0}".format(idx), end="\r")
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

        # print ("converting image number " + str(idx) + " from " + name + " : " + img_path)
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
    writer.close()
    return idx


def get_img_labels_list(dataset_dir, det_file):
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
    return zip(orig_file_names, det_annotations, imgs_shape)


def run(dataset_dir, det_file, name='test'):
    assert dataset_dir != '', 'no dataset directory'
    assert det_file != '', 'no detection file'
    img_labels_list = get_img_labels_list(dataset_dir, det_file)
    images_num = _create_tfrecord(img_labels_list, dataset_dir, name)
    print('\nDone converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str,
                        default='/local/data/MVTec/D2S/images')
    parser.add_argument('--det', '-det', help="detection ground truth", type=str,
                        default='/local/data/MVTec/D2S/annotations/D2S_validation.json')
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='test')
    args = parser.parse_args()
    run(args.img, args.det, args.name)
"""
----------------------------------------------------------------------------
CMD used to create a D2S_train.tfrecord for D2S training dataset:
----------------------------------------------------------------------------
python create_d2s_tfrecord.py
--img /local/data/MVTec/D2S/images
--det --det /local/data/MVTec/D2S/annotations/D2S_training.json
--name train

----------------------------------------------------------------------------
CMD used to create a D2S_test.tfrecord for D2S validation dataset:
----------------------------------------------------------------------------
python create_d2s_tfrecord.py
--img /local/data/MVTec/D2S/images
--det /local/data/MVTec/D2S/annotations/D2S_validation.json
--name test
"""
