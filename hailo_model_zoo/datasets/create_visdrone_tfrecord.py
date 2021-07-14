#!/usr/bin/env python

import os
import argparse
import io
import tensorflow as tf
import numpy as np
import PIL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

CLASS_MAPS = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11
}
CATEGORY_INDEX_MAP = {
    1: 'pedestrian',
    2: 'person',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others',
}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _float_list_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def create_tf_example(file_basename,
                      image_dir,
                      annotations,
                      image_idx):
    """Converts image and annotations to a tf.Example proto.

    Args:
      file_basename: str,
      image_dir: directory containing the image files.
      annotations: list of annotations corresponding to the image file.
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_full_path = os.path.join(image_dir, file_basename + ".jpg")

    with tf.io.gfile.GFile(img_full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    image_width, image_height = image.size
    filename = file_basename + ".jpg"

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    num_annotations_skipped = 0
    overall_annotations = 0
    for ann in annotations:
        x, y, width, height, category_id_original = ann
        if category_id_original == 0:
            num_annotations_skipped += 1
            continue
        category_id = CLASS_MAPS[category_id_original]
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(0)
        category_ids.append(category_id)
        category_names.append(CATEGORY_INDEX_MAP[category_id].encode('utf8'))
        area.append(width * height)
        overall_annotations += 1
    feature_dict = {
        'height': _int64_feature(image_height),
        'width': _int64_feature(image_width),
        'num_boxes': _int64_feature(overall_annotations),
        'image_id': _int64_feature(image_idx),
        'xmin': _float_list_feature(xmin),
        'xmax': _float_list_feature(xmax),
        'ymin': _float_list_feature(ymin),
        'ymax': _float_list_feature(ymax),
        'area': _float_list_feature(area),
        'category_id': _int64_feature(category_ids),
        'is_crowd': _int64_feature(is_crowd),
        'image_name': _bytes_feature(filename.encode('utf8')),
        'image_jpeg': _bytes_feature(encoded_jpg),
        'mask': _bytes_feature(np.array(0).tostring()),
        'text': _bytes_feature(category_names),
        'format': _bytes_feature('jpeg'.encode('utf8')),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_annotations_skipped, overall_annotations


def _create_tf_record_from_visdrone_annotations(annotations_dir, image_dir, output_path):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
      annotations_dir: Directory with all the annotations in txt files.
      image_dir: Directory containing the image files.
      output_path: Path to output tf.Record file.
      category_index_file: Path to category json file
      num_shards: number of output file shards.
    """
    writer = tf.io.TFRecordWriter(output_path)
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f[-3:] == 'jpg']

    overall_annotations = 0
    missing_annotation_count = 0
    total_num_annotations_skipped = 0
    for idx, image_filename in enumerate(images):
        if idx % 100 == 0:
            tf.compat.v1.logging.info('On image %d of %d', idx, len(images))
        file_basename = image_filename[:-4]
        annotation_filename = file_basename + ".txt"
        if not os.path.exists(os.path.join(annotations_dir, annotation_filename)):
            missing_annotation_count += 1
            tf.compat.v1.logging.info('{} missing annotations.'.format(annotation_filename))
            continue

        annotation_full_path = os.path.join(annotations_dir, file_basename + ".txt")
        with open(annotation_full_path, 'r') as f:
            annotation_file = f.read().splitlines()
        annotations = []
        for ann_str in annotation_file:
            try:
                line_str = [i for i in ann_str.split(",")]
                x, y, w, h, category_id = (int(line_str[0]), int(line_str[1]),
                                           int(line_str[2]), int(line_str[3]),
                                           int(line_str[5]))
            except IndexError:
                print("problem with {0}".format(image_filename))
            annotations.append([x, y, w, h, category_id])

        tf_example, num_annotations_skipped, image_overall_annotations =\
            create_tf_example(file_basename, image_dir, annotations, idx)
        overall_annotations += image_overall_annotations
        total_num_annotations_skipped += num_annotations_skipped
        writer.write(tf_example.SerializeToString())

    tf.compat.v1.logging.info('Finished writing, skipped %d annotations.', total_num_annotations_skipped)
    tf.compat.v1.logging.info('%d Annotations Found.', overall_annotations)
    tf.compat.v1.logging.info('%d images are missing annotations.', missing_annotation_count)
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-img', help="images directory", type=str,
                        default='/local/data/datasets/visDrone/VisDrone2019-DET-val/images/')
    parser.add_argument('--det', '-det', help="detection annotations dir", type=str,
                        default='/local/data/datasets/visDrone/VisDrone2019-DET-val/annotations')
    parser.add_argument('--output-dir', help="output directory", type=str, default='')
    parser.add_argument('--name', '-name', help="file name suffix", type=str, default='val')
    args = parser.parse_args()
    output_path = os.path.join(args.output_dir, "visdrone_" + args.name + ".tfrecord")
    _create_tf_record_from_visdrone_annotations(
        args.det,
        args.img,
        output_path)
"""
----------------------------------------------------------------------------
CMD used to create a visdrone_val.tfrecord for the VisDrone validation dataset:
----------------------------------------------------------------------------
python create_visdrone_tfrecord.py
--img /local/data/datasets/visDrone/VisDrone2019-DET-val/images/
--det /local/data/datasets/visDrone/VisDrone2019-DET-val/annotations
--name val

----------------------------------------------------------------------------
CMD used to create visdrone_train.tfrecord for the VisDrone training dataset:
----------------------------------------------------------------------------
python create_visdrone_tfrecord.py
--img /local/data/datasets/visDrone/VisDrone2019-DET-train/images/
--det /local/data/datasets/visDrone/VisDrone2019-DET-train/annotations
--name train
"""
