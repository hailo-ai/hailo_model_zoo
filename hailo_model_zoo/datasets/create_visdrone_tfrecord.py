#!/usr/bin/env python

import argparse
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import PIL
import tensorflow as tf
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_from_drive

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

VAL_LOCATION = "models_files/visdrone/2020-05-25/visdrone_val.tfrecord"
CALIB_LOCATION = "models_files/visdrone/2021-07-25/visdrone_calib.tfrecord"
TFRECORD_LOCATION = {
    'val': VAL_LOCATION,
    'train': CALIB_LOCATION
}
DOWNLOAD_URL = {
    'train': 'https://drive.google.com/uc?export=download&id=1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn',
    'val': 'https://drive.google.com/uc?export=download&id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59',
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
    img_full_path = image_dir.joinpath(file_basename).with_suffix('.jpg')

    with open(img_full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = PIL.Image.open(io.BytesIO(encoded_jpg))

    image_width, image_height = image.size
    filename = img_full_path.name

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
    images = sorted(list(image_dir.glob('*.jpg')))

    overall_annotations = 0
    missing_annotation_count = 0
    total_num_annotations_skipped = 0

    progress_bar = tqdm(images)
    with tf.io.TFRecordWriter(output_path) as writer:
        for idx, image_filename in enumerate(progress_bar):
            file_basename = image_filename.with_suffix('').name
            annotation_filename = f"{file_basename}.txt"
            annotation_full_path = annotations_dir / annotation_filename
            if not annotation_full_path.is_file():
                missing_annotation_count += 1
                print(f'{annotation_filename} missing annotations.')
                continue

            with open(annotation_full_path, 'r') as f:
                annotation_file = f.read().splitlines()
            annotations = []
            for ann_str in annotation_file:
                try:
                    line_str = [int(i) for i in ann_str.rstrip(',').split(",")]
                    x, y, w, h, category_id = (line_str[0], line_str[1],
                                               line_str[2], line_str[3],
                                               line_str[5])
                except IndexError:
                    print(f"problem with {image_filename}")
                    raise
                annotations.append([x, y, w, h, category_id])

            tf_example, num_annotations_skipped, image_overall_annotations =\
                create_tf_example(file_basename, image_dir, annotations, idx)
            overall_annotations += image_overall_annotations
            total_num_annotations_skipped += num_annotations_skipped
            writer.write(tf_example.SerializeToString())

    print(f'Finished writing, skipped {total_num_annotations_skipped} annotations.')
    print(f'{overall_annotations} Annotations Found.')
    print(f'{missing_annotation_count} images are missing annotations.')


def _download_dataset(name, dataset_dir):
    if dataset_dir.is_dir():
        return
    dataset_root = dataset_dir.parent
    print(f'{dataset_dir} not found. Downloading...')
    with tempfile.NamedTemporaryFile() as outfile:
        out_filename = f'VisDrone2019-DET-{name}.zip'
        download_from_drive(DOWNLOAD_URL[name], outfile, desc=out_filename)

        with zipfile.ZipFile(outfile, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_root))


def run(args):
    tfrecord_path = path_resolver.resolve_data_path(TFRECORD_LOCATION[args.name])
    if tfrecord_path.exists():
        print(f'tfrecord already exists at {tfrecord_path}. Skipping...')
        return
    output_path = f"visdrone_{args.name}.tfrecord"
    dataset_path = Path(args.dir or f'VisDrone2019-DET-{args.name}')
    _download_dataset(args.name, dataset_path)
    _create_tf_record_from_visdrone_annotations(
        dataset_path / 'annotations',
        dataset_path / 'images',
        output_path)
    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(output_path, tfrecord_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', help="Path to dataset root", type=str)
    parser.add_argument('name', help="Which set to download",
                        type=str, default='val', choices=['train', 'val'])
    args = parser.parse_args()
    run(args)
