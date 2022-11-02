#!/usr/bin/env python
import os
import argparse
import shutil
import tarfile
import tempfile
from collections import namedtuple
from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.downloader import download_to_file

Dataset = namedtuple('Dataset', ['download_url', 'tfrecord_location', 'default_root',
                     'dataset_dir', 'annotations_dir', 'split_name', 'split_dir', 'path_in_tar', 'mask_extension'])
VOC = Dataset(
    download_url="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    tfrecord_location={
        'calib': 'models_files/pascal_voc/2021-08-15/pascal_voc_train.tfrecord',
        'val': 'models_files/pascal_voc/2021-08-15/pascal_voc_val.tfrecord'
    },
    default_root='VOCdevkit/VOC2012',
    dataset_dir='JPEGImages',
    annotations_dir='SegmentationClass',
    split_name={
        'calib': 'train',
        'val': 'val'
    },
    split_dir='ImageSets/Segmentation',
    path_in_tar='VOCdevkit/VOC2012',
    mask_extension='.png'
)
AUG = Dataset(
    download_url='http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz',
    tfrecord_location={
        'calib': 'models_files/pascal_aug/2021-08-15/pascal_aug_train.tfrecord',
        'val': 'models_files/pascal_aug/2021-08-15/pascal_aug_val.tfrecord'
    },
    default_root='benchmark_RELEASE/dataset',
    dataset_dir='img',
    annotations_dir='cls',
    split_name={
        'calib': 'train',
        'val': 'val'
    },
    split_dir='.',
    path_in_tar='benchmark_RELEASE/dataset',
    mask_extension='.mat'
)

DATASETS = {
    'aug': AUG,
    'voc': VOC
}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _download_dataset(dataset, name, root_dir):
    if not root_dir.is_dir():
        print(f'Image directory not found in {root_dir}. Downloading...')
        with tempfile.NamedTemporaryFile() as outfile, tempfile.TemporaryDirectory() as temp_dir:
            download_to_file(dataset.download_url, outfile)

            with tarfile.open(outfile.name) as tar_ref:
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
                    
                
                safe_extract(tar_ref, temp_dir)
            shutil.move(Path(temp_dir) / dataset.path_in_tar, root_dir)


def _create_tfrecord(filenames, name='val'):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = f'pascal_{name}.tfrecord'
    filenames = list(filenames)
    progress_bar = tqdm(filenames)
    filenames_iterator = enumerate(progress_bar)

    with tf.io.TFRecordWriter(tfrecords_filename) as writer:
        for i, (img_path, mask_path) in filenames_iterator:
            img = open(img_path, "rb").read()
            img_pil = np.array(Image.open(img_path))
            image_height = img_pil.shape[0]
            image_width = img_pil.shape[1]
            if mask_path.endswith('.png'):
                mask = np.array(Image.open(mask_path), np.uint8)
            else:
                assert mask_path.endswith('.mat')
                mask = loadmat(mask_path)['GTcls']['Segmentation'][0][0]

            progress_bar.set_description(f"#{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(image_height),
                'width': _int64_feature(image_width),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'mask': _bytes_feature(mask.tostring()),
                'image_jpeg': _bytes_feature(img)}))
            writer.write(example.SerializeToString())
    images_num = i + 1
    print('\nDone converting {} images'.format(images_num))
    return tfrecords_filename


def get_images_list(val_images):
    with open(val_images, 'r') as f:
        data = f.read()
    return data.split('\n')[:-1]


def get_img_labels_list(dataset_dir, seg_dir, val_images, mask_extension):
    image_file_names, mask_file_names = [], []
    img_val_list = get_images_list(val_images)
    for img in os.listdir(dataset_dir):
        if os.path.splitext(img)[0] in img_val_list:
            image_file_names.append(os.path.join(dataset_dir, img))
            mask_file_names.append(os.path.join(seg_dir, img.replace('.jpg', mask_extension)))
    return zip(image_file_names, mask_file_names)


def run(dataset_name, name, root):
    dataset = DATASETS[dataset_name]
    root = Path(root or dataset.default_root)
    tfrecord_path = path_resolver.resolve_data_path(dataset.tfrecord_location[name])
    if tfrecord_path.exists():
        print(f'tfrecord already exists at {tfrecord_path}. Skipping...')
        return
    _download_dataset(dataset, name, root)
    dataset_dir = str(root / dataset.dataset_dir)
    seg_dir = str(root / dataset.annotations_dir)
    set_name = dataset.split_name[name]
    val_images = str(root / dataset.split_dir / f'{set_name}.txt')
    img_labels_list = get_img_labels_list(dataset_dir, seg_dir, val_images, dataset.mask_extension)
    result_tfrecord_path = _create_tfrecord(img_labels_list, name)
    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result_tfrecord_path, tfrecord_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help="TFRecord of which dataset to create", type=str,
                        choices=['calib', 'val'])
    parser.add_argument('dataset', help="Which variant of the dataset to create", type=str,
                        choices=['voc', 'aug'], default='aug', nargs='?')
    parser.add_argument('--root', help="dataset root directory", type=str,
                        default=None)
    args = parser.parse_args()
    run(args.dataset, args.type, args.root)
