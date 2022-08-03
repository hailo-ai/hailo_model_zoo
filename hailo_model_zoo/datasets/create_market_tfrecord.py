import argparse
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from hailo_model_zoo.utils import path_resolver, downloader
import zipfile
import tensorflow as tf
import numpy as np
import random

TF_RECORD_TYPE = 'calib', 'val'
TF_RECORD_LOC = {'val': 'models_files/market1501/2022-06-22/market1501_val.tfrecord',
                 'calib': 'models_files/market1501/2022-06-22/market1501_calib.tfrecord'}


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _create_tfrecord(filenames, name, num_images):
    """Loop over all the images in filenames and create the TFRecord
    """
    tfrecords_filename = path_resolver.resolve_data_path(TF_RECORD_LOC[name])
    (tfrecords_filename.parent).mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(filenames[:num_images])
    with tf.io.TFRecordWriter(str(tfrecords_filename)) as writer:
        for i, (img_path, label_index, type, cam_id) in enumerate(progress_bar):
            img_jpeg = open(img_path, 'rb').read()
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            progress_bar.set_description(f"{name} #{i}: {img_path}")
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'label_index': _int64_feature(label_index),
                'cam_id': _int64_feature(cam_id),
                'image_name': _bytes_feature(str.encode(os.path.basename(img_path))),
                'type': _bytes_feature(str.encode(type)),
                'image_jpeg': _bytes_feature(img_jpeg)}))
            writer.write(example.SerializeToString())
    return i + 1


def _get_files_and_labels_list(dataset_dir, type):
    """Get a list of filenames and labels from the dataset directory
    """
    file_list = []
    for lib in Path(dataset_dir).iterdir():
        if lib.parts[-1][0] != '-' and lib.parts[-1][6].isdigit():
            file_list.append([str(lib), int(lib.parts[-1][0:4]), type, int(lib.parts[-1][6])])

    random.shuffle(file_list)
    return file_list


def download_dataset(type):
    dataset_dir = Path.cwd()
    dataset_dir = dataset_dir / 'market1501'
    # create the libraries if needed
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if not (dataset_dir / 'Market-1501-v15.09.15').is_dir():
        filename = downloader.download_file(
            'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(str(dataset_dir))

    if type == 'val':
        return os.path.join(str(dataset_dir), 'Market-1501-v15.09.15/')
    else:
        return os.path.join(str(dataset_dir), 'Market-1501-v15.09.15', 'bounding_box_train')


def run(type, dataset_dir, num_images):
    if dataset_dir == '':
        dataset_dir = download_dataset(type)
    if type == 'val':
        img_labels_list = _get_files_and_labels_list(dataset_dir + 'bounding_box_test', 'gallery')
        img_labels_list_ = _get_files_and_labels_list(dataset_dir + 'query', 'query')
        img_labels_list = img_labels_list + img_labels_list_
    else:
        img_labels_list = _get_files_and_labels_list(dataset_dir, type)
    images_num = _create_tfrecord(img_labels_list, type, num_images)
    print('Done converting {} images'.format(images_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='which tf-record to create {}'.format(TF_RECORD_TYPE))
    parser.add_argument('--img', '-img', help="images directory", type=str, default='')
    parser.add_argument('--num-images', type=int, default=None, help='Limit num images')
    args = parser.parse_args()
    assert args.type in TF_RECORD_TYPE, \
        'need to provide which kind of tfrecord to create {}'.format(TF_RECORD_TYPE)
    num_images = args.num_images if args.num_images is not None else (19281 if args.type == 'val' else 8192)
    run(args.type, args.img, num_images)
