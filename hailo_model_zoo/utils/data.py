from builtins import object
import os
import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.datasets import dataset_factory
from hailo_model_zoo.utils.video_utils import VideoCapture


def _open_image_file(img_path):
    image = tf.io.read_file(img_path)
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.uint8)
    image_name = tf.compat.v1.string_split([img_path], os.path.sep).values[-1]
    return image, {
        'img_orig': image,
        'image_name': image_name,
    }


def _read_npz(item):
    img_name = item.decode()
    data = np.load(img_name, allow_pickle=True)
    base_name = os.path.basename(img_name).replace('.npz', '')
    data = {key: data[key].item() for key in data}
    image_info = data[base_name]['image_info']
    rpn_boxes = image_info['rpn_proposals']
    num_rpn_boxes = image_info['num_rpn_proposals']
    return data[base_name]['logits'], rpn_boxes, num_rpn_boxes, image_info['image_name'], \
        image_info['image_id']


def _open_featuremap(img_path):
    featuremap, rpn_boxes, num_rpn_boxes, \
        image_name, image_id = tf.compat.v1.py_func(_read_npz, [img_path], [tf.float32, tf.float32,
                                                                            tf.int64, tf.string, tf.int32])
    return featuremap, {"rpn_proposals": rpn_boxes,
                        "num_rpn_boxes": num_rpn_boxes,
                        "image_name": image_name,
                        "image_id": image_id}


def _parse_video_frame(image, name):
    image = tf.cast(image, tf.uint8)
    return image, {'label_index': tf.cast(0, tf.float32),
                   'img_orig': image,
                   'image_name': name,
                   'is_same': tf.cast(0, tf.float32),
                   'mask': tf.image.rgb_to_grayscale(image)}


def _video_generator(video_path):
    def _video_generator_implementation():
        filename = os.path.basename(video_path)
        base, _ = os.path.splitext(filename)
        with VideoCapture(video_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            required_digits = len(str(total_frames))
            number_format = '{{:0{}d}}'.format(required_digits)
            name_format = '{}_frame_' + number_format + '.png'
            frame_count = 0
            success = True
            while success:
                success, image = cap.read()
                if success:
                    image_name = name_format.format(base, frame_count)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    yield image, image_name
                frame_count += 1
    return _video_generator_implementation


class DataFeed(object):
    """DataFeed class. Use this class to handle input data"""

    def __init__(self, preprocessing_callback, batch_size=8):
        self._preproc_callback = preprocessing_callback
        self._batch_size = batch_size

    @property
    def iterator(self):
        return tf.compat.v1.data.make_initializable_iterator(self._dataset)


class TFRecordFeed(DataFeed):
    def __init__(self, preprocessing_callback, batch_size, tfrecord_file, dataset_name):
        super().__init__(preprocessing_callback, batch_size=batch_size)
        parse_func = dataset_factory.get_dataset_parse_func(dataset_name)
        dataset = tf.data.TFRecordDataset([str(tfrecord_file)]).map(parse_func)
        if self._preproc_callback:
            dataset = dataset.map(self._preproc_callback)
        self._dataset = dataset if batch_size is None else dataset.batch(self._batch_size)


def _dataset_from_folder(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            if os.path.splitext(name)[-1].lower() in ['.jpg', '.jpeg', '.png', '.npz']:
                all_files.append(os.path.join(root, name))
    all_files.sort()
    all_files = tf.convert_to_tensor(all_files, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(all_files)
    return dataset


class ImageFeed(DataFeed):
    def __init__(self, preprocessing_callback, batch_size, folder_path):
        super().__init__(preprocessing_callback, batch_size)

        dataset = _dataset_from_folder(folder_path).map(_open_image_file)
        if self._preproc_callback:
            dataset = dataset.map(self._preproc_callback)
        self._dataset = dataset if batch_size is None else dataset.batch(self._batch_size)


class RegionProposalFeed(DataFeed):
    def __init__(self, preprocessing_callback, batch_size, folder_path):
        super().__init__(preprocessing_callback, batch_size)

        dataset = _dataset_from_folder(folder_path).map(_open_featuremap)
        if self._preproc_callback:
            dataset = dataset.map(self._preproc_callback)
        dataset = dataset.apply(tf.data.experimental.unbatch())
        self._dataset = dataset if batch_size is None else dataset.batch(self._batch_size)


class VideoFeed(DataFeed):
    def __init__(self, preprocessing_callback, batch_size, file_path):
        super().__init__(preprocessing_callback, batch_size=batch_size)

        dataset = tf.data.Dataset.from_generator(_video_generator(file_path), (tf.float32, tf.string))
        dataset = dataset.map(_parse_video_frame)
        if self._preproc_callback:
            dataset = dataset.map(self._preproc_callback)
        self._dataset = dataset if batch_size is None else dataset.batch(self._batch_size)
