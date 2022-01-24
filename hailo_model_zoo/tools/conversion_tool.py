#!/usr/bin/env python

import argparse
import logging
from functools import partial
import tensorflow as tf
from pathlib import Path

from hailo_sdk_client import ClientRunner, HailoNN
from hailo_model_zoo.core.main_utils import get_network_info
from hailo_model_zoo.utils.logger import get_logger
from hailo_model_zoo.core.preprocessing import preprocessing_factory
from hailo_model_zoo.utils.data import TFRecordFeed

_logger = get_logger()


class UnsupportedNetworkException(Exception):
    pass


class SourceFileNotFound(Exception):
    pass


def _get_data_feed(network_info, model_name, data_path, dataset_name, height, width):
    preprocessing_args = network_info.preprocessing
    hn_editor = network_info.hn_editor
    flip = hn_editor.flip
    yuv2rgb = hn_editor.yuv2rgb
    input_resize = hn_editor.input_resize
    preproc_callback = preprocessing_factory.get_preprocessing(
        model_name, height=height, width=width, flip=flip, yuv2rgb=yuv2rgb,
        input_resize=input_resize, normalization_params=False,
        **preprocessing_args)
    data_feed = TFRecordFeed(preproc_callback, batch_size=1, dataset_name=dataset_name, tfrecord_file=data_path)
    return data_feed.iterator


def _init_dataset(runner, tf_path, network_info):
    model_arch = network_info['preprocessing']['meta_arch']
    dataset_name = network_info['evaluation'].get('dataset_name', None)
    height, width = HailoNN.from_parsed_hn(runner.get_hn()).get_input_layers()[0].output_shape[1:3]
    data_feed_callback = partial(_get_data_feed, network_info, model_arch, tf_path,
                                 dataset_name, height, width)

    return data_feed_callback


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.description = '''The tool used to convert tf_record into buffer of images,
                            which can be used as an input file for the benchmark flow of the HailoRTcli tool'''
    parser.add_argument('tfrecord_file',
                        help='''The tfrecord file to be processed''',
                        type=str)
    parser.add_argument('hef_path',
                        help='''The hef file path, har/hn file must be adjacent''',
                        type=str)
    parser.add_argument('--output_file',
                        help='The name of the file to generate, default is {tfrecord_file}.bin',
                        type=str)
    parser.add_argument('--num-of-images',
                        help='The number of images to export from the input',
                        default=None,
                        type=int)
    return parser


def _tf_preprocess(data_feed_callback, process_callback, num_of_images=None):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    with tf.Graph().as_default():
        _logger.info('Building preprocess...')
        iterator = data_feed_callback()
        [preprocessed_data, image_info] = iterator.get_next()

        # Default of num_of_images to process is all images in the dataset:
        if not num_of_images:
            num_of_images = 1e9

        image_index = 0
        with tf.compat.v1.Session() as sess:
            sess.run([iterator.initializer, tf.compat.v1.local_variables_initializer()])

            try:
                while image_index < num_of_images:
                    preprocessed_image, _ = sess.run([preprocessed_data, image_info])
                    # Calling the process_callback to process the current pre-processed image:
                    process_callback(preprocessed_image)
                    image_index += 1
            except tf.errors.OutOfRangeError:
                # Finished iterating all the images in the dataset
                pass
        return image_index


def convert_tf_record_to_bin_file(hef_path: Path,
                                  tf_record_path: Path,
                                  output_file_name,
                                  num_of_images,
                                  har_path: Path = None,
                                  hn_path: Path = None):
    '''By default uses hailo archive file, otherwise uses hn'''

    # Extract the network name from the hef_path:
    network_name = hef_path.stem
    network_info = get_network_info(network_name)

    _logger.info('Initializing the runner...')
    runner = ClientRunner(hw_arch='hailo8p')
    # Hack to filter out client_runner info logs
    runner._logger.setLevel(logging.ERROR)

    _logger.info('Loading HEF file ...')
    try:
        # Load HAR into runner
        runner.load_har(har_path)
    except IOError:
        try:
            with open(hn_path, 'r') as hn:
                runner.set_hn(hn)
        except IOError:
            raise SourceFileNotFound(f'Neither {har_path} nor {hn_path} files were found.')

    _logger.info('Initializing the dataset ...')
    data_feed_callback = _init_dataset(runner, tf_record_path, network_info)

    bin_file = open(output_file_name, 'wb')

    callback_part = partial(_save_pre_processed_image_callback, file_to_append=bin_file)

    numn_of_processed_images = _tf_preprocess(data_feed_callback, callback_part, num_of_images=num_of_images)

    _logger.info('Conversion is done')
    _logger.info(f'File {output_file_name} created with {numn_of_processed_images} images')


def _save_pre_processed_image_callback(preprocessed_image, file_to_append):
    '''Callback function which used to get a pre-processed image from the dataset'''
    file_to_append.write(preprocessed_image.tobytes())


if __name__ == '__main__':
    parser = create_args_parser()
    args = parser.parse_args()

    output_path = Path(args.output_file) if args.output_file else Path(args.tfrecord_file).with_suffix('.bin')
    hn_path = Path(args.hef_path).with_suffix('.hn')
    har_path = Path(args.hef_path).with_suffix('.har')

    convert_tf_record_to_bin_file(Path(args.hef_path),
                                  Path(args.tfrecord_file),
                                  output_path,
                                  args.num_of_images,
                                  har_path,
                                  hn_path)
