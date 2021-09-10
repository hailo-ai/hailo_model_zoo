#!/usr/bin/env python

import os
import argparse
import logging
from functools import partial
import tensorflow as tf

from hailo_platform.drivers.hailort.pyhailort import HEF, HailoRTTransformUtils
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


def _get_data_feed(model_name, data_path, dataset_name, height, width):
    preproc_callback = preprocessing_factory.get_preprocessing(model_name, height=height,
                                                               width=width, normalization_params=False)
    data_feed = TFRecordFeed(preproc_callback, batch_size=1, dataset_name=dataset_name, tfrecord_file=data_path)
    return data_feed.iterator


def _init_dataset(runner, tf_path, network_info):
    model_arch = network_info['preprocessing']['meta_arch']
    dataset_name = network_info['evaluation'].get('dataset_name', None)
    height, width = HailoNN.from_parsed_hn(runner.get_hn()).get_input_layers()[0].output_shape[1:3]
    data_feed_callback = partial(_get_data_feed, model_arch, tf_path,
                                 dataset_name, height, width)

    return data_feed_callback


def _change_file_extension(file_name, new_extension):
    file_name = os.path.splitext(file_name)[0]
    return '{}.{}'.format(file_name, new_extension)


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
    parser.add_argument('--transform',
                        action='store_true',
                        help='Should images be transformed',
                        default=False,
                        )
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
                    preprocessed_image, img_info = sess.run([preprocessed_data, image_info])
                    # Calling the process_callback to process the current pre-processed image:
                    process_callback(image_index, preprocessed_image, img_info)
                    image_index += 1
            except tf.errors.OutOfRangeError:
                # Finished iterating all the images in the dataset
                pass
        return image_index


def _get_default_output_file_name(tfrecord_file_path):
    file_name = os.path.basename(tfrecord_file_path)
    return _change_file_extension(file_name, 'bin')


def convert_tf_record_to_bin_file(hef_path, tf_record_path, output_file_name,
                                  num_of_images, transform, hn_path, har_path):
    '''By default uses hailo archive file, otherwise uses hn'''

    network_name = os.path.basename(hef_path).rstrip('.hef')
    network_info = get_network_info(network_name)

    _logger.info('Initializing the runner...')
    runner = ClientRunner(hw_arch='hailo8p')
    # Hack to filter out client_runner info logs
    runner._logger.setLevel(logging.ERROR)

    _logger.info('Loading HEF file ...')
    try:
        # Load HAR into runner
        with open(har_path, 'r') as hn:
            runner.load_har(har_path)
    except IOError:
        try:
            with open(hn_path, 'r') as hn:
                runner.set_hn(hn)
        except IOError:
            raise SourceFileNotFound(f'Neither {har_path} nor {hn_path} files were found.')

    hef = HEF(hef_path)
    _logger.info('HEF loaded')

    input_layers_info = hef.get_input_layers_info()
    if len(input_layers_info) > 1:
        raise UnsupportedNetworkException(f'''Networks with multiple input layers \
                                          ({len(input_layers_info)}) are not supported''')
    input_layer_info = input_layers_info[0]

    _logger.info('Initializing the dataset ...')
    data_feed_callback = _init_dataset(runner, tf_record_path, network_info)

    bin_file = open(output_file_name, 'wb')

    callback_part = partial(_save_pre_processed_image_callback, file_to_append=bin_file, should_transform=transform,
                            input_layer_info=input_layer_info)

    numn_of_processed_images = _tf_preprocess(data_feed_callback, callback_part, num_of_images=num_of_images)

    _logger.info('Conversion is done')
    _logger.info(f'File {output_file_name} created with {numn_of_processed_images} images')


def _save_pre_processed_image_callback(image_index, preprocessed_image, img_info, file_to_append,
                                       should_transform=False, input_layer_info=None):
    '''Callback function which used to get a pre-processed image from the dataset,
       transfom it if needed and save it in the binary file'''
    if should_transform:
        data = HailoRTTransformUtils.pre_infer(input_data=preprocessed_image,
                                               layer_info=input_layer_info,
                                               quantized=False)
    else:
        # No transform is needed:
        data = preprocessed_image.tobytes()

    file_to_append.write(data)


if __name__ == '__main__':
    parser = create_args_parser()
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = _get_default_output_file_name(args.tfrecord_file)

    hn_path = _change_file_extension(args.hef_path, 'hn')
    har_path = _change_file_extension(args.hef_path, 'har')
    convert_tf_record_to_bin_file(args.hef_path,
                                  args.tfrecord_file,
                                  args.output_file,
                                  args.num_of_images,
                                  args.transform,
                                  hn_path,
                                  har_path)
