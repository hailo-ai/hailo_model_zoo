#!/usr/bin/env python

import argparse
import json
import logging
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np

from hailo_sdk_client import ClientRunner

from hailo_model_zoo.core.main_utils import get_network_info, make_evalset_callback, make_preprocessing
from hailo_model_zoo.utils.logger import get_logger

_logger = get_logger()


class UnsupportedNetworkException(Exception):
    pass


class SourceFileNotFound(Exception):
    pass


def _init_dataset(runner, tf_path, network_info):
    preproc_callback = make_preprocessing(runner, network_info)
    dataset = make_evalset_callback(network_info, preproc_callback, tf_path)
    return dataset


def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.description = """Conversion tool for serialization of preprocessed input images as bin/npy file"""
    parser.add_argument("tfrecord_file", help="""The tfrecord file to be processed""", type=str)
    parser.add_argument("hef_path", help="""The hef file path, har/hn file must be adjacent""", type=str)
    parser.add_argument(
        "--output_file", help="The name of the file to generate, default is {tfrecord_file}.bin", type=str
    )
    parser.add_argument("--num-of-images", help="The number of images to export from the input", default=None, type=int)
    parser.add_argument("--npy", action="store_true", help="Output NPY format instead of bin")
    return parser


def _get_name(output_file_name, name):
    if name == "":
        return output_file_name

    original_suffix = output_file_name.suffix
    new_suffix = f'{name.replace("/","@")}{original_suffix}'
    path = output_file_name.parent / f"{output_file_name.with_suffix('').name}|{new_suffix}"
    return path


def _make_bin(dataset, output_file_name, num_of_images=None):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    _logger.info("Building preprocess...")
    if num_of_images:
        dataset = dataset.take(num_of_images)

    image_spec, _ = dataset.element_spec
    if not isinstance(image_spec, Mapping):
        image_spec = {"": image_spec}

    _image_index = -1
    files = {}
    with ExitStack() as stack:
        files = {name: stack.enter_context(_get_name(output_file_name, name).open("wb")) for name in image_spec}

        for _image_index, (preprocessed_data, _) in enumerate(dataset):
            if not isinstance(preprocessed_data, Mapping):
                preprocessed_data = {"": preprocessed_data}

            for name, preprocessed_image in preprocessed_data.items():
                files[name].write(preprocessed_image.numpy().tobytes())

    return _image_index + 1, [f.name for f in files.values()]


def _make_npy(dataset, output_file_name, num_of_images=None):
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    _logger.info("Building preprocess...")
    if num_of_images:
        dataset = dataset.take(num_of_images)

    image_list = []
    for preprocessed_data, _ in dataset:
        image_list.append(preprocessed_data)

    outpath = output_file_name.with_suffix(".npy")
    np.save(outpath, np.squeeze(image_list, axis=1))
    return len(image_list), [outpath]


def convert_tf_record_to_bin_file(
    hef_path: Path,
    tf_record_path: Path,
    output_file_name,
    num_of_images,
    har_path: Optional[Path] = None,
    hn_path: Optional[Path] = None,
    npy: bool = False,
):
    output_file_name = Path(output_file_name)
    """By default uses hailo archive file, otherwise uses hn"""

    # Extract the network name from the hef_path:
    network_name = hef_path.stem
    network_info = get_network_info(network_name)

    _logger.info("Initializing the runner...")
    runner = ClientRunner(hw_arch="hailo10h")
    # Hack to filter out client_runner info logs
    runner._logger.setLevel(logging.ERROR)  # noqa: SLF001 allow private member access

    _logger.info("Loading HEF file ...")
    try:
        # Load HAR into runner
        runner.load_har(har_path)
    except IOError:
        try:
            with open(hn_path, "r") as hn:
                runner.set_hn(hn)
        except IOError:
            raise SourceFileNotFound(f"Neither {har_path} nor {hn_path} files were found.") from None

    _logger.info("Initializing the dataset ...")
    dataset = _init_dataset(runner, tf_record_path, network_info)

    if npy:
        num_of_processed_images, files = _make_npy(dataset, output_file_name, num_of_images=num_of_images)
    else:
        num_of_processed_images, files = _make_bin(dataset, output_file_name, num_of_images=num_of_images)

    _logger.info("Conversion is done")
    _logger.info(f"{files} created with {num_of_processed_images} images")
    return files


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to a Python list
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()

    output_path = Path(args.output_file) if args.output_file else Path(args.tfrecord_file).with_suffix(".bin")
    hn_path = Path(args.hef_path).with_suffix(".hn")
    har_path = Path(args.hef_path).with_suffix(".har")

    convert_tf_record_to_bin_file(
        Path(args.hef_path), Path(args.tfrecord_file), output_path, args.num_of_images, har_path, hn_path, args.npy
    )
