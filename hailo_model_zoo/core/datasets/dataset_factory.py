"""Contains a factory for network infer."""
import importlib

import hailo_model_zoo.core.datasets
from hailo_model_zoo.core.factory import DATASET_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace

discovered_plugins = {
    name: importlib.import_module(name)
    for _, name, _ in iter_namespace(hailo_model_zoo.core.datasets)
}


def get_dataset_parse_func(ds_name):
    """Get the func to parse dictionary from a .tfrecord.
    Each parse function returns <image, image_info>
        image: image tensor
        image_info: dictionary that contains other information of the image (e.g., the label)
    """
    return DATASET_FACTORY.get(ds_name)
