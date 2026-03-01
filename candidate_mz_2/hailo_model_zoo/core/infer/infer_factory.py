"""Contains a factory for network infer."""

import importlib

import hailo_model_zoo.core.infer
from hailo_model_zoo.core.factory import INFER_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace


def get_infer(infer_type):
    return INFER_FACTORY.get(infer_type)


discovered_plugins = {name: importlib.import_module(name) for _, name, _ in iter_namespace(hailo_model_zoo.core.infer)}
