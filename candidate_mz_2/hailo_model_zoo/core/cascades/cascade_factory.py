"""Contains a factory for network cascades"""

import importlib

import hailo_model_zoo.core.cascades
from hailo_model_zoo.core.factory import CASCADE_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace

discovered_plugins = {
    name: importlib.import_module(name) for _, name, _ in iter_namespace(hailo_model_zoo.core.cascades)
}


def get_cascade(name):
    return CASCADE_FACTORY.get(name)
