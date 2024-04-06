"""Contains a factory for network evaluation."""
import importlib

import hailo_model_zoo.core.eval
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace

discovered_plugins = {
    name: importlib.import_module(name)
    for _, name, _
    in iter_namespace(hailo_model_zoo.core.eval)
}


@EVAL_FACTORY.register(name="landmark_detection")
@EVAL_FACTORY.register(name="empty")
class EmptyEval():
    def __init__(self, **kwargs):
        pass

    def get_accuracy(self, **kwargs):
        pass


def get_evaluation(name):
    """Returns evaluation object
    Args:
        name: The name of the task
    Returns:
        evaluation: An object that evaluate the network.

    Raises:
        ValueError: If task `name` is not recognized.
    """

    return EVAL_FACTORY.get(name)
