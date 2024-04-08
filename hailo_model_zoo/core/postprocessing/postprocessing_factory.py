"""Contains a factory for network postprocessing."""
import importlib

import hailo_model_zoo.core.postprocessing
from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace

discovered_plugins = {
    name: importlib.import_module(name)
    for _, name, _
    in iter_namespace(hailo_model_zoo.core.postprocessing)
    if 'post' in name.split('.')[-1]  # ignore roi_align module which isn't importable
}


def get_visualization(name, **kwargs):
    """ Returns visualization_fn(endnodes, image_info)
        Args:
            name: The name of the task.
        Returns:
            visualization_fn: A function that visualize the results.

        Raises:
            ValueError: If visualization `name` is not recognized.
    """
    unsupported_visualizations = {
        'face_verification',
        'person_reid',
    }

    if name in unsupported_visualizations:
        raise ValueError(f'Visualization is currently not supported for {name}')

    visualization_fn = VISUALIZATION_FACTORY.get(name)
    return visualization_fn


def get_postprocessing(name, flip=False):
    """ Returns postprocessing_fn(endnodes, **kwargs)
        Args:
            name: The name of the task.
        Returns:
            postprocessing_fn: A function that postprocess a batch.

        Raises:
            ValueError: If postprocessing `name` is not recognized.
    """

    postprocess_callback = POSTPROCESS_FACTORY.get(name)

    def postprocessing_fn(endnodes, device_pre_post_layers=None, **kwargs):
        return postprocess_callback(endnodes, device_pre_post_layers, **kwargs)

    return postprocessing_fn
