"""Contains a factory for network infer."""
from hailo_model_zoo.core.infer.tf_infer import tf_infer
from hailo_model_zoo.core.infer.model_infer import model_infer
from hailo_model_zoo.core.infer.model_infer_lite import model_infer_lite
try:
    # THIS CODE IS EXPERIMENTAL AND IN USE ONLY FOR TAPPAS VALIDATION
    from hailo_model_zoo.core.infer.so_infer import so_infer
except ModuleNotFoundError:
    so_infer = None
from hailo_model_zoo.core.infer.tf_infer_second_stage import tf_infer_second_stage
from hailo_model_zoo.core.infer.runner_infer import runner_infer

NAME_TO_INFER = {
    'tf_infer': tf_infer,
    'runner_infer': runner_infer,
    'np_infer': lambda *args, **kwargs: model_infer(*args, **kwargs, np_infer=True),
    'facenet_infer': model_infer,
    'model_infer': model_infer,
    'model_infer_lite': model_infer_lite,
    'np_infer_lite': lambda *args, **kwargs: model_infer_lite(*args, **kwargs, np_infer=True),
    'so_infer': so_infer,
    'tf_infer_second_stage': tf_infer_second_stage,
}


def get_infer(infer_type):
    """ Returns infer_fn(endnodes, **kwargs)
        Args:
            name: The name of the task.
        Returns:
            infer_fn: A function that postprocesses a batch.

        Raises:
            ValueError: If infer `name` is not recognized.
    """

    if infer_type not in NAME_TO_INFER:
        raise ValueError('infer key [%s] was not recognized' % infer_type)

    return NAME_TO_INFER[infer_type]
