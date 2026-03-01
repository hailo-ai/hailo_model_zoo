import numpy as np
import tensorflow as tf


def to_numpy(tensor, *, decode=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "numpy"):
        result = tensor.numpy()
        if decode:
            result = result.decode("utf8")
        return result
    if isinstance(tensor, str):
        return tensor
    if isinstance(tensor, bytes):
        if decode:
            tensor = tensor.decode("utf8")
        return tensor
    if isinstance(tensor, dict):
        return {k: v.numpy() if hasattr(v, "numpy") else v for k, v in tensor.items()}

    return tf.nest.map_structure(to_numpy, tensor)
