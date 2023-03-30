import numpy as np
import tensorflow as tf


def to_numpy(tensor, *, decode=False):
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, 'numpy'):
        result = tensor.numpy()
        if decode:
            result = result.decode('utf8')
        return result
    if isinstance(tensor, str):
        return tensor

    return tf.nest.map_structure(to_numpy, tensor)
