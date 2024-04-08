import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


def scdepthv3_postprocessing(logits):
    depth = tf.math.reciprocal(tf.math.sigmoid(logits) * 10 + 0.009)
    return {'predictions': depth}


def fast_depth_postprocessing(logits):
    return {'predictions': logits}


def mono_depth_postprocessing(endnodes):
    min_depth = 0.1
    max_depth = 100
    depth_scale_factor = 5.4
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * tf.sigmoid(endnodes)
    depth = 1 / scaled_disp
    return {'predictions': depth * depth_scale_factor}


@VISUALIZATION_FACTORY.register(name="depth_estimation")
def visualize_depth_estimation_result(logits, image, **kwargs):
    logits = logits['predictions']
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(image.squeeze(0))
    plt.title("Input", fontsize=22)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.squeeze(logits), cmap='magma', vmax=np.percentile(logits, 95))
    plt.imshow(np.squeeze(logits))
    plt.title("Depth prediction", fontsize=22)
    plt.axis('off')
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='raw')
        buffer.seek(0)
        result = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    fig = plt.gcf()
    w, h = fig.canvas.get_width_height()
    result = result.reshape((int(h), int(w), -1))
    return result


archs_postprocess_dict = {"scdepthv3": scdepthv3_postprocessing,
                          "fast_depth": fast_depth_postprocessing,
                          "mono_depth": mono_depth_postprocessing}


def _get_postprocessing_function(meta_arch):
    if meta_arch in archs_postprocess_dict:
        return archs_postprocess_dict[meta_arch]
    raise ValueError("Meta-architecture [{}] is not supported".format(meta_arch))


@POSTPROCESS_FACTORY.register(name="depth_estimation")
def depth_estimation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs["meta_arch"].lower()
    postprocess = _get_postprocessing_function(meta_arch)
    return postprocess(endnodes)
