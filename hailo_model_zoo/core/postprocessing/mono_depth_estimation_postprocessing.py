import io

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


MIN_DEPTH = 0.1
MAX_DEPTH = 100
DEPTH_SCALE_FACTOR = 5.4


def mono_depth_estimation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    min_disp = 1 / MAX_DEPTH
    max_disp = 1 / MIN_DEPTH
    scaled_disp = min_disp + (max_disp - min_disp) * tf.sigmoid(endnodes)
    depth = 1 / scaled_disp
    return {'predictions': depth * DEPTH_SCALE_FACTOR}


def visualize_mono_depth_result(logits, image, **kwargs):
    logits = logits['predictions']
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(image.squeeze(0))
    plt.title("Input", fontsize=22)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(np.squeeze(logits), cmap='magma', vmax=np.percentile(logits, 95))
    plt.title("Disparity prediction", fontsize=22)
    plt.axis('off')
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='raw')
        buffer.seek(0)
        result = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    fig = plt.gcf()
    w, h = fig.canvas.get_width_height()
    result = result.reshape((int(h), int(w), -1))
    return result
