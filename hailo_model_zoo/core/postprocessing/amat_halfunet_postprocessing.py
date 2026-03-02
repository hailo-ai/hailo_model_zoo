import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="amat_halfunet")
def amat_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    return {"predictions": tf.concat(endnodes, axis=-1)}


@VISUALIZATION_FACTORY.register(name="amat_halfunet")
def visualize_amat_halfunet(logits, img_gt, **kwargs):
    logits = logits["predictions"].squeeze(axis=0)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 5))
    channel_titles = ["Channel 1", "Channel 2", "Channel 3"]

    for i in range(3):
        ax = axes[i]
        ax.imshow(logits[..., i], cmap="viridis")  # Use a colormap like 'viridis'
        ax.set_title(channel_titles[i])
        ax.axis("off")  # Turn off axis ticks and labels

    # Save the figure to a buffer
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="raw")
        buffer.seek(0)
        result = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    fig = plt.gcf()
    w, h = fig.canvas.get_width_height()
    result = result.reshape((int(h), int(w), -1))
    plt.close(fig)
    return result
