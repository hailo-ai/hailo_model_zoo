import io

import numpy as np
import matplotlib.pyplot as plt


def fast_depth_postprocessing(logits, device_pre_post_layers=None, **kwargs):
    return {'predictions': logits}


def visualize_fast_depth_result(logits, image, **kwargs):
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
