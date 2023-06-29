import io

import numpy as np
import matplotlib.pyplot as plt


def stereonet_postprocessing(logits, device_pre_post_layers=None, **kwargs):
    return {'predictions': logits}


def visualize_stereonet_result(logits, image, **kwargs):
    logits = np.array(logits['predictions'])[0]
    image = np.array(kwargs['img_info']['img_orig'])
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Input", fontsize=22)
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(logits, cmap='gray')
    plt.title("Disparity Map", fontsize=22)
    plt.axis('off')
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='raw')
        buffer.seek(0)
        result = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    fig = plt.gcf()
    w, h = fig.canvas.get_width_height()
    result = result.reshape((int(h), int(w), -1))
    return result
