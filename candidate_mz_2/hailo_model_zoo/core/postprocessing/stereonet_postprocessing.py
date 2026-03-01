import io

import matplotlib.pyplot as plt
import numpy as np

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="stereonet")
def stereonet_postprocessing(logits, device_pre_post_layers=None, **kwargs):
    return {"predictions": logits}


@VISUALIZATION_FACTORY.register(name="stereonet")
def visualize_stereonet_result(logits, image, **kwargs):
    logits = np.array(logits["predictions"])[0]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.imshow(image[0])
    ax1.set_title("Input", fontsize=22)
    ax1.axis("off")
    ax2.imshow(logits, cmap="gray")
    ax2.set_title("Disparity Map", fontsize=22)
    ax2.axis("off")
    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="raw")
        buffer.seek(0)
        result = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    result = result.reshape((int(h), int(w), -1))
    plt.close()
    return result
