import cv2
import numpy as np

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY

"""
properties of patch for image visualization:
"""
""" size fit for screen comparison """


@POSTPROCESS_FACTORY.register(name="low_light_enhancement")
def low_light_enhancement_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    return {'predictions': endnodes}


@VISUALIZATION_FACTORY.register(name="low_light_enhancement")
def visualize_low_light_enhancement_result(logits, img, **kwargs):
    """
    Visualizing the output of the Low-Light Enhancement network.
    Args:
        logits: Numpy array. The low-light-enhancement network output batch
        img: Numpy array. The low-light-enhancement network input batch
    Returns:
        combined_image: Numpy array. An image composed of a low light image (right part),
        and the enhanced image (left part).
    """
    transpose = False
    if 'img_info' in kwargs and 'height' in kwargs['img_info']:
        transpose = kwargs['img_info']['height'] > kwargs['img_info']['width']

    if transpose:
        img = np.transpose(img, axes=[0, 2, 1, 3])
    logits = logits['predictions'][0]

    enhanced_img = (logits * 255).astype(np.uint8)  # Un-normalize
    enhanced_img = cv2.putText(enhanced_img, 'Enhanced image', (5, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (158, 253, 56), 2)
    orig_img = cv2.putText(img[0].astype(np.uint8), 'LowLight image', (5, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (158, 253, 56), 2)
    combined_image = np.concatenate([orig_img, enhanced_img], axis=1)

    return combined_image
