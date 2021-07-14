import tensorflow as tf
import numpy as np
import cv2

"""
properties of patch for image visualization:
"""
""" size fit for screen comparison """

THICKNESS = 3
NORMALIZATION_VALUE = 127.5


def super_resolution_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs['meta_arch'].lower()
    if 'sr_resnet' in meta_arch:
        endnodes = tf.clip_by_value(endnodes * 255, 0, 255)
    elif 'srgan' in meta_arch:
        endnodes = tf.cast(tf.clip_by_value(endnodes * NORMALIZATION_VALUE + NORMALIZATION_VALUE, 0, 255), tf.uint8)
    else:
        raise Exception("Super resolution postprocessing {} is not supported".format(meta_arch))
    return {'predictions': endnodes}


def create_mosaic_real_ratio(combined_images, input_resized):
    h_diff = combined_images.shape[0] - input_resized.shape[0]
    w_diff = combined_images.shape[0] - input_resized.shape[1]
    h_start = int(h_diff / 2)
    h_stop = h_start + input_resized.shape[0]
    w_start = int(w_diff / 2)
    w_stop = w_start + input_resized.shape[1]
    grey_matrix = 127 * np.ones((combined_images.shape[0], combined_images.shape[0], 3))
    grey_matrix[h_start:h_stop, w_start:w_stop, :] = input_resized
    mosaic = np.concatenate([combined_images, grey_matrix], 1)
    mosaic = mosaic.astype(np.uint8)
    return mosaic


def draw_patch(image, h_center, w_center, width):
    h_min = max([h_center - int(width / 2), 0])
    h_max = min([h_center + int(width / 2), image.shape[0]])
    w_min = max([w_center - int(width / 2), 0])
    w_max = min([w_center + int(width / 2), image.shape[1]])
    for h in range(h_min, h_min + THICKNESS):
        image[h, w_min:w_max] = [0, 255, 0]
    for h in range(h_max - THICKNESS, h_max):
        image[h, w_min:w_max] = [0, 255, 0]
    for w in range(w_min, w_min + THICKNESS):
        image[h_min:h_max, w] = [255, 0, 0]
    for w in range(w_max - THICKNESS, w_max):
        image[h_min:h_max, w] = [255, 0, 0]
    return image


def focus_on_patch(image, h_center, w_center, width):
    h_min = max([h_center - int(width / 2), 0])
    h_max = min([h_center + int(width / 2), image.shape[0]])
    w_min = max([w_center - int(width / 2), 0])
    w_max = min([w_center + int(width / 2), image.shape[1]])
    return image[h_min:h_max, w_min:w_max, :]


def visualize_super_resolution_result(logits, img, **kwargs):
    """
    Visualizing the output of the Super-Res network compared with a naive upscaling.
    Args:
        logits: Numpy array. The sr network output (sr image batch)
        img: Numpy array, The sr network input (low-res image batch)
    Returns:
        mosaic_image: Numpy array. An image mosaic of the network input, and two versions
        of an upsampled patch - a naive bicubic upsampling (upper-left), and a super-resolution
        upsampling (lower-left).
    """
    """ size fit for screen comparison """
    h_center = 380
    w_center = 490
    width = 400
    logits = logits['predictions']
    input_resized = np.clip(cv2.resize(img[0],
                            (logits.shape[2], logits.shape[1]),
                            interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

    input_resized_patch = focus_on_patch(input_resized, h_center, w_center, width)
    img_sr = logits[0]
    img_sr_patch = focus_on_patch(img_sr, h_center, w_center, width)
    combined_images = np.concatenate([input_resized_patch, img_sr_patch], 0)
    small_input = img[0]
    small_input_with_patch_drawn = \
        draw_patch(small_input, int(h_center / 4), int(w_center / 4), int(width / 4))
    mosaic_image = create_mosaic_real_ratio(combined_images, small_input_with_patch_drawn)
    return mosaic_image
