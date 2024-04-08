import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.preprocessing.super_resolution_preprocessing import RGB2YUV_mat, RGB2YUV_offset

"""
properties of patch for image visualization:
"""
""" size fit for screen comparison """

THICKNESS = 3
NORMALIZATION_VALUE = 127.5

YUV2RGB_mat = [[1.16438355, 1.16438355, 1.16438355],
               [0., -0.3917616, 2.01723105],
               [1.59602715, -0.81296805, 0.]]


@POSTPROCESS_FACTORY.register(name="super_resolution")
@POSTPROCESS_FACTORY.register(name="super_resolution_srgan")
def super_resolution_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs['meta_arch'].lower()
    if 'sr_resnet' in meta_arch:
        endnodes = tf.clip_by_value(endnodes * 255, 0, 255)
    elif 'srgan' in meta_arch:
        endnodes = tf.cast(tf.clip_by_value(endnodes * NORMALIZATION_VALUE + NORMALIZATION_VALUE, 0, 255), tf.uint8)
    elif 'espcn' in meta_arch:
        endnodes = tf.cast(tf.clip_by_value(endnodes, 0, 1), tf.float32)
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


@VISUALIZATION_FACTORY.register(name="super_resolution_srgan")
def visualize_srgan_result(logits, img, **kwargs):
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


@VISUALIZATION_FACTORY.register(name="super_resolution")
def visualize_super_resolution_result(logits, img, **kwargs):
    """
    Visualizing the output of the Super-Res network compared with a naive upscaling.
    Args:
        logits: Numpy array. The sr network output (sr image batch)
        img: Numpy array, The sr network input (low-res image batch)
    Returns:
        combined_image: Numpy array. An image composed of a naive bicubic upsampling (right part),
        and a super-resolution upsampling (left part).
    """

    transpose = False
    if 'img_info' in kwargs and 'height' in kwargs['img_info']:
        transpose = kwargs['img_info']['height'] > kwargs['img_info']['width']

    if transpose:
        img = np.transpose(img, axes=[0, 2, 1, 3])
    logits = logits['predictions']
    img_yuv = np.matmul(img[0], RGB2YUV_mat) + RGB2YUV_offset

    input_resized = np.clip(cv2.resize(img[0],
                            logits.shape[1:3] if transpose else logits.shape[1:3][::-1],
                            interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

    img_yuv_resized = np.clip(cv2.resize(img_yuv,
                              logits.shape[1:3] if transpose else logits.shape[1:3][::-1],
                              interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    img_sr = logits[0] * 255  # Un-normalize
    img_sr = np.transpose(img_sr, axes=[1, 0, 2]) if transpose else img_sr
    if img_sr.shape[-1] == 1:
        img_sr = np.concatenate([img_sr, img_yuv_resized[..., 1:]], axis=2)

    white_patch = np.array(255 * np.ones((50, img_sr.shape[1], 3))).astype(np.uint8)
    img_sr = np.clip(np.matmul(img_sr - RGB2YUV_offset, YUV2RGB_mat), 0, 255).astype(np.uint8)  # YUV ==> RGB
    img_sr = np.concatenate([white_patch, img_sr], axis=0)
    img_sr = cv2.putText(img_sr, 'SR', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    input_resized = np.concatenate([white_patch, input_resized], axis=0)
    input_resized = cv2.putText(input_resized, 'Bicubic Resize', (5, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    combined_image = np.concatenate([img_sr, input_resized], axis=1)

    return combined_image
