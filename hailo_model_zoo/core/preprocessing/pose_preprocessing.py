import math

import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


def _openpose_padding(img, desired_dims, pad_value=0):
    h, w, _ = img.shape
    assert h <= desired_dims[0] and w <= desired_dims[1]

    pad = []
    pad.append(int(math.floor((desired_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((desired_dims[1] - w) / 2.0)))
    pad.append(int(desired_dims[0] - h - pad[0]))
    pad.append(int(desired_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value) \
        if pad != [0, 0, 0, 0] else img
    return padded_img, pad


def _openpose_preproc(img, desired_height, desired_width):
    height_in, width_in, _ = img.shape
    ratio = min(desired_height / float(height_in), desired_width / float(width_in))
    scaled_img = cv2.resize(img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    desired_dims = [desired_height, desired_width]
    padded_img, pad = _openpose_padding(scaled_img, desired_dims)
    return padded_img, pad


@PREPROCESS_FACTORY.register(name="openpose")
def openpose_tf_preproc(img, image_info, desired_height, desired_width, **kwargs):
    res_tens, pad = tf.numpy_function(_openpose_preproc,
                                      [img, desired_height, desired_width], (tf.float32, tf.int64))
    image_info["pad"] = pad
    image_info["orig_shape"] = tf.shape(img)
    return (tf.cast(res_tens, tf.float32), image_info)


def openpose_denormalize(img, img_mean=128.0, img_scale=1 / 256.0):
    img = np.array(img, dtype=np.float32)
    img = (img / img_scale) + img_mean
    return img


def letterbox(img, height=608, width=1088, centered=True,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (int(shape[1] * ratio), int(shape[0] * ratio))  # new_shape = [width, height]
    new_width = new_shape[0]
    dw = (width - new_width) / 2 if centered else (width - new_width)  # width padding
    new_height = new_shape[1]
    dh = (height - new_height) / 2 if centered else (height - new_height)  # height padding
    if centered:
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
    else:
        top, bottom = 0, dh
        left, right = 0, dw
    img = cv2.resize(img, new_shape,
                     interpolation=(cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR))  # resized, no border
    # cv2 uses bgr format, need to switch the color
    color_bgr = color[::-1]
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT,
                             value=color_bgr)  # padded rectangular
    return img, new_width, new_height


@PREPROCESS_FACTORY.register(name="yolov8_pose")
def yolo_pose(image, image_info=None, height=None, width=None,
              scope=None, padding_color=114, **kwargs):
    """
    This is the preprocessing used by ultralytics
    - Normalize the image from [0,255] to [0,1]
    """
    if height and width:
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        image, new_width, new_height = tf.numpy_function(
            lambda image, height, width: letterbox(image, height, width, color=[padding_color] * 3,
                                                   centered=kwargs["centered"]),
            [image, height, width], [tf.uint8, tf.int64, tf.int64])
        image.set_shape((height, width, 3))

    if image.dtype == tf.uint8:
        image = tf.cast(image, tf.float32)

    image_info['height'] = image_height
    image_info['width'] = image_width
    image_info['letterbox_height'] = new_height
    image_info['letterbox_width'] = new_width
    image_info['horizontal_pad'] = width - new_width
    image_info['vertical_pad'] = height - new_height

    return image, image_info
