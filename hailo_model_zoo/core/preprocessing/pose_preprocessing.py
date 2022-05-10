import tensorflow as tf
import cv2
import numpy as np
import math


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
