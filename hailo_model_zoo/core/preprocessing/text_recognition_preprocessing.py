import cv2
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY

"""
    Image preprocessing for Paddle OCR recognition model.
    Based on OCRReisizeNormImg class from PaddleOCR repository:
    https://github.com/PaddlePaddle/PaddleX/blob/d824981d50790548432daea51e5861b4fc56b82d/paddlex/inference/models/text_recognition/processors.py
"""


def resize(img, output_height=48, output_width=320):
    h, w, c = img.shape
    wh_ratio = w / h

    # resized image width in chosen from minimum of output_width or output_height * aspect ratio of original image
    resized_w = min(int(output_height * wh_ratio) + 1, output_width)
    resized_image = cv2.resize(img, (resized_w, output_height))

    # padding image to output_width with 128 value (only in case of resized_w < output_width)
    # padding will be on the right side of the image stripe
    padding_im = np.ones((output_height, output_width, c), dtype=np.uint8) * 128
    padding_im[:, 0:resized_w, :] = resized_image

    return padding_im


@PREPROCESS_FACTORY.register
def paddle_rec_preprocess(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image_resize = tf.numpy_function(resize, [image, output_height, output_width], tf.uint8)
    image_resize = tf.ensure_shape(image_resize, (output_height, output_width, kwargs["channels"]))

    if image_info:
        image_info["img_orig"] = image_resize

    return image_resize, image_info
