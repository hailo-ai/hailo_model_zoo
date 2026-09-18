"""Provides utilities to preprocess images for the Detection OBB networks."""

import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY
from hailo_model_zoo.core.preprocessing.detection_preprocessing import (
    _cast_image_info_types,
    _extract_box_coords_from_image_info,
    _pad_tensor,
    letterbox,
)

MAX_PADDING_LENGTH = 100


@PREPROCESS_FACTORY.register
def yolo_obb(image, image_info=None, height=None, width=None, scope=None, padding_color=114, **kwargs):
    if height and width:
        image_shape = tf.shape(image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        image, new_width, new_height = tf.numpy_function(
            lambda image, height, width: letterbox(
                image, height, width, color=[padding_color] * 3, centered=kwargs["centered"]
            ),
            [image, height, width],
            [tf.uint8, tf.int64, tf.int64],
        )
        image.set_shape((height, width, 3))

    if image.dtype == tf.uint8:
        image = tf.cast(image, tf.float32)

    image_info["img_orig"] = tf.cast(image, tf.uint8)
    if image_info and "num_boxes" in image_info.keys():
        maxpad = kwargs.get("max_pad", MAX_PADDING_LENGTH)
        image_info = _cast_image_info_types(image_info, image, max_padding_length=maxpad)
        x1, x2, x3, x4, y1, y2, y3, y4 = _extract_box_coords_from_image_info(
            image_info, max_padding_length=maxpad, is_normalized=True
        )
        image_info["bbox"] = tf.concat([x1, y1, x2, y2, x3, y3, x4, y4], axis=1)
        image_info["height"] = image_height
        image_info["width"] = image_width
        image_info["area"] = tf.expand_dims(_pad_tensor(image_info["area"], maxpad), axis=1)
        image_info["letterbox_height"] = new_height
        image_info["letterbox_width"] = new_width
        image_info["horizontal_pad"] = width - new_width
        image_info["vertical_pad"] = height - new_height

    return image, image_info
