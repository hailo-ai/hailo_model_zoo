from __future__ import division

import tensorflow as tf


def mono_depth_2(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if output_height and output_width:
        image = tf.expand_dims(image, 0)
        image = tf.compat.v1.image.resize(image, [output_height, output_width], method=tf.image.ResizeMethod.AREA)
        image = tf.squeeze(image, [0])
    image = image * 255
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info
