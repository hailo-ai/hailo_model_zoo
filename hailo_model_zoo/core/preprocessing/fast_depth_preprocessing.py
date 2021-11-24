from __future__ import division
import tensorflow as tf


def val_transform(image):
    image = tf.image.resize(image, (250, 332))
    image = tf.image.central_crop(image, 0.91)  # (228, 304))
    image = tf.image.resize(image, (224, 224))
    return image


def fast_depth(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # from unit8 also divides by 255...
    image = val_transform(image)
    image = image * 255
    if image_info:
        depth = image_info['depth']
        depth = tf.expand_dims(depth, 2)
        depth = val_transform(depth)  # float32
        image_info['depth'] = depth
        image_info['img_orig'] = image
    return image, image_info
