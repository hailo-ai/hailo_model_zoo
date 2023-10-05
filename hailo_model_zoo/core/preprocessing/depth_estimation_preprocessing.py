from __future__ import division
import tensorflow as tf


def mono_depth_2(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if output_height and output_width:
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [output_height, output_width], method='area')
        image = tf.squeeze(image, [0])
    image = image * 255
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def fastdepth_transform(image):
    image = tf.image.resize(image, (250, 332))
    image = tf.image.central_crop(image, 0.91)  # (228, 304))
    image = tf.image.resize(image, (224, 224))
    return image


def fast_depth(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # from unit8 also divides by 255...
    image = fastdepth_transform(image)
    image = image * 255
    if image_info:
        image_info['img_orig'] = image
        if 'depth' in image_info:
            depth = image_info['depth']
            depth = tf.expand_dims(depth, 2)
            depth = fastdepth_transform(depth)  # float32
            image_info['depth'] = depth
    return image, image_info


def scdepthv3(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (output_height, output_width))
    image = image * 255

    if image_info:
        image_info['img_orig'] = image

    return image, image_info
