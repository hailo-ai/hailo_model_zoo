from __future__ import division
import tensorflow as tf


def stereonet(images, image_info=None, output_height=None, output_width=None, flip=None, **kwargs):
    image_l = images['image_l']
    image_r = images['image_r']
    height_l = image_info['height_l']
    width_l = image_info['width_l']
    height_r = image_info['height_r']
    width_r = image_info['width_r']
    crop_h, crop_w, _ = kwargs['input_shape']
    image_l = tf.image.crop_to_bounding_box(image_l, height_l - crop_h, width_l - crop_w, crop_h, crop_w)
    image_r = tf.image.crop_to_bounding_box(image_r, height_r - crop_h, width_r - crop_w, crop_h, crop_w)
    image = {'stereonet/input_layer1': image_l, 'stereonet/input_layer2': image_r}
    image_info['gt_l'] = tf.image.crop_to_bounding_box(
        image_info['gt_l'], height_l - crop_h, width_l - crop_w, crop_h, crop_w)
    return image, image_info
