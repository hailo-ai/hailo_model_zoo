from __future__ import division
import tensorflow as tf


def stereonet(images, image_info=None, output_height=None, output_width=None, flip=None, **kwargs):
    image_l = images['image_l']
    image_r = images['image_r']
    crop_h, crop_w, _ = kwargs['input_shape']
    image_l = pad_and_crop_tensor(image_l, crop_h, crop_w)
    image_l = tf.ensure_shape(image_l, [crop_h, crop_w, 3])
    image_r = pad_and_crop_tensor(image_r, crop_h, crop_w)
    image_r = tf.ensure_shape(image_r, [crop_h, crop_w, 3])
    image = {'stereonet/input_layer1': image_l, 'stereonet/input_layer2': image_r}
    image_info['gt_l'] = pad_and_crop_tensor(image_info['gt_l'], crop_h, crop_w)
    image_info['img_orig'] = tf.cast(image_l, tf.uint8)
    return image, image_info


def pad_and_crop_tensor(image, target_height=368, target_width=1232):
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    # Calculate the amount of padding required
    pad_height = tf.maximum(target_height - height, 0)
    pad_width = tf.maximum(target_width - width, 0)

    # Pad the tensor symmetrically on all sides
    paddings = tf.constant([[0, 0], [0, 0], [0, 0]], dtype=tf.int32)
    paddings = tf.tensor_scatter_nd_update(paddings, [[0, 1]], [pad_height])
    paddings = tf.tensor_scatter_nd_update(paddings, [[1, 1]], [pad_width])

    padded_image = tf.pad(image, paddings)

    # Crop or pad the tensor to the target shape
    cropped_image = tf.image.crop_to_bounding_box(padded_image, 0, 0, target_height, target_width)

    return cropped_image
