from __future__ import division

from past.utils import old_div
import tensorflow as tf
import numpy as np
from PIL import Image

RESIZE_SIDE = 256
VIT_RESIZE_SIDE = 248
MOBILENET_CENTRAL_FRACTION = 0.875
RESMLP_CENTRAL_FRACTION = 0.875
MEAN_IMAGENET = [123.675, 116.28, 103.53]
STD_IMAGENET = [58.395, 57.12, 57.375]


class PatchifyException(Exception):
    """Patchify exception."""


def mobilenet_v1(image, image_info=None, height=None, width=None, **kwargs):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if MOBILENET_CENTRAL_FRACTION:
        image = tf.image.central_crop(image, central_fraction=MOBILENET_CENTRAL_FRACTION)

    if height and width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [height, width], method='bilinear')
        image = tf.squeeze(image, [0])

    # retrieve a 0-255 that was implicitely changed by tf.image.convert_image_dtype:
    image = image * 255
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)

    return image, image_info


def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    smallest_side = tf.cast(smallest_side, tf.float32)

    scale = tf.cond(tf.greater(height, width),
                    lambda: old_div(smallest_side, width),
                    lambda: old_div(smallest_side, height))
    new_height = tf.cast(height * scale, tf.int32)
    new_width = tf.cast(width * scale, tf.int32)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side, **kwargs):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    if ("method" in kwargs) and (kwargs['method'] is not None):
        resized_image = tf.image.resize(image, [new_height, new_width], method=kwargs["method"],
                                        antialias=True)
    else:
        resized_image = tf.image.resize(image, [new_height, new_width], method='bilinear')
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = old_div((image_height - crop_height), 2)
        offset_width = old_div((image_width - crop_width), 2)

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _resnet_base_preprocessing(image, output_height=None, output_width=None, resize_side=None, method=None):
    if output_height is not None:
        assert output_width is not None
        assert resize_side is not None
        image = _aspect_preserving_resize(image, resize_side, method=method)
        image = _central_crop([image], output_height, output_width)[0]
        image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, tf.float32)
    return image


def resnet_v1_18_34(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = _resnet_base_preprocessing(image, output_height, output_width, RESIZE_SIDE)
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def efficientnet(image, image_info=None, output_height=None, output_width=None, **kwargs):
    shape = tf.shape(image)
    padded_center_crop_size = tf.cast(output_height / (output_height + 32)
                                      * tf.cast(tf.minimum(shape[0], shape[1]), tf.float32), tf.int32)
    offset_height = ((shape[0] - padded_center_crop_size) + 1) // 2
    offset_width = ((shape[1] - padded_center_crop_size) + 1) // 2
    image_crop = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, padded_center_crop_size, padded_center_crop_size)
    image_resize = tf.image.resize([image_crop], [output_height, output_width], method='bicubic')[0]

    if image_info:
        image_info['img_orig'] = tf.cast(tf.image.resize(
            [image], [output_height, output_width], method='bicubic')[0], tf.uint8)
    return tf.cast(image_resize, tf.float32), image_info


def resmlp(image, image_info=None, output_height=None, output_width=None, **kwargs):  # Full model in chip
    '''
    This version of preprocessing runs the base ResMLP preprocess (Resize + CenterCrop).
    The patchify is done on-chip
    '''
    if output_height is not None:
        assert output_width is not None
        image = _aspect_preserving_resize(image, RESIZE_SIDE, method='bicubic')
        image = _central_crop([image], output_height, output_width)[0]
        image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, tf.float32)
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def pil_resize(image, output_height, output_width):
    image_uint = np.array(image, np.uint8)
    pil_image = Image.fromarray(image_uint)
    resized_image = pil_image.resize((output_width, output_height), Image.BICUBIC)
    image_numpy = np.array(resized_image, np.uint8)
    return image_numpy


def clip(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.numpy_function(pil_resize, [image, output_height, output_width], tf.uint8)
    image = tf.cast(image, tf.float32)
    image.set_shape([output_height, output_width, 3])
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def lprnet(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = tf.image.resize([image], [output_height, output_width], method='bicubic')[0]
    image = tf.squeeze(image)
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def vit_tiny(image, image_info=None, output_height=None, output_width=None, **kwargs):  # Full model in chip
    if output_height is not None:
        assert output_width is not None
        image = _aspect_preserving_resize(image, VIT_RESIZE_SIDE, method=kwargs.get('resize_method'))
        image = _central_crop([image], output_height, output_width)[0]
        image.set_shape([output_height, output_width, 3])
    image = tf.cast(image, tf.float32)
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info


def resnet_pruned(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image = _resnet_base_preprocessing(image, output_height, output_width, RESIZE_SIDE, method='bilinear')
    if image_info:
        image_info['img_orig'] = tf.cast(image, tf.uint8)
    return image, image_info
