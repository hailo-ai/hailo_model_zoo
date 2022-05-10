import tensorflow as tf


def _resize(image, new_height, new_width, is_mask):
    image = tf.expand_dims(image, 0)
    if is_mask:
        resized_image = tf.image.resize(image, [new_height, new_width], method='nearest')
    else:
        resized_image = tf.image.resize(image, [new_height, new_width], method='bilinear')
    resized_image = tf.squeeze(resized_image)
    return resized_image


def _resnet_base_preprocessing(image, output_height=None, output_width=None, is_mask=False):
    image = _resize(image, output_height, output_width, is_mask)
    image = tf.cast(image, tf.float32)
    return image


def resnet_bw_18(image, image_info=None, output_height=None, output_width=None, **kwargs):
    image_orig = _resnet_base_preprocessing(image, output_height=output_height, output_width=output_width)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_gray = _resnet_base_preprocessing(image_gray, output_height=output_height, output_width=output_width)
    image_gray = tf.expand_dims(image_gray, axis=-1)
    if image_info and 'mask' in image_info.keys():
        image_info['mask'] = _resnet_base_preprocessing(image_info['mask'], output_height=output_height,
                                                        output_width=output_width, is_mask=True)
        image_info['img_orig'] = image_orig
    return image_gray, image_info


def resnet_v1_18(image, image_info=None, height=None, width=None, flip=False, **kwargs):
    image_orig = _resnet_base_preprocessing(image, height, width)

    if image_info and 'mask' in image_info.keys():
        image_info['mask'] = _resnet_base_preprocessing(image_info['mask'], output_height=height,
                                                        output_width=width, is_mask=True)
        image_info['img_orig'] = image_orig
    return image_orig, image_info
