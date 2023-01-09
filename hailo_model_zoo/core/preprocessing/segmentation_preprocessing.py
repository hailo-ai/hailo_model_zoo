import tensorflow as tf

from hailo_model_zoo.core.preprocessing.detection_preprocessing import MAX_PADDING_LENGTH


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


def _pad_tensor(x, max_tensor_padding=MAX_PADDING_LENGTH):
    paddings = [(0, 0), (0, max_tensor_padding - tf.shape(x)[0])]
    return tf.squeeze(tf.pad(tf.expand_dims(x, axis=0), paddings, "CONSTANT", constant_values=-1))


def sparseinst(image, image_info=None, height=None, width=None, max_pad=MAX_PADDING_LENGTH, **kwargs):
    image_resized = image
    if height and width:
        paddings = [[0, height - tf.shape(image)[0]],
                    [0, width - tf.shape(image)[1]],
                    [0, 0]]
        image_padded = tf.squeeze(tf.pad(image, paddings, mode="CONSTANT", constant_values=0))
        image_resized = tf.cast(image_padded, tf.float32)
    if image_info:
        image_info['height'] = tf.cast(height, tf.int32)
        image_info['width'] = tf.cast(width, tf.int32)
        image_info['orig_height'] = tf.cast(tf.shape(image)[0], tf.int32)
        image_info['orig_width'] = tf.cast(tf.shape(image)[1], tf.int32)
        image_info['img_orig'] = tf.cast(image_resized, tf.uint8)
        keys2pad = ['xmin', 'xmax', 'ymin', 'ymax', 'area', 'category_id', 'is_crowd']
        for key in keys2pad:
            if key in image_info:
                image_info[key] = _pad_tensor(image_info[key], max_pad)
    return image_resized, image_info
