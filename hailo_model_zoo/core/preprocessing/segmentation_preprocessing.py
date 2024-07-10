import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY
from hailo_model_zoo.core.preprocessing.detection_preprocessing import MAX_PADDING_LENGTH


def _resize(image, new_height, new_width, is_mask):
    image = tf.expand_dims(image, 0)
    if is_mask:
        resized_image = tf.image.resize(image, [new_height, new_width], method="nearest")
    else:
        resized_image = tf.image.resize(image, [new_height, new_width], method="bilinear")
    resized_image = tf.squeeze(resized_image)
    return resized_image


def _resnet_base_preprocessing(image, height=None, width=None, is_mask=False):
    image = _resize(image, height, width, is_mask)
    image = tf.cast(image, tf.float32)
    return image


@PREPROCESS_FACTORY.register(name="fcn_resnet_bw")
def resnet_bw_18(image, image_info=None, input_height=None, input_width=None, **kwargs):
    image_orig = _resnet_base_preprocessing(image, height=input_height, width=input_width)
    image_gray = tf.image.rgb_to_grayscale(image)
    image_gray = _resnet_base_preprocessing(image_gray, height=input_height, width=input_width)
    image_gray = tf.expand_dims(image_gray, axis=-1)
    if image_info and "mask" in image_info.keys():
        image_info["mask"] = _resnet_base_preprocessing(
            image_info["mask"], height=input_height, width=input_width, is_mask=True
        )
        image_info["img_orig"] = image_orig
    return image_gray, image_info


@PREPROCESS_FACTORY.register(name="fcn_resnet")
def resnet_v1_18(image, image_info=None, height=None, width=None, **kwargs):
    image_orig = _resnet_base_preprocessing(image, height, width)
    if image_info and "mask" in image_info.keys():
        image_info["mask"] = _resnet_base_preprocessing(image_info["mask"], height=height, width=width, is_mask=True)
        image_info["img_orig"] = image_orig
    return image_orig, image_info


def _pad_tensor(x, max_tensor_padding=MAX_PADDING_LENGTH):
    paddings = [(0, 0), (0, max_tensor_padding - tf.shape(x)[0])]
    return tf.squeeze(tf.pad(tf.expand_dims(x, axis=0), paddings, "CONSTANT", constant_values=-1))


def _get_resized_shape(size, height, width):
    size = tf.cast(size, tf.float32)
    h, w = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
    scale = size * 1.0 / tf.maximum(h, w)
    newh = h * scale
    neww = w * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return newh, neww


@PREPROCESS_FACTORY.register
def sparseinst(image, image_info=None, height=None, width=None, max_pad=MAX_PADDING_LENGTH, **kwargs):
    image_resized = image
    if height and width:
        assert height == width, f"sparseinst expects a square input but got {height}x{width}"
        orig_height, orig_width = tf.shape(image)[0], tf.shape(image)[1]
        newh, neww = _get_resized_shape(height, orig_height, orig_width)
        image_resized_ar = tf.squeeze(tf.image.resize(image, size=(newh, neww), method="bilinear"))
        paddings = [
            [0, tf.maximum(height - tf.shape(image_resized_ar)[0], 0)],
            [0, tf.maximum(width - tf.shape(image_resized_ar)[1], 0)],
            [0, 0],
        ]
        image_padded = tf.squeeze(tf.pad(image_resized_ar, paddings, mode="CONSTANT", constant_values=0))
        image_resized = tf.cast(image_padded, tf.float32)
        image_resized.set_shape((height, width, 3))
    if image_info:
        image_info["resized_height"] = tf.cast(newh, tf.int32)
        image_info["resized_width"] = tf.cast(neww, tf.int32)
        image_info["height"] = tf.cast(tf.shape(image)[0], tf.int32)
        image_info["width"] = tf.cast(tf.shape(image)[1], tf.int32)
        image_info["img_orig"] = tf.cast(image_resized, tf.uint8)
        keys2pad = ["xmin", "xmax", "ymin", "ymax", "area", "category_id", "is_crowd"]
        for key in keys2pad:
            if key in image_info:
                image_info[key] = _pad_tensor(image_info[key], max_pad)
    return image_resized, image_info
