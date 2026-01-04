"""Contains a factory for image preprocessing."""

import importlib

import tensorflow as tf

import hailo_model_zoo.core.preprocessing
from hailo_model_zoo.core.factory import PREPROCESS_FACTORY
from hailo_model_zoo.utils.plugin_utils import iter_namespace

discovered_plugins = {
    name: importlib.import_module(name) for _, name, _ in iter_namespace(hailo_model_zoo.core.preprocessing)
}


def convert_rgb_to_yuv(image):
    transition_matrix = tf.constant(
        [
            [0.2568619, -0.14823364, 0.43923104],
            [0.5042455, -0.2909974, -0.367758],
            [0.09799913, 0.43923104, -0.07147305],
        ],
        dtype=tf.float32,
    )
    image = tf.cast(image, tf.float32)
    image = tf.matmul(image, transition_matrix)
    image += tf.constant([16, 128, 128], dtype=tf.float32)
    return image


def convert_yuv_to_yuy2(image):
    y_img = image[..., 0]
    uv_img = image[..., 1:]
    uv_subsampled = uv_img[:, ::2, :]
    uv_unrolled = tf.reshape(uv_subsampled, (image.shape[-3], image.shape[-2]))
    yuy2_img = tf.stack([y_img, uv_unrolled], axis=-1)
    return yuy2_img


def convert_yuv_to_nv12(image):
    h, w, _ = image.shape
    y_img = image[..., 0]
    y = tf.reshape(y_img, (2 * h, w // 2))
    u = image[..., 1]
    v = image[..., 2]
    u_subsample = tf.expand_dims(u[::2, ::2], axis=-1)
    v_subsample = tf.expand_dims(v[::2, ::2], axis=-1)
    uv = tf.stack([u_subsample, v_subsample], axis=-1)
    uv = tf.reshape(uv, (h, w // 2))
    return tf.reshape(tf.concat([y, uv], axis=0), (h // 2, w, 3))


def convert_rgb_to_rgbx(image):
    h, w, _ = image.shape
    new_channel = tf.zeros((h, w, 1))
    return tf.concat([image, new_channel], axis=-1)


def image_resize(image, shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, tuple(shape), method="bilinear")
    return tf.squeeze(image, [0])


def normalize(image, normalization_params):
    mean, std = normalization_params
    return (image - list(mean)) / list(std)


def get_preprocessing(name, height, width, normalization_params, **kwargs):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
        name: meta architecture name
        height: height of the input image
        width: width of the input image
        normalization_params: parameters for normalizing the input image
        kwargs: additional preprocessing arguments

    Returns:
        preprocessing_fn: A function that preprocessing a single image (pre-batch).
            It has the following signature:
                image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
        ValueError: If Preprocessing `name` is not recognized.
    """

    preprocessing_callback = PREPROCESS_FACTORY.get(name)
    flip = kwargs.pop("flip", False)
    yuv2rgb = kwargs.pop("yuv2rgb", False)
    yuy2 = kwargs.pop("yuy2", False)
    nv12 = kwargs.pop("nv12", False)
    rgbx = kwargs.pop("rgbx", False)
    input_resize = kwargs.pop("input_resize", {})

    def preprocessing_fn(image, image_info=None):
        image, image_info = preprocessing_callback(image, image_info, height, width, flip=flip, **kwargs)
        if normalization_params and normalization_params[0] and normalization_params[1]:
            image = normalize(image, normalization_params)
        if input_resize.get("enabled", False):
            image = image_resize(image, input_resize.input_shape)
        if yuv2rgb:
            image = convert_rgb_to_yuv(image)
        if yuy2:
            image = convert_yuv_to_yuy2(image)
        if nv12:
            image = convert_yuv_to_nv12(image)
        if rgbx:
            image = convert_rgb_to_rgbx(image)
        return image, image_info

    return preprocessing_fn
