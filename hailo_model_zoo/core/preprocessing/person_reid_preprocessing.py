import tensorflow as tf


def repvgga0(image, image_info=None, height=256, width=128, **kwargs):
    image = tf.cast(image, tf.float32)

    # resize to 256x128
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, [height, width], preserve_aspect_ratio=False, antialias=False, name=None)
    image = tf.squeeze(image, [0])

    # Normalize to [0,1]
    image = image / 255.0

    return image, image_info
