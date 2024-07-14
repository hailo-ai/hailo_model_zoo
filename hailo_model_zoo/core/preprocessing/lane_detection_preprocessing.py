import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


def _cast_image_info_types(image_info, image):
    image_info["img_orig"] = tf.cast(image, tf.uint8)
    image_info["lanes"] = tf.cast(image_info["lanes"], tf.int32)
    return image_info


@PREPROCESS_FACTORY.register
def polylanenet(image, image_info=None, height=None, width=None, **kwargs):
    image = tf.cast(image, tf.float32)
    image_info = _cast_image_info_types(image_info, image)
    if height and width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [height, width], method="bilinear")
        # this is the working line. gave 91.44% when we expect 91.3%
        image = tf.squeeze(image, [0])
        image = image[..., ::-1]  # swapping rgb/bgr
        image_info["image_resized"] = image

    return image, image_info


@PREPROCESS_FACTORY.register
def laneaf(image, image_info=None, height=None, width=None, **kwargs):
    image = tf.cast(image, tf.float32)
    image_info = _cast_image_info_types(image_info, image)
    if height and width:
        image = tf.expand_dims(image, 0)
        image = image[:, 16:, :, :]  # Crop 16 pixels from the top
        image = tf.image.resize(image, [height, width], method="bilinear")
        image = tf.squeeze(image, [0])
        image_info["image_resized"] = image

    return image, image_info
