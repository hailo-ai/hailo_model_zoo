import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="lp_ocr")
def parse_lp_ocr_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
            "plate": tf.io.FixedLenFeature([], tf.string),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image_name = tf.cast(features["image_name"], tf.string)
    plate = tf.cast(features["plate"], tf.string)
    image = tf.image.decode_jpeg(features["image_jpeg"], channels=3)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)

    image_info = {"image_name": image_name}
    image_info["height"] = height
    image_info["width"] = width
    image_info["plate"] = plate

    return [image, image_info]
