import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="ss_dpm")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "mask": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image_name = tf.cast(features["image_name"], tf.string)
    image = tf.io.decode_raw(features["image"], tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    mask = tf.io.decode_raw(features["mask"], tf.uint8)
    mask_shape = tf.stack([height, width, 3])
    mask = tf.cast(tf.reshape(mask, mask_shape), tf.uint8)
    image_info = {"image_name": image_name, "mask": mask, "height": height, "width": width}
    return [image, image_info]
