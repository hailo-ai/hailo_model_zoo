import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="amat_halfunet")
def amat_halfunet_dataset(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "image": tf.io.FixedLenFeature([], dtype=tf.string),
            "gt": tf.io.FixedLenFeature([], dtype=tf.string),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )

    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image = tf.reshape(tf.io.decode_raw(features["image"], tf.float32), (height, width, 3))
    gt = tf.reshape(tf.io.decode_raw(features["gt"], tf.float32), (height, width, 3))

    image_info = {"img_orig": image, "gt": gt}
    return (image, image_info)
