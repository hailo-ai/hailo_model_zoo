import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="face_landmarks")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "landmarks": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
            "format": tf.io.FixedLenFeature([], tf.string),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image_id = tf.cast(features["image_id"], tf.int32)
    image_name = tf.cast(features["image_name"], tf.string)
    landmarks = tf.cast(features["landmarks"], tf.string)
    image = tf.cast(tf.image.decode_jpeg(features["image_jpeg"], channels=3), tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    image_info = {"image_name": image_name, "image_id": image_id, "landmarks": landmarks}
    return [image, image_info]


@DATASET_FACTORY.register(name="hand_detection")
@DATASET_FACTORY.register(name="hand_landmark")
def parse_hand_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image_name = tf.cast(features["image_name"], tf.string)
    image = tf.cast(tf.image.decode_jpeg(features["image_jpeg"], channels=3), tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    image_info = {"image_name": image_name}
    return [image, image_info]
