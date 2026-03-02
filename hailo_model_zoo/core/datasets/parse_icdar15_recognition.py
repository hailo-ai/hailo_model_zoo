import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="icdar15_rec")
def parse_icdar15_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "text": tf.io.FixedLenFeature([], tf.string),
            "cropped_text": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "text_id": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    text_tag = tf.cast(features["text"], tf.string)

    image_name = tf.cast(features["image_name"], tf.string)
    image = tf.image.decode_jpeg(features["cropped_text"], channels=3)

    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)

    image_id = tf.cast(features["image_id"], tf.int32)
    text_id = tf.cast(features["text_id"], tf.int32)

    image_info = {
        "image_name": image_name,
        "width": width,
        "height": height,
        "image_id": image_id,
        "text_id": text_id,
        "text_tag": text_tag,
    }

    return [image, image_info]
