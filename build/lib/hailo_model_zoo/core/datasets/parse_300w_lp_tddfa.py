import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        })
    face_image = tf.cast(tf.image.decode_jpeg(features['image'], channels=3), tf.uint8)

    image_info = {
        'image_name': features['image_name'],
    }
    return [face_image, image_info]
