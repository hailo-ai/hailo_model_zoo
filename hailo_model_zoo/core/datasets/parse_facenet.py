import tensorflow as tf


def parse_facenet_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string),
            'pair_index': tf.FixedLenFeature([], tf.int64),
            'is_same': tf.FixedLenFeature([], tf.int64),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['is_same'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.decode_raw(features['image'], tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    return [image, {'is_same': label, 'image_name': image_name}]
