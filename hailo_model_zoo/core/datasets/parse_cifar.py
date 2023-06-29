import tensorflow as tf


def parse_cifar100_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'label_index': tf.io.FixedLenFeature([], tf.int64),
            'image_numpy': tf.io.FixedLenFeature([], tf.string)
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label_index = tf.cast(features['label_index'], tf.int32)
    image = tf.io.decode_raw(features['image_numpy'], tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    image_info = {'label_index': label_index}
    return [image, image_info]
