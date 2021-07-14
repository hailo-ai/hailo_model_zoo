import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_id': tf.FixedLenFeature([], tf.int64),
            'landmarks': tf.FixedLenFeature([], tf.string),
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
            'format': tf.FixedLenFeature([], tf.string)
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_id = tf.cast(features['image_id'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    landmarks = tf.cast(features['landmarks'], tf.string)
    image = tf.cast(tf.image.decode_jpeg(features['image_jpeg'], channels=3), tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    image_info = {'image_name': image_name, 'image_id': image_id, 'landmarks': landmarks}
    return [image, image_info]
