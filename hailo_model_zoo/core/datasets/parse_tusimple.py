import tensorflow as tf


def parse(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'lanes': tf.io.VarLenFeature(tf.int64),
            'y_samples': tf.io.VarLenFeature(tf.float32),
            'num_of_lane_points': tf.io.VarLenFeature(tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image_jpeg': tf.io.FixedLenFeature([], tf.string),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    image_info = {'image_name': image_name}
    image_info['height'] = height
    image_info['width'] = width
    image_info['lanes'] = tf.cast(features['lanes'], tf.int32)
    image_info['num_of_lane_points'] = tf.cast(features['num_of_lane_points'], tf.int32)
    image_info['y_samples'] = tf.cast(features['y_samples'], tf.int32)
    return [image, image_info]
