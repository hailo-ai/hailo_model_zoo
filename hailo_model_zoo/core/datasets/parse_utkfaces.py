import tensorflow as tf


def parse_age_gender_record(serialized_example):
    """
    Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_id': tf.FixedLenFeature([], tf.int64),
            'age': tf.FixedLenFeature([], tf.int64),
            'is_female_int': tf.FixedLenFeature([], tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_id = tf.cast(features['image_id'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3)
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    image_info = {
        'image_name': image_name, 'height': height, 'width': width, 'image_id': image_id,
        'age': tf.cast(features['age'], tf.int32),
        'is_female_int': tf.cast(features['is_female_int'], tf.int32)
    }

    return [image, image_info]
