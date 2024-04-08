import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="market1501")
def parse_market_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image_jpeg': tf.io.FixedLenFeature([], tf.string),
            'type': tf.io.FixedLenFeature([], tf.string),
            'cam_id': tf.io.FixedLenFeature([], tf.int64),
            'label_index': tf.io.FixedLenFeature([], tf.int64),
        })
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    label = tf.cast(features['label_index'], tf.int64)
    image_name = tf.cast(features['image_name'], tf.string)
    type = tf.cast(features['type'], tf.string)
    cam_id = tf.cast(features['cam_id'], tf.int64)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3, dct_method='INTEGER_ACCURATE')
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)

    return [image, {'label': label, 'image_name': image_name, 'type': type, 'cam_id': cam_id}]
