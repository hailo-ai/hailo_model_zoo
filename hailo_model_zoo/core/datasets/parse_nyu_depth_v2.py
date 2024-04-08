import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480


@DATASET_FACTORY.register(name="nyu_depth_v2")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'depth': tf.io.FixedLenFeature([], tf.string),
            'rgb': tf.io.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)

    image = tf.io.decode_raw(features['rgb'], tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    img_depth = tf.io.decode_raw(features['depth'], tf.float32)
    depth_shape = tf.stack([height, width])
    img_depth = tf.reshape(img_depth, depth_shape)
    image_info = {'image_name': image_name, 'depth': img_depth}
    return [image, image_info]
