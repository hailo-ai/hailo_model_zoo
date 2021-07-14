import tensorflow as tf


DEPTH_WIDTH = 1242
DEPTH_HEIGHT = 375


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.string),
            'image_jpeg': tf.FixedLenFeature([], tf.string)
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.cast(tf.image.decode_jpeg(features['image_jpeg'], channels=3), tf.uint8)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    img_depth = tf.io.decode_raw(features['depth'], tf.float32)
    depth_shape = tf.stack([DEPTH_HEIGHT, DEPTH_WIDTH])
    img_depth = tf.reshape(img_depth, depth_shape)
    image_info = {'image_name': image_name, 'depth': img_depth}
    return [image, image_info]
