import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="bsd68")
@DATASET_FACTORY.register(name="cbsd68")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'index': tf.io.FixedLenFeature([], tf.int64),
            'img': tf.io.FixedLenFeature([], tf.string),
            'format': tf.io.FixedLenFeature([], tf.string)
        })

    index = tf.cast(features['index'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.io.decode_raw(features['img'], tf.uint8)
    image_shape = tf.stack([height, width, depth])
    image = tf.reshape(image, image_shape)
    image_info = {'image_name': image_name,
                  'index': index,
                  'height': height,
                  'width': width}
    return [image, image_info]
