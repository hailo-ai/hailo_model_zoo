import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="kitti_stereo")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height_l': tf.io.FixedLenFeature([], tf.int64),
            'width_l': tf.io.FixedLenFeature([], tf.int64),
            'height_r': tf.io.FixedLenFeature([], tf.int64),
            'width_r': tf.io.FixedLenFeature([], tf.int64),
            'image_l_name': tf.io.FixedLenFeature([], tf.string),
            'image_r_name': tf.io.FixedLenFeature([], tf.string),
            'image_l_png': tf.io.FixedLenFeature([], tf.string),
            'image_r_png': tf.io.FixedLenFeature([], tf.string),
            'label_l_name': tf.io.FixedLenFeature([], tf.string),
            'label_r_name': tf.io.FixedLenFeature([], tf.string),
            'label_l': tf.io.FixedLenFeature([], tf.string)
        })

    height_l = tf.cast(features['height_l'], tf.int32)
    width_l = tf.cast(features['width_l'], tf.int32)
    height_r = tf.cast(features['height_r'], tf.int32)
    width_r = tf.cast(features['width_r'], tf.int32)
    image_l_name = tf.cast(features['image_l_name'], tf.string)
    image_r_name = tf.cast(features['image_r_name'], tf.string)
    image_l = tf.cast(tf.image.decode_png(features['image_l_png'], channels=3), tf.float32)
    image_r = tf.cast(tf.image.decode_png(features['image_r_png'], channels=3), tf.float32)
    label_l = tf.io.decode_raw(features['label_l'], tf.float32) / 255.0
    label_l = tf.reshape(label_l, (height_l, width_l, 1))
    images = {'image_l': image_l, 'image_r': image_r}
    image_info = {'image_l_name': image_l_name, 'image_r_name': image_r_name,
                  'gt_l': label_l,
                  'height_l': height_l, 'width_l': width_l,
                  'height_r': height_r, 'width_r': width_r}
    return [images, image_info]
