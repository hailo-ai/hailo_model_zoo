import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'lr_img': tf.FixedLenFeature([], tf.string),
            'hr_img': tf.FixedLenFeature([], tf.string),
            'format': tf.FixedLenFeature([], tf.string)
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    lr_image = tf.decode_raw(features['lr_img'], tf.uint8)
    lr_image_shape = tf.stack([height, width, 3])
    lr_image = tf.reshape(lr_image, lr_image_shape)
    hr_image = tf.decode_raw(features['hr_img'], tf.uint8)
    hr_image_shape = tf.stack([4 * height, 4 * width, 3])
    hr_image = tf.reshape(hr_image, hr_image_shape)
    image_info = {'image_name': image_name, 'hr_img': hr_image, 'height': 4 * height, 'width': 4 * width}
    return [lr_image, image_info]
