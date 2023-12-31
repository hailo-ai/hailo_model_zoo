import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'upscale_factor': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'lr_img': tf.io.FixedLenFeature([], tf.string),
            'hr_img': tf.io.FixedLenFeature([], tf.string),
            'format': tf.io.FixedLenFeature([], tf.string)
        })

    upscale_factor = tf.cast(features['upscale_factor'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    lr_image = tf.io.decode_raw(features['lr_img'], tf.uint8)
    lr_image_shape = tf.stack([height, width, 3])
    lr_image = tf.reshape(lr_image, lr_image_shape)
    hr_image = tf.io.decode_raw(features['hr_img'], tf.uint8)
    hr_image_shape = tf.stack([upscale_factor * height, upscale_factor * width, 3])
    hr_image = tf.reshape(hr_image, hr_image_shape)
    image_info = {'image_name': image_name,
                  'hr_img': hr_image,
                  'height': height,
                  'width': width,
                  'upscale_factor': upscale_factor}
    return [lr_image, image_info]
