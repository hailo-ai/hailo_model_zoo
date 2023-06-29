import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'll_img': tf.io.FixedLenFeature([], tf.string),
            'll_enhanced_img': tf.io.FixedLenFeature([], tf.string),
            'format': tf.io.FixedLenFeature([], tf.string)
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    ll_image = tf.io.decode_raw(features['ll_img'], tf.uint8)
    ll_image_shape = tf.stack([height, width, 3])
    ll_image = tf.reshape(ll_image, ll_image_shape)
    ll_enhanced_image = tf.io.decode_raw(features['ll_enhanced_img'], tf.uint8)
    ll_enhanced_image = tf.reshape(ll_enhanced_image, ll_image_shape)
    image_info = {'image_name': image_name,
                  'll_enhanced_img': ll_enhanced_image,
                  'height': height,
                  'width': width}
    return [ll_image, image_info]
