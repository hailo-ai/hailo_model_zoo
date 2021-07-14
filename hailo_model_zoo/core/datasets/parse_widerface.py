import tensorflow as tf


def parse_detection_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_id': tf.FixedLenFeature([], tf.int64),
            'xmin': tf.VarLenFeature(tf.float32),
            'xmax': tf.VarLenFeature(tf.float32),
            'ymin': tf.VarLenFeature(tf.float32),
            'ymax': tf.VarLenFeature(tf.float32),
            'category_id': tf.VarLenFeature(tf.int64),
            'num_boxes': tf.FixedLenFeature([], tf.int64),
            'wider_hard_keep_index': tf.VarLenFeature(tf.int64),
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
        'num_boxes': tf.cast(features['num_boxes'], tf.int32),
        'xmin': tf.sparse.to_dense(features['xmin'], default_value=0),
        'xmax': tf.sparse.to_dense(features['xmax'], default_value=0),
        'ymin': tf.sparse.to_dense(features['ymin'], default_value=0),
        'ymax': tf.sparse.to_dense(features['ymax'], default_value=0),
        'wider_hard_keep_index': tf.sparse.to_dense(features['wider_hard_keep_index'], default_value=0),
        'category_id': tf.sparse.to_dense(features['category_id'], default_value=0)
    }

    return [image, image_info]
