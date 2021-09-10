import tensorflow as tf


def parse_mot_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'video_name': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'person_id': tf.io.VarLenFeature(tf.int64),
            'xmin': tf.io.VarLenFeature(tf.int64),
            'xmax': tf.io.VarLenFeature(tf.int64),
            'ymin': tf.io.VarLenFeature(tf.int64),
            'ymax': tf.io.VarLenFeature(tf.int64),
            'mark': tf.io.VarLenFeature(tf.int64),
            'label': tf.io.VarLenFeature(tf.int64),
            'visibility_ratio': tf.io.VarLenFeature(tf.float32),
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image_jpeg': tf.io.FixedLenFeature([], tf.string),
            'is_ignore': tf.io.VarLenFeature(tf.int64),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    video_name = tf.cast(features['video_name'], tf.string)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3, dct_method='INTEGER_ACCURATE')
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    image_info = {
        'image_name': image_name, 'video_name': video_name, 'height': height, 'width': width,
        'xmin': tf.sparse.to_dense(features['xmin'], default_value=0),
        'xmax': tf.sparse.to_dense(features['xmax'], default_value=0),
        'ymin': tf.sparse.to_dense(features['ymin'], default_value=0),
        'ymax': tf.sparse.to_dense(features['ymax'], default_value=0),
        'person_id': tf.sparse.to_dense(features['person_id'], default_value=0),
        'label': tf.sparse.to_dense(features['label'], default_value=0),
        'is_ignore': tf.sparse.to_dense(features['is_ignore'], default_value=0),
    }

    return [image, image_info]
