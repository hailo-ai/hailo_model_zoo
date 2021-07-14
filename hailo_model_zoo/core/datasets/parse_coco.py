import tensorflow as tf


def parse_pose_estimation_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_id': tf.FixedLenFeature([], tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_id = tf.cast(features['image_id'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3)
    image = image[..., ::-1]
    image_shape = tf.stack([height, width, 3])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    image_info = {'image_id': image_id, 'image_name': image_name}

    return [image, image_info]


def parse_segmentation_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'xmin': tf.VarLenFeature(tf.float32),
            'xmax': tf.VarLenFeature(tf.float32),
            'ymin': tf.VarLenFeature(tf.float32),
            'ymax': tf.VarLenFeature(tf.float32),
            'category_id': tf.VarLenFeature(tf.int64),
            'image_name': tf.FixedLenFeature([], tf.string),
            'mask': tf.FixedLenFeature([], tf.string),
            'image_jpeg': tf.FixedLenFeature([], tf.string),
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_name = tf.cast(features['image_name'], tf.string)
    image = tf.image.decode_jpeg(features['image_jpeg'], channels=3)
    mask = tf.decode_raw(features['mask'], tf.uint8)
    image_shape = tf.stack([height, width, 3])
    mask_shape = tf.stack([height, width, 1])
    image = tf.cast(tf.reshape(image, image_shape), tf.uint8)
    mask = tf.cast(tf.reshape(mask, mask_shape), tf.uint8)
    image_info = {'image_name': image_name, 'mask': mask}

    return [image, image_info]


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
            'area': tf.VarLenFeature(tf.float32),
            'category_id': tf.VarLenFeature(tf.int64),
            'is_crowd': tf.VarLenFeature(tf.int64),
            'num_boxes': tf.FixedLenFeature([], tf.int64),
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
    image_info = {'image_name': image_name}

    image_info['height'] = height
    image_info['width'] = width
    image_info['image_id'] = image_id

    image_info['num_boxes'] = tf.cast(features['num_boxes'], tf.int32)
    image_info['is_crowd'] = tf.sparse.to_dense(features['is_crowd'], default_value=0)

    image_info['xmin'] = tf.sparse.to_dense(features['xmin'], default_value=0)
    image_info['xmax'] = tf.sparse.to_dense(features['xmax'], default_value=0)
    image_info['ymin'] = tf.sparse.to_dense(features['ymin'], default_value=0)
    image_info['ymax'] = tf.sparse.to_dense(features['ymax'], default_value=0)
    image_info['area'] = tf.sparse.to_dense(features['area'], default_value=0)
    image_info['category_id'] = tf.sparse.to_dense(features['category_id'], default_value=0)

    return [image, image_info]
