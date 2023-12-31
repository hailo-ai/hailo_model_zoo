import tensorflow as tf


def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'landmarks_68_3d_xy_normalized': tf.io.FixedLenFeature([68, 2], tf.float32),
            'landmarks_68_3d_z': tf.io.FixedLenFeature([68, 1], tf.float32),
        })
    image = tf.cast(tf.image.decode_jpeg(features['image'], channels=3), tf.uint8)
    image = tf.reshape(image, (450, 450, 3))
    shape = tf.shape(image)
    landmarks = features['landmarks_68_3d_xy_normalized'] * 450.0

    height, width = shape[0], shape[1]

    unstacked_landmarks = tf.unstack(landmarks, axis=1)
    x_coords, y_coords = unstacked_landmarks[0], unstacked_landmarks[1]
    x_min = tf.reduce_min(x_coords)
    y_min = tf.reduce_min(y_coords)
    x_max = tf.reduce_max(x_coords)
    y_max = tf.reduce_max(y_coords)

    center_x = tf.reduce_mean([x_min, x_max])
    center_y = tf.reduce_mean([y_min, y_max])
    radius_x = (x_max - x_min) / 2.0
    radius_y = (y_max - y_min) / 2.0
    long_side = tf.maximum(radius_x, radius_y)
    length = tf.sqrt(2.0) * long_side

    roi_box = [center_x - length, center_y - length, center_x + length, center_y + length]
    roi_box = [tf.cast(tf.round(coord), tf.int32) for coord in roi_box]

    # Pad image in case roi escapes its borders
    pad_left = tf.maximum(0, -roi_box[0])
    pad_up = tf.maximum(0, -roi_box[1])

    padded_image = tf.image.pad_to_bounding_box(image,
                                                pad_up,
                                                pad_left,
                                                tf.maximum(roi_box[3], height) + pad_up,
                                                tf.maximum(roi_box[2], width) + pad_left)
    face_image = tf.image.crop_to_bounding_box(
        padded_image, roi_box[1] + pad_up, roi_box[0] + pad_left, roi_box[3] - roi_box[1], roi_box[2] - roi_box[0])

    landmarks = tf.concat([landmarks, features['landmarks_68_3d_z']], axis=-1)
    image_info = {
        'landmarks': landmarks,
        'roi_box': roi_box,
        'uncropped_image': image
    }
    return [face_image, image_info]
