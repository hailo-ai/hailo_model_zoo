import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="aflw2k3d_tddfa")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image_name': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'uncropped_image': tf.io.FixedLenFeature([], tf.string),
            'landmarks': tf.io.FixedLenFeature([68, 3], tf.float32),
            'landmarks_reannotated': tf.io.FixedLenFeature([68, 2], tf.float32),
            'yaw': tf.io.FixedLenFeature([1], tf.float32),
            'roi_box': tf.io.FixedLenFeature([4], tf.float32),
        })
    face_image = tf.cast(tf.image.decode_jpeg(features['image'], channels=3), tf.uint8)
    uncropped_image = tf.cast(tf.image.decode_jpeg(features['uncropped_image'], channels=3), tf.uint8)

    roi_box = tf.cast(tf.round(features['roi_box']), tf.int32)

    landmarks = features['landmarks']
    image_info = {
        'landmarks': landmarks,
        'roi_box': roi_box,
        'uncropped_image': uncropped_image,
        'yaw': features['yaw'],
    }
    return [face_image, image_info]
