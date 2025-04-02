import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="coco_sd1.5_encoded")
@DATASET_FACTORY.register(name="ip_adapter_faces_sd1_5")
def vae(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "encoded_image": tf.io.FixedLenFeature([], dtype=tf.string),
            "gt_image": tf.io.FixedLenFeature([], dtype=tf.string),
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )

    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    encoded_image = tf.reshape(tf.io.decode_raw(features["encoded_image"], tf.float32), (1, height // 8, width // 8, 4))
    gt_image = tf.reshape(tf.io.decode_raw(features["gt_image"], tf.uint8), (1, height, width, 3))

    image_info = {"img_orig": gt_image, "img_float": encoded_image}
    return (encoded_image, image_info)
