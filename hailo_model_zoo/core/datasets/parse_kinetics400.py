import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="kinetics400")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "video": tf.io.FixedLenFeature([], tf.string),
            "tensor_name": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        },
    )

    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    video_bytes = features["video"]
    label_index = tf.cast(features["label"], tf.int32)
    image_name = tf.cast(features["tensor_name"], tf.string)

    # Decode and reshape the video tensor
    image = tf.io.decode_raw(video_bytes, tf.uint8)
    num_frames = tf.shape(image)[0] // (height * width * 3)  # Calculate the number of frames
    image = tf.reshape(image, [height, width, 3, num_frames])

    image_info = {
        "image_name": image_name,
        "label_index": label_index,
        "height_orig": height,
        "width_org": width,
        "num_frames": num_frames,
    }
    return [image, image_info]
