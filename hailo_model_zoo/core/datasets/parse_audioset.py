import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="audioset")
def parse_record(serialized_example):
    """Parse one AudioSet log-mel patch record.

    Schema written by `datasets/create_audioset_tfrecord.py`. Each entry is
    one log-mel patch plus the clip-level multi-hot label vector. The patch
    shape (height x width x 1) and number of classes are read from the
    record itself, so a future audio model that wants different STFT
    parameters or a different label space can reuse this parser.

    Returns:
        [patch, info]
            patch: tf.float32, shape (height, width, 1)
            info:  dict with the keys the eval module needs to aggregate
                   per-clip predictions.
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "patch_bytes": tf.io.FixedLenFeature([], tf.string),
            "patch_height": tf.io.FixedLenFeature([], tf.int64),
            "patch_width": tf.io.FixedLenFeature([], tf.int64),
            "label_multi_hot_bytes": tf.io.FixedLenFeature([], tf.string),
            "num_classes": tf.io.FixedLenFeature([], tf.int64),
            "clip_id": tf.io.FixedLenFeature([], tf.string),
            "patch_index": tf.io.FixedLenFeature([], tf.int64),
        },
    )

    height = tf.cast(features["patch_height"], tf.int32)
    width = tf.cast(features["patch_width"], tf.int32)
    num_classes = tf.cast(features["num_classes"], tf.int32)

    patch = tf.io.decode_raw(features["patch_bytes"], tf.float32)
    patch = tf.reshape(patch, tf.stack([height, width, 1]))

    label_multi_hot = tf.io.decode_raw(features["label_multi_hot_bytes"], tf.int8)
    label_multi_hot = tf.reshape(label_multi_hot, tf.stack([num_classes]))

    info = {
        "image_name": features["clip_id"],
        "label_multi_hot": label_multi_hot,
        "patch_index": tf.cast(features["patch_index"], tf.int32),
    }
    return [patch, info]
