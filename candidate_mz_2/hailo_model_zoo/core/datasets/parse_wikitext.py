import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="wikitext2")
def parse_wikitext2(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "text": tf.io.FixedLenFeature([], tf.string),
            "attention_mask": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "model_input": tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            "channels": tf.io.FixedLenFeature([], tf.int64),
            "pad": tf.io.FixedLenFeature([], tf.int64),
            "mask_index": tf.io.FixedLenFeature([], tf.int64),
            "original_token_id": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    text = tf.cast(features["text"], tf.string)
    pad = tf.cast(features["pad"], tf.int32)
    channels = tf.cast(features["channels"], tf.int32)
    mask_index = tf.cast(features["mask_index"], tf.int32)
    label_index = tf.cast(features["original_token_id"], tf.int32)
    attention_mask = tf.cast(features["attention_mask"], tf.int32)
    model_input = tf.cast(features["model_input"], tf.float32)
    model_input = tf.reshape(model_input, [pad, channels])
    attention_mask = tf.reshape(attention_mask, [pad])
    image_info = {
        "text": text,
        "attention_mask": attention_mask,
        "pad": pad,
        "channels": channels,
        "mask_index": mask_index,
        "label_index": label_index,
    }
    return [model_input, image_info]
