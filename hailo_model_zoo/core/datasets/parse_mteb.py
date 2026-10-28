import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="mteb")
def parse_record(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "query_id": tf.io.FixedLenFeature([], tf.string),
            "query_text": tf.io.FixedLenFeature([], tf.string),
            "doc_id": tf.io.FixedLenFeature([], tf.string),
            "doc_text": tf.io.FixedLenFeature([], tf.string),
            "multiplicative_mask": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            "model_input": tf.io.VarLenFeature(tf.float32),
            "channels": tf.io.FixedLenFeature([], tf.int64),
            "pad": tf.io.FixedLenFeature([], tf.int64),
        },
    )

    pad = tf.cast(features["pad"], tf.int32)
    channels = tf.cast(features["channels"], tf.int32)

    model_input = tf.sparse.to_dense(features["model_input"], default_value=0.0)
    model_input = tf.reshape(model_input, tf.stack([pad, channels]))

    multiplicative_mask = tf.cast(features["multiplicative_mask"], tf.float32)

    query_id = tf.cast(features["query_id"], tf.string)
    query_text = tf.cast(features["query_text"], tf.string)
    doc_id = tf.cast(features["doc_id"], tf.string)
    doc_text = tf.cast(features["doc_text"], tf.string)

    image_info = {
        "query_id": query_id,
        "query_text": query_text,
        "doc_id": doc_id,
        "doc_text": doc_text,
        "multiplicative_mask": multiplicative_mask,
        "pad": pad,
        "channels": channels,
    }
    return [model_input, image_info]
