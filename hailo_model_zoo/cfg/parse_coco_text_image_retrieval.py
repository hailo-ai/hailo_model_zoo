import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY


@DATASET_FACTORY.register(name="coco_xtd10")
def coco_text_retrieval(serialized_example):
    """Parse serialized example of TfRecord and extract dictionary of all the information"""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "input_ids": tf.io.FixedLenFeature([], dtype=tf.string),
            "input_embeds": tf.io.FixedLenFeature([], dtype=tf.string),
            "image_embeds": tf.io.FixedLenFeature([], dtype=tf.string),
            "text_embeds": tf.io.FixedLenFeature([], dtype=tf.string),
            "input_embeds_dim": tf.io.FixedLenFeature([], dtype=tf.int64),
            "output_embeds_dim": tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )

    input_embed_dim = features["input_embeds_dim"]
    output_embed_dim = features["output_embeds_dim"]
    input_ids = tf.reshape(tf.io.decode_raw(features["input_ids"], tf.int64), (-1,))
    input_embeds = tf.reshape(tf.io.decode_raw(features["input_embeds"], tf.float32), (1, -1, input_embed_dim))
    image_embeds = tf.reshape(tf.io.decode_raw(features["image_embeds"], tf.float32), (1, -1, output_embed_dim))
    text_embeds = tf.reshape(tf.io.decode_raw(features["text_embeds"], tf.float32), (1, -1, output_embed_dim))

    image_info = {
        "input_ids": input_ids,
        "image_embeds": image_embeds,
        "text_embeds": text_embeds,
    }
    return input_embeds, image_info
