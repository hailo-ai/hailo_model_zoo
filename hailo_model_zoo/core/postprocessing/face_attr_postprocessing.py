import tensorflow as tf


def face_attr_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    preds = tf.reshape(endnodes, (-1, 40, 2))
    preds = tf.cast(tf.argmax(preds, axis=-1), tf.float32) - 0.5
    return {'predictions': preds}
