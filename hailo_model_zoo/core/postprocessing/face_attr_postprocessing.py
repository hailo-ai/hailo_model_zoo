import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY


@POSTPROCESS_FACTORY.register(name="face_attr")
def face_attr_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    preds = tf.reshape(endnodes, (-1, 40, 2))
    preds = tf.cast(tf.argmax(preds, axis=-1), tf.float32) - 0.5
    return {"predictions": preds}
