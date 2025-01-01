import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY


@POSTPROCESS_FACTORY.register(name="person_reid")
def person_reid_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    embeddings = endnodes
    embeddings = tf.nn.l2_normalize(endnodes, -1, 1e-10, name="embeddings")
    return {"predictions": embeddings}
