import tensorflow as tf


def person_reid_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    embeddings = endnodes
    embeddings = tf.nn.l2_normalize(endnodes, 1, 1e-10, name='embeddings')
    return {'predictions': embeddings}
