import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.utils import path_resolver


@POSTPROCESS_FACTORY.register(name="text_encoder")
def text_encoding_postprocessing(
    endnodes,
    device_pre_post_layers,
    postprocess_config_file,
    gt_images,
    **kwargs,
):
    last_token = gt_images["last_token"]
    text_projection = np.load(path_resolver.resolve_data_path(postprocess_config_file))["projection_layer"]
    final_state_token = tf.gather(endnodes, last_token, axis=2, batch_dims=1)
    result = final_state_token @ text_projection.T
    result = tf.math.l2_normalize(result, axis=-1)
    return {"predictions": result}


@VISUALIZATION_FACTORY.register(name="text_encoder")
def visualize_text_encoder_result(logits, image, **kwargs):
    raise NotImplementedError(f"Visualization for {kwargs['meta_arch']} is not implemented")
