import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="age_gender")
def age_gender_postprocessing(endnodes, device_pre_post_layers, **kwargs):
    age_predictions, gender_predictions = endnodes

    is_male = gender_predictions >= 0.6

    # Softmax is done outside of net due to HW issue when vector_sze % 8 != 0
    age_predictions = tf.nn.softmax(age_predictions)

    # Find 2 highest ranking ages per image
    indices = tf.argsort(age_predictions, direction='DESCENDING')[:, :2]
    batch_indices = tf.cast(tf.reshape(tf.range(0, tf.size(indices) / 2, 0.5), (tf.shape(indices)[0], 2)), tf.int32)
    indices_for_gather = tf.stack([batch_indices, indices], axis=-1)

    top_age_probs = tf.gather_nd(age_predictions, indices_for_gather)

    top_age_probs_sum = tf.reshape(tf.reduce_sum(top_age_probs, axis=1), (-1, 1))
    # Weighted average of age probability
    norm_preds = top_age_probs / top_age_probs_sum

    # Multiply age (index) by its weights average
    res_age = tf.reduce_sum(tf.cast(indices, tf.float32) * norm_preds, axis=1)
    res_age += 1
    return {'age': res_age, 'is_male': is_male}


@VISUALIZATION_FACTORY.register(name="age_gender")
def visualize_age_gender_result(logits, img, **kwargs):
    gender = 'Male' if logits['is_male'][0] else 'Female'
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text((0, 0), 'Age: {:.2f}\t Gender: {}'.format(logits['age'][0], gender), (255, 0, 0))
    return np.array(img_orig, np.uint8)
