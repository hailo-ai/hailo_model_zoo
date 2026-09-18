import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY


@POSTPROCESS_FACTORY.register(name="mivolo")
def mivolo_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    """MiVOLO output is [gender(2 logits), age(1 in years)].

    Age de-normalization (age = out*94 + 48) is folded into the network head, so the age
    channel is already in years. Gender is a 2-way argmax (index 0 = male); softmax is
    monotonic so the raw-logit compare is identical and avoids a redundant host softmax.
    Returns the {"age", "is_male"} dict the AgeGenderEval metric consumes.
    """
    out = endnodes[0] if isinstance(endnodes, (list, tuple)) else endnodes
    out = tf.reshape(out, (-1, 3))
    is_male = tf.reshape(out[:, 0] >= out[:, 1], (-1, 1))  # [B,1]: eval indexes [i][0]
    age = out[:, 2]
    return {"age": age, "is_male": is_male}


@VISUALIZATION_FACTORY.register(name="mivolo")
def visualize_mivolo_result(logits, img, **kwargs):
    gender = "Male" if logits["is_male"][0] else "Female"
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text((0, 0), "Age: {:.1f}\t Gender: {}".format(logits["age"][0], gender), (255, 0, 0))
    return np.array(img_orig, np.uint8)
