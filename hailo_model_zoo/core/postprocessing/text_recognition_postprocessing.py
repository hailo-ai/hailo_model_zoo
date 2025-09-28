import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.utils import path_resolver

"""
    Text Recognition Postprocessing for PaddleOCR Text Recognition.
    Based on CTCLabelDecode class from PaddleOCR repository:
    https://github.com/PaddlePaddle/PaddleX/blob/d824981d50790548432daea51e5861b4fc56b82d/paddlex/inference/models/text_recognition/processors.py#L183
"""


def decode(endnodes, character):
    is_remove_duplicate = True
    preds = endnodes.squeeze(1)
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)

    ignored_tokens = [0, 1]

    decoded_texts = []
    decoded_scores = []

    for curr_pred, curr_conf in zip(preds_idx, preds_prob):
        char_list = []
        conf_list = []

        for idx in range(len(curr_pred)):
            if curr_pred[idx] in ignored_tokens:
                continue

            if is_remove_duplicate:
                if idx > 0 and curr_pred[idx - 1] == curr_pred[idx]:
                    # Skip duplicate characters, which common in CTC outputs (Connectionist Temporal Classification)
                    continue

            if curr_pred[idx] >= len(character) + len(ignored_tokens):
                # This case where the prediction is out of bounds,
                # happens due to format change of the character array (txt->npz)
                continue

            char_list.append(character[int(curr_pred[idx]) - len(ignored_tokens)])
            conf_list.append(curr_conf[idx])

        text = b"".join(char_list).decode("utf-8")

        mean_conf = np.mean(conf_list, dtype=np.float32) if conf_list else 0.0
        decoded_texts.append(text)
        decoded_scores.append(mean_conf)

    return (
        np.array([t.encode("utf-8") for t in decoded_texts], dtype=np.object_),
        np.array(decoded_scores, dtype=np.float32),
    )


@POSTPROCESS_FACTORY.register(name="text_recognition")
def postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    character = np.load(path_resolver.resolve_data_path(kwargs["postprocess_config_file"]))["dictionary"]
    decoded_texts, decoded_scores = tf.numpy_function(decode, [endnodes, character], [tf.string, tf.float32])

    return {"text": decoded_texts, "score": decoded_scores}


@VISUALIZATION_FACTORY.register(name="text_recognition")
def visualize_text_recognition(logits, img, **kwargs):
    img_orig = Image.fromarray(img.squeeze(0).astype(np.uint8))
    draw = ImageDraw.Draw(img_orig)

    text_label = logits["text"][0].decode("utf-8")

    if not text_label.isascii():
        # If the text contains non-ASCII characters (such as chinese letters), we replace it with a placeholder
        text_label = "###"

    text_conf = logits["score"][0]
    vis_string = f"{text_label}: ({text_conf:.2f})"

    text_position = (3, 38)
    draw.text(text_position, vis_string, fill="blue")
    return np.array(img_orig, np.uint8)
