import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.utils import path_resolver

MAX_CLASSES_TO_VISUALIZE = 5


class PostprocessingException(Exception):
    pass


def _is_logits_shape_allowed(shape, classes):
    if len(shape) == 2:
        return True
    if len(shape) == 4 and shape[1] == 1 and shape[2] == 1 and shape[3] == classes:
        return True
    return False


def softmax(logits):
    e = np.exp(logits - np.max(logits))  # subtract max to avoid numerical instability
    return e / np.sum(e)


@POSTPROCESS_FACTORY.register(name="person_attr")
@POSTPROCESS_FACTORY.register(name="classification")
@POSTPROCESS_FACTORY.register(name="video_classification")
def classification_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    if device_pre_post_layers is not None and device_pre_post_layers["softmax"]:
        probs = endnodes
    else:
        logits = endnodes
        if not _is_logits_shape_allowed(logits.shape, kwargs["classes"]):
            raise PostprocessingException("Unexpected logits shape {}".format(logits.shape))
        # Verify the shape of the logits tensor has four dimensions
        logits = tf.reshape(logits, [-1, 1, 1, kwargs["classes"]])
        probs = tf.nn.softmax(tf.squeeze(logits, axis=(1, 2)), axis=1)
    return {"predictions": probs}


def _get_imagenet_labels():
    imagenet_names = json.load(open(os.path.join(os.path.dirname(__file__), "imagenet_names.json")))
    imagenet_names = [imagenet_names[str(i)] for i in range(1001)]
    return imagenet_names[1:]


def _get_peta_labels():
    peta_names = json.load(open(os.path.join(os.path.dirname(__file__), "peta_names.json")))
    peta_names = [peta_names[str(i)] for i in range(35)]
    return peta_names


def _get_kinetics400_labels():
    imagenet_names = json.load(open(os.path.join(os.path.dirname(__file__), "kinetics400_names.json")))
    imagenet_names = [imagenet_names[str(i)] for i in range(400)]
    return imagenet_names[0:]


@VISUALIZATION_FACTORY.register(name="classification")
@VISUALIZATION_FACTORY.register(name="zero_shot_classification")
def visualize_classification_result(logits, img, **kwargs):
    logits = logits["predictions"]
    # TODO: SDK-32906 (wrong shape for classifiers) remove this when sdk is fixed
    if len(logits.shape) == 4:
        logits = logits.squeeze((1, 2))
    labels_offset = kwargs.get("labels_offset", 0)
    top1 = np.argmax(logits, axis=1)
    conf = np.squeeze(logits[0, top1])
    imagenet_labels = _get_imagenet_labels()
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text(
        (0, 0), "{} ({:.2f})".format(imagenet_labels[int(top1[0] - labels_offset)], conf), (255, 0, 0)
    )
    return np.array(img_orig, np.uint8)


@VISUALIZATION_FACTORY.register(name="person_attr")
def visualize_multi_classification_result(logits, img, **kwargs):
    peta_labels = _get_peta_labels()
    logits = logits["predictions"].squeeze()
    preds = np.array(logits >= 0, np.int64)
    confidences = softmax(logits)
    matches = [(label, conf) for label, conf, pred in zip(peta_labels, confidences, preds) if pred == 1]
    matches = sorted(matches, key=lambda match: match[1], reverse=True)  # Sort by confidence
    matches_to_visualize = matches[:MAX_CLASSES_TO_VISUALIZE]

    img_orig = Image.fromarray(img[0])
    draw = ImageDraw.Draw(img_orig)
    text_position = (5, 5)
    for match in matches_to_visualize:
        label, _ = match
        draw.text(text_position, label, fill="blue")
        text_position = (text_position[0], text_position[1] + 10)
    return np.array(img_orig, np.uint8)


@VISUALIZATION_FACTORY.register(name="video_classification")
def visualize_kinetics400_classification_result(logits, img, **kwargs):
    logits = logits["predictions"]
    if len(logits.shape) == 4:
        logits = logits.squeeze((1, 2))
    labels_offset = kwargs.get("labels_offset", 0)
    top1 = np.argmax(logits, axis=1)
    conf = np.squeeze(logits[0, top1])
    kinetics400_labels = _get_kinetics400_labels()
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text(
        (0, 0), "{} ({:.2f})".format(kinetics400_labels[int(top1[0] - labels_offset)], conf), (255, 0, 0)
    )
    return np.array(img_orig, np.uint8)


@POSTPROCESS_FACTORY.register(name="zero_shot_classification")
def zero_shot_classification_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    endnodes /= tf.norm(endnodes, keepdims=True, axis=-1)
    path = path_resolver.resolve_data_path(kwargs["postprocess_config_file"])
    text_features = np.load(path)
    similarity = tf.linalg.matmul(100.0 * endnodes, text_features, transpose_b=True)
    if len(similarity.shape) == 4:
        similarity = tf.squeeze(similarity, [1, 2])
    probs = tf.nn.softmax(similarity, axis=-1)
    return {"predictions": probs}
