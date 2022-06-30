import tensorflow as tf
import numpy as np
import os
import json
from PIL import Image
from PIL import ImageDraw


class PostprocessingException(Exception):
    pass


def _is_logits_shape_allowed(shape, classes):
    if len(shape) == 2:
        return True
    if len(shape) == 4 and shape[1] == 1 and shape[2] == 1 and shape[3] == classes:
        return True
    return False


def classification_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    if device_pre_post_layers is not None and device_pre_post_layers['softmax']:
        probs = endnodes
    else:
        logits = endnodes
        if not _is_logits_shape_allowed(logits.shape, kwargs['classes']):
            raise PostprocessingException('Unexpected logits shape {}'.format(logits.shape))
        # Verify the shape of the logits tensor has four dimensions
        logits = tf.reshape(logits, [-1, 1, 1, kwargs['classes']])
        probs = tf.nn.softmax(tf.squeeze(logits, axis=(1, 2)), axis=1)
    return {'predictions': probs}


def _get_imagenet_labels():
    imagenet_names = json.load(open(os.path.join(os.path.dirname(__file__), 'imagenet_names.json')))
    imagenet_names = [imagenet_names[str(i)] for i in range(1001)]
    return imagenet_names[1:]


def visualize_classification_result(logits, img, **kwargs):
    logits = logits['predictions']
    labels_offset = kwargs.get('labels_offset', 0)
    top1 = np.argmax(logits, axis=1)
    conf = np.squeeze(logits[0, top1])
    imagenet_labels = _get_imagenet_labels()
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text((0, 0), "{} ({:.2f})".format(imagenet_labels[int(top1[0] - labels_offset)],
                                                               conf), (255, 0, 0))
    return np.array(img_orig, np.uint8)
