import numpy as np
from PIL import Image, ImageDraw


def face_landmarks_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    shape = kwargs['img_dims']
    return {'predictions': endnodes * shape[0]}


def visualize_face_landmarks_result(logits, image, **kwargs):
    logits = logits['predictions']
    img = Image.fromarray(image[0])
    img_draw = ImageDraw.Draw(img)
    img_draw.point(logits[0], fill=(255, 255, 255))
    return np.array(img)
