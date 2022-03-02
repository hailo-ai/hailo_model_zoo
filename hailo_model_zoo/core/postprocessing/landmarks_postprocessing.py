import numpy as np
from PIL import Image, ImageDraw
import cv2


def face_landmarks_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    shape = kwargs['img_dims']
    return {'predictions': endnodes * shape[0]}


def hand_landmarks_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    return {'predictions': endnodes[0]}


def visualize_face_landmarks_result(logits, image, **kwargs):
    logits = logits['predictions']
    img = Image.fromarray(image[0])
    img_draw = ImageDraw.Draw(img)
    img_draw.point(logits[0], fill=(255, 255, 255))
    return np.array(img)


def visualize_hand_landmarks_result(logits, image, **kwargs):
    logits = logits['predictions'][0]
    img = image[0]
    pts = [x for i, x in enumerate(logits) if (i + 1) % 3 != 0]
    for i in range(len(pts)):
        if i % 2 == 0:
            continue
        img = cv2.circle(img, (pts[i - 1], pts[i]), radius=1, color=(0, 0, 255), thickness=2)
    return img
