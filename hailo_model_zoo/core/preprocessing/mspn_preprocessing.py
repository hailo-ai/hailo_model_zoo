import tensorflow as tf
import cv2
import numpy as np
import math

from hailo_model_zoo.core.preprocessing.affine_utils import get_affine_transform


def _bbox_xywh2cs(bbox, aspect_ratio, pixel_std=200., padding=1.25):
    """Transform the bbox format from (x,y,w,h) into (center, scale)
    Args:
        bbox (ndarray): Single bbox in (x, y, w, h)
        aspect_ratio (float): The expected bbox aspect ratio (w over h)
        padding (float): Bbox padding factor that will be multilied to scale.
            Default: 1.25
        pixel_std (float): The scale normalization factor. Default: 200.0
    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """
    x, y, w, h = bbox

    center = np.array([x + w / 2., y + h / 2.], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32) / pixel_std
    scale = scale * padding

    return center, scale


def _mspn_preprocessing(image, aspect_ratio, bbox, height, width, pixel_std=200., padding=1.25):
    image = np.array(image)
    center, scale = _bbox_xywh2cs(bbox, aspect_ratio, pixel_std=pixel_std, padding=padding)
    trans_input = get_affine_transform(center, scale, 0, [float(width), float(height)], pixel_std=pixel_std)
    inp_image = cv2.warpAffine(image, trans_input, (int(width), int(height)), flags=cv2.INTER_LINEAR)

    return inp_image, center, scale


def _get_bbox_xywh(image_info):
    # Get box info if exists, otherwise assume box spans the entire image
    bbox = image_info.get('bbox', [[0, 1, 0, 1]])[0]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    xmin *= tf.cast(image_info['orig_width'], tf.float32)
    xmax *= tf.cast(image_info['orig_width'], tf.float32)
    ymin *= tf.cast(image_info['orig_height'], tf.float32)
    ymax *= tf.cast(image_info['orig_height'], tf.float32)
    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    return bbox


def mspn(image, image_info=None, height=None, width=None, **kwargs):
    image_info['orig_height'], image_info['orig_width'] = tf.shape(image)[0], tf.shape(image)[1]
    image_info['img_orig'] = tf.image.encode_jpeg(image, quality=100)

    aspect_ratio = width / height
    bbox = _get_bbox_xywh(image_info)

    image, center, scale = tf.py_function(_mspn_preprocessing,
                                          [image, aspect_ratio, bbox, height, width],
                                          [tf.float32, tf.float32, tf.float32])

    image.set_shape((height, width, 3))
    image_info['img_resized'] = image
    image_info['center'], image_info['scale'] = center, scale

    return image, image_info


def vit_pose(image, image_info=None, height=None, width=None, **kwargs):
    image_info['orig_height'], image_info['orig_width'] = tf.shape(image)[0], tf.shape(image)[1]
    image_info['img_orig'] = tf.image.encode_jpeg(image, quality=100)

    aspect_ratio = tf.cast(image_info['orig_height'], tf.float32) / tf.cast(image_info['orig_width'], tf.float32)
    aspect_ratio = width / height

    bbox = _get_bbox_xywh(image_info)

    image, center, scale = tf.py_function(_vit_pose_preprocessing,
                                          [image, aspect_ratio, bbox, height, width],
                                          [tf.float32, tf.float32, tf.float32])

    image.set_shape((height, width, 3))
    image_info['img_resized'] = image
    image_info['center'], image_info['scale'] = center, scale

    return image, image_info


def _vit_pose_preprocessing(image, aspect_ratio, bbox, height, width, pixel_std=200., padding=1.25):
    image = np.array(image).astype(np.uint8)
    center, scale = _bbox_xywh2cs(bbox, aspect_ratio, pixel_std=pixel_std, padding=padding)

    image_size_npy = np.array([width, height])

    trans = get_warp_matrix(0, center * 2.0, image_size_npy - 1.0, scale * 200.0)
    inp_image = cv2.warpAffine(image, trans, (image_size_npy[0], image_size_npy[1]), flags=cv2.INTER_LINEAR)

    return inp_image, center, scale


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta)
                              + 0.5 * size_input[1] * math.sin(theta)
                              + 0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta)
                              - 0.5 * size_input[1] * math.cos(theta)
                              + 0.5 * size_target[1])
    return matrix
