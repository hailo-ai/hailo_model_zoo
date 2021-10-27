"""Contains a factory for image preprocessing."""
import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.preprocessing import classification_preprocessing
from hailo_model_zoo.core.preprocessing import segmentation_preprocessing
from hailo_model_zoo.core.preprocessing import detection_preprocessing
from hailo_model_zoo.core.preprocessing import pose_preprocessing
from hailo_model_zoo.core.preprocessing import centerpose_preprocessing
from hailo_model_zoo.core.preprocessing import super_resolution_preprocessing
from hailo_model_zoo.core.preprocessing import mono_depth_estimation_preprocessing
from hailo_model_zoo.core.preprocessing import lane_detection_preprocessing
from hailo_model_zoo.core.preprocessing import face_landmarks_preprocessing


def convert_rgb_to_yuv(image):
    transition_matrix = np.array([[0.2568619, -0.14823364, 0.43923104],
                                  [0.5042455, -0.2909974, -0.367758],
                                  [0.09799913, 0.43923104, -0.07147305]])
    image = tf.matmul(image, transition_matrix)
    image += [16, 128, 128]
    return image


def image_resize(image, shape):
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, tuple(shape), align_corners=True)
    return tf.squeeze(image, [0])


def normalize(image, normalization_params):
    mean, std = normalization_params
    return (image - list(mean)) / list(std)


def get_preprocessing(name, height, width, normalization_params, **kwargs):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
        name: meta architecture name
        height: height of the input image
        width: width of the input image
        normalization_params: parameters for normalizing the input image
        kwargs: additional preprocessing arguments

    Returns:
        preprocessing_fn: A function that preprocessing a single image (pre-batch).
            It has the following signature:
                image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
        ValueError: If Preprocessing `name` is not recognized.
    """
    preprocessing_fn_map = {
        'basic_resnet': classification_preprocessing.resnet_v1_18_34,
        'fcn_resnet': segmentation_preprocessing.resnet_v1_18,
        'fcn_resnet_bw': segmentation_preprocessing.resnet_bw_18,
        'mobilenet': classification_preprocessing.mobilenet_v1,
        'efficientnet': classification_preprocessing.efficientnet,
        'mobilenet_ssd': detection_preprocessing.mobilenet_ssd,
        'mobilenet_ssd_ar': detection_preprocessing.mobilenet_ssd_ar_preserving,
        'resnet_ssd': detection_preprocessing.resnet_v1_18_detection,
        'regnet_detection': detection_preprocessing.regnet_detection,
        'yolo_v3': detection_preprocessing.yolo_v3,
        'yolo_v5': detection_preprocessing.yolo_v5,
        'faster_rcnn_stage2': detection_preprocessing.faster_rcnn_stage2,
        'centernet': detection_preprocessing.centernet_resnet_v1_18_detection,
        'retinaface': detection_preprocessing.retinaface,
        'face_ssd': detection_preprocessing.face_ssd,
        'sr_resnet': super_resolution_preprocessing.resnet,
        'srgan': super_resolution_preprocessing.srgan,
        'openpose': pose_preprocessing.openpose_tf_preproc,
        'centerpose': centerpose_preprocessing.centerpose_preprocessing,
        'mono_depth': mono_depth_estimation_preprocessing.mono_depth_2,
        'polylanenet': lane_detection_preprocessing.polylanenet,
        'fair_mot': detection_preprocessing.fair_mot,
        'face_landmark_cnn': face_landmarks_preprocessing.face_landmark_cnn,
        'smoke': detection_preprocessing.centernet_resnet_v1_18_detection,
        'face_landmark_cnn_3d': face_landmarks_preprocessing.face_landmark_cnn,
    }
    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)
    flip = kwargs.pop('flip', False)
    yuv2rgb = kwargs.pop('yuv2rgb', False)
    input_resize = kwargs.pop('input_resize', {})
    if flip:
        height, width = width, height

    def preprocessing_fn(image, image_info=None):
        image, image_info = preprocessing_fn_map[name](image, image_info, height, width, flip=flip, **kwargs)
        if normalization_params:
            image = normalize(image, normalization_params)
        if yuv2rgb:
            image = convert_rgb_to_yuv(image)
        if input_resize.get('enabled', False):
            image = image_resize(image, input_resize.input_shape)
        return image, image_info
    return preprocessing_fn
