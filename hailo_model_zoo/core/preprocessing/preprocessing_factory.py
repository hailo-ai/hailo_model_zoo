"""Contains a factory for image preprocessing."""
import numpy as np
import tensorflow as tf
from hailo_model_zoo.core.preprocessing import classification_preprocessing
from hailo_model_zoo.core.preprocessing import segmentation_preprocessing
from hailo_model_zoo.core.preprocessing import detection_preprocessing
from hailo_model_zoo.core.preprocessing import pose_preprocessing
from hailo_model_zoo.core.preprocessing import centerpose_preprocessing
from hailo_model_zoo.core.preprocessing import super_resolution_preprocessing
from hailo_model_zoo.core.preprocessing import lane_detection_preprocessing
from hailo_model_zoo.core.preprocessing import face_landmarks_preprocessing
from hailo_model_zoo.core.preprocessing import depth_estimation_preprocessing
from hailo_model_zoo.core.preprocessing import person_reid_preprocessing
from hailo_model_zoo.core.preprocessing import mspn_preprocessing
from hailo_model_zoo.core.preprocessing import low_light_enhancement_preprocessing
from hailo_model_zoo.core.preprocessing import stereonet_preprocessing
from hailo_model_zoo.core.preprocessing import image_denoising_preprocessing


def convert_rgb_to_yuv(image):
    transition_matrix = np.array([[0.2568619, -0.14823364, 0.43923104],
                                  [0.5042455, -0.2909974, -0.367758],
                                  [0.09799913, 0.43923104, -0.07147305]])
    image = tf.matmul(image, transition_matrix)
    image += [16, 128, 128]
    return image


def convert_yuv_to_yuy2(image):
    y_img = image[..., 0]
    uv_img = image[..., 1:]
    uv_subsampled = uv_img[:, ::2, :]
    uv_unrolled = tf.reshape(uv_subsampled, (image.shape[-3], image.shape[-2]))
    yuy2_img = tf.stack([y_img, uv_unrolled], axis=-1)
    return yuy2_img


def convert_yuv_to_nv12(image):
    h, w, _ = image.shape
    y_img = image[..., 0]
    y = tf.reshape(y_img, (2 * h, w // 2))
    u = image[..., 1]
    v = image[..., 2]
    u_subsample = tf.expand_dims(u[::2, ::2], axis=-1)
    v_subsample = tf.expand_dims(v[::2, ::2], axis=-1)
    uv = tf.stack([u_subsample, v_subsample], axis=-1)
    uv = tf.reshape(uv, (h, w // 2))
    return tf.reshape(tf.concat([y, uv], axis=0), (h // 2, w, 3))


def convert_rgb_to_rgbx(image):
    h, w, _ = image.shape
    new_channel = tf.zeros((h, w, 1))
    return tf.concat([image, new_channel], axis=-1)


def image_resize(image, shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, tuple(shape), method='bilinear')
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
        'fastvit': classification_preprocessing.fastvit,
        'efficientnet': classification_preprocessing.efficientnet,
        'mobilenet_ssd': detection_preprocessing.mobilenet_ssd,
        'mobilenet_ssd_ar': detection_preprocessing.mobilenet_ssd_ar_preserving,
        'resnet_ssd': detection_preprocessing.resnet_v1_18_detection,
        'regnet_detection': detection_preprocessing.regnet_detection,
        'yolo_v3': detection_preprocessing.yolo_v3,
        'yolo_v5': detection_preprocessing.yolo_v5,
        'detr': detection_preprocessing.detr,
        'faster_rcnn_stage2': detection_preprocessing.faster_rcnn_stage2,
        'centernet': detection_preprocessing.centernet_resnet_v1_18_detection,
        'retinaface': detection_preprocessing.retinaface,
        'face_ssd': detection_preprocessing.face_ssd,
        'sr_resnet': super_resolution_preprocessing.resnet,
        'srgan': super_resolution_preprocessing.srgan,
        'zero_dce': low_light_enhancement_preprocessing.zero_dce,
        'openpose': pose_preprocessing.openpose_tf_preproc,
        'yolov8_pose': pose_preprocessing.yolo_pose,
        'centerpose': centerpose_preprocessing.centerpose_preprocessing,
        'mono_depth': depth_estimation_preprocessing.mono_depth_2,
        'polylanenet': lane_detection_preprocessing.polylanenet,
        'fair_mot': detection_preprocessing.fair_mot,
        'face_landmark_cnn': face_landmarks_preprocessing.face_landmark_cnn,
        'smoke': detection_preprocessing.centernet_resnet_v1_18_detection,
        'face_landmark_cnn_3d': face_landmarks_preprocessing.face_landmark_cnn,
        'resmlp': classification_preprocessing.resmlp,
        'fast_depth': depth_estimation_preprocessing.fast_depth,
        'lprnet': classification_preprocessing.lprnet,
        'clip': classification_preprocessing.clip,
        'person_reid': person_reid_preprocessing.market1501,
        'mspn': mspn_preprocessing.mspn,
        'vit_pose': mspn_preprocessing.vit_pose,
        'vit': classification_preprocessing.vit_tiny,
        'espcn': super_resolution_preprocessing.espcn,
        'retinanet_resnext50': detection_preprocessing.retinanet_resnext50,
        'sparseinst': segmentation_preprocessing.sparseinst,
        'resnet_pruned': classification_preprocessing.resnet_pruned,
        'stereonet': stereonet_preprocessing.stereonet,
        'dncnn3': image_denoising_preprocessing.dncnn3,
        'scdepthv3': depth_estimation_preprocessing.scdepthv3
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)
    flip = kwargs.pop('flip', False)
    yuv2rgb = kwargs.pop('yuv2rgb', False)
    yuy2 = kwargs.pop('yuy2', False)
    nv12 = kwargs.pop('nv12', False)
    rgbx = kwargs.pop('rgbx', False)
    input_resize = kwargs.pop('input_resize', {})
    if flip:
        height, width = width, height

    def preprocessing_fn(image, image_info=None):
        image, image_info = preprocessing_fn_map[name](image, image_info, height, width, flip=flip, **kwargs)
        if normalization_params:
            image = normalize(image, normalization_params)
        if input_resize.get('enabled', False):
            image = image_resize(image, input_resize.input_shape)
        if yuv2rgb:
            image = convert_rgb_to_yuv(image)
        if yuy2:
            image = convert_yuv_to_yuy2(image)
        if nv12:
            image = convert_yuv_to_nv12(image)
        if rgbx:
            image = convert_rgb_to_rgbx(image)
        return image, image_info
    return preprocessing_fn
