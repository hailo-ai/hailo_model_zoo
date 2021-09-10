"""Contains a factory for image preprocessing."""
from hailo_model_zoo.core.preprocessing import classification_preprocessing
from hailo_model_zoo.core.preprocessing import segmentation_preprocessing
from hailo_model_zoo.core.preprocessing import detection_preprocessing
from hailo_model_zoo.core.preprocessing import pose_preprocessing
from hailo_model_zoo.core.preprocessing import centerpose_preprocessing
from hailo_model_zoo.core.preprocessing import super_resolution_preprocessing
from hailo_model_zoo.core.preprocessing import mono_depth_estimation_preprocessing
from hailo_model_zoo.core.preprocessing import lane_detection_preprocessing
from hailo_model_zoo.core.preprocessing import face_landmarks_preprocessing


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
    if flip:
        height, width = width, height

    def preprocessing_fn(image, image_info=None):
        image, image_info = preprocessing_fn_map[name](image, image_info, height, width, flip=flip, **kwargs)
        if normalization_params:
            image = normalize(image, normalization_params)
        return image, image_info
    return preprocessing_fn
