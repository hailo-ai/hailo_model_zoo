"""Contains a factory for network postprocessing."""
from hailo_model_zoo.core.postprocessing.age_gender_postprocessing import (age_gender_postprocessing,
                                                                           visualize_age_gender_result)
from hailo_model_zoo.core.postprocessing.classification_postprocessing import (classification_postprocessing,
                                                                               zero_shot_classification_postprocessing,
                                                                               visualize_classification_result)
from hailo_model_zoo.core.postprocessing.detection_postprocessing import (detection_postprocessing,
                                                                          visualize_detection_result)
from hailo_model_zoo.core.postprocessing.face_detection_postprocessing import (face_detection_postprocessing)
from hailo_model_zoo.core.postprocessing.segmentation_postprocessing import (segmentation_postprocessing,
                                                                             visualize_segmentation_result)
from hailo_model_zoo.core.postprocessing.facenet_postprocessing import (facenet_postprocessing, visualize_face_result)
from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import (
    instance_segmentation_postprocessing, visualize_instance_segmentation_result)
from hailo_model_zoo.core.postprocessing.pose_estimation_postprocessing import (pose_estimation_postprocessing,
                                                                                visualize_pose_estimation_result)
from hailo_model_zoo.core.postprocessing.mono_depth_estimation_postprocessing import (
    mono_depth_estimation_postprocessing, visualize_mono_depth_result)
from hailo_model_zoo.core.postprocessing.super_resolution_postprocessing import (super_resolution_postprocessing,
                                                                                 visualize_super_resolution_result,
                                                                                 visualize_srgan_result)
from hailo_model_zoo.core.postprocessing.low_light_enhancement_postprocessing import (
    low_light_enhancement_postprocessing,
    visualize_low_light_enhancement_result)
from hailo_model_zoo.core.postprocessing.head_pose_estimation_postprocessing import (
    head_pose_estimation_postprocessing, visualize_head_pose_result)
from hailo_model_zoo.core.postprocessing.lane_detection_postprocessing import (lane_detection_postprocessing,
                                                                               visualize_lane_detection_result)
from hailo_model_zoo.core.postprocessing.multiple_object_tracking_postprocessing import (
    multiple_object_tracking_postprocessing, visualize_tracking_result)
from hailo_model_zoo.core.postprocessing.landmarks_postprocessing import (
    face_landmarks_postprocessing, visualize_face_landmarks_result,
    visualize_hand_landmarks_result, hand_landmarks_postprocessing)
from hailo_model_zoo.core.postprocessing.face_landmarks_3d_postprocessing import (
    face_landmarks_3d_postprocessing, visualize_face_landmarks_3d_result)
from hailo_model_zoo.core.postprocessing.detection_3d_postprocessing import (
    detection_3d_postprocessing, visualize_3d_detection_result)
from hailo_model_zoo.core.postprocessing.fast_depth_postprocessing import (
    fast_depth_postprocessing, visualize_fast_depth_result)
from hailo_model_zoo.core.postprocessing.ocr_postprocessing import (
    ocr_postprocessing, visualize_ocr_result
)
from hailo_model_zoo.core.postprocessing.person_reid_postprocessing import (
    person_reid_postprocessing)
from hailo_model_zoo.core.postprocessing.face_attr_postprocessing import face_attr_postprocessing
from hailo_model_zoo.core.postprocessing.mspn_postprocessing import (
    mspn_postprocessing, visualize_single_person_pose_estimation_result
)
from hailo_model_zoo.core.postprocessing.stereonet_postprocessing import (
    stereonet_postprocessing, visualize_stereonet_result)
try:
    # THIS CODE IS EXPERIMENTAL AND IN USE ONLY FOR TAPPAS VALIDATION
    from hailo_model_zoo.core.postprocessing.tappas_postprocessing import tappas_postprocessing
except ModuleNotFoundError:
    tappas_postprocessing = None


def get_visualization(name, **kwargs):
    """ Returns visualization_fn(endnodes, image_info)
        Args:
            name: The name of the task.
        Returns:
            visualization_fn: A function that visualize the results.

        Raises:
            ValueError: If visualization `name` is not recognized.
    """
    unsupported_visualizations = {
        'face_verification',
        'person_reid',
    }

    if name in unsupported_visualizations:
        raise ValueError(f'Visualization is currently not supported for {name}')

    visualization_fn_map = {
        'classification': visualize_classification_result,
        'zero_shot_classification': visualize_classification_result,
        'segmentation': visualize_segmentation_result,
        'detection': visualize_detection_result,
        'pose_estimation': visualize_pose_estimation_result,
        'face_verification': visualize_face_result,
        'instance_segmentation': visualize_instance_segmentation_result,
        'super_resolution': visualize_super_resolution_result,
        'super_resolution_srgan': visualize_srgan_result,
        'low_light_enhancement': visualize_low_light_enhancement_result,
        'head_pose_estimation': visualize_head_pose_result,
        'age_gender': visualize_age_gender_result,
        'face_detection': visualize_detection_result,
        'mono_depth_estimation': visualize_mono_depth_result,
        'multiple_object_tracking': visualize_tracking_result,
        'face_landmark_detection': visualize_face_landmarks_result,
        'landmark_detection': visualize_hand_landmarks_result,
        'lane_detection': visualize_lane_detection_result,
        '3d_detection': visualize_3d_detection_result,
        'face_landmark_detection_3d': visualize_face_landmarks_3d_result,
        'fast_depth': visualize_fast_depth_result,
        'ocr': visualize_ocr_result,
        'single_person_pose_estimation': visualize_single_person_pose_estimation_result,
        'stereonet': visualize_stereonet_result
    }
    if name not in visualization_fn_map:
        raise ValueError('Visualization name [%s] was not recognized' % name)

    def visualization_fn(endnodes, image_info, **kwargs):
        return visualization_fn_map[name](endnodes, image_info, **kwargs)

    return visualization_fn


def get_postprocessing(name, flip=False):
    """ Returns postprocessing_fn(endnodes, **kwargs)
        Args:
            name: The name of the task.
        Returns:
            postprocessing_fn: A function that postprocess a batch.

        Raises:
            ValueError: If postprocessing `name` is not recognized.
    """
    postprocessing_fn_map = {
        'classification': classification_postprocessing,
        'zero_shot_classification': zero_shot_classification_postprocessing,
        'segmentation': segmentation_postprocessing,
        'detection': detection_postprocessing,
        'pose_estimation': pose_estimation_postprocessing,
        'mono_depth_estimation': mono_depth_estimation_postprocessing,
        'face_verification': facenet_postprocessing,
        'landmark_detection': hand_landmarks_postprocessing,
        'face_landmark_detection': face_landmarks_postprocessing,
        'instance_segmentation': instance_segmentation_postprocessing,
        'super_resolution': super_resolution_postprocessing,
        'super_resolution_srgan': super_resolution_postprocessing,
        'low_light_enhancement': low_light_enhancement_postprocessing,
        'head_pose_estimation': head_pose_estimation_postprocessing,
        'face_detection': face_detection_postprocessing,
        'age_gender': age_gender_postprocessing,
        'multiple_object_tracking': multiple_object_tracking_postprocessing,
        'lane_detection': lane_detection_postprocessing,
        '3d_detection': detection_3d_postprocessing,
        'face_landmark_detection_3d': face_landmarks_3d_postprocessing,
        'fast_depth': fast_depth_postprocessing,
        'ocr': ocr_postprocessing,
        'person_reid': person_reid_postprocessing,
        'person_attr': classification_postprocessing,
        'face_attr': face_attr_postprocessing,
        'single_person_pose_estimation': mspn_postprocessing,
        'stereonet': stereonet_postprocessing,
        'tappas_postprocessing': tappas_postprocessing
    }

    if name not in postprocessing_fn_map:
        raise ValueError('Postprocessing name [%s] was not recognized' % name)

    def postprocessing_fn(endnodes, device_pre_post_layers=None, **kwargs):
        return postprocessing_fn_map[name](endnodes, device_pre_post_layers, **kwargs)

    return postprocessing_fn
