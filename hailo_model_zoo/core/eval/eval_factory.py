"""Contains a factory for network evaluation."""
from hailo_model_zoo.core.eval.age_gender_evaluation import AgeGenderEval
from hailo_model_zoo.core.eval.classification_evaluation import ClassificationEval
from hailo_model_zoo.core.eval.face_detection_evaluation import FaceDetectionEval
from hailo_model_zoo.core.eval.segmentation_evaluation import SegmentationEval
from hailo_model_zoo.core.eval.detection_evaluation import DetectionEval
from hailo_model_zoo.core.eval.face_verification_evaluation import FaceVerificationEval
from hailo_model_zoo.core.eval.pose_estimation_evaluation import PoseEstimationEval
from hailo_model_zoo.core.eval.instance_segmentation_evaluation import InstanceSegmentationEval
from hailo_model_zoo.core.eval.depth_estimation_evaluation import DepthEstimationEval
from hailo_model_zoo.core.eval.super_resolution_evaluation import SuperResolutionEval
from hailo_model_zoo.core.eval.srgan_evaluation import SRGANEval
from hailo_model_zoo.core.eval.head_pose_estimation_evaluation import HeadPoseEstimationEval
from hailo_model_zoo.core.eval.multiple_object_tracking_evaluation import MultipleObjectTrackingEval
from hailo_model_zoo.core.eval.lane_detection_evaluation import LaneDetectionEval
from hailo_model_zoo.core.eval.face_landmark_evaluation import FaceLandmarkEval, FaceLandmark3DEval
from hailo_model_zoo.core.eval.detection_3d_evaluation import Detection3DEval
from hailo_model_zoo.core.eval.faster_rcnn_evaluation import FasterRCNNEval
from hailo_model_zoo.core.eval.fast_depth_evaluation import FastDepthEval
from hailo_model_zoo.core.eval.ocr_evaluation import OCREval
from hailo_model_zoo.core.eval.person_reid_evaluation import PersonReidEval
from hailo_model_zoo.core.eval.person_attr_evaluation import PersonAttrEval
from hailo_model_zoo.core.eval.single_person_pose_estimation_evaluation import SinglePersonPoseEstimationEval


class EmptyEval():
    def __init__(self, **kwargs):
        pass

    def get_accuracy(self, **kwargs):
        pass


def get_evaluation(name):
    """Returns evaluation object
    Args:
        name: The name of the task
    Returns:
        evaluation: An object that evaluate the network.

    Raises:
        ValueError: If task `name` is not recognized.
    """
    evaluation_map = {
        'classification': ClassificationEval,
        'segmentation': SegmentationEval,
        'detection': DetectionEval,
        'pose_estimation': PoseEstimationEval,
        'face_verification': FaceVerificationEval,
        'instance_segmentation': InstanceSegmentationEval,
        'mono_depth_estimation': DepthEstimationEval,
        'srgan': SRGANEval,
        'landmark_detection': EmptyEval,
        'face_landmark_detection': FaceLandmarkEval,
        'face_landmark_detection_3d': FaceLandmark3DEval,
        'head_pose_estimation': HeadPoseEstimationEval,
        'face_detection': FaceDetectionEval,
        'age_gender': AgeGenderEval,
        'multiple_object_tracking': MultipleObjectTrackingEval,
        'lane_detection': LaneDetectionEval,
        '3d_detection': Detection3DEval,
        'empty': EmptyEval,
        'faster_rcnn_stage2': FasterRCNNEval,
        'fast_depth': FastDepthEval,
        'ocr': OCREval,
        'person_reid': PersonReidEval,
        'person_attr': PersonAttrEval,
        'face_attr': PersonAttrEval,
        'single_person_pose_estimation': SinglePersonPoseEstimationEval,
        'super_resolution': SuperResolutionEval,
    }

    if name not in evaluation_map:
        raise ValueError('Task name [{}] was not recognized'.format(name))

    return evaluation_map[name]
