import json
import os

from detection_tools.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from hailo_model_zoo.core.postprocessing.detection.ssd import SSDPostProc
from hailo_model_zoo.core.postprocessing.detection.centernet import CenternetPostProc
from hailo_model_zoo.core.postprocessing.detection.yolo import YoloPostProc
from hailo_model_zoo.core.postprocessing.detection.efficientdet import EfficientDetPostProc
from hailo_model_zoo.core.postprocessing.detection.faster_rcnn_stage1_postprocessing import FasterRCNNStage1
from hailo_model_zoo.core.postprocessing.detection.faster_rcnn_stage2_postprocessing import FasterRCNNStage2
from hailo_model_zoo.core.postprocessing.detection.nanodet import NanoDetPostProc


DETECTION_ARCHS = {
    "ssd": SSDPostProc,
    "yolo": YoloPostProc,
    "centernet": CenternetPostProc,
    "efficientdet": EfficientDetPostProc,
    "faster_rcnn_stage1": FasterRCNNStage1,
    "faster_rcnn_stage2": FasterRCNNStage2,
    "nanodet": NanoDetPostProc,
}


def _get_postprocessing_class(meta_arch):
    for k in DETECTION_ARCHS:
        if k in meta_arch:
            return DETECTION_ARCHS[k]
    raise ValueError("Meta-architecture [{}] is not supported".format(meta_arch))


def detection_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs["meta_arch"].lower()
    kwargs["anchors"] = {} if kwargs["anchors"] is None else kwargs["anchors"]
    kwargs["device_pre_post_layers"] = device_pre_post_layers
    postproc = _get_postprocessing_class(meta_arch)(**kwargs)
    return postproc.postprocessing(endnodes, **kwargs)


def _get_coco_labels():
    coco_names = json.load(open(os.path.join(os.path.dirname(__file__), 'coco_names.json')))
    coco_names = {int(k): {'id': int(k), 'name': str(v)} for (k, v) in coco_names.items()}
    return coco_names


def _get_visdrone_labels():
    visdrone_names = json.load(open(os.path.join(os.path.dirname(__file__), 'visdrone_names.json')))
    visdrone_names = {int(k): {'id': int(k), 'name': str(v)} for (k, v) in visdrone_names.items()}
    return visdrone_names


def _get_face_detection_visualization_data(logits):
    boxes = logits['detection_boxes'][0]

    face_landmarks = logits.get('face_landmarks')
    if face_landmarks is not None:
        face_landmarks = face_landmarks[0].reshape((-1, 5, 2))[:, :, (1, 0)]
    boxes = boxes[:, (1, 0, 3, 2)]
    # No name to prevent clobbering the visualization
    labels = {1: {'id': 1, 'name': ''}}
    return boxes, labels, face_landmarks


def visualize_detection_result(logits, image, threshold=0.2, image_info=None, use_normalized_coordinates=True,
                               max_boxes_to_draw=20, dataset_name='coco', **kwargs):
    boxes = logits['detection_boxes'][0]
    keypoints = None
    if 'coco' in dataset_name:
        labels = _get_coco_labels()
    elif 'visdrone' in dataset_name:
        labels = _get_visdrone_labels()
    elif 'mot' in dataset_name:
        labels = {1: {'name': 'person', 'id': 1}},
    elif 'widerface' in dataset_name:
        boxes, labels, keypoints = _get_face_detection_visualization_data(logits)
    elif 'vehicle_detection' in dataset_name:
        labels = {0: {'name': 'vehicle'}}
    elif 'license_plates' in dataset_name:
        labels = {0: {'name': 'plate'}}
    return visualize_boxes_and_labels_on_image_array(image[0],
                                                     boxes,
                                                     logits['detection_classes'][0],
                                                     logits['detection_scores'][0],
                                                     labels,
                                                     instance_masks=logits.get('detection_masks'),
                                                     use_normalized_coordinates=use_normalized_coordinates,
                                                     max_boxes_to_draw=max_boxes_to_draw,
                                                     line_thickness=4,
                                                     min_score_thresh=threshold,
                                                     keypoints=keypoints)
