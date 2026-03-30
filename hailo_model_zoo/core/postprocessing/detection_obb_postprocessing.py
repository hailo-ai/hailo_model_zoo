import json
import os

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.postprocessing.detection.visualization_utils import (
    visualize_boxes_and_labels_on_image_array,
)
from hailo_model_zoo.core.postprocessing.detection.yolo_obb import YoloOBBPostProc

DETECTION_OBB_ARCHS = {
    "yobb": YoloOBBPostProc,
}


def _get_postprocessing_class(meta_arch):
    for k in DETECTION_OBB_ARCHS:
        if k in meta_arch:
            return DETECTION_OBB_ARCHS[k]
    raise ValueError("Meta-architecture [{}] is not supported for detection_obb".format(meta_arch))


@POSTPROCESS_FACTORY.register(name="detection_obb")
def detection_obb_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs["meta_arch"].lower()
    kwargs["anchors"] = {} if kwargs["anchors"] is None else kwargs["anchors"]
    kwargs["device_pre_post_layers"] = device_pre_post_layers
    postproc = _get_postprocessing_class(meta_arch)(**kwargs)
    return postproc.postprocessing(endnodes, **kwargs)


def _get_dotav1_labels():
    with open(os.path.join(os.path.dirname(__file__), "dotav1_names.json")) as f:
        dotav1_names = json.load(f)
    return {int(k): {"id": int(k), "name": str(v)} for (k, v) in dotav1_names.items()}


def _get_obb_labels(dataset_name):
    if "dotav1" in dataset_name:
        return _get_dotav1_labels()
    raise Exception("No OBB labels for dataset {}".format(dataset_name))


@VISUALIZATION_FACTORY.register(name="detection_obb")
def visualize_detection_obb_result(
    logits,
    image,
    threshold=0.2,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    dataset_name="dotav1",
    **kwargs,
):
    boxes = logits["detection_boxes"][0]
    labels = _get_obb_labels(dataset_name)
    return visualize_boxes_and_labels_on_image_array(
        image[0],
        boxes,
        logits["detection_classes"][0],
        logits["detection_scores"][0],
        labels,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=4,
        min_score_thresh=threshold,
    )
