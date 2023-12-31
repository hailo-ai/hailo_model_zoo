import numpy as np
import cv2

from hailo_model_zoo.core.postprocessing.lane_detection.polylanenet import PolyLaneNetPostProcessHailo
from hailo_model_zoo.core.postprocessing.lane_detection.laneaf import LaneAFPostProc


LANE_DETECTION_ARCHS = {
    "polylanenet": PolyLaneNetPostProcessHailo,
    "laneaf": LaneAFPostProc
}


def visualize_lane_detection_result(pred, im, dataset_name='tusimple', **kwargs):
    pred = pred['predictions']
    color = [0, 255, 0]
    conf = pred[0, :, -1]   # last element in each lane is confidence
    pred = pred[:, :, :-1]  # removing conf from pred for ease of expression
    pred = pred[0].astype(int)
    im = im[0]
    overlay = im.copy()
    ypoints = np.arange(im.shape[0])
    for i in range(pred.shape[0]):  # going over lanes in image:
        if not np.any(pred[i, :] > 0):
            # all the x values of this lane are non-positive
            continue
        lane_confidence = conf[i]
        xpoints = pred[i, :]
        lane_horizon_y = ypoints[xpoints > 0][0]
        points = [[ypoints[j], xpoints[j]] for j in range(len(xpoints))]
        for current_point in points:
            if current_point[1] > 0 and current_point[1] < im.shape[1]:
                overlay = cv2.circle(overlay,
                                     (int(current_point[1]), int(current_point[0])),
                                     radius=2, color=color, thickness=2)
        cv2.putText(overlay,
                    str('{:.1f}'.format(lane_confidence)),
                    (int(xpoints[ypoints == lane_horizon_y]), int(lane_horizon_y)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0))
    w = 0.6
    im = ((1. - w) * im + w * overlay).astype(np.uint8)
    return im


def _get_postprocessing_class(meta_arch):
    for k in LANE_DETECTION_ARCHS:
        if k in meta_arch:
            return LANE_DETECTION_ARCHS[k]
    raise ValueError("Meta-architecture [{}] is not supported".format(meta_arch))


def lane_detection_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs["meta_arch"].lower()
    kwargs["anchors"] = {} if kwargs["anchors"] is None else kwargs["anchors"]
    kwargs["device_pre_post_layers"] = device_pre_post_layers
    postproc = _get_postprocessing_class(meta_arch)(**kwargs)
    return postproc.postprocessing(endnodes, **kwargs)
