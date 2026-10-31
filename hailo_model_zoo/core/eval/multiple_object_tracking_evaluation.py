from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.tracking_evaluation_external.byte_tracker import BYTETracker
from hailo_model_zoo.core.eval.tracking_evaluation_external.mot_evaluator import Evaluator
from hailo_model_zoo.core.eval.tracking_evaluation_external.tracking_classes import JDETracker
from hailo_model_zoo.core.factory import EVAL_FACTORY

MIN_BOX_AREA = 200

DETECTION_SCORE_THRESHOLD = 0.4

MAX_DETECTIONS_PER_IMAGE = 100
TRACK_BUFFER_SIZE = 30
MEAN = [0.408, 0.447, 0.470]
STD = [0.289, 0.274, 0.278]
REID_DIMENSION = 128
BYTETRACK_MIN_BOX_AREA = 100


@EVAL_FACTORY.register(name="multiple_object_tracking")
class MultipleObjectTrackingEval(Eval):
    def __init__(self, **kwargs):
        self._bytetrack = kwargs.get("meta_arch") == "bytetrack"
        self._centered = kwargs.get("centered", True)
        self._video_trackers = {}
        self._results = {}
        self._evaluators = {}
        self._mota = 0
        self._idf1 = 0

    def update_op(self, net_output, img_info):
        for i in range(len(img_info["video_name"])):
            video_name = img_info["video_name"][i]
            if video_name not in self._video_trackers:
                self._video_trackers[video_name] = (
                    BYTETracker()
                    if self._bytetrack
                    else JDETracker(
                        MAX_DETECTIONS_PER_IMAGE, TRACK_BUFFER_SIZE, MEAN, STD, det_thresh=DETECTION_SCORE_THRESHOLD
                    )
                )

            dets = self._to_original_frame(net_output["detection_boxes"][i], img_info, i)
            scores = net_output["detection_scores"][i]
            if self._bytetrack:
                online_targets = self._video_trackers[video_name].update(dets[:, [1, 0, 3, 2]], scores)
                min_box_area = BYTETRACK_MIN_BOX_AREA
            else:
                dets = np.stack((dets[:, 1], dets[:, 0], dets[:, 3], dets[:, 2], scores), axis=1)
                id_features = net_output["re_id_values"][i].squeeze(0)
                remain_inds = dets[:, 4] > DETECTION_SCORE_THRESHOLD
                online_targets = self._video_trackers[video_name].update(dets[remain_inds], id_features[remain_inds])
                min_box_area = MIN_BOX_AREA
            online_tlwhs = []
            online_ids = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            frame_id = int(img_info["image_name"][i].decode().split(".")[0])
            if video_name not in self._results:
                self._results[video_name] = []
            self._results[video_name].append((frame_id, online_tlwhs, online_ids))

            gt_indices = img_info["is_ignore"][i] == 0
            gt_tlwhs = img_info["bbox"][i][gt_indices]

            gt_ids = img_info["person_id"][i][gt_indices]

            ignore_indices = img_info["is_ignore"][i] == 1
            ignore_tlwhs = img_info["bbox"][i][ignore_indices]

            if video_name not in self._evaluators:
                self._evaluators[video_name] = Evaluator()
            self._evaluators[video_name].eval_frame(gt_tlwhs, gt_ids, ignore_tlwhs, online_tlwhs, online_ids)

    def _to_original_frame(self, boxes, img_info, i):
        """Normalized letterboxed ymin/xmin/ymax/xmax -> original-image pixels."""
        boxes = np.array(boxes, dtype=np.float64)
        for axis, size, pad, original in (
            ((1, 3), "width", "horizontal_pad", "original_width"),
            ((0, 2), "height", "vertical_pad", "original_height"),
        ):
            offset = img_info[pad][i] / 2 if self._centered else 0
            scale = img_info[original][i] / (img_info[size][i] - img_info[pad][i])
            boxes[:, axis] = (boxes[:, axis] * img_info[size][i] - offset) * scale
        return boxes

    def evaluate(self):
        summary = Evaluator.get_summary(
            [self._evaluators[name].acc for name in self._evaluators], list(self._evaluators)
        )
        self._mota = summary["mota"][-1]
        self._idf1 = summary["idf1"][-1]

    def _get_accuracy(self):
        return OrderedDict([("MOTA", self._mota), ("IDF1", self._idf1)])

    def reset(self):
        self._video_trackers = {}
