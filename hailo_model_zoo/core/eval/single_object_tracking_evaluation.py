from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY

SUCCESS_THRESHOLDS = 21


def _iou(a, b):
    """IoU between two (x, y, w, h) boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    inter_h = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@EVAL_FACTORY.register(name="single_object_tracking")
class SingleObjectTrackingEval(Eval):
    """Success-plot AUC for single-object tracking, the standard LaSOT/OTB metric.

    ``tracker_infer`` runs the sequential tracking loop and hands over one predicted box
    per frame; this class only scores. AUC is the mean over IoU thresholds in [0, 1] of
    the fraction of frames exceeding each threshold, which is equivalent to the area
    under the success plot.
    """

    def __init__(self, **kwargs):
        self._ious = []
        self._per_video = {}
        self._metrics = {}
        self._skipped = 0

    def update_op(self, net_output, img_info):
        pred_boxes = np.asarray(net_output["pred_boxes"]).reshape(-1, 4)
        gt_boxes = np.asarray(img_info["gt_box"]).reshape(-1, 4)
        video_names = img_info["video_name"]
        visible = img_info.get("target_visible")
        visible = np.ones(len(gt_boxes), dtype=np.int32) if visible is None else np.asarray(visible).reshape(-1)

        for pred, gt, video, is_visible in zip(pred_boxes, gt_boxes, video_names, visible, strict=True):
            # Match the reference protocol (calc_seq_err_robust in pytracking, which
            # Stark's own evaluation uses): frames with a degenerate annotation or an
            # absent target are EXCLUDED, not scored as IoU 0.
            #     valid = ((anno_bb > 0.0).sum(1) == 4) & target_visible
            if not (np.all(gt > 0.0) and bool(is_visible)):
                self._skipped += 1
                continue
            name = video.decode() if isinstance(video, bytes) else str(video)
            value = _iou(pred, gt)
            self._ious.append(value)
            self._per_video.setdefault(name, []).append(value)

    def evaluate(self):
        if not self._ious:
            self._metrics = {"success_auc": 0.0, "mean_iou": 0.0}
            return
        ious = np.asarray(self._ious, dtype=np.float64)
        taus = np.linspace(0.0, 1.0, SUCCESS_THRESHOLDS)
        # Per-video AUC first, then averaged, so long sequences do not dominate.
        per_video_auc = [float(np.mean([(np.asarray(v) > t).mean() for t in taus])) for v in self._per_video.values()]
        self._metrics = {
            "success_auc": float(np.mean(per_video_auc)) if per_video_auc else 0.0,
            "mean_iou": float(ious.mean()),
        }

    def _get_accuracy(self):
        # The base class wraps this into EvalResult objects; do not override get_accuracy.
        return OrderedDict(self._metrics)

    def reset(self):
        self._ious = []
        self._per_video = {}
        self._metrics = {}
        self._skipped = 0
