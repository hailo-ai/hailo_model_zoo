import os
import numpy as np
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hailo_model_zoo.core.eval.eval_base_class import Eval

GT_LABELS_FILE = 'person_keypoints_val2017.json'

METRIC_NAMES = ['Average-Precision-IoU-0.50:0.95',
                'Average-Precision-IoU-0.50',
                'Average-Precision-IoU-0.75',
                'Average-Precision-IoU-0.50:0.95-medium',
                'Average-Precision-0.50:0.95-large',
                'Average-Recall-IoU-0.50:0.95',
                'Average-Recall-IoU-0.50',
                'Average-Recall-IoU-0.75',
                'Average-Recall-IoU-0.50:0.95-medium',
                'Average-Recall-0.50:0.95-large'
                ]


class PoseEstimationEval(Eval):

    def __init__(self, **kwargs):
        self._metrics_vals = OrderedDict()
        self._local_det_file = 'my_pose_detections.json'
        self._gt_labels_path = kwargs['gt_labels_path']
        self._coco_result = []

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        if "center" not in img_info:
            # OpenPose Evaluation
            net_output = self._parse_net_output(net_output)
            self._coco_result += net_output
            return

        # CenterPose evaluation
        bboxes, scores, keypoints, joint_scores = (
            net_output['bboxes'], net_output['scores'], net_output['keypoints'], net_output['joint_scores'])

        batch_size = bboxes.shape[0]
        for batch_index in range(batch_size):
            box, score, keypoint, keypoint_score = (
                bboxes[batch_index], scores[batch_index], keypoints[batch_index], joint_scores[batch_index])
            ground_truth = {k: v[batch_index] for k, v in img_info.items()}

            # change boxes to coco format
            box[:, 2] -= box[:, 0]
            box[:, 3] -= box[:, 1]
            for b, s, kps, kps_score in zip(box, score, keypoint, keypoint_score):
                detection_keypoints = np.concatenate([kps.reshape(17, 2), kps_score.reshape(17, 1)], axis=1)

                detection = {
                    "image_id": int(ground_truth['image_id']),
                    "category_id": 1,  # Always person
                    "bbox": b.tolist(),
                    "score": s[0],
                    "keypoints": detection_keypoints.reshape(-1).tolist(),
                }

                self._coco_result.append(detection)

            # undo modifications
            box[:, 2] += box[:, 0]
            box[:, 3] += box[:, 1]

    def evaluate(self):
        coco_gt = COCO(os.path.join(self._gt_labels_path, GT_LABELS_FILE))
        coco_dt = coco_gt.loadRes(self._coco_result)
        for annotation_type in ['keypoints', 'bbox']:
            if annotation_type not in self._coco_result[0]:
                continue
            result = COCOeval(coco_gt, coco_dt, annotation_type)
            result.evaluate()
            result.accumulate()
            result.summarize()
            self._metrics_vals['{}_{}'.format(annotation_type, METRIC_NAMES[0])] = result.stats[0]
            self._metrics_vals['{}_{}'.format(annotation_type, METRIC_NAMES[1])] = result.stats[1]

    def _get_accuracy(self):
        return self._metrics_vals

    def reset(self):
        pass
