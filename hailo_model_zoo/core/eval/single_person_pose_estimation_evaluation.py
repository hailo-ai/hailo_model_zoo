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


class SinglePersonPoseEstimationEval(Eval):

    def __init__(self, **kwargs):
        self._metrics_vals = OrderedDict()
        # self._local_det_file = 'my_pose_detections.json'
        self._gt_labels_path = kwargs['gt_labels_path']
        self._coco_result = []
        self.thr = 0.2
        self.sigmas = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
                       0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
                       ]

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)

        keypoints, joint_scores = net_output[..., :2], net_output[..., 2]

        batch_size = keypoints.shape[0]
        for batch_index in range(batch_size):
            keypoint, keypoint_score = keypoints[batch_index], joint_scores[batch_index]
            ground_truth = {k: v[batch_index] for k, v in img_info.items()}

            detection_keypoints = np.concatenate([keypoint, keypoint_score[:, None]], axis=1)

            # calculate average (threshold passing) keypoint score
            valid_scores = keypoint_score[keypoint_score > self.thr]
            score = np.mean(valid_scores) if len(valid_scores) > 0 else 0

            detection = {
                "image_id": int(ground_truth['image_id']),
                "category_id": 1,  # Always person
                "keypoints": detection_keypoints.reshape(-1).tolist(),
                "score": score,
            }
            self._coco_result.append(detection)

    def evaluate(self):
        coco_gt = COCO(os.path.join(self._gt_labels_path, GT_LABELS_FILE))
        coco_dt = coco_gt.loadRes(self._coco_result)
        result = COCOeval(coco_gt, coco_dt, 'keypoints')
        result.params.useSegm = None
        result.params.kpt_oks_sigmas = np.array(self.sigmas)
        result.evaluate()
        result.accumulate()
        result.summarize()
        self._metrics_vals['{}_{}'.format('keypoints', METRIC_NAMES[0])] = result.stats[0]
        self._metrics_vals['{}_{}'.format('keypoints', METRIC_NAMES[1])] = result.stats[1]

    def _get_accuracy(self):
        return self._metrics_vals

    def reset(self):
        pass
