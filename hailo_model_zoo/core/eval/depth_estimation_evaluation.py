import numpy as np
from collections import OrderedDict
import cv2

from hailo_model_zoo.core.eval.eval_base_class import Eval


class DepthEstimationEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['abs_rel']
        self._metrics_vals = [0]
        self.reset()
        self._min_depth = 1e-3
        self._max_depth = 80

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        for label, pred in zip(img_info['depth'], net_output):
            gt_height, gt_width = label.shape[:2]
            pred = cv2.resize(pred, (gt_width, gt_height))
            mask = np.logical_and(label > self._min_depth, label < self._max_depth)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
            label = label[mask]
            pred = pred[mask]
            pred[pred < self._min_depth] = self._min_depth
            pred[pred > self._max_depth] = self._max_depth
            self.err.append(np.mean(np.abs(label - pred) / label))

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self.err)

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self.err = []
