from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="person_attr")
@EVAL_FACTORY.register(name="face_attr")
class PersonAttrEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ["Accuracy", "Top1"]
        self._metrics_vals = [0, 0]
        self.num_attributes = kwargs["classes"]
        self.reset()

    def reset(self):
        self.gt_pos = np.zeros((self.num_attributes,))
        self.gt_neg = np.zeros((self.num_attributes,))
        self.pt_pos = np.zeros((self.num_attributes,))
        self.pt_neg = np.zeros((self.num_attributes,))
        self.top1 = []

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        label_index = img_info["attributes"]
        preds = np.array(net_output >= 0, np.int64)
        gt = label_index[:, : self.num_attributes]
        self.gt_pos += np.sum(gt == 1, axis=0)
        self.gt_neg += np.sum(gt == 0, axis=0)
        self.pt_pos += np.sum((gt == 1).astype(float) * (preds == 1).astype(float), axis=0)
        self.pt_neg += np.sum((gt == 0).astype(float) * (preds == 0).astype(float), axis=0)
        self.top1 += list(preds == gt)

    def evaluate(self):
        label_pos_acc = self.pt_pos / self.gt_pos
        label_neg_acc = self.pt_neg / self.gt_neg
        self._metrics_vals[0] = np.mean((label_pos_acc + label_neg_acc) / 2)
        self._metrics_vals[1] = np.mean(self.top1)

    def _get_accuracy(self):
        return OrderedDict(
            [(self._metric_names[0], self._metrics_vals[0]), (self._metric_names[1], self._metrics_vals[1])]
        )
