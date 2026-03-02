from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="l1_evaluation")
class L1Evaluation(Eval):
    def __init__(self, **kwargs):
        self._metrics_vals = [0]
        self.base_tsh = 1e-4
        self._metric_names = [f"L1({self.base_tsh} scale)"]
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self.l1_error += np.sum(np.abs(net_output - gt_labels["gt"]))
        self.n_obs += net_output.size

    def is_percentage(self):
        return False

    def is_bigger_better(self):
        return False

    def evaluate(self):
        # We use a scale of base_tsh for the L1 error because MZ evaluation outputs 3 digits after the decimal point.
        self._metrics_vals[0] = self.l1_error / (self.n_obs * self.base_tsh) if self.n_obs > 0 else 0

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self._psnr = []
        self.l1_error = 0
        self.n_obs = 0
