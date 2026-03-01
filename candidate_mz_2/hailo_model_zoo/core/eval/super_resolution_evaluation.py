from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="super_resolution")
class SuperResolutionEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ["psnr"]
        self._metrics_vals = [0]
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self._psnr += self.evaluate_psnr(net_output, gt_labels["hr_img"], gt_labels["height"], gt_labels["width"])

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self._psnr) / 100.0

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self._psnr = []

    def evaluate_psnr(self, y_pred, y_true, height, width):
        psnr_list = []
        for image_num in range(y_pred.shape[0]):
            y_pred_single_im = y_pred[image_num]
            y_true_single_im = y_true[image_num]
            psnr = 10.0 * np.log10(1.0 / (np.mean(np.square(y_pred_single_im - y_true_single_im))))
            psnr_list.append(psnr)
        return psnr_list
