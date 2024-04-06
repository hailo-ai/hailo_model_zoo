from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY

# Calculation is based on: https://torchmetrics.readthedocs.io/en/stable/image/peak_signal_noise_ratio.html


@EVAL_FACTORY.register(name="low_light_enhancement")
class LowLightEnhancementEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['PSNR']
        self._metrics_vals = [0]
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self.sum_squared_error += self._aggregate(net_output, gt_labels)
        self.n_obs += net_output.size

    def is_percentage(self):
        return False

    def evaluate(self):
        _data_range = self.target_max - self.target_min
        _psnr = 10.0 * np.log10(np.square(_data_range) / (self.sum_squared_error / self.n_obs))
        self._metrics_vals[0] = _psnr

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self._psnr = []
        self.sum_squared_error = 0
        self.n_obs = 0
        self.target_min = 1.0
        self.target_max = 0.0

    def _aggregate(self, net_output, gt_labels):
        _sum_squared_error = np.sum(np.square(net_output - gt_labels['enhanced_img_processed']))
        target_min, target_max = gt_labels['enhanced_img_processed'].min(), gt_labels['enhanced_img_processed'].max()
        self.target_min = min(self.target_min, target_min)
        self.target_max = max(self.target_max, target_max)
        return _sum_squared_error
