from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="head_pose_estimation")
class HeadPoseEstimationEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['mae', 'mae_yaw', 'mae_pitch', 'mae_roll']
        self._metrics_vals = [0, 0, 0, 0]
        self._normalize_results = kwargs.get('normalize_results', True)
        self.reset()

    def update_op(self, net_output, img_info):
        _pitch, _yaw, _roll = img_info['angles'][:, 0], img_info['angles'][:, 1], img_info['angles'][:, 2]
        _pitch_predicted, _roll_predicted, _yaw_predicted = (net_output['pitch'], net_output['roll'], net_output['yaw'])
        self.yaw_err += list(np.abs(_yaw_predicted - _yaw))
        self.pitch_err += list(np.abs(_pitch_predicted - _pitch))
        self.roll_err += list(np.abs(_roll_predicted - _roll))

    def evaluate(self):
        normalize_factor = 100.0 if self._normalize_results else 1.0
        self._metrics_vals[1] = np.mean(self.yaw_err) / normalize_factor
        self._metrics_vals[2] = np.mean(self.pitch_err) / normalize_factor
        self._metrics_vals[3] = np.mean(self.roll_err) / normalize_factor
        self._metrics_vals[0] = np.mean(self._metrics_vals[1:3])

    def _get_accuracy(self):
        return OrderedDict([(x, y) for x, y in zip(self._metric_names, self._metrics_vals)])

    def reset(self):
        self.yaw_err = []
        self.pitch_err = []
        self.roll_err = []
