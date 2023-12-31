import numpy as np
from collections import OrderedDict

from hailo_model_zoo.core.eval.eval_base_class import Eval


class FaceLandmarkEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['mse']
        self._metrics_vals = [0]
        self.reset()

    def reset(self):
        self._err = []

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        for b in range(len(gt_labels['landmarks'])):
            gt = gt_labels['landmarks'][b]
            res = net_output[b, :]
            self._err += [np.mean(np.abs(gt - res)) / len(res)]

    def evaluate(self):
        self._metrics_vals = [np.mean(self._err)]

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])


class FaceLandmark3DEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['nme[0,90]', 'nme[0,30]', 'nme[30,60]', 'nme[60,90]']
        self._metrics_vals = [0.0, 0.0, 0.0, 0.0]
        self.reset()

    def reset(self):
        self.nme_list = [[], [], []]

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        gt_landmarks = gt_labels['landmarks']
        minx = np.min(gt_landmarks[:, :, 0], axis=1)
        maxx = np.max(gt_landmarks[:, :, 0], axis=1)
        miny = np.min(gt_landmarks[:, :, 1], axis=1)
        maxy = np.max(gt_landmarks[:, :, 1], axis=1)
        box_length = np.sqrt((maxx - minx) * (maxy - miny))
        # Drop the depth dimension since we normalize by the box
        diff = net_output[:, :, :2] - gt_landmarks[:, :, :2]
        mean_error = np.mean(np.linalg.norm(diff, axis=2), axis=1)
        normalized_mean_error = mean_error / box_length

        yaw_abs = np.squeeze(np.abs(gt_labels['yaw']))
        small_angles_ind = yaw_abs <= 30
        if small_angles_ind.any():
            self.nme_list[0].append(normalized_mean_error[small_angles_ind])

        medium_angles_ind = np.bitwise_and(yaw_abs > 30, yaw_abs <= 60)
        if medium_angles_ind.any():
            self.nme_list[1].append(normalized_mean_error[medium_angles_ind])

        wide_angles_ind = yaw_abs > 60
        if wide_angles_ind.any():
            self.nme_list[2].append(normalized_mean_error[wide_angles_ind])

    def evaluate(self):
        self._metrics_vals = [np.mean(np.concatenate(nme_list)) for nme_list in self.nme_list]
        self._metrics_vals.insert(0, np.mean(self._metrics_vals))

    def _get_accuracy(self):
        return OrderedDict(zip(self._metric_names, self._metrics_vals))

    def is_bigger_better(self):
        return False
