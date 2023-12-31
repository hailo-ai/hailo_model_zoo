from collections import OrderedDict
import numpy as np
from hailo_model_zoo.core.eval.eval_base_class import Eval


class StereoNetEval(Eval):
    def __init__(self, **kwargs):
        self.TotalEPE = 0
        self.count = 0
        self.avgEPE = 0
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        for i in range(net_output.shape[0]):  # update per inference in batch
            pred_disp = net_output[i]
            true_disp = img_info['gt_l'][i]
            diff = np.zeros(true_disp.shape)
            index = np.argwhere(true_disp > 0)  # ignore non-relevant indices
            diff[index[:, 0], index[:, 1], index[:, 2]] = np.abs(
                true_disp[index[:, 0], index[:, 1], index[:, 2]] - pred_disp[index[:, 0], index[:, 1], index[:, 2]])
            correct = (diff[index[:, 0], index[:, 1], index[:, 2]] < 3) | (
                diff[index[:, 0], index[:, 1], index[:, 2]] < true_disp[index[:, 0], index[:, 1], index[:, 2]] * 0.05)
            # pixel prediction is true if diff is smaller than 3 or 5%
            three_pixel_correct_rate = float(np.sum(correct)) / float(len(index[:, 0]))
            self.TotalEPE += three_pixel_correct_rate

        self.count += net_output.shape[0]

    def evaluate(self):
        self.avgEPE = self.TotalEPE / self.count

    def _get_accuracy(self):
        return OrderedDict([('EPE', self.avgEPE)])

    def reset(self):
        self.TotalEPE = 0
        self.count = 0
