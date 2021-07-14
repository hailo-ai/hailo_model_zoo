import numpy as np
from collections import OrderedDict

from hailo_model_zoo.core.eval.eval_base_class import Eval


class ClassificationEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['top1', 'top5']
        self._metrics_vals = [0, 0]
        self._labels_offset = kwargs.get('labels_offset', 0)
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        label_index = img_info['label_index']
        self.top1 += list(np.argmax(net_output, 1) == (label_index + self._labels_offset))
        top5 = np.argpartition(net_output, -5, axis=1)[:, -5:]
        self.top5 += [label_index[batch_idx] + self._labels_offset in top5[batch_idx, :]
                      for batch_idx in range(len(label_index))]

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self.top1)
        self._metrics_vals[1] = np.mean(self.top5)

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1])])

    def reset(self):
        self.top1 = []
        self.top5 = []
