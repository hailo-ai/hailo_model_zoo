from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.core.postprocessing.ocr_postprocessing import CHARS, greedy_decoder


@EVAL_FACTORY.register(name="ocr")
class OCREval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['Accuracy']
        self._metrics_vals = [0, 0]
        self._labels_offset = kwargs.get('labels_offset', 0)
        self.reset()

    def _greedy_decode(self, output):
        return greedy_decoder(output)

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        gt_plates = list(p.decode('utf-8') for p in img_info['plate'])
        net_output = self._parse_net_output(net_output)  # (B x 19)
        pred_labels = self._greedy_decode(net_output)
        plates = list(''.join([CHARS[int(x)] for x in label]) for label in pred_labels)
        acc = list((np.asarray(x) == np.asarray(y)).all() for (x, y) in zip(gt_plates, plates))
        self.accuracy += acc

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self.accuracy)

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self.accuracy = []
