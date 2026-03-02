import string
from collections import OrderedDict

import Levenshtein
import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="text_recognition")
class TextRecognitionEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ["WordAcc", "CharAcc"]
        self._metrics_vals = [0, 0]
        self.reset()
        self.is_filter = kwargs.get("is_filter", True)
        self.filter_list = list(string.digits + string.ascii_letters)

    def _normalize_text(self, text):
        text = "".join(filter(lambda x: x in self.filter_list, text))
        return text.lower()

    def _normalized_distance(self, a, b):
        raw_dist = Levenshtein.distance(a, b)
        return raw_dist / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 0.0

    def update_op(self, net_output, img_info):
        pred_text = net_output["text"]
        gt_text = img_info["text_tag"]

        for pred, gt in zip(pred_text, gt_text):
            if isinstance(pred, bytes):
                pred = pred.decode("utf-8")
            if isinstance(gt, bytes):
                gt = gt.decode("utf-8")

            if self.is_filter:
                pred = self._normalize_text(pred)
                gt = self._normalize_text(gt)

            self.word_acc += [pred == gt]
            self.char_acc += [self._normalized_distance(pred, gt)]

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self.word_acc)
        self._metrics_vals[1] = 1 - np.mean(self.char_acc)

    def _get_accuracy(self):
        return OrderedDict(
            [(self._metric_names[0], self._metrics_vals[0]), (self._metric_names[1], self._metrics_vals[1])]
        )

    def reset(self):
        self.word_acc = []
        self.char_acc = []
