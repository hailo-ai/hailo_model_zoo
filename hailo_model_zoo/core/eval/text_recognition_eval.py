import string
from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


def levenshtein_distance_optimized(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein distance between two strings
    using O(min(len(s1), len(s2))) space.
    """
    # Ensure s2 is the shorter string to minimize memory
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    m, n = len(s1), len(s2)

    # Only two rows needed
    previous = list(range(n + 1))
    current = [0] * (n + 1)

    for i in range(1, m + 1):
        current[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            current[j] = min(
                previous[j] + 1,  # Deletion
                current[j - 1] + 1,  # Insertion
                previous[j - 1] + cost,  # Substitution
            )
        previous, current = current, previous  # swap

    return previous[n]


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
        raw_dist = levenshtein_distance_optimized(a, b)
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
