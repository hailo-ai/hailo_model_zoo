from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


def confusion_matrix(y_true, y_pred, N):
    y = np.array(N * y_true + y_pred, np.int32)
    y = np.bincount(y.ravel())
    if len(y) < N * N:
        y = np.concatenate((y, np.zeros((N * N - len(y),))))
    y = y.reshape(N, N)
    return y[:N - 1, :N - 1]


@EVAL_FACTORY.register(name="segmentation")
class SegmentationEval(Eval):
    def __init__(self, *, labels_map=None, **kwargs):
        self._labels_offset = kwargs['labels_offset']
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        self._labels_map = np.array(labels_map, dtype=np.uint8) if labels_map else None
        if self._channels_remove:
            self._filtered_classes = list(np.where(np.array(self._channels_remove['mask'][0]) == 0)[0])
            self._classes = kwargs.get('classes', 2) - len(self._filtered_classes)
        else:
            self._classes = kwargs.get('classes', 2)
        self._metric_names = ['mIoU']
        self._metrics_vals = [0]
        self.reset()

    def _filter_classes(self, mask):
        if self._channels_remove:
            for cl in range(self._classes + len(self._filtered_classes)):
                if cl in self._filtered_classes:
                    mask[mask == cl] = 255
                elif (cl > np.array(self._filtered_classes)).any():
                    mask[mask == cl] = cl - sum(cl > np.array(self._filtered_classes))

        if self._labels_map is not None:
            idx = np.where(mask != 255)
            mask[idx] = self._labels_map[mask.astype(np.uint8)[idx]]
        return np.clip(mask, 0, self._classes) - self._labels_offset

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        for b in range(net_output.shape[0]):
            self._overall_confusion_matrix += confusion_matrix(
                y_true=self._filter_classes(img_info['mask'][b]),
                y_pred=net_output[b], N=self._classes + 1)

    def evaluate(self):
        intersection = np.diag(self._overall_confusion_matrix)
        ground_truth_set = self._overall_confusion_matrix.sum(axis=1)
        predicted_set = self._overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        union = np.where(np.greater(union, 0.), union, np.ones_like(union))
        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)
        self._metrics_vals[0] = mean_intersection_over_union

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self._overall_confusion_matrix = np.zeros((self._classes, self._classes))
