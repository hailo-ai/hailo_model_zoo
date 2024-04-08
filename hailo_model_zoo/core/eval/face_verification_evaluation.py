import math
import re
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import KFold

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


def _accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    return float(tp + tn) / dist.size


def _distance(embeddings1, embeddings2):
    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = dot / norm
    dist = np.arccos(similarity) / math.pi
    return dist


@EVAL_FACTORY.register(name="face_verification")
class FaceVerificationEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['acc']
        self._tf_path = kwargs.get('tf_path', False)
        self._metrics_vals = [0]
        self._folds_num = 10
        self.reset()

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def _parse_gt_data(self, gt_data):
        # facenet_infer tf1 legacy support
        if isinstance(gt_data, tuple):
            return gt_data
        return gt_data['image_name'], gt_data['is_same']

    def update_op(self, logits_batch, gt_data):
        logits_batch = self._parse_net_output(logits_batch)
        image_name, is_same = self._parse_gt_data(gt_data)
        idx = 0
        for logits, name, same in zip(logits_batch, image_name, is_same):
            if idx % 2 == 0:
                # Create pairs gt from tf-record or by filename
                if self._tf_path:
                    self._actual_issame.append(same)
                else:
                    self._actual_issame.append(re.sub(b"[0-9]", b"_", name)
                                               == re.sub(b"[0-9]", b"_", image_name[idx + 1]))
                self._embeddings1.append(logits)
            else:
                self._embeddings2.append(logits)
            idx += 1

    def evaluate(self):
        thresholds = np.arange(0, 4, 0.01)
        k_fold = KFold(n_splits=self._folds_num, shuffle=False)
        accuracy = np.zeros((self._folds_num))
        nrof_pairs = min(len(self._actual_issame), len(self._embeddings1))
        indices = np.arange(nrof_pairs)
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            emb_mean = np.mean(np.concatenate([np.array(self._embeddings1)[train_set.astype(
                int)], np.array(self._embeddings2)[train_set.astype(int)]]), axis=0)
            dist = _distance(np.array(self._embeddings1) - emb_mean, np.array(self._embeddings2) - emb_mean)
            acc_train = np.zeros((len(thresholds)))
            for threshold_idx, threshold in enumerate(thresholds):
                acc_train[threshold_idx] = _accuracy(threshold, np.array(
                    dist)[train_set.astype(int)], np.array(self._actual_issame)[train_set.astype(int)])
            best_threshold_index = np.argmax(acc_train)
            accuracy[fold_idx] = _accuracy(thresholds[best_threshold_index], np.array(
                dist)[test_set.astype(int)], np.array(self._actual_issame)[test_set.astype(int)])
        self._metrics_vals[0] = np.mean(accuracy)

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])

    def reset(self):
        self._embeddings1 = []
        self._embeddings2 = []
        self._actual_issame = []
