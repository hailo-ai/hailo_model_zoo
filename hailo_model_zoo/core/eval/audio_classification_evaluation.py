"""Multi-label audio classification evaluation (mean Average Precision).

AudioSet is a multi-label task — every 10-second clip carries 1–7 positive
labels from a 521-class ontology subset (YamNet's outputs). The canonical
metric is mean Average Precision across the 521 classes, with predictions
aggregated to clip-level before scoring (a 10 s clip yields ~19 patches at
0.48 s hop; the model's per-patch logits are mean-pooled per clip).

This module:

1. Accumulates per-patch sigmoid scores and clip-level multi-hot labels.
2. Aggregates predictions per `clip_id` (mean over the clip's patches).
3. Reports mAP (macro-average of per-class AP) plus per-class AP if requested.

References:
- Google's YamNet README ("Clip-level scores are obtained by averaging the
  framewise scores over the patches of the clip").
- AudioSet paper §3.3 — uses mAP as the headline metric.
"""

from collections import OrderedDict, defaultdict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


def _sigmoid(x):
    # Stable sigmoid (avoid overflow on large negative logits)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _average_precision(y_true, y_score):
    """Per-class binary Average Precision.

    Equivalent to sklearn.metrics.average_precision_score but implemented
    inline so the model zoo doesn't gain a sklearn dependency for one
    function. AP = sum over thresholds of (recall_n - recall_{n-1}) * precision_n.
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")  # undefined; the macro-mean skips NaNs

    # Sort descending by score
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]

    # Cumulative TP / (TP + FP)
    tp_cum = np.cumsum(y_true_sorted).astype(np.float64)
    fp_cum = np.cumsum(1 - y_true_sorted).astype(np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / n_pos

    # Stepwise AP: dr * p (recall_n - recall_{n-1}) * precision_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


@EVAL_FACTORY.register(name="audio_classification")
class AudioClassificationEval(Eval):
    """Multi-label mAP over 521 classes, predictions aggregated per clip."""

    def __init__(self, **kwargs):
        self._num_classes = int(kwargs.get("classes", 521))
        self._metric_names = ["mAP"]
        self._metrics_vals = [0.0]
        self.reset()

    def reset(self):
        # clip_id -> {sum: (C,) float64, count: int}
        self._clip_sum = defaultdict(lambda: np.zeros(self._num_classes, dtype=np.float64))
        self._clip_count = defaultdict(int)
        # clip_id -> (C,) int8 multi-hot
        self._clip_labels = {}

    def _parse_net_output(self, net_output):
        x = net_output["predictions"]
        # Same shape-squeeze pattern as ClassificationEval (SDK-32906 quirk)
        if hasattr(x, "shape") and len(x.shape) == 4:
            x = x.squeeze((1, 2))
        return x

    def update_op(self, net_output, img_info):
        """Accumulate per-patch logits → sigmoid → per-clip running mean.

        net_output["predictions"]:  (B, 521) logits
        img_info["image_name"]:     (B,) bytes — clip_id
        img_info["label_multi_hot"]: (B, 521) int8
        """
        logits = np.asarray(self._parse_net_output(net_output))
        probs = _sigmoid(logits.astype(np.float64))
        clip_ids = img_info["image_name"]
        labels = np.asarray(img_info["label_multi_hot"])

        for i in range(probs.shape[0]):
            cid = clip_ids[i]
            # TFRecord clip_ids come back as numpy.bytes_; the framework may
            # also hand them in as str. Normalise to str.
            if isinstance(cid, (bytes, np.bytes_)):
                cid = cid.decode("utf-8")
            self._clip_sum[cid] += probs[i]
            self._clip_count[cid] += 1
            # Labels are clip-level — same vector for every patch of the clip.
            # We overwrite each time (cheap; consistent by construction).
            self._clip_labels[cid] = labels[i].astype(np.int8)

    def evaluate(self):
        if not self._clip_labels:
            self._metrics_vals[0] = float("nan")
            return

        clip_ids = sorted(self._clip_labels.keys())
        y_true = np.stack([self._clip_labels[c] for c in clip_ids])  # (N, C)
        y_score = np.stack([self._clip_sum[c] / max(self._clip_count[c], 1) for c in clip_ids])  # (N, C)

        aps = np.array([_average_precision(y_true[:, k], y_score[:, k]) for k in range(self._num_classes)])
        # Macro mAP — classes with zero positives produce NaN AP and are
        # excluded from the mean (canonical AudioSet practice).
        self._metrics_vals[0] = float(np.nanmean(aps))

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0])])
