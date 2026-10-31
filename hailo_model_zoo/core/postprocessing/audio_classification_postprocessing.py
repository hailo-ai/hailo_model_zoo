"""Audio-classification postprocessor: shape-normalise raw network outputs.

YamNet (and audio classifiers like it) produces a `(B, 1, 1, num_classes)`
tensor — the trailing `(1, 1)` are spurious spatial dims left over from the
Gemm being wrapped in an HW-friendly NHWC layout. We squeeze them and pass
the raw logits through under the `predictions` key the eval module reads.

Sigmoid is intentionally NOT applied here:

* AudioSet is multi-label — per-class binary scoring, not softmax-over-classes.
* AP / mAP are rank-based metrics, so applying sigmoid before scoring would
  shift values toward 0.5 monotonically but not change the ordering. Doing
  the sigmoid inside the eval module (`audio_classification_evaluation`)
  keeps the contract simple — postprocess returns logits, eval owns the
  probability interpretation.

If a future deployment wants on-device sigmoid, set
`postprocessing.device_pre_post_layers.sigmoid: true` in the yaml; in that
case the network already emits probabilities and we just pass them through.
"""

import numpy as np
import tensorflow as tf

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY


class PostprocessingException(Exception):
    pass


def _validate_logits_shape(shape, classes):
    """Allow (B, C) or (B, 1, 1, C) where C == classes."""
    if len(shape) == 2 and shape[-1] == classes:
        return True
    if len(shape) == 4 and shape[1] == 1 and shape[2] == 1 and shape[-1] == classes:
        return True
    return False


@POSTPROCESS_FACTORY.register(name="audio_classification")
def audio_classification_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    """Squeeze (B, 1, 1, C) -> (B, C) and return as the canonical predictions dict.

    Args:
        endnodes: tf.Tensor or np.ndarray, shape (B, 1, 1, C) or (B, C).
        device_pre_post_layers: dict from yaml; not consulted here (the eval
                                module owns sigmoid because AP/mAP are
                                rank-based and a device-side sigmoid would
                                not change the metric).
        **kwargs: `classes` must match the network's output dim; mismatch
                  surfaces here instead of producing garbage downstream.

    Returns:
        {"predictions": tensor of shape (B, classes)}.
    """
    classes = kwargs.get("classes")
    logits = endnodes
    if classes is not None and not _validate_logits_shape(logits.shape, classes):
        raise PostprocessingException(
            f"Unexpected logits shape {tuple(logits.shape)} for classes={classes}; "
            "expected (B, classes) or (B, 1, 1, classes)."
        )

    if isinstance(logits, np.ndarray):
        if logits.ndim == 4:
            logits = logits.reshape(logits.shape[0], -1)
    else:  # tf.Tensor
        if logits.shape.ndims == 4:
            logits = tf.reshape(logits, [-1, logits.shape[-1]])
    return {"predictions": logits}
