"""Audio-classification preprocessor (YamNet-style log-mel patches).

The expensive audio → STFT → mel → log → patch pipeline lives in
`datasets/create_audioset_tfrecord.py` and runs once at dataset-build time.
At inference / calibration / eval time we just pass the precomputed
(PATCH_FRAMES, MEL_BANDS, 1) log-mel patch through largely unchanged.

This module exists for three reasons:

1. Plug into the Model Zoo's PREPROCESS_FACTORY so `network_type:
   audio_classification` in yaml resolves to *something*.
2. Apply the standard `image_info`-passthrough contract the dataset
   feeder expects (`preprocessing_factory.get_preprocessing` wraps us).
3. Ensure shape is fixed to (height, width, 1) for graph-mode batching.

If a future audio model needs runtime feature extraction (e.g. compiled
HEF takes raw waveform), implement the STFT pipeline here using
`tf.signal.stft` — the canonical parameters live at the top of
`datasets/create_audioset_tfrecord.py`.
"""

import tensorflow as tf

from hailo_model_zoo.core.factory import PREPROCESS_FACTORY


@PREPROCESS_FACTORY.register(name="audio_classification")
def audio_classification_preprocess(image, image_info, height, width, **kwargs):
    """Pass-through for precomputed log-mel patches.

    Args:
        image:      (height, width, C) float32 log-mel patch from
                    `parse_audioset.parse_record`. Named `image` because the
                    factory wrapper hardcodes that kwarg.
        image_info: dict carrying `image_name` (clip_id) and `label_multi_hot`
                    — needed by the audio_classification eval to aggregate
                    per-clip predictions.
        height/width: from the parsed network's input shape (96 / 64 for YamNet).
        **kwargs:   `channels` is propagated from the network's input shape
                    (1 for YamNet — single mel-spectrogram patch).

    Returns:
        (image, image_info) with the static shape enforced.
    """
    channels = int(kwargs.get("channels", 1))
    image = tf.cast(image, tf.float32)
    image.set_shape([height, width, channels])
    return image, image_info
