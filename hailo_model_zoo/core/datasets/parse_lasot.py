import tensorflow as tf

from hailo_model_zoo.core.factory import DATASET_FACTORY

TEMPLATE_SIZE = 128
SEARCH_SIZE = 320


@DATASET_FACTORY.register(name="lasot")
def parse_lasot_record(serialized_example):
    """Parse a whole-frame LaSOT record for tracking evaluation.

    Frames are returned at their native resolution: the search region has to be cropped
    from the tracker's previous prediction, so it cannot be prepared ahead of inference.
    ``tracker_infer`` does the cropping.
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "video_name": tf.io.FixedLenFeature([], tf.string),
            "frame_index": tf.io.FixedLenFeature([], tf.int64),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image_jpeg": tf.io.FixedLenFeature([], tf.string),
            "gt_box": tf.io.FixedLenFeature([4], tf.float32),
            # 0 when the target is fully occluded or out of view; such frames are
            # excluded from the success plot rather than scored as IoU 0.
            "target_visible": tf.io.FixedLenFeature([], tf.int64, default_value=1),
        },
    )
    height = tf.cast(features["height"], tf.int32)
    width = tf.cast(features["width"], tf.int32)
    image = tf.image.decode_jpeg(features["image_jpeg"], channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.cast(tf.reshape(image, tf.stack([height, width, 3])), tf.uint8)
    image_info = {
        "image_name": tf.cast(features["image_name"], tf.string),
        "video_name": tf.cast(features["video_name"], tf.string),
        "frame_index": tf.cast(features["frame_index"], tf.int32),
        "height": height,
        "width": width,
        "gt_box": features["gt_box"],
        "target_visible": tf.cast(features["target_visible"], tf.int32),
    }
    return [image, image_info]


@DATASET_FACTORY.register(name="lasot_calib")
def parse_lasot_calib_record(serialized_example):
    """Parse a pre-cropped template/search pair for calibration.

    The pair is already at network resolution, so it bypasses cropping entirely and is
    handed straight to the two input layers.
    """
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "template_png": tf.io.FixedLenFeature([], tf.string),
            "search_png": tf.io.FixedLenFeature([], tf.string),
        },
    )
    template = tf.image.decode_png(features["template_png"], channels=3)
    search = tf.image.decode_png(features["search_png"], channels=3)
    template = tf.cast(tf.reshape(template, [TEMPLATE_SIZE, TEMPLATE_SIZE, 3]), tf.uint8)
    search = tf.cast(tf.reshape(search, [SEARCH_SIZE, SEARCH_SIZE, 3]), tf.uint8)
    # The template rides in image_info so the single-tensor preprocessing contract holds;
    # the preprocessing pairs them back up into the two-input dict.
    return [search, {"template": template}]
