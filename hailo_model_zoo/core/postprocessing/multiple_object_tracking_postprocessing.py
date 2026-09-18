import numpy as np
import tensorflow as tf
from detection_tools.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.postprocessing.detection.centernet import CenternetPostProc

# STARK corner-head geometry. The head emits two score maps (top-left and bottom-right
# corner) over a FEAT_SIZE x FEAT_SIZE grid of the SEARCH_SIZE search crop; the soft-argmax
# weights each cell by its pixel centre, hence the 0.5 offset.
SOT_FEAT_SIZE = 20
SOT_SEARCH_SIZE = 320
_SOT_STRIDE = SOT_SEARCH_SIZE / SOT_FEAT_SIZE
_SOT_CELL_CENTRES = (np.arange(SOT_FEAT_SIZE, dtype=np.float32) + 0.5) * _SOT_STRIDE
# Row-major flatten of the (H, W) score map: x is the column, y is the row.
_SOT_COORD_X = np.tile(_SOT_CELL_CENTRES, SOT_FEAT_SIZE)
_SOT_COORD_Y = np.repeat(_SOT_CELL_CENTRES, SOT_FEAT_SIZE)


@POSTPROCESS_FACTORY.register(name="single_object_tracking")
def single_object_tracking_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    """Corner-head soft-argmax for STARK, run on the host.

    The reference graph does this on chip, but the 400-wide softmax it needs cannot be
    mapped on Hailo-8 / Hailo-8L, so the network is cut after the corner-score convolution
    and the soft-argmax is reproduced here.

    Args:
        endnodes: Corner score maps, ``(B, FEAT, FEAT, 2)``; channel 0 is the top-left
            corner and channel 1 the bottom-right, matching the ONNX slice order.
        device_pre_post_layers: Unused, kept for the postprocessing-callback contract.
        **kwargs: Unused.

    Returns:
        ``{"pred_boxes": (B, 4)}`` as xyxy normalized to the search crop, i.e. the same
        contract the on-chip head produced.
    """
    scores = tf.reshape(tf.cast(endnodes, tf.float32), (-1, SOT_FEAT_SIZE * SOT_FEAT_SIZE, 2))
    probs = tf.nn.softmax(scores, axis=1)
    coord_x = tf.constant(_SOT_COORD_X)[None, :]
    coord_y = tf.constant(_SOT_COORD_Y)[None, :]
    corners = [tf.reduce_sum(coord * probs[:, :, corner], axis=1) for corner in (0, 1) for coord in (coord_x, coord_y)]
    return {"pred_boxes": tf.stack(corners, axis=-1) / SOT_SEARCH_SIZE}


@POSTPROCESS_FACTORY.register(name="multiple_object_tracking")
def multiple_object_tracking_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    kwargs["meta_arch"] = kwargs.get("meta_arch", {})
    if kwargs["meta_arch"] != "fair_mot":
        raise NotImplementedError("Tracking post-processing currently supports only FairMOT based architectures")

    centernet_postprocessing = CenternetPostProc(
        device_pre_post_layers=device_pre_post_layers, score_threshold=kwargs["score_threshold"]
    )
    detection_dict = centernet_postprocessing.postprocessing(
        [endnodes[2], endnodes[1], endnodes[3]], device_pre_post_layers=device_pre_post_layers, **kwargs
    )

    re_id_values = tf.nn.l2_normalize(endnodes[0], axis=-1)
    top_indices = tf.numpy_function(_get_top_indices, [re_id_values, detection_dict["top_k_indices"]], [tf.int64])
    detection_dict["re_id_values"] = tf.gather_nd(re_id_values, top_indices)
    return {**detection_dict}


def _get_top_indices(re_id_values, top_k):
    top_indices_including_all_features = []
    for ind in top_k:
        top_indices_including_all_features.append(np.stack([ind + [0, 0, 0, j] for j in range(128)]))
    top_indices_including_all_features = np.stack(top_indices_including_all_features)
    return top_indices_including_all_features.reshape((re_id_values.shape[0], -1, 128, top_k.shape[-1]))


@VISUALIZATION_FACTORY.register(name="multiple_object_tracking")
def visualize_tracking_result(
    logits,
    image,
    threshold=0.4,
    image_info=None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    dataset_name=None,
    **kwargs,
):
    boxes = logits["detection_boxes"][0]
    labels = {1: {"name": "person", "id": 1}}
    keypoints = None
    return visualize_boxes_and_labels_on_image_array(
        image[0],
        boxes,
        logits["detection_classes"][0],
        logits["detection_scores"][0],
        labels,
        instance_masks=logits.get("detection_masks"),
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=4,
        min_score_thresh=threshold,
        keypoints=keypoints,
    )
