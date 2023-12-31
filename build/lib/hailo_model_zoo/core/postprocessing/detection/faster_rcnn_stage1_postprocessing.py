import tensorflow as tf
import numpy as np
import collections


def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes


def generate_rpn_proposals(boxes, scores, img_shape,
                           pre_nms_topk, post_nms_topk=None):
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    topk = tf.minimum(pre_nms_topk, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_valid_boxes = topk_boxes
    topk_valid_scores = topk_scores

    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes,
        topk_valid_scores,
        max_output_size=post_nms_topk,
        iou_threshold=0.7)

    proposal_boxes = tf.gather(topk_valid_boxes, nms_indices)
    proposal_scores = tf.gather(topk_valid_scores, nms_indices)
    return tf.stop_gradient(proposal_boxes, name='boxes'), tf.stop_gradient(proposal_scores, name='scores')


def get_all_anchors(stride, sizes, ratios, max_size):
    cell_anchors = _generate_anchors(stride, sizes, ratios)
    field_size = int(np.ceil(max_size / stride))
    shifts = (np.arange(0, field_size) * stride).astype("float32")
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype("float32")
    return field_of_anchors


def _generate_anchors(base_size, scales, aspect_ratios):
    anchor = np.array([1, 1, base_size, base_size], dtype=float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors


def _bbox_boundaries_to_whctrs(anchor):
    """ Trnasform (xmin, ymin, xmax, ymax) to (width, height, x_center, y_center) for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _bbox_boundaries_to_whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _bbox_boundaries_to_whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def decode_bbox_target(box_predictions, anchors):
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = np.log(1000 / 16.)
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


class RPNAnchors(collections.namedtuple('_RPNAnchors', ['boxes'])):
    def decode_logits(self, logits):
        return decode_bbox_target(logits, self.boxes)

    def narrow_to(self, height, width):
        shape2d = [height, width]
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)
        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        return RPNAnchors(boxes)


class FasterRCNNStage1(object):
    def __init__(self, img_dims, nms_iou_thresh,
                 score_threshold, anchors,
                 featuremap_shape=(38, 50),
                 pre_nms_topk=6000, post_nms_topk=1000,
                 **kwargs):
        self._image_dims = img_dims
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')
        self._stride = int(anchors['strides'][0])
        self._aspect_ratios = np.array(anchors['aspect_ratios'], dtype=float)
        self._sizes = np.array(anchors['sizes'], dtype=float) / self._stride
        self._pre_nms_topk = pre_nms_topk
        self._post_nms_topk = post_nms_topk
        self._feature_map_height = int(anchors['featuremap_shape'][0])
        self._feature_map_width = int(anchors['featuremap_shape'][1])
        self._anchors = self.get_full_anchor_map()

    def get_full_anchor_map(self):
        anchors_full_map = RPNAnchors(get_all_anchors(stride=self._stride,
                                                      sizes=self._sizes,
                                                      ratios=self._aspect_ratios,
                                                      max_size=2048))
        return anchors_full_map.narrow_to(self._feature_map_height, self._feature_map_width)

    def postprocessing(self, endnodes, **kwargs):
        featuremap = endnodes[0]
        scores = endnodes[1]
        boxes = endnodes[2]
        boxes = tf.reshape(boxes, [-1, self._feature_map_height * self._feature_map_width * 15, 4])
        decoded_roi_boxes = self._anchors.decode_logits(boxes)
        proposal_boxes, proposal_scores = generate_rpn_proposals(tf.reshape(decoded_roi_boxes, (-1, 4)),
                                                                 tf.reshape(scores, (-1,)), self._image_dims,
                                                                 self._pre_nms_topk, self._post_nms_topk)
        return [featuremap, proposal_boxes]
