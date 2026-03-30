import math

import numpy as np
import tensorflow as tf


def covariance_matrix(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate covariance matrix from oriented bounding boxes.
    Args:
        boxes (np.ndarray): A (N, 5) tensor of rotated bounding boxes with xywhr format.

    Returns:
        (np.ndarray): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = np.concatenate((boxes[:, 2:4] ** 2 / 12, boxes[:, 4:]), axis=-1)
    ax1, ax2, angle = np.split(gbbs, 3, axis=-1)
    cos = np.cos(angle)
    sin = np.sin(angle)
    cos2 = cos**2
    sin2 = sin**2
    return ax1 * cos2 + ax2 * sin2, ax1 * sin2 + ax2 * cos2, (ax1 - ax2) * cos * sin


def batch_probiou(obb1: np.ndarray, obb2: np.ndarray) -> np.ndarray:
    """Calculate probabilistic IoU between oriented bounding boxes (https://arxiv.org/pdf/2106.06072v1.pdf).

    Args:
        obb1 (np.ndarray): A (N, 5) tensor of ground truth obbs, with xywhr format.
        obb2 (np.ndarray): A (M, 5) tensor of predicted obbs, with xywhr format.

    Returns:
        (np.ndarray): A tensor of shape (N, M) representing obb similarities.
    """
    eps = 1e-7
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (np.squeeze(x, axis=-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = covariance_matrix(obb1)
    a2, b2, c2 = (np.squeeze(x, axis=-1)[None] for x in covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)) * 0.5
    t3 = ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) / (
        4 * np.sqrt(np.maximum(0, a1 * b1 - c1**2) * np.maximum(0, a2 * b2 - c2**2)) + eps
    ) + eps
    t3 = np.log(t3) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def fast_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Fast-NMS implementation from https://arxiv.org/pdf/1904.02689 using upper triangular matrix operations.

    Args:
        boxes (np.ndarray): Bounding boxes with shape (N, 5) in xywhr format.
        scores (np.ndarray): Confidence scores with shape (N,).
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        (np.ndarray): Indices of boxes to keep after NMS.
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)  # upper triangular matrix (excluding diagonal)
    pick = np.where((ious >= iou_threshold).sum(0) <= 0)[0]
    return sorted_idx[pick]


class YoloOBBPostProc:
    def __init__(
        self,
        img_dims=(1024, 1024),
        nms_iou_thresh=0.7,
        labels_offset=0,
        score_threshold=0.25,
        anchors=None,
        classes=15,
        nms_max_output_per_class=None,
        post_nms_topk=None,
        meta_arch="yobb",
        **kwargs,
    ):
        self._num_classes = classes
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._strides = anchors.strides
        self.reg_max = anchors.regression_length
        self._labels_offset = labels_offset
        # since scale factors don't make sense in nanodet we abuse it to store the offsets
        self._offset_factors = anchors.scale_factors
        self._network_arch = meta_arch
        self._nms_max_output_per_class = 100 if nms_max_output_per_class is None else nms_max_output_per_class
        self._nms_max_output = 100 if post_nms_topk is None else post_nms_topk

    def _nms_numpy(self, preds, return_idxs=False, max_nms=30000, max_wh=7680):
        bs = preds.shape[0]
        nc = self._num_classes
        extra = preds.shape[2] - nc - 4  # number of extra info (single angle)
        ai = 4 + nc  # angle index
        # score filtering
        xc = np.max(preds[:, :, 4:ai], axis=2) > self._score_threshold
        xinds = np.arange(preds.shape[1])[None, :].repeat(bs, axis=0)[..., None]

        output = [np.zeros((0, 6 + extra), dtype=np.float32)] * bs
        keepi = [np.zeros((0, 1), dtype=np.int32)] * bs

        for xi, (x, xk) in enumerate(zip(preds, xinds, strict=True)):
            filt = xc[xi]  # confidence
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

            # If none remain process next image
            if not x.shape[0]:
                continue

            box, cls, angle = np.split(x, [4, 4 + nc], axis=1)  #  (xywh, conf, cls)

            # best class only
            j = np.argmax(cls, axis=1, keepdims=True)
            conf = np.max(cls, axis=1, keepdims=True)
            filt = conf.flatten() > self._score_threshold
            x = np.concatenate((box, conf, j.astype(float), angle), axis=1)[filt]
            if return_idxs:
                xk = xk[filt]

            num_boxes = x.shape[0]
            if not num_boxes:
                continue
            if num_boxes > max_nms:
                # sort by confidence and remove excess boxes
                filt = np.argsort(x[:, 4])[::-1][:max_nms]
                x = x[filt]
                if return_idxs:
                    xk = xk[filt]

            # classes and scores
            class_offsets = x[:, 5:6] * max_wh
            scores = x[:, 4]

            boxes = np.concatenate((x[:, :2] + class_offsets, x[:, 2:4], x[:, -1:]), axis=-1)  # xywhr
            i = fast_nms(boxes, scores, self._nms_iou_thresh)
            i = i[: self._nms_max_output]
            output[xi] = x[i]
            if return_idxs:
                keepi[xi] = xk[i].reshape(-1)

        outputs = [out.astype(np.float32) for out in output]
        boxes = np.zeros((bs, self._nms_max_output, 4), dtype=np.float32)
        scores = np.zeros((bs, self._nms_max_output), dtype=np.float32)
        classes = np.zeros((bs, self._nms_max_output), dtype=np.int32)
        angles = np.zeros((bs, self._nms_max_output), dtype=np.float32)
        num_detections = np.zeros((bs,), dtype=np.int32)
        for i in range(bs):
            out = outputs[i]
            if out.shape[0] == 0:
                continue
            boxes_i = out[:, :4]
            scores_i = out[:, 4]
            classes_i = out[:, 5].astype(np.int32)
            angles_i = out[:, 6]
            num_detections[i] = out.shape[0]
            boxes[i, : len(boxes_i), :] = boxes_i
            scores[i, : len(scores_i)] = scores_i
            classes[i, : len(classes_i)] = classes_i
            angles[i, : len(angles_i)] = angles_i

        return boxes, scores, classes, angles, num_detections

    @staticmethod
    def _split_decode_yolo(endnodes, reg_max, num_classes):
        # YOLOv8 score and boxes split
        scores, boxes = [], []
        for node in endnodes[::2]:
            fm_size_h, fm_size_w = node.shape[1:3]
            box = tf.reshape(node, (-1, fm_size_h * fm_size_w, 4, (reg_max + 1)))
            boxes.append(box)
        for node in endnodes[1::2]:
            fm_size_h, fm_size_w = node.shape[1:3]
            score = tf.reshape(node, (-1, fm_size_h * fm_size_w, num_classes))
            score = score[:, :, :num_classes]
            scores.append(score)

        return tf.concat(scores, axis=1), boxes

    def _box_decoding(self, raw_boxes, angles):
        boxes = None
        for box_distribute, angle, stride in zip(raw_boxes, angles, self._strides, strict=True):
            # create grid
            shape = [int(x / stride) for x in self._image_dims]
            grid_x = np.arange(shape[1])
            grid_y = np.arange(shape[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            x_offset, y_offset = self._offset_factors
            ct_row = (grid_y.flatten() + x_offset) * stride
            ct_col = (grid_x.flatten() + y_offset) * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # box distribution to distance
            reg_range = np.arange(self.reg_max + 1)
            box_distance = tf.nn.softmax(box_distribute, axis=-1)
            box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = tf.reduce_sum(box_distance, axis=-1)

            box_distance = box_distance * stride

            # decode box
            lt, rb = box_distance[:, :, :2], box_distance[:, :, 2:]
            xf, yf = tf.split((rb - lt) / 2, 2, axis=-1)  # Half-width and half-height
            cos_angle, sin_angle = tf.cos(angle), tf.sin(angle)  # Rotation components
            x, y = xf * cos_angle - yf * sin_angle, xf * sin_angle + yf * cos_angle
            xy = tf.concat([x, y], axis=-1) + center[:, :2]
            decoded_box = tf.concat(
                [xy, lt + rb], axis=-1
            )  # [rotated center x, rotated center y, total width, total height]
            boxes = decoded_box if boxes is None else tf.concat([boxes, decoded_box], axis=1)
        return boxes

    def postprocessing(self, endnodes, *, device_pre_post_layers, **kwargs):
        angle_endnodes = [
            tf.reshape(endnode, (-1, endnode.shape[1] * endnode.shape[2], endnode.shape[3])) for endnode in endnodes[6:]
        ]
        angles = []
        for angle_node in angle_endnodes:
            angles.append((tf.sigmoid(angle_node) - 0.25) * math.pi)

        # scores, raw_boxes = self._nanodet_postproc._get_scores_boxes(endnodes[:6])
        scores, raw_boxes = self._split_decode_yolo(endnodes[:6], self.reg_max, self._num_classes)
        scores = tf.sigmoid(scores) if not device_pre_post_layers.sigmoid else scores
        boxes = self._box_decoding(raw_boxes, angles)
        angles = tf.concat(angles, axis=1)

        preds = tf.concat([boxes, scores, angles], axis=2)
        # obb-nms
        [boxes, scores, classes, angles, num_detections] = tf.numpy_function(
            self._nms_numpy,
            [preds],
            [tf.float32, tf.float32, tf.int32, tf.float32, tf.int32],
        )
        # xywh -> xyxy
        boxes = tf.concat([boxes[:, :, :2] - boxes[:, :, 2:] / 2, boxes[:, :, :2] + boxes[:, :, 2:] / 2], axis=-1)
        # clip to image dims
        xmin = tf.maximum(0.0, boxes[:, :, 0]) / self._image_dims[1]
        ymin = tf.maximum(0.0, boxes[:, :, 1]) / self._image_dims[0]
        xmax = tf.minimum(tf.cast(self._image_dims[1], tf.float32), boxes[:, :, 2]) / self._image_dims[1]
        ymax = tf.minimum(tf.cast(self._image_dims[0], tf.float32), boxes[:, :, 3]) / self._image_dims[0]
        boxes = tf.transpose([ymin, xmin, ymax, xmax], [1, 2, 0])
        boxes = tf.concat([boxes, angles[..., tf.newaxis]], axis=-1)

        return {
            "detection_boxes": boxes,
            "detection_scores": scores,
            "detection_classes": classes,
            "num_detections": num_detections,
        }
