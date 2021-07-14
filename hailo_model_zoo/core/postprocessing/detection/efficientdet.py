import tensorflow as tf
import numpy as np

from .ssd import collect_box_class_predictions
from tensorflow.image import combined_non_max_suppression


class EfficientDetPostProc(object):

    def __init__(self, img_dims, nms_iou_thresh, score_threshold, anchors,
                 classes, labels_offset, max_detections=100, **kwargs):
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_classes = classes
        self._max_detections = max_detections
        self._label_offset = labels_offset
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')
        self._anchors_input = tf.reshape(tf.py_func(self.anchors_for_shape,
                                                    [img_dims, anchors["aspect_ratios"],
                                                     anchors["scales"],
                                                     anchors["sizes"], anchors["strides"]],
                                                    ['float32'])[0], (1, -1, 4))

    def bbox_transform_inv(self, deltas):
        cxa = (self._anchors_input[..., 0] + self._anchors_input[..., 2]) / 2
        cya = (self._anchors_input[..., 1] + self._anchors_input[..., 3]) / 2
        wa = self._anchors_input[..., 2] - self._anchors_input[..., 0]
        ha = self._anchors_input[..., 3] - self._anchors_input[..., 1]
        ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
        w = tf.math.exp(tw) * wa
        h = tf.math.exp(th) * ha
        cy = ty * ha + cya
        cx = tx * wa + cxa
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    def clip_boxes(self, boxes):
        height, width = self._image_dims
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1) / width
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1) / height
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1) / width
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1) / height
        return tf.stack([y1, x1, y2, x2], axis=2)

    def postprocessing(self, endnodes, **kwargs):
        with tf.name_scope('Postprocessor'):
            regression, classification = collect_box_class_predictions(endnodes, self._num_classes)
            classification = tf.sigmoid(classification)
            boxes = self.bbox_transform_inv(regression)
            boxes = self.clip_boxes(boxes)
            (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = \
                combined_non_max_suppression(boxes=tf.expand_dims(boxes, axis=[2]),
                                             scores=classification,
                                             score_threshold=self._score_threshold,
                                             iou_threshold=self._nms_iou_thresh,
                                             max_output_size_per_class=self._max_detections,
                                             max_total_size=self._max_detections)
        nmsed_classes = tf.cast(tf.add(nmsed_classes, self._label_offset), tf.int16)
        return {'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': num_detections}

    def shift(self, feature_map_shape, stride, anchors):
        # create a grid starting from half stride from the top left corner
        shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = np.array(all_anchors.reshape((K * A, 4)), np.float32)
        return all_anchors

    def generate_anchors(self, base_size, aspect_ratio, scales):
        num_anchors = len(aspect_ratio) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(aspect_ratio))[None], (2, 1)).T
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.tile(aspect_ratio, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.tile(aspect_ratio, len(scales))
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def anchors_for_shape(self, image_dims, aspect_ratio, scales, sizes, strides):
        pyramid_levels = [3, 4, 5, 6, 7]
        feature_map_shapes = [(image_dims + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
        all_anchors = np.zeros((0, 4), dtype=np.float32)
        for idx, p in enumerate(pyramid_levels):
            anchors = self.generate_anchors(sizes[idx], aspect_ratio, scales)
            shifted_anchors = self.shift(feature_map_shapes[idx], strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        return all_anchors
