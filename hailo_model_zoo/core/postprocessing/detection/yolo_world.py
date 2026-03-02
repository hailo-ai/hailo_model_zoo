import numpy as np
import tensorflow as tf

from .centernet import COCO_2017_TO_2014_TRANSLATION


class YoloWorldPostProc:
    def __init__(
        self,
        img_dims=(640, 640),
        nms_iou_thresh=0.7,
        labels_offset=1,
        score_threshold=0.001,
        anchors=None,
        classes=80,
        nms_max_output_per_class=None,
        post_nms_topk=None,
        **kwargs,
    ):
        self._num_classes = classes
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._strides = anchors.strides
        self.reg_max = anchors.regression_length
        self._labels_offset = labels_offset
        self._offset_factors = anchors.scale_factors
        self._nms_max_output_per_class = 100 if nms_max_output_per_class is None else nms_max_output_per_class
        self._nms_max_output = 100 if post_nms_topk is None else post_nms_topk

    @staticmethod
    def split_decode(endnodes, num_classes):
        scores, boxes = [], []
        for node in endnodes[3:]:
            fm_size_h, fm_size_w = node.shape[1:3]
            box = tf.reshape(node, (-1, fm_size_h * fm_size_w, 4, 1))
            boxes.append(box)
        for node in endnodes[:3]:
            fm_size_h, fm_size_w = node.shape[1:3]
            score = tf.reshape(node, (-1, fm_size_h * fm_size_w, num_classes))
            score = score[:, :, :num_classes]
            scores.append(score)
        return tf.concat(scores, axis=1), boxes

    def _get_scores_boxes(self, endnodes):
        return self.split_decode(endnodes, self._num_classes)

    def _box_decoding(self, raw_boxes):
        boxes = None
        for box_distribute, stride in zip(raw_boxes, self._strides):
            # create grid
            shape = [int(x / stride) for x in self._image_dims]
            grid_x = np.arange(shape[1])
            grid_y = np.arange(shape[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            x_offset, y_offset = self._offset_factors
            ct_row = (grid_y.flatten() + x_offset) * stride
            ct_col = (grid_x.flatten() + y_offset) * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)
            # decode box
            box_distance = box_distribute * stride
            box_distance = tf.concat([box_distance[:, :, :2, 0] * (-1), box_distance[:, :, 2:, 0]], axis=-1)
            decode_box = np.expand_dims(center, axis=0) + box_distance
            # clipping
            xmin = tf.maximum(0.0, decode_box[:, :, 0]) / self._image_dims[1]
            ymin = tf.maximum(0.0, decode_box[:, :, 1]) / self._image_dims[0]
            xmax = tf.minimum(tf.cast(self._image_dims[1], tf.float32), decode_box[:, :, 2]) / self._image_dims[1]
            ymax = tf.minimum(tf.cast(self._image_dims[0], tf.float32), decode_box[:, :, 3]) / self._image_dims[0]
            decode_box = tf.transpose([ymin, xmin, ymax, xmax], [1, 2, 0])
            boxes = decode_box if boxes is None else tf.concat([boxes, decode_box], axis=1)
        return tf.expand_dims(boxes, axis=2)

    def postprocessing(self, endnodes, *, device_pre_post_layers, **kwargs):
        scores, raw_boxes = self._get_scores_boxes(endnodes)
        scores = tf.sigmoid(scores) if not device_pre_post_layers.sigmoid else scores
        # decode boxes
        boxes = self._box_decoding(raw_boxes)

        # nms
        (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=scores,
            score_threshold=self._score_threshold,
            iou_threshold=self._nms_iou_thresh,
            max_output_size_per_class=self._nms_max_output_per_class,
            max_total_size=self._nms_max_output,
        )

        # adding offset to the class prediction and cast to integer
        def translate_coco_2017_to_2014(nmsed_classes):
            return np.vectorize(COCO_2017_TO_2014_TRANSLATION.get)(nmsed_classes).astype(np.int32)

        nmsed_classes = tf.cast(tf.add(nmsed_classes, self._labels_offset), tf.int16)
        [nmsed_classes] = tf.numpy_function(translate_coco_2017_to_2014, [nmsed_classes], ["int32"])
        nmsed_classes.set_shape((1, kwargs["nms_max_output_per_class"]))
        return {
            "detection_boxes": nmsed_boxes,
            "detection_scores": nmsed_scores,
            "detection_classes": nmsed_classes,
            "num_detections": num_detections,
        }
