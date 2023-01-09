import tensorflow as tf
import numpy as np
from tensorflow.image import combined_non_max_suppression
import math


def collect_box_class_predictions(output_branches, num_classes, type):
    # RESHAPING AND CONCAT RESULTS BEFORE RUNNING THROUGH POST PROCESSING STAGE:
    box_predictors_list = []
    class_predictors_list = []
    sorted_output_branches = output_branches
    for i, BoxTensor in enumerate(sorted_output_branches):
        num_of_batches, branch_h, branch_w, branch_features = BoxTensor.shape
        # Odd locations are the box predictors
        if i % 2 == 0:
            reshaped_tensor = tf.reshape(BoxTensor,
                                         shape=[-1,
                                                branch_h * branch_w * tf.cast(branch_features / 4, tf.int32),
                                                4])
            box_predictors_list.append(reshaped_tensor)
        # Even locations are the class preidctors
        else:
            reshaped_tensor = \
                tf.reshape(BoxTensor,
                           shape=[-1,
                                  branch_h * branch_w * tf.cast(branch_features / (num_classes), tf.int32),
                                  num_classes])
            class_predictors_list.append(reshaped_tensor)
    return box_predictors_list[::-1], class_predictors_list[::-1]


class AnchorGen(object):

    def __init__(self, anchors, image_dims):
        predefined_anchors_flag = anchors.get('predefined', None)
        if predefined_anchors_flag:
            pass
        else:
            self._type = anchors['type']
            self.scales = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            self.aspect_ratios = ((0.5, 1.0, 2.0),) * len(self.scales)
            self.orig_image_size = image_dims
            self.feature_maps_sizes = self.calc_feature_maps_size()
            self.scale_factors = [1.0, 1.0, 1.0, 1.0]
        self.cell_anchors = [self.generate_anchors(size, aspect_ratio)
                             for size, aspect_ratio in zip(self.scales, self.aspect_ratios)]
        self.anchors = self.get_anchors(self.orig_image_size, list(self.feature_maps_sizes))

    def calc_feature_maps_size(self):
        features_maps_size = []
        for ratio in [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128]:
            features_maps_size.append([math.ceil(self.orig_image_size[0] * ratio),
                                      math.ceil(self.orig_image_size[1] * ratio)])
        return features_maps_size

    def generate_anchors(self, scales, aspect_ratios, dtype=np.float32):
        scales = np.asarray(list(scales), dtype=dtype)
        aspect_ratios = np.asarray(list(aspect_ratios), dtype=dtype)
        h_ratios = np.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = np.reshape(w_ratios[:, None] * scales[None, :], (-1,))
        hs = np.reshape(h_ratios[:, None] * scales[None, :], (-1,))

        base_anchors = np.stack([-ws, -hs, ws, hs], axis=1) / 2
        return base_anchors.astype(int)

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        if not (len(grid_sizes) == len(strides) == len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride

            shifts_x = np.arange(0, grid_width, dtype=np.float32) * stride_width
            shifts_y = np.arange(0, grid_height, dtype=np.float32) * stride_height
            shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
            shift_x = np.reshape(shift_x, (-1,))
            shift_y = np.reshape(shift_y, (-1,))
            shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)
            anchors.append(
                np.reshape(np.reshape(shifts, (-1, 1, 4))
                           + np.reshape(base_anchors, (1, -1, 4)), (-1, 4)).astype(np.float32)
            )

        return anchors

    def get_anchors(self, image_size, grid_sizes):
        strides = [[np.asarray(image_size[0] // g[0], dtype=np.float32),
                    np.asarray(image_size[1] // g[1], dtype=np.float32)] for g in grid_sizes]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]

        return anchors_in_image


class retinanet_postproc(object):
    # The following params are corresponding to those used for training the model

    def __init__(self, img_dims=(800, 800), nms_iou_thresh=0.2,
                 score_threshold=0.01, anchors=None, classes=264, topk=1000,
                 **kwargs):

        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_classes = classes
        self.topk_candidates = topk
        self._anchors = AnchorGen(anchors, self._image_dims)
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')

    def _decode_boxes(self, rel_codes, anchors):
        xa_min, ya_min, xa_max, ya_max = tf.unstack(anchors, num=4, axis=-1)
        wa = xa_max - xa_min
        ha = ya_max - ya_min
        xcenter_a = xa_min + 0.5 * wa
        ycenter_a = ya_min + 0.5 * ha
        dx, dy, dw, dh = tf.unstack(rel_codes, num=4, axis=-1)
        if self._anchors.scale_factors:
            dx /= self._anchors.scale_factors[0]
            dy /= self._anchors.scale_factors[1]
            dw /= self._anchors.scale_factors[2]
            dh /= self._anchors.scale_factors[3]
        dw = tf.clip_by_value(dw, clip_value_min=tf.math.reduce_min(dw), clip_value_max=tf.math.log(1000. / 16))
        dh = tf.clip_by_value(dh, clip_value_min=tf.math.reduce_min(dh), clip_value_max=tf.math.log(1000. / 16))
        w = tf.exp(dw) * wa
        h = tf.exp(dh) * ha
        ycenter = dy * ha + ycenter_a
        xcenter = dx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def clip_boxes_to_image(self, boxes):
        return tf.clip_by_value(boxes, clip_value_min=[0, 0, 0, 0],
                                clip_value_max=[self._image_dims[0], self._image_dims[1],
                                                self._image_dims[0], self._image_dims[1]])

    def tf_select_best_pred(self, inputs):
        class_pred, boxes_pred, anchors = inputs
        detection_scores = tf.reshape(class_pred, (1, -1))
        top_k_idxs = tf.math.top_k(detection_scores[0, ...], k=self.topk_candidates)[1]
        anchor_idxs = tf.cast(tf.math.divide(top_k_idxs, self._num_classes), tf.int32)
        top_k_box = tf.gather(boxes_pred, indices=anchor_idxs, axis=-2)
        top_k_scores = tf.gather(class_pred, indices=anchor_idxs, axis=-2)
        top_k_anchors = tf.gather(anchors, indices=anchor_idxs, axis=-2)
        return top_k_scores, top_k_box, top_k_anchors

    def np_select_best_pred(self, class_pred, boxes_pred, anchors):
        batch_size = class_pred.shape[0]
        top_k_boxes = np.zeros((batch_size, self.topk_candidates, 4))
        top_k_scores = np.zeros((batch_size, self.topk_candidates, self._num_classes))
        top_k_anchors = np.zeros((batch_size, self.topk_candidates, 4))
        for k in range(batch_size):
            idxs = np.argpartition(class_pred[k, ...].flatten(), -self.topk_candidates)[-self.topk_candidates:]
            anchor_idxs = (idxs / self._num_classes).astype('int64')
            top_k_boxes[k, ...] = boxes_pred[k, anchor_idxs, ...]
            top_k_scores[k, ...] = class_pred[k, anchor_idxs, ...]
            top_k_anchors[k, ...] = anchors[anchor_idxs]
        return top_k_scores.astype('float32'), top_k_boxes.astype('float32'), top_k_anchors.astype('float32')

    def tf_postproc(self, endnodes):

        with tf.name_scope('Postprocessor'):
            # Collect all output branches into Boxes/classes objects
            box_predictions, classes_predictions = \
                collect_box_class_predictions(endnodes, self._num_classes, self._anchors._type)
            image_boxes = []
            image_scores = []
            for class_per_level, box_per_level, anchors_per_level in zip(classes_predictions,
                                                                         box_predictions, self._anchors.anchors):
                top_k_scores, top_k_box, top_k_anchors = tf.numpy_function(self.np_select_best_pred,
                                                                           [class_per_level, box_per_level,
                                                                               anchors_per_level],
                                                                           [tf.float32, tf.float32, tf.float32])

                boxes_per_level = self._decode_boxes(top_k_box, top_k_anchors)
                image_boxes.append(boxes_per_level)
                image_scores.append(top_k_scores)
            image_boxes = tf.expand_dims(tf.concat(image_boxes, axis=1), axis=-2) / self._image_dims[0]
            image_scores = tf.concat(image_scores, axis=1)
            (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = \
                combined_non_max_suppression(boxes=image_boxes,
                                             scores=image_scores,
                                             iou_threshold=self._nms_iou_thresh,
                                             score_threshold=self._score_threshold,
                                             max_output_size_per_class=5000,
                                             max_total_size=5000,
                                             clip_boxes=True)
            # adding offset to the class prediction and cast to integer
            nmsed_classes = tf.cast(nmsed_classes, tf.int16)
        return {'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': num_detections}

    def postprocessing(self, endnodes, **kwargs):
        return self.tf_postproc(endnodes)
