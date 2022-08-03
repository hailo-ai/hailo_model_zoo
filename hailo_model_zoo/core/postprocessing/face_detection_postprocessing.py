from itertools import product

import numpy as np
import tensorflow as tf
from detection_tools.core.post_processing import batch_multiclass_non_max_suppression


class FaceDetectionPostProc(object):
    # The following params are corresponding to those used for training the model
    LABEL_OFFSET = 1
    NUM_CLASSES = 1
    SCALE_FACTORS = (10., 5.)

    def __init__(self, image_dims=(300, 300), nms_iou_thresh=0.6, score_threshold=0.3, anchors=None):
        self._image_dims = image_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_branches = len(anchors['steps'])
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')
        self._anchors = self.extract_anchors(anchors['min_sizes'], anchors['steps'])

    def collect_box_class_predictions(self, output_branches):
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []
        sorted_output_branches = output_branches
        num_branches = self._num_branches
        assert len(sorted_output_branches) % num_branches == 0, "All branches must have the same number of output nodes"
        num_output_nodes_per_branch = len(sorted_output_branches) // num_branches
        for branch_index in range(0, len(sorted_output_branches), num_output_nodes_per_branch):
            num_of_batches, _, _, _ = tf.unstack(tf.shape(sorted_output_branches[branch_index]))
            box_predictors_list.append(tf.reshape(sorted_output_branches[branch_index], shape=[num_of_batches, -1, 4]))
            class_predictors_list.append(tf.reshape(sorted_output_branches[branch_index + 1],
                                                    shape=[num_of_batches, -1, self.NUM_CLASSES + 1]))

            if num_output_nodes_per_branch > 2:
                # Assume output is landmarks
                landmarks_predictors_list.append(tf.reshape(sorted_output_branches[branch_index + 2],
                                                            shape=[num_of_batches, -1, 10]))
        box_predictors = tf.concat(box_predictors_list, axis=1)
        class_predictors = tf.concat(class_predictors_list, axis=1)
        landmarks_predictors = tf.concat(landmarks_predictors_list, axis=1) if landmarks_predictors_list else None
        return box_predictors, class_predictors, landmarks_predictors

    def extract_anchors(self, min_sizes, steps):
        feature_maps = [[int(np.ceil(self._image_dims[0] / step)), int(np.ceil(self._image_dims[1] / step))] for step in
                        steps]

        anchors = []
        for feature_map_index, feature_map in enumerate(feature_maps):
            current_min_sizes = min_sizes[feature_map_index]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in current_min_sizes:
                    s_kx = min_size / self._image_dims[1]
                    s_ky = min_size / self._image_dims[0]
                    cx = (j + 0.5) / feature_map[1]
                    cy = (i + 0.5) / feature_map[0]
                    anchor = np.clip(np.array([cx, cy, s_kx, s_ky], dtype=np.float32), 0.0, 1.0)
                    anchors.append(tf.convert_to_tensor(anchor))

        anchors_tensor = tf.convert_to_tensor(anchors, name='Anchors')
        return anchors_tensor

    def _decode_landmarks(self, landmarks_detections, anchors):
        return tf.concat((anchors[:, :2] + landmarks_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 2:4] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 4:6] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 6:8] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 8:10] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          ), axis=1)

    def _decode_boxes(self, box_detections, anchors):
        boxes = tf.concat((
            anchors[:, :2] + box_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
            anchors[:, 2:] * tf.exp(box_detections[:, 2:] / self.SCALE_FACTORS[1])), 1)
        boxes_low_dims = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_high_dims = boxes[:, 2:] + boxes_low_dims
        new_boxes = tf.concat((boxes_low_dims, boxes_high_dims), axis=1)
        return new_boxes

    def tf_postproc(self, endnodes):
        with tf.name_scope('Postprocessor'):
            box_predictions, classes_predictions, landmarks_predictors = self.collect_box_class_predictions(endnodes)
            additional_fields = {}
            classes_predictions_softmax = tf.nn.softmax(classes_predictions, axis=2)

            # Slicing Background class score
            detection_scores = tf.slice(classes_predictions_softmax, [0, 0, 1], [-1, -1, -1])

            batch_size, num_proposals = tf.unstack(tf.slice(tf.shape(box_predictions), [0], [2]))

            tiled_anchor_boxes = tf.tile(tf.expand_dims(self._anchors, 0), [batch_size, 1, 1])
            tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])

            decoded_boxes = self._decode_boxes(tf.reshape(box_predictions, (-1, 4)), tiled_anchors_boxlist)
            detection_boxes = tf.reshape(decoded_boxes, [batch_size, num_proposals, 4])

            decoded_landmarks = None
            if tf.is_tensor(landmarks_predictors):
                decoded_landmarks = self._decode_landmarks(tf.reshape(landmarks_predictors, (-1, 10)),
                                                           tiled_anchors_boxlist)
                decoded_landmarks = tf.reshape(decoded_landmarks, [batch_size, num_proposals, 10])
                additional_fields['landmarks'] = decoded_landmarks

            detection_boxes = tf.identity(tf.expand_dims(detection_boxes, axis=[2]), 'raw_box_locations')
            (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, nmsed_additional_fields, num_detections) = \
                batch_multiclass_non_max_suppression(boxes=detection_boxes, scores=detection_scores,
                                                     score_thresh=self._score_threshold,
                                                     iou_thresh=self._nms_iou_thresh,
                                                     additional_fields=additional_fields,
                                                     max_size_per_class=1000, max_total_size=1000)
            # adding offset to the class prediction and cast to integer
            nmsed_classes = tf.cast(tf.add(nmsed_classes, self.LABEL_OFFSET), tf.int16)

        results = {'detection_boxes': nmsed_boxes,
                   'detection_scores': nmsed_scores,
                   'detection_classes': nmsed_classes,
                   'num_detections': num_detections, }

        nmsed_additional_fields = nmsed_additional_fields or {}
        face_landmarks = nmsed_additional_fields.get('landmarks')
        if tf.is_tensor(face_landmarks):
            results['face_landmarks'] = face_landmarks

        return results


class LibFaceDetectionPostProc(object):
    # The following params are corresponding to those used for training the model
    LABEL_OFFSET = 1
    NUM_CLASSES = 1
    SCALE_FACTORS = (10., 5.)

    def __init__(self, image_dims=(300, 300), nms_iou_thresh=0.6, score_threshold=0.3, anchors=None):
        self._image_dims = image_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_branches = len(anchors['steps'])
        self.num_anchors = [len(x) for x in anchors['min_sizes']]
        if anchors is None:
            raise ValueError('Missing detection anchors metadata')
        self._anchors = self.extract_anchors(anchors['min_sizes'], anchors['steps'])

    def collect_box_class_predictions(self, output_branches):
        boxlandmark_predictors_list = []
        class_predictors_list = []
        iou_predictors_list = []

        sorted_output_branches = output_branches
        num_branches = self._num_branches
        assert len(sorted_output_branches) % num_branches == 0, "All branches must have the same number of output nodes"
        num_output_nodes_per_branch = len(sorted_output_branches) // num_branches
        for branch_index in range(0, len(sorted_output_branches), num_output_nodes_per_branch):
            num_of_batches, _, _, _ = tf.unstack(tf.shape(sorted_output_branches[branch_index]))

            boxlandmarks_predictor = sorted_output_branches[branch_index]
            boxlandmarks_predictor = tf.reshape(boxlandmarks_predictor, shape=[num_of_batches, -1, 14])
            boxlandmark_predictors_list.append(boxlandmarks_predictor)
            class_predictors_list.append(tf.reshape(sorted_output_branches[branch_index + 1],
                                                    shape=[num_of_batches, -1, self.NUM_CLASSES + 1]))
            iou_predictors_list.append(tf.reshape(
                sorted_output_branches[branch_index + 2], shape=[num_of_batches, -1, 1]))

        boxlandmarks_predictors = tf.concat(boxlandmark_predictors_list, axis=1)
        box_predictors = tf.slice(boxlandmarks_predictors, [0, 0, 0], [-1, -1, 4])
        landmarks_predictors = tf.slice(boxlandmarks_predictors, [0, 0, 4], [-1, -1, -1])
        class_predictors = tf.concat(class_predictors_list, axis=1)
        iou_predictors = tf.concat(iou_predictors_list, axis=1)
        return box_predictors, class_predictors, landmarks_predictors, iou_predictors

    def extract_anchors(self, min_sizes, steps):
        feature_maps = [[int(np.floor(self._image_dims[0] / step)),
                         int(np.floor(self._image_dims[1] / step))] for step in steps]

        anchors = []
        for feature_map_index, feature_map in enumerate(feature_maps):
            current_min_sizes = min_sizes[feature_map_index]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in current_min_sizes:
                    s_kx = min_size / self._image_dims[1]
                    s_ky = min_size / self._image_dims[0]
                    cx = (j + 0.5) * steps[feature_map_index] / self._image_dims[1]
                    cy = (i + 0.5) * steps[feature_map_index] / self._image_dims[0]
                    anchor = np.clip(np.array([cx, cy, s_kx, s_ky], dtype=np.float32), 0.0, 1.0)
                    anchors.append(tf.convert_to_tensor(anchor))

        anchors_tensor = tf.convert_to_tensor(anchors, name='Anchors')
        return anchors_tensor

    def _decode_landmarks(self, landmarks_detections, anchors):
        return tf.concat((anchors[:, :2] + landmarks_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 2:4] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 4:6] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 6:8] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          anchors[:, :2] + landmarks_detections[:, 8:10] / self.SCALE_FACTORS[0] * anchors[:, 2:],
                          ), axis=1)

    def _decode_boxes(self, box_detections, anchors):
        boxes = tf.concat((
            anchors[:, :2] + box_detections[:, :2] / self.SCALE_FACTORS[0] * anchors[:, 2:],
            anchors[:, 2:] * tf.exp(box_detections[:, 2:] / self.SCALE_FACTORS[1])), 1)
        boxes_low_dims = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_high_dims = boxes[:, 2:] + boxes_low_dims
        new_boxes = tf.concat((boxes_low_dims, boxes_high_dims), axis=1)
        return new_boxes

    def tf_postproc(self, endnodes):
        with tf.name_scope('Postprocessor'):
            (box_predictions, classes_predictions,
             landmarks_predictors, iou_predictors) = self.collect_box_class_predictions(endnodes)
            additional_fields = {}
            classes_predictions_softmax = tf.nn.softmax(classes_predictions, axis=2)

            # Slicing Background class score
            raw_detection_scores = tf.slice(classes_predictions_softmax, [0, 0, 1], [-1, -1, -1])
            detection_scores = tf.sqrt(raw_detection_scores * tf.clip_by_value(iou_predictors, 0.0, 1.0))

            batch_size, num_proposals = tf.unstack(tf.slice(tf.shape(box_predictions), [0], [2]))

            tiled_anchor_boxes = tf.tile(tf.expand_dims(self._anchors, 0), [batch_size, 1, 1])
            tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])

            decoded_boxes = self._decode_boxes(tf.reshape(box_predictions, (-1, 4)), tiled_anchors_boxlist)
            detection_boxes = tf.reshape(decoded_boxes, [batch_size, num_proposals, 4])

            decoded_landmarks = None
            if tf.is_tensor(landmarks_predictors):
                decoded_landmarks = self._decode_landmarks(tf.reshape(landmarks_predictors, (-1, 10)),
                                                           tiled_anchors_boxlist)
                decoded_landmarks = tf.reshape(decoded_landmarks, [batch_size, num_proposals, 10])
                additional_fields['landmarks'] = decoded_landmarks

            detection_boxes = tf.identity(tf.expand_dims(detection_boxes, axis=[2]), 'raw_box_locations')
            (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, nmsed_additional_fields, num_detections) = \
                batch_multiclass_non_max_suppression(boxes=detection_boxes, scores=detection_scores,
                                                     score_thresh=self._score_threshold,
                                                     iou_thresh=self._nms_iou_thresh,
                                                     additional_fields=additional_fields,
                                                     max_size_per_class=1000, max_total_size=1000)
            # adding offset to the class prediction and cast to integer
            nmsed_classes = tf.cast(tf.add(nmsed_classes, self.LABEL_OFFSET), tf.int16)

        results = {'raw_detection_scores': raw_detection_scores,
                   'iou_scores': iou_predictors,
                   'detection_boxes': nmsed_boxes,
                   'detection_scores': nmsed_scores,
                   'detection_classes': nmsed_classes,
                   'num_detections': num_detections, }

        face_landmarks = nmsed_additional_fields.get('landmarks')
        if tf.is_tensor(face_landmarks):
            results['face_landmarks'] = face_landmarks

        return results


META_ARCH_TO_CLASS = {
    "libfacedetection": LibFaceDetectionPostProc,
    "retinaface": FaceDetectionPostProc,
}
DEFAULT_CLASS = FaceDetectionPostProc


def face_detection_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs.get('meta_arch', None)
    if meta_arch:
        postproc_class = META_ARCH_TO_CLASS[meta_arch]
    else:
        postproc_class = DEFAULT_CLASS
    postproc = postproc_class(image_dims=kwargs['img_dims'],
                              nms_iou_thresh=kwargs['nms_iou_thresh'],
                              score_threshold=kwargs['score_threshold'],
                              anchors=kwargs['anchors'])
    return postproc.tf_postproc(endnodes)
