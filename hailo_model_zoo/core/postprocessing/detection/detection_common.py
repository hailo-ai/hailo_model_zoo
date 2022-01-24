import tensorflow as tf
import numpy as np


COCO_2017_TO_2014_TRANSLATION = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                                 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19,
                                 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28,
                                 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
                                 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47,
                                 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55,
                                 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63,
                                 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75,
                                 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84,
                                 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}


def translate_coco_2017_to_2014(nmsed_classes):
    return np.vectorize(COCO_2017_TO_2014_TRANSLATION.get)(nmsed_classes).astype(np.int32)


def tf_postproc_nms(endnodes, score_threshold, coco_2017_to_2014=True):
    def _single_batch_parse(args):
        frame_detections = args[:, :, :]
        indices = tf.where(frame_detections[:, :, 4] > score_threshold)[0:100]
        boxes_and_scores = tf.gather_nd(frame_detections, indices)
        num_detections = tf.shape(indices)[0]
        pad_size = 100 - num_detections
        classes_expanded = tf.expand_dims(tf.cast(indices[:, 0], tf.int32), axis=1) + 1
        classes_expanded = tf.squeeze(tf.pad(classes_expanded, paddings=[[0, pad_size], [0, 0]],
                                      mode='CONSTANT', constant_values=0), axis=1)
        final_frame_results = boxes_and_scores
        final_frame_results_padded = tf.pad(final_frame_results,
                                            paddings=[[0, pad_size],
                                                      [0, 0]], mode='CONSTANT', constant_values=0)
        return final_frame_results_padded[:, :4], final_frame_results_padded[:, 4], classes_expanded, num_detections

    with tf.compat.v1.name_scope('Postprocessor'):
        detections = tf.transpose(endnodes, [0, 1, 3, 2])
        post_processing_boxes, post_processing_scores, post_processing_classes, post_num_detections = \
            tf.map_fn(_single_batch_parse, detections, dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
                      parallel_iterations=32, back_prop=False)
    if coco_2017_to_2014:
        [post_processing_classes] = tf.compat.v1.py_func(
            translate_coco_2017_to_2014, [post_processing_classes], ['int32'])
    return {'detection_boxes': post_processing_boxes,
            'detection_scores': post_processing_scores,
            'detection_classes': post_processing_classes,
            'num_detections': post_num_detections}


def tf_postproc_nms_centernet(endnodes, max_detections_per_class, coco_2017_to_2014=True, input_division_factor=4):
    # SDK Tensor -> ADK
    def _single_batch_parse(args):
        label_offset = 1
        frame_detections = args[:, :, :]
        num_of_classes = args.shape[0]
        boxes_and_scores = tf.reshape(frame_detections, [num_of_classes * max_detections_per_class, 5])

        # Taking top k scores from all classes
        _, topk_indices = tf.math.top_k(tf.reshape(boxes_and_scores[:, -1], [-1]), sorted=True,
                                        k=int(max_detections_per_class / input_division_factor))
        boxes_and_scores = tf.gather(boxes_and_scores, topk_indices)
        num_detections = tf.shape(topk_indices)[0]
        pad_size = max_detections_per_class - num_detections
        classes_expanded = tf.unravel_index(topk_indices, [num_of_classes, max_detections_per_class])[0] + label_offset
        classes_expanded = tf.expand_dims(classes_expanded, axis=1)
        classes_expanded = tf.squeeze(tf.pad(classes_expanded, paddings=[[0, pad_size], [0, 0]],
                                      mode='CONSTANT', constant_values=0), axis=1)
        final_frame_results = boxes_and_scores
        final_frame_results_padded = tf.pad(final_frame_results,
                                            paddings=[[0, pad_size],
                                                      [0, 0]], mode='CONSTANT', constant_values=0)
        return final_frame_results_padded[:, :4], final_frame_results_padded[:, 4], classes_expanded, num_detections

    with tf.name_scope('Postprocessor'):
        detections = tf.transpose(endnodes, [0, 1, 3, 2])
        post_processing_boxes, post_processing_scores, post_processing_classes, post_num_detections = \
            tf.map_fn(_single_batch_parse, detections, dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
                      parallel_iterations=32, back_prop=False)
    if coco_2017_to_2014:
        [post_processing_classes] = tf.compat.v1.py_func(translate_coco_2017_to_2014,
                                                         [post_processing_classes], ['int32'])
    return {'detection_boxes': post_processing_boxes,
            'detection_scores': post_processing_scores,
            'detection_classes': post_processing_classes,
            'num_detections': post_num_detections}
