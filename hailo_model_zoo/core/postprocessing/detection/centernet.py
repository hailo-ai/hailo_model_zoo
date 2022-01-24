import tensorflow as tf
import numpy as np
from hailo_model_zoo.core.postprocessing.detection.detection_common import (COCO_2017_TO_2014_TRANSLATION,
                                                                            tf_postproc_nms_centernet)


def pad_list(lst, pad_size):
    if type(lst[0]) is np.ndarray:
        padding_element = np.zeros(lst[0].shape)
    elif type(lst[0]) is list:
        padding_element = len(lst[0]) * [0]
    elif type(lst[0]) in [np.float32, float]:
        padding_element = 0.
    elif type(lst[0]) in [np.int32, int]:
        padding_element = 0
    else:
        raise ValueError('type of lst items should be ndarray, list, float or int. instead got:', type(lst[0]))
    lst = lst + pad_size * [padding_element]
    return lst


def _find_topk(probs, topk):
    probsarray = np.reshape(probs, [probs.size])
    sortedprobs = np.sort(probsarray)
    descending_probs = sortedprobs[::-1]
    smallest_allowed_val = descending_probs[topk - 1]
    filtered_probs = probs * (probs >= smallest_allowed_val)
    return filtered_probs


class CenternetPostProc(object):
    def __init__(self, **kwargs):
        self._nms_on_device = kwargs["device_pre_post_layers"] and kwargs["device_pre_post_layers"].get('nms', False)
        self._nms_topk_perclass = kwargs.get('post_nms_topk', 400)

    def postprocessing(self, endnodes, device_pre_post_layers=None, **kwargs):
        """since the preprocess not only resizes the image but also pads the smaller
        dimension with zeros to a fixed width=height=512, it
        affects the relative dimensions of bboxes in the image (in a scale of (0,1)).
        we need to restore the relative bbox coordinates to
        the original convention based on the unpadded unresized image."""
        if self._nms_on_device:
            return tf_postproc_nms_centernet(endnodes, max_detections_per_class=self._nms_topk_perclass)
        bb_dict = {}
        if device_pre_post_layers and device_pre_post_layers.get('max_finder', False):
            endnodes.append(endnodes[0])  # following code expects prob tensor to be 3rd.
        else:
            endnodes0_padded = tf.pad(endnodes[0], [[0, 0], [1, 1], [1, 1], [0, 0]])
            maxpooled_probs = tf.nn.max_pool2d(endnodes0_padded, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
            probs_maxima_booleans = tf.cast(tf.math.equal(endnodes[0], maxpooled_probs), 'float32')
            probs_maxima_values = tf.math.multiply(probs_maxima_booleans, endnodes[0])
            endnodes.append(probs_maxima_values)
            # we discard the 0 element in the endnodes list. This is the probabilities tensor.
            # Instead we pass the sparse probabilities tensor probs_maxima_values:

        bb_probs, bb_classes, bb_boxes, num_detections, top_k_indices = tf.compat.v1.py_func(
            self._centernet_postprocessing, endnodes[1:], ['float32', 'int32', 'float32', 'int32', 'int64'],
            name='centernet_postprocessing')
        bb_dict['detection_scores'] = bb_probs
        bb_dict['detection_classes'] = bb_classes
        bb_dict['detection_boxes'] = bb_boxes
        bb_dict['num_detections'] = num_detections
        bb_dict['top_k_indices'] = top_k_indices
        return bb_dict

    def _generate_boxes(self, probs, coors, classes, widths, offsets, output_height, output_width):
        """expecting to receive descending order lists"""
        MAX_NUM_OF_DETECTIONS = 100
        SCALE = 4.
        label_offset = 1
        coors = [[SCALE * h, SCALE * w] for h, w in coors]
        widths = [[SCALE * width_x, SCALE * width_y] for width_x, width_y in widths]
        offsets = [[SCALE * offset_x, SCALE * offset_y] for offset_x, offset_y in offsets]
        bb_probs = []
        bb_classes = []
        bb_boxes = []
        num_detections = len(probs)
        required_zero_detections_padding = MAX_NUM_OF_DETECTIONS - num_detections
        for ind in range(len(probs)):
            x = coors[ind][1]
            y = coors[ind][0]
            w = widths[ind][0]
            h = widths[ind][1]
            dx = offsets[ind][0]
            dy = offsets[ind][1]
            """returning box coordinates normalized to [0, 1] out of the image"""
            x0 = np.float32((x + dx - 0.5 * w) / (SCALE * output_width))
            y0 = np.float32((y + dy - 0.5 * h) / (SCALE * output_height))
            x1 = np.float32((x + dx + 0.5 * w) / (SCALE * output_width))
            y1 = np.float32((y + dy + 0.5 * h) / (SCALE * output_height))
            obj_prob = np.float32(probs[ind])
            obj_class = np.float32(classes[ind] + label_offset)
            bb_probs.append(obj_prob)
            bb_classes.append(obj_class)
            bb_boxes.append([y0, x0, y1, x1])
        if bb_probs == []:
            bb_probs = [0.]
            bb_classes = [0]
            bb_boxes = [[0., 0., 0., 0.]]
            required_zero_detections_padding -= 1
        bb_probs = pad_list(bb_probs, required_zero_detections_padding)
        bb_classes = pad_list(bb_classes, required_zero_detections_padding)
        bb_boxes = pad_list(bb_boxes, required_zero_detections_padding)
        return np.expand_dims(np.array(bb_probs), 0).astype('float32'),\
            np.expand_dims(np.array(bb_classes), 0).astype('int32'),\
            np.expand_dims(np.array(bb_boxes), 0).astype('float32'),\
            (np.ones([1]) * num_detections).astype('int32')

    def _centernet_postprocessing(self, box_widths, box_offsets, sparse_probs, **kwargs):
        # endnodes3 is the sparse tensor of maximum prob values.
        TOPK = 100
        PROB_THRESH = 0.0
        output_height = sparse_probs.shape[1]
        output_width = sparse_probs.shape[2]

        for i in range(sparse_probs.shape[0]):  # iterating over images in batch
            topk_probs_list = []
            topk_coors_list = []
            topk_class_list = []
            topk_widths_list = []
            topk_offsets_list = []
            # return a map with nonzeros only if the value is in the top k value list:
            topk_probs = _find_topk(sparse_probs[i], TOPK)
            # a [batch_size, h_out, w_out, num_of_classes] array
            topk_probs_im = topk_probs * (topk_probs >= PROB_THRESH).astype('float')
            top_indices = np.unravel_index(np.argsort(topk_probs_im.ravel())[-TOPK:][::-1], topk_probs_im.shape)
            widths_im = box_widths[i]
            offsets_im = box_offsets[i]
            for h, w, cls in zip(*top_indices):
                if topk_probs_im[h, w, cls] > 0:
                    topk_probs_list.append(topk_probs_im[h, w, cls])
                    topk_coors_list.append([h, w])
                    topk_class_list.append(cls)
                    topk_widths_list.append(widths_im[h, w, :])
                    topk_offsets_list.append(offsets_im[h, w, :])

            bb_probs_for_im, bb_classes_for_im, bb_boxes_for_im, num_detections_for_im = \
                self._generate_boxes(topk_probs_list, topk_coors_list, topk_class_list,
                                     topk_widths_list, topk_offsets_list, output_height, output_width)
            if i == 0:
                bb_probs_batch = bb_probs_for_im
                bb_classes_batch = bb_classes_for_im
                bb_boxes_batch = bb_boxes_for_im
                num_detections_batch = num_detections_for_im
                topk_indices_per_image = np.stack([np.array([i] * TOPK)] + list(top_indices), axis=1)
            else:
                bb_probs_batch = np.concatenate([bb_probs_batch, bb_probs_for_im], 0)
                bb_classes_batch = np.concatenate([bb_classes_batch, bb_classes_for_im], 0)
                bb_boxes_batch = np.concatenate([bb_boxes_batch, bb_boxes_for_im], 0)
                num_detections_batch = np.concatenate([num_detections_batch, num_detections_for_im], 0)
                topk_indices_per_image = np.concatenate(
                    [topk_indices_per_image, np.stack([np.array([i] * TOPK)] + list(top_indices), axis=1)], 0)
        bb_classes_batch = np.vectorize(COCO_2017_TO_2014_TRANSLATION.get)(bb_classes_batch).astype(np.int32)
        return bb_probs_batch, bb_classes_batch, bb_boxes_batch, num_detections_batch, topk_indices_per_image
