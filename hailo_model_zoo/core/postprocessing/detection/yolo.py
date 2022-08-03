import tensorflow as tf
import numpy as np

from .centernet import COCO_2017_TO_2014_TRANSLATION
from .detection_common import translate_coco_2017_to_2014
from tensorflow.image import combined_non_max_suppression


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class YoloPostProc(object):
    def __init__(self, img_dims=(608, 608), nms_iou_thresh=0.45, score_threshold=0.01,
                 anchors=None, output_scheme=None, classes=80,
                 labels_offset=0, meta_arch="yolo_v3", **kwargs):

        self._network_arch = meta_arch
        self._image_dims = img_dims
        self._nms_iou_thresh = nms_iou_thresh
        self.score_threshold = score_threshold
        if anchors is None or anchors["strides"] is None:
            raise ValueError("Missing detection anchors/strides metadata")
        self._anchors_list = anchors["sizes"]
        self._strides = anchors["strides"]
        self._num_classes = classes
        self._labels_offset = labels_offset
        self._yolo_decoding = {
            "yolo_v3": YoloPostProc._yolo3_decode,
            'yolo_v4': YoloPostProc._yolo4_decode,
            "yolo_v5": YoloPostProc._yolo5_decode,
            "yolox": YoloPostProc._yolox_decode,
        }
        self._nms_on_device = False
        if kwargs["device_pre_post_layers"] and kwargs["device_pre_post_layers"].get('nms', False):
            self._nms_on_device = True

    @staticmethod
    def _yolo3_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (sigmoid(raw_box_centers) + offsets) * stride  # dim [N, HxW, 3, 2]
        box_scales = (np.exp(raw_box_scales) * anchors_for_stride)  # dim [N, HxW, 3, 2]
        confidence = sigmoid(objness)  # dim [N, HxW, 3, 1]
        class_pred = sigmoid(class_pred)
        return box_centers, box_scales, confidence, class_pred

    @staticmethod
    def _yolo4_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride,
                      scale_x_y=1.05):
        box_centers = (raw_box_centers * scale_x_y - 0.5 * (scale_x_y - 1) + offsets) * stride  # dim [N, HxW, 3, 2]
        box_scales = (np.exp(raw_box_scales) * anchors_for_stride)  # dim [N, HxW, 3, 2]
        return box_centers, box_scales, objness, class_pred

    @staticmethod
    def _yolo5_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (raw_box_centers * 2. - 0.5 + offsets) * stride
        box_scales = (raw_box_scales * 2) ** 2 * anchors_for_stride  # dim [N, HxW, 3, 2]
        return box_centers, box_scales, objness, class_pred

    @staticmethod
    def _yolox_decode(raw_box_centers, raw_box_scales, objness, class_pred, anchors_for_stride, offsets, stride):
        box_centers = (raw_box_centers + offsets) * stride  # dim [N, HxW, 3, 2]
        box_scales = (np.exp(raw_box_scales) * stride)  # dim [N, HxW, 3, 2]
        return box_centers, box_scales, objness, class_pred

    def iou_nms(self, endnodes):
        endnodes = tf.transpose(endnodes, [0, 3, 1, 2])
        detection_boxes = endnodes[:, :, :, :4]
        detection_scores = tf.squeeze(endnodes[:, :, :, 4:], axis=3)

        (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = \
            combined_non_max_suppression(boxes=detection_boxes,
                                         scores=detection_scores,
                                         score_threshold=self.score_threshold,
                                         iou_threshold=self._nms_iou_thresh,
                                         max_output_size_per_class=100,
                                         max_total_size=100)

        nmsed_classes = tf.cast(tf.add(nmsed_classes, self._labels_offset), tf.int16)
        [nmsed_classes] = tf.py_function(translate_coco_2017_to_2014, [nmsed_classes], ['int32'])
        return {'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': num_detections}

    def yolo_postprocessing(self, endnodes, **kwargs):
        """
        endnodes is a list of 3 output tensors:
        endnodes[0] - stride 32 of input
        endnodes[1] - stride 16 of input
        endnodes[2] - stride 8 of input
        Returns:
        a tensor with dims: [BS, Total_num_of_detections_in_image, 6]
        where:
            total_num_of_detections_in_image = H*W*((1/32^2) + (1/16^2) + (1/8^2))*num_anchors*num_classes,
            with H, W as input dims.
            If H=W=608, num_anchors=3, num_classes=80 (coco 2017), we get:
            total_num_of_detections = 1819440 ~ 1.8M detections per image for the NMS
        """
        H_input = self._image_dims[0]
        W_input = self._image_dims[1]
        anchors_list = self._anchors_list
        strides = self._strides
        num_classes = self._num_classes
        """bringing output layers back to original form:"""
        if len(endnodes) > 4:
            endnodes = self.reorganize_split_output(endnodes)

        for output_ind, output_branch in enumerate(endnodes):  # iterating over the output layers:
            stride = strides[::-1][output_ind]
            anchors_for_stride = np.array(anchors_list[::-1][output_ind])
            anchors_for_stride = np.reshape(anchors_for_stride, (1, 1, -1, 2))  # dim [1, 1, 3, 2]
            output_branch_and_data = [output_branch, anchors_for_stride, stride]
            detection_boxes, detection_scores = tf.numpy_function(self.yolo_postprocess_numpy,
                                                                  output_branch_and_data,
                                                                  ['float32', 'float32'],
                                                                  name=f'{self._network_arch}_postprocessing')

            # detection_boxes is a [BS, num_detections, 1, 4] tensor, detection_scores is a
            # [BS, num_detections, num_classes] tensor
            BS = endnodes[0].shape[0]
            H = H_input // stride
            W = W_input // stride
            num_anchors = anchors_for_stride.size // 2
            num_detections = H * W * num_anchors
            detection_boxes.set_shape((BS, num_detections, 1, 4))
            detection_scores.set_shape((BS, num_detections, num_classes))
            # concatenating the detections from the different output layers:
            if output_ind == 0:
                detection_boxes_full = detection_boxes
                detection_scores_full = detection_scores
            else:
                detection_boxes_full = tf.concat([detection_boxes_full, detection_boxes], axis=1)
                detection_scores_full = tf.concat([detection_scores_full, detection_scores], axis=1)

        (nmsed_boxes, nmsed_scores, nmsed_classes, num_detections) = \
            combined_non_max_suppression(boxes=detection_boxes_full,
                                         scores=detection_scores_full,
                                         score_threshold=self.score_threshold,
                                         iou_threshold=self._nms_iou_thresh,
                                         max_output_size_per_class=100,
                                         max_total_size=100)

        # adding offset to the class prediction and cast to integer
        def translate_coco_2017_to_2014(nmsed_classes):
            return np.vectorize(COCO_2017_TO_2014_TRANSLATION.get)(nmsed_classes).astype(np.int32)

        nmsed_classes = tf.cast(tf.add(nmsed_classes, self._labels_offset), tf.int16)
        [nmsed_classes] = tf.py_function(translate_coco_2017_to_2014, [nmsed_classes], ['int32'])
        nmsed_classes.set_shape((1, 100))

        return {'detection_boxes': nmsed_boxes,
                'detection_scores': nmsed_scores,
                'detection_classes': nmsed_classes,
                'num_detections': num_detections}

    def yolo_postprocess_numpy(self, net_out, anchors_for_stride, stride):
        """
        net_out is shape: [N, 19, 19, 255] or [N, 38, 38, 255] or [N, 76, 76, 255]
        first we reshape it to be as in gluon and then follow gluon's shapes.
        output_ind = 0 for stride 32, 1 for stride 16, 2 for stride 8.
        """
        num_classes = self._num_classes
        BS = net_out.shape[0]  # batch size
        H = net_out.shape[1]
        W = net_out.shape[2]
        num_anchors = anchors_for_stride.size // 2  # 2 params for each anchor.
        num_pred = 1 + 4 + num_classes  # 2 box centers, 2 box scales, 1 objness, num_classes class scores
        alloc_size = (H, W)

        grid_x = np.arange(alloc_size[1])
        grid_y = np.arange(alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)  # dims [128,128], [128,128]

        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # dim [128,128,2]
        offsets = np.expand_dims(np.expand_dims(offsets, 0), 0)  # dim [1,1,128,128,2]

        pred = net_out.transpose((0, 3, 1, 2))  # now dims are: [N,C,H,W] as in Gluon.
        pred = np.reshape(pred, (BS, num_anchors * num_pred, -1))  # dim [N, 255, HxW]
        # dim [N, 361, 255], we did it so that the 255 be the last dim and can be reshaped.
        pred = pred.transpose((0, 2, 1))
        pred = np.reshape(pred, (BS, -1, num_anchors, num_pred))  # dim [N, HxW, 3, 85]]

        raw_box_centers = pred[:, :, :, 0:2]  # dim [N, HxW, 3, 2]
        raw_box_scales = pred[:, :, :, 2:4]  # dim [N,HxW, 3, 2]
        objness = pred[:, :, :, 4:5]  # dim [N, HxW, 3, 1]
        class_pred = pred[:, :, :, 5:]  # dim [N, HxW, 3, 80]
        offsets = offsets[:, :, :H, :W, :]  # dim [1, 1, H, W, 2]
        offsets = np.reshape(offsets, (1, -1, 1, 2))  # dim [1, HxW, 1, 2]
        box_centers, box_scales, confidence, class_pred = self._yolo_decoding[self._network_arch](
            raw_box_centers=raw_box_centers,
            raw_box_scales=raw_box_scales,
            objness=objness,
            class_pred=class_pred,
            anchors_for_stride=anchors_for_stride,
            offsets=offsets,
            stride=stride)

        class_score = class_pred * confidence  # dim [N, HxW, 3, 80]
        wh = box_scales / 2.0
        # dim [N, HxW, 3, 4]. scheme xmin, ymin, xmax, ymax
        bbox = np.concatenate((box_centers - wh, box_centers + wh), axis=-1)

        detection_boxes = np.reshape(bbox, (BS, -1, 1, 4))  # dim [N, num_detections, 1, 4]
        detection_scores = np.reshape(class_score, (BS, -1, num_classes))  # dim [N, num_detections, 80]

        # switching scheme from xmin, ymin, xmanx, ymax to ymin, xmin, ymax, xmax and normalize to 1:
        detection_boxes_tmp = np.zeros(detection_boxes.shape)
        detection_boxes_tmp[:, :, :, 0] = detection_boxes[:, :, :, 1] / self._image_dims[0]
        detection_boxes_tmp[:, :, :, 1] = detection_boxes[:, :, :, 0] / self._image_dims[1]
        detection_boxes_tmp[:, :, :, 2] = detection_boxes[:, :, :, 3] / self._image_dims[0]
        detection_boxes_tmp[:, :, :, 3] = detection_boxes[:, :, :, 2] / self._image_dims[1]

        detection_boxes = detection_boxes_tmp  # now scheme is: ymin, xmin, ymax, xmax
        return detection_boxes.astype(np.float32), detection_scores.astype(np.float32)

    def reorganize_split_output(self, endnodes):
        """endnodes is a list of output tensors. we split them into groups of 4,
           since the remodeling created 4 output tensors out of each single output tensor,
           and we reorganize those 4 tensors back into the form of a single output tensor.
           if originally there were 3 output branches, the remodeling created 12 output tensors.
           we split them into 3 groups of 4 and for each group return a single tensor.
        """
        reorganized_endnodes_list = []
        for index in range(len(self._anchors_list)):
            branch_index = int(4 * index)
            if 'yolox' in self._network_arch:
                # special case for yolox: 9 branches
                branch_index = int(3 * index)
                centers = endnodes[branch_index][:, :, :, :2]
                scales = endnodes[branch_index][:, :, :, 2:]
                obj = endnodes[branch_index + 1]
                probs = endnodes[branch_index + 2]
            else:
                centers = endnodes[branch_index]
                scales = endnodes[branch_index + 1]
                obj = endnodes[branch_index + 2]
                probs = endnodes[branch_index + 3]
            branch_endnodes = tf.py_function(self.reorganize_split_output_numpy,
                                             [centers, scales, obj, probs],
                                             ['float32'], name='yolov3_match_remodeled_output')

            reorganized_endnodes_list.append(branch_endnodes[0])  # because the py_func returns a list
        return reorganized_endnodes_list

    def reorganize_split_output_numpy(self, centers, scales, obj, probs):
        num_anchors = len(self._anchors_list[0]) // 2  # the ith element in anchors_list is a list for the x,y
        # anchor values in the ith output layer (stride)
        for anchor in range(num_anchors):
            concat_arrays_for_anchor = [centers[:, :, :, 2 * anchor:2 * anchor + 2],
                                        scales[:, :, :, 2 * anchor:2 * anchor + 2],
                                        obj[:, :, :, anchor:anchor + 1],
                                        probs[:, :, :, anchor * self._num_classes:(anchor + 1) * self._num_classes]]

            partial_concat = np.concatenate(concat_arrays_for_anchor, 3)

            if anchor == 0:
                full_concat_array = partial_concat
            else:
                full_concat_array = np.concatenate([full_concat_array, partial_concat], 3)
        return full_concat_array

    def postprocessing(self, endnodes, **kwargs):
        if self._nms_on_device:
            return self.iou_nms(endnodes)
        else:
            return self.yolo_postprocessing(endnodes, **kwargs)
