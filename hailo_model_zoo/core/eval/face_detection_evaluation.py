from collections import OrderedDict

import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.widerface_evaluation_external.evaluation import (
    dataset_pr_info, image_eval, img_pr_info, voc_ap)
from hailo_model_zoo.core.factory import EVAL_FACTORY

THRESH_NUM = 1000
IOU_THRESH = 0.5


@EVAL_FACTORY.register(name="face_detection")
class FaceDetectionEval(Eval):
    """
    Widerface evaluation metric class.
    """

    def __init__(self, **kwargs):
        super(FaceDetectionEval, self).__init__()
        self._pr_curve = np.zeros((THRESH_NUM, 2)).astype('float')
        self._ap = 0
        self._face_count = 0

    def reset(self):
        pass

    def _parse_net_output(self, net_output):
        return net_output

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        batch_size = net_output["detection_scores"].shape[0]
        num_boxes = net_output["detection_scores"].shape[1]
        new_det = np.reshape(net_output["detection_scores"], (batch_size, num_boxes, 1))
        predictions = np.concatenate((net_output["detection_boxes"], new_det), axis=2)
        for i in range(predictions.shape[0]):
            width = gt_labels['width'][i]
            height = gt_labels['height'][i]
            if 'original_width' in gt_labels:
                horizontal_resize_factor = (gt_labels['original_width'][i]
                                            * (width / (width - gt_labels['horizontal_pad'][i])))
                vertical_resize_factor = (gt_labels['original_height'][i]
                                          * (height / (height - gt_labels['vertical_pad'][i])))
            else:
                horizontal_resize_factor = width
                vertical_resize_factor = height

            predictions[i, :, (0, 2)] *= horizontal_resize_factor
            predictions[i, :, (1, 3)] *= vertical_resize_factor

        predictions = np.concatenate((np.expand_dims(predictions[:, :, 0], axis=-1),
                                      np.expand_dims(predictions[:, :, 1], axis=-1),
                                      np.expand_dims(predictions[:, :, 2] - predictions[:, :, 0], axis=-1),
                                      np.expand_dims(predictions[:, :, 3] - predictions[:, :, 1], axis=-1),
                                      np.expand_dims(predictions[:, :, 4], axis=-1)), axis=-1)

        for i in range(gt_labels['num_boxes'].shape[0]):
            keep_index = gt_labels['wider_hard_keep_index'][i].tolist()
            while 0 in keep_index:
                keep_index.remove(0)
            keep_index = np.array(keep_index)
            self._face_count += len(keep_index)
            if len(predictions[i]) == 0 or len(gt_labels['bbox'][i]) == 0:
                continue
            ignore = np.zeros(gt_labels['bbox'][i].shape[0])
            if len(keep_index) != 0:
                ignore[keep_index - 1] = 1

            pred_recall, proposal_list = image_eval(predictions[i].astype('float64'),
                                                    gt_labels["bbox"][i].astype('float64'), ignore, IOU_THRESH)
            _img_pr_info = img_pr_info(THRESH_NUM, predictions[i], proposal_list, pred_recall)

            self._pr_curve += _img_pr_info

        return 0

    def evaluate(self):
        """
        Evaluates with detections from all images in our data set with widerface API.
        """
        pr_curve = dataset_pr_info(THRESH_NUM, self._pr_curve, self._face_count)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        self._ap = voc_ap(recall, propose)

    def _get_accuracy(self):
        return OrderedDict((('widerface HARD ap', self._ap),))
