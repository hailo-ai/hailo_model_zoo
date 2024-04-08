from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info
from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.faster_rcnn_proposals_nms import FasterRCNNProposalsNMS
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="faster_rcnn_stage2")
class FasterRCNNEval(Eval):
    """COCO evaluation metric class."""

    def __init__(self, **kwargs):
        """Constructs COCO evaluation class.
        GT is being loaded at initialization time from JSON.
        Rescaling of bboxes is being done - to correspond with the preprocessing
        where the images went through "resize_and_pad" operator.

        """
        dataset_name = kwargs.get('dataset_name', None)
        self.reset()
        image_height, image_width, _ = kwargs['input_shape']
        dataset_info = get_dataset_info(dataset_name=dataset_name)
        self._label_inv_map = {v: k for k, v in dataset_info.label_map.items()}
        self._metric_names = ['bbox AP', 'bbox AP50']
        self._metrics_vals = [0, 0]
        self._gt_ann_file = kwargs['gt_json_path']
        self._coco_gt = COCO(self._gt_ann_file)
        self._last_image_id = None
        self._faster_proposlas_nms = FasterRCNNProposalsNMS(image_height, image_width, self._coco_gt,
                                                            kwargs['score_threshold'], kwargs['nms_iou_thresh'],
                                                            label_inv_map=self._label_inv_map,
                                                            detections=self._detections)

    def reset(self):
        """Reset COCO API object."""
        self._coco_gt = COCO()
        # Create an empty detection array with 7 columns:
        # (image_id, xmin, ymin, width, height, score, class)
        self._detections = np.empty(shape=(0, 7))
        self._images = set()

    def update_op(self, net_output, gt_labels):
        batch_dets = net_output["detection_boxes"]
        image_ids = net_output["image_id"]
        new_dets = list()
        for i, (img_dets, image_id) in enumerate(zip(batch_dets, image_ids)):
            if self._last_image_id is None:
                self._last_image_id = image_id
            new_img_dets = self._faster_proposlas_nms._convert_resize_and_pad(img_dets, image_id)
            new_dets.append(new_img_dets)
            if not self._last_image_id == image_id:
                self._faster_proposlas_nms._finished_collect_proposals[self._last_image_id] = True
                self._last_image_id = image_id

        net_output["detection_boxes"] = np.stack(new_dets)

        update_inputs = [image_ids,
                         net_output["detection_boxes"],
                         net_output["detection_scores"]
                         ]
        return self._update_op(*update_inputs)

    def _update_op(self, image_ids, detection_boxes, detection_scores):
        for i, (image_id, bboxes, scores) in enumerate(zip(image_ids, detection_boxes, detection_scores)):
            self._faster_proposlas_nms.init_image_id_det_results(image_id, bboxes, scores)
        return 0

    def evaluate(self, force_last_img=False):
        """Evaluates with detections from all images in our data set with COCO API.

        Returns:
            coco_metric: float numpy array with shape [2] representing the
            coco-style evaluation metrics mAP / mAP50
        """
        detections = self._faster_proposlas_nms.nms_per_image(self._images, force_last_img=force_last_img,
                                                              _last_image_id=self._last_image_id)
        # detections = self._detections
        coco_dt = self._coco_gt.loadRes(detections)
        coco_eval = COCOeval(self._coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = list(self._images)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self._metrics_vals = np.array(coco_eval.stats, dtype=np.float32)

    @property
    def num_evaluated_images(self):
        return self._faster_proposlas_nms.num_evaluated_images

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1])])
