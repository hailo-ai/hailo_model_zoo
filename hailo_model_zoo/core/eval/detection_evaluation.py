from collections import OrderedDict

import numpy as np
from prettytable import PrettyTable
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.core.postprocessing.detection_postprocessing import _get_labels


@EVAL_FACTORY.register(name="detection")
class DetectionEval(Eval):
    """COCO evaluation metric class."""

    def __init__(self, **kwargs):
        """Constructs COCO evaluation class.

        The class provides the interface to metrics_fn in TPUEstimator. The
        _update_op() takes detections from each image and push them to
        self.detections. The _evaluate() loads a JSON file in COCO annotation format
        as the groundtruths and runs COCO evaluation.
        """
        self._metric_names = [
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "ARmax1",
            "ARmax10",
            "ARmax100",
            "ARs",
            "ARm",
            "ARl",
        ]
        self._metrics_vals = [0, 0]
        self._centered = kwargs["centered"]
        self.labels_offset = kwargs["labels_offset"]
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self._cls_mapping, self._filtered_classes = self._create_class_mapping()
        self.show_results_per_class = kwargs["show_results_per_class"]
        self.dataset_name = kwargs["dataset_name"]
        self.mask = kwargs["channels_remove"].get("mask")
        self.reset()

    def _create_class_mapping(self):
        mask_list = list(np.where(np.array(self._channels_remove["mask"][0]) == 0)[0] + self.labels_offset)
        num_classes = len(self._channels_remove["mask"][0])
        cls_mapping = {}
        idx = 0
        for cl in range(num_classes):
            if cl in mask_list:
                cls_mapping[cl] = 0
            else:
                cls_mapping[cl] = idx
                idx += 1
        return cls_mapping, mask_list

    def reset(self):
        """Reset COCO API object."""
        self.coco_gt = COCO()
        self.detections = np.empty(shape=(0, 7))
        self.dataset = {"images": [], "annotations": [], "categories": []}
        self.images = set()
        self.annotation_id = 1
        self.category_ids = set()

    def _convert_letterbox_detections(self, img_dets, img_info, img_index):
        # fix latterbox preprocessing for either centered in the middle of the input frame (symmetric padding)
        # or aligned to the top left
        img_dets[:, 1] *= img_info["horizontal_pad"][img_index] + img_info["letterbox_width"][img_index]
        if self._centered:
            img_dets[:, 1] -= img_info["horizontal_pad"][img_index] / 2
        img_dets[:, 1] /= img_info["letterbox_width"][img_index]
        img_dets[:, 3] *= img_info["horizontal_pad"][img_index] + img_info["letterbox_width"][img_index]
        if self._centered:
            img_dets[:, 3] -= img_info["horizontal_pad"][img_index] / 2
        img_dets[:, 3] /= img_info["letterbox_width"][img_index]
        img_dets[:, 0] *= img_info["vertical_pad"][img_index] + img_info["letterbox_height"][img_index]
        if self._centered:
            img_dets[:, 0] -= img_info["vertical_pad"][img_index] / 2
        img_dets[:, 0] /= img_info["letterbox_height"][img_index]
        img_dets[:, 2] *= img_info["vertical_pad"][img_index] + img_info["letterbox_height"][img_index]
        if self._centered:
            img_dets[:, 2] -= img_info["vertical_pad"][img_index] / 2
        img_dets[:, 2] /= img_info["letterbox_height"][img_index]
        return img_dets

    def update_op(self, net_output, gt_labels):
        # net_output_detections_remapping:
        if "horizontal_pad" in gt_labels.keys() and "vertical_pad" in gt_labels.keys():
            # In the case of LetterBox preprocessing we will have to remap our detections
            # locations to the original image
            batch_dets = net_output["detection_boxes"]
            new_dets = []
            for i, img_dets in enumerate(batch_dets):
                new_img_dets = self._convert_letterbox_detections(img_dets, gt_labels, i)
                new_dets.append(new_img_dets)
            net_output["detection_boxes"] = np.stack(new_dets)

        update_inputs = [
            net_output["detection_scores"],
            net_output["detection_classes"],
            net_output["detection_boxes"],
            net_output["num_detections"],
            gt_labels["height"],
            gt_labels["width"],
            gt_labels["image_id"],
            gt_labels["num_boxes"],
            gt_labels["bbox"],
            gt_labels["area"],
            gt_labels["category_id"],
            gt_labels["is_crowd"],
        ]
        return self._update_op(*update_inputs)

    def _update_op(
        self,
        detection_scores,
        detection_classes,
        detection_boxes,
        num_detections,
        gt_data_height,
        gt_data_width,
        gt_data_image_id,
        gt_data_num_boxes,
        gt_data_bbox,
        gt_data_area,
        gt_data_category_id,
        gt_data_is_crowd,
    ):
        """Update detection results and groundtruth data.

        Append detection results to self.detections to aggregate results,
        and add parsed groundruth data to dictinoary with the same format
        as COCO dataset, which can be used for evaluation.

        Args:
         detections: Detection results is a dictionary with the following keys:
             (num_detections, detection_boxes, detection_classes, detection_scores)
             num_detections - number of detection per image, shape=
             detection_boxes - bbox output in the following format [ymin, xmin, ymax, xmax], shape=
             detection_classes - detectioed classes, shape=[ymin, xmin, ymax, xmax]
             detection_scores - detections confidence scores , shape=
             [image_id, x, y, width, height, score, class].
             detections are being parsed to be [?, 7] <-- (image_id, xmin, ymin, w, h, scores, classes)
         gt_data: Groundtruth annotations, part of the tfrecord, which are being used to update the
             coco dictionaries for evaluation with the following keys: (images,annotations,categories)
             images - a list of dictionaries of the image ids
             categories - a list of dictionaries of the class ids
             dataset - list of annotations where each annotation contains
                       (annotation_id, image_id, category_id, bbox, area, iscrowd)
        """
        for detection_count, boxes, classes, scores, height, width, image_id in zip(
            num_detections,
            detection_boxes,
            detection_classes,
            detection_scores,
            gt_data_height,
            gt_data_width,
            gt_data_image_id,
        ):
            # Rearrange detections bbox to be in the coco format - xmin, ymin, width, height
            bbox = np.transpose(
                np.vstack(
                    [
                        boxes[:, 1] * width,
                        boxes[:, 0] * height,
                        (boxes[:, 3] - boxes[:, 1]) * width,
                        (boxes[:, 2] - boxes[:, 0]) * height,
                    ]
                )
            )
            # Concat image id per detection in the first column
            image_id_vec = np.ones(shape=(bbox.shape[0], 1)) * image_id
            # Finalize the detections np.array as shape=(num_detections, 7)
            # (image_id, xmin, ymin, width, height, score, class)
            new_detections = np.hstack([image_id_vec, bbox, scores[:, np.newaxis], classes[:, np.newaxis]])[
                :detection_count, :
            ]
            self.detections = np.vstack([self.detections, new_detections])
        for num_boxes, image_id, _width, _height, bboxes, area, category_id, is_crowd in zip(
            gt_data_num_boxes,
            gt_data_image_id,
            gt_data_width,
            gt_data_height,
            gt_data_bbox,
            gt_data_area,
            gt_data_category_id,
            gt_data_is_crowd,
        ):

            def _add_gt_bbox(annotation_id, image_id, category_id, bbox, is_crowd, area):
                self.dataset["annotations"].append(
                    {
                        "id": int(annotation_id),
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": is_crowd,
                    }
                )

            for i in range(num_boxes):
                if self._channels_remove is None or category_id[i] not in self._filtered_classes:
                    _add_gt_bbox(
                        annotation_id=self.annotation_id,
                        image_id=image_id,
                        category_id=self._cls_mapping.get(category_id[i], 0)
                        if self._channels_remove
                        else category_id[i],
                        bbox=bboxes[i, :],
                        is_crowd=is_crowd[i],
                        area=area[i],
                    )
                    self.category_ids.add(category_id[i])
                    self.annotation_id += 1
            self.images.add(image_id)
            self.dataset["images"] = [{"id": int(img_id)} for img_id in self.images]
            self.dataset["categories"] = [{"id": int(category_id)} for category_id in self.category_ids]

        return 0

    def _print_per_class(self, coco_eval):
        if self.mask:
            coco_eval.params.catIds = [
                self._cls_mapping[id] for id in coco_eval.params.catIds if id in list(self._cls_mapping.keys())
            ]
        s = coco_eval.eval["precision"][
            ..., coco_eval.params.areaRngLbl.index("all"), coco_eval.params.maxDets.index(100)
        ]
        table = PrettyTable()
        table.field_names = ["Class Name", "Average Precision (AP)"]
        ap_values = np.mean(s, axis=(0, 1))
        labels = _get_labels(self.dataset_name)

        class_names = [
            labels[k]["name"] for k in coco_eval.params.catIds if k - self.labels_offset in np.where(ap_values > -1)[0]
        ]
        for class_name, ap in zip(class_names, ap_values[ap_values > -1]):
            table.add_row([class_name, f"{ap:.2f}"])
        print(table)

    def evaluate(self):
        """Evaluates with detections from all images in our data set with COCO API.

        Returns:
          coco_metric: float numpy array with shape [12] representing the
            coco-style evaluation metrics.
        """
        self.coco_gt.dataset = self.dataset
        self.coco_gt.createIndex()
        detections = self.detections
        coco_dt = self.coco_gt.loadRes(detections)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = list(self.images)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        if self.show_results_per_class:
            self._print_per_class(coco_eval)
        self._metrics_vals = np.array(coco_eval.stats, dtype=np.float32)

    def _get_accuracy(self):
        return OrderedDict(
            [(self._metric_names[0], self._metrics_vals[0]), (self._metric_names[1], self._metrics_vals[1])]
        )
