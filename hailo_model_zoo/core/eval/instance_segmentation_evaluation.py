from collections import OrderedDict

import numpy as np
from prettytable import PrettyTable
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info
from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.eval.instance_segmentation_evaluation_utils import SparseInstEval, YolactEval, Yolov5SegEval
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.core.postprocessing.detection_postprocessing import _get_labels

EVAL_CLASS_MAP = {
    "yolact": YolactEval,
    "yolov5_seg": Yolov5SegEval,
    "sparseinst": SparseInstEval,
    "yolov8_seg": Yolov5SegEval,
}


@EVAL_FACTORY.register(name="instance_segmentation")
class InstanceSegmentationEval(Eval):
    """COCO evaluation metric class."""

    def __init__(self, **kwargs):
        """Constructs COCO evaluation class."""
        self.meta_arch = kwargs.get("meta_arch", None)
        self.mask_thresh = kwargs.get("mask_thresh", None)
        self.input_shape = kwargs.get("input_shape", None)
        dataset_name = kwargs.get("dataset_name", None)
        self._labels_map = kwargs.get("labels_map", None)
        dataset_info = get_dataset_info(dataset_name=dataset_name)
        self._label_map = dataset_info.label_map
        self._label_inv_map = {v: k for k, v in self._label_map.items()}
        self.labels_offset = kwargs["labels_offset"]
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self.cls_mapping, self.catIds = self._create_class_mapping()
        else:
            self._channels_remove = None
        self._mask_data = []

        self.eval_config = EVAL_CLASS_MAP[self.meta_arch]()
        self._metric_names = self.eval_config._metric_names  # noqa: SLF001 allow access to private member
        self._metrics_vals = [0] * len(self._metric_names)
        self.scale_boxes = self.eval_config.scale_boxes
        self.scale_masks = self.eval_config.scale_masks
        self.eval_mask = self.eval_config.eval_mask
        self.eval_bbox = self.eval_config.eval_bbox

        self._gt_ann_file = kwargs["gt_json_path"]
        self.show_results_per_class = kwargs["show_results_per_class"]
        self.dataset_name = kwargs["dataset_name"]
        self.mask = kwargs["channels_remove"].get("mask")
        self.reset()

    def _create_class_mapping(self):
        cats_to_eval = list(np.where(np.array(self._channels_remove["mask"][0]) == 1)[0])
        num_classes = len(self._channels_remove["mask"][0])
        cls_mapping = {}
        idx = 0
        for cl in range(num_classes):
            if cl in cats_to_eval:
                cls_mapping[idx] = cl
                idx += 1
        cats_to_eval.remove(0)
        cls_mapping.pop(0)
        return cls_mapping, cats_to_eval

    def reset(self):
        """Reset COCO API object."""
        self._coco_gt = COCO()
        self._detections = np.empty(shape=(0, 7))
        self._dataset = {"images": [], "annotations": [], "categories": []}
        self._images = set()
        self._annotation_id = 1
        self._category_ids = set()

    def _parse_net_output(self, net_output):
        return net_output["predictions"]

    def detections_to_coco(self, boxes, detection_scores, detection_classes, detection_masks, img_id):
        bbox = np.transpose(np.vstack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]))
        # Concat image id per detection in the first column
        image_id_vec = np.ones(shape=(bbox.shape[0], 1)) * img_id
        # Finalize the detections np.array as shape=(num_detections, 7)
        # (image_id, xmin, ymin, width, height, score, class)
        num_detections = detection_classes.shape[0]
        for i in range(num_detections):
            rle = maskUtils.encode(np.asfortranarray(detection_masks[i]).astype(np.uint8))
            rle["counts"] = rle["counts"].decode("ascii")  # json.dump doesn't like bytes strings
            detection_category_id = (
                detection_classes[i] + 1 if not self._channels_remove else self.cls_mapping[detection_classes[i] + 1]
            )
            self._mask_data.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(self._label_inv_map[detection_category_id]),
                    "segmentation": rle,
                    "bbox": [bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3]],
                    "score": float(detection_scores[i]),
                }
            )
            if self._channels_remove:
                detection_classes[i] = self.cls_mapping[detection_classes[i] + 1] - 1

        new_detections = np.hstack(
            [image_id_vec, bbox, detection_scores[:, np.newaxis], detection_classes[:, np.newaxis]]
        )[:num_detections, :]
        self._detections = np.vstack([self._detections, new_detections])

    def _get_category_id(self, category_id):
        if self._labels_map == [0]:
            return int(0)
        return int(category_id)

    def arrange_dataset(self, img_info):
        gt_data_height = img_info["height"]
        gt_data_width = img_info["width"]
        gt_data_image_id = img_info["image_id"]
        gt_data_num_boxes = img_info["num_boxes"]
        gt_data_bbox = img_info.get("bbox", np.zeros((gt_data_width.shape[0], 100, 4)))
        gt_data_area = img_info["area"]
        gt_data_category_id = img_info["category_id"]
        gt_data_is_crowd = img_info["is_crowd"]
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
                self._dataset["annotations"].append(
                    {
                        "id": int(annotation_id),
                        "image_id": int(image_id),
                        "category_id": self._get_category_id(category_id),
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": is_crowd,
                    }
                )

            for i in range(num_boxes):
                if category_id[i] in self._label_map.keys():
                    cat = self._label_map[category_id[i]] - 1
                    _add_gt_bbox(
                        annotation_id=self._annotation_id,
                        image_id=image_id,
                        category_id=cat,
                        bbox=bboxes[i, :],
                        is_crowd=is_crowd[i],
                        area=area[i],
                    )
                    self._category_ids.add(cat)
                    self._annotation_id += 1
            self._images.add(image_id)
            self._dataset["images"] = [{"id": int(img_id)} for img_id in self._images]
            self._dataset["categories"] = [{"id": int(category_id)} for category_id in self._category_ids]

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)
        gt_data_height = img_info["height"]
        gt_data_width = img_info["width"]
        gt_data_image_id = img_info["image_id"]

        for batch_idx, output in enumerate(net_output):
            output_shape = gt_data_height[batch_idx], gt_data_width[batch_idx]

            num_detections = output["detection_classes"].shape[0]
            detection_boxes = output.get("detection_boxes", np.zeros((num_detections, 4)))
            detection_masks = output["mask"]
            detection_scores = output["detection_scores"]
            detection_classes = output["detection_classes"]
            if num_detections:
                detection_boxes = self.scale_boxes(detection_boxes, shape_out=output_shape, shape_in=self.input_shape)
                detection_masks = self.scale_masks(detection_masks, shape_out=output_shape, shape_in=self.input_shape)
                # Binarize masks
                detection_masks = detection_masks > self.mask_thresh

                # Rearrange detections bbox to be in the coco format - xmin, ymin, width, height
                self.detections_to_coco(
                    detection_boxes, detection_scores, detection_classes, detection_masks, gt_data_image_id[batch_idx]
                )

        # Arrange annotations
        self.arrange_dataset(img_info)

    def _print_per_class(self, seg_eval):
        s = seg_eval.eval["precision"][..., seg_eval.params.areaRngLbl.index("all"), seg_eval.params.maxDets.index(100)]
        table = PrettyTable()
        table.field_names = ["Class Name", "Mask Average Precision (AP)"]
        ap_values = np.mean(s, axis=(0, 1))
        labels = _get_labels(self.dataset_name)
        class_names = [labels[k]["name"] for k in seg_eval.params.catIds]
        for class_name, ap in zip(class_names, ap_values[ap_values > -1]):
            table.add_row([class_name, f"{ap:.2f}"])
        print(table)

    def _evaluate_mask(self):
        gt_annotations = COCO(self._gt_ann_file)
        if self._labels_map == [0]:
            for ann in gt_annotations.anns.keys():
                # for zero-shot instance segmentation (segment anything) we map all
                # classes to the same category
                gt_annotations.anns[ann]["category_id"] = 1
        mask_dets = gt_annotations.loadRes(self._mask_data)
        seg_eval = COCOeval(gt_annotations, mask_dets, "segm")
        if self._labels_map == [0]:
            # zero-shot instance segmentation (segment anything) works on AR1000
            # COCO default is [1, 10, 100]
            seg_eval.params.maxDets = [1, 10, 1000]
            self._metric_names[8] = "mask ARmax1000"
        seg_eval.params.imgIds = list(self._images)
        if self._channels_remove:
            seg_eval.params.catIds = [self._label_inv_map[i] for i in self.catIds]
        seg_eval.evaluate()
        seg_eval.accumulate()
        seg_eval.summarize()
        if self.show_results_per_class:
            self._print_per_class(seg_eval)
        self._metrics_vals = list(np.array(seg_eval.stats, dtype=np.float32))

    def _evaluate_bbox(self):
        detections = self._detections
        coco_dt = self._coco_gt.loadRes(detections)
        coco_eval = COCOeval(self._coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = list(self._images)
        if self._channels_remove:
            coco_eval.params.catIds = [i - 1 for i in self.catIds]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self._metrics_vals += list(np.array(coco_eval.stats, dtype=np.float32))

    def evaluate(self):
        """Evaluates with detections from all images in our data set with COCO API.

        Returns:
            coco_metric: float numpy array with shape [12] representing the
            coco-style evaluation metrics.
        """
        self._coco_gt.dataset = self._dataset
        self._coco_gt.createIndex()
        if self.eval_mask:
            self._evaluate_mask()
        if self.eval_bbox:
            self._evaluate_bbox()

    def _get_accuracy(self):
        # in zero-shot instance segmentation we map all classes to zero and
        # report the AR1000 instead of mAP
        if self._labels_map == [0]:
            accuracy = OrderedDict(
                [(self._metric_names[8], self._metrics_vals[8]), (self._metric_names[11], self._metrics_vals[11])]
            )
        else:
            accuracy = OrderedDict(
                [(self._metric_names[0], self._metrics_vals[0]), (self._metric_names[1], self._metrics_vals[1])]
            )
        return accuracy
