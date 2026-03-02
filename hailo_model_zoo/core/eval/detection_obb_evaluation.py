from collections import OrderedDict

import cv2
import numpy as np
from prettytable import PrettyTable

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY
from hailo_model_zoo.core.postprocessing.detection.yolo_obb import batch_probiou
from hailo_model_zoo.core.postprocessing.detection_postprocessing import _get_labels


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): Recall curve.
        precision (list): Precision curve.

    Returns:
        ap (float): Average precision.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # integrate area under curve
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    return ap


def compute_ap_per_class_and_iou(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16,
) -> tuple:
    """Compute the average precision per class per IoU threshold for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Average precision, precision and recall curves
    ap = np.zeros((nc, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_labels = nt[ci]  # number of labels
        n_preds = i.sum()  # number of predictions
        if n_preds == 0 or n_labels == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_labels + eps)  # recall curve
        precision = tpc / (tpc + fpc)  # precision curve

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return ap, unique_classes.astype(int)


def match_predictions(
    predicted_classes,
    ground_truth_classes,
    iou_matrix,
    iou_thresholds,
):
    """Match predictions to ground truth objects using IoU-based greedy assignment.

    Args:
        predicted_classes (np.ndarray): Predicted class IDs, shape (N,)
        ground_truth_classes (np.ndarray): Ground truth class IDs, shape (G,)
        iou_matrix (np.ndarray): IoU values between GT and predictions, shape (G, N)
        iou_thresholds (list): IoU thresholds for evaluation, shape (T,)

    Returns:
        np.ndarray: True positive matrix of shape (N, T) indicating which predictions
                   are correct at each IoU threshold.
    """
    num_predictions = len(predicted_classes)
    num_thresholds = len(iou_thresholds)
    true_positives = np.zeros((num_predictions, num_thresholds), dtype=bool)

    if num_predictions == 0 or len(ground_truth_classes) == 0:
        return true_positives

    # create class compatibility matrix: GT classes (rows) vs predicted classes (cols)
    class_matches = ground_truth_classes[:, None] == predicted_classes[None, :]

    # zero out IoU for incompatible classes (different class predictions)
    class_filtered_ious = iou_matrix * class_matches

    # process each IoU threshold separately
    for threshold_idx, iou_threshold in enumerate(iou_thresholds):
        # find all GT-prediction pairs that exceed the IoU threshold
        valid_matches_mask = class_filtered_ious >= iou_threshold
        gt_indices, pred_indices = np.nonzero(valid_matches_mask)

        if len(gt_indices) == 0:
            continue

        # create matches array with [gt_idx, pred_idx] pairs
        match_pairs = np.column_stack([gt_indices, pred_indices])

        if len(match_pairs) > 1:
            # sort matches by IoU in descending order (highest IoU first)
            match_ious = class_filtered_ious[gt_indices, pred_indices]
            sorted_indices = np.argsort(match_ious)[::-1]
            match_pairs = match_pairs[sorted_indices]

            # keep first occurrence of each prediction (highest IoU)
            _, unique_pred_indices = np.unique(match_pairs[:, 1], return_index=True)
            match_pairs = match_pairs[unique_pred_indices]

            # keep first occurrence of each ground truth (highest IoU)
            _, unique_gt_indices = np.unique(match_pairs[:, 0], return_index=True)
            match_pairs = match_pairs[unique_gt_indices]

        # mark successful matches as true positives
        matched_prediction_indices = match_pairs[:, 1]
        true_positives[matched_prediction_indices, threshold_idx] = True

    return true_positives


def xyxyxyxy2xywhr(x):
    """Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.

    Args:
        x (np.ndarray): Input box corners with shape (N, 8) in [xy1, xy2, xy3, xy4] format.

    Returns:
        (np.ndarray): Converted data in [cx, cy, w, h, rotation] format with shape (N, 5). Rotation
            values are in radians from 0 to pi/2.
    """
    points = x.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return np.asarray(rboxes)


def xyxy2xywh(x):
    """Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray): Input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        (np.ndarray): Bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x, dtype=np.float32)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def scale_boxes(boxes, src_img_shape, tgt_img_shape, ratio_pad=None, padding: bool = True, xywh: bool = True):
    """Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Supports
    both xyxy and xywh box formats.

    Args:
        boxes (np.ndarray): Bounding boxes to rescale in format (N, 4).
        src_img_shape (tuple): Shape of the source image (height, width).
        tgt_img_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (np.ndarray): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(src_img_shape[0] / tgt_img_shape[0], src_img_shape[1] / tgt_img_shape[1])  # gain  = old / new
        pad_x = round((src_img_shape[1] - tgt_img_shape[1] * gain) / 2 - 0.1)
        pad_y = round((src_img_shape[0] - tgt_img_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x  # x padding
            boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain
    if not xywh:
        # clip bounding boxes to target image boundaries
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, tgt_img_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, tgt_img_shape[0])  # y1, y2
    return boxes


@EVAL_FACTORY.register(name="detection_obb")
class DetectionOBBEval(Eval):
    """OBB evaluation metric class."""

    def __init__(self, **kwargs):
        self._metric_names = [
            "AP",
            "AP50",
            "AP75",
        ]
        self._metrics_vals = [0, 0, 0]
        self._centered = kwargs["centered"]
        self.labels_offset = kwargs["labels_offset"]
        self._channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
        if self._channels_remove:
            self._cls_mapping, self._filtered_classes = self._create_class_mapping()
        self.show_results_per_class = kwargs["show_results_per_class"]
        self.dataset_name = kwargs["dataset_name"]
        self.names = _get_labels(self.dataset_name)
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
        self.detections = []
        self.dataset = {"images": [], "annotations": [], "categories": []}
        self.images = set()
        self.annotation_id = 1
        self.category_ids = set()

    def _convert_letterbox_detections(self, img_dets, img_info, img_index, normalize=False):
        # fix letterbox preprocessing for either centered in the middle of the input frame (symmetric padding)
        # or aligned to the top left
        img_dets[:, 1] *= img_info["horizontal_pad"][img_index] + img_info["letterbox_width"][img_index]
        if self._centered:
            img_dets[:, 1] -= img_info["horizontal_pad"][img_index] / 2
        if normalize:
            img_dets[:, 1] /= img_info["letterbox_width"][img_index]
        img_dets[:, 3] *= img_info["horizontal_pad"][img_index] + img_info["letterbox_width"][img_index]
        if self._centered:
            img_dets[:, 3] -= img_info["horizontal_pad"][img_index] / 2
        if normalize:
            img_dets[:, 3] /= img_info["letterbox_width"][img_index]
        img_dets[:, 0] *= img_info["vertical_pad"][img_index] + img_info["letterbox_height"][img_index]
        if self._centered:
            img_dets[:, 0] -= img_info["vertical_pad"][img_index] / 2
        if normalize:
            img_dets[:, 0] /= img_info["letterbox_height"][img_index]
        img_dets[:, 2] *= img_info["vertical_pad"][img_index] + img_info["letterbox_height"][img_index]
        if self._centered:
            img_dets[:, 2] -= img_info["vertical_pad"][img_index] / 2
        if normalize:
            img_dets[:, 2] /= img_info["letterbox_height"][img_index]
        return img_dets

    def update_op(self, net_output, gt_labels):
        self.image_name_id_map = {}
        for name, id in zip(gt_labels["image_name"], gt_labels["image_id"], strict=True):
            self.image_name_id_map[name.decode("utf-8")] = id
        # rescale gt boxes to original image size and convert to xywhr
        gt_boxes = []
        for i, boxes in enumerate(gt_labels["bbox"]):
            scales = np.array([gt_labels["width"][i], gt_labels["height"][i]])
            scales = np.tile(scales, 4)
            gt_box = np.rint(boxes * scales).astype(np.int32)
            gt_box = xyxyxyxy2xywhr(gt_box)
            gt_boxes.append(gt_box)
        gt_labels["bbox"] = gt_boxes

        # net_output_detections_remapping:
        if "horizontal_pad" in gt_labels.keys() and "vertical_pad" in gt_labels.keys():
            # In the case of LetterBox preprocessing we will have to remap our detections
            # locations to the original image
            batch_dets = net_output["detection_boxes"][:, :, :4]
            batch_angles = net_output["detection_boxes"][:, :, 4]
            new_dets = []
            for i, img_dets in enumerate(batch_dets):
                new_img_dets = self._convert_letterbox_detections(img_dets, gt_labels, i)
                new_img_dets = new_img_dets[:, [1, 0, 3, 2]]  # yxyx to xyxy
                new_img_dets = xyxy2xywh(new_img_dets)
                new_img_dets = scale_boxes(
                    new_img_dets,
                    (gt_labels["letterbox_height"][i], gt_labels["letterbox_width"][i]),
                    (gt_labels["height"][i], gt_labels["width"][i]),
                )
                new_dets.append(new_img_dets)
            net_output["detection_boxes"] = np.stack(new_dets)
            net_output["detection_boxes"] = np.concatenate(
                [net_output["detection_boxes"], batch_angles[:, :, np.newaxis]], axis=2
            )

        update_inputs = [
            net_output["detection_scores"],
            net_output["detection_classes"],
            net_output["detection_boxes"],
            net_output["num_detections"],
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
             detection_boxes - bbox output in the following format [ymin, xmin, ymax, xmax, angle], shape=
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
        for detection_count, boxes, classes, scores, image_id in zip(
            num_detections,
            detection_boxes,
            detection_classes,
            detection_scores,
            gt_data_image_id,
            strict=True,
        ):
            for i in range(detection_count):
                self.detections.append(
                    {
                        "image_id": int(image_id),
                        "bbox": boxes[i, :].tolist(),
                        "score": float(scores[i]),
                        "category_id": int(classes[i]),
                    }
                )

        for num_boxes, image_id, bboxes, area, category_id, is_crowd in zip(
            gt_data_num_boxes,
            gt_data_image_id,
            gt_data_bbox,
            gt_data_area,
            gt_data_category_id,
            gt_data_is_crowd,
            strict=True,
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
                        bbox=bboxes[i, :].tolist(),
                        is_crowd=int(is_crowd[i]),
                        area=float(area[i]),
                    )
                    self.category_ids.add(category_id[i])
                    self.annotation_id += 1
            self.images.add(image_id)
            self.dataset["images"] = [{"id": int(img_id)} for img_id in self.images]
            self.dataset["categories"] = [{"id": int(category_id)} for category_id in self.category_ids]

        return 0

    def _print_per_class(self, results):
        all_ap, ap_class_index = results
        aps = all_ap.mean(1)  # average over iou thresholds
        class_names = [self.names[i]["name"] for i in ap_class_index]
        table = PrettyTable()
        table.field_names = ["Class Name", "Average Precision (AP)"]
        for class_name, ap in zip(class_names, aps, strict=True):
            table.add_row([class_name, f"{ap:.2f}"])
        print(table)

    def evaluate(self):
        """Evaluates Mean Average Precision of detections from all images in the dataset."""
        stats = {
            "tp": [],
            "conf": [],
            "pred_cls": [],
            "target_cls": [],
        }
        iou_thresholds = np.linspace(0.50, 0.95, 10).tolist()

        for img_id in self.images:
            detections = [d for d in self.detections if d["image_id"] == img_id]
            labels = [a for a in self.dataset["annotations"] if a["image_id"] == img_id]

            det_scores = np.array([d["score"] for d in detections]) if detections else np.array([])
            det_classes = np.array([d["category_id"] for d in detections]) if detections else np.array([])
            det_boxes = np.array([d["bbox"] for d in detections]) if detections else np.empty((0, 5))

            gt_classes = np.array([a["category_id"] for a in labels]) if labels else np.array([])
            gt_boxes = np.array([a["bbox"] for a in labels]) if labels else np.empty((0, 5))

            if len(gt_classes) > 0:
                stats["target_cls"].append(gt_classes)

            if len(det_scores) > 0:
                if len(gt_classes) > 0:
                    ious = batch_probiou(gt_boxes, det_boxes)  # shape (gt, pred)
                    true_positives = match_predictions(
                        predicted_classes=det_classes,
                        ground_truth_classes=gt_classes,
                        iou_matrix=ious,
                        iou_thresholds=iou_thresholds,
                    )  # shape (pred, len(iou_thresholds))
                else:
                    # If no ground truth, all predictions are false positives
                    true_positives = np.zeros((len(det_scores), len(iou_thresholds)), dtype=bool)

                stats["tp"].append(true_positives)
                stats["conf"].append(det_scores)
                stats["pred_cls"].append(det_classes)

        # Handle empty cases
        if not stats["tp"]:
            stats["tp"] = [np.empty((0, len(iou_thresholds)), dtype=bool)]
        if not stats["conf"]:
            stats["conf"] = [np.array([])]
        if not stats["pred_cls"]:
            stats["pred_cls"] = [np.array([])]
        if not stats["target_cls"]:
            stats["target_cls"] = [np.array([])]

        stats["tp"] = np.concatenate(stats["tp"], axis=0)
        stats["conf"] = np.concatenate(stats["conf"], axis=0)
        stats["pred_cls"] = np.concatenate(stats["pred_cls"], axis=0)
        stats["target_cls"] = np.concatenate(stats["target_cls"], axis=0)

        results = compute_ap_per_class_and_iou(stats["tp"], stats["conf"], stats["pred_cls"], stats["target_cls"])

        mean_ap, mean_ap50, mean_ap75 = (results[0].mean(), results[0][:, 0].mean(), results[0][:, 5].mean())

        if self.show_results_per_class:
            self._print_per_class(results)
        self._metrics_vals = np.array([mean_ap, mean_ap50, mean_ap75], dtype=np.float32)

    def _get_accuracy(self):
        return OrderedDict(
            [(self._metric_names[0], self._metrics_vals[0]), (self._metric_names[1], self._metrics_vals[1])]
        )
