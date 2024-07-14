from functools import namedtuple

import numpy as np

from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import nms as cnms

PadInfo = namedtuple("PadInfo", ["height_pad", "width_pad", "scaling_factor"])


class Memoize:
    """Caching Class"""

    def __init__(self, fn):
        self._fn = fn
        self._memo = {}

    def __call__(self, *args):
        if args not in self._memo:
            self._memo[args] = self._fn(*args)
        return self._memo[args]


@Memoize
def compute_padding_values(h, w, target_height=600, target_width=800):
    """
    Compute the padding values out of the the original image shapes and the
    target shape which is (600,800)
    """
    H_ratio = target_height / h
    W_ratio = target_width / w
    scaling_factor = np.min([H_ratio, W_ratio])

    is_w_padded = int(H_ratio < W_ratio)
    is_h_padded = int(W_ratio < H_ratio)

    w_padding = np.floor(0.5 * is_w_padded * np.abs((target_width - w * scaling_factor)))
    h_padding = np.floor(0.5 * is_h_padded * np.abs((target_height - h * scaling_factor)))
    return int(h_padding), int(w_padding), scaling_factor


class FasterRCNNProposalsNMS(object):
    """This class manages the postprocessing parts added the evaluation: collect image proposals and NMS"""

    def __init__(self, height, width, coco_gt, score_threshold, iou_threshold, label_inv_map, detections):
        self._image_height = height
        self._image_width = width
        self._coco_gt = coco_gt
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._per_image_detection_results = {}
        self._is_nmsed = {}
        self._bbox_pad_mapping = {}
        self._create_bbox_padding_map()
        self._finished_collect_proposals = {}
        self._create_bbox_padding_map()
        self._label_inv_map = label_inv_map
        self._detections = detections

    def _create_bbox_padding_map(self):
        """
        Create an internal map between image id to padding info which contains - (h_pad, w_pad, scale)
        The pad and resize information is being extracted from the real image shape and the target shapes
        and later used for mapping the bboxes to the original image for evaluation
        """
        for _i, img_coco_info in enumerate(self._coco_gt.dataset["images"]):
            h_pad, w_pad, scaling = compute_padding_values(
                img_coco_info["height"], img_coco_info["width"], self._image_height, self._image_width
            )
            self._bbox_pad_mapping[img_coco_info["id"]] = PadInfo(h_pad, w_pad, scaling)

    def _convert_resize_and_pad(self, img_dets, img_id):
        """
        Detections transformation to the original image coordinates
        """
        conversion_info = self._bbox_pad_mapping[img_id]
        img_dets[:, 0] -= conversion_info.width_pad
        img_dets[:, 0] /= conversion_info.scaling_factor
        # clip to 0 xmin ymin
        img_dets[:, 1] -= conversion_info.height_pad
        img_dets[:, 1] /= conversion_info.scaling_factor
        # clip to img_size xmax / ymax
        img_dets[:, 2] -= conversion_info.width_pad
        img_dets[:, 2] /= conversion_info.scaling_factor
        img_dets[:, 3] -= conversion_info.height_pad
        img_dets[:, 3] /= conversion_info.scaling_factor
        return self._clip(img_dets, img_id)

    def _clip(self, img_dets, img_id):
        """
        Clip the detections to be inside the original image
        """
        xmin, ymin, xmax, ymax = np.split(img_dets, 4, axis=1)
        xmin[xmin < 0] = 0
        ymin[ymin < 0] = 0
        conversion_info = self._bbox_pad_mapping[img_id]
        xmax_clip = (self._image_width - conversion_info.width_pad * 2) / conversion_info.scaling_factor
        ymax_clip = (self._image_height - conversion_info.height_pad * 2) / conversion_info.scaling_factor
        xmax[xmax > xmax_clip] = xmax_clip
        ymax[ymin > ymax_clip] = ymax_clip
        return np.hstack([xmin, ymin, xmax, ymax])

    def nms_detections(self, img_id, top_k=200):
        """
        Perform per class nms for detections of one image according to the image id
        """
        detection_boxes = np.stack(self._per_image_detection_results[img_id]["detection_boxes"])
        detection_scores = np.stack(self._per_image_detection_results[img_id]["detection_scores"])
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        num_classes = detection_scores.shape[1]
        ind = np.arange(0, num_classes)
        found_classes = ind[np.amax(detection_scores, axis=0) > self._score_threshold]

        cls_lst = []
        scr_lst = []
        box_lst = []

        for _i, _cls in enumerate(sorted(found_classes)):
            cls_scores = detection_scores[:, _cls]
            cls_boxes = detection_boxes[:, _cls, :]
            mask_scores = cls_scores > self._score_threshold
            cls_scores_masked = cls_scores[mask_scores]
            cls_boxes_masked = cls_boxes[mask_scores, :]
            preds = np.hstack([cls_boxes_masked, cls_scores_masked[:, np.newaxis]])
            keep = cnms(preds, self._iou_threshold)
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores_masked[keep])
            box_lst.append(cls_boxes_masked[keep, :])

        if len(scr_lst) == 0:
            return []

        classes = np.concatenate(cls_lst, axis=0)
        scores = np.concatenate(scr_lst, axis=0)
        boxes = np.concatenate(box_lst, axis=0)

        scores_idx = np.argsort(scores)[::-1]
        scores_idx = scores_idx[:top_k]

        scores = scores[scores_idx]
        boxes = boxes[scores_idx]
        classes = classes[scores_idx]

        image_id_vec = np.ones_like(scores, dtype=np.int32) * img_id
        # (image_id, xmin, ymin, width, height, score, class)
        xmin, ymin, xmax, ymax = np.split(boxes, 4, axis=1)
        boxes_wh_coding = np.hstack([xmin, ymin, xmax - xmin, ymax - ymin])
        new_detections = np.hstack(
            [image_id_vec[:, np.newaxis], boxes_wh_coding, scores[:, np.newaxis], classes[:, np.newaxis]]
        )
        new_detections[:, 6] = [self._label_inv_map[i] for i in (new_detections[:, 6] + 1)]
        return new_detections

    def nms_per_image(self, images_set, force_last_img=False, _last_image_id=None):
        """
        The function manages the collected results NMS -
        NMS is being performed on images that weren't nmsed yet (is_nmsed=False)
        only if collecting the detections of this image was finished (_finished_collect_proposals=True)
        """
        if force_last_img:
            self._finished_collect_proposals[_last_image_id] = True
        for img_id in self._finished_collect_proposals:
            if not self._is_nmsed[img_id]:
                # perform nms on the data for this image_id
                # map categories, and write to self._detections
                new_detections = self.nms_detections(img_id)
                if len(new_detections) > 0:
                    self._detections = np.vstack([self._detections, new_detections])
                self._is_nmsed[img_id] = True
                images_set.add(img_id)
        return self._detections

    def _set_detections_data(self, image_id, bboxes, scores):
        self._per_image_detection_results[image_id]["detection_boxes"].append(bboxes)
        self._per_image_detection_results[image_id]["detection_scores"].append(scores)

    def init_image_id_det_results(self, image_id, bboxes, scores):
        self._per_image_detection_results.setdefault(image_id, {"detection_scores": [], "detection_boxes": []})
        self._is_nmsed.setdefault(image_id, False)
        self._set_detections_data(image_id, bboxes, scores)

    @property
    def num_evaluated_images(self):
        """
        Number of images which all of their detections were gathered
        """
        return len(self._finished_collect_proposals)
