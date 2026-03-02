import cv2
import numpy as np
import tensorflow as tf


class PaddleDetPostProc(object):
    """
    Paddle OCR Detection Post-processing class.
    This class code is adapted from:
    https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py
    """

    def __init__(
        self,
        thresh=0.3,  # threshold for DB segmentation
        box_thresh=0.6,  # threshold for filtering high confidence polygons
        max_candidates=100,
        unclip_ratio=1.4,
        axis_aligned=True,
        **kwargs,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3

        self.axis_aligned = axis_aligned
        self.cat_label = kwargs.get("cat_label", 0)
        self.use_seg_score = kwargs.get("use_seg_score", True)

    def unclip(self, box, unclip_ratio):
        """
        Unclip a polygon by expanding it outward while preserving the original aspect ratio.
        box: numpy array of shape (4, 2) polygon vertice
        unclip_ratio: float, how far to expand (scaling factor)
        returns: numpy array of shape (4, 2)
        """
        center = np.mean(box, axis=0)
        vectors = box - center  # direction vectors from center to corners

        # Compute distance to each corner (Euclidean norm)
        distances = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Normalize direction vectors (unit vectors)
        unit_vectors = vectors / (distances + 1e-6)

        # Expand each point outward by unclip_ratio * original distance
        expanded_pts = box + unit_vectors * distances * (unclip_ratio - 1.0)

        return expanded_pts

    def get_mini_boxes(self, contour):
        """
        Get the minimum bounding box for a contour.
        Returns the box points and the shorter side length.
        contour: numpy array of shape (N, 2) representing the contour points
        returns: tuple (box points, shortest side length)
        """
        # Get the minimum bounding box for a contour
        bounding_box = cv2.minAreaRect(contour)  # bounding_box is ((x, y), (width, height), angle)

        # Get 4 corner points of the rectangle (not necessarily axis-aligned)
        points = cv2.boxPoints(bounding_box)
        points = sorted(points, key=lambda p: p[0])  # sort by x-coordinate

        # Left and right pairs, sorted by y-coordinate
        left = sorted(points[:2], key=lambda p: p[1])
        right = sorted(points[2:], key=lambda p: p[1])

        # Order: clockwise - top-left, top-right, bottom-right, bottom-left
        box = [left[0], right[0], right[1], left[1]]

        # Return box points and the shorter side length
        return np.array(box), min(bounding_box[1])

    def box_score(self, bitmap, _box):
        """
        Calculate the mean score of a bounding box in a bitmap.
        bitmap: single map with shape (H, W),
                whose values are binarized as {0, 1}
        _box: numpy array of shape (4, 2) polygon vertices
        returns: mean score within the axis-align bounding box
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()

        # Find the tightest axis-aligned bounding box enclosing the polygon
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        # Create a mask for the bounding box area
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin

        # Fill the mask with the polygon defined by the box
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)

        # Calculate the mean score within the rotated bounding box
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def polygon_score(self, bitmap, contour):
        """
        Calculate the mean score of the polygon in a bitmap.
        bitmap: single map with shape (H, W),
                whose values are binarized as {0, 1}
        contour: numpy array of shape (N, 2) representing the polygon vertices
        returns: mean score within the polygon area
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))
        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred):
        """
        Extract bounding boxes from a binary segmentation map.
        pred: numpy array of shape (1, H, W, 1), where values are binarized as {0, 1}
        returns: tuple (boxes, scores, valid_box_index)
                 boxes: numpy array of shape (max_candidates, 4, 2),
                        where each box is represented by 4 points (x, y)
                 scores: numpy array of shape (max_candidates,),
                         containing scores for each box
                 valid_box_index: int, number of valid boxes found
        """
        # extract the segmentation mask from the prediction based on DB scheme
        pred = np.squeeze(pred, axis=-1)
        mask = pred > self.thresh

        # Find Connected Components in the prediction mask
        height, width = mask.shape
        outs = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        boxes = np.zeros((self.max_candidates, 4, 2), dtype=np.float32)
        scores = np.zeros((self.max_candidates,), dtype=np.float32)
        valid_box_index = 0

        # Iterate over detection candidates, filter and process bounding boxes
        num_contours = min(len(contours), self.max_candidates)
        for index in range(num_contours):
            contour = contours[index]
            points, short_side = self.get_mini_boxes(contour)
            if short_side < self.min_size:
                continue  # Skip too small boxes

            if self.use_seg_score:
                score = self.polygon_score(pred, contour)
            else:
                score = self.box_score(pred, points)
            # check segmentation confidence
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, short_side = self.get_mini_boxes(box)  # box dimensions: [4, 2]
            if short_side < self.min_size + 2:
                continue

            box[:, 0] = np.clip(box[:, 0] / width, 0, 1)
            box[:, 1] = np.clip(box[:, 1] / height, 0, 1)

            boxes[valid_box_index] = box
            scores[valid_box_index] = score
            valid_box_index += 1

        return boxes, scores, valid_box_index

    def np_collect_boxes(self, endnodes):
        boxes_batch = []
        scores_batch = []
        num_det_batch = []
        for pred in endnodes:
            boxes, scores, num_detections = self.boxes_from_bitmap(pred)

            if self.axis_aligned:
                min_xy = np.min(boxes, axis=1)  # [100, 2]
                max_xy = np.max(boxes, axis=1)  # [100, 2]
                boxes = np.concatenate(
                    [min_xy[:, [1, 0]], max_xy[:, [1, 0]]], axis=1
                )  # [100, 4]: (ymin, xmin, ymax, xmax)

            boxes_batch.append(boxes)
            scores_batch.append(scores)
            num_det_batch.append(num_detections)

        return boxes_batch, scores_batch, num_det_batch

    def paddle_post_processing(self, endnodes, **kwargs):
        boxes_batch, scores_batch, num_det_batch = tf.numpy_function(
            self.np_collect_boxes, [endnodes], [tf.float32, tf.float32, tf.int64]
        )

        det_boxes = tf.stack(boxes_batch)
        det_scores = tf.stack(scores_batch, 0)
        def_cats = tf.ones_like(det_scores, dtype=tf.int32) * self.cat_label
        total_num_dets = tf.stack(num_det_batch, 0)

        return {
            "detection_boxes": det_boxes,
            "detection_scores": det_scores,
            "detection_classes": def_cats,
            "num_detections": total_num_dets,
        }

    def postprocessing(self, endnodes, **kwargs):
        return self.paddle_post_processing(endnodes, **kwargs)
