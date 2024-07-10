import cv2
import numpy as np

from hailo_model_zoo.core.postprocessing.instance_segmentation_postprocessing import _sanitize_coordinates

BBOX_METRIC_NAMES = [
    "bbox AP",
    "bbox AP50",
    "bbox AP75",
    "bbox APs",
    "bbox APm",
    "bbox APl",
    "bbox ARmax1",
    "bbox ARmax10",
    "bbox ARmax100",
    "bbox ARs",
    "bbox ARm",
    "bbox ARl",
]

SEGM_METRIC_NAMES = [
    "mask AP",
    "mask AP50",
    "mask AP75",
    "mask APs",
    "mask APm",
    "mask APl",
    "mask ARmax1",
    "mask ARmax10",
    "mask ARmax100",
    "mask ARs",
    "mask ARm",
    "mask ARl",
]


class InstSegEvalBase:
    def __init__(self):
        self.eval_bbox = True
        self.eval_mask = True
        self._metric_names = SEGM_METRIC_NAMES + BBOX_METRIC_NAMES

    def scale_boxes(self, boxes, **kwargs):
        return boxes

    def scale_masks(self, masks, **kwargs):
        return masks


class YolactEval(InstSegEvalBase):
    def __init__(self):
        super().__init__()

    def scale_boxes(self, boxes, shape_out, **kwargs):
        boxes[:, 0], boxes[:, 2] = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], shape_out[1], cast=False)
        boxes[:, 1], boxes[:, 3] = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], shape_out[0], cast=False)
        boxes = np.array(boxes, np.float64)
        return boxes

    def scale_masks(self, masks, shape_out, **kwargs):
        masks = cv2.resize(masks, (shape_out[1], shape_out[0]))
        if len(masks.shape) < 3:
            masks = np.expand_dims(masks, axis=0)
        else:
            masks = np.transpose(masks, (2, 0, 1))
        return masks


class Yolov5SegEval(InstSegEvalBase):
    def __init__(self):
        super().__init__()

    def scale_boxes(self, boxes, shape_out, shape_in=None, **kwargs):
        # scales the boxes values to the input size
        boxes[:, 0::2] *= shape_in[1]
        boxes[:, 1::2] *= shape_in[0]

        ratio_pad = kwargs.get("ratio_pad", None)
        if ratio_pad is None:  # calculate from shape_out
            if shape_in is None:
                raise ValueError("Expected shape_in to be a tuple of size 2 when ratio is not provided but got None")
            gain = min(shape_in[0] / shape_out[0], shape_in[1] / shape_out[1])  # gain  = old / new
            pad = [
                (shape_in - shape_out * gain) / 2 for shape_in, shape_out in zip(shape_in[:2], shape_out[:2])
            ]  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[1]  # x padding
        boxes[:, [1, 3]] -= pad[0]  # y padding
        boxes[:, :4] /= gain

        # Clip boxes (xyxy) to image shape (height, width)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape_out[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape_out[0])  # y1, y2

        return boxes

    def scale_masks(self, masks, shape_out, shape_in=None, **kwargs):
        ratio_pad = kwargs.get("ratio_pad", None)
        if ratio_pad is None:  # calculate from shape_out and shape_in
            if shape_in is None:
                raise ValueError("Expected shape_in to be a tuple of size 2 but got None")
            gain = min(shape_in[0] / shape_out[0], shape_in[1] / shape_out[1])  # gain  = old / new
            pad = (shape_in[1] - shape_out[1] * gain) / 2, (shape_in[0] - shape_out[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]
        top, left = int(pad[1]), int(pad[0])  # y, x
        bottom, right = int(shape_in[0] - pad[1]), int(shape_in[1] - pad[0])

        masks = masks.transpose((1, 2, 0))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(masks, (shape_out[1], shape_out[0]))

        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        masks = masks.transpose((2, 0, 1))
        return masks


class SparseInstEval(InstSegEvalBase):
    def __init__(self):
        super().__init__()
        self.eval_bbox = False
        self._metric_names = SEGM_METRIC_NAMES
