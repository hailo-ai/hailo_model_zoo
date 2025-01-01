from itertools import product
from math import sqrt

import cv2
import numpy as np

from hailo_model_zoo.core.datasets.datasets_info import CLASS_NAMES_COCO, get_dataset_info
from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import nms as cnms
from hailo_model_zoo.utils import path_resolver

COLORS = (
    (244, 67, 54),
    (233, 30, 99),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 188, 212),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 193, 7),
    (255, 152, 0),
    (255, 87, 34),
    (121, 85, 72),
    (158, 158, 158),
    (96, 125, 139),
)


def _sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = np.array(_x1, np.float64)
        _x2 = np.array(_x2, np.float64)
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=None)
    x2 = np.clip(x2 + padding, a_max=img_size, a_min=None)

    return x1, x2


def _crop(masks, boxes, padding=1):
    h, w, n = masks.shape
    x1, x2 = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = np.reshape(np.arange(w), (1, -1, 1))
    cols = np.reshape(np.arange(h), (-1, 1, 1))

    # TODO: no expand
    # cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= np.reshape(x1, (1, 1, -1))
    masks_right = rows < np.reshape(x2, (1, 1, -1))
    masks_up = cols >= np.reshape(y1, (1, 1, -1))
    masks_down = cols < np.reshape(y2, (1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * (1.0 * crop_mask)


def _intersect(box_a, box_b):
    max_xy = np.minimum(np.expand_dims(box_a[:, :, 2:], axis=2), np.expand_dims(box_b[:, :, 2:], axis=1))
    min_xy = np.maximum(np.expand_dims(box_a[:, :, :2], axis=2), np.expand_dims(box_b[:, :, :2], axis=1))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

    return inter[:, :, :, 0] * inter[:, :, :, 1]


def _jaccard(box_a, box_b, iscrowd=False):
    use_batch = True
    if len(box_a.shape) == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = _intersect(box_a, box_b)
    area_a = np.expand_dims((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), axis=2)  # [A,B]
    area_b = np.expand_dims((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), axis=1)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union

    return out if use_batch else np.squeeze(out, axis=0)


def _decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


class Detect(object):
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self._num_classes = num_classes
        self._background_label = bkg_label
        self._top_k = top_k
        # Parameters used in nms.
        self._nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non negative.")
        self._conf_thresh = conf_thresh

    def __call__(self, loc_data, proto_data, conf_data, mask_data, prior_data):
        out = []

        batch_size = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        conf_preds = np.transpose(np.reshape(conf_data, (batch_size, num_priors, self._num_classes)), (0, 2, 1))

        for batch_idx in range(batch_size):
            decoded_boxes = _decode(loc_data[batch_idx], prior_data)
            result = self._detect(batch_idx, conf_preds, decoded_boxes, mask_data)

            if result is not None and proto_data is not None:
                result["proto"] = proto_data[batch_idx]

            out.append(result)

        return out

    def _detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """Perform nms for only the max scoring class that isn't background (class 0)"""
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores = np.amax(cur_scores, axis=0)

        keep = conf_scores > self._conf_thresh
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if scores.shape[1] == 0:
            return None

        boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self._nms_thresh, self._top_k)

        return {"detection_boxes": boxes, "mask": masks, "detection_classes": classes, "detection_scores": scores}

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200, second_threshold=False):
        max_num_detections = 100
        idx = np.flip(np.argsort(scores, axis=1), axis=1)
        scores = np.flip(np.sort(scores, axis=1), axis=1)

        idx = idx[:, :top_k]
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.shape

        boxes = np.reshape(boxes[idx[:]], (num_classes, num_dets, 4))
        masks = np.reshape(masks[idx[:]], (num_classes, num_dets, -1))

        iou = _jaccard(boxes, boxes)
        iou = np.triu(iou, 1)
        iou_max = np.amax(iou, axis=1)

        # Now just filter out the ones higher than the threshold
        keep = iou_max <= iou_threshold
        if second_threshold:
            keep *= scores > self._conf_thresh

        # Assign each kept detection to its corresponding class
        classes = np.arange(num_classes)[:, None]
        classes = np.reshape(np.repeat(classes, keep.shape[1]), (num_classes, keep.shape[1]))
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        idx = np.flip(np.argsort(scores, axis=0), axis=0)
        scores = np.flip(np.sort(scores, axis=0), axis=0)
        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prep_display(
    dets_out,
    img,
    score_threshold,
    class_color=False,
    mask_alpha=0.45,
    channels_remove=None,
    class_names=None,
    mask_thresh=0.5,
):
    top_k = 5
    img_gpu = img / 255.0
    h, w, _ = img.shape
    if not channels_remove["enabled"]:
        visualization_class_names = class_names
    else:
        channels_remove = np.array(channels_remove["mask"][0])
        class_names_mask = channels_remove[1:]  # Remove background class
        cats = np.where(np.array(class_names_mask) == 1)[0]
        visualization_class_names = list(np.array(class_names)[cats])

    boxes = dets_out["detection_boxes"]
    masks = dets_out["mask"]
    classes = dets_out["detection_classes"]
    scores = dets_out["detection_scores"]

    # Scale Boxes
    boxes[:, 0], boxes[:, 2] = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = np.array(boxes, np.float64)

    # Scale Masks
    masks = cv2.resize(masks, (w, h))
    if len(masks.shape) < 3:
        masks = np.expand_dims(masks, axis=0)
    else:
        masks = np.transpose(masks, (2, 0, 1))
    # Binarize the masks
    masks = masks > mask_thresh

    if not masks.shape[0]:
        return np.array(img_gpu * 255, np.uint8)
    boxes = boxes[:top_k]
    masks = masks[:top_k]
    classes = classes[:top_k]
    scores = scores[:top_k]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return np.array(img_gpu * 255, np.uint8)

    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = 1.0 * color / 255.0
                color_cache[on_gpu][color_idx] = color
            return color

    masks = masks[:num_dets_to_consider, :, :, None]

    # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
    colors = np.concatenate(
        [np.reshape(get_color(j, on_gpu=None), (1, 1, 1, 3)) for j in range(num_dets_to_consider)], axis=0
    )
    masks_color = np.repeat(masks, 3, axis=-1) * colors * mask_alpha

    # This is 1 everywhere except for 1-mask_alpha where the mask is
    inv_alpha_masks = masks * (-mask_alpha) + 1
    masks_color_summand = masks_color[0]
    if num_dets_to_consider > 1:
        inv_alpha_cumul = np.cumprod(inv_alpha_masks[: (num_dets_to_consider - 1)], axis=0)
        masks_color_cumul = masks_color[1:] * inv_alpha_cumul
        masks_color_summand += np.sum(masks_color_cumul, axis=0)

    img_gpu = img_gpu * np.prod(inv_alpha_masks, axis=0) + masks_color_summand
    img_numpy = img_gpu * 255

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        score = scores[j]
        color = list(get_color(j))
        # if args.display_bboxes:
        cv2.rectangle(img_numpy, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        _class = visualization_class_names[classes[j]]
        text_str = "%s: %.2f" % (_class, score)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (x1, y1 - 3)

        cv2.rectangle(img_numpy, (int(x1), int(y1)), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
        cv2.putText(
            img_numpy,
            text_str,
            (int(text_pt[0]), int(text_pt[1])),
            font_face,
            font_scale,
            [255.0, 255.0, 255.0],
            font_thickness,
            cv2.LINE_AA,
        )

    return np.array(img_numpy, np.uint8)


def _softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def _make_priors(anchors, img_size):
    priors = []
    square_anchors = True if len(anchors["scales"][0]) == 1 else False
    for conv_size, pred_scale in zip(anchors["feature_map"], anchors["scales"]):
        prior_data = []
        for j, i in product(range(conv_size), range(conv_size)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_size
            y = (j + 0.5) / conv_size
            for scale in pred_scale:
                for ar in anchors["aspect_ratios"]:
                    ar = sqrt(ar)
                    w = scale * ar / img_size
                    h = w if square_anchors else scale / ar / img_size
                    prior_data += [x, y, w, h]
        prior_data = np.reshape(prior_data, (-1, 4))
        priors.append(prior_data)
    return np.concatenate(priors, axis=-2)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nm=32, multi_label=True):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Args:
        prediction: numpy.ndarray with shape (batch_size, num_proposals, 351)
        conf_thres: confidence threshold for NMS
        iou_thres: IoU threshold for NMS
        max_det: Maximal number of detections to keep after NMS
        nm: Number of masks
        multi_label: Consider only best class per proposal or all conf_thresh passing proposals
    Returns:
         A list of per image detections, where each is a dictionary with the following structure:
         {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
         }
    """

    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append(
                {
                    "detection_boxes": np.zeros((0, 4)),
                    "mask": np.zeros((0, 32)),
                    "detection_classes": np.zeros((0, 80)),
                    "detection_scores": np.zeros((0, 80)),
                }
            )
            continue

        # Confidence = Objectness X Class Score
        x[:, 5:] *= x[:, 4:5]

        # (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]

        multi_label &= nc > 1
        if not multi_label:
            conf = np.expand_dims(x[:, 5:mi].max(1), 1)
            j = np.expand_dims(x[:, 5:mi].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, mask), 1)[keep]
        else:
            i, j = (x[:, 5:mi] > conf_thres).nonzero()
            x = np.concatenate((boxes[i], x[i, 5 + j, None], j[:, None].astype(np.float32), mask[i]), 1)

        # sort by confidence
        x = x[x[:, 4].argsort()[::-1]]

        # per-class NMS
        cls_shift = x[:, 5:6] * max_wh
        boxes = x[:, :4] + cls_shift
        conf = x[:, 4:5]
        preds = np.hstack([boxes.astype(np.float32), conf.astype(np.float32)])

        keep = cnms(preds, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        out = x[keep]
        scores = out[:, 4]
        classes = out[:, 5]
        boxes = out[:, :4]
        masks = out[:, 6:]

        out = {"detection_boxes": boxes, "mask": masks, "detection_classes": classes, "detection_scores": scores}

        output.append(out)

    return output


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def crop_mask(masks, boxes):
    """
    Zeroing out mask region outside of the predicted bbox.
    Args:
        masks: numpy array of masks with shape [n, h, w]
        boxes: numpy array of bbox coords with shape [n, 4]
    """

    n_masks, _, _ = masks.shape
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)
    for k in range(n_masks):
        masks[k, : y1[k, 0], :] = 0
        masks[k, y2[k, 0] :, :] = 0
        masks[k, :, : x1[k, 0]] = 0
        masks[k, :, x2[k, 0] :] = 0
    return masks


def process_mask(protos, masks_in, bboxes, shape, upsample=True, downsample=False):
    mh, mw, c = protos.shape
    ih, iw = shape
    masks = _sigmoid(masks_in @ protos.reshape((-1, c)).transpose((1, 0))).reshape((-1, mh, mw))

    downsampled_bboxes = bboxes.copy()
    if downsample:
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih

        masks = crop_mask(masks, downsampled_bboxes)

    if upsample:
        if not masks.shape[0]:
            return None
        masks = cv2.resize(np.transpose(masks, axes=(1, 2, 0)), shape, interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[..., np.newaxis]
        masks = np.transpose(masks, axes=(2, 0, 1))  # CHW

    if not downsample:
        masks = crop_mask(masks, downsampled_bboxes)  # CHW

    return masks


def _yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes):
    BS, H, W = output.shape[0:3]
    stride = stride_list[branch_idx]
    anchors = anchor_list[branch_idx] / stride
    num_anchors = len(anchors) // 2

    grid, anchor_grid = _make_grid(anchors, stride, BS, W, H)

    output = output.transpose((0, 3, 1, 2)).reshape((BS, num_anchors, -1, H, W)).transpose((0, 1, 3, 4, 2))
    xy, wh, conf, mask = np.array_split(output, [2, 4, 4 + num_classes + 1], axis=4)

    # decoding
    xy = (_sigmoid(xy) * 2 + grid) * stride
    wh = (_sigmoid(wh) * 2) ** 2 * anchor_grid

    out = np.concatenate((xy, wh, _sigmoid(conf), mask), 4)
    out = out.reshape((BS, num_anchors * H * W, -1)).astype(np.float32)

    return out


def yolov5_seg_postprocess(endnodes, device_pre_post_layers=None, **kwargs):
    """
    endnodes is a list of 4 tensors:
        endnodes[0]:  mask protos with shape (BS, 160, 160, 32)
        endnodes[1]:  stride 32 of input with shape (BS, 20, 20, 351)
        endnodes[2]:  stride 16 of input with shape (BS, 40, 40, 351)
        endnodes[3]:  stride 8 of input with shape (BS, 80, 80, 351)
    Returns:
        A list of per image detections, where each is a dictionary with the following structure:
        {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 32),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
        }
    """
    img_dims = kwargs["img_dims"]
    if kwargs.get("hpp", False):
        # the outputs where decoded by the emulator (as part of the network)
        # organizing the output for evaluation
        return _organize_hpp_yolov5_seg_outputs(endnodes, img_dims=img_dims)

    protos = endnodes[0]
    outputs = []
    anchor_list = np.array(kwargs["anchors"]["sizes"][::-1])
    stride_list = kwargs["anchors"]["strides"][::-1]
    num_classes = kwargs["classes"]

    outputs = []
    for branch_idx, output in enumerate(endnodes[1:]):
        decoded_info = _yolov5_decoding(branch_idx, output, stride_list, anchor_list, num_classes)
        outputs.append(decoded_info)

    outputs = np.concatenate(outputs, 1)  # (BS, num_proposals, 117)

    # NMS
    score_thres = kwargs["score_threshold"]
    iou_thres = kwargs["nms_iou_thresh"]
    outputs = non_max_suppression(outputs, score_thres, iou_thres, nm=protos.shape[-1])
    outputs = _finalize_detections_yolov5_seg(outputs, protos, **kwargs)

    # reorder and normalize bboxes
    for output in outputs:
        output["detection_boxes"] = _normalize_yolov5_seg_bboxes(output, img_dims)
    return outputs


def _organize_hpp_yolov5_seg_outputs(outputs, img_dims):
    # the outputs structure is [-1, num_of_proposals, 6 + h*w]
    # were the mask information is ordered as follows
    # [x_min, y_min, x_max, y_max, score, class, flattened mask] of each detection
    # this function separates the structure to informative dict
    predictions = []
    batch_size, num_of_proposals = outputs.shape[0], outputs.shape[-1]
    outputs = np.transpose(np.squeeze(outputs, axis=1), [0, 2, 1])
    for i in range(batch_size):
        predictions.append(
            {
                "detection_boxes": outputs[i, :, :4][:, [1, 0, 3, 2]],
                "detection_scores": outputs[i, :, 4],
                "detection_classes": outputs[i, :, 5],
                "mask": outputs[i, :, 6:].reshape((num_of_proposals, *img_dims)),
            }
        )
    return predictions


def _normalize_yolov5_seg_bboxes(output, img_dims):
    # normalizes bboxes and change the bboxes format to y_min, x_min, y_max, x_max
    bboxes = output["detection_boxes"]
    bboxes[:, [0, 2]] /= img_dims[1]
    bboxes[:, [1, 3]] /= img_dims[0]

    return bboxes


def _finalize_detections_yolov5_seg(outputs, protos, **kwargs):
    for batch_idx, output in enumerate(outputs):
        shape = kwargs.get("img_dims", None)
        boxes = output["detection_boxes"]
        masks = output["mask"]
        proto = protos[batch_idx]

        masks = process_mask(proto, masks, boxes, shape, upsample=True)
        output["mask"] = masks

    return outputs


def _make_grid(anchors, stride, bs=8, nx=20, ny=20):
    na = len(anchors) // 2
    y, x = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing="ij")

    grid = np.stack((xv, yv), 2)
    grid = np.stack([grid for _ in range(na)], 0) - 0.5
    grid = np.stack([grid for _ in range(bs)], 0)

    anchor_grid = np.reshape(anchors * stride, (na, -1))
    anchor_grid = np.stack([anchor_grid for _ in range(ny)], axis=1)
    anchor_grid = np.stack([anchor_grid for _ in range(nx)], axis=2)
    anchor_grid = np.stack([anchor_grid for _ in range(bs)], 0)

    return grid, anchor_grid


def sparseinst_postprocess(endnodes, device_pre_post_layers=None, scale_factor=2, num_groups=4, **kwargs):
    inst_kernels_path = path_resolver.resolve_data_path(kwargs["postprocess_config_file"])
    meta_arch = kwargs.get("meta_arch", "sparseinst_giam")
    inst_kernels = np.load(inst_kernels_path, allow_pickle=True)["arr_0"][()]

    mask_features = endnodes[0].copy()  # 80 x 80 x 128
    features = endnodes[1].copy()  # 80 x 80 x 256
    iam = endnodes[2].copy()  # 80 x 80 x 100
    iam_prob = _sigmoid(iam)

    B, H, W, N = iam_prob.shape
    iam_prob = np.reshape(iam_prob, (B, H * W, N))
    iam_prob_trans = np.transpose(iam_prob, axes=[0, 2, 1])
    C = features.shape[-1]
    features = np.reshape(features, (B, H * W, C))

    inst_features = []
    for batch_idx in range(B):
        np.expand_dims(np.matmul(iam_prob_trans[batch_idx], features[batch_idx]), axis=0)
        # for each and every batch element
        inst_features.append(np.expand_dims(np.matmul(iam_prob_trans[batch_idx], features[batch_idx]), axis=0))
    inst_features = np.vstack(inst_features)

    normalizer = np.clip(np.sum(iam_prob, axis=1), a_min=1e-6, a_max=None)
    inst_features /= normalizer[:, :, None]

    if "giam" in meta_arch:
        inst_features = inst_features.reshape(B, num_groups, -1, C)
        inst_features = inst_features.transpose(0, 2, 1, 3)
        inst_features = inst_features.reshape((B, -1, num_groups * C))

        features = []
        for batch_idx in range(B):
            features.append(
                np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels["fc"]["weights"].transpose()), axis=0)
            )
        inst_features = np.vstack(features)

        inst_features = inst_features + inst_kernels["fc"]["bias"]
        inst_features[inst_features < 0.0] = 0.0

    pred_logits = []
    pred_kernel = []
    pred_scores = []
    for batch_idx in range(B):
        pred_scores.append(np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels["obj"]["weights"]), axis=0))
        pred_kernel.append(
            np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels["mask_kernel"]["weights"]), axis=0)
        )
        pred_logits.append(
            np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels["cls_score"]["weights"]), axis=0)
        )
    pred_scores = np.vstack(pred_scores) + inst_kernels["obj"]["bias"]
    pred_kernel = np.vstack(pred_kernel) + inst_kernels["mask_kernel"]["bias"]
    pred_logits = np.vstack(pred_logits) + inst_kernels["cls_score"]["bias"]

    pred_masks = []
    C = mask_features.shape[-1]
    for batch_idx in range(B):
        pred_masks.append(
            np.expand_dims(
                np.matmul(
                    pred_kernel[batch_idx], np.transpose(mask_features.reshape(B, H * W, C), axes=[0, 2, 1])[batch_idx]
                ),
                axis=0,
            )
        )
    N = pred_kernel.shape[1]
    pred_masks = np.vstack(pred_masks).reshape(B, N, H, W)
    pred_masks_tmp = np.zeros((B, N, H * scale_factor, W * scale_factor))

    for i, _ in enumerate(pred_masks):
        pred_masks_tmp[i] = np.transpose(
            cv2.resize(
                np.transpose(pred_masks[i], axes=(1, 2, 0)),
                (H * scale_factor, W * scale_factor),
                interpolation=cv2.INTER_LINEAR,
            ),
            axes=(2, 0, 1),
        )
    pred_masks = np.vstack(pred_masks_tmp).reshape(B, N, H * scale_factor, W * scale_factor)

    pred_objectness = _sigmoid(pred_scores)
    pred_scores = _sigmoid(pred_logits)
    pred_masks = _sigmoid(pred_masks)
    pred_scores = np.sqrt(pred_scores * pred_objectness)

    return _finalize_detections_sparseinst(pred_masks, pred_scores, **kwargs)


def _finalize_detections_sparseinst(pred_masks, pred_scores, **kwargs):
    img_info = kwargs["gt_images"]
    hin, win = img_info["img_orig"].shape[1:3]
    cls_threshold = kwargs.get("score_threshold", 0.005)
    mask_threshold = kwargs.get("mask_threshold", 0.45)

    outputs = []
    for idx, (scores_per_image, masks_per_image) in enumerate(zip(pred_scores, pred_masks)):
        output = {}
        hout, wout = img_info["height"][idx], img_info["width"][idx]
        h_resized, w_resized = img_info["resized_height"][idx], img_info["resized_width"][idx]

        scores = np.max(scores_per_image, axis=-1)
        labels = np.argmax(scores_per_image, axis=-1)
        keep = scores > cls_threshold
        scores = scores[keep]
        labels = labels[keep]
        masks_per_image = masks_per_image[keep]

        if not scores.shape[0]:
            output["detection_scores"] = scores
            output["detection_classes"] = labels
            outputs.append(output)
            continue

        scores = _rescoring_mask(scores, masks_per_image > mask_threshold, masks_per_image)

        # (1) upsampling the masks to input size, remove the padding area
        masks_per_image = np.transpose(
            cv2.resize(np.transpose(masks_per_image, axes=(2, 1, 0)), (hin, win), interpolation=cv2.INTER_LINEAR),
            axes=(2, 1, 0),
        )[:, :h_resized, :w_resized]

        # (2) upsampling/downsampling the masks to the original sizes
        masks_per_image = np.transpose(
            cv2.resize(np.transpose(masks_per_image, axes=(2, 1, 0)), (hout, wout), interpolation=cv2.INTER_LINEAR),
            axes=(2, 1, 0),
        )
        output["mask"] = masks_per_image
        output["detection_scores"] = scores
        output["detection_classes"] = labels
        output["orig_shape"] = img_info["height"][idx], img_info["width"][idx]
        output["resized_shape"] = img_info["resized_height"][idx], img_info["resized_width"][idx]
        outputs.append(output)

    return outputs


def _rescoring_mask(scores, masks_pred, masks):
    return scores * (np.sum(masks * masks_pred, axis=(1, 2)) / (np.sum(masks_pred, axis=(1, 2)) + 1e-6))


def mask_to_polygons(mask):
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x + 0.5 for x in res if len(x) >= 6]
    return res, has_holes


def _get_pol_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def visualize_sparseinst_results(
    detections, img, class_names=None, alpha=0.5, confidence_threshold=0.5, mask_thresh=0.45, **kwargs
):
    img_idx = 0
    img_out = img[img_idx].copy()

    output_shape = detections["orig_shape"]
    resized_shape = detections["resized_shape"]
    keep = detections["detection_scores"] > confidence_threshold
    scores = detections["detection_scores"][keep]
    classes = detections["detection_classes"][keep]
    masks = detections["mask"][keep]

    # Binarize the masks
    masks = masks > mask_thresh

    # remove padding
    img_out = img_out[: resized_shape[0], : resized_shape[1], :]
    img_out = cv2.resize(img_out, output_shape[::-1], interpolation=cv2.INTER_LINEAR)

    for idx, mask in enumerate(masks):
        label = f"{CLASS_NAMES_COCO[classes[idx]]}"

        if not np.sum(mask):
            continue
        color = np.random.randint(low=0, high=255, size=3, dtype=np.uint8)
        polygons, _ = mask_to_polygons(mask)

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color
        color = [int(c) for c in color]

        # Draw mask
        img_out = cv2.addWeighted(mask, alpha, img_out, 1, 0)

        # Draw mask contour
        pol_areas = []
        for pol in polygons:
            pol_areas.append(_get_pol_area(pol[::2], pol[1::2]))
            img_out = cv2.polylines(
                img_out, [pol.reshape((-1, 1, 2)).astype(np.int32)], isClosed=True, color=color, thickness=2
            )

        # Draw class and score info
        score = "{:.0f}".format(100 * scores[idx])
        label = f"{CLASS_NAMES_COCO[classes[idx]]}"
        text = label + ": " + score + "%"
        x0, y0 = int(np.mean(polygons[np.argmax(pol_areas)][::2])), int(np.mean(polygons[np.argmax(pol_areas)][1::2]))
        (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        org = (x0 - w // 2, y0 - h // 2)
        # Black rectangle for label background
        deltaY = max(h - org[1], 0)
        deltaX = max(-org[0], 0)
        img_out[np.max([org[1] - h, 0]) : org[1] + h // 2 + deltaY, np.max([org[0], 0]) : org[0] + w + deltaX, :] = [
            0,
            0,
            0,
        ]

        img_out = cv2.putText(
            img_out,
            text,
            org=(max(org[0], 0), max(org[1], 0)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=2,
            lineType=cv2.FILLED,
        )

    return img_out


def yolact_postprocessing(endnodes, device_pre_post_layers=None, score_thresh=0.2, crop_masks=True, **kwargs):
    channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
    if channels_remove:
        mask_list = list(np.where(np.array(kwargs["channels_remove"]["mask"][0]) == 0)[0])
        num_classes = kwargs["classes"] - int(len(mask_list))
    else:
        num_classes = kwargs["classes"]
    priors = _make_priors(kwargs["anchors"], kwargs["img_dims"][0])
    proto, bbox0, mask0, conf0, bbox1, mask1, conf1, bbox2, mask2, conf2, bbox3, mask3, conf3, bbox4, mask4, conf4 = (
        endnodes
    )
    bbox0 = np.reshape(bbox0, [bbox0.shape[0], -1, 4])
    bbox1 = np.reshape(bbox1, [bbox1.shape[0], -1, 4])
    bbox2 = np.reshape(bbox2, [bbox2.shape[0], -1, 4])
    bbox3 = np.reshape(bbox3, [bbox3.shape[0], -1, 4])
    bbox4 = np.reshape(bbox4, [bbox4.shape[0], -1, 4])
    loc = np.concatenate([bbox0, bbox1, bbox2, bbox3, bbox4], axis=-2)
    conf0 = np.reshape(conf0, [conf0.shape[0], -1, num_classes])
    conf1 = np.reshape(conf1, [conf1.shape[0], -1, num_classes])
    conf2 = np.reshape(conf2, [conf2.shape[0], -1, num_classes])
    conf3 = np.reshape(conf3, [conf3.shape[0], -1, num_classes])
    conf4 = np.reshape(conf4, [conf4.shape[0], -1, num_classes])
    conf = _softmax(np.concatenate([conf0, conf1, conf2, conf3, conf4], axis=-2))
    mask0 = np.reshape(mask0, [mask0.shape[0], -1, 32])
    mask1 = np.reshape(mask1, [mask1.shape[0], -1, 32])
    mask2 = np.reshape(mask2, [mask2.shape[0], -1, 32])
    mask3 = np.reshape(mask3, [mask3.shape[0], -1, 32])
    mask4 = np.reshape(mask4, [mask4.shape[0], -1, 32])
    mask = np.concatenate([mask0, mask1, mask2, mask3, mask4], axis=-2)
    detect = Detect(
        num_classes, bkg_label=0, top_k=200, conf_thresh=kwargs["score_threshold"], nms_thresh=kwargs["nms_iou_thresh"]
    )

    det_output = detect(loc, proto, conf, mask, priors)

    return _finalize_detections_yolact(det_output, proto)


def _finalize_detections_yolact(det_output, protos, score_thresh=0.2, crop_masks=True, **kwargs):
    outputs = []
    for batch_idx, dets in enumerate(det_output):
        proto = protos[batch_idx]
        empty_output = {
            "detection_boxes": np.zeros((0, 4)),
            "mask": np.zeros((proto.shape[0], proto.shape[1], 0)),
            "detection_classes": np.zeros(0),
            "detection_scores": np.zeros(0),
        }

        if dets is None:
            outputs.append(empty_output)
            continue

        if score_thresh > 0:
            keep = dets["detection_scores"] > score_thresh
            for k in dets:
                if k != "proto":
                    dets[k] = dets[k][keep]

            if dets["detection_scores"].shape[0] == 0:
                outputs.append(empty_output)
                continue

        # Actually extract everything from dets now
        classes = dets["detection_classes"]
        boxes = dets["detection_boxes"]
        scores = dets["detection_scores"]
        masks = dets["mask"]

        # At this points masks is only the coefficients
        proto_data = dets["proto"]

        # Test flag, do not upvote

        masks = np.matmul(proto_data, masks.transpose())
        masks = _sigmoid(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = _crop(masks, boxes)

        output = {}
        output["detection_boxes"] = boxes
        output["mask"] = masks
        output["detection_scores"] = scores
        output["detection_classes"] = classes
        outputs.append(output)

    return outputs


def _yolov8_decoding(raw_boxes, strides, image_dims, reg_max):
    boxes = None
    for box_distribute, stride in zip(raw_boxes, strides):
        # create grid
        shape = [int(x / stride) for x in image_dims]
        grid_x = np.arange(shape[1]) + 0.5
        grid_y = np.arange(shape[0]) + 0.5
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        ct_row = grid_y.flatten() * stride
        ct_col = grid_x.flatten() * stride
        center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

        # box distribution to distance
        reg_range = np.arange(reg_max + 1)
        box_distribute = np.reshape(
            box_distribute, (-1, box_distribute.shape[1] * box_distribute.shape[2], 4, reg_max + 1)
        )
        box_distance = _softmax(box_distribute)
        box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
        box_distance = np.sum(box_distance, axis=-1)
        box_distance = box_distance * stride

        # decode box
        box_distance = np.concatenate([box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]], axis=-1)
        decode_box = np.expand_dims(center, axis=0) + box_distance

        xmin = decode_box[:, :, 0]
        ymin = decode_box[:, :, 1]
        xmax = decode_box[:, :, 2]
        ymax = decode_box[:, :, 3]
        decode_box = np.transpose([xmin, ymin, xmax, ymax], [1, 2, 0])

        xywh_box = np.transpose([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], [1, 2, 0])
        boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=1)
    return boxes  # tf.expand_dims(boxes, axis=2)


def yolov8_seg_postprocess(endnodes, device_pre_post_layers=None, **kwargs):
    """
    endnodes is a list of 10 tensors:
        endnodes[0]:  bbox output with shapes (BS, 20, 20, 64)
        endnodes[1]:  scores output with shapes (BS, 20, 20, 80)
        endnodes[2]:  mask coeff output with shapes (BS, 20, 20, 32)
        endnodes[3]:  bbox output with shapes (BS, 40, 40, 64)
        endnodes[4]:  scores output with shapes (BS, 40, 40, 80)
        endnodes[5]:  mask coeff output with shapes (BS, 40, 40, 32)
        endnodes[6]:  bbox output with shapes (BS, 80, 80, 64)
        endnodes[7]:  scores output with shapes (BS, 80, 80, 80)
        endnodes[8]:  mask coeff output with shapes (BS, 80, 80, 32)
        endnodes[9]:  mask protos with shape (BS, 160, 160, 32)
    Returns:
        A list of per image detections, where each is a dictionary with the following structure:
        {
            'detection_boxes':   numpy.ndarray with shape (num_detections, 4),
            'mask':              numpy.ndarray with shape (num_detections, 160, 160),
            'detection_classes': numpy.ndarray with shape (num_detections, 80),
            'detection_scores':  numpy.ndarray with shape (num_detections, 80)
        }
    """
    num_classes = kwargs["classes"]
    strides = kwargs["anchors"]["strides"][::-1]
    image_dims = tuple(kwargs["img_dims"])
    reg_max = kwargs["anchors"]["regression_length"]
    raw_boxes = endnodes[:7:3]
    scores = [np.reshape(s, (-1, s.shape[1] * s.shape[2], num_classes)) for s in endnodes[1:8:3]]
    scores = np.concatenate(scores, axis=1)
    outputs = []
    decoded_boxes = _yolov8_decoding(raw_boxes, strides, image_dims, reg_max)
    score_thres = kwargs["score_threshold"]
    iou_thres = kwargs["nms_iou_thresh"]
    proto_data = endnodes[9]
    batch_size, _, _, n_masks = proto_data.shape

    # add objectness=1 for working with yolov5_nms
    fake_objectness = np.ones((scores.shape[0], scores.shape[1], 1))
    scores_obj = np.concatenate([fake_objectness, scores], axis=-1)

    coeffs = [np.reshape(c, (-1, c.shape[1] * c.shape[2], n_masks)) for c in endnodes[2:9:3]]
    coeffs = np.concatenate(coeffs, axis=1)

    # re-arrange predictions for yolov5_nms
    predictions = np.concatenate([decoded_boxes, scores_obj, coeffs], axis=2)

    nms_res = non_max_suppression(predictions, conf_thres=score_thres, iou_thres=iou_thres, multi_label=True)
    masks = []
    outputs = []
    for b in range(batch_size):
        protos = proto_data[b]
        masks = process_mask(protos, nms_res[b]["mask"], nms_res[b]["detection_boxes"], image_dims, upsample=True)
        output = {}
        output["detection_boxes"] = np.array(nms_res[b]["detection_boxes"]) / np.tile(image_dims, 2)
        if masks is not None:
            output["mask"] = np.transpose(masks, (0, 1, 2))
        else:
            output["mask"] = masks
        output["detection_scores"] = np.array(nms_res[b]["detection_scores"])
        output["detection_classes"] = np.array(nms_res[b]["detection_classes"]).astype(int)
        outputs.append(output)
    return outputs


@POSTPROCESS_FACTORY.register(name="instance_segmentation")
def instance_segmentation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs.get("meta_arch", "")
    if "sparseinst" in meta_arch:
        predictions = sparseinst_postprocess(endnodes, device_pre_post_layers=device_pre_post_layers, **kwargs)
    elif "yolov5_seg" in meta_arch:
        predictions = yolov5_seg_postprocess(endnodes, device_pre_post_layers=device_pre_post_layers, **kwargs)
    elif "yolact" in meta_arch:
        predictions = yolact_postprocessing(endnodes, device_pre_post_layers=device_pre_post_layers, **kwargs)
    elif "yolov8_seg" in meta_arch:
        predictions = yolov8_seg_postprocess(endnodes, device_pre_post_layers=device_pre_post_layers, **kwargs)
    else:
        raise NotImplementedError(f"Postprocessing {meta_arch} not found")
    return {"predictions": predictions}


def visualize_yolov5_seg_results(
    detections, img, class_names=None, alpha=0.5, score_thres=0.25, mask_thresh=0.5, max_boxes_to_draw=20, **kwargs
):
    img_idx = 0
    img_out = img[img_idx].copy()

    boxes = detections["detection_boxes"]

    # scales the box to input shape
    boxes[:, 0::2] *= img_out.shape[1]
    boxes[:, 1::2] *= img_out.shape[0]

    masks = detections["mask"] > mask_thresh
    scores = detections["detection_scores"]
    classes = detections["detection_classes"]
    # for SAM model we want to skip boxes and draw only the masks (single class)
    skip_boxes = kwargs.get("meta_arch", "") == "yolov8_seg_postprocess" and kwargs.get("classes", "") == 1

    keep = scores > score_thres
    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    classes = classes[keep]

    # take only the best max_boxes_to_draw (proposals are sorted)
    max_boxes = min(max_boxes_to_draw, len(keep))
    boxes = boxes[:max_boxes]
    masks = masks[:max_boxes]
    scores = scores[:max_boxes]
    classes = classes[:max_boxes]

    for idx, mask in enumerate(masks):
        xmin, ymin, xmax, ymax = boxes[idx].astype(np.int32)

        color = np.random.randint(low=0, high=255, size=3, dtype=np.uint8)
        # Draw bbox
        if not skip_boxes:
            img_out = cv2.rectangle(img_out, (xmin, ymin), (xmax, ymax), [int(c) for c in color], 3)

        if not np.sum(mask):
            continue
        polygons, _ = mask_to_polygons(mask)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) * color
        color = [int(c) for c in color]

        # Draw mask
        img_out = cv2.addWeighted(mask, alpha, img_out, 1, 0)

        # Draw mask contour
        pol_areas = []
        for pol in polygons:
            pol_areas.append(_get_pol_area(pol[::2], pol[1::2]))
            img_out = cv2.polylines(
                img_out, [pol.reshape((-1, 1, 2)).astype(np.int32)], isClosed=True, color=color, thickness=1
            )

        # Draw class and score info
        if not skip_boxes:
            label = f"{CLASS_NAMES_COCO[int(classes[idx])]}"
            score = "{:.0f}".format(100 * scores[idx])

            text = label + ": " + score + "%"
            (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
            org = (xmin, ymin)
            # Rectangle for label background
            deltaY = max(h - org[1], 0)
            deltaX = max(-org[0], 0)
            img_out[
                np.max([org[1] - h, 0]) : org[1] + h // 2 + deltaY, np.max([org[0], 0]) : org[0] + w + deltaX, :
            ] = color

            img_out = cv2.putText(
                img_out,
                text,
                org=(xmin, ymin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=[255, 255, 255],
                thickness=2,
                lineType=cv2.FILLED,
            )

    return img_out


@VISUALIZATION_FACTORY.register(name="instance_segmentation")
def visualize_instance_segmentation_result(detections, img, **kwargs):
    detections = detections["predictions"]
    meta_arch = kwargs.get("meta_arch", "")
    dataset_name = kwargs.get("dataset_name", None)
    dataset_info = get_dataset_info(dataset_name=dataset_name)

    if "sparseinst" in meta_arch:
        return visualize_sparseinst_results(detections, img, class_names=dataset_info.class_names, **kwargs)
    elif "yolov5_seg" or "yolov8_seg" in meta_arch:
        return visualize_yolov5_seg_results(detections, img, class_names=dataset_info.class_names, **kwargs)
    elif "yolact" in meta_arch:
        channels_remove = kwargs["channels_remove"]
        return prep_display(
            dets_out=detections,
            img=img[0],
            score_threshold=0.2,
            channels_remove=channels_remove,
            class_names=dataset_info.class_names,
        )
    else:
        raise NotImplementedError(f"Visualization for {meta_arch} is not implemented")
