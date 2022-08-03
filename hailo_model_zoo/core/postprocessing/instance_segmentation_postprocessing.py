import numpy as np
from itertools import product
from math import sqrt
import cv2

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info

COLORS = ((244, 67, 54),
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
          (96, 125, 139))


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
    max_xy = np.minimum(np.expand_dims(box_a[:, :, 2:], axis=2),
                        np.expand_dims(box_b[:, :, 2:], axis=1))
    min_xy = np.maximum(np.expand_dims(box_a[:, :, :2], axis=2),
                        np.expand_dims(box_b[:, :, :2], axis=1))
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
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
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
            raise ValueError('nms_threshold must be non negative.')
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
                result['proto'] = proto_data[batch_idx]

            out.append(result)

        return out

    def _detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores = np.amax(cur_scores, axis=0)

        keep = (conf_scores > self._conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if scores.shape[1] == 0:
            return None

        boxes, masks, classes, scores = self._fast_nms(boxes, masks, scores, self._nms_thresh, self._top_k)

        return {'detection_boxes': boxes, 'mask': masks, 'detection_classes': classes, 'detection_scores': scores}

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
        keep = (iou_max <= iou_threshold)
        if second_threshold:
            keep *= (scores > self._conf_thresh)

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


def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0.2):
    dets = det_output[batch_idx]

    if dets is None:
        return [None] * 4  # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['detection_scores'] > score_threshold
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]

        if dets['detection_scores'].shape[0] == 0:
            return [None] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Actually extract everything from dets now
    classes = dets['detection_classes']
    boxes = dets['detection_boxes']
    scores = dets['detection_scores']
    masks = dets['mask']

    # At this points masks is only the coefficients
    proto_data = dets['proto']

    # Test flag, do not upvote

    masks = np.matmul(proto_data, masks.transpose())
    masks = _sigmoid(masks)

    # Crop masks before upsampling because you know why
    if crop_masks:
        masks = _crop(masks, boxes)

    # Scale masks up to the full image
    masks = cv2.resize(masks, (w, h))
    if len(masks.shape) < 3:
        masks = np.expand_dims(masks, axis=0)
    else:
        masks = np.transpose(masks, (2, 0, 1))

    # Binarize the masks
    masks = masks > 0.5

    boxes[:, 0], boxes[:, 2] = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = np.array(boxes, np.float64)

    return classes, scores, boxes, masks


def prep_display(dets_out, img, score_threshold, class_color=False, mask_alpha=0.45,
                 channels_remove=None, class_names=None):
    top_k = 5
    img_gpu = img / 255.0
    h, w, _ = img.shape
    if not channels_remove["enabled"]:
        visualization_class_names = class_names
    else:
        channels_remove = np.array(channels_remove['mask'][0])
        class_names_mask = channels_remove[1:]  # Remove background class
        cats = np.where(np.array(class_names_mask) == 1)[0]
        visualization_class_names = list(np.array(class_names)[cats])
    t = postprocess(dets_out, w, h, visualize_lincomb=False,
                    crop_masks=True, score_threshold=score_threshold)
    if t[3] is None:
        return np.array(img_gpu * 255, np.uint8)
    masks = t[3][:top_k]
    classes, scores, boxes = [x[:top_k] for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255)

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
                color = 1.0 * color / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    masks = masks[:num_dets_to_consider, :, :, None]

    # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
    colors = np.concatenate([np.reshape(get_color(j, on_gpu=None),
                             (1, 1, 1, 3)) for j in range(num_dets_to_consider)], axis=0)
    masks_color = np.repeat(masks, 3, axis=-1) * colors * mask_alpha

    # This is 1 everywhere except for 1-mask_alpha where the mask is
    inv_alph_masks = masks * (-mask_alpha) + 1
    masks_color_summand = masks_color[0]
    if num_dets_to_consider > 1:
        inv_alph_cumul = np.cumprod(inv_alph_masks[:(num_dets_to_consider - 1)], axis=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += np.sum(masks_color_cumul, axis=0)

    img_gpu = img_gpu * np.prod(inv_alph_masks, axis=0) + masks_color_summand
    img_numpy = (img_gpu * 255)

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        score = scores[j]
        color = list(get_color(j))
        # if args.display_bboxes:
        cv2.rectangle(img_numpy, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        _class = visualization_class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        text_pt = (x1, y1 - 3)

        cv2.rectangle(img_numpy, (int(x1), int(y1)), (int(x1 + text_w), int(y1 - text_h - 4)), color, -1)
        cv2.putText(img_numpy, text_str, (int(text_pt[0]), int(text_pt[1])), font_face, font_scale,
                    [255., 255., 255.], font_thickness, cv2.LINE_AA)

    return np.array(img_numpy, np.uint8)


def _softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def _make_priors(anchors, img_size):
    priors = []
    square_anchors = True if len(anchors['scales'][0]) == 1 else False
    for conv_size, pred_scale in zip(anchors['feature_map'], anchors['scales']):
        prior_data = []
        for j, i in product(range(conv_size), range(conv_size)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_size
            y = (j + 0.5) / conv_size
            for scale in pred_scale:
                for ar in anchors['aspect_ratios']:
                    ar = sqrt(ar)
                    w = scale * ar / img_size
                    h = w if square_anchors else scale / ar / img_size
                    prior_data += [x, y, w, h]
        prior_data = np.reshape(prior_data, (-1, 4))
        priors.append(prior_data)
    return np.concatenate(priors, axis=-2)


def instance_segmentation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
    if channels_remove:
        mask_list = list(np.where(np.array(kwargs['channels_remove']['mask'][0]) == 0)[0])
        num_classes = kwargs['classes'] - int(len(mask_list))
    else:
        num_classes = kwargs['classes']
    priors = _make_priors(kwargs['anchors'], kwargs['img_dims'][0])
    proto, bbox0, mask0, conf0, bbox1, mask1, conf1, bbox2, mask2, \
        conf2, bbox3, mask3, conf3, bbox4, mask4, conf4 = endnodes
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
    detect = Detect(num_classes, bkg_label=0, top_k=200, conf_thresh=kwargs['score_threshold'],
                    nms_thresh=kwargs['nms_iou_thresh'])
    return {'predictions': detect(loc, proto, conf, mask, priors)}


def visualize_instance_segmentation_result(detections, img, **kwargs):
    detections = detections['predictions']
    channels_remove = kwargs["channels_remove"] if kwargs["channels_remove"]["enabled"] else None
    dataset_name = kwargs.get('dataset_name', None)
    dataset_info = get_dataset_info(dataset_name=dataset_name)
    return prep_display(dets_out=detections[0:1], img=img[0], score_threshold=0.2,
                        channels_remove=channels_remove, class_names=dataset_info.class_names)
