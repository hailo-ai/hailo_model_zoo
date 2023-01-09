import numpy as np
from itertools import product
from math import sqrt
import cv2

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info, CLASS_NAMES_COCO
from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.core.postprocessing.cython_utils.cython_nms import nms as cnms

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


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        max_det=300, nm=32, multi_label=True):
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

    assert 0 <= conf_thres <= 1, \
        f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, \
        f'Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0'

    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680  # (pixels) maximum box width and height
    mi = 5 + nc  # mask start index
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            output.append({'detection_boxes': np.zeros((0, 4)),
                           'mask': np.zeros((0, 32)),
                           'detection_classes': np.zeros((0, 80)),
                           'detection_scores': np.zeros((0, 80))})
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

        out = {'detection_boxes': boxes,
               'mask': masks,
               'detection_classes': classes,
               'detection_scores': scores}

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

    n_masks, h, w = masks.shape
    x1, y1, x2, y2 = np.array_split(boxes[:, :, None], 4, axis=1)
    rows = np.arange(w)[None, None, :]
    cols = np.arange(h)[None, :, None]

    return masks * ((rows >= x1) * (rows < x2) * (cols >= y1) * (cols < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=True,
                 downsample=False):
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

    return masks > 0.5


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

    protos = endnodes[0]
    outputs = list()
    anchor_list = np.array(kwargs['anchors']['sizes'][::-1])
    stride_list = kwargs['anchors']['strides'][::-1]
    num_classes = kwargs['classes']

    outputs = []
    for branch_idx, output in enumerate(endnodes[1:]):
        decoded_info = _yolov5_decoding(branch_idx,
                                        output,
                                        stride_list,
                                        anchor_list,
                                        num_classes)
        outputs.append(decoded_info)

    outputs = np.concatenate(outputs, 1)  # (BS, num_proposals, 117)

    # NMS
    score_thres = kwargs['score_threshold']
    iou_thres = kwargs['nms_iou_thresh']
    outputs = non_max_suppression(outputs, score_thres, iou_thres)

    for batch_idx, output in enumerate(outputs):
        shape = kwargs.get('img_dims', None)
        boxes = output['detection_boxes']
        masks = output['mask']
        proto = protos[batch_idx]

        masks = process_mask(proto, masks, boxes, shape, upsample=True)
        output['mask'] = masks

    return outputs


def _make_grid(anchors, stride, bs=8, nx=20, ny=20):
    na = len(anchors) // 2
    y, x = np.arange(ny), np.arange(nx)
    yv, xv = np.meshgrid(y, x, indexing='ij')

    grid = np.stack((xv, yv), 2)
    grid = np.stack([grid for _ in range(na)], 0) - 0.5
    grid = np.stack([grid for _ in range(bs)], 0)

    anchor_grid = np.reshape(anchors * stride, (na, -1))
    anchor_grid = np.stack([anchor_grid for _ in range(ny)], axis=1)
    anchor_grid = np.stack([anchor_grid for _ in range(nx)], axis=2)
    anchor_grid = np.stack([anchor_grid for _ in range(bs)], 0)

    return grid, anchor_grid


def sparseinst_postprocess(endnodes, device_pre_post_layers=None, scale_factor=2, **kwargs):

    inst_kernels_path = path_resolver.resolve_data_path(kwargs['postprocess_config_json'])
    inst_kernels = np.load(inst_kernels_path, allow_pickle=True)['arr_0'][()]

    mask_features = endnodes[0].copy()  # 80 x 80 x 128
    features = endnodes[1].copy()       # 80 x 80 x 256
    iam = endnodes[2].copy()            # 80 x 80 x 100
    iam_prob = _sigmoid(iam)

    B, H, W, N = iam_prob.shape
    iam_prob = np.reshape(iam_prob, (B, H * W, N))
    iam_prob_trans = np.transpose(iam_prob, axes=[0, 2, 1])
    C = features.shape[-1]
    features = np.reshape(features, (B, H * W, C))

    inst_features = list()
    for batch_idx in range(B):
        np.expand_dims(np.matmul(iam_prob_trans[batch_idx], features[batch_idx]), axis=0)
        # for each and every batch element
        inst_features.append(np.expand_dims(np.matmul(iam_prob_trans[batch_idx],
                                                      features[batch_idx]), axis=0))
    inst_features = np.vstack(inst_features)

    normalizer = np.clip(np.sum(iam_prob, axis=1), a_min=1e-6, a_max=None)
    inst_features /= normalizer[:, :, None]

    pred_logits = list()
    pred_kernel = list()
    pred_scores = list()
    for batch_idx in range(B):
        pred_scores.append(np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels['obj']['weights']), axis=0))
        pred_kernel.append(np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels['mask_kernel']['weights']),
                                          axis=0))
        pred_logits.append(np.expand_dims(np.matmul(inst_features[batch_idx], inst_kernels['cls_score']['weights']),
                                          axis=0))
    pred_scores = np.vstack(pred_scores) + inst_kernels['obj']['bias']
    pred_kernel = np.vstack(pred_kernel) + inst_kernels['mask_kernel']['bias']
    pred_logits = np.vstack(pred_logits) + inst_kernels['cls_score']['bias']

    pred_masks = list()
    C = mask_features.shape[-1]
    for batch_idx in range(B):
        pred_masks.append(np.expand_dims(
                          np.matmul(pred_kernel[batch_idx],
                                    np.transpose(mask_features.reshape(B, H * W, C), axes=[0, 2, 1])[batch_idx]),
                          axis=0))
    pred_masks = np.vstack(pred_masks).reshape(B, N, H, W)
    pred_masks_tmp = np.zeros((B, N, H * scale_factor, W * scale_factor))

    for i, _ in enumerate(pred_masks):
        pred_masks_tmp[i] = np.transpose(cv2.resize(np.transpose(pred_masks[i], axes=(1, 2, 0)),
                                                    (H * scale_factor, W * scale_factor),
                                                    interpolation=cv2.INTER_LINEAR), axes=(2, 0, 1))
    pred_masks = np.vstack(pred_masks_tmp).reshape(B, N, H * scale_factor, W * scale_factor)

    pred_scores = _sigmoid(pred_logits)
    pred_masks = _sigmoid(pred_masks)
    pred_objectness = _sigmoid(pred_scores)
    pred_scores = np.sqrt(pred_scores * pred_objectness)

    img_info = kwargs['image_info']
    orig_height, orig_width = img_info['orig_height'], img_info['orig_width']
    output_shapes = (orig_height, orig_width)

    output = {'pred_scores': pred_scores, 'pred_masks': pred_masks, 'output_shapes': output_shapes}

    return output


def _spraseinst_post(output, input_shape, output_shape,
                     cls_threshold=0.005, mask_threshold=0.45):
    pred_scores = output['pred_scores']
    pred_masks = output['pred_masks']
    results = []
    for idx, (scores_per_image, masks_per_image) in enumerate(zip(pred_scores, pred_masks)):
        result = {}
        hout, wout = output_shape[0][idx], output_shape[1][idx]
        hin, win = input_shape[0][idx], input_shape[1][idx]

        scores = np.max(scores_per_image, axis=-1)
        labels = np.argmax(scores_per_image, axis=-1)
        keep = scores > cls_threshold
        scores = scores[keep]
        labels = labels[keep]
        masks_per_image = masks_per_image[keep]

        if not scores.shape[0]:
            result['scores'] = scores
            result['pred_classes'] = labels
            results.append(result)
            continue

        scores = _rescoring_mask(scores, masks_per_image > mask_threshold, masks_per_image)

        # (1) upsampling the masks to input size, remove the padding area
        masks_per_image = np.transpose(cv2.resize(np.transpose(masks_per_image, axes=(2, 1, 0)),
                                       (hin, win), interpolation=cv2.INTER_LINEAR),
                                       axes=(2, 1, 0))[:, :hout, :wout]
        # (2) upsampling/downsampling the masks to the original sizes
        masks_per_image = np.transpose(cv2.resize(np.transpose(masks_per_image, axes=(2, 1, 0)),
                                       (hout, wout), interpolation=cv2.INTER_LINEAR),
                                       axes=(2, 1, 0))
        mask_pred = masks_per_image > mask_threshold

        result['pred_masks'] = mask_pred
        result['scores'] = scores
        result['pred_classes'] = labels

        results.append(result)

    return results


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


def visualize_sparseinst_results(detections, img, class_names=None, alpha=0.5, confidence_threshold=0.5, **kwargs):

    output_shapes = detections['output_shapes']
    input_shapes = ([im.shape[0] for im in img], [im.shape[1] for im in img])
    detections = _spraseinst_post(detections, input_shapes, input_shapes)

    img_idx = 0
    results = detections[img_idx]
    output_shape = output_shapes[0][img_idx], output_shapes[1][img_idx]
    scores = results['scores']
    keep = scores > confidence_threshold

    masks = results['pred_masks'][keep]
    scores = results['scores'][keep]
    classes = results['pred_classes'][keep]
    img_out = img[img_idx].copy()
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
            img_out = cv2.polylines(img_out, [pol.reshape((-1, 1, 2)).astype(np.int32)],
                                    isClosed=True, color=color, thickness=2)

        # Draw class and score info
        score = "{:.0f}".format(100 * scores[idx])
        label = f"{CLASS_NAMES_COCO[classes[idx]]}"
        text = label + ': ' + score + '%'
        x0, y0 = int(np.mean(polygons[np.argmax(pol_areas)][::2])), int(np.mean(polygons[np.argmax(pol_areas)][1::2]))
        (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=2)
        org = (x0 - w // 2, y0 - h // 2)
        # Black rectangle for label backround
        deltaY = max(h - org[1], 0)
        deltaX = max(-org[0], 0)
        img_out[np.max([org[1] - h, 0]):org[1] + h // 2 + deltaY,
                np.max([org[0], 0]):org[0] + w + deltaX, :] = [0, 0, 0]

        img_out = cv2.putText(img_out, text,
                              org=(max(org[0], 0), max(org[1], 0)),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.5,
                              color=color,
                              thickness=2,
                              lineType=cv2.FILLED)

    # remove padding
    img_out = img_out[:output_shape[0], :output_shape[1], :]
    return img_out


def yolact_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
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
    return detect(loc, proto, conf, mask, priors)


def instance_segmentation_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    meta_arch = kwargs.get('meta_arch', '')
    if 'sparseinst' in meta_arch:
        predictions = sparseinst_postprocess(endnodes,
                                             device_pre_post_layers=device_pre_post_layers,
                                             **kwargs)
    elif 'yolov5_seg' in meta_arch:
        predictions = yolov5_seg_postprocess(endnodes,
                                             device_pre_post_layers=device_pre_post_layers,
                                             **kwargs)
    elif 'yolact' in meta_arch:
        predictions = yolact_postprocessing(endnodes,
                                            device_pre_post_layers=device_pre_post_layers,
                                            **kwargs)
    else:
        raise NotImplementedError(f'Postprocessing {meta_arch} not found')
    return {'predictions': predictions}


def visualize_yolov5_seg_results(detections, img, class_names=None, alpha=0.5, score_thres=0.25, **kwargs):
    img_idx = 0
    det = detections[img_idx]
    img_out = img[img_idx].copy()

    boxes = det['detection_boxes']
    masks = det['mask']
    scores = det['detection_scores']
    classes = det['detection_classes']

    keep = scores > score_thres
    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]
    classes = classes[keep]

    for idx, mask in enumerate(masks):
        xmin, ymin, xmax, ymax = boxes[idx].astype(np.int32)

        color = np.random.randint(low=0, high=255, size=3, dtype=np.uint8)
        # Draw bbox
        img_out = cv2.rectangle(img_out,
                                (xmin, ymin),
                                (xmax, ymax),
                                [int(c) for c in color],
                                3)

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
            img_out = cv2.polylines(img_out, [pol.reshape((-1, 1, 2)).astype(np.int32)],
                                    isClosed=True, color=color, thickness=1)

        # Draw class and score info
        label = f"{CLASS_NAMES_COCO[int(classes[idx])]}"
        score = "{:.0f}".format(100 * scores[idx])

        text = label + ': ' + score + '%'
        (w, h), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=2)
        org = (xmin, ymin)
        # Rectangle for label backround
        deltaY = max(h - org[1], 0)
        deltaX = max(-org[0], 0)
        img_out[np.max([org[1] - h, 0]):org[1] + h // 2 + deltaY,
                np.max([org[0], 0]):org[0] + w + deltaX, :] = color

        img_out = cv2.putText(img_out, text,
                              org=(xmin, ymin),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=0.5,
                              color=[255, 255, 255],
                              thickness=2,
                              lineType=cv2.FILLED)

    return img_out


def visualize_instance_segmentation_result(detections, img, **kwargs):
    detections = detections['predictions']
    meta_arch = kwargs.get('meta_arch', '')
    dataset_name = kwargs.get('dataset_name', None)
    dataset_info = get_dataset_info(dataset_name=dataset_name)

    if 'sparseinst' in meta_arch:
        return visualize_sparseinst_results(detections, img, class_names=dataset_info.class_names, **kwargs)
    elif 'yolov5_seg' in meta_arch:
        return visualize_yolov5_seg_results(detections, img, class_names=dataset_info.class_names, **kwargs)
    elif 'yolact' in meta_arch:
        channels_remove = kwargs["channels_remove"]
        return prep_display(dets_out=detections[0:1], img=img[0], score_threshold=0.2,
                            channels_remove=channels_remove, class_names=dataset_info.class_names)
    else:
        raise NotImplementedError(f'Visualization {meta_arch} not found')
