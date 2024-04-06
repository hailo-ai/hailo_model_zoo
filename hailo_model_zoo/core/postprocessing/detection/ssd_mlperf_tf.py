import math

import tensorflow as tf

from hailo_model_zoo.core.postprocessing.detection.detection_common import translate_coco_2017_to_2014


def center2point(center_x, center_y, width, height):
    return center_x - width / 2., center_y - height / 2., center_x + width / 2., center_y + height / 2.


def point2center(xmin, ymin, xmax, ymax):
    width, height = (xmax - xmin), (ymax - ymin)
    return xmin + width / 2., ymin + height / 2., width, height


class DefaultBoxes(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales,
                 extra_anchor_scales, anchor_ratios, layer_steps, clip=False):
        super(DefaultBoxes, self).__init__()
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)
        self._clip = clip

    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset=0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        with tf.name_scope('get_layer_anchors'):
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]
            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            num_anchors_along_depth = 2 * len(anchor_scale) * len(anchor_ratio) + \
                len(anchor_scale) + len(extra_anchor_scale)
            num_anchors_along_spatial = layer_shape[0] * layer_shape[1]
            list_w_on_image = []
            list_h_on_image = []
            for _, scales in enumerate(zip(anchor_scale, extra_anchor_scale)):
                list_w_on_image.append(scales[0])
                list_h_on_image.append(scales[0])
                list_w_on_image.append(math.sqrt(scales[0] * scales[1]))
                list_h_on_image.append(math.sqrt(scales[0] * scales[1]))
            for _, scale in enumerate(anchor_scale):
                for _, ratio in enumerate(anchor_ratio):
                    w, h = scale * math.sqrt(ratio), scale / math.sqrt(ratio),
                    list_w_on_image.append(w)
                    list_h_on_image.append(h)
                    list_w_on_image.append(h)
                    list_h_on_image.append(w)

            return tf.expand_dims(x_on_image, axis=-1), tf.expand_dims(y_on_image, axis=-1), \
                tf.constant(list_w_on_image, dtype=tf.float32), tf.constant(list_h_on_image, dtype=tf.float32), \
                num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        list_anchors_xmin = []
        list_anchors_ymin = []
        list_anchors_xmax = []
        list_anchors_ymax = []
        for ind, layer_shape in enumerate(self._layers_shapes):
            anchors_in_layer = self.get_layer_anchors(layer_shape,
                                                      self._anchor_scales[ind],
                                                      self._extra_anchor_scales[ind],
                                                      self._anchor_ratios[ind],
                                                      self._layer_steps[ind],
                                                      self._anchor_offset[ind])
            anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax = (
                center2point(anchors_in_layer[0], anchors_in_layer[1],
                             anchors_in_layer[2], anchors_in_layer[3]))
            anchors_xmin = tf.transpose(anchors_xmin, perm=[2, 0, 1])
            anchors_ymin = tf.transpose(anchors_ymin, perm=[2, 0, 1])
            anchors_xmax = tf.transpose(anchors_xmax, perm=[2, 0, 1])
            anchors_ymax = tf.transpose(anchors_ymax, perm=[2, 0, 1])
            list_anchors_xmin.append(tf.reshape(anchors_xmin, [-1]))
            list_anchors_ymin.append(tf.reshape(anchors_ymin, [-1]))
            list_anchors_xmax.append(tf.reshape(anchors_xmax, [-1]))
            list_anchors_ymax.append(tf.reshape(anchors_ymax, [-1]))
            all_num_anchors_depth.append(anchors_in_layer[-2])
            all_num_anchors_spatial.append(anchors_in_layer[-1])
        anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
        anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
        anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')
        anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
        if self._clip:
            anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
            anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
            anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)
            anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
        default_bboxes_ltrb = (anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax)
        anchor_cx, anchor_cy, anchor_w, anchor_h = point2center(anchors_xmin, anchors_ymin, anchors_xmax, anchors_ymax)
        default_bboxes = (anchor_cx, anchor_cy, anchor_w, anchor_h)

        return default_bboxes, default_bboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial


class En_Decoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling):
        super(En_Decoder, self).__init__()
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling

    def decode_all_anchors(self, pred_location, default_bboxes, num_anchors_per_layer):
        with tf.name_scope('decode_all_anchors'):
            anchor_cx, anchor_cy, anchor_w, anchor_h = default_bboxes
            pred_cx = pred_location[:, 0] * self._prior_scaling[0] * anchor_w + anchor_cx
            pred_cy = pred_location[:, 1] * self._prior_scaling[1] * anchor_h + anchor_cy
            pred_w = tf.exp(pred_location[:, 2] * self._prior_scaling[2]) * anchor_w
            pred_h = tf.exp(pred_location[:, 3] * self._prior_scaling[3]) * anchor_h

            return tf.stack(center2point(pred_cx, pred_cy, pred_w, pred_h), axis=-1)


def parse_by_class_fixed_bboxes(cls_pred, bboxes_pred, params):
    selected_bboxes, selected_scores = parse_by_class(
        cls_pred, bboxes_pred, params['num_classes'], params['select_threshold'],
        params['min_size'], params['keep_topk'], params['nms_topk'], params['nms_threshold'])
    pred_bboxes = []
    pred_scores = []
    pred_classes = []
    predictions = {}
    for class_ind in range(1, params['num_classes']):
        predictions['scores_{}'.format(class_ind)] = tf.expand_dims(selected_scores[class_ind], axis=0)
        predictions['bboxes_{}'.format(class_ind)] = tf.expand_dims(selected_bboxes[class_ind], axis=0)
        labels_mask = selected_scores[class_ind] > -0.5
        labels_mask = tf.cast(labels_mask, tf.float32)
        selected_labels = tf.cast(labels_mask * class_ind, tf.int32)
        pred_bboxes.append(selected_bboxes[class_ind])
        pred_scores.append(selected_scores[class_ind])
        pred_classes.append(selected_labels)
    detection_bboxes = tf.concat(pred_bboxes, axis=0)
    detection_scores = tf.concat(pred_scores, axis=0)
    detection_classes = tf.concat(pred_classes, axis=0)
    num_bboxes = tf.shape(detection_bboxes)[0]
    detection_scores, idxes = tf.nn.top_k(detection_scores, k=tf.minimum(
        params['keep_max_boxes'], num_bboxes), sorted=True)
    detection_bboxes = tf.gather(detection_bboxes, idxes)
    detection_classes = tf.gather(detection_classes, idxes)
    keep_max_boxes = tf.convert_to_tensor(params['keep_max_boxes'])

    cur_num = tf.shape(detection_classes)[0]
    detection_bboxes = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_bboxes, tf.zeros(
        shape=(params['keep_max_boxes'] - cur_num, 4), dtype=tf.float32)], axis=0), lambda: detection_bboxes)
    detection_scores = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_scores, tf.zeros(
        shape=(params['keep_max_boxes'] - cur_num,), dtype=tf.float32)], axis=0), lambda: detection_scores)
    detection_classes = tf.cond(cur_num < keep_max_boxes, lambda: tf.concat([detection_classes, tf.zeros(
        shape=(params['keep_max_boxes'] - cur_num,), dtype=tf.int32)], axis=0), lambda: detection_classes)
    num_detections = cur_num
    return detection_bboxes, detection_scores, detection_classes, num_detections


def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes'):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold
            select_mask = tf.cast(select_mask, tf.float32)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)
    return selected_bboxes, selected_scores


def clip_bboxes(xmin, ymin, xmax, ymax, name):
    with tf.name_scope(name):
        xmin = tf.maximum(xmin, 0.)
        ymin = tf.maximum(ymin, 0.)
        xmax = tf.minimum(xmax, 1.)
        ymax = tf.minimum(ymax, 1.)
        xmin = tf.minimum(xmin, xmax)
        ymin = tf.minimum(ymin, ymax)
        return xmin, ymin, xmax, ymax


def filter_bboxes(scores_pred, xmin, ymin, xmax, ymax, min_size, name):
    with tf.name_scope(name):
        width = xmax - xmin
        height = ymax - ymin
        filter_mask = tf.logical_and(width > min_size, height > min_size)
        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(xmin, filter_mask), tf.multiply(ymin, filter_mask), \
            tf.multiply(xmax, filter_mask), tf.multiply(ymax, filter_mask), tf.multiply(scores_pred, filter_mask)


def sort_bboxes(scores_pred, xmin, ymin, xmax, ymax, keep_topk, name):
    with tf.name_scope(name):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)
        xmin, ymin, xmax, ymax = tf.gather(xmin, idxes), tf.gather(
            ymin, idxes), tf.gather(xmax, idxes), tf.gather(ymax, idxes)
        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk - cur_bboxes, 0)], axis=0), axis=0)
        return tf.pad(xmin, paddings_scores, "CONSTANT"), tf.pad(ymin, paddings_scores, "CONSTANT"), \
            tf.pad(xmax, paddings_scores, "CONSTANT"), tf.pad(ymax, paddings_scores, "CONSTANT"), \
            tf.pad(scores, paddings_scores, "CONSTANT")


def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name):
        xmin, ymin, xmax, ymax = tf.unstack(bboxes_pred, 4, axis=-1)
        bboxes_pred = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        ymin, xmin, ymax, xmax = tf.unstack(bboxes_pred, 4, axis=-1)
        bboxes_pred = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)


def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes'):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            xmin, ymin, xmax, ymax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            # ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            # ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
            #                                    ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            xmin, ymin, xmax, ymax, selected_scores[class_ind] = (
                sort_bboxes(selected_scores[class_ind],
                            xmin, ymin, xmax, ymax, keep_topk, 'sort_bboxes_{}'.format(class_ind)))
            selected_bboxes[class_ind] = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = (
                nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind],
                           nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind)))
        return selected_bboxes, selected_scores


class SSDMLPerfPostProc:
    def __init__(self, **kwargs) -> None:
        pass

    def postprocessing(self, endnodes, **kwargs):

        # TODO magicke
        params = dict(num_classes=81, select_threshold=0.05, min_size=0.003,
                      keep_topk=200, nms_topk=200, nms_threshold=0.5, keep_max_boxes=200)
        match_threshold = 0.5
        neg_threshold = 0.5
        out_shape = (1200, 1200)
        defaultboxes_creator = DefaultBoxes(out_shape,
                                            layers_shapes=[(50, 50), (25, 25),
                                                           (13, 13), (7, 7), (3, 3), (3, 3)],
                                            anchor_scales=[(0.07,), (0.15,), (0.33,),
                                                           (0.51,), (0.69,), (0.87,)],
                                            extra_anchor_scales=[(0.15,), (0.33,), (0.51,), (0.69,), (0.87,), (1.05,)],
                                            anchor_ratios=[(2,), (2., 3.,), (2., 3.,), (2., 3.,), (2.,), (2.,)],
                                            layer_steps=[24, 48, 92, 171, 400, 400])
        defaultboxes, defaultboxes_ltrb, all_num_anchors_depth, all_num_anchors_spatial = (
            defaultboxes_creator.get_all_anchors())
        num_anchors_per_layer = []
        for ind in range(len(all_num_anchors_depth)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
        en_decoder = En_Decoder(allowed_borders=[1.0] * 6,
                                positive_threshold=match_threshold,
                                ignore_threshold=neg_threshold,
                                prior_scaling=[0.1, 0.1, 0.2, 0.2])

        def decode_fn(pred):
            return en_decoder.decode_all_anchors(pred, defaultboxes, num_anchors_per_layer)
        location_pred, cls_pred = endnodes[::2], endnodes[1::2]

        cls_pred = [tf.transpose(pred, [0, 3, 1, 2]) for pred in cls_pred]
        location_pred = [tf.transpose(pred, [0, 3, 1, 2]) for pred in location_pred]

        batch_size = tf.shape(endnodes[0])[0]
        cls_pred = [tf.reshape(pred, [batch_size, params['num_classes'], -1]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [batch_size, 4, -1]) for pred in location_pred]
        cls_pred = tf.concat(cls_pred, axis=2)
        location_pred = tf.concat(location_pred, axis=2)
        tf.identity(cls_pred, name='py_cls_pred')
        tf.identity(location_pred, name='py_location_pred')
        cls_pred = tf.transpose(cls_pred, [0, 2, 1])
        location_pred = tf.transpose(location_pred, [0, 2, 1])

        with tf.device('/cpu:0'):
            bboxes_pred = tf.map_fn(lambda _preds: decode_fn(_preds),
                                    location_pred,
                                    dtype=tf.float32, back_prop=False)
            # bboxes_pred = tf.concat(bboxes_pred, axis=1)

            def parse_bboxes_fn(x):
                return parse_by_class_fixed_bboxes(x[0], x[1], params)
            pred_results = tf.map_fn(parse_bboxes_fn, (cls_pred, bboxes_pred), dtype=(
                tf.float32, tf.float32, tf.int32, tf.int32), back_prop=False)

            predictions = {}
            detection_bboxes = tf.concat(pred_results[0], axis=0)
            detection_scores = tf.concat(pred_results[1], axis=0)
            detection_classes = tf.concat(pred_results[2], axis=0)
            num_detections = tf.concat(pred_results[3], axis=0)
            xmin, ymin, xmax, ymax = tf.unstack(detection_bboxes, axis=-1)
            detection_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            [detection_classes] = tf.numpy_function(translate_coco_2017_to_2014, [detection_classes], ['int32'])
            predictions['detection_classes'] = detection_classes
            predictions['detection_scores'] = detection_scores
            predictions['detection_boxes'] = detection_bboxes
            predictions['num_detections'] = num_detections

        return predictions
