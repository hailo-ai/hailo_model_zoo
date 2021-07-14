import tensorflow as tf


class BBoxCornerToCenter(object):
    def __init__(self, axis=-1, split=False):
        self._split = split
        self._axis = axis

    def __call__(self, bboxes_corners):
        xmin, ymin, xmax, ymax = tf.split(bboxes_corners, 4, axis=self._axis)
        width = xmax - xmin
        height = ymax - ymin
        cx = xmin + width * 0.5
        cy = ymin + height * 0.5
        if not self._split:
            return tf.concat([cx, cy, width, height], axis=-1)
        else:
            return cx, cy, width, height


class NormalizedBoxCenterDecoder(object):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), convert_anchor=False, clip=None):
        self._stds = stds
        self._clip = clip
        if convert_anchor:
            self._corner_to_center = BBoxCornerToCenter(split=True)
        else:
            self._corner_to_center = False

    def __call__(self, bboxes, anchors):
        if self._corner_to_center is not None:
            cx, cy, width, height = self._corner_to_center(anchors)
        else:
            cx, cy, width, height = tf.split(anchors, 4, axis=-1)
        p = tf.split(bboxes, 4, axis=-1)
        ox = tf.squeeze(p[0]) * self._stds[0] * width + cx
        oy = tf.squeeze(p[1]) * self._stds[1] * height + cy
        dw = p[2] * self._stds[2]
        dh = p[3] * self._stds[3]
        if self._clip:
            dw = tf.minimum(dw, self._clip)
            dh = tf.minimum(dh, self._clip)
        dw = tf.exp(dw)
        dh = tf.exp(dh)

        ow = tf.squeeze(dw) * width * 0.5
        oh = tf.squeeze(dh) * height * 0.5

        return tf.stack([ox - ow, oy - oh, ox + ow, oy + oh], axis=2)


class FasterRCNNStage2(object):
    def __init__(self, classes, **kwargs):
        self._num_classes = classes

    def postprocessing(self, endnodes, image_info, **kwargs):
        rpn_boxes = image_info['rpn_proposals']
        num_roi = tf.shape(rpn_boxes)[0]
        cls_preds = endnodes[0]
        box_preds = endnodes[1]
        bboxes_preds = tf.reshape(box_preds, shape=(num_roi, self._num_classes, 4))
        cls_preds = tf.nn.softmax(cls_preds, axis=-1)

        box_decoder = NormalizedBoxCenterDecoder(convert_anchor=True, clip=4.42)
        decoded_boxes = box_decoder(bboxes_preds, rpn_boxes)
        detection_boxes = decoded_boxes
        proposals, classes = tf.split(tf.shape(cls_preds), 2)
        detection_scores = tf.expand_dims(tf.slice(cls_preds, begin=[0, 1],
                                                   size=[proposals[0], classes[0] - 1]), axis=0)
        detection_scores = tf.reshape(detection_scores, shape=[-1, self._num_classes])
        return {'detection_scores': detection_scores, 'detection_boxes': detection_boxes,
                'image_id': image_info['image_id']}
