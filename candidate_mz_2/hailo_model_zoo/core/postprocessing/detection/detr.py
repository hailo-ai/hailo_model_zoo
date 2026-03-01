import tensorflow as tf


class DetrPostProc(object):
    def __init__(self, **kwargs):
        pass

    def detr_postprocessing(self, endnodes, sigmoid=True, **kwargs):
        logits = tf.squeeze(endnodes[0], [1])  # [batch, 1, 100, 92] -> [batch, 100, 92]
        boxes = tf.squeeze(endnodes[1], [1])  # [batch, 1, 100, 4] -> [batch, 100, 4]
        if sigmoid:
            boxes = tf.nn.sigmoid(boxes)

        # SoftMax to get scores and class predictions
        probs = tf.nn.softmax(logits, axis=-1)  # [batch, queries, 92]
        labels = tf.math.argmax(probs[:, :, 0:-1], axis=-1)  # [batch, queries]
        scores = tf.gather(probs, labels, axis=2, batch_dims=2)  # [batch, queries] = sample probs at the label location

        # [Cx, Cy, W, H] -> [y_min, x_min, y_max, x_max]
        x_c, y_c, w, h = tf.split(boxes, 4, axis=2)  # [batch, 100, 1] [batch, 100, 1] [batch, 100, 1] [batch, 100, 1]
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        boxes = tf.concat([y_min, x_min, y_max, x_max], axis=2)  # [batch, 100, 4]

        # num detection -> [batch, 100]
        num_detections = tf.repeat(tf.constant([100]), tf.shape(boxes)[0], axis=0)

        return {
            "detection_boxes": boxes,
            "detection_scores": scores,
            "detection_classes": labels,
            "num_detections": num_detections,
        }

    def postprocessing(self, endnodes, **kwargs):
        return self.detr_postprocessing(endnodes, **kwargs)
