import numpy as np
import cv2
import tensorflow as tf


class PolyLaneNetPostProcessHailo(object):
    def __init__(self):
        return

    def recombine_split_endnodes(self, confs, upper_lower, coeffs_1_2, coeffs_3_4):
        bs = confs.shape[0]
        output = np.zeros((bs, 5, 7))
        for lane in range(5):
            output[:, lane, 0:1] = confs[:, 1 * lane:1 * lane + 1]
            output[:, lane, 1:3] = upper_lower[:, 2 * lane:2 * lane + 2]
            output[:, lane, 3:5] = coeffs_1_2[:, 2 * lane:2 * lane + 2]
            output[:, lane, 5:7] = coeffs_3_4[:, 2 * lane:2 * lane + 2]
        return output.astype(np.float32)

    def sigmoid(self, x):
        return (np.exp(x)) / (np.exp(x) + 1.0)

    def enforce_shared_y(self, pred):
        pred = pred.reshape(-1, 5, 7)
        pred_lowers = pred[:, :, 1]
        first_lowers = pred_lowers[:, 0]
        first_lowers = np.expand_dims(first_lowers, 1)
        first_lowers = np.repeat(first_lowers, 5, axis=1)
        pred[:, :, 1] = first_lowers
        return pred

    def decode(self, outputs, conf_threshold=0.5):
        outputs = self.enforce_shared_y(outputs)
        outputs[:, :, 0] = self.sigmoid(outputs[:, :, 0])
        outputs[outputs[:, :, 0] < conf_threshold] = 0
        return outputs

    def polynomize_pred(self, pred):
        pred = pred[0]  # [0] zero is because it had to be given as a list
        batch_lanes = []
        for image in range(pred.shape[0]):
            # running over images in batch:
            lanes = []
            for lane_index in range(pred.shape[1]):
                confidence = pred[image, lane_index, 0]
                lower = pred[image, lane_index, 1]
                upper = pred[image, lane_index, 2]
                xvals = (np.polyval(pred[image, lane_index, 3:], self.h_range) * self.img_w)
                xvals[self.h_range < lower] = -2.
                xvals[self.h_range > upper] = -2.
                xvals = np.append(xvals, confidence)
                lanes.append(xvals.astype(np.float32))
            batch_lanes.append(lanes)
        return [batch_lanes]


def visualize_lane_detection_result(pred, im, dataset_name='tusimple', **kwargs):
    pred = pred['predictions']
    color = [0, 255, 0]
    conf = pred[0, :, -1]   # last element in each lane is confidence
    pred = pred[:, :, :-1]  # removing conf from pred for ease of expression
    pred = pred[0].astype(int)
    im = im[0]
    overlay = im.copy()
    ypoints = np.arange(im.shape[0])
    for i in range(pred.shape[0]):  # going over lanes in image:
        if not np.any(pred[i, :] > 0):
            # all the x values of this lane are non-positive
            continue
        lane_confidence = conf[i]
        xpoints = pred[i, :]
        lane_horizon_y = ypoints[xpoints > 0][0]
        points = [[ypoints[j], xpoints[j]] for j in range(len(xpoints))]
        for current_point in points:
            if current_point[1] > 0 and current_point[1] < im.shape[1]:
                overlay = cv2.circle(overlay, (current_point[1], current_point[0]), radius=2, color=color, thickness=2)
        cv2.putText(overlay,
                    str('{:.1f}'.format(lane_confidence)),
                    (int(xpoints[ypoints == lane_horizon_y]), int(lane_horizon_y)),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0))
    w = 0.6
    im = ((1. - w) * im + w * overlay).astype(np.uint8)
    return im


def lane_detection_postprocessing(endnodes, device_pre_post_layers=None, img_w=1280,
                                  img_h=720, output_scheme=None, **kwargs):
    lane_detection_postproc = PolyLaneNetPostProcessHailo()
    lane_detection_postproc.h_range_int = np.arange(img_h)
    lane_detection_postproc.h_range = lane_detection_postproc.h_range_int / (img_h - 1)
    lane_detection_postproc.img_w = img_w
    lane_detection_postproc.img_h = img_h
    if output_scheme and output_scheme.get('split_output', False):
        endnodes = tf.compat.v1.py_func(lane_detection_postproc.recombine_split_endnodes, endnodes, [tf.float32])
    decoded = tf.compat.v1.py_func(lane_detection_postproc.decode, [endnodes], [tf.float32])
    postprocessed = tf.compat.v1.py_func(lane_detection_postproc.polynomize_pred,
                                         [decoded], [tf.float32])  # network always returns 5 lane predictions.
    return {'predictions': postprocessed[0]}
