import numpy as np
import tensorflow as tf


class PolyLaneNetPostProcessHailo(object):
    def __init__(self, **kwargs):
        self.img_h = 720
        self.img_w = 1280
        self.h_range_int = np.arange(self.img_h)
        self.h_range = self.h_range_int / (self.img_h - 1)
        return

    def recombine_split_endnodes(self, confs, upper_lower, coeffs_1_2, coeffs_3_4):
        bs = confs.shape[0]
        output = np.zeros((bs, 5, 7))
        for lane in range(5):
            output[:, lane, 0:1] = confs[:, 1 * lane : 1 * lane + 1]
            output[:, lane, 1:3] = upper_lower[:, 2 * lane : 2 * lane + 2]
            output[:, lane, 3:5] = coeffs_1_2[:, 2 * lane : 2 * lane + 2]
            output[:, lane, 5:7] = coeffs_3_4[:, 2 * lane : 2 * lane + 2]
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
                xvals = np.polyval(pred[image, lane_index, 3:], self.h_range) * self.img_w
                xvals[self.h_range < lower] = -2.0
                xvals[self.h_range > upper] = -2.0
                xvals = np.append(xvals, confidence)
                lanes.append(xvals.astype(np.float32))
            batch_lanes.append(lanes)
        return [batch_lanes]

    def postprocessing(self, endnodes, device_pre_post_layers=None, output_scheme=None, **kwargs):
        if output_scheme and output_scheme.get("split_output", False):
            endnodes = tf.numpy_function(self.recombine_split_endnodes, endnodes, [tf.float32])
        decoded = tf.numpy_function(self.decode, [endnodes], [tf.float32])
        # network always returns 5 lane predictions.
        postprocessed = tf.numpy_function(self.polynomize_pred, [decoded], [tf.float32])
        return {"predictions": postprocessed[0]}
