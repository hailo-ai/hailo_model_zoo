import math

import numpy as np

from hailo_model_zoo.core.eval.super_resolution_evaluation import SuperResolutionEval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="image_denoising")
class ImageDenoisingEval(SuperResolutionEval):
    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self._psnr += self.evaluate_psnr(net_output, gt_labels['img'], gt_labels['height'], gt_labels['width'])

    def evaluate_psnr(self, y_pred, y_true, height, width):
        psnr_list = []
        for image_num in range(y_pred.shape[0]):
            y_pred_single_im = y_pred[image_num].astype(np.float64)
            y_true_single_im = y_true[image_num].astype(np.float64)
            psnr = 20 * math.log10(255.0 / math.sqrt(np.mean((y_pred_single_im - y_true_single_im) ** 2)))
            psnr_list.append(psnr)
        return psnr_list

    def evaluate(self):
        self._metrics_vals[0] = sum(self._psnr) / (len(self._psnr) * 100)
