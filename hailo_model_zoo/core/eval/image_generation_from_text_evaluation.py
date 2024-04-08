from collections import OrderedDict

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY


@EVAL_FACTORY.register(name="stable_diffusion_v2_decoder")
@EVAL_FACTORY.register(name="stable_diffusion_v2_unet")
class ImagegenerationFromTextEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['snr_db', 'clip_score', 'FID_score']
        self._metrics_vals = [0, 0, 0]
        self.clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        self.fid = FrechetInceptionDistance(normalize=True, reset_real_features=False)
        self.reset()

    def is_percentage(self):
        return False

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, img_info):
        net_output = self._parse_net_output(net_output)

        # SNR calc
        gt = img_info['img_float'].squeeze(1)
        snr = np.mean(gt**2, axis=(1, 2, 3)) / np.mean((net_output - gt)**2, axis=(1, 2, 3))
        self.snr_list.append(10 * np.log10(snr))

        # clip score
        for prompt, curr_pred, curr_gt in zip(img_info['prompt'], net_output, gt):
            prompt = prompt.decode('utf-8')
            curr_pred_ = curr_pred.transpose(2, 0, 1) * 255
            curr_pred_ = curr_pred_.astype("uint8")
            curr_pred_ = torch.from_numpy(curr_pred_)
            self.clip_metric.update(curr_pred_, prompt)

            # FID score
            self.fid.update(torch.from_numpy(np.expand_dims(curr_gt.transpose(2, 0, 1), 0)), real=True)
            self.fid.update(torch.from_numpy(np.expand_dims(curr_pred.transpose(2, 0, 1), 0)), real=False)

    def evaluate(self):
        self._metrics_vals[0] = np.mean(self.snr_list)
        self._metrics_vals[1] = np.asarray(self.clip_metric.score / self.clip_metric.n_samples)
        self._metrics_vals[2] = np.asarray(self.fid.compute())

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1]),
                            (self._metric_names[2], self._metrics_vals[2])])

    def reset(self):
        self.snr_list = []
        self.clip_metric.reset()
        self.fid.reset()
