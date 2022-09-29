import numpy as np
from collections import OrderedDict
import cv2

from hailo_model_zoo.core.eval.eval_base_class import Eval


class SRGANEval(Eval):
    def __init__(self, **kwargs):
        self._metric_names = ['psnr', 'ssim']
        self._normalize_results = kwargs.get('normalize_results', True)
        self._metrics_vals = [0, 0]
        self.reset()

    def shave(self, image, height, width, margin=4):
        start_w = max(margin, (image.shape[1] - width) // 2 + margin)
        start_h = max(margin, (image.shape[0] - height) // 2 + margin)
        end_w = min(image.shape[1] - margin, image.shape[1] - (image.shape[1] - width) // 2 - margin)
        end_h = min(image.shape[0] - margin, image.shape[0] - (image.shape[0] - height) // 2 - margin)
        new_image = image[start_h:end_h, start_w:end_w, :]
        return new_image

    def convert_rgb_to_y(self, image):
        if len(image.shape) <= 2 or image.shape[2] == 1:
            return image
        xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
        y_image = image.dot(xform.T) + 16.0
        return y_image

    def _parse_net_output(self, net_output):
        return net_output['predictions']

    def update_op(self, net_output, gt_labels):
        net_output = self._parse_net_output(net_output)
        self._psnr += self.evaluate_psnr(net_output, gt_labels['hr_img'], gt_labels['height'], gt_labels['width'])
        self._ssim += self.evaluate_ssim(net_output, gt_labels['hr_img'], gt_labels['height'], gt_labels['width'])

    def evaluate(self):
        normalize_factor = 100.0 if self._normalize_results else 1.0
        self._metrics_vals[0] = np.mean(self._psnr) / normalize_factor
        self._metrics_vals[1] = np.mean(self._ssim) / normalize_factor

    def _get_accuracy(self):
        return OrderedDict([(self._metric_names[0], self._metrics_vals[0]),
                            (self._metric_names[1], self._metrics_vals[1])])

    def reset(self):
        self._psnr = []
        self._ssim = []

    def evaluate_psnr(self, y_pred, y_true, height, width):
        """
        ###  TF implementaion:  ###
        PSNR is Peak Signal to Noise Ratio, which is similar to mean squared error.
        It can be calculated as
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
        When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
        Args:
           y_true: float32 numpy array, the original (high resolution) image batch.
           y_pred: float32 numpy array, the super-resolution network output.
        Returns:
            PSNR: float32 scalar, the performance metric on that batch.

        """
        psnr_list = []
        for image_num in range(y_pred.shape[0]):
            y_pred_single_im = self.shave(y_pred[image_num], height[image_num], width[image_num], margin=4)
            y_true_single_im = self.shave(y_true[image_num], height[image_num], width[image_num], margin=4)
            y_pred_single_im = self.convert_rgb_to_y(y_pred_single_im)
            y_true_single_im = self.convert_rgb_to_y(y_true_single_im)
            psnr_list.append(48.1308 - 10. * np.log10(np.mean(np.square(y_pred_single_im - y_true_single_im))))
        return psnr_list

    def evaluate_ssim(self, y_pred, y_true, height, width):
        """
        calculate structural similarity index between 2 images.
        Args:
            img1, img2 - float32 numpy arrays of the same dimensions (channel dimension allowed to be 1).
            input images are assumed to use scale 0 - 255.
        Returns:
            ssim_map.mean() - float32 scalar.

        """
        C1 = (0.01 * 255.0) ** 2
        C2 = (0.03 * 255.0) ** 2
        ssim_list = []
        for image_num in range(y_pred.shape[0]):
            y_pred_resized = self.shave(y_pred[image_num], height[image_num], width[image_num], margin=4)
            y_true_resized = self.shave(y_true[image_num], height[image_num], width[image_num], margin=4)
            img1 = self.convert_rgb_to_y(y_pred_resized)
            img2 = self.convert_rgb_to_y(y_true_resized)
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())
            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_list.append(ssim_map.mean())
        return ssim_list
