from collections import OrderedDict

import cv2
import numpy as np

from hailo_model_zoo.core.eval.eval_base_class import Eval
from hailo_model_zoo.core.factory import EVAL_FACTORY

DATASETS_INFO = {
    "nyu_depth_v2": {"min_depth": 0.1, "max_depth": 10, "crop_array": np.array([45, 471, 41, 601]).astype(np.int32)},
    "kitti_depth": {
        "min_depth": 1e-3,
        "max_depth": 80,
        "crop_array": np.array([0.40810811 * 375, 0.99189189 * 375, 0.03594771 * 1242, 0.96405229 * 1242]).astype(
            np.int32
        ),
    },
}


@EVAL_FACTORY.register(name="depth_estimation")
class DepthEstimationEval(Eval):
    """
    DepthEstimationEval is a class to evaluate depth estimation models.

    Attributes:
        meta_arch (str): Name of the meta architecture being evaluated (lowercased).
        full_errors (list): List to store depth error metrics for all samples.
        avg (None or numpy array): Variable to store the average depth error metrics.
        min_depth (float): Minimum valid depth value for the dataset.
        max_depth (float): Maximum valid depth value for the dataset.
        crop (numpy array): Array specifying crop dimensions for the dataset.

    """

    def __init__(self, **kwargs):
        """
        Initialize the DepthEvaluation class with configuration settings.

        Args:
            **kwargs: Variable keyword arguments containing various configuration settings.
        """
        self.meta_arch = kwargs["meta_arch"].lower()
        self.full_errors = []
        self.avg = None

        dataset_name = kwargs["dataset_name"].lower()
        dataset_info = DATASETS_INFO[dataset_name]
        self.min_depth = dataset_info["min_depth"]
        self.max_depth = dataset_info["max_depth"]
        self.crop = dataset_info["crop_array"]

    def _parse_net_output(self, net_output):
        """
        Parse the network output and return the depth predictions.

        Args:
            net_output (dict): Network output containing 'predictions' key.

        Returns:
            numpy array: Depth predictions.
        """
        return net_output["predictions"]

    def is_percentage(self):
        """
        Check if the evaluation metrics are percentages.

        Returns:
            bool: False, as the depth error metrics are not percentages.
        """
        return False

    def is_bigger_better(self):
        """
        Check if higher evaluation metric values are better.

        Returns:
            bool: False, as lower values of depth error metrics are better.
        """
        return False

    def compute_depth_errors(self, gt, pred):
        """
        Compute various depth error metrics between ground truth and predicted depth maps.

        Args:
            gt (numpy array): Ground truth depth map.
            pred (numpy array): Predicted depth map.

        Returns:
            tuple: Tuple containing different depth error metrics
            (rmse, rmse_log, abs_rel, sq_rel, log10, delta1, delta2, delta3).
        """

        # Compute the threshold ratio between ground truth and predicted depth
        thresh = np.maximum((gt / pred), (pred / gt))

        delta1 = (thresh < 1.25).mean()  # fraction of threshold ratios below 1.25
        delta2 = (thresh < 1.25**2).mean()  # fraction of threshold ratios below 1.25^2
        delta3 = (thresh < 1.25**3).mean()  # fraction of threshold ratios below 1.25^3

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())  # root mean squared error
        rmse_log = (np.log(gt + 1e-10) - np.log(pred + 1e-10)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())  # root mean squared error in the logarithm domain

        abs_rel = np.mean(np.abs(gt - pred) / gt)  # mean absolute relative error
        sq_rel = np.mean(((gt - pred) ** 2) / gt)  # squared relative error
        log10 = np.mean(
            np.abs((np.log10(gt + 1e-10) - np.log10(pred + 1e-10)))
        )  # mean absolute error in the logarithm domain (log10)

        return rmse, rmse_log, abs_rel, sq_rel, log10, delta1, delta2, delta3

    def get_valid_depthmaps(self, gt, pred):
        """
        Get valid depth maps by applying crop and range filters.

        Args:
            gt (numpy array): Ground truth depth map.
            pred (numpy array): Predicted depth map.

        Returns:
            tuple: Tuple containing the filtered ground truth and predicted depth maps.
        """
        if self.meta_arch == "fast_depth":
            # For fast_depth, only use a mask to filter out invalid depth values (<= 0).
            valid_mask = ((gt > 0) + (pred > 0)) > 0
            return gt[valid_mask], pred[valid_mask]
        # For other meta_archs:
        gt, pred = self.apply_valid_masks(gt, pred)

        # Align scales
        ratio = np.median(gt) / np.median(pred)
        pred *= ratio

        # Clip values in predicted depth map to be within [self.min_depth, self.max_depth]
        np.clip(pred, self.min_depth, self.max_depth, out=pred)

        return gt, pred

    def apply_valid_masks(self, gt, pred):
        # Resize the predicted depth map to the same dimensions as the ground truth depth map (gt)
        gt_h, gt_w = gt.shape[:2]
        pred = cv2.resize(pred, (gt_w, gt_h))

        # Create a crop_mask to identify regions to be cropped out from the depth maps
        crop_mask = np.isnan(gt)
        crop_mask[self.crop[0] : self.crop[1], self.crop[2] : self.crop[3]] = 1
        # Create a range mask to filter out depth values outside [self.min_depth, self.max_depth]
        range_mask = np.logical_and(gt > self.min_depth, gt < self.max_depth)
        # Combine range mask and crop mask
        valid_mask = np.logical_and(range_mask, crop_mask)

        # Apply the mask
        return gt[valid_mask], pred[valid_mask]

    def update_op(self, net_output, img_info):
        """
        Update the evaluation with predictions from the network output and ground truth.

        Args:
            net_output (dict): Network output containing 'predictions' key.
            img_info (dict): Image information dictionary containing 'depth' key with ground truth depth maps.
        """
        pred = self._parse_net_output(net_output)
        gt = img_info["depth"]

        # Loop through each sample in the batch.
        for i in range(pred.shape[0]):
            # Get valid ground truth and predicted depth maps by applying crop and range filters
            valid_gt, valid_pred = self.get_valid_depthmaps(gt[i], pred[i])
            # Compute depth errors
            self.full_errors.append(self.compute_depth_errors(valid_gt, valid_pred))

    def evaluate(self):
        """
        Calculate the average depth errors over all instances.
        """
        self.avg = np.mean(self.full_errors, axis=0)

    def _get_accuracy(self):
        """
        Return the computed depth error metrics in an ordered dictionary.

        Returns:
            OrderedDict: Dictionary containing depth error metrics with keys and their corresponding values.
        """
        return OrderedDict(
            [
                ("rmse", self.avg[0]),
                ("rmse_log", self.avg[1]),
                ("abs_rel", self.avg[2]),
                ("sq_rel", self.avg[3]),
                ("log10", self.avg[4]),
                ("delta1", self.avg[5]),
                ("delta2", self.avg[6]),
                ("delta3", self.avg[7]),
            ]
        )

    def reset(self):
        """
        Reset the full_errors list for the next evaluation.
        """
        self.full_errors = []


@EVAL_FACTORY.register(name="zero_shot_depth_estimation")
class ZeroShotDepthEstimationEval(DepthEstimationEval):
    """
    ZeroShotDepthEstimationEval is a class to evaluate zero-shot depth estimation models.

    Attributes:
        meta_arch (str): Name of the meta architecture being evaluated (lowercased).
        full_errors (list): List to store depth error metrics for all samples.
        avg (None or numpy array): Variable to store the average depth error metrics.
        min_depth (float): Minimum valid depth value for the dataset.
        max_depth (float): Maximum valid depth value for the dataset.
        crop (numpy array): Array specifying crop dimensions for the dataset.

    """

    def get_valid_depthmaps(self, gt, pred):
        """
        Get valid depth maps by applying crop and range filters.

        Args:
            gt (numpy array): Ground truth depth map.
            pred (numpy array): Predicted depth map.

        Returns:
            tuple: Tuple containing the filtered ground truth and predicted depth maps.
        """
        gt, pred = self.apply_valid_masks(gt, pred)

        scale, shift = self.compute_scale_and_shift(pred, 1 / gt)
        pred_aligned = scale * pred + shift
        prediction_depth = 1 / pred_aligned

        # Clip values in predicted depth map to be within [self.min_depth, self.max_depth]
        np.clip(prediction_depth, self.min_depth, self.max_depth, out=prediction_depth)

        return gt, prediction_depth

    def compute_scale_and_shift(self, prediction, ground_truth):
        """
        Calculate scale and shift for prediction compare to ground truth using linear regression

        Parameters:
        - prediction: np.array, prediction after masking (N,)
        - target: np.array, ground truth after masking (N,)

        Returns:
        - scale: scale for predictions
        - shift: shift for predictions
        """
        a_00 = np.sum(prediction * prediction)
        a_01 = np.sum(prediction)
        a_11 = np.prod(prediction.shape)

        b_0 = np.sum(prediction * ground_truth)
        b_1 = np.sum(ground_truth)
        det = a_00 * a_11 - a_01 * a_01

        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det

        return scale, shift

    def _get_accuracy(self):
        """
        Return the computed depth error metrics in an ordered dictionary.

        Returns:
            OrderedDict: Dictionary containing depth error metrics with keys and their corresponding values.
        """
        return OrderedDict(
            [
                ("abs_rel", self.avg[2]),
                ("rmse", self.avg[0]),
                ("rmse_log", self.avg[1]),
                ("sq_rel", self.avg[3]),
                ("log10", self.avg[4]),
                ("delta1", self.avg[5]),
                ("delta2", self.avg[6]),
                ("delta3", self.avg[7]),
            ]
        )
