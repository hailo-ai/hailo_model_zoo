import ctypes
import os
from ctypes import c_bool, c_double, c_int, c_void_p
from functools import partial

import numpy as np

import hailo_model_zoo.core


def roi_align_wrapper(
    featuremap,
    rpn_boxes,
    roi_callback,
    crop_size=(14, 14),
    spatial_scale=1 / 16.0,
    sampling_ratio=2,
    position_sensitive=False,
    continuous_coordinate=False,
    roi_cols=5,
):
    """
    This function is the wrapper to the c function used to perform roi align.
    The parameters are being casted to c types and given to the roi align c function.
    Inputs:
        featuremap - the image extracted featuremap - [B,H,W,C]
        rpn_boxes - the bboxes of the proposals - [N, 4]
    Output:
        top_data - ROI output  - in our case the shape would be [N, 14, 14, C]
    """
    height, width, channels = featuremap.shape
    f_transposed = np.expand_dims(featuremap, axis=0).transpose([0, 3, 1, 2]).copy()
    pooled_height = crop_size[0]
    pooled_width = crop_size[1]
    n_rois = rpn_boxes.shape[0]
    # Adding batch index - assuming the batch size is 1 we add zeros to the first column
    bottom_rois = np.concatenate([np.zeros(shape=(n_rois, 1), dtype=np.float32), rpn_boxes], axis=1)
    top_data = np.zeros(shape=(n_rois, channels, pooled_height, pooled_width), dtype=np.float32)
    roi_callback(
        c_int(n_rois),
        c_void_p(f_transposed.ctypes.data),
        c_double(spatial_scale),
        c_bool(position_sensitive),
        c_bool(continuous_coordinate),
        c_int(channels),
        c_int(height),
        c_int(width),
        c_int(pooled_height),
        c_int(pooled_width),
        c_int(sampling_ratio),
        c_void_p(bottom_rois.ctypes.data),
        c_int(roi_cols),
        c_void_p(top_data.ctypes.data),
    )
    return top_data.transpose([0, 2, 3, 1])


class ROIAlignWrapper(object):
    """This class gives a callback to the roi align operation compiled in cpp to so file."""

    def __init__(
        self,
        crop_size=(14, 14),
        spatial_scale=1 / 16.0,
        sampling_ratio=2,
        roi_cols=5,
        position_sensitive=False,
        continuous_coordinate=False,
        roi_align_filename="roi_align_float.so",
    ):
        so_path = os.path.join(hailo_model_zoo.core.__path__[0], "postprocessing", "roi_align_float.so")
        self._lib = ctypes.CDLL(so_path)
        self._roialign_func = self._lib.ROIAlignC
        self._roi_align_callback = partial(roi_align_wrapper, roi_callback=self._roialign_func)

    def __call__(self, featuremap, rpn_boxes):
        return self._roi_align_callback(featuremap=featuremap, rpn_boxes=rpn_boxes)
