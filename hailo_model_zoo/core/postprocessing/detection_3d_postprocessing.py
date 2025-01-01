import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nuscenes.eval.detection.config import config_factory
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from nuscenes.utils.map_mask import MapMask
from pyquaternion import Quaternion
from tqdm import tqdm

from hailo_model_zoo.core.datasets.datasets_info import get_dataset_info
from hailo_model_zoo.core.eval.detection_3d_evaluation import (
    DefaultAttribute,
    lidar_nusc_box_to_global,
    output_to_nusc_box,
)
from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY, VISUALIZATION_FACTORY
from hailo_model_zoo.core.postprocessing.visualize_3d import visualization3Dbox
from hailo_model_zoo.core.postprocessing.visualize_3d.utils.visual_nuscenes import NuScenesExplorer, category_mapping
from hailo_model_zoo.utils import path_resolver
from hailo_model_zoo.utils.logger import get_logger

"""
KITTI_KS and KITTI_TRANS_MATS are from KITTI image calibration files.
KITTI_DEPTH_REF and KITTI_DIM_REF are taken from the code
"""
KITTI_KS = np.array(
    [
        [721.5377, 000.0000, 609.5593, 44.85728],
        [000.0000, 721.5377, 172.8540, 0.2163791],
        [000.0000, 000.0000, 1.000000, 0.002745884],
    ]
)
KITTI_TRANS_MATS = np.array(
    [[2.5765e-1, 2.4772e-17, 4.2633e-14], [-2.4772e-17, 2.5765e-1, -3.0918e-1], [0.0, 0.0, 1.0]]
)
CALIB_DATA_PATH = str(path_resolver.resolve_data_path("models_files/kitti_3d/label/calib/")) + "/"
KITTI_DEPTH_REF = np.array([28.01, 16.32])
KITTI_DIM_REF = np.array([[3.88, 1.63, 1.53], [1.78, 1.70, 0.58], [0.88, 1.73, 0.67]])


def get_calibration_matrix_from_data(data):
    for line in data:
        if "P2" in line:
            P2 = line.split(" ")
            P2 = np.asarray([float(i) for i in P2[1:]])
            P2 = np.reshape(P2, (3, 4))
            P2 = P2.astype("float32")
            return P2


def denormalize_bbox(normalized_bboxes):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = tf.math.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    width = normalized_bboxes[..., 2:3]
    length = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = tf.exp(width)
    length = tf.exp(length)
    height = tf.exp(height)
    if normalized_bboxes.shape[-1] > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = tf.concat([cx, cy, cz, width, length, height, rot, vx, vy], axis=-1)

    else:
        denormalized_bboxes = tf.concat([cx, cy, cz, width, length, height, rot], axis=-1)
    return denormalized_bboxes


class NMSFreeCoder:
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(
        self, pc_range, voxel_size=None, post_center_range=None, max_num=100, score_threshold=None, num_classes=10
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = tf.math.sigmoid(cls_scores)
        scores, indices = tf.math.top_k(tf.reshape(cls_scores, [-1]), k=max_num)

        labels = tf.math.floormod(indices, self.num_classes)
        bbox_index = tf.math.floordiv(indices, self.num_classes)

        bbox_preds = tf.gather(bbox_preds, bbox_index)

        final_box_preds = denormalize_bbox(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = tf.constant(self.post_center_range)

            mask = tf.math.reduce_all((final_box_preds[..., :3] >= self.post_center_range[:3]), axis=1)
            mask &= tf.math.reduce_all((final_box_preds[..., :3] <= self.post_center_range[3:]), axis=1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            tmp = boxes3d[:, 2:3] - boxes3d[:, 5:6] * 0.5
            boxes3d = tf.concat([boxes3d[..., 0:2], tmp, boxes3d[..., 3:]], axis=-1)
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {"boxes_3d": boxes3d, "scores_3d": scores, "labels_3d": labels}

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        # NOTE: PETR-v2 takes only the last output branch results
        all_cls_scores = preds_dicts["cls_scores"][-1]
        all_bbox_preds = preds_dicts["bbox_pred"][-1]

        batch_size = all_cls_scores.shape[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


@POSTPROCESS_FACTORY.register(name="object_detection_3d")
def petrv2_transformer(endnodes, device_pre_post_layers=None, **kwargs):
    postprocess_config_file = kwargs["postprocess_config_file"]
    ref_points_path = postprocess_config_file
    ref_points_path = str(path_resolver.resolve_data_path(ref_points_path)) if ref_points_path else None
    if not os.path.isfile(ref_points_path):
        raise FileNotFoundError(f"Could not find {ref_points_path}")

    ref_points = np.load(ref_points_path)
    return petrv2_postprocess(
        endnodes,
        timestamp=kwargs["gt_images"]["timestamp"],
        ref_points=ref_points,
        classes=kwargs["classes"],
    )


def petrv2_postprocess(endnodes, timestamp, ref_points, classes):
    batch_size = endnodes[0].shape[0]

    timestamp = tf.reshape(timestamp, [batch_size, -1, 6])
    mean_time_stamp = tf.math.reduce_mean((timestamp[:, 1, :] - timestamp[:, 0, :]), -1)
    mean_time_stamp = tf.expand_dims(tf.expand_dims(mean_time_stamp, -1), -1)
    endnodes = [tf.squeeze(endnode, [0]) for endnode in endnodes]
    num_pred = len(endnodes) // 2
    reg_branches = endnodes[:num_pred]
    cls_branches = endnodes[num_pred:]

    ref_points = tf.reshape(ref_points, [reg_branches[0].shape[0], reg_branches[0].shape[1], 3])

    outputs_coords = []
    for lvl in range(len(reg_branches)):
        reg_branch = reg_branches[lvl]
        x1 = reg_branch[..., 0:2] + ref_points[..., 0:2]
        x1 = tf.math.sigmoid(x1)
        x2 = reg_branch[..., 2:4]
        x3 = reg_branch[..., 4:5] + ref_points[..., 2:3]
        x3 = tf.math.sigmoid(x3)
        x4 = reg_branch[..., 5:8]
        x5 = reg_branch[..., 8:] / mean_time_stamp

        output_coord = tf.concat([x1, x2, x3, x4, x5], axis=-1)
        assert output_coord.shape == reg_branch.shape
        outputs_coords.append(output_coord)

    all_cls_scores = tf.stack(cls_branches)
    all_bbox_preds = tf.stack(outputs_coords)

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    x1 = all_bbox_preds[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    x2 = all_bbox_preds[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    x3 = all_bbox_preds[..., 2:4]
    x4 = all_bbox_preds[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2]
    x5 = all_bbox_preds[..., 5:]
    all_bbox_pred = tf.concat([x1, x2, x3, x4, x5], axis=-1)
    assert all_bbox_pred.shape == all_bbox_preds.shape

    preds_dict = {"cls_scores": all_cls_scores, "bbox_pred": all_bbox_pred}
    num_classes = classes
    bbox_coder = NMSFreeCoder(
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=pc_range,
        max_num=300,
        voxel_size=[0.2, 0.2, 8],
        num_classes=num_classes,
    )

    preds_dicts = bbox_coder.decode(preds_dict)

    return {"predictions": preds_dicts}


@POSTPROCESS_FACTORY.register(name="object_detection_3d_backbone")
def petrv2_backbone(endnodes, device_pre_post_layers=None, **kwargs):
    return {"predictions": endnodes}


@POSTPROCESS_FACTORY.register(name="3d_detection")
def detection_3d_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    output_scheme = kwargs.get("output_scheme", None)
    if output_scheme:
        recombine_output = output_scheme.get("split_output", False)
    else:
        recombine_output = False
    smoke_postprocess = SMOKEPostProcess()
    results = smoke_postprocess.smoke_postprocessing(
        endnodes, device_pre_post_layers=device_pre_post_layers, recombine_output=recombine_output, **kwargs
    )
    return {"predictions": results}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms_heatmap(pred_heatmap):
    heatmap_padded = tf.pad(pred_heatmap, [[0, 0], [1, 1], [1, 1], [0, 0]])
    maxpooled_probs = tf.nn.max_pool2d(heatmap_padded, [1, 3, 3, 1], [1, 1, 1, 1], "VALID")
    probs_maxima_booleans = tf.cast(tf.math.equal(pred_heatmap, maxpooled_probs), "float32")
    probs_maxima_values = tf.math.multiply(probs_maxima_booleans, pred_heatmap)
    return probs_maxima_values


def _rad_to_matrix(rotys, N):
    cos, sin = np.cos(rotys), np.sin(rotys)
    i_temp = np.array([[1, 0, 1], [0, 1, 0], [-1, 0, 1]]).astype(np.float32)
    ry = np.reshape(np.tile(i_temp, (N, 1)), [N, -1, 3])
    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos
    return ry


class SMOKEPostProcess(object):
    # The following params are corresponding to those used for training the model
    def __init__(
        self,
        image_dims=(1280, 384),
        det_threshold=0.25,
        output_scheme=None,
        num_classes=3,
        regression_head=8,
        Ks=KITTI_KS,
        trans_mats=KITTI_TRANS_MATS,
        depth_ref=KITTI_DEPTH_REF,
        dim_ref=KITTI_DIM_REF,
        **kwargs,
    ):
        self._dim_ref = dim_ref
        self._depth_ref = depth_ref
        self._regression_head = regression_head
        self._image_dims = image_dims
        self._num_classes = num_classes
        self._Ks = np.expand_dims(Ks[:, :3], axis=0)
        self._trans_mat = trans_mats
        self._trans_mats = np.expand_dims(trans_mats, axis=0)
        self._trans_mats_inv = np.linalg.inv(self._trans_mats)
        self._Ks_inv = np.linalg.inv(self._Ks)
        self._det_threshold = det_threshold
        self._pred_2d = True

    def _get_calib_from_tensor_filename(self, filepath):
        if os.path.isfile(os.path.join(CALIB_DATA_PATH, "single_calib.txt")):
            calib_file_path = os.path.join(CALIB_DATA_PATH, "single_calib.txt")
        else:
            calib_file_path = filepath[0]
        with open(calib_file_path, "r") as f:
            P2 = get_calibration_matrix_from_data(f)
            Ks = np.expand_dims(P2[:, :3], axis=0)
            Ks_inv = np.linalg.inv(Ks)
        return Ks, Ks_inv

    def _read_calib_data(self, image_name):
        calib_dir = tf.constant(CALIB_DATA_PATH)
        calib_file_name = tf.strings.regex_replace(image_name, "png", "txt")
        calib_file_path = tf.strings.join([calib_dir, calib_file_name])
        Ks, Ks_inv = tf.numpy_function(self._get_calib_from_tensor_filename, [calib_file_path], ["float32", "float32"])
        return Ks, Ks_inv

    def smoke_postprocessing(
        self, endnodes, device_pre_post_layers=None, recombine_output=False, image_info=None, **kwargs
    ):
        image_name = kwargs["gt_images"]["image_name"]
        self._Ks, self._Ks_inv = self._read_calib_data(image_name)

        # recombining split regression output:
        if recombine_output:
            endnodes = tf.numpy_function(self._recombine_split_endnodes, endnodes, ["float32", "float32"])
        if device_pre_post_layers and device_pre_post_layers.get("max_finder", False):
            print("Not building postprocess NMS. Assuming the hw contains one!")
            heatmap = endnodes[1]
        else:
            print("Building postprocess NMS.")
            heatmap = _nms_heatmap(endnodes[1])
        self.bs = tf.shape(heatmap)[0]
        pred_regression = endnodes[0]
        pred_regression = tf.numpy_function(self.add_activations, [pred_regression], ["float32"])[0]
        scores, indices, classes, ys, xs = tf.numpy_function(
            self._select_topk, [heatmap], ["float32", "float32", "float32", "float32", "float32"]
        )
        tensor_list_for_select_point_of_interest = [self.bs, indices, pred_regression]
        # returning a [BS, K, regression_head] tensor:
        pred_regression = tf.numpy_function(
            self._select_point_of_interest, tensor_list_for_select_point_of_interest, ["float32"]
        )
        pred_regression_pois = tf.reshape(pred_regression, [-1, self._regression_head])
        pred_proj_points = tf.concat([tf.reshape(xs, [-1, 1]), tf.reshape(ys, [-1, 1])], axis=1)

        pred_depths_offset = pred_regression_pois[:, 0]  # delta_z
        pred_proj_offsets = pred_regression_pois[:, 1:3]  # delta_xc, delta_yc
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]  # delta_h, delta_w, delta_l
        pred_orientation = pred_regression_pois[:, 6:]  # sin_alpha, cos_alpha

        pred_depths = self._decode_depth(pred_depths_offset)
        preds_for_location_decoding = [
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            self._Ks_inv,
            self._trans_mats_inv,
            self.bs,
        ]

        pred_locations = tf.numpy_function(self._decode_location, preds_for_location_decoding, ["float32"])[0]
        preds_for_dimension_decoding = [classes, pred_dimensions_offsets]
        pred_dimensions = tf.numpy_function(self._decode_dimension, preds_for_dimension_decoding, ["float32"])[0]
        pred_locations = tf.numpy_function(self._assign_items, [pred_locations, pred_dimensions], ["float32"])[0]
        # now for the rotations
        preds_for_orientation_decoding = [pred_orientation, pred_locations]
        pred_rotys, pred_alphas = tf.numpy_function(
            self._decode_orientation, preds_for_orientation_decoding, ["float32", "float32"]
        )

        if self._pred_2d:
            inputs_for_box2d_encode = [self._Ks, pred_rotys, pred_dimensions, pred_locations, self._image_dims]
            box2d = tf.numpy_function(self._encode_box2d, inputs_for_box2d_encode, ["float32"])
        else:
            box2d = tf.zeros([4])
        classes = tf.reshape(classes, [-1, 1])
        pred_alphas = tf.reshape(pred_alphas, [-1, 1])
        pred_rotys = tf.reshape(pred_rotys, [-1, 1])
        scores = tf.reshape(scores, [-1, 1])
        # change dims back to h,w,l:
        pred_dimensions = tf.roll(pred_dimensions, shift=-1, axis=1)
        """sqeeuzing dim 0"""
        pred_dimensions = tf.squeeze(pred_dimensions)
        box2d = tf.squeeze(box2d)
        res_list_for_concat = [classes, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores]
        result = tf.numpy_function(self._concatenate_results, res_list_for_concat, ["float32"])[0]
        return result

    def _recombine_split_endnodes(self, reg_depth, heatmap, reg_offset, reg_dims, reg_sin, reg_cos):
        regression = np.concatenate([reg_depth, reg_offset, reg_dims, reg_sin, reg_cos], axis=3)
        return [regression, heatmap]

    def add_activations(self, pred_regression):
        eps = 1e-12
        pred_regression[:, :, :, 3:6] = sigmoid(pred_regression[:, :, :, 3:6]) - 0.5
        orientation_slice = pred_regression[:, :, :, 6:]
        # rescaling cosine:
        cosine_scaling = 1.0
        orientation_slice[:, :, :, 1] = cosine_scaling * orientation_slice[:, :, :, 1]

        orientation_slice_norm = np.linalg.norm(orientation_slice, ord=2, axis=3, keepdims=True).clip(eps)
        orientation_slice = orientation_slice / orientation_slice_norm
        pred_regression[:, :, :, 6:] = orientation_slice
        return pred_regression

    def _concatenate_results(self, clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores):
        clses = clses.astype(np.float32)
        result = np.concatenate(
            [clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores], axis=1
        )
        keep_idx = result[:, -1] > self._det_threshold
        result = result[keep_idx]
        """adding a dim 0 for batch because right now it's not handled. so for now i require bs=1 !!"""
        result = np.expand_dims(result, axis=0)
        return result

    def _assign_items(self, array_1, array_2):
        array_1[:, 1] += array_2[:, 1] / 2
        return array_1

    def _encode_box2d(self, K, rotys, dims, locs, img_size):
        K = K[0, :, :]
        img_size = img_size.flatten()
        box3d = self._encode_box3d(rotys, dims, locs)
        box3d_image = np.matmul(K, box3d)
        box3d_image = box3d_image[:, :2, :] / (np.reshape(box3d_image[:, 2, :], (box3d.shape[0], 1, box3d.shape[2])))
        xmins = box3d_image[:, 0, :].min(axis=1)
        xmaxs = box3d_image[:, 0, :].max(axis=1)
        ymins = box3d_image[:, 1, :].min(axis=1)
        ymaxs = box3d_image[:, 1, :].max(axis=1)

        xmins = np.clip(xmins, 0, img_size[0])
        xmaxs = np.clip(xmaxs, 0, img_size[0])
        ymins = np.clip(ymins, 0, img_size[1])
        ymaxs = np.clip(ymaxs, 0, img_size[1])
        bboxfrom3d = np.concatenate(
            [
                np.expand_dims(xmins, axis=1),
                np.expand_dims(ymins, axis=1),
                np.expand_dims(xmaxs, axis=1),
                np.expand_dims(ymaxs, axis=1),
            ],
            axis=1,
        )
        return bboxfrom3d.astype(np.float32)

    def _encode_box3d(self, rotys, dims, locs):
        if len(rotys.shape) == 2:
            rotys = rotys.flatten()
        if len(dims.shape) == 3:
            dims = np.reshape(dims, [-1, 3])
        if len(locs.shape) == 3:
            locs = np.reshape(locs, [-1, 3])

        N = rotys.shape[0]
        ry = _rad_to_matrix(rotys, N)
        dims = np.tile(np.reshape(dims, (-1, 1)), (1, 8))

        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        dims[1::3, :4], dims[1::3, 4:] = 0.0, -dims[1::3, 4:]
        index = np.array([[4, 0, 1, 2, 3, 5, 6, 7], [4, 5, 0, 1, 6, 7, 2, 3], [4, 5, 6, 0, 1, 2, 3, 7]])
        index = np.tile(index, (N, 1))
        box_3d_object = np.take_along_axis(dims, index, 1)  # replaces torch.gather()
        box_3d = np.matmul(ry, np.reshape(box_3d_object, (N, 3, -1)))
        box_3d += np.tile(np.expand_dims(locs, axis=-1), (1, 1, 8))
        return box_3d

    def _decode_orientation(self, vector_ori, locations, flip_mask=None):
        eps = 1e-7
        locations = np.reshape(locations, [-1, 3])
        rays = np.arctan(locations[:, 0] / (locations[:, 2] + eps))
        alphas = np.arctan(vector_ori[:, 0] / (vector_ori[:, 1] + eps))
        cos_pos_ids = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_ids = (vector_ori[:, 1] < 0).nonzero()
        alphas[cos_pos_ids] -= np.pi / 2
        alphas[cos_neg_ids] += np.pi / 2
        # rotate the object corners?? I think there's a conceptual error in the rotation:
        # the dims seem to have been calculated as if the box was already in the normal frame.
        rotys = alphas + rays
        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < np.pi).nonzero()
        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        if flip_mask is not None:
            fm = flip_mask.flatten()
            rotys_flip = float(fm) * rotys
            rotys_flip_pos_idx = rotys_flip > 0
            rotys_flip_neg_idx = rotys_flip < 0
            rotys_flip[rotys_flip_pos_idx] -= np.pi
            rotys_flip[rotys_flip_neg_idx] += np.pi
            rotys_all = float(fm) * rotys_flip + (1 - float(fm)) * rotys
            return rotys_all
        else:
            return rotys, alphas

    def _decode_dimension(self, cls_id, dims_offset):
        cls_id = cls_id.flatten()
        dims_select = self._dim_ref[cls_id.astype(np.int32), :]  # a [C, 2] array
        dimensions = np.exp(dims_offset) * dims_select
        return dimensions.astype(np.float32)

    def _decode_location(self, points, points_offset, depths, Ks_inv_single, trans_mats_inv_single, bs):
        N = points_offset.shape[0]  # this is a [BS,2,N] array N<=K (if there aren't enough objects).
        N_batch = bs  # batch size
        batch_id = np.arange(N_batch)
        batch_id = np.expand_dims(batch_id, 1)  # a [BS, 1] array
        obj_id = np.concatenate((N // N_batch) * [batch_id], axis=1)
        # now dim 0 is im number in batch and dim 1 is the object number dimension.
        obj_id = obj_id.flatten()
        trans_mats_inv = np.concatenate(N * [trans_mats_inv_single], axis=0)
        Ks_inv = np.concatenate(N * [Ks_inv_single], axis=0)
        Ks_inv = Ks_inv[obj_id]
        points = np.reshape(points, [-1, 2])
        assert points.shape[0] == N
        # the fine-grained location of the projected 3D center points on the image:
        proj_points = points + points_offset
        # this is an [N, 3] array with the last column being 1's:
        proj_points_extend = np.concatenate([proj_points, np.ones([N, 1])], axis=1)
        # now it's an [N, 3, 1] array:
        proj_points_extend = np.expand_dims(proj_points_extend, axis=-1)
        # from feature map plane to image plane (still 2D):
        proj_points_img = np.matmul(trans_mats_inv, proj_points_extend)
        # now the center points are in 3D: [x, y, z], but still in image frame:
        proj_points_img = proj_points_img * np.reshape(depths, [N, -1, 1])
        # transformed to real world:
        locations = np.matmul(Ks_inv, proj_points_img)
        return np.squeeze(locations, axis=2).astype(np.float32)

    def _decode_depth(self, depths_offset):
        """transform depth offset to depth"""
        depth = depths_offset * self._depth_ref[1] + self._depth_ref[0]
        return depth

    def _select_topk(self, heat_map, K=100):
        batch, height, width, cls = heat_map.shape
        #  let's reshape to make long HxW index frames:
        heat_map = heat_map.transpose([0, 3, 1, 2])  # now the order is [BS, C, H, W]
        heat_map = heat_map.reshape(batch, cls, -1)  # now the shape is [BS, C, H*W]
        # we now want to get two [BS, K] arrays: one for the K highest values in each class,
        # and one for their dim=-1 indices.
        # a [BS, C, K] array with the K highest per-class scores:
        topk_scores_all = -np.sort(-heat_map, axis=-1)[:, :, :K]
        # this array is [BS, C, K] with their H*W-scale indices:
        topk_inds_all = np.argsort(-heat_map, axis=-1)[:, :, :K]
        topk_ys = (topk_inds_all // width).astype(np.float32)
        topk_xs = (topk_inds_all % width).astype(np.float32)
        # now we will select the top K elements out of the entire array:
        topk_scores_all = np.reshape(topk_scores_all, (batch, -1))  # this array is [BS, C*K]
        # a [BS, K] array with the K highest per-image scores:
        topk_scores = -np.sort(-topk_scores_all, axis=-1)[:, :K]
        # a [BS, K] array whose values are indices in the C*K notation:
        topk_inds = np.argsort(-topk_scores_all, axis=-1)[:, :K]
        # renormalizing the index to get the class for each of the K points:
        topk_clses = (topk_inds // K).astype(np.float32)

        # now we want to get the indices in a [BS, K] size array with the H*W-notation indices:
        topk_inds_all = self._take_along_axis(np.reshape(topk_inds_all, (batch, -1, 1)), topk_inds)
        topk_ys = self._take_along_axis(np.reshape(topk_ys, (batch, -1, 1)), topk_inds)
        topk_xs = self._take_along_axis(np.reshape(topk_xs, (batch, -1, 1)), topk_inds)

        topk_inds_all = np.reshape(topk_inds_all, (batch, K)).astype(np.float32)
        topk_ys = np.reshape(topk_ys, (batch, K))
        topk_xs = np.reshape(topk_xs, (batch, K))
        return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs

    def _take_along_axis(self, feat, ind):
        channel = feat.shape[-1]
        ind = np.expand_dims(ind, axis=-1)
        ind = np.concatenate(channel * [ind], axis=2)
        feat = np.take_along_axis(feat, ind, axis=1)
        return feat

    def _select_point_of_interest(self, batch, index, feature_maps):
        W = feature_maps.shape[2]  # the width. in the source it's =3 because the torch convention is NCHW.
        if len(index.shape) == 3:
            index = index[:, :, 1] * W + index[:, :, 0]
        index = np.reshape(index, (batch, -1)).astype(np.int32)
        channel = feature_maps.shape[-1]
        feature_maps = np.reshape(feature_maps, (batch, -1, channel))  # now it's in [BS, H*W, C]
        index = np.expand_dims(index, axis=-1)
        index = np.concatenate(channel * [index], axis=2)
        feature_maps = np.take_along_axis(feature_maps, index, axis=1)
        return feature_maps


@VISUALIZATION_FACTORY.register(name="3d_detection")
def visualize_3d_detection_result(
    logits,
    image,
    image_name=None,
    threshold=0.25,
    image_info=None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    dataset_name="kitti_3d",
    channels_remove=None,
    **kwargs,
):
    """our code writes 1 image at a time while smoke vis writes them all together.
    i will convert the smoke vis."""
    logits = logits["predictions"]
    if os.path.isfile(os.path.join(CALIB_DATA_PATH, "single_calib.txt")):
        calib_file_path = os.path.join(CALIB_DATA_PATH, "single_calib.txt")
    else:
        calib_file_path = os.path.join(CALIB_DATA_PATH, image_name.decode("utf-8").replace("png", "txt"))
    with open(calib_file_path) as data:
        KITTI_KS = get_calibration_matrix_from_data(data)
    return visualization3Dbox.visualize_hailo(logits, image, image_name.decode("utf-8"), threshold, KITTI_KS)


@VISUALIZATION_FACTORY.register(name="object_detection_3d")
class PETRV2Visualizer:
    dataroot = "/fastdata/data/nuscenes/nuesence/"

    def __init__(
        self,
        score_threshold=0.25,
        map_resolution: float = 0.1,
        version="v1.0-trainval",
        eval_version="detection_cvpr_2019",
        show_lidarseg=False,
        show_panoptic=False,
        box_vis_level=BoxVisibility.ANY,
        nsweeps=1,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path=None,
        **kwargs,
    ):
        self.eval_version = eval_version
        self.modality = {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        }

        self.logger = get_logger()
        self.version = version
        self.map_resolution = map_resolution
        self.score_threshold = score_threshold
        self.show_lidarseg = show_lidarseg
        self.show_panoptic = show_panoptic
        self.box_vis_level = box_vis_level
        self.nsweeps = nsweeps
        self.filter_lidarseg_labels = filter_lidarseg_labels
        self.lidarseg_preds_bin_path = lidarseg_preds_bin_path

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        self.table_names = ["sample", "sample_data", "calibrated_sensor", "sensor", "map", "log", "ego_pose", "scene"]
        self.sample = self.__load_table__("sample")
        self.scene = self.__load_table__("scene")
        self.sample_data = self.__load_table__("sample_data")
        self.calibrated_sensor = self.__load_table__("calibrated_sensor")
        self.sensor = self.__load_table__("sensor")
        self.map = self.__load_table__("map")
        self.log = self.__load_table__("log")
        self.ego_pose = self.__load_table__("ego_pose")

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record["mask"] = MapMask(os.path.join(self.dataroot, map_record["filename"]), resolution=map_resolution)

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__()

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    def __load_table__(self, table_name) -> dict:
        table_path = os.path.join(self.dataroot, self.version, f"{table_name}.json")
        with open(table_path) as f:
            table = json.load(f)
        return table

    def __make_reverse_index__(self) -> None:
        # Store the mapping from token to table index for each table.
        self.logger.info("Reverse indexing ...")
        self._token2ind = {}
        for table_name in self.table_names:
            self._token2ind[table_name] = {}
            for ind, member in tqdm(enumerate(getattr(self, table_name))):
                self._token2ind[table_name][member["token"]] = ind

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        log_to_map = {}
        for map_record in self.map:
            for log_token in map_record["log_tokens"]:
                log_to_map[log_token] = map_record["token"]
        for log_record in self.log:
            log_record["map_token"] = log_to_map[log_record["token"]]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get("sample_annotation", sample_annotation_token)
        return Box(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=record["category_name"],
            token=record["token"],
        )

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        curr_sample_record = self.get("sample", sd_record["sample_token"])

        if curr_sample_record["prev"] == "" or sd_record["is_key_frame"]:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record["anns"]))

        else:
            prev_sample_record = self.get("sample", curr_sample_record["prev"])

            curr_ann_recs = [self.get("sample_annotation", token) for token in curr_sample_record["anns"]]
            prev_ann_recs = [self.get("sample_annotation", token) for token in prev_sample_record["anns"]]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry["instance_token"]: entry for entry in prev_ann_recs}

            t0 = prev_sample_record["timestamp"]
            t1 = curr_sample_record["timestamp"]
            t = sd_record["timestamp"]

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:
                if curr_ann_rec["instance_token"] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec["instance_token"]]

                    # Interpolate center.
                    center = [
                        np.interp(t, [t0, t1], [c0, c1])
                        for c0, c1 in zip(prev_ann_rec["translation"], curr_ann_rec["translation"])
                    ]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(
                        q0=Quaternion(prev_ann_rec["rotation"]),
                        q1=Quaternion(curr_ann_rec["rotation"]),
                        amount=(t - t0) / (t1 - t0),
                    )

                    box = Box(
                        center,
                        curr_ann_rec["size"],
                        rotation,
                        name=curr_ann_rec["category_name"],
                        token=curr_ann_rec["token"],
                    )
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec["token"])

                boxes.append(box)
        return boxes

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        if table_name != "sample_annotation":
            assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Returns the path to a sample_data."""

        sd_record = self.get("sample_data", sample_data_token)
        return os.path.join(self.dataroot, sd_record["filename"])

    def get_sample_data(
        self,
        sample_data_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        selected_anntokens: List[str] = None,
        use_flat_vehicle_coordinates: bool = False,
    ) -> Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get("sample_data", sample_data_token)
        cs_record = self.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        sensor_record = self.get("sensor", cs_record["sensor_token"])
        pose_record = self.get("ego_pose", sd_record["ego_pose_token"])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record["modality"] == "camera":
            cam_intrinsic = np.array(cs_record["camera_intrinsic"])
            imsize = (sd_record["width"], sd_record["height"])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record["rotation"]).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record["translation"]))
                box.rotate(Quaternion(pose_record["rotation"]).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record["translation"]))
                box.rotate(Quaternion(cs_record["rotation"]).inverse)

            if sensor_record["modality"] == "camera" and not box_in_image(
                box, cam_intrinsic, imsize, vis_level=box_vis_level
            ):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def __call__(self, logits, img, **kwargs):
        dataset_name = kwargs.get("dataset_name", None)
        dataset_info = get_dataset_info(dataset_name=dataset_name)
        classes = dataset_info.class_names
        eval_detection_configs = config_factory(self.eval_version)

        img_info = kwargs.get("img_info", None)
        sample_token = img_info["token"].numpy().decode("utf-8")
        data_info = {
            "lidar2ego_rotation": img_info["lidar2ego_rotation"],
            "lidar2ego_translation": img_info["lidar2ego_translation"],
            "ego2global_rotation": img_info["ego2global_rotation"],
            "ego2global_translation": img_info["ego2global_translation"],
        }

        boxes = output_to_nusc_box(logits["predictions"])
        boxes = lidar_nusc_box_to_global(data_info, boxes, classes, eval_detection_configs, self.eval_version)

        nusc_dets = {}
        dets = []
        for _, box in enumerate(boxes):
            name = classes[box.label]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in [
                    "car",
                    "construction_vehicle",
                    "bus",
                    "truck",
                    "trailer",
                ]:
                    attr = "vehicle.moving"
                elif name in ["bicycle", "motorcycle"]:
                    attr = "cycle.with_rider"
                else:
                    attr = DefaultAttribute[name]
            else:
                if name in ["pedestrian"]:
                    attr = "pedestrian.standing"
                elif name in ["bus"]:
                    attr = "vehicle.stopped"
                else:
                    attr = DefaultAttribute[name]

            nusc_det = {
                "sample_token": sample_token,
                "translation": box.center.tolist(),
                "size": box.wlh.tolist(),
                "rotation": box.orientation.elements.tolist(),
                "velocity": box.velocity[:2].tolist(),
                "detection_name": name,
                "detection_score": box.score,
                "attribute_name": attr,
            }
            dets.append(nusc_det)
        nusc_dets[sample_token] = dets

        self.sample_annotation = {
            "meta": self.modality,
            "results": nusc_dets,
        }
        # Format predictions results
        results_pred = []
        token = 0
        for _, sample_anno in self.sample_annotation["results"].items():
            for record in sample_anno:
                if record["detection_score"] > self.score_threshold and record["detection_name"] in category_mapping:
                    record["token"] = str(token)
                    record["category_name"] = category_mapping[record["detection_name"]]
                    results_pred.append(record)
                    token = token + 1
        self.sample_annotation = results_pred

        self._token2ind["sample_annotation"] = {}
        for ind, member in enumerate(self.sample_annotation):
            self._token2ind["sample_annotation"][member["token"]] = ind

        for pred_record in self.sample_annotation:
            sample_pred = self.get("sample", pred_record["sample_token"])
            sample_pred["anns"].append(pred_record["token"])

        # Plot detections
        self.logger.info(f"Rendering sample token {sample_token}...")
        fig = self.explorer.render_sample(
            sample_token,
            self.box_vis_level,
            nsweeps=self.nsweeps,
            show_lidarseg=self.show_lidarseg,
            filter_lidarseg_labels=self.filter_lidarseg_labels,
            lidarseg_preds_bin_path=self.lidarseg_preds_bin_path,
            verbose=False,
            show_panoptic=self.show_panoptic,
        )

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        drawn_image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close("all")

        return drawn_image
