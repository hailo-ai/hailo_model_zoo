import tensorflow as tf
import numpy as np
import os

from hailo_model_zoo.core.postprocessing.visualize_3d import visualization3Dbox

"""
KITTI_KS and KITTI_TRANS_MATS are from KITTI image calibration files.
KITTI_DEPTH_REF and KITTI_DIM_REF are taken from the code
"""
KITTI_KS = np.array([[721.5377, 000.0000, 609.5593, 44.85728],
                     [000.0000, 721.5377, 172.8540, 0.2163791],
                     [000.0000, 000.0000, 1.000000, 0.002745884]])
KITTI_TRANS_MATS = np.array([[2.5765e-1, 2.4772e-17, 4.2633e-14],
                             [-2.4772e-17, 2.5765e-1, -3.0918e-1],
                             [0., 0., 1.]])
CALIB_DATA_PATH = 'models_files/kitti_3d/label/calib/'
KITTI_DEPTH_REF = np.array([28.01, 16.32])
KITTI_DIM_REF = np.array([[3.88, 1.63, 1.53], [1.78, 1.70, 0.58], [0.88, 1.73, 0.67]])


def get_calibration_matrix_from_data(data):
    for line in data:
        if 'P2' in line:
            P2 = line.split(' ')
            P2 = np.asarray([float(i) for i in P2[1:]])
            P2 = np.reshape(P2, (3, 4))
            P2 = P2.astype('float32')
            return P2


def detection_3d_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    output_scheme = kwargs.get('output_scheme', None)
    if output_scheme:
        recombine_output = output_scheme.get('split_output', False)
    else:
        recombine_output = False
    smoke_postprocess = SMOKEPostProcess()
    results = smoke_postprocess.smoke_postprocessing(endnodes, device_pre_post_layers=device_pre_post_layers,
                                                     recombine_output=recombine_output, **kwargs)
    return {'predictions': results}


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _nms_heatmap(pred_heatmap):
    heatmap_padded = tf.pad(pred_heatmap, [[0, 0], [1, 1], [1, 1], [0, 0]])
    maxpooled_probs = tf.nn.max_pool2d(heatmap_padded, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
    probs_maxima_booleans = tf.cast(tf.math.equal(pred_heatmap, maxpooled_probs), 'float32')
    probs_maxima_values = tf.math.multiply(probs_maxima_booleans, pred_heatmap)
    return probs_maxima_values


def _rad_to_matrix(rotys, N):
    cos, sin = np.cos(rotys), np.sin(rotys)
    i_temp = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [-1, 0, 1]]).astype(np.float32)
    ry = np.reshape(np.tile(i_temp, (N, 1)), [N, -1, 3])
    ry[:, 0, 0] *= cos
    ry[:, 0, 2] *= sin
    ry[:, 2, 0] *= sin
    ry[:, 2, 2] *= cos
    return ry


class SMOKEPostProcess(object):
    # The following params are corresponding to those used for training the model
    def __init__(self, image_dims=(1280, 384), det_threshold=0.25,
                 output_scheme=None, num_classes=3, regression_head=8,
                 Ks=KITTI_KS, trans_mats=KITTI_TRANS_MATS,
                 depth_ref=KITTI_DEPTH_REF,
                 dim_ref=KITTI_DIM_REF, **kwargs):
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
        if os.path.isfile(os.path.join(CALIB_DATA_PATH, 'single_calib.txt')):
            calib_file_path = os.path.join(CALIB_DATA_PATH, 'single_calib.txt')
        else:
            calib_file_path = str(filepath)[3:-2]
        with open(calib_file_path, 'r') as f:
            P2 = get_calibration_matrix_from_data(f)
            Ks = np.expand_dims(P2[:, :3], axis=0)
            Ks_inv = np.linalg.inv(Ks)
        return Ks, Ks_inv

    def _read_calib_data(self, image_name):
        calib_dir = tf.constant(CALIB_DATA_PATH)
        calib_file_name = tf.strings.regex_replace(image_name, 'png', 'txt')
        calib_file_path = tf.strings.join([calib_dir, calib_file_name])
        Ks, Ks_inv = tf.compat.v1.py_func(self._get_calib_from_tensor_filename, [
                                          calib_file_path], ['float32', 'float32'])
        return Ks, Ks_inv

    def smoke_postprocessing(self, endnodes, device_pre_post_layers=None, recombine_output=False,
                             image_info=None, **kwargs):
        image_name = kwargs['gt_images']['image_name']
        self._Ks, self._Ks_inv = self._read_calib_data(image_name)

        # recombining split regression output:
        if recombine_output:
            endnodes = tf.compat.v1.py_func(self._recombine_split_endnodes, endnodes, ['float32', 'float32'])
        if device_pre_post_layers and device_pre_post_layers.get('max_finder', False):
            print('Not building postprocess NMS. Assuming the hw contains one!')
            heatmap = endnodes[1]
        else:
            print('Building postprocess NMS.')
            heatmap = _nms_heatmap(endnodes[1])
        self.bs = tf.shape(heatmap)[0]
        pred_regression = endnodes[0]
        pred_regression = tf.compat.v1.py_func(self.add_activations, [pred_regression], ['float32'])[0]
        scores, indexs, clses, ys, xs = tf.compat.v1.py_func(self._select_topk, [heatmap],
                                                             ['float32', 'float32', 'float32', 'float32', 'float32'])
        tensor_list_for_select_point_of_interest = [self.bs, indexs, pred_regression]
        # returning a [BS, K, regression_head] tensor:
        pred_regression = tf.compat.v1.py_func(self._select_point_of_interest, tensor_list_for_select_point_of_interest,
                                               ['float32'])
        pred_regression_pois = tf.reshape(pred_regression, [-1, self._regression_head])
        pred_proj_points = tf.concat([tf.reshape(xs, [-1, 1]), tf.reshape(ys, [-1, 1])], axis=1)

        pred_depths_offset = pred_regression_pois[:, 0]  # delta_z
        pred_proj_offsets = pred_regression_pois[:, 1:3]  # delta_xc, delta_yc
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]  # delta_h, delta_w, delta_l
        pred_orientation = pred_regression_pois[:, 6:]  # sin_alpha, cos_alpha

        pred_depths = self._decode_depth(pred_depths_offset)
        preds_for_location_decoding = [pred_proj_points,
                                       pred_proj_offsets,
                                       pred_depths,
                                       self._Ks_inv,
                                       self._trans_mats_inv,
                                       self.bs]

        pred_locations = tf.compat.v1.py_func(self._decode_location, preds_for_location_decoding, ['float32'])[0]
        preds_for_dimension_decoding = [clses, pred_dimensions_offsets]
        pred_dimensions = tf.compat.v1.py_func(self._decode_dimension, preds_for_dimension_decoding, ['float32'])[0]
        pred_locations = tf.compat.v1.py_func(self._assign_items, [pred_locations, pred_dimensions], ['float32'])[0]
        # now for the rotations
        preds_for_orientation_decoding = [pred_orientation, pred_locations]
        pred_rotys, pred_alphas = tf.compat.v1.py_func(self._decode_orientation, preds_for_orientation_decoding,
                                                       ['float32', 'float32'])

        if self._pred_2d:
            inputs_for_box2d_encode = [self._Ks, pred_rotys, pred_dimensions, pred_locations, self._image_dims]
            box2d = tf.compat.v1.py_func(self._encode_box2d, inputs_for_box2d_encode, ['float32'])
        else:
            box2d = tf.zeros([4])
        clses = tf.reshape(clses, [-1, 1])
        pred_alphas = tf.reshape(pred_alphas, [-1, 1])
        pred_rotys = tf.reshape(pred_rotys, [-1, 1])
        scores = tf.reshape(scores, [-1, 1])
        # change dims back to h,w,l:
        pred_dimensions = tf.roll(pred_dimensions, shift=-1, axis=1)
        """sqeeuzing dim 0"""
        pred_dimensions = tf.squeeze(pred_dimensions)
        box2d = tf.squeeze(box2d)
        res_list_for_concat = [clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores]
        result = tf.compat.v1.py_func(self._concatenate_results, res_list_for_concat, ['float32'])[0]
        return result

    def _recombine_split_endnodes(self, reg_depth, heatmap, reg_offset, reg_dims, reg_sin, reg_cos):
        regression = np.concatenate([reg_depth, reg_offset, reg_dims, reg_sin, reg_cos], axis=3)
        return [regression, heatmap]

    def add_activations(self, pred_regression):
        eps = 1e-12
        pred_regression[:, :, :, 3:6] = sigmoid(pred_regression[:, :, :, 3:6]) - 0.5
        orientation_slice = pred_regression[:, :, :, 6:]
        # rescaling cosine:
        cosine_scaling = 1.
        orientation_slice[:, :, :, 1] = cosine_scaling * orientation_slice[:, :, :, 1]

        orientation_slice_norm = np.linalg.norm(orientation_slice, ord=2, axis=3, keepdims=True).clip(eps)
        orientation_slice = orientation_slice / orientation_slice_norm
        pred_regression[:, :, :, 6:] = orientation_slice
        return pred_regression

    def _concatenate_results(self, clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores):
        clses = clses.astype(np.float32)
        result = np.concatenate([clses, pred_alphas, box2d,
                                 pred_dimensions, pred_locations, pred_rotys,
                                 scores], axis=1)
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
        bboxfrom3d = np.concatenate([np.expand_dims(xmins, axis=1), np.expand_dims(ymins, axis=1),
                                     np.expand_dims(xmaxs, axis=1), np.expand_dims(ymaxs, axis=1)], axis=1)
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
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        index = np.array([[4, 0, 1, 2, 3, 5, 6, 7],
                          [4, 5, 0, 1, 6, 7, 2, 3],
                          [4, 5, 6, 0, 1, 2, 3, 7]])
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
        # the dims seeem to have been calculated as if the box was already in the normal frame.
        rotys = alphas + rays
        larger_idx = (rotys > np.pi).nonzero()
        small_idx = (rotys < np.pi).nonzero()
        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * np.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * np.pi

        if flip_mask is not None:
            fm = flip_mask.flatten()
            rotys_flip = np.float(fm) * rotys
            rotys_flip_pos_idx = rotys_flip > 0
            rotys_flip_neg_idx = rotys_flip < 0
            rotys_flip[rotys_flip_pos_idx] -= np.pi
            rotys_flip[rotys_flip_neg_idx] += np.pi
            rotys_all = np.float(fm) * rotys_flip + (1 - np.float(fm)) * rotys
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


def visualize_3d_detection_result(logits, image, image_name=None, threshold=0.25, image_info=None,
                                  use_normalized_coordinates=True, max_boxes_to_draw=20,
                                  dataset_name='kitti_3d', channels_remove=None, **kwargs):
    '''our code writes 1 image at a time while smoke vis writes them all together.
       i will convert the smoke vis. '''
    logits = logits['predictions']
    if os.path.isfile(os.path.join(CALIB_DATA_PATH, 'single_calib.txt')):
        calib_file_path = os.path.join(CALIB_DATA_PATH, 'single_calib.txt')
    else:
        calib_file_path = os.path.join(CALIB_DATA_PATH, image_name.decode('utf-8').replace('png', 'txt'))
    with open(calib_file_path) as data:
        KITTI_KS = get_calibration_matrix_from_data(data)
    return visualization3Dbox.visualize_hailo(logits, image, image_name.decode('utf-8'), threshold, KITTI_KS)
