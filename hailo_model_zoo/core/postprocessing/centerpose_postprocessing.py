import numpy as np
import tensorflow as tf
from hailo_model_zoo.core.preprocessing.affine_utils import transform_preds


def _nms(heatmap):
    heatmap_padded = tf.pad(heatmap, [[0, 0], [1, 1], [1, 1], [0, 0]])
    pooled_heatmap = tf.nn.max_pool(heatmap_padded, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID')
    keep = tf.cast(tf.math.equal(heatmap, pooled_heatmap), 'float32')
    return tf.math.multiply(keep, heatmap)


def _np_topk(array, K, *, axis=-1, sort_output=True):
    if array.shape[axis] <= K:
        assert sort_output
        index_array = np.argsort(-array, axis=axis)
        return np.take_along_axis(array, index_array, axis=axis), index_array
    index_array = np.argpartition(-array, K, axis=axis)
    index_array = np.take(index_array, np.arange(K), axis=axis)
    result = np.take_along_axis(array, index_array, axis=axis)
    if sort_output:
        sorted_index_array = np.argsort(-result, axis=axis)
        result = np.take_along_axis(result, sorted_index_array, axis=axis)
        index_array = np.take_along_axis(index_array, sorted_index_array, axis=axis)
    return result, index_array


def _topk(scores, K=40):
    batch_size, out_height, out_width, category = scores.shape

    topk_scores, topk_indices = _np_topk(np.reshape(scores, (batch_size, category, -1)), K)

    topk_indices = topk_indices % (out_height * out_width)
    topk_ys = np.floor(topk_indices / out_width)
    topk_xs = np.floor(topk_indices % out_width)

    topk_score, topk_ind = _np_topk(topk_scores.reshape((batch_size, -1)), K)
    topk_clses = (topk_ind / K).astype(int)
    topk_indices = _gather_feat(
        topk_indices.reshape((batch_size, -1, 1)), topk_ind).reshape(batch_size, K)
    topk_ys = _gather_feat(topk_ys.reshape((batch_size, -1, 1)), topk_ind).reshape(batch_size, K)
    topk_xs = _gather_feat(topk_xs.reshape((batch_size, -1, 1)), topk_ind).reshape(batch_size, K)

    return topk_score, topk_indices, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch_size, out_height, out_width, category = scores.shape
    scores = scores.transpose((0, 3, 1, 2))

    topk_scores, topk_indices = _np_topk(scores.reshape((batch_size, category, -1)), K)

    topk_indices = topk_indices % (out_height * out_width)
    topk_ys = np.floor(topk_indices / out_width)
    topk_xs = np.floor(topk_indices % out_width)

    return topk_scores, topk_indices, topk_ys, topk_xs


def _gather_feat(feat, ind):
    dim = feat.shape[2]
    ind = np.tile(np.expand_dims(ind, 2), dim)
    feat = np.take_along_axis(feat, ind, axis=1)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
    feat = _gather_feat(feat, ind)
    return feat


def _centerpose_postprocessing(center_heatmap, center_wh, joint_heatmap, center_offset, joint_center_offset,
                               joint_offset, centers, scales, *, K=100, heatmap_score_thresh=0.1, **kwargs):
    center_heatmap, center_wh, joint_heatmap, center_offset, joint_center_offset, joint_offset = (
        np.array(x) for x in (center_heatmap, center_wh, joint_heatmap,
                              center_offset, joint_center_offset, joint_offset))

    batch_size, out_height, out_width, category = center_heatmap.shape
    num_joints = joint_center_offset.shape[-1] // 2

    scores, topk_indices, classes, ys, xs = _topk(center_heatmap, K)
    keypoints = _transpose_and_gather_feat(joint_center_offset, topk_indices)
    keypoints = keypoints.reshape((batch_size, K, num_joints * 2))
    keypoints[..., ::2] += np.tile(xs.reshape((batch_size, K, 1)), num_joints)
    keypoints[..., 1::2] += np.tile(ys.reshape((batch_size, K, 1)), num_joints)
    if center_offset is not None:
        reg = _transpose_and_gather_feat(center_offset, topk_indices)
        reg = reg.reshape((batch_size, K, 2))
        xs = xs.reshape((batch_size, K, 1)) + reg[:, :, 0:1]
        ys = ys.reshape((batch_size, K, 1)) + reg[:, :, 1:2]
    else:
        xs = xs.reshape((batch_size, K, 1)) + 0.5
        ys = ys.reshape((batch_size, K, 1)) + 0.5
    wh = _transpose_and_gather_feat(center_wh, topk_indices)
    wh = wh.reshape((batch_size, K, 2))
    classes = classes.reshape((batch_size, K, 1)).astype(float)
    scores = scores.reshape((batch_size, K, 1))

    bboxes = np.concatenate([xs - wh[..., 0:1] / 2,
                             ys - wh[..., 1:2] / 2,
                             xs + wh[..., 0:1] / 2,
                             ys + wh[..., 1:2] / 2], axis=2)
    if joint_heatmap is not None:
        keypoints = keypoints.reshape((batch_size, K, num_joints, 2)).transpose((0, 2, 1, 3))  # b x J x K x 2
        reg_kps = np.tile(np.expand_dims(keypoints, 3), (1, 1, 1, K, 1))
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(joint_heatmap, K=K)  # b x J x K
        if joint_offset is not None:
            hp_offset = _transpose_and_gather_feat(joint_offset, hm_inds.reshape((batch_size, -1)))
            hp_offset = hp_offset.reshape((batch_size, num_joints, K, 2))
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > heatmap_score_thresh).astype(float)
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = np.tile(np.expand_dims(np.stack([hm_xs, hm_ys], axis=-1), 2), (1, 1, K, 1, 1))
        dist = (((reg_kps - hm_kps) ** 2).sum(axis=4) ** 0.5)
        min_ind = np.argmin(dist, axis=3)  # b x J x K
        min_dist = np.take_along_axis(dist, np.expand_dims(min_ind, axis=3), axis=3)
        hm_score = np.expand_dims(np.take_along_axis(hm_score, min_ind, axis=2), -1)  # b x J x K x 1
        min_ind = np.tile(min_ind.reshape((batch_size, num_joints, K, 1, 1)), 2)
        hm_kps = np.take_along_axis(hm_kps, min_ind, axis=3)
        hm_kps = hm_kps.reshape((batch_size, num_joints, K, 2))
        left = np.tile(bboxes[:, :, 0].reshape((batch_size, 1, K, 1)), (1, num_joints, 1, 1))
        top = np.tile(bboxes[:, :, 1].reshape((batch_size, 1, K, 1)), (1, num_joints, 1, 1))
        right = np.tile(bboxes[:, :, 2].reshape((batch_size, 1, K, 1)), (1, num_joints, 1, 1))
        bottom = np.tile(bboxes[:, :, 3].reshape((batch_size, 1, K, 1)), (1, num_joints, 1, 1))
        mask = (hm_kps[..., 0:1] < left) + (hm_kps[..., 0:1] > right) + \
            (hm_kps[..., 1:2] < top) + (hm_kps[..., 1:2] > bottom) + \
            (hm_score < heatmap_score_thresh) + (min_dist > (np.maximum(bottom - top, right - left) * 0.3))
        mask = np.tile((mask > 0).astype(float), 2)
        keypoints = (1 - mask) * hm_kps + mask * keypoints
        keypoints = keypoints.transpose((0, 2, 1, 3)).reshape(batch_size, K, num_joints * 2)

    for batch_index in range(batch_size):
        box, keypoint = bboxes[batch_index], keypoints[batch_index]
        center, scale = centers[batch_index], scales[batch_index]

        box = transform_preds(box.reshape(-1, 2),
                              center, scale, (out_height, out_width)).reshape(-1, 4)
        keypoint = transform_preds(keypoint.reshape(-1, 2),
                                   center, scale, (out_height, out_width)).reshape(-1, 34)

        bboxes[batch_index], keypoints[batch_index] = box, keypoint
    return [bboxes, scores, keypoints, hm_score.transpose((0, 2, 1, 3))]


def centerpose_postprocessing(endnodes, device_pre_post_layers=None, gt_images=None, integrated_postprocessing=None,
                              **kwargs):
    center_heatmap, center_wh, joint_center_offset, center_offset, joint_heatmap, joint_offset = endnodes

    if not integrated_postprocessing or not integrated_postprocessing.get('enabled', True):
        center_heatmap = _nms(center_heatmap)
        joint_heatmap = _nms(joint_heatmap)

    bboxes, scores, keypoints, joint_scores = tf.py_function(_centerpose_postprocessing,
                                                             [center_heatmap, center_wh,
                                                              joint_heatmap, center_offset,
                                                              joint_center_offset, joint_offset,
                                                              gt_images["center"], gt_images["scale"]],
                                                             [tf.float32, tf.float32, tf.float32, tf.float32],
                                                             name='centerpose_postprocessing')

    return [bboxes, scores, keypoints, joint_scores]
