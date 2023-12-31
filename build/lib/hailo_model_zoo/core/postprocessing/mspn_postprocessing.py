import numpy as np
import cv2

from hailo_model_zoo.core.preprocessing.affine_utils import transform_preds


pose_kpt_color = np.array([[0, 255, 0],
                           [0, 255, 0],
                           [0, 255, 0],
                           [0, 255, 0],
                           [0, 255, 0],
                           [51, 153, 255],
                           [51, 153, 255],
                           [51, 153, 255],
                           [51, 153, 255],
                           [51, 153, 255],
                           [51, 153, 255],
                           [255, 128, 0],
                           [255, 128, 0],
                           [255, 128, 0],
                           [255, 128, 0],
                           [255, 128, 0],
                           [255, 128, 0]])


skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
            [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]


pose_link_color = np.array([[255, 128, 0],
                            [255, 128, 0],
                            [255, 128, 0],
                            [255, 128, 0],
                            [255, 51, 255],
                            [255, 51, 255],
                            [255, 51, 255],
                            [51, 153, 255],
                            [51, 153, 255],
                            [51, 153, 255],
                            [51, 153, 255],
                            [51, 153, 255],
                            [0, 255, 0],
                            [0, 255, 0],
                            [0, 255, 0],
                            [0, 255, 0],
                            [0, 255, 0],
                            [0, 255, 0],
                            [0, 255, 0]])


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _gaussian_blur(heatmaps, kernel=11, eps=1e-12):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / max(np.max(heatmaps[i, j]), eps)
    return heatmaps


def bbox_xyxy2cs(bbox, orig_height, orig_width, aspect_ratio, padding=1.25, pixel_std=200.):
    assert bbox.shape[1] == 1, \
        f"Expected a single box per image but got {bbox.shape[1]}"
    box = np.squeeze(bbox.copy(), axis=1)
    xmin, xmax, ymin, ymax = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    xmin *= orig_width
    xmax *= orig_width
    ymin *= orig_height
    ymax *= orig_height
    width = xmax - xmin
    height = ymax - ymin

    center = np.array([xmin + width * 0.5, ymin + height * 0.5], dtype=np.float32).T
    for i, (w, h) in enumerate(zip(width, height)):
        if w > aspect_ratio * h:
            height[i] = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            width[i] = height[i] * aspect_ratio
    scale = np.array([width, height], dtype=np.float32).T / pixel_std
    scale = scale * padding

    return center, scale


def _get_default_bbox(batch_size):
    box = np.array([[0, 1, 0, 1]])
    default_bbox = np.expand_dims(np.vstack([box for _ in range(batch_size)]), axis=1)
    return default_bbox  # Bx1x4


def mspn_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    image_info = kwargs['gt_images']
    height, width = image_info['img_resized'].shape[1:3]
    aspect_ratio = width / height

    # Get box info if exists, otherwise assume box spans the entire image
    bbox = image_info.get('bbox', _get_default_bbox(endnodes.shape[0]))
    center, scale = bbox_xyxy2cs(bbox, image_info['orig_height'], image_info['orig_width'], aspect_ratio)
    heatmaps = np.transpose(endnodes, axes=[0, 3, 1, 2])

    heatmaps = _gaussian_blur(heatmaps, kernel=5)
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)
    for n in range(N):
        for k in range(K):
            heatmap = heatmaps[n][k]
            px = int(preds[n][k][0])
            py = int(preds[n][k][1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                preds[n][k] += np.sign(diff) * .25
                preds[n][k] += 0.5

    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i],
                                   [W, H], pixel_std=200.0
                                   )

    maxvals = maxvals / 255.0 + 0.5
    all_preds = np.zeros((N, preds.shape[1], 3), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals

    return {'predictions': all_preds}


def visualize_single_person_pose_estimation_result(probs, image, kpt_score_thr=0.3,
                                                   radius=8, thickness=2, **kwargs):

    idx = 0
    img = cv2.cvtColor(image[idx], cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape
    probs = probs['predictions'][idx]
    kpts = np.array(probs, copy=False)

    # draw keypoint on image
    if pose_kpt_color is not None:
        assert len(pose_kpt_color) == len(kpts)

        for kid, kpt in enumerate(kpts):
            x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

            if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                # skip the point that should not be drawn
                continue

            color = tuple(int(c) for c in pose_kpt_color[kid])
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

    # draw links
    if skeleton is not None and pose_link_color is not None:
        assert len(pose_link_color) == len(skeleton)

        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

            if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                    or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                    or pos2[1] <= 0 or pos2[1] >= img_h
                    or kpts[sk[0], 2] < kpt_score_thr
                    or kpts[sk[1], 2] < kpt_score_thr
                    or pose_link_color[sk_id] is None):
                # skip the link that should not be drawn
                continue
            color = tuple(int(c) for c in pose_link_color[sk_id])
            cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img
