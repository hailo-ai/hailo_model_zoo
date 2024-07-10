import cv2
import numpy as np

from hailo_model_zoo.core.factory import POSTPROCESS_FACTORY

pose_kpt_color = np.array(
    [
        [0, 255, 0],
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
        [255, 128, 0],
    ]
)


skeleton = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]


pose_link_color = np.array(
    [
        [255, 128, 0],
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
        [0, 255, 0],
    ]
)


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
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim == 4, "batch_images should be 4-ndim"

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    # we factor np.eye(2) by 10^-6 because originally we had a case of a
    # too small epsilon, resulting in a non inversible matrix.
    hessian = np.linalg.inv(hessian + 1e-6 * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


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
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / max(np.max(heatmaps[i, j]), eps)
    return heatmaps


def bbox_xyxy2cs(bbox, orig_height, orig_width, aspect_ratio, padding=1.25, pixel_std=200.0):
    assert bbox.shape[1] == 1, f"Expected a single box per image but got {bbox.shape[1]}"
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


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


@POSTPROCESS_FACTORY.register(name="vit_pose")
def vit_pose_postprocessing(endnodes, device_pre_post_layers=None, kernel=11, **kwargs):
    img_info = kwargs["gt_images"]
    center = img_info["center"]
    scale = img_info["scale"]

    endnodes_ = endnodes.transpose(0, 3, 1, 2)
    N, K, H, W = endnodes_.shape
    preds, maxvals = _get_max_preds(endnodes_)
    preds = post_dark_udp(preds, endnodes_, kernel=kernel)

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=True)

    all_preds = np.zeros((N, preds.shape[1], 3), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals

    return {"predictions": all_preds}
