import numpy as np
import tensorflow as tf


class LaneAFPostProc(object):
    def __init__(self, **kwargs):
        return

    def _decode_wrapper(self, masks_out, vaf, haf, fg_thresh=128, err_thresh=5):
        results = []
        for i, mask_out in enumerate(masks_out):
            result = self._decodeAFs(mask_out[:, :, 0], vaf[i], haf[i], fg_thresh=128, err_thresh=5)
            results.append(result)
        return np.array(results, dtype=np.uint8)

    def _decodeAFs(self, BW, VAF, HAF, fg_thresh=128, err_thresh=5):
        output = np.zeros_like(BW, dtype=np.uint8)  # initialize output array
        lane_end_pts = []  # keep track of latest lane points
        next_lane_id = 1  # next available lane ID

        # start decoding from last row to first
        for row in range(BW.shape[0] - 1, -1, -1):
            cols = np.where(BW[row, :] > fg_thresh)[0]  # get fg cols
            clusters = [[]]
            if cols.size > 0:
                prev_col = cols[0]

            # parse horizontally
            for col in cols:
                if col - prev_col > err_thresh:  # if too far away from last point
                    clusters.append([])
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                if HAF[row, prev_col] >= 0 and HAF[row, col] >= 0:  # keep moving to the right
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] >= 0 and HAF[row, col] < 0:  # found lane center, process VAF
                    clusters[-1].append(col)
                    prev_col = col
                elif HAF[row, prev_col] < 0 and HAF[row, col] >= 0:  # found lane end, spawn new lane
                    clusters.append([])
                    clusters[-1].append(col)
                    prev_col = col
                    continue
                elif HAF[row, prev_col] < 0 and HAF[row, col] < 0:  # keep moving to the right
                    clusters[-1].append(col)
                    prev_col = col
                    continue

            # parse vertically
            # assign existing lanes
            assigned = [False for _ in clusters]
            C = np.Inf * np.ones((len(lane_end_pts), len(clusters)), dtype=np.float64)
            for r, pts in enumerate(lane_end_pts):  # for each end point in an active lane
                for c, cluster in enumerate(clusters):
                    if len(cluster) == 0:
                        continue
                    # mean of current cluster
                    cluster_mean = np.array([[np.mean(cluster), row]], dtype=np.float32)
                    # get vafs from lane end points
                    vafs = np.array([VAF[int(round(x[1])), int(round(x[0])), :] for x in pts], dtype=np.float32)
                    vafs = vafs / np.linalg.norm(vafs, axis=1, keepdims=True)
                    # get predicted cluster center by adding vafs
                    pred_points = pts + vafs * np.linalg.norm(pts - cluster_mean, axis=1, keepdims=True)
                    # get error between predicated cluster center and actual cluster center
                    error = np.mean(np.linalg.norm(pred_points - cluster_mean, axis=1))
                    C[r, c] = error
            # assign clusters to lane (in ascending order of error)
            row_ind, col_ind = np.unravel_index(np.argsort(C, axis=None), C.shape)
            for r, c in zip(row_ind, col_ind):
                if C[r, c] >= err_thresh:
                    break
                if assigned[c]:
                    continue
                assigned[c] = True
                # update best lane match with current pixel
                output[row, clusters[c]] = r + 1
                lane_end_pts[r] = np.stack(
                    (np.array(clusters[c], dtype=np.float32), row * np.ones_like(clusters[c])), axis=1
                )
            # initialize unassigned clusters to new lanes
            for c, cluster in enumerate(clusters):
                if len(cluster) == 0:
                    continue
                if not assigned[c]:
                    output[row, cluster] = next_lane_id
                    lane_end_pts.append(
                        np.stack((np.array(cluster, dtype=np.float32), row * np.ones_like(cluster)), axis=1)
                    )
                    next_lane_id += 1
        return output

    def postprocessing(self, endnodes, device_pre_post_layers=None, **kwargs):
        """
        endnodes:
            [0]: haf - [B, 88, 160, 1]
            [1]: vaf - [B, 88, 160, 2]
            [2]: hm  - [B, 88, 160, 1]
        """
        haf, vaf, hm = endnodes
        hm_repeat = tf.tile(tf.math.sigmoid(hm), [1, 1, 1, 3])
        meann = tf.zeros(3, dtype=tf.float32)
        stdd = tf.ones(3, dtype=tf.float32)
        hm_images = tf.cast(255.0 * (stdd * hm_repeat + meann), tf.uint8)
        hm_images_bgr = tf.reverse(hm_images, axis=[-1])  # RGB to BGR conversion

        seg_out = tf.numpy_function(
            lambda x, y, z: self._decode_wrapper(x, y, z, fg_thresh=128, err_thresh=5),
            (hm_images_bgr, vaf, haf),
            tf.uint8,
        )
        return {"predictions": seg_out}
