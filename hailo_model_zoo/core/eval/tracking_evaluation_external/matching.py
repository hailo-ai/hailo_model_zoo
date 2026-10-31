import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from hailo_model_zoo.core.eval.tracking_evaluation_external import kalman_filter
from hailo_model_zoo.core.eval.widerface_evaluation_external.python_box_overlaps import bbox_overlaps


def merge_matches(m1, m2, shape):
    O, P, Q = shape  # noqa: E741
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1], strict=True))
    unmatched_O = tuple(set(range(O)) - {i for i, j in match})
    unmatched_Q = tuple(set(range(Q)) - {j for i, j in match})

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices, strict=True))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """Solve the assignment problem with ``lap.lapjv(extend_cost=True, cost_limit=thresh)`` semantics.

    Leaving a row or a column unmatched costs ``thresh / 2``, so a pair is taken only when it is
    cheaper than leaving both sides open, not whenever it costs less than ``thresh``.

    Args:
        cost_matrix: 2D array of pairwise costs (tracks x detections).
        thresh: Maximum cost of an accepted pair.

    Returns:
        Tuple of (matches, unmatched_a, unmatched_b), where matches is an
        (N, 2) array of (row, column) index pairs.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    n_rows, n_cols = cost_matrix.shape
    extended = np.full((n_rows + n_cols, n_rows + n_cols), thresh / 2.0)
    extended[n_rows:, n_cols:] = 0
    extended[:n_rows, :n_cols] = np.where(np.isfinite(cost_matrix), cost_matrix, thresh + 1.0)
    rows, cols = linear_sum_assignment(extended)
    paired = (rows < n_rows) & (cols < n_cols)
    matches = np.stack((rows[paired], cols[paired]), axis=1)
    unmatched_a = np.setdiff1d(np.arange(n_rows), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(n_cols), matches[:, 1])
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_overlaps(np.ascontiguousarray(atlbrs, dtype=float), np.ascontiguousarray(btlbrs, dtype=float))

    return ious


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Normalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric="maha")
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """Scale IoU similarity by detection confidence (ByteTrack)."""
    if cost_matrix.size == 0:
        return cost_matrix
    det_scores = np.array([det.score for det in detections])[None, :]
    return 1 - (1 - cost_matrix) * det_scores
