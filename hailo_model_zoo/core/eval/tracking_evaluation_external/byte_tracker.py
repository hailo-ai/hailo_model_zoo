"""BYTE association, ported from https://github.com/ifzhang/ByteTrack (MIT) yolox/tracker/byte_tracker.py."""

import numpy as np

from hailo_model_zoo.core.eval.tracking_evaluation_external import matching
from hailo_model_zoo.core.eval.tracking_evaluation_external.kalman_filter import KalmanFilter
from hailo_model_zoo.core.eval.tracking_evaluation_external.tracking_classes import STrack as ReIDSTrack
from hailo_model_zoo.core.eval.tracking_evaluation_external.tracking_classes import (
    TrackState,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)

LOW_SCORE_THRESHOLD = 0.1
SECOND_MATCH_THRESHOLD = 0.5
UNCONFIRMED_MATCH_THRESHOLD = 0.7


class STrack(ReIDSTrack):
    """A ReIDSTrack without appearance features."""

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

    def re_activate(self, new_track, frame_id):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score


class BYTETracker:
    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.9, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilter()

    def update(self, bboxes, scores):
        """Advance one frame.

        Args:
            bboxes: (N, 4) x1y1x2y2 boxes in original-image pixels.
            scores: (N,) detection confidences.

        Returns:
            The confirmed tracks alive after this frame.
        """
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        high = scores > self.track_thresh
        second = (scores > LOW_SCORE_THRESHOLD) & (scores < self.track_thresh)
        detections = [STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(bboxes[high], scores[high], strict=True)]

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]

        # First association: high-score detections against tracked and lost tracks.
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.fuse_score(matching.iou_distance(strack_pool, detections), detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        for itracked, idet in matches:
            self._apply_match(strack_pool[itracked], detections[idet], activated_stracks, refind_stracks)

        # Second association: low-score detections against the still-unmatched tracked tracks.
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(b), s) for b, s in zip(bboxes[second], scores[second], strict=True)
        ]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=SECOND_MATCH_THRESHOLD)
        for itracked, idet in matches:
            self._apply_match(r_tracked_stracks[itracked], detections_second[idet], activated_stracks, refind_stracks)
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = matching.fuse_score(matching.iou_distance(unconfirmed, detections), detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=UNCONFIRMED_MATCH_THRESHOLD)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            unconfirmed[it].mark_removed()
            removed_stracks.append(unconfirmed[it])

        for inew in u_detection:
            track = detections[inew]
            if track.score >= self.det_thresh:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        return [t for t in self.tracked_stracks if t.is_activated]

    def _apply_match(self, track, det, activated_stracks, refind_stracks):
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        else:
            track.re_activate(det, self.frame_id)
            refind_stracks.append(track)
