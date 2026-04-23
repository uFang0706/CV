import numpy as np
from collections import deque
import copy
import scipy.linalg


class BaseTrack:
    _count = 0

    @classmethod
    def reset_count(cls):
        cls._count = 0

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, value):
        self._track_id = value


class TrackState:
    Tracked = 0
    Lost = 1
    Removed = 2


class STrack(BaseTrack):
    shared_kalman = None

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.Tracked

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, use_kalman=True):
        self.frame_id = frame_id
        self.tracklet_len += 1
        if use_kalman:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        else:
            self.mean = self.tlwh_to_xyah(new_track.tlwh)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def xcycwh_to_tlbr(xcycwh):
        ret = np.asarray(xcycwh).copy()
        ret[0] = xcycwh[0] - xcycwh[2] // 2
        ret[1] = xcycwh[1] - xcycwh[3] // 2
        ret[2] = xcycwh[0] + xcycwh[2] // 2
        ret[3] = xcycwh[1] + xcycwh[3] // 2
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class STrackEmbeding(STrack):
    def __init__(self, tlwh, score, feat=None):
        super().__init__(tlwh, score)
        self.curr_feat = None
        self.smooth_feat = None
        self.alpha = 0.1
        if feat is not None:
            self.update_feature(feat)

    def update_feature(self, feat):
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    def re_activate(self, new_track, frame_id, new_id=False):
        super().re_activate(new_track, frame_id, new_id)
        if new_track.curr_feat is not None:
            self.update_feature(new_track.curr_feat)

    def update(self, new_track, frame_id, use_kalman=True):
        super().update(new_track, frame_id, use_kalman)
        if new_track.curr_feat is not None:
            self.update_feature(new_track.curr_feat)


class KalmanFilter:
    def __init__(self):
        ndim = 4
        dt = 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._motion_cov = np.eye(2 * ndim)
        self._motion_cov[ndim:, ndim:] *= 1e-4
        self._measurement_cov = np.eye(ndim) * 1e-1

    def initiate(self, measurement):
        mean_pos = measurement[:4]
        mean_vel = np.zeros(4)
        mean = np.r_[mean_pos, mean_vel]
        covariance = np.eye(8) * 1e-4

        return mean, covariance

    def predict(self, mean, covariance):
        motion_cov = self._motion_cov.copy()
        motion_cov[4:, 4:] *= 1e3
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T]) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        try:
            chol = np.linalg.cholesky(covariance)
            if np.any(np.isnan(chol)):
                return mean, covariance
        except np.linalg.LinAlgError:
            return mean, covariance

        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=lower, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T])
        return new_mean, new_covariance

    def multi_predict(self, mean, covariance):
        if len(mean) > 0:
            motion_cov = self._motion_cov.copy()
            motion_cov[4:, 4:] *= 1e3
            mean = np.dot(self._motion_mat, mean.T).T
            covariance = np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T]) + motion_cov

        return mean, covariance


def iou_distance(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    def box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = box_a_area + box_b_area - inter_area
        return inter_area / (union_area + 1e-6)

    distances = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            distances[i, j] = 1 - box_iou(box_a, box_b)

    return distances


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int).tolist(), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches = []
    unmatched_a = []
    unmatched_b = []

    return matches, unmatched_a, unmatched_b
