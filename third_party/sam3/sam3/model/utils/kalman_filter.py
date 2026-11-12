import numpy as np


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]

        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )

        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        projected_mean, projected_cov = self.project(mean, covariance)

        PHt = np.dot(covariance, self._update_mat.T)
        kalman_gain = np.linalg.solve(projected_cov.T, PHt.T).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )

        return new_mean, new_covariance

    def xyxy_to_xyah(self, bbox):
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            h = 1
        return [xc, yc, w / h, h]

    def xyah_to_xyxy(self, bbox):
        xc, yc, a, h = bbox
        x1 = xc - a * h / 2
        y1 = yc - h / 2
        x2 = xc + a * h / 2
        y2 = yc + h / 2
        return [x1, y1, x2, y2]

    def _compute_iou(self, bbox1, bbox2):
        if bbox2 == [0, 0, 0, 0]:
            return 0.0

        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        intersection_area = max(0.0, min(x2, x2_) - max(x1, x1_)) * max(
            0.0, min(y2, y2_) - max(y1, y1_)
        )

        union_area = (
            (x2 - x1) * (y2 - y1)
            + (x2_ - x1_) * (y2_ - y1_)
            - intersection_area
        )

        return float(intersection_area / union_area) if union_area != 0 else 0.0

    def compute_iou(self, pred_bbox_xyah, bboxes_xyxy):
        pred_bbox_xyxy = self.xyah_to_xyxy(pred_bbox_xyah)
        return [self._compute_iou(pred_bbox_xyxy, b) for b in bboxes_xyxy]
