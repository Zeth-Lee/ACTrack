from __future__ import annotations

import numpy as np

from actrack.types import BBox
from actrack.utils.geometry import bbox_iou, center_distance


def success_auc(preds: list[BBox], gts: list[BBox], bins: int = 21) -> float:
    n = min(len(preds), len(gts))
    if n == 0:
        return 0.0
    ious = np.array([bbox_iou(preds[i], gts[i]) for i in range(n)], dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, bins)
    success = [(ious >= t).mean() for t in thresholds]
    return float(np.mean(success))


def precision_at_threshold(preds: list[BBox], gts: list[BBox], dist_thr: float = 20.0) -> float:
    n = min(len(preds), len(gts))
    if n == 0:
        return 0.0
    dists = np.array([center_distance(preds[i], gts[i]) for i in range(n)], dtype=np.float32)
    return float((dists <= dist_thr).mean())

