from __future__ import annotations

import math

from actrack.types import BBox


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    return math.hypot(ax - bx, ay - by)

