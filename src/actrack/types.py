from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from typing import Any


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clamp(self, width: int, height: int) -> "BBox":
        x1 = max(0.0, min(self.x, float(width - 1)))
        y1 = max(0.0, min(self.y, float(height - 1)))
        x2 = max(x1 + 1.0, min(self.x + self.w, float(width)))
        y2 = max(y1 + 1.0, min(self.y + self.h, float(height)))
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def expanded(self, ratio: float) -> "BBox":
        cx, cy = self.center()
        nw, nh = self.w * ratio, self.h * ratio
        return BBox(cx - nw / 2.0, cy - nh / 2.0, nw, nh)

    @staticmethod
    def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> "BBox":
        return BBox(x=x1, y=y1, w=max(1.0, x2 - x1), h=max(1.0, y2 - y1))


@dataclass
class CandidatePrediction:
    bbox: BBox
    score: float
    source: str = "tracker"


@dataclass
class SegmentationDetection:
    bbox: BBox
    score: float
    mask: np.ndarray | None = None
    source: str = "sam3_detect"


@dataclass
class TrackerOutput:
    best: CandidatePrediction
    candidates: list[CandidatePrediction]
    effective_search_region: BBox | None = None


@dataclass
class MotionPrediction:
    predicted_bbox: BBox
    search_region: BBox
    fallback_used: bool = False


@dataclass
class FrameResult:
    frame_id: int
    bbox: BBox
    tracker_score: float
    state: str
    extra: dict[str, Any] = field(default_factory=dict)
