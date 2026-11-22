from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from actrack.types import BBox, MotionPrediction, SegmentationDetection


class BaseMotionModel(ABC):
    def prepare_sequence(self, frame_paths: list[str] | None = None, frame_dir: str | None = None) -> None:

        return None

    @abstractmethod
    def initialize(self, init_bbox: BBox) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_width: int, image_height: int) -> MotionPrediction:
        raise NotImplementedError

    def detect_similar_objects(
        self,
        image: np.ndarray,
        exemplar_bbox: BBox,
        search_region: BBox | None = None,
    ) -> list[SegmentationDetection]:
        _ = image, exemplar_bbox, search_region
        return []

    @abstractmethod
    def update(self, bbox: BBox) -> None:
        raise NotImplementedError


    @abstractmethod
    def reprompt(self, bbox: BBox) -> None:
        raise NotImplementedError
