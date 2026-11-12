from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from actrack.types import BBox, TrackerOutput


class BaseTracker(ABC):
    @abstractmethod
    def initialize(self, image: np.ndarray, init_bbox: BBox) -> None:
        raise NotImplementedError

    @abstractmethod
    def track(self, image: np.ndarray, search_region: BBox | None, top_k: int) -> TrackerOutput:
        raise NotImplementedError


    def get_last_effective_search_region(self) -> BBox | None:
        return None
