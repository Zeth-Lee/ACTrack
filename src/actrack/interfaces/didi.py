from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

from actrack.interfaces.rgb import (FrameAudit, apply_recorded_seed, load_rgb,
                                    logical_frame_path, require_vlm_credentials,
                                    validate_sam_frame_order, working_directory)


def frame_path(frame):
    if isinstance(frame, dict):
        frame = frame.get("color") or next(iter(frame.values()))
    return logical_frame_path(frame)


def box_payload(region):
    if isinstance(region, (tuple, list)) and len(region) == 2 and isinstance(region[1], dict):
        region = region[0]
    if isinstance(region, (tuple, list)) and len(region) >= 3 and all(
        isinstance(point, (tuple, list)) and len(point) == 2 for point in region
    ):
        xs, ys = zip(*region)
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        raise ValueError("DiDi requires convert=rectangle")
    return tuple(float(value) for value in region)


class Tracker:
    def __init__(self, repo_root, config_path, log_dir=None):
        self.root = Path(repo_root).resolve()
        self.config = (self.root / config_path).resolve()
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        self.next_frame = None

    def init(self, frame, new=None, properties=None):
        import torch
        from actrack.config import load_config
        from actrack.types import BBox

        path = frame_path(frame)

        self.frames = sorted(item for item in path.parent.iterdir()
                             if item.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.frames = validate_sam_frame_order(self.frames)
        if self.frames.index(path) != 0:
            raise ValueError("DiDi must initialize at the first frame")
        if self.next_frame is not None:
            raise RuntimeError("DiDi tracker instances process exactly one sequence")
        self.index = {str(item): index for index, item in enumerate(self.frames)}
        self.sequence_name = path.parent.parent.name
        self.trace_dir = self.log_dir / self.sequence_name if self.log_dir is not None else None
        initial_values = box_payload(new)
        if not all(math.isfinite(float(value)) for value in initial_values):
            raise ValueError("DiDi initialization requires finite xywh values")
        box = BBox(*initial_values)
        with working_directory(self.root), torch.inference_mode():
            cfg = load_config(self.config)
            require_vlm_credentials(cfg)
            if self.trace_dir is not None:
                self.trace_dir.mkdir(parents=True, exist_ok=False)
            apply_recorded_seed(cfg)
            from actrack.pipeline import ACTrackPipeline

            self.pipe = ACTrackPipeline(cfg, sequence_name=self.sequence_name)
            self.pipe.motion.prepare_sequence(
                frame_paths=[str(item) for item in self.frames], frame_dir=str(path.parent))
            result = self.pipe.initialize(load_rgb(path), box)
        save_logs = bool(getattr(getattr(cfg, "experiment", None), "save_logs", True))
        self.audit = FrameAudit(self.trace_dir / "frame_logs.jsonl" if self.trace_dir is not None and save_logs else None)
        self.audit.record(0, (box.x, box.y, box.w, box.h), 1.0, result)
        self.next_frame = 1
        self._write_audit(0)
        return (box.x, box.y, box.w, box.h), {"confidence": 1.0}

    def update(self, frame, new=None, properties=None):
        import torch

        path = frame_path(frame)
        if self.next_frame is None:
            raise RuntimeError("DiDi must initialize before update")
        frame_id = self.index.get(str(path))
        if frame_id != self.next_frame:
            raise ValueError(f"DiDi requires the next sequential frame (index {self.next_frame})")
        with working_directory(self.root), torch.inference_mode():
            result = self.pipe.step(load_rgb(path), frame_id=frame_id)
        box = result.bbox
        self.audit.record(frame_id, (box.x, box.y, box.w, box.h), float(result.tracker_score), result)
        self.next_frame += 1
        self._write_audit(frame_id)
        return (float(box.x), float(box.y), float(box.w), float(box.h)), {
            "confidence": float(result.tracker_score)}

    def _write_audit(self, frame_id):
        if self.trace_dir is not None:
            complete = frame_id == len(self.frames) - 1 and self.audit.frames == len(self.frames)
            receipt = {"sequence": self.sequence_name,
                       "frames_expected": len(self.frames), "frames_recorded": self.audit.frames,
                       "complete": complete,
                       "frame_logs_sha256": hashlib.sha256(self.audit.log_path.read_bytes()).hexdigest()
                       if complete and self.audit.log_path is not None else None,
                       **self.audit.summary()}
            temporary = self.trace_dir / "AUDIT.json.tmp"
            temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.trace_dir / "AUDIT.json")


def create_tracker(repo_root, config_path, log_dir=None):
    return Tracker(repo_root, config_path, log_dir=log_dir)
