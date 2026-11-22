from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path


@contextmanager
def working_directory(path):

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_rgb(path):

    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _box_values(box):
    values = tuple(float(value) for value in (box.x, box.y, box.w, box.h))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The pipeline returned a non-finite bounding box")
    return values


def logical_frame_path(path):

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def validate_sam_frame_order(frame_paths):

    frames = [logical_frame_path(path) for path in frame_paths]
    if not frames:
        raise ValueError("The sequence has no frames")
    if len(set(frames)) != len(frames):
        raise ValueError("Duplicate frame paths are not allowed")
    directory = frames[0].parent
    if any(frame.parent != directory for frame in frames):
        raise ValueError("Frozen SAM3 requires all frames in one directory")
    extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    if any(frame.suffix not in extensions for frame in frames):
        raise ValueError("Frozen SAM3 requires JPEG frames (.jpg/.jpeg/.JPG/.JPEG)")
    actual = [path for path in directory.iterdir() if path.suffix in extensions]
    for frame in actual:
        if not frame.is_file():
            raise FileNotFoundError(frame)
    try:
        indices = [int(frame.stem) for frame in actual]
    except ValueError as exc:
        raise ValueError("Frozen SAM3 requires integer JPEG frame names") from exc
    if len(set(indices)) != len(indices):
        raise ValueError("JPEG filenames must have unique integer frame indices")
    expected = sorted(actual, key=lambda path: int(path.stem))
    if frames != expected:
        raise ValueError("Frames must contain the complete directory in SAM3 integer-filename order")
    return frames


def apply_recorded_seed(config):

    seed = getattr(getattr(config, "experiment", None), "random_seed", None)
    if seed is None:
        return
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_vlm_credentials(config):

    agent = getattr(config, "agent", None)
    if not bool(getattr(agent, "vlm_reprompt_gate_enabled", False)):
        return
    configured = str(getattr(agent, "vllm_api_key", "") or "").strip()
    environment_name = str(getattr(agent, "vllm_api_key_env", "ARK_API_KEY")).strip()
    if not configured and not os.environ.get(environment_name, "").strip():
        raise RuntimeError(f"VLM gate is enabled but no API key is configured; set {environment_name} in the environment")


def _json_value(value):

    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize frame diagnostic type {type(value).__name__}")


class FrameAudit:

    def __init__(self, log_path=None):
        self.log_path = Path(log_path) if log_path is not None else None
        self.frames = 0
        self.counts = {"gate_triggers": 0, "vlm_decision_calls": 0,
                       "vlm_call_skips": 0, "reprompts": 0, "api_failures": 0}

    def record(self, frame_id, bbox, tracker_score, result):
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
            raise ValueError("Frame audit requires four finite xywh values")
        if not math.isfinite(float(tracker_score)):
            raise ValueError("Frame audit requires a finite tracker confidence")
        extra = getattr(result, "extra", {}) or {}
        gate = extra.get("vlm_reprompt_gate", {}) or {}
        triggered = bool(gate.get("triggered", False))
        skipped = bool(gate.get("vlm_call_skipped_by_loop", False))
        self.counts["gate_triggers"] += int(triggered)
        self.counts["vlm_call_skips"] += int(triggered and skipped)
        self.counts["vlm_decision_calls"] += int(triggered and not skipped)
        self.counts["reprompts"] += int(bool(gate.get("reprompt_applied", False)))
        failures = extra.get("api_failures", []) or []
        self.counts["api_failures"] += len(failures) if isinstance(failures, (list, tuple)) else 1
        self.frames += 1
        if self.log_path is not None:
            record = {"frame_id": frame_id, "bbox": list(bbox), "tracker_score": tracker_score,
                      "state": getattr(result, "state", "init" if frame_id == 0 else None), "extra": extra}
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=_json_value) + "\n")

    def summary(self):
        return {"audit": dict(self.counts), "valid_for_reproduction": self.counts["api_failures"] == 0,
                "audit_note": "VLM decision calls exclude direct-loop skips; HTTP retry attempts are not counted here"}


def run_sequence(config_path, frame_paths, init_bbox, output_dir, *,
                 sequence_name=None, repo_root=None):

    root = Path(repo_root or Path(__file__).resolve().parents[3]).resolve()
    config = Path(config_path).expanduser()
    config = (config if config.is_absolute() else root / config).resolve()
    frames = validate_sam_frame_order(frame_paths)
    if len(init_bbox) != 4 or not all(math.isfinite(float(x)) for x in init_bbox):
        raise ValueError("init_bbox must contain four finite xywh values")
    if not config.is_file():
        raise FileNotFoundError(config)
    for frame in frames:
        if not frame.is_file():
            raise FileNotFoundError(frame)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    import torch
    from actrack.config import load_config
    from actrack.types import BBox

    name = sequence_name or frames[0].parent.name
    with working_directory(root), torch.inference_mode():
        cfg = load_config(config)
        require_vlm_credentials(cfg)
        apply_recorded_seed(cfg)
        from actrack.pipeline import ACTrackPipeline

        pipeline = ACTrackPipeline(cfg, sequence_name=name)
        pipeline.motion.prepare_sequence(
            frame_paths=[str(frame) for frame in frames],
            frame_dir=str(frames[0].parent),
        )
        save_logs = bool(getattr(getattr(cfg, "experiment", None), "save_logs", True))
        audit = FrameAudit(output / "frame_logs.jsonl" if save_logs else None)
        box = BBox(*(float(value) for value in init_bbox))
        with (output / "predictions.txt.partial").open("w", encoding="utf-8") as predictions, \
                (output / "confidence.txt.partial").open("w", encoding="utf-8") as confidences:
            for frame_id, path in enumerate(frames):
                frame = load_rgb(path)
                if frame_id == 0:
                    result = pipeline.initialize(frame, box)
                    values, confidence = _box_values(box), 1.0
                else:
                    result = pipeline.step(frame, frame_id=frame_id)
                    values, confidence = _box_values(result.bbox), float(result.tracker_score)
                if not math.isfinite(confidence):
                    raise ValueError("The pipeline returned a non-finite confidence")
                predictions.write(",".join(format(value, ".17g") for value in values) + "\n")
                confidences.write(format(confidence, ".17g") + "\n")
                audit.record(frame_id, values, confidence, result)

    (output / "predictions.txt.partial").rename(output / "predictions.txt")
    (output / "confidence.txt.partial").rename(output / "confidence.txt")
    receipt = {
        "sequence": name,
        "frames": len(frames),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256((output / "predictions.txt").read_bytes()).hexdigest(),
        "confidence_sha256": hashlib.sha256((output / "confidence.txt").read_bytes()).hexdigest(),
        "frame_logs_sha256": hashlib.sha256((output / "frame_logs.jsonl").read_bytes()).hexdigest() if save_logs else None,
        **audit.summary(),
    }
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
