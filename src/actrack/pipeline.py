from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from actrack.agent import (
    ArkVLLMClient,
    BaseVLLMClient,
)
from actrack.config import PipelineConfig
from actrack.motion import BaseMotionModel, SAM3MotionModel
from actrack.tracker import BaseTracker, MCITrackTracker
from actrack.types import (
    BBox,
    FrameResult,
    SegmentationDetection,
)
from actrack.utils.geometry import bbox_iou
from actrack.utils.image_ops import crop_image, draw_labeled_boxes


def _bbox_to_dict(box: BBox) -> dict[str, float]:
    return {
        "x": float(box.x),
        "y": float(box.y),
        "w": float(box.w),
        "h": float(box.h),
    }


def _mask_to_log_dict(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or arr.size == 0:
        return None

    flat = arr.reshape(-1)
    counts: list[int] = []
    current = False
    run_len = 0
    for value in flat:
        value_bool = bool(value)
        if value_bool == current:
            run_len += 1
        else:
            counts.append(int(run_len))
            current = value_bool
            run_len = 1
    counts.append(int(run_len))

    ys, xs = np.where(arr)
    bbox = None
    if len(xs) > 0 and len(ys) > 0:
        bbox = {
            "x": int(xs.min()),
            "y": int(ys.min()),
            "w": int(xs.max() - xs.min() + 1),
            "h": int(ys.max() - ys.min() + 1),
        }

    return {
        "encoding": "rle_c_order_start_false",
        "size": [int(arr.shape[0]), int(arr.shape[1])],
        "counts": counts,
        "positive_count": int(arr.sum()),
        "bbox": bbox,
    }


def _point_in_bbox(point: tuple[float, float], box: BBox) -> bool:
    px, py = point
    x1, y1, x2, y2 = box.as_xyxy()
    return x1 <= px <= x2 and y1 <= py <= y2


def _bbox_area(box: BBox) -> float:
    return max(0.0, float(box.w)) * max(0.0, float(box.h))


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return arr[:, :, :3]


def _draw_plain_boxes(image: np.ndarray, boxes: list[tuple[BBox, tuple[int, int, int]]]) -> np.ndarray:
    arr = _as_rgb_uint8(image)
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr.copy())
    draw = ImageDraw.Draw(pil)
    width = max(2, int(round(max(w, h) / 320.0)))
    for bbox, color in boxes:
        clipped = bbox.clamp(w, h)
        x1, y1, x2, y2 = clipped.as_xyxy()
        rect = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        for offset in range(width):
            draw.rectangle(
                [rect[0] - offset, rect[1] - offset, rect[2] + offset, rect[3] + offset],
                outline=tuple(int(c) for c in color),
            )
    return np.asarray(pil, dtype=np.uint8)


def _crop_with_local_box(
    frame: np.ndarray,
    bbox: BBox,
    color: tuple[int, int, int],
    expand_ratio: float = 2.2,
) -> np.ndarray:
    arr = _as_rgb_uint8(frame)
    h, w = arr.shape[:2]
    tight = bbox.clamp(w, h)
    context = tight.expanded(max(1.0, float(expand_ratio))).clamp(w, h)
    x1, y1, x2, y2 = context.as_xyxy()
    ix1 = max(0, min(w - 1, int(np.floor(x1))))
    iy1 = max(0, min(h - 1, int(np.floor(y1))))
    ix2 = max(ix1 + 1, min(w, int(np.ceil(x2))))
    iy2 = max(iy1 + 1, min(h, int(np.ceil(y2))))
    crop = arr[iy1:iy2, ix1:ix2].copy()
    local = BBox(tight.x - float(ix1), tight.y - float(iy1), tight.w, tight.h)
    return _draw_plain_boxes(crop, [(local, color)])


def _resize_letterbox(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    arr = _as_rgb_uint8(image)
    target_w, target_h = size
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    if arr.size == 0:
        return canvas
    pil = Image.fromarray(arr)
    src_w, src_h = pil.size
    scale = min(float(target_w) / max(1.0, float(src_w)), float(target_h) / max(1.0, float(src_h)))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    resized = pil.resize((new_w, new_h), resample)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def _header_panel(image: np.ndarray, title: str, size: tuple[int, int] = (520, 360), header_h: int = 30) -> Image.Image:
    panel_w, panel_h = size
    canvas = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, panel_w, header_h], fill=(242, 242, 242))
    draw.text((8, 8), title, fill=(32, 32, 32))
    body = _resize_letterbox(image, (panel_w, max(1, panel_h - header_h)))
    canvas.paste(body, (0, header_h))
    return canvas


@dataclass
class PipelineStats:
    total_frames: int = 0
    trigger_conflict: int = 0
    trigger_low_conf: int = 0
    vlm_reprompt_gate_calls: int = 0
    vlm_reprompt_gate_tracker: int = 0
    vlm_reprompt_gate_sam: int = 0
    vlm_reprompt_gate_uncertain: int = 0
    vlm_reprompts: int = 0


@dataclass
class _RuntimeState:
    initialized: bool = False
    confirm_streak: int = 0
    motion_win_streak: int = 0
    initial_reference_frame: np.ndarray | None = None
    sequence_name: str = "sequence"
    vlm_reprompt_branch: str = ""
    vlm_reprompt_key: str = ""
    vlm_reprompt_streak: int = 0
    vlm_reprompt_last_frame: int = -10**9
    vlm_reprompt_fired_episode: bool = False
    vlm_reprompt_direct_next_after_identity_diff_tracker: bool = False


class ACTrackPipeline:
    _VLM_REPROMPT_PROMPT = (
        "图片的左上角是确定的首帧追踪物，首先，你需要确定它是个什么物体，并保证一个准则——"
        "追踪的物体类型一定要一致，然后请你注意它的外观而非动作，作为接下来选择的依据，"
        "右上角为当前帧的全帧图片，蓝色框中是SAM追踪的bbox，绿色框是Tracker追踪的bbox，用于把控整体，"
        "下边的两张图片则是我裁切出的Tracker和SAM的裁切图，用于观察具体细节。\n"
        "现在，我需要你分析Tracker和SAM谁的选择是正确的，你需要结合四张图片你获取到的信息，"
        "做出最置信的选择，你可以选择“SAM”，“Tracker”和“Uncertain”，注意，Tracker需要你完全确定这是一个清晰的选择，"
        "如果有任何一点不清晰就要选择Uncertain，Tracker必须高置信才可以选择，不要有多余回答。\n"
        "注意：\n"
        "1.不要把动作和位置的变化作为参考因素，因为在一段长视频中，跟踪物和干扰物随时可以出现运动变化，"
        "所以如果是同类型物体，并且自身没有外观上的显著特征（如颜色，花纹等），那你就需要回答“Uncertain”\n"
        "2.请注意场景当中出现多个类似物体时，往往是SAM更能追踪到正确的物体上，所以如果你认为SAM和Tracker"
        "的特征与首帧GT的差异并不是非常显著，你需要选择“Uncertain”\n"
        "3.请注意SAM是否像是被遮挡了，因为当SAM被遮挡时，Tracker有可能漂移到类似物体上，你需要警惕这种情况，"
        "如果SAM的框有任何一点像是被遮挡的情况，请回复“Uncertain”\n"
        "4.有些情况是SAM没有分清一个物体的整体和局部，当你发现SAM和Tracker框的是同一个位置，"
        "区别只在于是否框住了尾巴等内容时，你可以直接选择“Tracker”，但是前提是一定要保证二者框住的是同一个物体"
    )

    def __init__(
        self,
        cfg: PipelineConfig,
        tracker: BaseTracker | None = None,
        motion: BaseMotionModel | None = None,
        sequence_name: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker or self._build_tracker(cfg)
        self.motion = motion or self._build_motion(cfg)

        self.vllm = self._build_vllm_client(cfg)

        self.stats = PipelineStats()
        self.state = _RuntimeState()
        self.state.sequence_name = self._safe_name(sequence_name or "sequence")

    @staticmethod
    def _build_tracker(cfg: PipelineConfig) -> BaseTracker:
        _ = cfg
        return MCITrackTracker(cfg.tracker)

    @staticmethod
    def _build_motion(cfg: PipelineConfig) -> BaseMotionModel:
        _ = cfg
        return SAM3MotionModel(cfg.motion)

    @staticmethod
    def _build_vllm_client(cfg: PipelineConfig) -> BaseVLLMClient:
        return ArkVLLMClient(cfg.agent)

    @staticmethod
    def _safe_name(raw: str) -> str:
        text = str(raw).strip()
        if not text:
            return "sequence"
        cleaned = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in text)
        cleaned = cleaned.strip("_")
        return cleaned or "sequence"


    def initialize(self, frame: np.ndarray, init_bbox: BBox) -> FrameResult:
        self.tracker.initialize(frame, init_bbox)
        self.motion.initialize(init_bbox)

        self.state.initialized = True
        self.state.confirm_streak = 1
        self.state.motion_win_streak = 0


        h, w = frame.shape[:2]
        tight_bbox = init_bbox.clamp(w, h)
        self.state.initial_reference_frame = (
            np.asarray(draw_labeled_boxes(frame, [("REF", tight_bbox, (196, 148, 28))])).copy()
            if crop_image(frame, tight_bbox).size > 0 else None
        )
        self._reset_vlm_reprompt_episode()
        self.stats.total_frames = 1
        return FrameResult(
            frame_id=0,
            bbox=init_bbox,
            tracker_score=1.0,
            state="init",
            extra={
                "sam3_conflict_detect": None,
            },
        )

    def step(self, frame: np.ndarray, frame_id: int) -> FrameResult:
        self._check_initialized()
        h, w = frame.shape[:2]

        motion_pred = self.motion.predict(w, h)


        track_out = self.tracker.track(frame, motion_pred.predicted_bbox, top_k=self.cfg.tracker.top_k_candidates)
        tracker_bbox = track_out.best.bbox
        tracker_score = track_out.best.score
        effective_search_region = (
            track_out.effective_search_region or self.tracker.get_last_effective_search_region()
        )

        api_failures: list[dict[str, Any]] = []
        accepted_bbox = tracker_bbox
        accepted_source = "tracker"
        frame_state = "track"
        motion_reset_applied = False
        sam3_conflict_detect: dict[str, Any] | None = None
        if self._should_run_sam3_conflict_detect(motion_pred=motion_pred):
            sam3_conflict_detect = self._evaluate_sam3_conflict_detect(
                frame=frame,
                tracker_bbox=tracker_bbox,
                motion_bbox=motion_pred.predicted_bbox,
                effective_search_region=effective_search_region,
            )
        diverged = bool(sam3_conflict_detect and sam3_conflict_detect.get("instance_conflict", False))
        low_conf = tracker_score < float(self.cfg.tracker.confidence_threshold)
        if low_conf:
            self.stats.trigger_low_conf += 1
        if diverged:
            self.stats.trigger_conflict += 1

        reference = self.state.initial_reference_frame
        initial_ref_frame = (
            np.asarray(reference).copy()
            if isinstance(reference, np.ndarray) and reference.size > 0
            else np.asarray([])
        )

        vlm_reprompt_gate = self._maybe_run_vlm_reprompt_gate(
            frame=frame,
            frame_id=frame_id,
            tracker_bbox=tracker_bbox,
            motion_bbox=motion_pred.predicted_bbox,
            sam3_conflict_detect=sam3_conflict_detect,
            initial_reference_frame=initial_ref_frame,
        )
        gate_api_failures = vlm_reprompt_gate.pop("api_failures", [])
        if gate_api_failures:
            api_failures.extend(gate_api_failures)
        vlm_reprompt_applied = bool(vlm_reprompt_gate.get("reprompt_applied", False))

        if vlm_reprompt_applied:
            accepted_bbox = tracker_bbox
            accepted_source = "tracker"
            frame_state = "vlm_reprompt_tracker"
            motion_reset_applied = True
            self.motion.reprompt(tracker_bbox)
            self.stats.vlm_reprompts += 1
        elif bool(sam3_conflict_detect and sam3_conflict_detect.get("override_motion", False)):
            accepted_bbox = motion_pred.predicted_bbox
            accepted_source = "motion"
            frame_state = "sam3_detect_motion"


        confirmed = True
        self.state.confirm_streak += 1
        if accepted_source == "motion":
            self.state.motion_win_streak += 1
        else:
            self.state.motion_win_streak = 0

        if not motion_reset_applied:
            self.motion.update(accepted_bbox)
        self.stats.total_frames += 1

        extra = {
            "low_conf": low_conf,
            "diverged": diverged,
            "confirm_streak": self.state.confirm_streak,
            "motion_win_streak": self.state.motion_win_streak,
            "candidate_count": len(track_out.candidates),
            "tracker_bbox": _bbox_to_dict(tracker_bbox),
            "motion_pred_bbox": _bbox_to_dict(motion_pred.predicted_bbox),
            "motion_search_region": _bbox_to_dict(motion_pred.search_region),
            "motion_fallback_used": bool(getattr(motion_pred, "fallback_used", False)),
            "sam3_fallback_used": bool(getattr(motion_pred, "fallback_used", False)),
            "tracker_effective_search_region": (
                _bbox_to_dict(effective_search_region) if effective_search_region is not None else None
            ),
            "accepted_bbox": _bbox_to_dict(accepted_bbox),
            "accepted_source": accepted_source,
            "template_confirmed": bool(confirmed),
            "motion_reanchor_confirmed": bool(confirmed),
            "sam3_conflict_detect": sam3_conflict_detect,
            "vlm_reprompt_gate": vlm_reprompt_gate,
        }
        if api_failures:
            extra["api_failures"] = list(api_failures)
        extra["confirmed"] = bool(confirmed)

        return FrameResult(
            frame_id=frame_id,
            bbox=accepted_bbox,
            tracker_score=tracker_score,
            state=frame_state,
            extra=extra,
        )


    def _should_run_sam3_conflict_detect(self, motion_pred) -> bool:
        if not bool(getattr(self.cfg.motion, "sam3_conflict_detect_enabled", True)):
            return False
        return not bool(getattr(motion_pred, "fallback_used", False))

    def _reset_vlm_reprompt_episode(self) -> None:
        self.state.vlm_reprompt_branch = ""
        self.state.vlm_reprompt_key = ""
        self.state.vlm_reprompt_streak = 0
        self.state.vlm_reprompt_last_frame = -10**9
        self.state.vlm_reprompt_fired_episode = False

    def _maybe_run_vlm_reprompt_gate(
        self,
        frame: np.ndarray,
        frame_id: int,
        tracker_bbox: BBox,
        motion_bbox: BBox,
        sam3_conflict_detect: dict[str, Any] | None,
        initial_reference_frame: np.ndarray,
    ) -> dict[str, Any]:
        enabled = bool(getattr(self.cfg.agent, "vlm_reprompt_gate_enabled", False))
        info: dict[str, Any] = {
            "enabled": enabled,
            "triggered": False,
            "reprompt_applied": False,
            "answer": None,
            "raw_answer": "",
        }
        if not enabled:
            return info

        event = self._classify_vlm_reprompt_event(
            tracker_bbox=tracker_bbox,
            motion_bbox=motion_bbox,
            sam3_conflict_detect=sam3_conflict_detect,
        )
        if event is None:
            self._reset_vlm_reprompt_episode()
            return info

        branch = str(event["branch"])
        key = str(event["key"])
        is_contiguous = int(frame_id) == int(self.state.vlm_reprompt_last_frame) + 1
        if (
            branch == self.state.vlm_reprompt_branch
            and key == self.state.vlm_reprompt_key
            and is_contiguous
        ):
            self.state.vlm_reprompt_streak += 1
        else:
            self.state.vlm_reprompt_branch = branch
            self.state.vlm_reprompt_key = key
            self.state.vlm_reprompt_streak = 1
            self.state.vlm_reprompt_fired_episode = False
        self.state.vlm_reprompt_last_frame = int(frame_id)

        info.update(event)
        info["streak"] = int(self.state.vlm_reprompt_streak)
        info["k"] = max(1, int(getattr(self.cfg.agent, "vlm_reprompt_gate_k", 5)))
        info["episode_fired"] = bool(self.state.vlm_reprompt_fired_episode)
        if self.state.vlm_reprompt_streak < int(info["k"]) or self.state.vlm_reprompt_fired_episode:
            return info

        self.state.vlm_reprompt_fired_episode = True
        info["triggered"] = True
        if self.state.vlm_reprompt_direct_next_after_identity_diff_tracker:
            self.state.vlm_reprompt_direct_next_after_identity_diff_tracker = False
            info["answer"] = "Tracker"
            info["raw_answer"] = "identity_diff_motion_tracker_loop_direct_reprompt"
            info["reprompt_applied"] = True
            info["vlm_call_skipped_by_loop"] = True
            info["direct_reprompt_after_identity_diff_tracker"] = True
            self.stats.vlm_reprompt_gate_tracker += 1
            return info

        info["vlm_call_skipped_by_loop"] = False
        info["direct_reprompt_after_identity_diff_tracker"] = False
        panel = self._build_vlm_reprompt_panel(
            frame=frame,
            tracker_bbox=tracker_bbox,
            motion_bbox=motion_bbox,
            initial_reference_frame=initial_reference_frame,
        )
        self.stats.vlm_reprompt_gate_calls += 1
        self.vllm.set_api_failure_context(
            None,
            {
                "source": "vlm_reprompt_gate",
                "sequence": self.state.sequence_name,
                "frame_id": int(frame_id),
                "branch": branch,
                "key": key,
            },
        )
        try:
            answer, raw_answer = self.vllm.choose_sam_tracker_uncertain(
                prompt=self._VLM_REPROMPT_PROMPT,
                panel_image=panel,
                max_output_tokens=int(getattr(self.cfg.agent, "vlm_reprompt_gate_max_output_tokens", 16)),
                image_max_side=int(getattr(self.cfg.agent, "vlm_reprompt_gate_image_max_side", 1100)),
            )
        finally:
            info["api_failures"] = self.vllm.consume_api_failures()
            self.vllm.clear_api_failure_context()

        info["answer"] = answer
        info["raw_answer"] = raw_answer
        if answer == "Tracker":
            self.stats.vlm_reprompt_gate_tracker += 1
            info["reprompt_applied"] = True
            if branch == "identity_diff_motion":
                self.state.vlm_reprompt_direct_next_after_identity_diff_tracker = True
                info["direct_next_after_identity_diff_tracker_set"] = True
        elif answer == "SAM":
            self.stats.vlm_reprompt_gate_sam += 1
        else:
            self.stats.vlm_reprompt_gate_uncertain += 1
        return info

    def _classify_vlm_reprompt_event(
        self,
        tracker_bbox: BBox,
        motion_bbox: BBox,
        sam3_conflict_detect: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(sam3_conflict_detect, dict) or not bool(sam3_conflict_detect.get("ran", False)):
            return None

        reason = str(sam3_conflict_detect.get("reason", "")).strip()
        detected_count = int(sam3_conflict_detect.get("detected_count_in_search_region", 0) or 0)
        tracker_assignment = sam3_conflict_detect.get("tracker_assignment", {})
        sam3_assignment = sam3_conflict_detect.get("sam3_assignment", {})
        if not isinstance(tracker_assignment, dict):
            tracker_assignment = {}
        if not isinstance(sam3_assignment, dict):
            sam3_assignment = {}
        tracker_instance = tracker_assignment.get("instance_id")
        sam3_instance = sam3_assignment.get("instance_id")
        tracker_sam_iou = float(bbox_iou(tracker_bbox, motion_bbox))
        area_ratio = _bbox_area(tracker_bbox) / max(_bbox_area(motion_bbox), 1e-6)
        base = {
            "reason": reason,
            "detected_count": detected_count,
            "tracker_sam_iou": tracker_sam_iou,
            "tracker_sam_area_ratio": area_ratio,
            "tracker_instance": tracker_instance,
            "sam3_instance": sam3_instance,
        }

        if (
            reason == "different_detector_instances"
            and tracker_sam_iou <= 1e-9
            and detected_count >= 2
            and tracker_instance is not None
            and sam3_instance is not None
            and str(tracker_instance) != str(sam3_instance)
        ):
            return {
                **base,
                "branch": "identity_diff_motion",
                "key": f"{tracker_instance}->{sam3_instance}",
            }

        if reason == "unassigned_instance" and tracker_sam_iou <= 1e-9 and detected_count <= 2:
            return {
                **base,
                "branch": "unassigned_tracker",
                "key": str(sam3_instance),
            }

        if (
            reason == "same_detector_instance"
            and _point_in_bbox(tracker_bbox.center(), motion_bbox)
            and not _point_in_bbox(motion_bbox.center(), tracker_bbox)
        ):
            return {
                **base,
                "branch": "same_instance_extent",
                "key": str(sam3_instance),
            }
        return None

    def _build_vlm_reprompt_panel(
        self,
        frame: np.ndarray,
        tracker_bbox: BBox,
        motion_bbox: BBox,
        initial_reference_frame: np.ndarray,
    ) -> np.ndarray:
        green = (34, 170, 80)
        blue = (40, 120, 255)
        initial = (
            initial_reference_frame
            if isinstance(initial_reference_frame, np.ndarray) and initial_reference_frame.size > 0
            else frame
        )
        current = _draw_plain_boxes(frame, [(tracker_bbox, green), (motion_bbox, blue)])
        tracker_crop = _crop_with_local_box(frame, tracker_bbox, green)
        motion_crop = _crop_with_local_box(frame, motion_bbox, blue)

        panels = [
            _header_panel(initial, "First frame target"),
            _header_panel(current, "Current frame: green=Tracker, blue=SAM"),
            _header_panel(tracker_crop, "Tracker crop"),
            _header_panel(motion_crop, "SAM crop"),
        ]
        cell_w, cell_h = panels[0].size
        canvas = Image.new("RGB", (cell_w * 2, cell_h * 2), (255, 255, 255))
        canvas.paste(panels[0], (0, 0))
        canvas.paste(panels[1], (cell_w, 0))
        canvas.paste(panels[2], (0, cell_h))
        canvas.paste(panels[3], (cell_w, cell_h))
        draw = ImageDraw.Draw(canvas)
        draw.line([(cell_w, 0), (cell_w, cell_h * 2)], fill=(220, 220, 220), width=2)
        draw.line([(0, cell_h), (cell_w * 2, cell_h)], fill=(220, 220, 220), width=2)
        return np.asarray(canvas, dtype=np.uint8)

    def _evaluate_sam3_conflict_detect(
        self,
        frame: np.ndarray,
        tracker_bbox: BBox,
        motion_bbox: BBox,
        effective_search_region: BBox | None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "ran": False,
            "override_motion": False,
            "instance_conflict": False,
            "conflict_type": "sam3_detector_instance",
            "reason": "not_run",
            "detected_count_in_search_region": 0,
        }
        factor = self.cfg.motion.perception_crop_factor
        if factor is not None:
            factor = float(factor)
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("perception_crop_factor must be positive and finite, or None")
            if factor != 4.0:


                anchor = getattr(self.tracker, "_last_search_anchor", None)
                if anchor is None:
                    raise RuntimeError("An explicit perception crop requires the current pre-track search anchor")
                side = int(np.ceil(np.sqrt(max(1.0, float(anchor.w)) * max(1.0, float(anchor.h))) * factor))
                cx, cy = float(anchor.x) + float(anchor.w) / 2, float(anchor.y) + float(anchor.h) / 2
                effective_search_region = BBox(
                    float(round(cx - side * .5)), float(round(cy - side * .5)),
                    float(side), float(side),
                ).clamp(frame.shape[1], frame.shape[0])
            info["perception_crop_factor"] = factor
        if effective_search_region is None:
            info["reason"] = "missing_effective_search_region"
            return info

        detections = self.motion.detect_similar_objects(
            frame,
            motion_bbox,
            search_region=effective_search_region,
        )
        info["ran"] = True
        info["search_region"] = _bbox_to_dict(effective_search_region)
        info["detected_count_in_search_region"] = len(detections)
        info["mask_coordinate_space"] = "search_region_crop"
        info["search_region_detections"] = [
            {
                "instance_id": int(idx),
                "bbox": _bbox_to_dict(det.bbox),
                "score": float(det.score),
                "mask": _mask_to_log_dict(det.mask),
                "source": det.source,
            }
            for idx, det in enumerate(detections)
        ]
        if not detections:
            info["reason"] = "no_similar_objects_detected"
            return info

        tracker_assignment = self._assign_bbox_to_sam3_instance(
            tracker_bbox,
            detections,
            effective_search_region,
            frame_shape=frame.shape,
        )
        motion_assignment = self._assign_bbox_to_sam3_instance(
            motion_bbox,
            detections,
            effective_search_region,
            frame_shape=frame.shape,
        )
        info["tracker_assignment"] = tracker_assignment
        info["sam3_assignment"] = motion_assignment
        tracker_instance = tracker_assignment.get("instance_id")
        motion_instance = motion_assignment.get("instance_id")
        if tracker_instance is None or motion_instance is None:
            info["reason"] = "unassigned_instance"
            return info
        if int(tracker_instance) == int(motion_instance):
            info["reason"] = "same_detector_instance"
            return info

        info["instance_conflict"] = True
        info["override_motion"] = True
        info["reason"] = "different_detector_instances"
        return info

    def _assign_bbox_to_sam3_instance(
        self,
        bbox: BBox,
        detections: list[SegmentationDetection],
        effective_search_region: BBox,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        center = bbox.center()
        assignment: dict[str, Any] = {
            "instance_id": None,
            "method": "unassigned",
            "center": {"x": float(center[0]), "y": float(center[1])},
        }
        mask_matches = self._mask_center_matches(
            center=center,
            detections=detections,
            effective_search_region=effective_search_region,
            frame_shape=frame_shape,
        )
        if mask_matches:
            instance_id, det = min(
                mask_matches,
                key=lambda item: (-float(item[1].score), self._center_distance(center, item[1].bbox.center())),
            )
            assignment.update(
                {
                    "instance_id": int(instance_id),
                    "method": "mask_center",
                    "instance_bbox": _bbox_to_dict(det.bbox),
                    "instance_score": float(det.score),
                }
            )
            return assignment

        box_matches = [
            (idx, det)
            for idx, det in enumerate(detections)
            if _point_in_bbox(center, det.bbox)
        ]
        if box_matches:
            instance_id, det = min(
                box_matches,
                key=lambda item: (-float(item[1].score), self._center_distance(center, item[1].bbox.center())),
            )
            assignment.update(
                {
                    "instance_id": int(instance_id),
                    "method": "box_center",
                    "instance_bbox": _bbox_to_dict(det.bbox),
                    "instance_score": float(det.score),
                }
            )
        return assignment

    def _mask_center_matches(
        self,
        center: tuple[float, float],
        detections: list[SegmentationDetection],
        effective_search_region: BBox,
        frame_shape: tuple[int, ...],
    ) -> list[tuple[int, SegmentationDetection]]:
        h, w = frame_shape[:2]
        rx1, ry1, rx2, ry2 = effective_search_region.clamp(w, h).as_xyxy()
        crop_x1 = max(0, min(w - 1, int(np.floor(rx1))))
        crop_y1 = max(0, min(h - 1, int(np.floor(ry1))))
        crop_x2 = max(crop_x1 + 1, min(w, int(np.ceil(rx2))))
        crop_y2 = max(crop_y1 + 1, min(h, int(np.ceil(ry2))))
        crop_w = max(1, crop_x2 - crop_x1)
        crop_h = max(1, crop_y2 - crop_y1)
        local_x = center[0] - float(crop_x1)
        local_y = center[1] - float(crop_y1)
        if local_x < 0.0 or local_y < 0.0 or local_x >= float(crop_w) or local_y >= float(crop_h):
            return []

        matches: list[tuple[int, SegmentationDetection]] = []
        for idx, det in enumerate(detections):
            if det.mask is None or det.mask.size == 0:
                continue
            mask = np.asarray(det.mask, dtype=bool)
            if mask.ndim != 2:
                continue
            mask_h, mask_w = mask.shape
            mx = int(np.floor(local_x * float(mask_w) / float(crop_w)))
            my = int(np.floor(local_y * float(mask_h) / float(crop_h)))
            mx = max(0, min(mask_w - 1, mx))
            my = max(0, min(mask_h - 1, my))
            if bool(mask[my, mx]):
                matches.append((idx, det))
        return matches

    @staticmethod
    def _center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    def _check_initialized(self) -> None:
        if not self.state.initialized:
            raise RuntimeError("Pipeline not initialized.")
