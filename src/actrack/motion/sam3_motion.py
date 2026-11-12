from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from actrack.config import MotionConfig
from actrack.motion.base import BaseMotionModel
from actrack.types import BBox, MotionPrediction, SegmentationDetection


class SAM3MotionModel(BaseMotionModel):
    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._video_path: str | None = None
        self._predictor = None
        self._processor = None
        self._state = None
        self._torch = None

        self._initialized = False
        self._num_frames: int | None = None
        self._current_frame_idx = 0
        self._last_bbox: BBox | None = None
        self._propagate_gen = None
        self._bbox_cache: dict[int, BBox] = {}

    def prepare_sequence(self, frame_paths: list[str] | None = None, frame_dir: str | None = None) -> None:
        if frame_dir:
            self._video_path = str(Path(frame_dir).resolve())
            return
        if frame_paths:
            self._video_path = str(Path(frame_paths[0]).resolve().parent)
            return
        self._video_path = None

    def initialize(self, init_bbox: BBox) -> None:
        if self._video_path is None:
            raise RuntimeError(
                "SAM3MotionModel requires sequence frames directory. "
                "Call prepare_sequence(frame_paths=..., frame_dir=...) before initialize."
            )

        predictor, processor, torch_mod = self._build_predictor()
        self._predictor = predictor
        self._processor = processor
        self._torch = torch_mod

        self._state = self._predictor.init_state(
            video_path=self._video_path,
            offload_video_to_cpu=bool(self.cfg.sam3_offload_video_to_cpu),
            offload_state_to_cpu=bool(self.cfg.sam3_offload_state_to_cpu),
            async_loading_frames=bool(self.cfg.sam3_async_loading_frames),
        )
        self._num_frames = int(self._state["num_frames"])
        self._current_frame_idx = 0
        self._last_bbox = init_bbox
        self._bbox_cache.clear()
        self._prompt_box(frame_idx=0, bbox=init_bbox)
        self._restart_propagation(start_frame_idx=1)
        self._initialized = True

    def predict(self, image_width: int, image_height: int) -> MotionPrediction:
        self._check_initialized()

        target_idx = self._current_frame_idx + 1
        if self._num_frames is not None:
            target_idx = min(target_idx, self._num_frames - 1)
        if target_idx <= self._current_frame_idx:
            pred = (self._last_bbox or BBox(0, 0, 32, 32)).clamp(image_width, image_height)
            return MotionPrediction(
                predicted_bbox=pred,
                search_region=pred.expanded(float(self.cfg.search_expand_ratio)).clamp(image_width, image_height),
                fallback_used=False,
            )

        bbox = self._predict_from_propagation(target_idx, image_width, image_height)
        fallback_used = bbox is None
        if bbox is None:
            bbox = (self._last_bbox or BBox(0, 0, 32, 32)).clamp(image_width, image_height)

        self._current_frame_idx = target_idx
        self._last_bbox = bbox
        self._bbox_cache.pop(target_idx, None)
        return MotionPrediction(
            predicted_bbox=bbox,
            search_region=bbox.expanded(float(self.cfg.search_expand_ratio)).clamp(image_width, image_height),
            fallback_used=fallback_used,
        )

    def update(self, bbox: BBox) -> None:
        self._check_initialized()
        self._last_bbox = bbox


    def reprompt(self, bbox: BBox) -> None:
        self._check_initialized()
        self._last_bbox = bbox
        self._prompt_box(frame_idx=self._current_frame_idx, bbox=bbox)
        self._restart_propagation(start_frame_idx=self._current_frame_idx + 1)


    def _predict_from_propagation(self, target_idx: int, image_width: int, image_height: int) -> BBox | None:
        cached = self._bbox_cache.get(target_idx)
        if cached is not None:
            return cached

        if self._propagate_gen is None:
            self._restart_propagation(start_frame_idx=target_idx)

        while self._propagate_gen is not None:
            with self._inference_context():
                try:
                    out_frame_idx, out_obj_ids, _low_res_masks, out_video_res_masks, _obj_scores = next(
                        self._propagate_gen
                    )
                except StopIteration:
                    self._close_propagation()
                    break

            out_idx = int(out_frame_idx)
            bbox = self._mask_to_bbox(
                out_obj_ids=out_obj_ids,
                out_video_res_masks=out_video_res_masks,
                image_width=image_width,
                image_height=image_height,
            )
            if bbox is None:
                bbox = (self._last_bbox or BBox(0, 0, 32, 32)).clamp(image_width, image_height)
            self._bbox_cache[out_idx] = bbox
            if out_idx >= target_idx:
                return self._bbox_cache.get(target_idx)

        return self._bbox_cache.get(target_idx)

    def _restart_propagation(self, start_frame_idx: int) -> None:
        self._close_propagation()
        self._bbox_cache.clear()
        if self._num_frames is None:
            return
        start_idx = int(max(0, start_frame_idx))
        if start_idx >= self._num_frames:
            return
        assert self._predictor is not None
        assert self._state is not None
        self._propagate_gen = self._predictor.propagate_in_video(
            self._state,
            start_frame_idx=start_idx,
            max_frame_num_to_track=None,
            reverse=False,
            propagate_preflight=True,
        )

    def _close_propagation(self) -> None:
        gen = self._propagate_gen
        self._propagate_gen = None
        if gen is None:
            return
        try:
            gen.close()
        except Exception:
            return

    def _prompt_box(self, frame_idx: int, bbox: BBox) -> None:
        assert self._predictor is not None
        assert self._state is not None
        video_w = float(self._state["video_width"])
        video_h = float(self._state["video_height"])
        x1, y1, x2, y2 = bbox.as_xyxy()
        box = np.array(
            [
                x1 / max(video_w, 1.0),
                y1 / max(video_h, 1.0),
                x2 / max(video_w, 1.0),
                y2 / max(video_h, 1.0),
            ],
            dtype=np.float32,
        )
        with self._inference_context():
            self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=int(frame_idx),
                obj_id=int(self.cfg.sam3_obj_id),
                box=box,
                points=None,
                labels=None,
                clear_old_points=True,
                rel_coordinates=True,
            )

    def _mask_to_bbox(self, out_obj_ids, out_video_res_masks, image_width: int, image_height: int) -> BBox | None:
        obj_ids = list(out_obj_ids)
        wanted = int(self.cfg.sam3_obj_id)
        if wanted in obj_ids:
            obj_idx = obj_ids.index(wanted)
        elif len(obj_ids) > 0:
            obj_idx = 0
        else:
            return None

        logits = out_video_res_masks[obj_idx]
        if hasattr(logits, "detach"):
            logits_np = logits.detach().float().cpu().numpy()
        else:
            logits_np = np.asarray(logits)
        while logits_np.ndim > 2:
            logits_np = logits_np[0]

        thr = float(self.cfg.sam3_mask_logit_threshold)
        mask = logits_np > thr
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None

        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return BBox.from_xyxy(x1, y1, x2, y2).clamp(image_width, image_height)

    def _inference_context(self):
        if self._torch is None:
            return nullcontext()
        torch_mod = self._torch
        if str(self.cfg.sam3_device).startswith("cuda"):
            return torch_mod.autocast(device_type="cuda", dtype=torch_mod.bfloat16)
        return nullcontext()

    def detect_similar_objects(
        self,
        image: np.ndarray,
        exemplar_bbox: BBox,
        search_region: BBox | None = None,
    ) -> list[SegmentationDetection]:
        self._check_initialized()
        processor = self._processor
        if processor is None:
            return []

        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        region = (search_region or BBox(0, 0, float(w), float(h))).clamp(w, h)
        rx1, ry1, rx2, ry2 = region.as_xyxy()
        crop_x1 = max(0, min(w - 1, int(np.floor(rx1))))
        crop_y1 = max(0, min(h - 1, int(np.floor(ry1))))
        crop_x2 = max(crop_x1 + 1, min(w, int(np.ceil(rx2))))
        crop_y2 = max(crop_y1 + 1, min(h, int(np.ceil(ry2))))
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]
        prompt = exemplar_bbox.clamp(w, h)
        x1, y1, x2, y2 = prompt.as_xyxy()
        x1 = min(max(x1 - float(crop_x1), 0.0), float(crop_w - 1))
        y1 = min(max(y1 - float(crop_y1), 0.0), float(crop_h - 1))
        x2 = min(max(x2 - float(crop_x1), x1 + 1.0), float(crop_w))
        y2 = min(max(y2 - float(crop_y1), y1 + 1.0), float(crop_h))
        norm_box = [
            ((x1 + x2) * 0.5) / max(float(crop_w), 1.0),
            ((y1 + y2) * 0.5) / max(float(crop_h), 1.0),
            (x2 - x1) / max(float(crop_w), 1.0),
            (y2 - y1) / max(float(crop_h), 1.0),
        ]
        with self._inference_context():
            state = processor.set_image(Image.fromarray(crop))
            state = processor.add_geometric_prompt(state=state, box=norm_box, label=True)

        boxes = state.get("boxes")
        scores = state.get("scores")
        masks = state.get("masks")
        if boxes is None or scores is None:
            return []

        detections: list[SegmentationDetection] = []
        total = min(len(boxes), len(scores))
        for idx in range(total):
            box_i = boxes[idx]
            score_i = scores[idx]
            box_np = (
                box_i.detach().float().cpu().numpy()
                if hasattr(box_i, "detach")
                else np.asarray(box_i, dtype=np.float32)
            )
            if box_np.size < 4:
                continue
            bbox = BBox.from_xyxy(
                float(box_np[0]) + float(crop_x1),
                float(box_np[1]) + float(crop_y1),
                float(box_np[2]) + float(crop_x1),
                float(box_np[3]) + float(crop_y1),
            ).clamp(w, h)
            score = float(score_i.item()) if hasattr(score_i, "item") else float(score_i)
            mask_np: np.ndarray | None = None
            if masks is not None and idx < len(masks):
                raw_mask = masks[idx]
                mask_arr = (
                    raw_mask.detach().cpu().numpy()
                    if hasattr(raw_mask, "detach")
                    else np.asarray(raw_mask)
                )
                while mask_arr.ndim > 2:
                    mask_arr = mask_arr[0]
                mask_np = np.asarray(mask_arr, dtype=bool)
                if mask_np.size == 0 or (not bool(mask_np.any())):
                    mask_np = None
            detections.append(
                SegmentationDetection(
                    bbox=bbox,
                    score=score,
                    mask=mask_np,
                    source="sam3_detect",
                )
            )
        return detections

    def _build_predictor(self):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("SAM3MotionModel requires torch.") from exc

        device = str(self.cfg.sam3_device).strip()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("sam3_device is CUDA but CUDA is not available.")

        pkg_root = self._resolve_sam3_pkg_root()
        self._ensure_sam3_import_points_to(pkg_root)
        if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))
        try:
            from sam3.model_builder import build_sam3_video_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(
                "Failed to import SAM3 package. Run `pip install -e ./sam3` "
                "in your environment or set motion.sam3_root correctly."
            ) from exc

        ckpt = Path(self.cfg.sam3_checkpoint).expanduser()
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {ckpt}. Set motion.sam3_checkpoint in config."
            )

        bpe_path = pkg_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            raise FileNotFoundError(f"SAM3 BPE vocab not found: {bpe_path}")

        model = build_sam3_video_model(
            checkpoint_path=str(ckpt),
            load_from_HF=False,
            bpe_path=str(bpe_path),
            apply_temporal_disambiguation=bool(self.cfg.sam3_apply_temporal_disambiguation),
            device=device,
            compile=bool(self.cfg.sam3_compile),
            min_obj_score_logits=self.cfg.sam3_min_obj_score_logits,
        )
        predictor = model.tracker
        predictor.backbone = model.detector.backbone
        processor = Sam3Processor(
            model.detector,
            device=device,
        )
        return predictor, processor, torch

    def _resolve_sam3_pkg_root(self) -> Path:
        root = Path(self.cfg.sam3_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()
        candidates = [root, root / "sam3"]
        for cand in candidates:
            if (cand / "sam3" / "model_builder.py").exists():
                return cand
        raise FileNotFoundError(
            "SAM3 package root not found. Expected either "
            f"{root / 'sam3' / 'model_builder.py'} or {(root / 'sam3') / 'sam3' / 'model_builder.py'}."
        )

    def _ensure_sam3_import_points_to(self, pkg_root: Path) -> None:
        loaded = sys.modules.get("sam3")
        if loaded is None:
            return

        module_ok = False
        module_file = getattr(loaded, "__file__", None)
        if module_file is not None:
            module_path = Path(module_file).resolve()
            module_ok = pkg_root in module_path.parents
        if not module_ok:
            module_paths = getattr(loaded, "__path__", [])
            for module_dir in module_paths:
                mod_dir = Path(str(module_dir)).resolve()
                if pkg_root in mod_dir.parents or mod_dir == pkg_root:
                    module_ok = True
                    break
        if module_ok:
            return

        for name in list(sys.modules):
            if name == "sam3" or name.startswith("sam3."):
                del sys.modules[name]

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("SAM3MotionModel not initialized.")
