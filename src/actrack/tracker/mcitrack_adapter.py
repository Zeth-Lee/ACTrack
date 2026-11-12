from __future__ import annotations

import math
from pathlib import Path
from importlib import import_module
from types import SimpleNamespace
import sys

import numpy as np

from actrack.config import TrackerConfig
from actrack.tracker.base import BaseTracker
from actrack.types import BBox, CandidatePrediction, TrackerOutput


def _to_xywh_list(bbox: BBox) -> list[float]:
    return [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)]


def _to_bbox(xywh: list[float] | tuple[float, float, float, float]) -> BBox:
    x, y, w, h = xywh
    return BBox(float(x), float(y), float(w), float(h))


def _ensure_u8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


class MCITrackTracker(BaseTracker):


    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self._tracker = None
        self._torch = None
        self._mci_sample_target = None
        self._mci_clip_box = None
        self._tracker_supports_topk: bool | None = None
        self._initialized = False
        self._last_effective_search_region: BBox | None = None
        self._last_search_anchor: BBox | None = None
        self._mci_root = Path(cfg.mcitrack_root).resolve()
        self._load_official_tracker()

    def _load_official_tracker(self) -> None:
        if not self._mci_root.exists():
            raise FileNotFoundError(
                f"MCITrack root not found: {self._mci_root}. "
                "Set tracker.mcitrack_root in config."
            )

        if str(self._mci_root) not in sys.path:
            sys.path.insert(0, str(self._mci_root))

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Failed to import torch for MCITrack.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "MCITrack official tracker requires CUDA in current implementation. "
                "Please run on GPU node."
            )

        try:
            from lib.config.mcitrack.config import cfg as mci_cfg
            from lib.config.mcitrack.config import update_config_from_file
            from lib.test.tracker.mcitrack import MCITRACK
            from lib.test.tracker.utils import sample_target as mci_sample_target
            from lib.utils.box_ops import clip_box as mci_clip_box
        except Exception as exc:
            raise RuntimeError(
                "Failed to import MCITrack modules. Ensure MCITrack dependencies are installed "
                "(see MCITrack/install.sh)."
            ) from exc

        yaml_name = self.cfg.mcitrack_yaml_name
        yaml_path = self._mci_root / "experiments" / "mcitrack" / f"{yaml_name}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"MCITrack yaml not found: {yaml_path}")
        update_config_from_file(str(yaml_path))
        self._configure_pretrained_path(mci_cfg, import_module("lib.models.mcitrack.fastitpn"))

        checkpoint = self._resolve_checkpoint(mci_cfg, yaml_name)
        params = SimpleNamespace(
            cfg=mci_cfg,
            yaml_name=yaml_name,
            checkpoint=checkpoint,
            template_factor=float(mci_cfg.TEST.TEMPLATE_FACTOR),
            template_size=int(mci_cfg.TEST.TEMPLATE_SIZE),
            search_factor=float(mci_cfg.TEST.SEARCH_FACTOR),
            search_size=int(mci_cfg.TEST.SEARCH_SIZE),
            save_all_boxes=False,
            debug=int(self.cfg.mcitrack_debug),
        )
        self._tracker = MCITRACK(params, dataset_name=str(self.cfg.mcitrack_dataset_name))
        self._torch = torch
        self._mci_sample_target = mci_sample_target
        self._mci_clip_box = mci_clip_box

    def _configure_pretrained_path(self, mci_cfg, fastitpn_module) -> None:

        filename = Path(str(mci_cfg.MODEL.ENCODER.PRETRAIN_TYPE)).name
        checkpoint = Path(self.cfg.mcitrack_pretrained_root).expanduser().resolve() / filename
        if not filename or not checkpoint.is_file():
            raise FileNotFoundError(
                f"MCITrack encoder checkpoint not found: {checkpoint}. "
                "Set tracker.mcitrack_pretrained_root to the pretrained weights directory."
            )
        fastitpn_module.current_file_path = ""
        mci_cfg.MODEL.ENCODER.PRETRAIN_TYPE = str(checkpoint)

    def _resolve_checkpoint(self, mci_cfg, yaml_name: str) -> str:
        if self.cfg.mcitrack_checkpoint:
            ckpt = Path(self.cfg.mcitrack_checkpoint).expanduser().resolve()
        else:
            ckpt = (
                self._mci_root
                / "checkpoints"
                / "train"
                / "mcitrack"
                / yaml_name
                / f"MCITRACK_ep{int(mci_cfg.TEST.EPOCH):04d}.pth.tar"
            )
        if not ckpt.exists():
            raise FileNotFoundError(
                f"MCITrack checkpoint not found: {ckpt}. "
                "Set tracker.mcitrack_checkpoint to your downloaded weight path."
            )
        return str(ckpt)

    def initialize(self, image: np.ndarray, init_bbox: BBox) -> None:
        img = _ensure_u8_rgb(image)
        info = {"init_bbox": _to_xywh_list(init_bbox), "seq_name": "actrack_seq"}
        self._tracker.initialize(img, info)
        self._initialized = True

    def track(self, image: np.ndarray, search_region: BBox | None, top_k: int) -> TrackerOutput:
        self._check_initialized()
        img = _ensure_u8_rgb(image)
        tk = max(1, int(top_k))
        if search_region is not None:
            self._apply_motion_prior(search_region=search_region, image_shape=img.shape)
        effective_search_region = self._compute_effective_search_region(img.shape)
        self._last_effective_search_region = effective_search_region

        fallback_candidates: list[CandidatePrediction] = []
        if self._tracker_supports_topk is not True:
            fallback_candidates = self._decode_topk_candidates_from_internal(img, top_k=tk)

        if self._tracker_supports_topk is False:
            out = self._tracker.track(img, info={})
        else:
            try:
                out = self._tracker.track(img, info={}, top_k=tk)
                self._tracker_supports_topk = True
            except TypeError:
                out = self._tracker.track(img, info={})
                self._tracker_supports_topk = False

        bbox = _to_bbox(out["target_bbox"])
        score = float(out["best_score"].item()) if hasattr(out["best_score"], "item") else float(out["best_score"])
        best = CandidatePrediction(bbox=bbox, score=score, source="mcitrack")
        candidates = self._parse_topk_candidates(out, image_shape=img.shape, top_k=tk)
        if not candidates:
            candidates = fallback_candidates
        if not candidates:
            candidates = [best]
        else:
            candidates = candidates[:tk]

        return TrackerOutput(
            best=best,
            candidates=candidates,
            effective_search_region=effective_search_region,
        )

    def _parse_topk_candidates(
            self,
            out: dict,
            image_shape: tuple[int, int, int],
            top_k: int,
    ) -> list[CandidatePrediction]:
        raw_boxes = out.get("candidate_bboxes")
        if not isinstance(raw_boxes, (list, tuple)):
            return []

        raw_scores = out.get("candidate_scores")
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        h, w = image_shape[:2]
        parsed: list[CandidatePrediction] = []
        for idx, raw_box in enumerate(raw_boxes):
            if idx >= top_k:
                break
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
                continue

            try:
                bbox = _to_bbox((raw_box[0], raw_box[1], raw_box[2], raw_box[3])).clamp(w, h)
            except Exception:
                continue

            score = 0.0
            if isinstance(raw_scores, (list, tuple)) and idx < len(raw_scores):
                try:
                    score = float(raw_scores[idx])
                except Exception:
                    score = 0.0
            parsed.append(CandidatePrediction(bbox=bbox, score=score, source="mcitrack_topk"))

        parsed.sort(key=lambda c: c.score, reverse=True)
        return parsed

    def _decode_topk_box_tensors(self, response, out_dict: dict, top_k: int):
        torch = self._torch
        if torch is None:
            return None, None
        flat_response = response.flatten(1)
        if flat_response.numel() == 0:
            return None, None

        k = max(1, min(int(top_k), flat_response.shape[1]))
        topk_scores, topk_idx = torch.topk(flat_response, k=k, dim=1, largest=True, sorted=True)
        feat_sz = response.shape[-1]
        idx_y = torch.div(topk_idx, feat_sz, rounding_mode="floor")
        idx_x = topk_idx % feat_sz

        if "size_map" in out_dict:
            gather_idx = topk_idx.unsqueeze(1).expand(-1, 2, -1)
            size = out_dict["size_map"].flatten(2).gather(dim=2, index=gather_idx)
            offset = out_dict["offset_map"].flatten(2).gather(dim=2, index=gather_idx)
            cx = (idx_x.to(torch.float32) + offset[:, 0, :]) / feat_sz
            cy = (idx_y.to(torch.float32) + offset[:, 1, :]) / feat_sz
            w = size[:, 0, :]
            h = size[:, 1, :]
            boxes = torch.stack([cx, cy, w, h], dim=-1)
        else:
            gather_idx = topk_idx.unsqueeze(1).expand(-1, 4, -1)
            offset = out_dict["offset_map"].flatten(2).gather(dim=2, index=gather_idx)
            x1 = idx_x.to(torch.float32) / feat_sz - offset[:, 0, :]
            y1 = idx_y.to(torch.float32) / feat_sz - offset[:, 1, :]
            x2 = idx_x.to(torch.float32) / feat_sz + offset[:, 2, :]
            y2 = idx_y.to(torch.float32) / feat_sz + offset[:, 3, :]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            w = x2 - x1
            h = y2 - y1
            boxes = torch.stack([cx, cy, w, h], dim=-1)
        return boxes, topk_scores

    def _decode_topk_candidates_from_internal(self, image: np.ndarray, top_k: int) -> list[CandidatePrediction]:
        if self._tracker is None or self._torch is None:
            return []
        if self._mci_sample_target is None or self._mci_clip_box is None:
            return []

        state = getattr(self._tracker, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) < 4:
            return []

        h, w = image.shape[:2]
        x_patch_arr, resize_factor = self._mci_sample_target(
            image,
            state,
            self._tracker.params.search_factor,
            output_sz=self._tracker.params.search_size,
        )
        search = self._tracker.preprocessor.process(x_patch_arr)
        search_list = [search]
        with self._torch.no_grad():
            enc_opt = self._tracker.network.forward_encoder(
                self._tracker.template_list,
                search_list,
                self._tracker.template_anno_list,
            )
            hidden_state = self._tracker.h_state.copy()
            _, out_neck, _ = self._tracker.network.forward_neck(enc_opt, hidden_state)
            out_dict = self._tracker.network.forward_decoder(feature=out_neck)

        pred_score_map = out_dict["score_map"]
        if bool(self._tracker.cfg.TEST.WINDOW):
            response = self._tracker.output_window * pred_score_map
        else:
            response = pred_score_map

        boxes, scores = self._decode_topk_box_tensors(response=response, out_dict=out_dict, top_k=top_k)
        if boxes is None or scores is None:
            return []

        boxes = boxes[0]
        scores = scores[0]
        scale = float(self._tracker.params.search_size) / float(resize_factor)
        parsed: list[CandidatePrediction] = []
        for box, score in zip(boxes, scores):
            pred_box = (box * scale).tolist()
            mapped = self._mci_clip_box(self._tracker.map_box_back(pred_box, resize_factor), h, w, margin=10)
            parsed.append(
                CandidatePrediction(
                    bbox=_to_bbox(mapped).clamp(w, h),
                    score=float(score.item()),
                    source="mcitrack_topk",
                )
            )
        parsed.sort(key=lambda c: c.score, reverse=True)
        return parsed[: max(1, int(top_k))]

    def _apply_motion_prior(self, search_region: BBox, image_shape: tuple[int, int, int]) -> None:
        if not bool(self.cfg.mcitrack_use_motion_prior):
            return
        if self._tracker is None:
            return
        state = getattr(self._tracker, "state", None)
        if state is None:
            return
        if not isinstance(state, (list, tuple)) or len(state) < 4:
            return

        h, w = image_shape[:2]
        prev_bbox = _to_bbox(state)
        prior = search_region.clamp(w, h)
        blend = float(self.cfg.mcitrack_motion_center_blend)
        blend = min(1.0, max(0.0, blend))

        prev_cx, prev_cy = prev_bbox.center()
        prior_cx, prior_cy = prior.center()
        new_cx = (1.0 - blend) * prev_cx + blend * prior_cx
        new_cy = (1.0 - blend) * prev_cy + blend * prior_cy

        anchored = BBox(


            x=new_cx - prior.w / 2.0,
            y=new_cy - prior.h / 2.0,
            w=prior.w,
            h=prior.h,
        ).clamp(w, h)
        self._tracker.state = _to_xywh_list(anchored)

    def _compute_effective_search_region(self, image_shape: tuple[int, int, int]) -> BBox | None:
        self._last_search_anchor = None
        if self._tracker is None:
            return None
        state = getattr(self._tracker, "state", None)
        if not isinstance(state, (list, tuple)) or len(state) < 4:
            return None

        h, w = image_shape[:2]
        anchor = _to_bbox(state)
        self._last_search_anchor = anchor
        crop_side = math.ceil(
            math.sqrt(max(1.0, float(anchor.w)) * max(1.0, float(anchor.h)))
            * float(getattr(self._tracker.params, "search_factor", 4.0))
        )
        cx, cy = anchor.center()
        x1 = round(cx - crop_side * 0.5)
        y1 = round(cy - crop_side * 0.5)
        return BBox(float(x1), float(y1), float(crop_side), float(crop_side)).clamp(w, h)

    def get_last_effective_search_region(self) -> BBox | None:
        return self._last_effective_search_region


    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("MCITrackTracker not initialized.")
