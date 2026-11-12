from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import os
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "actrack"
    output_dir: str = "outputs/actrack"
    save_logs: bool = True
    random_seed: int | None = None


@dataclass
class TrackerConfig:
    confidence_threshold: float = 0.0
    top_k_candidates: int = 1
    mcitrack_root: str = "MCITrack"
    mcitrack_pretrained_root: str = "pretrained"
    mcitrack_yaml_name: str = "mcitrack_b224"
    mcitrack_checkpoint: str = "MCITrack/checkpoints/train/mcitrack/mcitrack_b224/MCITRACK_ep0300.pth.tar"
    mcitrack_dataset_name: str = "lasot"
    mcitrack_debug: int = 0
    mcitrack_use_motion_prior: bool = True
    mcitrack_motion_center_blend: float = 1.0


@dataclass
class MotionConfig:
    sam3_min_obj_score_logits: float = -1.0
    search_expand_ratio: float = 2.5
    perception_crop_factor: float | None = 2.5
    sam3_conflict_detect_enabled: bool = True
    sam3_root: str = "sam3"
    sam3_checkpoint: str = "sam3/checkpoints/sam3.pt"
    sam3_device: str = "cuda"
    sam3_offload_video_to_cpu: bool = True
    sam3_offload_state_to_cpu: bool = True
    sam3_async_loading_frames: bool = True
    sam3_mask_logit_threshold: float = 0.0
    sam3_obj_id: int = 1
    sam3_apply_temporal_disambiguation: bool = True
    sam3_compile: bool = False


@dataclass
class AgentConfig:
    vllm_model: str = "doubao-seed-2-0-pro-260215"
    vllm_api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    vllm_api_path: str = "/responses"
    vllm_api_key: str = ""
    vllm_api_key_env: str = "ARK_API_KEY"
    vllm_timeout_sec: float = 45.0
    vllm_retries: int = 2
    vllm_retry_backoff_sec: float = 1.2
    vllm_circuit_on_length_error: bool = False
    vllm_image_max_side: int = 320
    vllm_image_min_side: int = 14
    vllm_circuit_breaker_sec: float = 120.0
    vlm_reprompt_gate_enabled: bool = True
    vlm_reprompt_gate_k: int = 5
    vlm_reprompt_gate_image_max_side: int = 1100
    vlm_reprompt_gate_max_output_tokens: int = 16


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _merge_dataclass(dc_cls: type[Any], data: dict[str, Any] | None, section: str) -> Any:
    if data is None:
        return dc_cls()
    if not isinstance(data, dict):
        raise ValueError(f"Configuration section {section!r} must be a mapping.")
    if dc_cls is AgentConfig and "enabled" in data:
        raise ValueError("The legacy agent.enabled option is retired; use agent.vlm_reprompt_gate_enabled.")
    unknown = set(data) - {item.name for item in fields(dc_cls)}
    if unknown:
        names = ", ".join(f"{section}.{key}" for key in sorted(unknown, key=str))
        raise ValueError(f"Unknown configuration field(s): {names}")
    return dc_cls(**data)


def load_config(path: str | Path, *, root: str | Path | None = None) -> PipelineConfig:

    project_root = Path(root).expanduser().resolve() if root is not None else Path(__file__).resolve().parents[2]
    cfg_path = Path(os.path.expandvars(str(path))).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    unknown = set(raw) - {item.name for item in fields(PipelineConfig)}
    if unknown:
        names = ", ".join(str(key) for key in sorted(unknown, key=str))
        raise ValueError(f"Unknown configuration section(s): {names}")

    config = PipelineConfig(
        experiment=_merge_dataclass(ExperimentConfig, raw.get("experiment"), "experiment"),
        tracker=_merge_dataclass(TrackerConfig, raw.get("tracker"), "tracker"),
        motion=_merge_dataclass(MotionConfig, raw.get("motion"), "motion"),
        agent=_merge_dataclass(AgentConfig, raw.get("agent"), "agent"),
    )

    for section, names in (
        (config.experiment, ("output_dir",)),
        (config.tracker, ("mcitrack_root", "mcitrack_checkpoint", "mcitrack_pretrained_root")),
        (config.motion, ("sam3_root", "sam3_checkpoint")),
    ):
        for name in names:
            if name == "mcitrack_checkpoint" and not str(getattr(section, name)).strip():
                continue
            value = Path(os.path.expandvars(str(getattr(section, name)))).expanduser()
            setattr(section, name, str((value if value.is_absolute() else project_root / value).resolve()))
    return config
