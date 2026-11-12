from __future__ import annotations

from abc import ABC
import base64
import io
import json
import logging
import os
from pathlib import Path
import re
import socket
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
from PIL import Image

from actrack.config import AgentConfig

_LOG = logging.getLogger(__name__)


class _ArkLengthLimitError(RuntimeError):
    pass


def _to_data_url(image: np.ndarray) -> str:
    if image.size == 0:
        return ""
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return ""
    arr = arr[:, :, :3]
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _normalize_sam_tracker_uncertain_answer(text: Any) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    lower = raw.lower()
    first_line = lower.splitlines()[0].strip().strip("\"'“”‘’")
    if first_line in {"tracker", "sam", "uncertain"}:
        return {"tracker": "Tracker", "sam": "SAM", "uncertain": "Uncertain"}[first_line]

    positions: list[tuple[int, str]] = []
    for pattern, label in (
        (r"\btracker\b", "Tracker"),
        (r"\bsam\b", "SAM"),
        (r"\buncertain\b", "Uncertain"),
        (r"不确定", "Uncertain"),
    ):
        match = re.search(pattern, lower)
        if match is not None:
            positions.append((int(match.start()), label))
    if not positions:
        return None
    positions.sort(key=lambda item: item[0])
    return positions[0][1]


class BaseVLLMClient(ABC):

    def choose_sam_tracker_uncertain(
        self,
        prompt: str,
        panel_image: np.ndarray,
        max_output_tokens: int = 16,
        image_max_side: int | None = None,
    ) -> tuple[str, str]:
        _ = prompt, panel_image, max_output_tokens, image_max_side
        return "Uncertain", ""

    @staticmethod
    def _summarize_api_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        summary: dict[str, Any] = {}
        for key in ("model", "max_output_tokens", "max_tokens", "temperature"):
            if key in payload:
                summary[key] = payload.get(key)
        if isinstance(payload.get("input"), list):
            summary["input_items"] = len(payload["input"])
        if isinstance(payload.get("messages"), list):
            summary["message_count"] = len(payload["messages"])
        return summary or None

    @staticmethod
    def _summarize_api_response(response: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(response, dict):
            return None
        summary: dict[str, Any] = {}
        for key in ("id", "status", "incomplete_details"):
            if key in response:
                summary[key] = response.get(key)
        if "error" in response:
            summary["error"] = response.get("error")
        return summary or None

    def set_api_failure_context(self, sample_dir: str | Path | None, context: dict[str, Any] | None = None) -> None:
        self._api_failure_sample_dir = (Path(sample_dir) if sample_dir else None)
        self._api_failure_context = dict(context or {})

    def clear_api_failure_context(self) -> None:
        self._api_failure_sample_dir = None
        self._api_failure_context = {}

    def consume_api_failures(self) -> list[dict[str, Any]]:
        events = list(getattr(self, "_api_failure_events", []))
        self._api_failure_events = []
        return events

    def _record_api_failure(
        self,
        *,
        provider: str,
        stage: str,
        error: Exception | str,
        payload: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": str(provider).strip(),
            "stage": str(stage).strip(),
            "error": str(error).strip(),
            "exception_type": type(error).__name__ if isinstance(error, Exception) else "Error",
            "context": dict(getattr(self, "_api_failure_context", {}) or {}),
            "payload": self._summarize_api_payload(payload),
            "response": self._summarize_api_response(response),
        }
        events = list(getattr(self, "_api_failure_events", []))
        events.append(event)
        self._api_failure_events = events

        sample_dir = getattr(self, "_api_failure_sample_dir", None)
        if isinstance(sample_dir, Path):
            sample_dir.mkdir(parents=True, exist_ok=True)
            with (sample_dir / "api_failures.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


class ArkVLLMClient(BaseVLLMClient):


    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.model = str(cfg.vllm_model).strip() or "doubao-seed-2-0-pro-260215"
        self.api_base = str(cfg.vllm_api_base).rstrip("/")
        self.api_path = str(getattr(cfg, "vllm_api_path", "/responses") or "/responses").strip()
        self.api_key_env = str(cfg.vllm_api_key_env).strip()
        self.api_key = str(getattr(cfg, "vllm_api_key", "") or "").strip()
        self.timeout_sec = float(cfg.vllm_timeout_sec)
        self.retries = max(0, int(getattr(cfg, "vllm_retries", 1)))
        self.retry_backoff_sec = max(0.0, float(getattr(cfg, "vllm_retry_backoff_sec", 1.0)))
        self.circuit_on_length_error = bool(getattr(cfg, "vllm_circuit_on_length_error", False))
        self.image_max_side = max(96, int(getattr(cfg, "vllm_image_max_side", 384)))
        self.image_min_side = max(14, int(getattr(cfg, "vllm_image_min_side", 14)))
        self.circuit_breaker_sec = max(0.0, float(getattr(cfg, "vllm_circuit_breaker_sec", 120.0)))
        self._circuit_open_until = 0.0
        self._warned_circuit = False

    def choose_sam_tracker_uncertain(
        self,
        prompt: str,
        panel_image: np.ndarray,
        max_output_tokens: int = 16,
        image_max_side: int | None = None,
    ) -> tuple[str, str]:
        if self._circuit_open():
            return "Uncertain", "circuit_open"
        if not isinstance(panel_image, np.ndarray) or panel_image.size == 0:
            return "Uncertain", "empty_panel"

        payload = self._build_single_image_choice_payload(
            prompt=prompt,
            image=panel_image,
            max_output_tokens=max(1, int(max_output_tokens)),
            temperature=0.0,
            image_max_side=image_max_side,
        )
        try:
            api_key = self.api_key or os.getenv(self.api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(f"Missing Ark API key. Set agent.vllm_api_key or env {self.api_key_env}.")
            response = self._call_ark_responses(payload=payload, api_key=api_key)
            self._raise_if_api_error(response)
            if self._is_incomplete_length(response):
                raise _ArkLengthLimitError(self._format_empty_output_reason(response))
            raw_text = self._extract_output_text(response)
            answer = _normalize_sam_tracker_uncertain_answer(raw_text)
            if answer is None:
                raise RuntimeError(f"Unexpected SAM/Tracker/Uncertain answer: {raw_text!r}")
            return answer, raw_text
        except Exception as exc:
            self._record_api_failure(
                provider="ark",
                stage="choose_sam_tracker_uncertain",
                error=exc,
                payload=payload,
            )
            self._open_circuit(exc)
            return "Uncertain", str(exc)

    def _circuit_open(self) -> bool:
        if self.circuit_breaker_sec <= 0:
            return False
        return time.time() < float(self._circuit_open_until)

    def _open_circuit(self, exc: Exception) -> None:
        if isinstance(exc, _ArkLengthLimitError) and (not self.circuit_on_length_error):
            return
        if self.circuit_breaker_sec <= 0:
            return
        now = time.time()
        self._circuit_open_until = now + float(self.circuit_breaker_sec)
        if not self._warned_circuit:
            _LOG.warning(
                "Ark client error; open circuit for %.1fs and use tracker-only fallback. reason=%s",
                float(self.circuit_breaker_sec),
                exc,
            )
            self._warned_circuit = True

    def _build_single_image_choice_payload(
        self,
        prompt: str,
        image: np.ndarray,
        max_output_tokens: int,
        temperature: float = 0.0,
        image_max_side: int | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        data_url = _to_data_url(self._prepare_image_for_api(image, max_side=image_max_side))
        if data_url:
            content.append({"type": "input_image", "image_url": data_url})
        return {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(max_output_tokens),
            "temperature": float(temperature),
            "thinking": {"type": "disabled"},
        }

    def _call_ark_responses(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        endpoint = self._resolve_endpoint()
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urlrequest.Request(
                url=endpoint,
                data=data,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise RuntimeError("Ark response is not a JSON object.")
                return parsed
            except urlerror.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")

                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.retries:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(f"Ark API HTTP {exc.code}: {body[:400]}") from exc
            except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
                raise RuntimeError(f"Ark API request failed: {exc}") from exc

        raise RuntimeError(f"Ark API request failed: {last_error}")

    def _resolve_endpoint(self) -> str:
        path = self.api_path
        if path.startswith("http://") or path.startswith("https://"):
            return path.rstrip("/")
        base = self.api_base
        if base.endswith("/responses"):
            return base
        if not path:
            path = "/responses"
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _extract_output_text(self, response: dict[str, Any]) -> str:
        top_text = response.get("output_text")
        if isinstance(top_text, str) and top_text.strip():
            return top_text.strip()
        if isinstance(top_text, list):
            chunks = [t.strip() for t in top_text if isinstance(t, str) and t.strip()]
            if chunks:
                return "\n".join(chunks).strip()

        chunks: list[str] = []
        output = response.get("output", [])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if isinstance(content, str) and content.strip():
                    chunks.append(content.strip())
                    continue
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks).strip()


        choices = response.get("choices", [])
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message", {}) if isinstance(first, dict) else {}
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                inner = [str(p.get("text", "")).strip() for p in content if isinstance(p, dict)]
                return "\n".join([t for t in inner if t]).strip()
        return ""

    def _raise_if_api_error(self, response: dict[str, Any]) -> None:
        err = response.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message", "") or err)
            code = str(err.get("code", ""))
            raise RuntimeError(f"Ark API error: code={code} message={msg}")
        if isinstance(err, str) and err.strip():
            raise RuntimeError(f"Ark API error: {err}")

    def _format_empty_output_reason(self, response: dict[str, Any]) -> str:
        status = str(response.get("status", "")).strip()
        details = response.get("incomplete_details")
        rid = response.get("id")
        if status:
            return f"Ark response has empty output text. status={status} incomplete_details={details} id={rid}"
        return f"Ark response has empty output text. keys={list(response.keys())[:12]}"

    def _is_incomplete_length(self, response: dict[str, Any]) -> bool:
        status = str(response.get("status", "")).strip().lower()
        details = response.get("incomplete_details")
        reason = ""
        if isinstance(details, dict):
            reason = str(details.get("reason", "")).strip().lower()
        elif isinstance(details, str):
            reason = details.strip().lower()
        return status == "incomplete" and "length" in reason

    def _prepare_image_for_api(self, image: np.ndarray, max_side: int | None = None) -> np.ndarray:
        arr = np.asarray(image)
        if arr.size == 0:
            return arr
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return arr
        arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        source_max_side = max(h, w)
        limit = self.image_max_side if max_side is None else max(96, int(max_side))
        if source_max_side > limit:
            scale = float(limit) / float(source_max_side)
            nh = max(1, int(round(h * scale)))
            nw = max(1, int(round(w * scale)))
            arr = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
            h, w = arr.shape[:2]

        min_side = min(h, w)
        if min_side < self.image_min_side:
            scale = float(self.image_min_side) / float(min_side)
            nh = max(self.image_min_side, int(round(h * scale)))
            nw = max(self.image_min_side, int(round(w * scale)))
            arr = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
        return arr
