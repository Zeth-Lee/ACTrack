"""CPU regressions for prompt languages, HTTP parsing, and tracking decisions.

Run with ``python -m unittest discover -s tests -v``. Only model predictions,
detector output, and urllib's HTTP transport are replaced by test doubles.
The real client, request encoder, panel renderer, K5 gate, and step routing run.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from actrack.agent.prompts import VLM_REPROMPT_PROMPT_EN, VLM_REPROMPT_PROMPT_ZH
from actrack.agent.vllm_client import (
    ArkVLLMClient,
    _normalize_matching_motion_uncertain_answer,
    _normalize_sam_tracker_uncertain_answer,
)
from actrack.config import AgentConfig, PipelineConfig, load_config
import actrack.pipeline as pipeline_module
from actrack.pipeline import ACTrackPipeline
from actrack.types import BBox, CandidatePrediction, MotionPrediction, TrackerOutput


FRAME = (np.arange(100 * 120 * 3).reshape(100, 120, 3) % 251).astype(np.uint8)
TRACKER_BOX = BBox(0, 0, 8, 8)
BRANCHES = ("identity_diff_motion", "unassigned_tracker", "same_instance_extent")
PROMPTS = {"zh": VLM_REPROMPT_PROMPT_ZH, "en": VLM_REPROMPT_PROMPT_EN}
RAW_LABELS = {
    "zh": {"Tracker": "Tracker", "SAM": "SAM", "Uncertain": "Uncertain"},
    "en": {"Tracker": "Matching", "SAM": "Motion", "Uncertain": "Uncertain"},
}


def detection(branch):
    reason, ids, count = {
        "identity_diff_motion": ("different_detector_instances", (0, 1), 2),
        "unassigned_tracker": ("unassigned_instance", (None, 1), 2),
        "same_instance_extent": ("same_detector_instance", (0, 0), 1),
    }[branch]
    return dict(ran=True, reason=reason, detected_count_in_search_region=count,
                tracker_assignment=dict(instance_id=ids[0]),
                sam3_assignment=dict(instance_id=ids[1]),
                instance_conflict=branch == "identity_diff_motion",
                override_motion=branch == "identity_diff_motion")


class HTTPResponses:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(json.loads(request.data.decode("utf-8")))
        response = {"output": [{"content": [{"type": "output_text", "text": self.answer}]}]}
        return io.BytesIO(json.dumps(response).encode("utf-8"))


class FakeTracker:
    def initialize(self, image, bbox):
        pass

    def track(self, image, search_region, top_k):
        return TrackerOutput(CandidatePrediction(TRACKER_BOX, 1.0), [], BBox(0, 0, 70, 70))


class FakeMotion:
    def __init__(self):
        self.box = BBox(30, 0, 10, 10)
        self.updates = []
        self.reprompts = []

    def initialize(self, bbox):
        pass

    def predict(self, width, height):
        return MotionPrediction(self.box, self.box, False)

    def update(self, bbox):
        self.updates.append(bbox)

    def reprompt(self, bbox):
        self.reprompts.append(bbox)


def set_branch(pipe, branch):
    pipe.test_detection = detection(branch) if branch else None
    pipe.motion.box = BBox(0, 0, 20, 20) if branch == "same_instance_extent" else BBox(30, 0, 10, 10)


def fixture(language="zh", branch="identity_diff_motion"):
    cfg = PipelineConfig(agent=AgentConfig(
        vlm_reprompt_prompt_language=language,
        vllm_api_key="unit-test-only-placeholder",
        vllm_api_base="https://api.invalid",
        vllm_retries=0,
        vllm_circuit_breaker_sec=0,
    ))
    pipe = ACTrackPipeline(cfg, tracker=FakeTracker(), motion=FakeMotion(), sequence_name="test")
    pipe.initialize(FRAME, TRACKER_BOX)
    set_branch(pipe, branch)
    pipe._evaluate_sam3_conflict_detect = lambda **kwargs: pipe.test_detection
    return pipe


def frames(pipe, start, count):
    return [pipe.step(FRAME, index) for index in range(start, start + count)]


def gate(result):
    return result.extra["vlm_reprompt_gate"]


class PromptLanguageTests(unittest.TestCase):
    def setUp(self):
        # No test may reach the network, including accidental new HTTP paths.
        network = patch("actrack.agent.vllm_client.urlrequest.urlopen",
                        side_effect=AssertionError("Network is disabled in CPU tests"))
        network.start()
        self.addCleanup(network.stop)

    def test_configuration_defaults_explicit_languages_and_invalid_values(self):
        self.assertEqual(AgentConfig().vlm_reprompt_prompt_language, "zh")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(load_config(path).agent.vlm_reprompt_prompt_language, "zh")
            for language in ("zh", "en"):
                path.write_text(yaml.safe_dump({"agent": {"vlm_reprompt_prompt_language": language}}))
                self.assertEqual(load_config(path).agent.vlm_reprompt_prompt_language, language)
            for invalid in (None, "", "EN", "english", "fr", 1, True, ["en"], {"language": "en"}):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        AgentConfig(vlm_reprompt_prompt_language=invalid)
                    path.write_text(yaml.safe_dump({"agent": {"vlm_reprompt_prompt_language": invalid}}))
                    with self.assertRaises(ValueError):
                        load_config(path)

    def test_thirteen_public_configs_keep_two_search_factors_and_k5(self):
        paths = sorted((ROOT / "configs").glob("*/*.yaml"))
        self.assertEqual(len(paths), 13)
        for path in paths:
            with self.subTest(config=path.name):
                cfg = load_config(path, root=ROOT)
                mci = Path(cfg.tracker.mcitrack_root) / "experiments/mcitrack" / (cfg.tracker.mcitrack_yaml_name + ".yaml")
                self.assertEqual(yaml.safe_load(mci.read_text())["TEST"]["SEARCH_FACTOR"], 2.5)
                self.assertEqual(cfg.motion.perception_crop_factor, 2.5)
                self.assertEqual(cfg.agent.vlm_reprompt_prompt_language, "zh")
                self.assertEqual(cfg.agent.vlm_reprompt_gate_k, 5)
                self.assertTrue(cfg.agent.vlm_reprompt_gate_enabled)

    def test_frozen_chinese_prompt_bytes_and_english_vocabulary(self):
        self.assertEqual(hashlib.sha256(VLM_REPROMPT_PROMPT_ZH.encode("utf-8")).hexdigest(),
                         "e51ddc7879c126e5749e27d4dc676257659a57d1cb5156d5f3e4bf7a90c257d1")
        self.assertEqual(ACTrackPipeline._VLM_REPROMPT_PROMPT, VLM_REPROMPT_PROMPT_ZH)
        self.assertTrue(VLM_REPROMPT_PROMPT_EN.isascii())
        for label in ("Matching", "Motion", "Uncertain"):
            self.assertIn(label, VLM_REPROMPT_PROMPT_EN)

    def test_english_parser_accepts_only_single_label_with_case_or_quotes(self):
        normalize = _normalize_matching_motion_uncertain_answer
        for label, expected in (("Matching", "Tracker"), ("Motion", "SAM"), ("Uncertain", "Uncertain")):
            for raw in (label, label.lower(), label.upper(), " \t" + label + "\n",
                        '"' + label + '"', "'" + label + "'", "“" + label + "”", "‘" + label + "’"):
                with self.subTest(raw=raw):
                    self.assertEqual(normalize(raw), expected)
        for raw in (None, True, 0, [], {}, "", "Tracker", "SAM", "不确定", "Motion because it matches",
                    "Motion\nExplanation", "Matching or Motion", "Motion/Matching", "Motion.",
                    "{\"answer\": \"Motion\"}", "Answer: Motion", "```Motion```", "Emotion"):
            with self.subTest(invalid=raw):
                self.assertIsNone(normalize(raw))

    def test_chinese_parser_retains_legacy_semantics(self):
        for raw, expected in (("Tracker", "Tracker"), ("'sam'", "SAM"), ("不确定", "Uncertain"),
                              ("Tracker because the appearance matches", "Tracker"),
                              ("SAM\nExplanation", "SAM"), ("SAM or Tracker", "SAM"),
                              ("Matching", None), ("Motion", None)):
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_sam_tracker_uncertain_answer(raw), expected)

    def test_real_http_client_preserves_english_raw_text_and_internal_verdict(self):
        client = fixture("en").vllm
        self.assertIsInstance(client, ArkVLLMClient)
        for raw, internal in (('"mOtIoN"', "SAM"), ("MATCHING", "Tracker"), ("Uncertain", "Uncertain")):
            transport = HTTPResponses(raw)
            with self.subTest(raw=raw), patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                answer, original = client.choose_sam_tracker_uncertain(VLM_REPROMPT_PROMPT_EN, FRAME)
                self.assertEqual((answer, original), (internal, raw))
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(client.consume_api_failures(), [])

    def test_english_explanation_is_rejected_through_real_client_failure_path(self):
        transport = HTTPResponses("Motion because its appearance matches")
        client = fixture("en").vllm
        with patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
            answer, raw = client.choose_sam_tracker_uncertain(VLM_REPROMPT_PROMPT_EN, FRAME)
        self.assertEqual(answer, "Uncertain")
        self.assertIn("Motion because", raw)
        failures = client.consume_api_failures()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["stage"], "choose_sam_tracker_uncertain")

    def test_two_languages_three_events_three_verdicts_through_real_step(self):
        # 18 end-to-end routing cases, including both Ct=0 events for Motion.
        for language in ("zh", "en"):
            for branch in BRANCHES:
                for internal in ("Tracker", "SAM", "Uncertain"):
                    with self.subTest(language=language, branch=branch, answer=internal):
                        raw = RAW_LABELS[language][internal]
                        transport = HTTPResponses(raw)
                        pipe = fixture(language, branch)
                        with patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                            out = frames(pipe, 1, 7)
                        self.assertEqual([gate(r)["triggered"] for r in out], [False] * 4 + [True, False, False])
                        self.assertEqual(len(transport.requests), 1)
                        self.assertTrue(all(gate(r)["prompt_language"] == language for r in out))
                        chosen = out[4]
                        self.assertEqual((gate(chosen)["answer"], gate(chosen)["raw_answer"]), (internal, raw))
                        self.assertNotIn("api_failures", chosen.extra)
                        default_source = "motion" if branch == "identity_diff_motion" else "tracker"
                        self.assertEqual([r.extra["accepted_source"] for r in out[:4]], [default_source] * 4)
                        source = "tracker" if internal == "Tracker" else "motion" if internal == "SAM" else default_source
                        self.assertEqual(chosen.extra["accepted_source"], source)
                        self.assertEqual(chosen.bbox, TRACKER_BOX if source == "tracker" else pipe.motion.box)
                        self.assertEqual(len(pipe.motion.reprompts), int(internal == "Tracker"))
                        if internal == "SAM":
                            self.assertEqual(chosen.state, "vlm_motion")
                            self.assertEqual(pipe.motion.updates[4], pipe.motion.box)
                        if internal == "Tracker":
                            self.assertEqual(chosen.state, "vlm_reprompt_tracker")
                            self.assertEqual(pipe.motion.reprompts, [TRACKER_BOX])
                        self.assertTrue(all(gate(r)["answer"] is None for r in out[5:]))
                        payload = transport.requests[0]
                        self.assertEqual(payload["input"][0]["content"][0], {"type": "input_text", "text": PROMPTS[language]})
                        self.assertTrue(payload["input"][0]["content"][1]["image_url"].startswith("data:image/"))
                        self.assertEqual(payload["temperature"], 0.0)

    def test_k5_streak_resets_on_gap_identity_change_and_absent_event(self):
        for language in ("zh", "en"):
            pipe = fixture(language)
            transport = HTTPResponses("Uncertain")
            with self.subTest(language=language), patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                frames(pipe, 1, 4)
                self.assertEqual(gate(frames(pipe, 6, 1)[0])["streak"], 1)
                frames(pipe, 7, 3)
                pipe.test_detection["tracker_assignment"]["instance_id"] = 2
                self.assertEqual(gate(frames(pipe, 10, 1)[0])["streak"], 1)
                frames(pipe, 11, 3)
                set_branch(pipe, None)
                self.assertFalse(gate(frames(pipe, 14, 1)[0])["triggered"])
                self.assertEqual(pipe.state.vlm_reprompt_streak, 0)
                self.assertEqual(transport.requests, [])
                set_branch(pipe, "identity_diff_motion")
                self.assertEqual([gate(r)["triggered"] for r in frames(pipe, 15, 5)], [False] * 4 + [True])
                self.assertEqual(len(transport.requests), 1)

    def test_direct_reuse_preserves_all_three_next_episode_types(self):
        for language in ("zh", "en"):
            for next_branch in BRANCHES:
                with self.subTest(language=language, next_branch=next_branch):
                    pipe = fixture(language)
                    transport = HTTPResponses(RAW_LABELS[language]["Tracker"])
                    with patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                        frames(pipe, 1, 5)
                        self.assertTrue(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                        set_branch(pipe, None)
                        frames(pipe, 6, 1)
                        set_branch(pipe, next_branch)
                        direct = frames(pipe, 7, 5)[-1]
                        self.assertTrue(gate(direct)["triggered"])
                        self.assertTrue(gate(direct)["vlm_call_skipped_by_loop"])
                        self.assertEqual(gate(direct)["answer"], "Tracker")
                        self.assertEqual(gate(direct)["prompt_language"], language)
                        self.assertEqual(len(transport.requests), 1)
                        self.assertEqual(pipe.motion.reprompts, [TRACKER_BOX, TRACKER_BOX])
                        self.assertFalse(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                        set_branch(pipe, None)
                        frames(pipe, 12, 1)
                        set_branch(pipe, next_branch)
                        transport.answer = RAW_LABELS[language]["SAM"]
                        resumed = frames(pipe, 13, 5)[-1]
                        self.assertFalse(gate(resumed)["vlm_call_skipped_by_loop"])
                        self.assertEqual((gate(resumed)["answer"], gate(resumed)["raw_answer"]),
                                         ("SAM", RAW_LABELS[language]["SAM"]))
                        self.assertEqual(len(transport.requests), 2)
                        self.assertEqual(len(pipe.motion.reprompts), 2)
                        self.assertEqual(resumed.bbox, pipe.motion.box)

    def test_default_chinese_panel_and_language_specific_titles(self):
        kwargs = dict(frame=FRAME, tracker_bbox=TRACKER_BOX, motion_bbox=BBox(30, 0, 10, 10),
                      initial_reference_frame=FRAME[::-1].copy())
        zh, en = fixture("zh"), fixture("en")
        default = fixture()
        with patch.object(pipeline_module, "_header_panel", wraps=pipeline_module._header_panel) as headers:
            chinese = zh._build_vlm_reprompt_panel(**kwargs)
        self.assertEqual([call.args[1] for call in headers.call_args_list],
                         ["First frame target", "Current frame: green=Tracker, blue=SAM", "Tracker crop", "SAM crop"])
        np.testing.assert_array_equal(chinese, default._build_vlm_reprompt_panel(**kwargs))
        with patch.object(pipeline_module, "_header_panel", wraps=pipeline_module._header_panel) as headers:
            english = en._build_vlm_reprompt_panel(**kwargs)
        self.assertEqual([call.args[1] for call in headers.call_args_list],
                         ["First frame target", "Current frame: green=Matching, blue=Motion", "Matching crop", "Motion crop"])
        self.assertEqual(chinese.shape, english.shape)
        # Font rendering may vary by Pillow version. All image/crop/box body
        # pixels must be byte-identical between languages, independently of fonts.
        rows = np.ones(chinese.shape[0], dtype=bool)
        rows[:30] = False
        rows[chinese.shape[0] // 2:chinese.shape[0] // 2 + 30] = False
        np.testing.assert_array_equal(chinese[rows], english[rows])
        self.assertFalse(np.array_equal(chinese, english))


if __name__ == "__main__":
    unittest.main(verbosity=2)
