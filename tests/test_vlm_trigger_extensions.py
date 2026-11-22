"""CPU regressions for the single VLM trigger-extension switch."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from test_vlm_prompt_languages import (
    BRANCHES,
    HTTPResponses,
    RAW_LABELS,
    ROOT,
    TRACKER_BOX,
    fixture,
    frames,
    gate,
    set_branch,
)
from actrack.config import AgentConfig, load_config


EXTEND_BRANCH = "extend_branch"
RETIRED_FLAGS = ("vlm_reprompt_unassigned_enabled", "vlm_reprompt_same_instance_extent_enabled")


def set_extend_branch(pipe, value):
    pipe.cfg.agent.extend_branch = value


def enabled(branch, extend_branch):
    return branch == "identity_diff_motion" or extend_branch


class TriggerExtensionTests(unittest.TestCase):
    def setUp(self):
        network = patch("actrack.agent.vllm_client.urlrequest.urlopen",
                        side_effect=AssertionError("Network is disabled in CPU tests"))
        network.start()
        self.addCleanup(network.stop)

    def test_configuration_requires_real_booleans_and_defaults_to_enabled(self):
        self.assertIs(AgentConfig().extend_branch, True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            for data in ({}, {EXTEND_BRANCH: False}, {EXTEND_BRANCH: True}):
                with self.subTest(valid=data):
                    path.write_text(yaml.safe_dump({"agent": data}), encoding="utf-8")
                    cfg = load_config(path).agent
                    direct = AgentConfig(**data)
                    self.assertIs(cfg.extend_branch, data.get(EXTEND_BRANCH, True))
                    self.assertIs(direct.extend_branch, data.get(EXTEND_BRANCH, True))
            for value in ("false", "true", None, 0, 1, 0.0, 1.0, [], {}):
                with self.subTest(invalid=value):
                    with self.assertRaises(ValueError):
                        AgentConfig(extend_branch=value)
                    path.write_text(yaml.safe_dump({"agent": {EXTEND_BRANCH: value}}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)
            for retired in RETIRED_FLAGS:
                with self.subTest(retired=retired):
                    self.assertFalse(hasattr(AgentConfig(), retired))
                    path.write_text(yaml.safe_dump({"agent": {retired: False}}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "Unknown configuration field"):
                        load_config(path)

    def test_public_yaml_files_explicitly_enable_extend_branch(self):
        paths = sorted((ROOT / "configs").glob("*/*.yaml"))
        self.assertEqual(len(paths), 13)
        for path in paths:
            with self.subTest(config=str(path.relative_to(ROOT))):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIs(data["agent"][EXTEND_BRANCH], True)
                self.assertTrue(all(name not in data["agent"] for name in RETIRED_FLAGS))
                loaded = load_config(path, root=ROOT).agent
                self.assertIs(loaded.extend_branch, True)

    def test_two_languages_two_switch_values_three_real_step_events(self):
        # 12 routes: false retains Branch1; true retains all three branches.
        # Disabled extension events leave ordinary coordination intact.
        for language, extend_branch, branch in product(("zh", "en"), (False, True), BRANCHES):
            with self.subTest(language=language, extend_branch=extend_branch, branch=branch):
                pipe = fixture(language, branch)
                set_extend_branch(pipe, extend_branch)
                transport = HTTPResponses(RAW_LABELS[language]["Tracker"])
                with patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                    result = frames(pipe, 1, 7)
                active = enabled(branch, extend_branch)
                self.assertEqual([gate(r)["triggered"] for r in result],
                                 [False] * 4 + [active, False, False])
                self.assertEqual(len(transport.requests), int(active))
                self.assertEqual(pipe.motion.reprompts, [TRACKER_BOX] if active else [])
                self.assertEqual(pipe.stats.vlm_reprompt_gate_calls, int(active))
                self.assertTrue(all(gate(r)["extend_branch"] is extend_branch for r in result))
                self.assertTrue(all("extensions" not in gate(r) for r in result))
                self.assertTrue(all(gate(r)["prompt_language"] == language for r in result))
                self.assertTrue(all("api_failures" not in r.extra for r in result))
                default_source = "motion" if branch == "identity_diff_motion" else "tracker"
                for index, output in enumerate(result):
                    source = "tracker" if active and index == 4 else default_source
                    self.assertEqual(output.extra["accepted_source"], source)
                    self.assertEqual(output.bbox, TRACKER_BOX if source == "tracker" else pipe.motion.box)
                    if active and index == 4:
                        self.assertEqual(gate(output)["branch"], branch)
                        self.assertEqual(gate(output)["answer"], "Tracker")
                        self.assertEqual(gate(output)["raw_answer"], RAW_LABELS[language]["Tracker"])
                        self.assertEqual(gate(output)["streak"], 5)
                    else:
                        self.assertIsNone(gate(output)["answer"])

    def test_pending_direct_reuse_is_not_consumed_by_disabled_extension(self):
        # Four paths cover each disabled extension in both languages.
        # An eligible Branch1 episode
        # must still wait five frames before consuming the pending one-use flag.
        for language, branch in product(("zh", "en"), BRANCHES[1:]):
            with self.subTest(language=language, branch=branch):
                pipe = fixture(language, "identity_diff_motion")
                set_extend_branch(pipe, False)
                transport = HTTPResponses(RAW_LABELS[language]["Tracker"])
                with patch("actrack.agent.vllm_client.urlrequest.urlopen", side_effect=transport):
                    frames(pipe, 1, 5)
                    self.assertTrue(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                    set_branch(pipe, branch)
                    for index in range(6, 12):
                        output = frames(pipe, index, 1)[0]
                        self.assertFalse(gate(output)["triggered"])
                        self.assertFalse(gate(output)["reprompt_applied"])
                        self.assertEqual(output.bbox, TRACKER_BOX)
                        self.assertTrue(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                        self.assertEqual(len(transport.requests), 1)
                        self.assertEqual(pipe.motion.reprompts, [TRACKER_BOX])
                    set_branch(pipe, "identity_diff_motion")
                    pre_trigger = frames(pipe, 12, 4)
                    self.assertFalse(any(gate(r)["triggered"] for r in pre_trigger))
                    self.assertTrue(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                    direct = frames(pipe, 16, 1)[0]
                    self.assertTrue(gate(direct)["triggered"])
                    self.assertTrue(gate(direct)["vlm_call_skipped_by_loop"])
                    self.assertEqual(gate(direct)["answer"], "Tracker")
                    self.assertEqual(direct.bbox, TRACKER_BOX)
                    self.assertEqual(len(transport.requests), 1)
                    self.assertEqual(pipe.motion.reprompts, [TRACKER_BOX, TRACKER_BOX])
                    self.assertFalse(pipe.state.vlm_reprompt_direct_next_after_identity_diff_tracker)
                    after = frames(pipe, 17, 2)
                    self.assertFalse(any(gate(r)["triggered"] for r in after))
                    self.assertEqual(len(pipe.motion.reprompts), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
