import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

import scripts.v11_matched_region_runner as runner


ROOT = Path(__file__).resolve().parents[1]


class V11RunnerTests(unittest.TestCase):
    def test_runner_imports_no_label_gt_or_posthoc_loader(self):
        tree = ast.parse((ROOT / "scripts/v11_matched_region_runner.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = {"load_gt_transform", "load_anchor_ids",
                     "load_oracle_anchor_ids", "v8_stage_order_posthoc"}
        self.assertFalse(imported & forbidden)
        source = (ROOT / "scripts/v11_matched_region_runner.py").read_text()
        self.assertNotIn("with_labels=True", source)

    def test_rule_b_constants_are_exact(self):
        expected = copy.deepcopy(runner.EXPECTED_RULE_B)
        self.assertEqual(expected, runner.dfx.RULE_THRESHOLDS)
        self.assertEqual(runner.assert_frozen_rule_b(),
                         runner.stable_json_hash(expected))

    def test_atomic_json_resume_and_stale_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            payload = {"schema": "x", "pair_id": "p", "value": 1}
            first = runner.load_or_create_json(
                path, payload, {"pair_id": "p"}, "x")
            second = runner.load_or_create_json(
                path, payload, {"pair_id": "p"}, "x")
            self.assertEqual(first["_file_sha256"], second["_file_sha256"])
            with self.assertRaises(runner.V11EvidenceError):
                runner.load_or_create_json(
                    path, {**payload, "value": 99}, {"pair_id": "p"}, "x")
            changed = json.loads(path.read_text())
            changed["value"] = 2
            path.write_text(json.dumps(changed))
            with self.assertRaises(runner.V11EvidenceError):
                runner.load_or_create_json(
                    path, payload, {"pair_id": "p"}, "x")

    def test_atomic_torch_resume_and_stale_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            payload = {"schema": "x", "pair_id": "p", "value": 1}
            runner.load_or_create_torch(
                path, payload, {"pair_id": "p"}, "x")
            same = runner.load_or_create_torch(
                path, {}, {"pair_id": "p"}, "x")
            self.assertEqual(same["value"], 1)
            changed = torch.load(path, map_location="cpu", weights_only=False)
            changed["pair_id"] = "other"
            torch.save(changed, path)
            with self.assertRaises(runner.V11EvidenceError):
                runner.load_or_create_torch(
                    path, {}, {"pair_id": "p"}, "x")

    @staticmethod
    def gate(sha, usable=True):
        return {
            "hypothesis_sha256": sha, "worker_failures": 0,
            "gate": {
                "cross_final": {"usable": usable},
                "medoid_rule_b": {
                    "forward": {"usable": usable},
                    "reverse": {"usable": usable}},
                "fresh_v8_qualified": usable,
            },
        }

    def test_multiple_safe_is_rejected(self):
        result = runner.selector_from_hypothesis_gates(
            [self.gate("a"), self.gate("b")], known_bad=False)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "multiple_safe_hypotheses")

    def test_known_bad_is_always_vetoed(self):
        result = runner.selector_from_hypothesis_gates(
            [self.gate("a")], known_bad=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "known_bad_veto")

    def test_whole_scene_cannot_enter_selector(self):
        diagnostic = self.gate("whole_scene", usable=True)
        result = runner.selector_from_hypothesis_gates([], known_bad=False)
        self.assertFalse(result["accepted"])
        self.assertNotIn(diagnostic["hypothesis_sha256"], str(result))

    def test_pilot_pairlist_is_frozen_positions_plus_known_bad(self):
        rows = [{"pair_id": f"p{index}"} for index in range(89)]
        rows[10]["pair_id"] = runner.KNOWN_BAD
        observed = runner.pilot_pair_ids(rows)
        self.assertEqual(observed[:3], ["p0", "p44", "p88"])
        self.assertEqual(observed[3], runner.KNOWN_BAD)


if __name__ == "__main__":
    unittest.main()
