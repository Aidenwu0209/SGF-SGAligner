from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v8_calibration90_cache_prepare as prepare


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Calibration90CachePrepareTests(unittest.TestCase):
    def test_reachable_cache_builder_is_ast_gt_free(self):
        audit = prepare.reachable_gt_ast_audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertFalse(audit["with_labels_true"])
        names = [row["function"] for row in audit["reachable_functions"]]
        self.assertIn("v6fix_consistency_audit.build_or_load_cache", names)

    def test_plan_binds_same_B_checkpoint_pairlist_sources_and_no_labels(self):
        plan = prepare.build_plan()
        self.assertEqual(plan["schema"], prepare.PLAN_SCHEMA)
        self.assertEqual(plan["pair_count"], 90)
        self.assertEqual(plan["pair_ids_sha256"],
                         prepare.locked.CANONICAL_PAIRLIST_SHA256)
        self.assertEqual(plan["checkpoint"]["id"], "B")
        self.assertEqual(plan["checkpoint"]["sha256"],
                         prepare.pilot.CHECKPOINT_SHA256)
        self.assertNotEqual(plan["checkpoint"]["sha256"],
                            prepare.locked.OFFICIAL_EPOCH6_SHA256)
        self.assertEqual(plan["authorization"], {
            "labels": False, "workers": False, "posthoc": False,
            "fixed12": False, "official92": False,
        })
        self.assertEqual(
            plan["builder"]["reuse"],
            "v6fix_consistency_audit.build_or_load_cache")

    def test_frozen_plan_validates_and_authorization_tamper_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = prepare.build_plan()
            path.write_text(json.dumps(plan, sort_keys=True))
            validated = prepare.validate_plan(path, file_sha(path))
            self.assertFalse(validated["authorization"]["labels"])
            plan["authorization"]["labels"] = True
            plan["payload_sha256"] = prepare.locked.stable_hash({
                key: value for key, value in plan.items()
                if key != "payload_sha256"})
            path.write_text(json.dumps(plan, sort_keys=True))
            with self.assertRaises(prepare.CachePrepareError):
                prepare.validate_plan(path, file_sha(path))

    def test_execute_failure_never_publishes_partial_final_root(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "fresh-B-calibration90"
            plan = {"_path": "/frozen/plan.json",
                    "_file_sha256": "a" * 64,
                    "payload_sha256": "b" * 64}
            with mock.patch.object(prepare.builder, "load_model",
                                   return_value=object()), \
                    mock.patch.object(
                        prepare.builder, "build_or_load_cache",
                        side_effect=RuntimeError("synthetic builder failure")):
                with self.assertRaises(RuntimeError):
                    prepare.execute(plan, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(parent.glob(".*.incomplete-*")), [])

    def test_execute_refuses_existing_output_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with mock.patch.object(prepare.builder, "load_model") as loader:
                with self.assertRaises(prepare.CachePrepareError):
                    prepare.execute({}, output)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
