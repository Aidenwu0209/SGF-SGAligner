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

import v8_calibration90_locked as locked


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Calibration90LockedTests(unittest.TestCase):
    def test_canonical_pairlist_is_only_opaque_90_ids(self):
        pair_ids = locked.read_pairlist()
        self.assertEqual(len(pair_ids), 90)
        self.assertEqual(len(set(pair_ids)), 90)
        self.assertEqual(locked.pair_ids_sha256(pair_ids),
                         locked.CANONICAL_PAIRLIST_SHA256)

    def test_controller_has_no_gt_loader_or_posthoc_metric_symbols(self):
        source = Path(locked.__file__).read_text()
        self.assertNotIn("load_gt_transform", source)
        self.assertNotIn("load_anchor_ids", source)
        self.assertNotIn("with_labels=True", source)

    def test_historical_floor_is_frozen_not_cli_tunable(self):
        self.assertEqual(locked.THRESHOLDS, {
            "completed": 90, "strict_min": 6, "relaxed_min": 8,
            "accepted_correct_min": 5, "accepted_error_max": 0,
            "repeatable_pairs": 90, "exceptions_max": 0,
            "nonfinite_max": 0, "cache_mismatches_max": 0,
        })
        source = Path(locked.__file__).read_text()
        self.assertNotIn("--strict-min", source)
        self.assertNotIn("--accepted-correct-min", source)

    def test_legacy_cache_audit_is_metadata_only_and_incompatible(self):
        audit = locked.audit_legacy_cache()
        self.assertEqual(audit["pair_count"], 90)
        self.assertTrue(audit["ordered_pair_ids_match"])
        self.assertFalse(audit["eligible_as_v8_B_worker_cache"])
        self.assertFalse(audit["arrays_or_labels_opened"])
        self.assertEqual(audit["checkpoint_sha256_values"],
                         [locked.OFFICIAL_EPOCH6_SHA256])

    def test_single_use_claim_is_durable_and_second_claim_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "_file_sha256": "a" * 64,
                "_path": "/frozen/manifest.json",
                "selection89_winner": {"file_sha256": "b" * 64},
                "single_use": {"claim_root": str(root)},
            }
            destination = locked.claim_single_use(manifest, claim_root=root)
            self.assertTrue(destination.is_file())
            with self.assertRaises(locked.Calibration90Error):
                locked.claim_single_use(manifest, claim_root=root)

    def test_winner_validator_accepts_only_unique_fixed_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "winner.json"
            value = {
                "schema": locked.WINNER_SCHEMA,
                "status": "FROZEN_UNIQUE_WINNER",
                "split": "selection89",
                "candidate_id": "V8_FINAL_FIRST_Q4_R5_T0.10_FIXED_TRACE",
                "checkpoint_sha256": locked.pilot.CHECKPOINT_SHA256,
                "config": {"repeats": 5, "quorum": 4,
                           "max_rotation_deg": 5.0,
                           "max_translation_m": 0.10},
                "selection89_gate_passed": True,
                "unique_winner": True,
                "selection_manifest_sha256": "1" * 64,
                "worker_batch_sha256": "2" * 64,
                "posthoc_sha256": "3" * 64,
                "cache_inventory_sha256": "4" * 64,
                "code_inventory_sha256": "5" * 64,
            }
            value["evidence_sha256"] = locked.stable_hash(value)
            path.write_text(json.dumps(value))
            validated = locked.validate_winner(path, file_sha(path))
            self.assertTrue(validated["unique_winner"])
            value["config"]["quorum"] = 3
            value["evidence_sha256"] = locked.stable_hash({
                key: item for key, item in value.items()
                if key != "evidence_sha256"})
            path.write_text(json.dumps(value))
            with self.assertRaises(locked.Calibration90Error):
                locked.validate_winner(path, file_sha(path))

    def test_freezer_binds_winner_cache_code_and_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            winner_path = root / "winner.json"
            winner = {
                "evidence_sha256": "6" * 64,
                "candidate_id": "V8_FINAL_FIRST_Q4_R5_T0.10_FIXED_TRACE",
                "cache_inventory_sha256": "7" * 64,
                "code_inventory_sha256": "8" * 64,
                "config": {"repeats": 5, "quorum": 4,
                           "max_rotation_deg": 5.0,
                           "max_translation_m": 0.10},
            }
            winner_path.write_text("{}")
            pair_ids = locked.read_pairlist()
            rows = [{"pair_id": pair_id, "cache_sha256": "9" * 64,
                     "cache_bytes": 1, "input_sha256": "a" * 64,
                     "node_corr_count": 1} for pair_id in pair_ids]
            with mock.patch.object(locked, "validate_winner",
                                   return_value=winner), \
                    mock.patch.object(locked, "_cache_inventory",
                                      return_value=(rows, "b" * 64)), \
                    mock.patch.object(
                        locked, "_cache_preparation_receipt",
                        return_value={"path": "/frozen/cache_receipt.json",
                                      "file_sha256": "d" * 64,
                                      "evidence_sha256": "e" * 64,
                                      "plan": {"path": "/frozen/plan.json",
                                               "file_sha256": "f" * 64,
                                               "payload_sha256": "0" * 64},
                                      "cache_inventory_sha256": "1" * 64}):
                manifest = locked.freeze_manifest(
                    winner_path=winner_path, winner_sha256="c" * 64,
                    cache_root=root)
            self.assertEqual(manifest["thresholds"], locked.THRESHOLDS)
            self.assertEqual(manifest["config"], winner["config"])
            self.assertEqual(manifest["cache_contract"]["inventory_sha256"],
                             "b" * 64)
            self.assertEqual(
                manifest["cache_contract"]["preparation_receipt"]
                ["plan"]["file_sha256"], "f" * 64)
            self.assertEqual(
                set(manifest["source_files"])
                & {"calibration_controller", "calibration_worker",
                   "calibration_posthoc"},
                {"calibration_controller", "calibration_worker",
                 "calibration_posthoc"})
            self.assertFalse(manifest["gt_separation"]["fixed12_authorized"])
            self.assertFalse(manifest["gt_separation"]["official92_authorized"])

    def test_cache_receipt_binds_plan_and_every_input_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = [{"pair_id": pair_id, "cache_sha256": "1" * 64,
                      "cache_bytes": 7, "input_sha256": "2" * 64,
                      "node_corr_count": 3}
                     for pair_id in locked.read_pairlist()]
            prepared = [dict(row, embedding_sha256="3" * 64,
                             similarity_sha256="4" * 64,
                             geot_entry_count=5) for row in pairs]
            value = {
                "schema": locked.CACHE_PREPARE_RECEIPT_SCHEMA,
                "status": "GT_FREE_B_CACHE_COMPLETE",
                "split": locked.SPLIT,
                "plan": {"path": "/frozen/plan.json",
                         "file_sha256": "5" * 64,
                         "payload_sha256": "6" * 64},
                "checkpoint_sha256": locked.pilot.CHECKPOINT_SHA256,
                "pair_count": 90,
                "pair_ids_sha256": locked.pair_ids_sha256(
                    [row["pair_id"] for row in pairs]),
                "pairs": prepared,
                "cache_inventory_sha256": locked.stable_hash(prepared),
                "gt_ast_audit": {"status": "PASS"},
                "labels_loaded": False,
                "workers_run": False,
                "posthoc_run": False,
            }
            value["evidence_sha256"] = locked.stable_hash(value)
            path = root / "cache_receipt.json"
            path.write_text(json.dumps(value))
            receipt = locked._cache_preparation_receipt(root, pairs)
            self.assertEqual(receipt["plan"]["file_sha256"], "5" * 64)
            value["pairs"][0]["input_sha256"] = "7" * 64
            value["evidence_sha256"] = locked.stable_hash({
                key: item for key, item in value.items()
                if key != "evidence_sha256"})
            path.write_text(json.dumps(value))
            with self.assertRaises(locked.Calibration90Error):
                locked._cache_preparation_receipt(root, pairs)

    def test_posthoc_exists_but_is_not_imported_by_controller(self):
        path = ROOT / "scripts/v8_calibration90_posthoc_gate.py"
        source = path.read_text()
        self.assertIn("load_gt_transform", source)
        self.assertIn("official92_authorized", source)
        controller_source = Path(locked.__file__).read_text()
        self.assertNotIn("import v8_calibration90_posthoc_gate",
                         controller_source)


if __name__ == "__main__":
    unittest.main()
