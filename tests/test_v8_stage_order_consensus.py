import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config,
    cluster_direction,
    evaluate_stage_order,
    fixed_trace_gate,
)
import v7_registration_batch as v7_batch  # noqa: E402
import v7_registration_pilot as v7_pilot  # noqa: E402
import v8_stage_order_replay as replay  # noqa: E402


def pose(tx=0.0, yaw_deg=0.0):
    angle = np.radians(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [[cosine, -sine, 0.0],
                     [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    value[0, 3] = tx
    return value


def worker(direction, replicate, *, rule_pass=True, raw=None, final=None,
           fixed_trace=True):
    final = pose(replicate * .001, replicate * .05) if final is None else final
    raw = final if raw is None else raw
    reason = [] if rule_pass else ["synthetic_rule_failure"]
    step = {
        "rmse_before_m": .02, "rmse_after_m": .03,
        "update_rotation_deg": 0.0, "update_translation_m": 0.0,
    }
    if fixed_trace:
        step.update({"fixed_correspondence_rmse_before_m": .02,
                     "fixed_correspondence_rmse_after_m": .01})
    return {
        "direction": direction, "replicate": replicate, "status": "ok",
        "raw_transform": raw, "final_transform": final,
        "permutation_provenance_sha256": f"{direction}-{replicate}",
        "evidence_sha256": f"sha-{direction}-{replicate}",
        "rule_b_features": {"pass": rule_pass},
        "rule_b_accepted": rule_pass,
        "decision": {"rejection_reasons": reason},
        "icp": {"trace": [step]},
    }


def synthetic_rule(features):
    return [] if features.get("pass") else ["synthetic_rule_failure"]


class DirectionalConsensusTests(unittest.TestCase):
    def test_unique_quorum_cluster_selects_observed_medoid(self):
        records = [{"status": "ok", "transform": pose(i * .01),
                    "stable_signature": str(i)} for i in range(4)]
        records.append({"status": "ok", "transform": pose(1.0),
                        "stable_signature": "4"})
        result = cluster_direction(records, V8Config())
        self.assertTrue(result["usable"])
        self.assertEqual(4, result["clique_sizes"][0])
        self.assertIn(result["medoid_original_index"], range(4))

    def test_equal_largest_clusters_fail_closed(self):
        records = [
            {"status": "ok", "transform": pose(0.00)},
            {"status": "ok", "transform": pose(0.09)},
            {"status": "ok", "transform": pose(0.18)},
            {"status": "ok", "transform": pose(0.27)},
            {"status": "ok", "transform": pose(0.36)},
        ]
        result = cluster_direction(
            records, V8Config(quorum=3, max_translation_m=.19))
        self.assertFalse(result["usable"])
        self.assertIn("largest_clique_not_unique",
                      result["rejection_reasons"])


class StageOrderTests(unittest.TestCase):
    def test_rule_b_is_applied_only_to_observed_medoid_after_clustering(self):
        rows = []
        for direction in ("forward", "reverse"):
            for replicate in range(5):
                # Four non-medoid workers fail Rule-B. V7 prefiltering would
                # remove the geometric quorum; V8 still clusters all five.
                rows.append(worker(direction, replicate,
                                   rule_pass=(replicate == 2)))
        result = evaluate_stage_order(rows, V8Config(), synthetic_rule)
        self.assertTrue(result["usable_for_reconstruction"])
        self.assertTrue(result["medoid_rule_b"]["forward"]["usable"])
        self.assertEqual(
            5, result["directional_final_consensus"]["forward"][
                "clique_sizes"][0])

    def test_medoid_rule_b_failure_vetoes(self):
        rows = [worker(direction, replicate, rule_pass=False)
                for direction in ("forward", "reverse")
                for replicate in range(5)]
        result = evaluate_stage_order(rows, V8Config(), synthetic_rule)
        self.assertFalse(result["usable_for_reconstruction"])
        self.assertFalse(result["medoid_rule_b"]["forward"]["usable"])

    def test_raw_consensus_is_diagnostic_not_hard_gate(self):
        rows = [worker(direction, replicate, raw=pose(replicate * 2.0))
                for direction in ("forward", "reverse")
                for replicate in range(5)]
        result = evaluate_stage_order(rows, V8Config(), synthetic_rule)
        self.assertTrue(result["usable_for_reconstruction"])
        self.assertFalse(result["raw_consensus_diagnostic_only"][
            "forward"]["usable"])

    def test_legacy_trace_can_replay_but_cannot_fresh_qualify(self):
        rows = [worker(direction, replicate, fixed_trace=False)
                for direction in ("forward", "reverse")
                for replicate in range(5)]
        research = evaluate_stage_order(
            rows, V8Config(), synthetic_rule, require_fixed_trace=False)
        fresh = evaluate_stage_order(
            rows, V8Config(), synthetic_rule, require_fixed_trace=True)
        self.assertTrue(research["usable_for_reconstruction"])
        self.assertFalse(research["fresh_v8_qualified"])
        self.assertFalse(fresh["usable_for_reconstruction"])

    def test_fixed_trace_uses_fixed_correspondences_not_dynamic_rmse(self):
        row = worker("forward", 0)
        gate = fixed_trace_gate(row)
        self.assertTrue(gate["usable"])
        self.assertTrue(gate["fixed_rmse_non_increasing"])


class IsolationTests(unittest.TestCase):
    def test_gt_import_exists_only_in_posthoc_process(self):
        runner = ast.parse(
            (ROOT / "scripts/v8_stage_order_replay.py").read_text())
        posthoc = ast.parse(
            (ROOT / "scripts/v8_stage_order_posthoc.py").read_text())

        def names(tree):
            return {alias.name for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in node.names}

        self.assertNotIn("load_gt_transform", names(runner))
        self.assertNotIn("load_anchor_ids", names(runner))
        self.assertIn("load_gt_transform", names(posthoc))

    def test_protocol_freezes_only_one_configuration(self):
        source = (ROOT / "scripts/v8_stage_order_replay.py").read_text()
        self.assertIn("CONFIG = V8Config()", source)
        self.assertNotIn("POLICIES", source)


class LegacyReadOnlyCompatibilityTests(unittest.TestCase):
    def _synthetic_worker(self, path):
        pair_id = "a_to_b"
        protocol_sha = "a" * 64
        cache_sha = "b" * 64
        count = 9
        permutation, provenance = v7_pilot.stable_row_permutation(
            count, pair_id=pair_id, direction="forward", replicate=0,
            protocol_sha=protocol_sha)
        permutation_sha = __import__("hashlib").sha256(
            np.ascontiguousarray(permutation.astype(np.int64)).tobytes()
        ).hexdigest()
        transform = np.eye(4)
        row = {
            "schema": v7_pilot.WORKER_SCHEMA,
            "pair_id": pair_id,
            "direction": "forward",
            "replicate": 0,
            "status": "ok",
            "cache": {
                "sha256": cache_sha,
                "checkpoint_id": v7_pilot.CHECKPOINT_ID,
                "checkpoint_sha256": v7_pilot.CHECKPOINT_SHA256,
            },
            "protocol_sha256": protocol_sha,
            "raw_transform": transform.tolist(),
            "raw_transform_sha256": v7_pilot.array_sha256(transform),
            "final_transform": transform.tolist(),
            "final_transform_sha256": v7_pilot.array_sha256(transform),
            "correspondence_count": count,
            "permutation_provenance_sha256": provenance,
            "permutation_sha256": permutation_sha,
            "source_hashes": {"runner": "c" * 64,
                              "consensus": "d" * 64},
        }
        row["evidence_sha256"] = v7_pilot.stable_json_hash(row)
        path.write_text(json.dumps(row))
        return row, {
            "pair_id": pair_id, "cache_sha256": cache_sha,
        }, protocol_sha, {
            "source_files": {
                "pilot_runner": {"sha256": "c" * 64},
                "consensus": {"sha256": "d" * 64},
            }}

    def test_synthetic_legacy_worker_binds_transform_permutation_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forward_00.json"
            _, pair, protocol, snapshot = self._synthetic_worker(path)
            loaded = replay._validate_legacy_worker(
                path, pair=pair, direction="forward", replicate=0,
                protocol_sha=protocol, snapshot=snapshot)
            self.assertEqual("ok", loaded["status"])

            tampered = json.loads(path.read_text())
            tampered["final_transform"][0][3] = .5
            tampered["evidence_sha256"] = v7_pilot.stable_json_hash({
                key: value for key, value in tampered.items()
                if key != "evidence_sha256"})
            path.write_text(json.dumps(tampered))
            with self.assertRaises(replay.V8ReplayError):
                replay._validate_legacy_worker(
                    path, pair=pair, direction="forward", replicate=0,
                    protocol_sha=protocol, snapshot=snapshot)

    def test_arbitrary_legacy_receipt_cannot_enter_known_compatibility_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            raw = {
                "schema": v7_batch.BATCH_SCHEMA,
                "source_snapshot": {"snapshot_sha256": "x"},
                "evidence_sha256": "y",
            }
            path.write_text(json.dumps(raw))
            with self.assertRaises(replay.V8ReplayError):
                replay._validate_legacy_source_batch(
                    path, {"_file_sha256": "a" * 64, "pairs": []}, raw,
                    enforce_known_identity=True)

    @unittest.skipUnless(os.environ.get("V8_LEGACY_BATCH_RECEIPT"),
                         "set V8_LEGACY_BATCH_RECEIPT for real frozen audit")
    def test_real_known_legacy_batch_full_chain(self):
        manifest = ROOT / "outputs/v7_pilot_manifest_seal_20260830" \
            / "v7_pilot_manifest.json"
        _, receipt, mode = replay.validate_source_batch(
            Path(os.environ["V8_LEGACY_BATCH_RECEIPT"]), manifest,
            v7_batch.DEFAULT_MANIFEST_SHA256)
        self.assertEqual("KNOWN_SUPERSEDED_V7_READ_ONLY", mode)
        self.assertEqual(replay.LEGACY_V7_BATCH_EVIDENCE_SHA256,
                         receipt["evidence_sha256"])


if __name__ == "__main__":
    unittest.main()
