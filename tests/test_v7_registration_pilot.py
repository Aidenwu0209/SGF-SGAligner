import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v7_registration_pilot as pilot  # noqa: E402
import v7_registration_posthoc as posthoc  # noqa: E402
from safety.registration_consensus import ConsensusConfig  # noqa: E402


def pose(tx=0.0, yaw_deg=0.0):
    angle = np.radians(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    transform[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[0, 3] = tx
    return transform


def worker(direction, replicate, transform=None):
    transform = pose(replicate * 0.001, replicate * 0.05) \
        if transform is None else transform
    return {
        "direction": direction,
        "replicate": replicate,
        "status": "ok",
        "raw_transform": transform,
        "final_transform": transform,
        "rule_b_accepted": True,
        "permutation_provenance_sha256": f"{direction}-{replicate}",
        "evidence_sha256": f"sha-{direction}-{replicate}",
        "icp": {"trace": [{
            "rmse_before_m": 0.01,
            "rmse_after_m": 0.009,
            "fixed_correspondence_rmse_before_m": 0.01,
            "fixed_correspondence_rmse_after_m": 0.009,
            "update_rotation_deg": 0.0,
            "update_translation_m": 0.0,
        }]},
    }


def surface_set_change_regression_case():
    """A fixed random case where the thresholded NN set changes after Kabsch."""
    rng = np.random.default_rng(20260830)
    for trial in range(5):
        source_count = int(rng.integers(10, 40))
        reference_count = int(rng.integers(8, 35))
        source = rng.uniform(-0.4, 0.4, (source_count, 3))
        reference = rng.uniform(-0.4, 0.4, (reference_count, 3))
        if trial % 2 == 0:
            count = min(source_count, reference_count)
            reference[:count] = (
                source[:count] + rng.normal(0, 0.10, (count, 3)))
        angle = rng.uniform(-0.4, 0.4)
        cosine, sine = np.cos(angle), np.sin(angle)
        initial = np.eye(4)
        initial[:3, :3] = [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
        initial[:3, 3] = rng.normal(0, 0.08, 3)
    return source, reference, initial


class StableReplicateTests(unittest.TestCase):
    def test_permutation_is_stable_and_context_separated(self):
        kwargs = dict(
            count=37, pair_id="a_to_b", direction="forward",
            replicate=0, protocol_sha="1" * 64)
        first, first_sha = pilot.stable_row_permutation(**kwargs)
        second, second_sha = pilot.stable_row_permutation(**kwargs)
        other, other_sha = pilot.stable_row_permutation(
            **{**kwargs, "direction": "reverse"})
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_sha, second_sha)
        self.assertFalse(np.array_equal(first, other))
        self.assertNotEqual(first_sha, other_sha)

    def test_reverse_record_is_independently_inverted(self):
        native = pose(0.3, 4.0)
        row = worker("reverse", 0, native)
        record = pilot.worker_consensus_record(
            row, "raw_transform", invert=True)
        np.testing.assert_allclose(record["transform"], np.linalg.inv(native))

    def test_complete_policy_uses_raw_final_and_both_directions(self):
        workers = [worker(direction, replicate)
                   for direction in ("forward", "reverse")
                   for replicate in range(5)]
        result = pilot.aggregate_policy(
            workers, ConsensusConfig(
                repeats=5, quorum=4,
                max_rotation_deg=2.5, max_translation_m=0.05))
        self.assertTrue(result["usable_for_reconstruction"])
        self.assertTrue(result["consensus"]["forward_raw"]["usable"])
        self.assertTrue(result["consensus"]["forward_final"]["usable"])
        self.assertTrue(result["consensus"]["reverse_raw_inverted"]["usable"])
        self.assertTrue(result["consensus"]["reverse_final_inverted"]["usable"])
        self.assertTrue(result["consensus"]["cross_raw"]["usable"])
        self.assertTrue(result["consensus"]["cross_final"]["usable"])


class IcpTraceTests(unittest.TestCase):
    def test_trace_preserves_raw_and_converges_on_translation(self):
        rng = np.random.default_rng(12)
        source = rng.normal(size=(100, 3)) * 0.05
        reference = source + np.array([0.03, 0.0, 0.0])
        result = pilot.segment_icp_with_trace(
            source, reference, np.eye(4), seed=42)
        self.assertTrue(result["converged"])
        self.assertGreaterEqual(result["iterations_run"], 1)
        np.testing.assert_allclose(
            result["transform"][:3, 3], [0.03, 0.0, 0.0], atol=1e-6)
        self.assertTrue(all(
            step["fixed_correspondence_rmse_after_m"]
            <= step["fixed_correspondence_rmse_before_m"] + 1e-12
            for step in result["trace"]))
        self.assertTrue(pilot.trace_gate({"icp": result})["usable"])

    def test_non_monotonic_trace_is_fail_closed(self):
        row = worker("forward", 0)
        row["icp"]["trace"][0][
            "fixed_correspondence_rmse_after_m"] = 0.02
        result = pilot.trace_gate(row)
        self.assertFalse(result["usable"])
        self.assertIn("icp_rmse_not_monotonic", result["rejection_reasons"])

    def test_missing_fixed_metric_is_fail_closed(self):
        row = worker("forward", 0)
        del row["icp"]["trace"][0][
            "fixed_correspondence_rmse_after_m"]
        result = pilot.trace_gate(row)
        self.assertFalse(result["usable"])
        self.assertFalse(result["fixed_metric_complete"])

    def test_surface_set_change_does_not_false_reject_kabsch(self):
        source, reference, initial = surface_set_change_regression_case()
        result = pilot.segment_icp_with_trace(
            source, reference, initial, seed=42)
        first = result["trace"][0]
        self.assertGreater(
            first["surface_rmse_after_m"], first["surface_rmse_before_m"])
        self.assertLessEqual(
            first["fixed_correspondence_rmse_after_m"],
            first["fixed_correspondence_rmse_before_m"] + 1e-12)
        # Isolate the monotonicity decision from the separately unchanged
        # last-update threshold.
        first = dict(first, update_rotation_deg=0.0,
                     update_translation_m=0.0)
        gate = pilot.trace_gate({"icp": {"trace": [first]}})
        self.assertTrue(gate["usable"])
        self.assertEqual("fixed_correspondence_rmse_m", gate["rmse_metric"])


class FrozenWorkerOfflineTests(unittest.TestCase):
    def test_replays_real_frozen_worker_with_fixed_correspondence_gate(self):
        worker_value = os.environ.get("V7_FROZEN_WORKER_JSON")
        if not worker_value:
            self.skipTest("set V7_FROZEN_WORKER_JSON for frozen-worker audit")
        worker_path = Path(worker_value)
        frozen = json.loads(worker_path.read_text())
        unsigned = {key: value for key, value in frozen.items()
                    if key != "evidence_sha256"}
        self.assertEqual(
            frozen["evidence_sha256"], pilot.stable_json_hash(unsigned))
        cached = pilot.load_validated_cache(
            Path(frozen["cache"]["path"]), frozen["pair_id"],
            frozen["cache"]["sha256"])
        data, _ = pilot.build_canonical_pair(
            frozen["pair_id"], with_labels=False)
        pilot.validate_canonical_surfaces(data, cached)
        used = [tuple(row) for row in
                frozen["node_pairs_used_original_index_frame"]]
        source, reference, _ = pilot.surface_union(
            data, used, frozen["direction"])
        replay = pilot.segment_icp_with_trace(
            source, reference, np.asarray(frozen["raw_transform"]),
            seed=int(frozen["icp"]["seed"]))
        np.testing.assert_allclose(
            replay["transform"], np.asarray(frozen["final_transform"]),
            atol=1e-12, rtol=0.0)
        self.assertFalse(all(
            step["rmse_after_m"] <= step["rmse_before_m"] + 1e-12
            for step in frozen["icp"]["trace"]))
        self.assertTrue(all(
            step["fixed_correspondence_rmse_after_m"]
            <= step["fixed_correspondence_rmse_before_m"] + 1e-12
            for step in replay["trace"]))
        self.assertTrue(pilot.trace_gate({"icp": replay})["usable"])


class EvidenceTests(unittest.TestCase):
    def test_atomic_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            pilot.atomic_create_json(path, {"first": True})
            with self.assertRaises(pilot.PilotEvidenceError):
                pilot.atomic_create_json(path, {"second": True})
            self.assertEqual({"first": True}, json.loads(path.read_text()))

    def test_cache_validation_checks_checkpoint_and_provenance(self):
        cache = {
            "cache_schema": pilot.CACHE_SCHEMA,
            "pair_id": "a_to_b",
            "checkpoint_id": pilot.CHECKPOINT_ID,
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
            "input_sha256": "x" * 64,
            "node_corrs": [(0, 1)],
            "provenance": {
                "cache_key": "x" * 64,
                "pair_id": "a_to_b",
                "checkpoint_id": pilot.CHECKPOINT_ID,
                "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
            },
            "geot": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            import torch
            torch.save(cache, path)
            loaded = pilot.load_validated_cache(path, "a_to_b")
            self.assertEqual(pilot.CHECKPOINT_ID, loaded["checkpoint_id"])
            cache["checkpoint_sha256"] = "bad"
            torch.save(cache, path)
            with self.assertRaises(pilot.PilotEvidenceError):
                pilot.load_validated_cache(path, "a_to_b")


class PosthocTests(unittest.TestCase):
    def test_official_label_uses_raw_and_reconstruction_label_uses_final(self):
        aggregate = {
            "pair_id": "a_to_b",
            "outer_repeat": 0,
            "evidence_sha256": "frozen",
            "policies": {"policy": {
                "usable_for_reconstruction": True,
                "selected_observed_forward_medoid": {
                    "raw_transform": pose(0.25),
                    "final_transform": pose(0.05),
                    "worker_evidence_sha256": "worker",
                },
            }},
        }
        with mock.patch.object(
                posthoc, "load_gt_transform", return_value=np.eye(4)):
            result = posthoc.label_aggregate(aggregate)
        labels = result["policies"]["policy"]
        self.assertFalse(labels["official_raw"]["strict"])
        self.assertTrue(labels["reconstruction_final"]["strict"])
        self.assertTrue(labels["accepted_strict_error"])


class IsolationTests(unittest.TestCase):
    def test_gt_loader_is_absent_from_runner_and_present_only_posthoc(self):
        runner = ast.parse((ROOT / "scripts/v7_registration_pilot.py").read_text())
        evaluator = ast.parse(
            (ROOT / "scripts/v7_registration_posthoc.py").read_text())
        runner_imports = {
            alias.name for node in ast.walk(runner)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        evaluator_imports = {
            alias.name for node in ast.walk(evaluator)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("load_gt_transform", runner_imports)
        self.assertNotIn("load_anchor_ids", runner_imports)
        self.assertIn("load_gt_transform", evaluator_imports)

    def test_controller_spawns_worker_processes(self):
        tree = ast.parse((ROOT / "scripts/v7_registration_pilot.py").read_text())
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        run_outer = functions["run_outer"]
        subprocess_calls = [node for node in ast.walk(run_outer)
                            if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "run"]
        worker_literals = {node.value for node in ast.walk(run_outer)
                           if isinstance(node, ast.Constant)
                           and isinstance(node.value, str)}
        self.assertTrue(subprocess_calls)
        self.assertIn("--worker", worker_literals)


if __name__ == "__main__":
    unittest.main()
