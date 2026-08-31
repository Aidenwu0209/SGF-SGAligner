import ast
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v7_registration_batch as batch  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402


def pair_id(index):
    return (f"00000000-0000-0000-0000-{index:012x}_to_"
            f"10000000-0000-0000-0000-{index:012x}")


def manifest_payload():
    pair_ids = [pair_id(index) for index in range(12)]
    return {
        "schema": batch.MANIFEST_SCHEMA,
        "status": "FROZEN",
        "pair_count": 12,
        "pairs": [
            {"pair_id": value, "cache_sha256": f"{index + 1:064x}",
             "role": "audit"}
            for index, value in enumerate(pair_ids)
        ],
        "pair_ids_sha256": batch.pair_ids_sha256(pair_ids),
        "checkpoint_id": pilot.CHECKPOINT_ID,
        "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
        "protocol_sha256": "a" * 64,
        "audit_metadata": {"note": "non-decisional"},
    }


def write_manifest(directory, payload=None):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(payload or manifest_payload(), sort_keys=True))
    return path, pilot.sha256_file(path)


def aggregate(pair="p", outer=0, state=False):
    policies = {
        pilot.policy_name(config): {"usable_for_reconstruction": state}
        for config in pilot.POLICIES
    }
    return {
        "schema": pilot.SCHEMA,
        "pair_id": pair,
        "outer_repeat": outer,
        "cache": {"sha256": "c" * 64},
        "protocol": {"sha256": "d" * 64},
        "batch": {"manifest_sha256": "e" * 64,
                  "source_snapshot_sha256": "f" * 64},
        "workers": {"requested": 10, "completed": 10},
        "worker_permutation_bindings": [
            {"direction": direction, "replicate": replicate,
             "correspondence_count": 10,
             "permutation_provenance_sha256": f"{index + 1:064x}",
             "permutation_sha256": f"{index + 11:064x}"}
            for index, (direction, replicate) in enumerate(
                ([("forward", i) for i in range(5)]
                 + [("reverse", i) for i in range(5)]))
        ],
        "policies": policies,
    }


class ManifestTests(unittest.TestCase):
    def test_committed_default_manifest_matches_frozen_schema_and_sha(self):
        result = batch.validate_manifest(
            batch.DEFAULT_MANIFEST, batch.DEFAULT_MANIFEST_SHA256)
        self.assertEqual(12, len(result["pairs"]))
        self.assertEqual(pilot.CHECKPOINT_SHA256,
                         result["checkpoint_sha256"])
        self.assertEqual(batch.FORMAL_EVIDENCE_MODE,
                         result["_evidence_mode"])
        self.assertTrue(result["_formal_preregistered"])

    def test_frozen_manifest_validates_exact_12_and_order_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = write_manifest(directory)
            with mock.patch.object(batch.pilot, "protocol_sha256",
                                   return_value="a" * 64):
                result = batch.validate_manifest(
                    path, digest, allow_non_preregistered=True)
        self.assertEqual(12, len(result["pairs"]))
        self.assertEqual(digest, result["_file_sha256"])
        self.assertEqual(batch.RESEARCH_EVIDENCE_MODE,
                         result["_evidence_mode"])
        self.assertFalse(result["_formal_preregistered"])

    def test_non_default_manifest_requires_explicit_research_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = write_manifest(directory)
            with mock.patch.object(batch.pilot, "protocol_sha256",
                                   return_value="a" * 64):
                with self.assertRaisesRegex(
                        batch.BatchEvidenceError,
                        "research-non-preregistered"):
                    batch.validate_manifest(path, digest)

    def test_duplicate_pair_and_unknown_decision_field_fail_closed(self):
        for mutation in ("duplicate", "unknown"):
            payload = manifest_payload()
            if mutation == "duplicate":
                payload["pairs"][1]["pair_id"] = payload["pairs"][0]["pair_id"]
                payload["pair_ids_sha256"] = batch.pair_ids_sha256(
                    [row["pair_id"] for row in payload["pairs"]])
            else:
                payload["threshold_override"] = 1
            with self.subTest(mutation=mutation), \
                    tempfile.TemporaryDirectory() as directory:
                path, digest = write_manifest(directory, payload)
                with mock.patch.object(batch.pilot, "protocol_sha256",
                                       return_value="a" * 64):
                    with self.assertRaises(batch.BatchEvidenceError):
                        batch.validate_manifest(
                            path, digest, allow_non_preregistered=True)

    def test_checkpoint_protocol_and_cache_sha_are_bound(self):
        payload = manifest_payload()
        payload["checkpoint_sha256"] = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            path, digest = write_manifest(directory, payload)
            with mock.patch.object(batch.pilot, "protocol_sha256",
                                   return_value="a" * 64):
                with self.assertRaises(batch.BatchEvidenceError):
                    batch.validate_manifest(
                        path, digest, allow_non_preregistered=True)

    def test_preflight_binds_every_cache_before_workers(self):
        payload = manifest_payload()
        fake_paths = {
            row["pair_id"]: Path(f"/cache/{row['pair_id']}.pt")
            for row in payload["pairs"]
        }
        with mock.patch.object(
                batch.pilot, "cache_path",
                side_effect=lambda _root, pair: fake_paths[pair]) as paths, \
                mock.patch.object(
                    batch.pilot, "sha256_file",
                    side_effect=lambda path: next(
                        row["cache_sha256"] for row in payload["pairs"]
                        if fake_paths[row["pair_id"]] == path)):
            batch.preflight_caches(payload)
        self.assertEqual(12, paths.call_count)


class RepositoryTests(unittest.TestCase):
    def _results(self, status):
        return [
            types.SimpleNamespace(stdout="1" * 40),
            types.SimpleNamespace(stdout=status),
        ]

    def test_only_designated_output_may_be_untracked(self):
        output = batch.CODE_ROOT / "outputs" / "new_batch"
        allowed = b"?? outputs/new_batch/receipt.json\0"
        with mock.patch.object(batch.subprocess, "run",
                               side_effect=self._results(allowed)):
            state = batch.repository_state(output)
        self.assertFalse(state["tracked_dirty"])

        source = b"?? scripts/unreviewed.py\0"
        with mock.patch.object(batch.subprocess, "run",
                               side_effect=self._results(source)):
            with self.assertRaises(batch.BatchEvidenceError):
                batch.repository_state(output)

    def test_tracked_change_is_rejected_even_under_output(self):
        output = batch.CODE_ROOT / "outputs" / "new_batch"
        dirty = b" M outputs/new_batch/tracked.json\0"
        with mock.patch.object(batch.subprocess, "run",
                               side_effect=self._results(dirty)):
            with self.assertRaises(batch.BatchEvidenceError):
                batch.repository_state(output)

    def test_broad_repository_output_root_is_rejected(self):
        for output in (batch.CODE_ROOT, batch.CODE_ROOT / "outputs"):
            with self.subTest(output=output):
                with self.assertRaises(batch.BatchEvidenceError):
                    batch.validate_output_root(output)

    def test_source_snapshot_binds_required_sources_and_runtime(self):
        repository = {"head": "1" * 40, "tracked_dirty": False,
                      "untracked_outside_output": False}
        with mock.patch.object(batch, "_package_record",
                               return_value={"available": True,
                                             "version": "test",
                                             "module_file": "module"}):
            result = batch.source_snapshot(repository)
        self.assertEqual(set(batch.SOURCE_FILES),
                         set(result["source_files"]))
        self.assertEqual({"numpy", "scipy", "torch", "pygcransac"},
                         set(result["packages"]))
        self.assertEqual("1", result["worker_environment"]["OMP_NUM_THREADS"])
        self.assertEqual("", result["worker_environment"]["CUDA_VISIBLE_DEVICES"])


class ResumeAndRepeatabilityTests(unittest.TestCase):
    def test_partial_outer_is_dirty_and_complete_shape_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "forward_00.json").write_text("{}")
            with self.assertRaises(batch.BatchEvidenceError):
                batch._validate_directory_shape(run_dir, complete=False)

    def test_repeatability_ignores_random_transforms_and_rule_b_patterns(self):
        first = aggregate(outer=0, state=True)
        second = aggregate(outer=1, state=True)
        first["random_transform_hash"] = "a"
        second["random_transform_hash"] = "b"
        first["rule_b_pattern"] = [True, False]
        second["rule_b_pattern"] = [False, True]
        result = batch.policy_repeatability([first, second])
        self.assertTrue(result["input_and_schema_repeatable"])
        self.assertTrue(all(row["repeatable"]
                            for row in result["policies"].values()))

    def test_repeatability_reports_mixed_final_decision(self):
        first = aggregate(outer=0, state=True)
        second = aggregate(outer=1, state=False)
        result = batch.policy_repeatability([first, second])
        self.assertTrue(all(row["outcome"] == "mixed"
                            for row in result["policies"].values()))
        self.assertTrue(all(not row["repeatable"]
                            for row in result["policies"].values()))

    def test_repeatability_binds_direction_replicate_permutations(self):
        first = aggregate(outer=0, state=False)
        second = aggregate(outer=1, state=False)
        second["worker_permutation_bindings"][0][
            "permutation_sha256"] = "f" * 64
        with self.assertRaisesRegex(batch.BatchEvidenceError,
                                    "bindings differ"):
            batch.policy_repeatability([first, second])

    def test_worker_permutation_is_recomputed_not_merely_present(self):
        permutation = np.array([2, 0, 1], dtype=np.int64)
        permutation_sha = hashlib.sha256(permutation.tobytes()).hexdigest()
        row = {
            "status": "ok", "direction": "forward", "replicate": 2,
            "cache": {"checkpoint_id": pilot.CHECKPOINT_ID,
                      "checkpoint_sha256": pilot.CHECKPOINT_SHA256},
            "raw_transform": np.eye(4).tolist(),
            "final_transform": np.eye(4).tolist(),
            "correspondence_count": 3,
            "permutation_provenance_sha256": "a" * 64,
            "permutation_sha256": permutation_sha,
        }
        with mock.patch.object(batch.pilot, "load_worker",
                               return_value=row), \
                mock.patch.object(batch.pilot, "stable_row_permutation",
                                  return_value=(permutation, "a" * 64)):
            loaded = batch._load_batch_worker(
                Path("unused"), pair_id="pair", direction="forward",
                replicate=2, cache_sha="b" * 64, protocol_sha="c" * 64)
            self.assertEqual(permutation_sha, loaded["permutation_sha256"])
            row["permutation_sha256"] = "d" * 64
            with self.assertRaisesRegex(batch.BatchEvidenceError,
                                        "permutation binding"):
                batch._load_batch_worker(
                    Path("unused"), pair_id="pair", direction="forward",
                    replicate=2, cache_sha="b" * 64,
                    protocol_sha="c" * 64)

    def test_atomic_receipt_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            pilot.atomic_create_json(path, {"schema": batch.BATCH_SCHEMA})
            with self.assertRaises(pilot.PilotEvidenceError):
                pilot.atomic_create_json(path, {"schema": "replacement"})


class IsolationTests(unittest.TestCase):
    def test_pair_posthoc_binds_pair_outers_aggregates_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = pair_id(1)
            aggregate_rows = []
            runs = []
            policy_names = {pilot.policy_name(config)
                            for config in pilot.POLICIES}
            for outer in range(2):
                aggregate = {
                    "policies": {name: {} for name in policy_names},
                }
                aggregate["evidence_sha256"] = pilot.stable_json_hash(
                    aggregate)
                path = root / f"aggregate_{outer}.json"
                path.write_text(json.dumps(aggregate))
                aggregate_rows.append({
                    "path": str(path), "sha256": pilot.sha256_file(path)})
                runs.append({
                    "pair_id": pair, "outer_repeat": outer,
                    "gt_free_evidence_sha256": aggregate["evidence_sha256"],
                    "policies": {name: {} for name in policy_names},
                })
            pair_receipt = {
                "schema": batch.PAIR_RECEIPT_SCHEMA,
                "status": "GT_FREE_COMPLETE", "pair_id": pair,
                "outer_repeats": 2, "posthoc_not_run": True,
                "aggregates": aggregate_rows,
                "batch": {"manifest_sha256": "a" * 64,
                          "source_snapshot_sha256": "b" * 64,
                          "evidence_mode": batch.FORMAL_EVIDENCE_MODE},
            }
            pair_receipt["evidence_sha256"] = pilot.stable_json_hash(
                pair_receipt)
            pair_receipt_path = root / "pair.json"
            pair_receipt_path.write_text(json.dumps(pair_receipt))
            posthoc = {
                "schema": batch.PAIR_POSTHOC_SCHEMA,
                "status": "POSTHOC_COMPLETE", "pair_id": pair,
                "outer_repeats": 2,
                "receipt": {"sha256": pilot.sha256_file(pair_receipt_path)},
                "manifest_sha256": "a" * 64,
                "source_snapshot_sha256": "b" * 64,
                "evidence_mode": batch.FORMAL_EVIDENCE_MODE,
                "source_sha256": pilot.sha256_file(
                    batch.CODE_ROOT / "scripts/v7_registration_posthoc.py"),
                "aggregate_bindings": [
                    {"outer_repeat": outer, "path": row["path"],
                     "sha256": row["sha256"]}
                    for outer, row in enumerate(aggregate_rows)],
                "runs": runs,
            }
            posthoc["evidence_sha256"] = pilot.stable_json_hash(posthoc)
            posthoc_path = root / "posthoc.json"
            posthoc_path.write_text(json.dumps(posthoc))
            batch._validate_pair_posthoc(posthoc_path, pair_receipt_path)
            posthoc["pair_id"] = pair_id(2)
            posthoc["evidence_sha256"] = pilot.stable_json_hash(
                {key: value for key, value in posthoc.items()
                 if key != "evidence_sha256"})
            posthoc_path.write_text(json.dumps(posthoc))
            with self.assertRaisesRegex(batch.BatchEvidenceError,
                                        "receipt binding"):
                batch._validate_pair_posthoc(
                    posthoc_path, pair_receipt_path)

    def test_batch_has_no_gt_import_and_posthoc_is_subprocess_only(self):
        tree = ast.parse(
            (ROOT / "scripts/v7_registration_batch.py").read_text())
        imports = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("load_gt_transform", imports)
        self.assertNotIn("load_anchor_ids", imports)
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        posthoc = functions["run_posthoc"]
        validations = [node.lineno for node in ast.walk(posthoc)
                       if isinstance(node, ast.Call)
                       and getattr(node.func, "id", "")
                       == "validate_batch_receipt"]
        processes = [node.lineno for node in ast.walk(posthoc)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "run"]
        self.assertTrue(validations and processes)
        self.assertLess(min(validations), min(processes))

    def test_batch_does_not_authorise_downstream_formal_splits(self):
        text = (ROOT / "scripts/v7_registration_batch.py").read_text()
        self.assertNotIn("v6_calib_fixed12", text)
        self.assertNotIn("official92", text.lower())

    def test_resume_posthoc_path_revalidates_frozen_receipt(self):
        tree = ast.parse(
            (ROOT / "scripts/v7_registration_batch.py").read_text())
        functions = {node.name: node for node in tree.body
                     if isinstance(node, ast.FunctionDef)}
        run_posthoc = functions["run_posthoc"]
        called = {getattr(node.func, "id", "") for node in ast.walk(run_posthoc)
                  if isinstance(node, ast.Call)}
        self.assertIn("validate_posthoc_batch_receipt", called)


if __name__ == "__main__":
    unittest.main()
