import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from safety.v16_b716_candidate_plan import (
    atomic_json, sha256_file, stable_json_sha256,
)
from safety.v16_b716_geot_backfill import (
    AUTH_SCHEMA, CLEAN_SCHEMA, TASK_SCHEMA, BackfillError,
    authorization_derivation_contract, build_attempt_receipt,
    compare_exact_missing, derive_authorized_task_view,
    enforce_cuda_clean_gate, enforce_cuda_runtime_gate,
    expected_missing_rows, future_merge_contract,
    merge_completed_geot_ledger, normalise_result,
    query_cuda_snapshot, validate_external_authorities, validate_preregister,
    revalidate_runtime_registration_points, validate_authorization,
    validate_resumed_result,
)
from safety.v16_matched_region_colorpcr import array_sha256 as surface_sha256


ROOT = Path(__file__).resolve().parents[1]


class TestV16B716GeoTBackfill(unittest.TestCase):
    def setUp(self):
        self.prereg = json.loads((
            ROOT / "manifests/v16_b716_geot_backfill_preregister.json"
        ).read_text())

    def execution_fixture(self, root: Path):
        task = {
            "schema": TASK_SCHEMA, "state": "planned_disabled",
            "execution_authorized": False,
            "execution_transition_contract": authorization_derivation_contract(),
            "short_id": "p", "pair_id": "a_to_b",
            "candidate_index": 0, "node_pair": [0, 1],
            "object_pair": [10, 20],
            "source_surface": {}, "reference_surface": {},
            "candidate_plan_sha256": "9" * 64,
            "official_release_checkpoint_sha256": "8" * 64,
            "official_geotransformer_checkpoint_sha256": "7" * 64,
            "canonical_boundary_sha256": "6" * 64,
            "canonical_surface_binding_sha256": "5" * 64,
            "forbidden_inputs": ["GT", "combos", "official92"],
        }
        task["task_sha256"] = stable_json_sha256(task)
        enabled = json.loads(json.dumps(self.prereg))
        enabled["disabled"] = False
        enabled["execution_contract"]["real_execution_allowed"] = True
        binding = {
            "authorization_sha256": "a" * 64,
            "preregister_sha256": "b" * 64,
            "preflight_manifest_sha256": "c" * 64,
            "preflight_payload_sha256": "d" * 64,
            "recursive_source_closure_sha256": "e" * 64,
            "recursive_artifact_closure_sha256": "f" * 64,
            "task_closure_sha256": "1" * 64,
            "immutable_runtime_source_bundle_sha256": "2" * 64,
            "runtime_module_entrypoint_closure_sha256": "3" * 64,
            "cuda_device_uuid": "GPU-abc",
        }
        view = derive_authorized_task_view(task, binding, enabled)
        view_path = root / "authorized_task_view.json"
        atomic_json(view_path, view)
        view_sha256 = sha256_file(view_path)
        snapshot = {
            "uuid": "GPU-abc", "index": 0, "memory_used_mib": 20,
            "utilization_percent": 0, "compute_processes": [],
        }
        receipt = build_attempt_receipt(
            task, view_sha256, binding, snapshot)
        attempt_path = root / "attempt_receipt.json"
        atomic_json(attempt_path, receipt)
        return task, binding, view_sha256, attempt_path

    def test_preregister_freezes_exact_72_ordered_keys(self):
        rows = validate_preregister(self.prereg)
        self.assertEqual(len(rows), 72)
        self.assertEqual(rows[:2], [
            {"short_id": "09582205_1883", "node_pair": [29, 66]},
            {"short_id": "09582205_1883", "node_pair": [5, 70]},
        ])
        self.assertEqual(len({(x["short_id"], *x["node_pair"]) for x in rows}), 72)
        changed = json.loads(json.dumps(self.prereg))
        changed["cuda_hard_gate"]["runtime_max_utilization_percent"] = 99
        with self.assertRaisesRegex(BackfillError, "preregistration contract"):
            validate_preregister(changed)

    def test_disabled_task_requires_explicit_enabled_derivation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, _view_sha, _attempt = self.execution_fixture(root)
            with self.assertRaisesRegex(BackfillError, "cannot derive"):
                derive_authorized_task_view(task, binding, self.prereg)
            enabled = json.loads(json.dumps(self.prereg))
            enabled["disabled"] = False
            enabled["execution_contract"]["real_execution_allowed"] = True
            view = derive_authorized_task_view(task, binding, enabled)
            self.assertTrue(view["execution_authorized"])
            self.assertEqual(task["state"], "planned_disabled")
            self.assertFalse(task["execution_authorized"])

    def test_exact_batch_rejects_reorder_subset_and_extra(self):
        rows = expected_missing_rows(self.prereg)
        observed = [{**row, "ignored": True} for row in rows]
        compare_exact_missing(rows, observed)
        for changed in (observed[::-1], observed[:-1], observed + [observed[0]]):
            with self.assertRaisesRegex(BackfillError, "exact 72"):
                compare_exact_missing(rows, changed)

    def test_deterministic_119_plus_72_merger_covers_exact_191(self):
        expected = [
            {"short_id": "fixed", "candidate_index": index,
             "node_pair": [index, index + 1]}
            for index in range(191)]
        contract = {
            **future_merge_contract(),
            "expected_candidate_keys": expected,
            "expected_candidate_key_closure_sha256": stable_json_sha256(expected),
        }
        existing = [{
            **row, "origin": "official_pair_cache", "immutable": True,
            "entry_sha256": "a" * 64,
        } for row in expected[:119]]
        backfill = [{
            **row, "origin": "official_geotransformer_backfill",
            "selector_eligible": False, "status": "ok",
            "task_sha256": "b" * 64,
            "attempt_receipt_sha256": "c" * 64,
            "result_sha256": "d" * 64,
        } for row in expected[119:]]
        merged = merge_completed_geot_ledger(
            contract, existing[::-1], backfill[::-1])
        self.assertEqual(
            [(row["candidate_index"], row["origin"]) for row in merged],
            [(index, "official_pair_cache" if index < 119
              else "official_geotransformer_backfill")
             for index in range(191)])
        with self.assertRaisesRegex(BackfillError, "count mismatch"):
            merge_completed_geot_ledger(contract, existing, backfill[:-1])

    def test_external_v3_and_v13_authorities_bind_full_fixed4(self):
        candidate_path = ROOT / self.prereg["candidate_manifest_path"]
        candidate = json.loads(candidate_path.read_text())
        sources, summary = validate_external_authorities(
            self.prereg, candidate, candidate_path.parent)
        self.assertEqual(summary["fixed4_artifact_file_count"], 16)
        self.assertEqual(
            summary["selection89_readonly_audit"],
            self.prereg["selection89_readonly_audit"])
        self.assertEqual(
            summary["formal_fixed4_full_pair_ids"],
            [row["pair_id"] for row in candidate["pairs"]])
        roles = {row["role"] for row in sources}
        self.assertIn("authoritative_v3_artifact_manifest", roles)
        self.assertIn("authoritative_v3_cache_manifest", roles)
        self.assertIn("formal_v13_fixed4_preregister", roles)
        self.assertEqual(
            len([role for role in roles
                 if role.startswith("authoritative_v3_fixed4:")]), 16)

    def test_selection89_digest_tamper_fails_closed(self):
        candidate_path = ROOT / self.prereg["candidate_manifest_path"]
        candidate = json.loads(candidate_path.read_text())
        changed = json.loads(json.dumps(self.prereg))
        changed["selection89_readonly_audit"]["stable_rows_sha256"] = "0" * 64
        with self.assertRaisesRegex(BackfillError, "audit digest"):
            validate_external_authorities(changed, candidate, candidate_path.parent)

    def test_preregister_declares_forbidden_fields_fail_closed(self):
        policy = self.prereg["forbidden_field_declarations"]
        self.assertEqual(
            policy["pair_cache_top_level_fields_lexically_skipped"], ["combos"])
        self.assertEqual(policy["nested_fields_never_decoded"], ["node_metrics"])
        for field in (
                "canonical_builder_with_labels", "gt_or_label_inputs_allowed",
                "posthoc_inputs_allowed", "official92_inputs_allowed",
                "result_or_outcome_selection_allowed"):
            self.assertFalse(policy[field])

    def test_cuda_snapshot_requires_one_numeric_visible_device(self):
        outputs = iter([
            "0, GPU-abc, 100, 0\n1, GPU-def, 20, 1\n", "",
        ])

        def fake_run(*_args, **_kwargs):
            return subprocess.CompletedProcess([], 0, stdout=next(outputs), stderr="")

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            row = query_cuda_snapshot(fake_run)
        self.assertEqual(row["uuid"], "GPU-def")
        self.assertEqual(row["compute_processes"], [])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False):
            with self.assertRaisesRegex(BackfillError, "exactly one"):
                query_cuda_snapshot(fake_run)

    def test_cuda_gate_requires_idle_clean_receipt_and_sentinels(self):
        with tempfile.TemporaryDirectory() as directory:
            clean_path = Path(directory) / "clean.json"
            checked = datetime.now(timezone.utc)
            clean_path.write_text(json.dumps({
                "schema": CLEAN_SCHEMA, "clean": True,
                "cuda_device_uuid": "GPU-abc",
                "checked_utc": checked.isoformat(),
                "expires_utc": (checked + timedelta(seconds=120)).isoformat(),
                "compute_process_count": 0,
                "services_checked": self.prereg["cuda_hard_gate"][
                    "required_services_checked"],
            }))
            auth = {
                "schema": AUTH_SCHEMA, "cuda_device_uuid": "GPU-abc",
                "clean_service_receipt_path": str(clean_path),
                "clean_service_receipt_sha256": sha256_file(clean_path),
            }
            snapshot = {"uuid": "GPU-abc", "memory_used_mib": 20,
                        "utilization_percent": 0, "compute_processes": []}
            env = {"V16_B716_CLEAN_SERVICE": "1",
                   "V16_B716_ISOLATED_GPU": "1"}
            with mock.patch.dict(os.environ, env, clear=False):
                enforce_cuda_clean_gate(snapshot, auth, self.prereg)
                busy = {**snapshot, "compute_processes": ["GPU-abc, 1, python, 2"]}
                with self.assertRaisesRegex(BackfillError, "not isolated"):
                    enforce_cuda_clean_gate(busy, auth, self.prereg)
                expired = checked - timedelta(seconds=240)
                clean_path.write_text(json.dumps({
                    "schema": CLEAN_SCHEMA, "clean": True,
                    "cuda_device_uuid": "GPU-abc",
                    "checked_utc": expired.isoformat(),
                    "expires_utc": (expired + timedelta(seconds=120)).isoformat(),
                    "compute_process_count": 0,
                    "services_checked": self.prereg["cuda_hard_gate"][
                        "required_services_checked"],
                }))
                expired_auth = {
                    **auth, "clean_service_receipt_sha256": sha256_file(clean_path)}
                with self.assertRaisesRegex(BackfillError, "receipt contract"):
                    enforce_cuda_clean_gate(snapshot, expired_auth, self.prereg)
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(BackfillError, "sentinels"):
                    enforce_cuda_clean_gate(snapshot, auth, self.prereg)

    def test_runtime_recheck_allows_only_the_current_runner_process(self):
        pid = 4242
        auth = {"cuda_device_uuid": "GPU-abc"}
        own = {
            "uuid": "GPU-abc", "memory_used_mib": 2048,
            "utilization_percent": 95,
            "compute_processes": [{
                "gpu_uuid": "GPU-abc", "pid": pid,
                "process_name": "python", "used_gpu_memory_mib": 2000,
            }],
        }
        env = {"V16_B716_CLEAN_SERVICE": "1",
               "V16_B716_ISOLATED_GPU": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            enforce_cuda_runtime_gate(
                own, auth, self.prereg, current_pid=pid)
            foreign = json.loads(json.dumps(own))
            foreign["compute_processes"][0]["pid"] = pid + 1
            with self.assertRaisesRegex(BackfillError, "runtime isolation"):
                enforce_cuda_runtime_gate(
                    foreign, auth, self.prereg, current_pid=pid)

    def test_fresh_registration_points_are_rebound_immediately_before_geot(self):
        src = np.arange(12, dtype=np.float64).reshape(4, 3)
        ref = np.arange(15, dtype=np.float64).reshape(5, 3)
        src_row = {
            "node_index": 0, "object_id": 10,
            "canonical_registration_points": 4,
            "canonical_registration_surface_sha256": surface_sha256(src),
            "raw_inseg_path": "/raw/src.npz", "raw_inseg_sha256": "a" * 64,
        }
        ref_row = {
            "node_index": 1, "object_id": 20,
            "canonical_registration_points": 5,
            "canonical_registration_surface_sha256": surface_sha256(ref),
            "raw_inseg_path": "/raw/ref.npz", "raw_inseg_sha256": "b" * 64,
        }
        task = {
            "node_pair": [0, 1], "source_surface": src_row,
            "reference_surface": ref_row,
        }
        data = {"registration_pts": {0: src, 1: ref}}
        observed = revalidate_runtime_registration_points(
            task, data, [src_row, ref_row], array_fingerprint=surface_sha256)
        self.assertTrue(np.array_equal(observed[0], src))
        changed = {"registration_pts": {0: src.copy(), 1: ref}}
        changed["registration_pts"][0][0, 0] += 1
        with self.assertRaisesRegex(BackfillError, "array changed"):
            revalidate_runtime_registration_points(
                task, changed, [src_row, ref_row],
                array_fingerprint=surface_sha256)
        wrong_raw = [{**src_row, "raw_inseg_sha256": "c" * 64}, ref_row]
        with self.assertRaisesRegex(BackfillError, "raw/object/surface"):
            revalidate_runtime_registration_points(
                task, data, wrong_raw, array_fingerprint=surface_sha256)

    def test_authorization_expiry_and_preregister_preflight_bindings(self):
        enabled = json.loads(json.dumps(self.prereg))
        enabled["disabled"] = False
        enabled["execution_contract"]["real_execution_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "schema": AUTH_SCHEMA, "authorized": True,
                "expires_utc": (
                    datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
                "candidate_manifest_sha256": "1" * 64,
                "missing_key_closure_sha256": "2" * 64,
                "preregister_sha256": "3" * 64,
                "preflight_manifest_sha256": "4" * 64,
                "preflight_payload_sha256": "6" * 64,
                "recursive_source_closure_sha256": "7" * 64,
                "recursive_artifact_closure_sha256": "8" * 64,
                "task_closure_sha256": "9" * 64,
                "immutable_runtime_source_bundle_sha256": "a" * 64,
                "runtime_module_entrypoint_closure_sha256": "b" * 64,
                "exact_batch_count": 72, "key_selection_allowed": False,
                "result_selection_allowed": False, "gt_allowed": False,
                "official92_allowed": False, "output_root": str(root),
                "cuda_device_uuid": "GPU-abc",
                "clean_service_receipt_path": str(root / "clean.json"),
                "clean_service_receipt_sha256": "5" * 64,
            }
            path = root / "authorization.json"
            path.write_text(json.dumps(values))
            validate_authorization(
                path, sha256_file(path), enabled,
                candidate_manifest_sha256="1" * 64,
                missing_closure_sha256="2" * 64,
                preregister_sha256="3" * 64,
                preflight_manifest_sha256="4" * 64,
                preflight_payload_sha256="6" * 64,
                recursive_source_closure_sha256="7" * 64,
                recursive_artifact_closure_sha256="8" * 64,
                task_closure_sha256="9" * 64,
                immutable_runtime_source_bundle_sha256="a" * 64,
                runtime_module_entrypoint_closure_sha256="b" * 64,
                output_root=root)
            with self.assertRaisesRegex(BackfillError, "scope/expiry"):
                validate_authorization(
                    path, sha256_file(path), enabled,
                    candidate_manifest_sha256="1" * 64,
                    missing_closure_sha256="2" * 64,
                    preregister_sha256="6" * 64,
                    preflight_manifest_sha256="4" * 64,
                    preflight_payload_sha256="6" * 64,
                    recursive_source_closure_sha256="7" * 64,
                    recursive_artifact_closure_sha256="8" * 64,
                    task_closure_sha256="9" * 64,
                    immutable_runtime_source_bundle_sha256="a" * 64,
                    runtime_module_entrypoint_closure_sha256="b" * 64,
                    output_root=root)
            expired = {
                **values,
                "expires_utc": (
                    datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
            }
            expired_path = root / "expired_authorization.json"
            expired_path.write_text(json.dumps(expired))
            with self.assertRaisesRegex(BackfillError, "scope/expiry"):
                validate_authorization(
                    expired_path, sha256_file(expired_path), enabled,
                    candidate_manifest_sha256="1" * 64,
                    missing_closure_sha256="2" * 64,
                    preregister_sha256="3" * 64,
                    preflight_manifest_sha256="4" * 64,
                    preflight_payload_sha256="6" * 64,
                    recursive_source_closure_sha256="7" * 64,
                    recursive_artifact_closure_sha256="8" * 64,
                    task_closure_sha256="9" * 64,
                    immutable_runtime_source_bundle_sha256="a" * 64,
                    runtime_module_entrypoint_closure_sha256="b" * 64,
                    output_root=root)

    def test_per_key_result_is_atomic_hash_bound_and_resumable(self):
        value = {
            "src_corr_points": np.ones((4, 3), np.float32),
            "ref_corr_points": np.ones((4, 3), np.float32) * 2,
            "corr_scores": np.arange(4, dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, view_sha, attempt = self.execution_fixture(root)
            result = normalise_result(
                "ok", value, task, root, execution_binding=binding,
                attempt_receipt_sha256=sha256_file(attempt),
                authorized_task_view_sha256=view_sha)
            path = root / "result.json"
            atomic_json(path, result)
            resumed = validate_resumed_result(
                path, task, attempt_receipt_path=attempt,
                execution_binding=binding,
                authorized_task_view_sha256=view_sha)
            self.assertEqual(resumed["status"], "ok")
            self.assertEqual(
                resumed["attempt_receipt_sha256"], sha256_file(attempt))
            (root / "correspondences.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(BackfillError, "artifact mismatch"):
                validate_resumed_result(
                    path, task, attempt_receipt_path=attempt,
                    execution_binding=binding,
                    authorized_task_view_sha256=view_sha)

    def test_resume_rejects_missing_or_tampered_attempt_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, view_sha, attempt = self.execution_fixture(root)
            result = normalise_result(
                "geotransformer_runtime_error", {"reason": "typed"}, task,
                root, execution_binding=binding,
                attempt_receipt_sha256=sha256_file(attempt),
                authorized_task_view_sha256=view_sha)
            result_path = root / "result.json"
            atomic_json(result_path, result)
            attempt_bytes = attempt.read_bytes()
            attempt.unlink()
            with self.assertRaisesRegex(BackfillError, "lacks its attempt"):
                validate_resumed_result(
                    result_path, task, attempt_receipt_path=attempt,
                    execution_binding=binding,
                    authorized_task_view_sha256=view_sha)
            attempt.write_bytes(attempt_bytes)
            receipt = json.loads(attempt.read_text())
            receipt["cuda_snapshot"]["memory_used_mib"] += 1
            receipt["payload_sha256"] = stable_json_sha256({
                key: value for key, value in receipt.items()
                if key != "payload_sha256"})
            attempt.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(BackfillError, "binding/contract"):
                validate_resumed_result(
                    result_path, task, attempt_receipt_path=attempt,
                    execution_binding=binding,
                    authorized_task_view_sha256=view_sha)

    def test_resume_rejects_wrong_auth_preregister_preflight_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, view_sha, attempt = self.execution_fixture(root)
            result = normalise_result(
                "geotransformer_runtime_error", {"reason": "typed"}, task,
                root, execution_binding=binding,
                attempt_receipt_sha256=sha256_file(attempt),
                authorized_task_view_sha256=view_sha)
            result_path = root / "result.json"
            atomic_json(result_path, result)
            changes = {
                "authorization_sha256": "d" * 64,
                "preregister_sha256": "e" * 64,
                "preflight_manifest_sha256": "f" * 64,
                "runtime_module_entrypoint_closure_sha256": "0" * 64,
                "cuda_device_uuid": "GPU-other",
            }
            for field, changed_value in changes.items():
                with self.subTest(field=field):
                    changed = {**binding, field: changed_value}
                    with self.assertRaisesRegex(BackfillError, "binding/contract"):
                        validate_resumed_result(
                            result_path, task, attempt_receipt_path=attempt,
                            execution_binding=changed,
                            authorized_task_view_sha256=view_sha)

    def test_resume_rejects_extra_npz_fields_and_unknown_failure_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, view_sha, attempt = self.execution_fixture(root)
            values = {
                "src_corr_points": np.ones((3, 3), np.float32),
                "ref_corr_points": np.ones((3, 3), np.float32) * 2,
                "corr_scores": np.ones(3, np.float32),
            }
            result = normalise_result(
                "ok", values, task, root, execution_binding=binding,
                attempt_receipt_sha256=sha256_file(attempt),
                authorized_task_view_sha256=view_sha)
            npz = root / "correspondences.npz"
            np.savez(
                npz, src_corr=values["src_corr_points"],
                ref_corr=values["ref_corr_points"],
                scores=values["corr_scores"], extra=np.ones(1, np.float32))
            result["correspondences"]["bytes"] = npz.stat().st_size
            result["correspondences"]["sha256"] = sha256_file(npz)
            result["payload_sha256"] = stable_json_sha256({
                key: value for key, value in result.items()
                if key != "payload_sha256"})
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(BackfillError, "NPZ field set"):
                validate_resumed_result(
                    result_path, task, attempt_receipt_path=attempt,
                    execution_binding=binding,
                    authorized_task_view_sha256=view_sha)
            with self.assertRaisesRegex(BackfillError, "not whitelisted"):
                normalise_result(
                    "lucky_unknown_status", {"reason": "not allowed"}, task,
                    root, execution_binding=binding,
                    attempt_receipt_sha256=sha256_file(attempt),
                    authorized_task_view_sha256=view_sha)

    def test_geot_output_cannot_introduce_forbidden_selection_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, binding, view_sha, attempt = self.execution_fixture(root)
            with self.assertRaisesRegex(BackfillError, "forbidden GeoT"):
                normalise_result(
                    "geotransformer_runtime_error",
                    {"reason": "blocked", "outcome": "never consume"},
                    task, root, execution_binding=binding,
                    attempt_receipt_sha256=sha256_file(attempt),
                    authorized_task_view_sha256=view_sha)

    def test_execution_remains_disabled_and_no_key_selector_exists(self):
        self.assertFalse(self.prereg["execution_contract"]["real_execution_allowed"])
        source = (ROOT / "scripts/v16_b716_geot_backfill.py").read_text()
        self.assertNotIn('add_argument("--pair-id"', source)
        self.assertNotIn('add_argument("--key"', source)
        self.assertNotIn("load_gt_transform", source)
        self.assertNotIn("load_anchor_ids", source)
        self.assertIn('"attempt_receipt_sha256":', source)
        self.assertIn('"attempt_receipt_closure_sha256":', source)
        self.assertIn("validate_clean_service_receipt", source)
        # Exact module paths are imported only inside the post-gate block.
        self.assertIn("Delayed exact-path import", source)
        self.assertIn("validate_runtime_module_resolution(preflight)", source)
        self.assertIn("runtime_module_entrypoint_closure_sha256", source)

    def test_frozen_preflight_binds_exact_runtime_module_entrypoints(self):
        path = (ROOT / "outputs/v16_b716_geot_backfill_preflight_20260830"
                / "preflight_manifest.json")
        preflight = json.loads(path.read_text())
        rows = preflight["runtime_module_entrypoints"]
        self.assertEqual(
            {row["module"] for row in rows},
            {
                "v16_b716_frozen_inference", "GeoTransformer.config",
                "GeoTransformer.model",
                "GeoTransformer.geotransformer.utils.data",
                "engine.registration_evaluator", "utils.torch_util",
            })
        self.assertEqual(
            stable_json_sha256(rows),
            preflight["runtime_module_entrypoint_closure_sha256"])
        bundle = {
            (row["path"], row["bytes"], row["sha256"])
            for row in preflight["runtime_source_bundle"]
        }
        self.assertTrue(all(
            (row["path"], row["bytes"], row["sha256"]) in bundle
            for row in rows))


if __name__ == "__main__":
    unittest.main()
