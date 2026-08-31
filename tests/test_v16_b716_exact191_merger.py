from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from safety.v16_b716_candidate_plan import (
    array_sha256, atomic_json, sha256_file, stable_json_sha256,
)
from safety.v16_b716_exact191_merger import (
    ATTEMPT_SCHEMA, BATCH_SCHEMA, EXPECTED_CANDIDATES,
    EXPECTED_EXISTING, EXPECTED_HYPOTHESES, EXPECTED_MISSING,
    EXPECTED_ORDERED_KEY_CLOSURE_SHA256, Exact191Error, _load_npz,
    _candidate_repository_root, _expected_missing_in_preregistered_order,
    _ordered_task_id_closure_sha256, _validate_corr_arrays, merge_exact191,
)
from safety.v16_b716_geot_backfill import (
    AUDITED_BASE_COMMIT, AUTH_SCHEMA, BASELINE_SERVICE_SCHEMA,
    INDEPENDENT_AUDIT_SCHEMA, BackfillError, build_attempt_receipt,
    derive_authorized_task_view, enabled_cuda_hard_gate_contract,
    normalise_result, repository_state,
)


ROOT = Path(__file__).resolve().parents[1]


class Exact191Fixture:
    def __init__(self, root: Path, *, future_contract_mutator=None):
        self.root = root
        self.preflight_root = root / "execution"
        self.output_root = root / "merged"
        base_preflight_path = (
            ROOT / "outputs/v16_b716_geot_backfill_preflight_20260830/"
            "preflight_manifest.json")
        base_preflight = json.loads(base_preflight_path.read_text())
        candidate = (
            ROOT / "outputs/v16_b716_candidate_plan_fixed4_20260830/"
            "fixed4_manifest.json")
        self.candidate_path = candidate
        self.candidate_sha = sha256_file(candidate)
        self.execution_repo_root = _candidate_repository_root(candidate)
        self.ordered72_sha = _ordered_task_id_closure_sha256(base_preflight)

        base_prereg_path = (
            ROOT / "manifests/v16_b716_geot_backfill_preregister.json")
        prereg = json.loads(base_prereg_path.read_text())
        prereg["disabled"] = False
        prereg["execution_contract"]["real_execution_allowed"] = True
        baseline = {
            "schema": BASELINE_SERVICE_SCHEMA,
            "pid": 2849,
            "executable_path": "/usr/libexec/gnome-remote-desktop-daemon",
            "executable_sha256": "a" * 64,
            "proc_starttime_ticks": 14545,
            "cmdline": ["/usr/libexec/gnome-remote-desktop-daemon"],
            "cmdline_sha256": "b" * 64,
        }
        baseline["identity_sha256"] = stable_json_sha256(baseline)
        prereg["cuda_hard_gate"] = enabled_cuda_hard_gate_contract(baseline)
        repo = repository_state(self.execution_repo_root)
        prereg["enable_scope"] = {
            "schema": "v16-b716-geot-enabled-preregister-v1",
            "audited_base_commit": AUDITED_BASE_COMMIT,
            "base_preregister_sha256": sha256_file(base_prereg_path),
            "base_preflight_manifest_sha256": sha256_file(base_preflight_path),
            "base_preflight_payload_sha256": base_preflight["payload_sha256"],
            "base_recursive_source_closure_sha256": base_preflight[
                "recursive_source_closure_sha256"],
            "base_recursive_artifact_closure_sha256": base_preflight[
                "recursive_artifact_closure_sha256"],
            "base_task_closure_sha256": base_preflight["task_closure_sha256"],
            "base_runtime_source_bundle_sha256": base_preflight[
                "immutable_runtime_source_bundle_sha256"],
            "base_runtime_module_entrypoint_closure_sha256": base_preflight[
                "runtime_module_entrypoint_closure_sha256"],
            "expected_short_id_order": list(
                prereg["expected_missing_node_pairs_by_short_id"]),
            "exact_batch_count": 72,
            "ordered72_sha256": self.ordered72_sha,
            "future_merge_contract_sha256": stable_json_sha256(
                base_preflight["future_merge_contract"]),
            "output_root": str(self.preflight_root.resolve()),
            "cuda_device_uuid": "GPU-test-exact191",
            "baseline_service_identity": baseline,
            "runner_executable": {
                "path": "/usr/bin/python3", "sha256": "c" * 64},
            "repository_head": repo["head"],
            "repository_tree": repo["tree"],
            "repository_clean": repo["clean"],
            "repository_status_sha256": repo["status_sha256"],
            "selected": False,
            "result_selection_allowed": False,
            "gt_allowed": False,
            "official92_allowed": False,
            "downstream_authorized": False,
            "independent_audit_required": True,
        }
        self.prereg_path = root / "authorized_preregister.json"
        # Production receipts use canonical sorted JSON.  The explicit order
        # array, never object-key insertion order, defines exact72 semantics.
        self.prereg_path.write_text(json.dumps(
            prereg, indent=2, sort_keys=True) + "\n")
        self.prereg_sha = sha256_file(self.prereg_path)

        tasks = []
        for row in base_preflight["tasks"]:
            source = base_preflight_path.parent / row["path"]
            target = self.preflight_root / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            tasks.append(copy.deepcopy(row))
        preflight = copy.deepcopy(base_preflight)
        preflight["candidate_manifest_path"] = str(candidate)
        preflight["disabled"] = False
        for row in preflight["source_closure"]:
            if row.get("role") == "frozen_backfill_preregistration":
                row.update({
                    "path": str(self.prereg_path),
                    "bytes": self.prereg_path.stat().st_size,
                    "sha256": self.prereg_sha,
                })
            elif str(row.get("role", "")).startswith("source:"):
                local = ROOT / str(row["role"])[len("source:"):]
                if (local.is_file() and local.stat().st_size == row["bytes"]
                        and sha256_file(local) == row["sha256"]):
                    row["path"] = str(local)
        preflight["recursive_source_closure_sha256"] = stable_json_sha256(
            preflight["source_closure"])
        if future_contract_mutator is not None:
            future_contract_mutator(preflight["future_merge_contract"])
        preflight.pop("payload_sha256", None)
        preflight["payload_sha256"] = stable_json_sha256(preflight)
        self.preflight_path = self.preflight_root / "preflight_manifest.json"
        atomic_json(self.preflight_path, preflight)
        self.preflight_sha = sha256_file(self.preflight_path)

        audit = {
            "schema": INDEPENDENT_AUDIT_SCHEMA,
            "status": "pass",
            "execution_authorized_after_audit": True,
            "preregister_sha256": self.prereg_sha,
            "preflight_manifest_sha256": self.preflight_sha,
            "ordered72_sha256": self.ordered72_sha,
            "repository_head": repo["head"],
            "repository_tree": repo["tree"],
            "output_root": str(self.preflight_root.resolve()),
            "auditor": "exact191-test-independent-auditor",
        }
        audit["payload_sha256"] = stable_json_sha256(audit)
        audit_path = self.preflight_root / "independent_audit_receipt.json"
        atomic_json(audit_path, audit)
        audit_sha = sha256_file(audit_path)

        auth = {
            "schema": AUTH_SCHEMA, "authorized": True,
            "audited_base_commit": AUDITED_BASE_COMMIT,
            "candidate_manifest_sha256": self.candidate_sha,
            "missing_key_closure_sha256": preflight["missing_key_closure_sha256"],
            "preregister_sha256": self.prereg_sha,
            "preflight_manifest_sha256": self.preflight_sha,
            "preflight_payload_sha256": preflight["payload_sha256"],
            "recursive_source_closure_sha256": preflight[
                "recursive_source_closure_sha256"],
            "recursive_artifact_closure_sha256": preflight[
                "recursive_artifact_closure_sha256"],
            "task_closure_sha256": preflight["task_closure_sha256"],
            "immutable_runtime_source_bundle_sha256": preflight[
                "immutable_runtime_source_bundle_sha256"],
            "runtime_module_entrypoint_closure_sha256": preflight[
                "runtime_module_entrypoint_closure_sha256"],
            "future_merge_contract_sha256": stable_json_sha256(
                preflight["future_merge_contract"]),
            "ordered72_sha256": self.ordered72_sha,
            "exact_batch_count": 72, "key_selection_allowed": False,
            "result_selection_allowed": False, "gt_allowed": False,
            "official92_allowed": False,
            "output_root": str(self.preflight_root),
            "expires_utc": (datetime.now(timezone.utc)
                            + timedelta(hours=2)).isoformat(),
            "cuda_device_uuid": "GPU-test-exact191",
            "clean_service_receipt_path": str(root / "unused-clean.json"),
            "clean_service_receipt_sha256": "0" * 64,
            "independent_audit_receipt_path": str(audit_path.resolve()),
            "independent_audit_receipt_sha256": audit_sha,
            "repository_head": repo["head"],
            "repository_tree": repo["tree"],
            "repository_status_sha256": repo["status_sha256"],
            "baseline_service_identity_sha256": baseline["identity_sha256"],
            "runner_executable": prereg["enable_scope"]["runner_executable"],
            "downstream_authorized": False,
            "selected": False,
        }
        auth["payload_sha256"] = stable_json_sha256(auth)
        self.authorization_path = root / "authorization.json"
        atomic_json(self.authorization_path, auth)
        self.authorization_sha = sha256_file(self.authorization_path)
        self.execution_binding = {
            "authorization_sha256": self.authorization_sha,
            "preregister_sha256": self.prereg_sha,
            "preflight_manifest_sha256": self.preflight_sha,
            "preflight_payload_sha256": preflight["payload_sha256"],
            "recursive_source_closure_sha256": preflight[
                "recursive_source_closure_sha256"],
            "recursive_artifact_closure_sha256": preflight[
                "recursive_artifact_closure_sha256"],
            "task_closure_sha256": preflight["task_closure_sha256"],
            "immutable_runtime_source_bundle_sha256": preflight[
                "immutable_runtime_source_bundle_sha256"],
            "runtime_module_entrypoint_closure_sha256": preflight[
                "runtime_module_entrypoint_closure_sha256"],
            "cuda_device_uuid": "GPU-test-exact191",
        }

        batch_rows = []
        for row in tasks:
            task_id = row["task_id"]
            directory = self.preflight_root / "tasks" / task_id
            task = json.loads((directory / "task.json").read_text())
            authorized_view = derive_authorized_task_view(
                task, self.execution_binding, prereg)
            view_path = directory / "authorized_task_view.json"
            atomic_json(view_path, authorized_view)
            view_sha = sha256_file(view_path)
            snapshot = {
                "uuid": "GPU-test-exact191", "index": 0,
                "memory_used_mib": 12, "utilization_percent": 0,
                "compute_processes": [],
            }
            attempt = build_attempt_receipt(
                task, view_sha, self.execution_binding, snapshot)
            atomic_json(directory / "attempt_receipt.json", attempt)
            attempt_sha = sha256_file(directory / "attempt_receipt.json")
            offset = float(task["candidate_index"] + 1)
            values = {
                "src_corr_points": np.array(
                    [[offset, 0, 0], [offset, 1, 0], [offset, 0, 1]],
                    dtype=np.float32),
                "ref_corr_points": np.array(
                    [[offset, 0, 0.1], [offset, 1, 0.1], [offset, 0, 1.1]],
                    dtype=np.float32),
                "corr_scores": np.array([0.9, 0.8, 0.7], dtype=np.float32),
            }
            result = normalise_result(
                "ok", values, task, directory,
                execution_binding=self.execution_binding,
                attempt_receipt_sha256=attempt_sha,
                authorized_task_view_sha256=view_sha)
            atomic_json(directory / "result.json", result)
            batch_rows.append({
                "task_id": task_id, "status": "ok", "resumed": False,
                "attempt_receipt_sha256": attempt_sha,
                "result_sha256": sha256_file(directory / "result.json"),
            })
        batch = {
            "schema": BATCH_SCHEMA, "exact_batch_count": 72,
            "selector_eligible": False,
            "result_based_selection_allowed": False,
            "results": batch_rows,
            "execution_binding": self.execution_binding,
            "attempt_receipt_closure_sha256": stable_json_sha256([
                {"task_id": row["task_id"],
                 "attempt_receipt_sha256": row["attempt_receipt_sha256"]}
                for row in batch_rows]),
        }
        batch["payload_sha256"] = stable_json_sha256(batch)
        self.batch_path = self.preflight_root / "batch_result.json"
        atomic_json(self.batch_path, batch)
        self.batch_sha = sha256_file(self.batch_path)

    def run(self, output_root: Path | None = None):
        return merge_exact191(
            candidate_path=self.candidate_path,
            candidate_sha256=self.candidate_sha,
            preflight_path=self.preflight_path,
            preflight_sha256=self.preflight_sha,
            preregister_path=self.prereg_path,
            preregister_sha256=self.prereg_sha,
            authorization_path=self.authorization_path,
            authorization_sha256=self.authorization_sha,
            batch_path=self.batch_path,
            batch_sha256=self.batch_sha,
            output_root=output_root or self.output_root,
        )

    def refresh_batch(self):
        batch = json.loads(self.batch_path.read_text())
        batch.pop("payload_sha256", None)
        batch["payload_sha256"] = stable_json_sha256(batch)
        self.batch_path.unlink()
        atomic_json(self.batch_path, batch)
        self.batch_sha = sha256_file(self.batch_path)


class TestV16B716Exact191Merger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Exact191Fixture(Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_01_positive_merge_is_exact_ordered_and_b716_only(self):
        manifest = self.fixture.run()
        self.assertEqual(manifest["candidate_count"], 191)
        self.assertEqual(manifest["existing_count"], 119)
        self.assertEqual(manifest["new_authorized_count"], 72)
        self.assertEqual(manifest["new_authorized_ok_count"], 72)
        self.assertEqual(manifest["new_authorized_typed_failure_count"], 0)
        self.assertEqual(manifest["hypothesis_count"], 34)
        self.assertEqual(manifest["typed_failure_existing_count"], 16)
        self.assertEqual(manifest["typed_failure_total_count"], 16)
        self.assertEqual(manifest["hypotheses_with_typed_failure_members"], 8)
        self.assertTrue(manifest["typed_failures_visible_and_never_filtered"])
        self.assertEqual(
            manifest["ordered_candidate_key_closure_sha256"],
            EXPECTED_ORDERED_KEY_CLOSURE_SHA256)
        self.assertEqual(manifest["fixed_hypothesis_distribution"], [12, 8, 2, 12])
        self.assertTrue(manifest["b716_domain_only"])
        self.assertFalse(manifest["legacy_B_ep20_or_89ed_consumed"])
        self.assertFalse(manifest["candidate_selection_allowed"])
        self.assertFalse(manifest["result_based_selection_allowed"])
        self.assertFalse(manifest["official92_allowed"])
        self.assertEqual(len(manifest["input_closure"]), 5)
        self.assertEqual(len(manifest["existing_entry_closure"]), 119)
        self.assertEqual(len(manifest["new_result_closure"]), 72)
        self.assertEqual(
            [row["candidate_count"] for row in manifest["pairs"]],
            list(EXPECTED_CANDIDATES))
        self.assertEqual(
            [row["existing_count"] for row in manifest["pairs"]],
            list(EXPECTED_EXISTING))
        self.assertEqual(
            [row["new_count"] for row in manifest["pairs"]],
            list(EXPECTED_MISSING))
        self.assertEqual(
            [row["hypothesis_count"] for row in manifest["pairs"]],
            list(EXPECTED_HYPOTHESES))

    def test_02_create_only_replay_is_byte_deterministic(self):
        before = sha256_file(self.fixture.output_root / "exact191_manifest.json")
        self.fixture.run()
        self.assertEqual(
            before, sha256_file(self.fixture.output_root / "exact191_manifest.json"))

    def test_03_frozen_allowlists_force_all_34_hypotheses(self):
        manifest = json.loads((
            self.fixture.output_root / "exact191_manifest.json").read_text())
        total = 0
        for pair in manifest["pairs"]:
            value = json.loads((self.fixture.output_root /
                                pair["allowlist_path"]).read_text())
            self.assertFalse(value["candidate_selection_allowed"])
            self.assertFalse(value["hypothesis_selection_allowed"])
            self.assertTrue(value["all_hypotheses_must_be_replayed"])
            self.assertTrue(
                value["typed_failure_members_visible_and_never_filtered"])
            total += value["hypothesis_count"]
            self.assertEqual(len(value["hypotheses"]), value["hypothesis_count"])
        self.assertEqual(total, 34)

    def test_04_reordered_batch_is_rejected(self):
        original = self.fixture.batch_path.read_bytes()
        try:
            value = json.loads(original)
            value["results"][0], value["results"][1] = (
                value["results"][1], value["results"][0])
            value.pop("payload_sha256")
            value["payload_sha256"] = stable_json_sha256(value)
            self.fixture.batch_path.write_text(json.dumps(value, indent=2,
                                                           sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(Exact191Error, "order/subset"):
                self.fixture.run(self.fixture.root / "bad-reorder")
        finally:
            self.fixture.batch_path.write_bytes(original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_05_failed_new_result_is_rejected(self):
        original = self.fixture.batch_path.read_bytes()
        batch = json.loads(self.fixture.batch_path.read_text())
        batch["results"][0]["status"] = "geotransformer_runtime_error"
        batch.pop("payload_sha256")
        batch["payload_sha256"] = stable_json_sha256(batch)
        try:
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(Exact191Error, "failed, foreign"):
                self.fixture.run(self.fixture.root / "bad-failed")
        finally:
            self.fixture.batch_path.write_bytes(original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_06_attempt_with_foreign_authorization_is_rejected(self):
        task_id = json.loads(self.fixture.batch_path.read_text())["results"][0]["task_id"]
        path = self.fixture.preflight_root / "tasks" / task_id / "attempt_receipt.json"
        original = path.read_bytes()
        try:
            value = json.loads(original)
            value["authorization_sha256"] = "f" * 64
            value.pop("payload_sha256")
            value["payload_sha256"] = stable_json_sha256(value)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(BackfillError, "binding/contract"):
                self.fixture.run(self.fixture.root / "bad-auth")
        finally:
            path.write_bytes(original)

    def test_07_tampered_correspondence_is_rejected(self):
        task_id = json.loads(self.fixture.batch_path.read_text())["results"][0]["task_id"]
        path = self.fixture.preflight_root / "tasks" / task_id / "correspondences.npz"
        original = path.read_bytes()
        try:
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(BackfillError, "artifact mismatch"):
                self.fixture.run(self.fixture.root / "bad-corr")
        finally:
            path.write_bytes(original)

    def test_08_foreign_result_identity_is_rejected(self):
        batch = json.loads(self.fixture.batch_path.read_text())
        task_id = batch["results"][0]["task_id"]
        path = self.fixture.preflight_root / "tasks" / task_id / "result.json"
        original_result = path.read_bytes()
        original_batch = self.fixture.batch_path.read_bytes()
        try:
            result = json.loads(original_result)
            result["pair_id"] = "foreign_to_pair"
            result.pop("payload_sha256")
            result["payload_sha256"] = stable_json_sha256(result)
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            batch["results"][0]["result_sha256"] = sha256_file(path)
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(Exact191Error, "failed, foreign"):
                self.fixture.run(self.fixture.root / "bad-foreign")
        finally:
            path.write_bytes(original_result)
            self.fixture.batch_path.write_bytes(original_batch)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_09_npz_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.npz"
            path.write_bytes(b"not-an-npz")
            with self.assertRaises(Exact191Error):
                _load_npz(path, sha256_file(path), path.stat().st_size)

    def test_10_cli_has_no_selector_gt_or_official92_surface(self):
        source = (ROOT / "scripts/v16_b716_exact191_merger.py").read_text()
        for option in ("--pair-id", "--key", "--selector", "--gt", "--official92"):
            self.assertNotIn(f'add_argument("{option}"', source)
        self.assertNotIn("geotransformer_forward", source)
        self.assertNotIn("official_matching", source)

    def test_11_wrong_attempt_preregister_sha_is_rejected_with_outer_chain_valid(self):
        batch_original = self.fixture.batch_path.read_bytes()
        batch = json.loads(batch_original)
        task_id = batch["results"][0]["task_id"]
        directory = self.fixture.preflight_root / "tasks" / task_id
        attempt_path = directory / "attempt_receipt.json"
        result_path = directory / "result.json"
        attempt_original = attempt_path.read_bytes()
        result_original = result_path.read_bytes()
        try:
            attempt = json.loads(attempt_original)
            attempt["preregister_sha256"] = "f" * 64
            attempt.pop("payload_sha256")
            attempt["payload_sha256"] = stable_json_sha256(attempt)
            attempt_path.write_text(json.dumps(
                attempt, indent=2, sort_keys=True) + "\n")
            attempt_sha = sha256_file(attempt_path)
            result = json.loads(result_original)
            result["attempt_receipt_sha256"] = attempt_sha
            result.pop("payload_sha256")
            result["payload_sha256"] = stable_json_sha256(result)
            result_path.write_text(json.dumps(
                result, indent=2, sort_keys=True) + "\n")
            batch["results"][0]["attempt_receipt_sha256"] = attempt_sha
            batch["results"][0]["result_sha256"] = sha256_file(result_path)
            batch["attempt_receipt_closure_sha256"] = stable_json_sha256([
                {"task_id": row["task_id"],
                 "attempt_receipt_sha256": row["attempt_receipt_sha256"]}
                for row in batch["results"]])
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(BackfillError, "binding/contract"):
                self.fixture.run(self.fixture.root / "bad-preregister-binding")
        finally:
            attempt_path.write_bytes(attempt_original)
            result_path.write_bytes(result_original)
            self.fixture.batch_path.write_bytes(batch_original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_12_batch_row_selected_true_is_rejected(self):
        original = self.fixture.batch_path.read_bytes()
        try:
            batch = json.loads(original)
            batch["results"][0]["selected"] = True
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(BackfillError, "forbidden.*selected"):
                self.fixture.run(self.fixture.root / "bad-selected")
        finally:
            self.fixture.batch_path.write_bytes(original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_13_wrong_src_corr_per_array_sha_is_rejected(self):
        batch_original = self.fixture.batch_path.read_bytes()
        batch = json.loads(batch_original)
        task_id = batch["results"][0]["task_id"]
        result_path = self.fixture.preflight_root / "tasks" / task_id / "result.json"
        result_original = result_path.read_bytes()
        try:
            result = json.loads(result_original)
            result["correspondences"]["arrays"]["src_corr"]["sha256"] = "0" * 64
            result.pop("payload_sha256")
            result["payload_sha256"] = stable_json_sha256(result)
            result_path.write_text(json.dumps(
                result, indent=2, sort_keys=True) + "\n")
            batch["results"][0]["result_sha256"] = sha256_file(result_path)
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(BackfillError, "per-array evidence"):
                self.fixture.run(self.fixture.root / "bad-array-sha")
        finally:
            result_path.write_bytes(result_original)
            self.fixture.batch_path.write_bytes(batch_original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_14_all_array_declarations_dtype_and_finite_are_checked(self):
        arrays = {
            "src_corr_new": np.array([[0, 0, 0]], dtype=np.float32),
            "ref_corr_new": np.array([[0, 0, 1]], dtype=np.float32),
            "scores_new": np.array([0.5], dtype=np.float32),
        }
        declared = {
            name: {
                "shape": list(arrays[f"{name}_new"].shape),
                "dtype": str(arrays[f"{name}_new"].dtype),
                "sha256": array_sha256(arrays[f"{name}_new"]),
            }
            for name in ("src_corr", "ref_corr", "scores")
        }
        _validate_corr_arrays(arrays, "new", declared)
        for name in declared:
            changed = copy.deepcopy(declared)
            changed[name]["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    Exact191Error, "array declaration mismatch"):
                _validate_corr_arrays(arrays, "new", changed)
        wrong_dtype = dict(arrays)
        wrong_dtype["src_corr_new"] = arrays["src_corr_new"].astype(np.float64)
        with self.assertRaisesRegex(Exact191Error, "arrays are malformed"):
            _validate_corr_arrays(wrong_dtype, "new", declared)
        nonfinite = dict(arrays)
        nonfinite["scores_new"] = np.array([np.nan], dtype=np.float32)
        with self.assertRaisesRegex(Exact191Error, "arrays are malformed"):
            _validate_corr_arrays(nonfinite, "new", declared)

    def test_15_self_signed_wrong_future_key_sha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Exact191Fixture(
                Path(directory), future_contract_mutator=lambda contract:
                contract.__setitem__(
                    "expected_candidate_key_closure_sha256", "0" * 64))
            with self.assertRaisesRegex(
                    Exact191Error, "future merge ordered-key contract"):
                fixture.run(Path(directory) / "bad-future-key")

    def test_16_self_signed_extra_batch_top_field_is_rejected(self):
        original = self.fixture.batch_path.read_bytes()
        try:
            batch = json.loads(original)
            batch["audit_note"] = "self-signed foreign top-level field"
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            with self.assertRaisesRegex(Exact191Error, "batch result contract"):
                self.fixture.run(self.fixture.root / "bad-batch-extra")
        finally:
            self.fixture.batch_path.write_bytes(original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)

    def test_17_implementation_manifest_closure_matches_current_sources(self):
        path = (
            ROOT / "outputs/v16_b716_exact191_order_fix_20260831/"
            "implementation_manifest.json")
        manifest = json.loads(path.read_text())
        self.assertEqual(
            manifest["status"], "ORDER_FIX_IMPLEMENTED_NOT_EXECUTED")
        self.assertFalse(manifest["execution_attempted"])
        self.assertFalse(manifest["gpu_used"])
        for row in manifest["implementation_closure"]:
            self.assertEqual(sha256_file(ROOT / row["path"]), row["sha256"])

    def test_18_sorted_json_mapping_uses_explicit_preregistered_order(self):
        base = json.loads((
            ROOT / "manifests/v16_b716_geot_backfill_preregister.json"
        ).read_text())
        base["enable_scope"] = {
            "expected_short_id_order": [
                "09582205_1883", "68bae76c_5364",
                "f38169cf_56fe", "6a36052f_c2b5",
            ],
        }
        canonical = json.loads(json.dumps(base, sort_keys=True))
        flattened = _expected_missing_in_preregistered_order(canonical)
        self.assertEqual(len(flattened), 72)
        self.assertEqual(
            [short_id for short_id, _pair in flattened[:2]],
            ["09582205_1883", "09582205_1883"])
        self.assertEqual(flattened[2][0], "68bae76c_5364")
        self.assertEqual(flattened[23][0], "f38169cf_56fe")
        self.assertEqual(flattened[44][0], "6a36052f_c2b5")

    def test_19_missing_extra_or_reordered_short_ids_are_rejected(self):
        base = json.loads((
            ROOT / "manifests/v16_b716_geot_backfill_preregister.json"
        ).read_text())
        expected = [
            "09582205_1883", "68bae76c_5364",
            "f38169cf_56fe", "6a36052f_c2b5",
        ]
        for invalid in (
                expected[:-1], list(reversed(expected)), expected + ["foreign"]):
            value = copy.deepcopy(base)
            value["enable_scope"] = {"expected_short_id_order": invalid}
            with self.assertRaisesRegex(
                    Exact191Error, "short-id order/set mismatch"):
                _expected_missing_in_preregistered_order(value)

    def test_20_candidate_repository_root_is_derived_from_bound_path(self):
        self.assertEqual(
            _candidate_repository_root(self.fixture.candidate_path),
            self.fixture.execution_repo_root)
        self.fixture.candidate_path.resolve().relative_to(
            self.fixture.execution_repo_root)
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "candidate.json"
            link.symlink_to(self.fixture.candidate_path)
            with self.assertRaisesRegex(Exact191Error, "not canonical"):
                _candidate_repository_root(link)

    def test_21_ordered72_authorization_closure_uses_task_ids(self):
        preflight = json.loads(self.fixture.preflight_path.read_text())
        preregister = json.loads(self.fixture.prereg_path.read_text())
        self.assertEqual(
            _ordered_task_id_closure_sha256(preflight),
            preregister["enable_scope"]["ordered72_sha256"])
        self.assertNotEqual(
            preregister["enable_scope"]["ordered72_sha256"],
            preregister["expected_missing_key_closure_sha256"])
        changed = copy.deepcopy(preflight)
        changed["tasks"][0]["task_id"] = changed["tasks"][1]["task_id"]
        with self.assertRaisesRegex(Exact191Error, "contain duplicates"):
            _ordered_task_id_closure_sha256(changed)

    def test_22_typed_insufficient_result_remains_visible(self):
        batch_original = self.fixture.batch_path.read_bytes()
        batch = json.loads(batch_original)
        task_id = batch["results"][0]["task_id"]
        directory = self.fixture.preflight_root / "tasks" / task_id
        result_path = directory / "result.json"
        result_original = result_path.read_bytes()
        try:
            task = json.loads((directory / "task.json").read_text())
            result = normalise_result(
                "insufficient_post_voxel_points",
                {"stage0_src": 5, "stage0_ref": 4}, task, directory,
                execution_binding=self.fixture.execution_binding,
                attempt_receipt_sha256=batch["results"][0][
                    "attempt_receipt_sha256"],
                authorized_task_view_sha256=sha256_file(
                    directory / "authorized_task_view.json"))
            result_path.unlink()
            atomic_json(result_path, result)
            batch["results"][0]["status"] = (
                "insufficient_post_voxel_points")
            batch["results"][0]["result_sha256"] = sha256_file(result_path)
            batch.pop("payload_sha256")
            batch["payload_sha256"] = stable_json_sha256(batch)
            self.fixture.batch_path.write_text(json.dumps(
                batch, indent=2, sort_keys=True) + "\n")
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)
            manifest = self.fixture.run(
                self.fixture.root / "typed-insufficient-visible")
            self.assertEqual(manifest["new_authorized_ok_count"], 71)
            self.assertEqual(
                manifest["new_authorized_typed_failure_count"], 1)
            self.assertEqual(manifest["typed_failure_total_count"], 17)
            self.assertTrue(manifest["typed_failures_visible_and_never_filtered"])
        finally:
            result_path.write_bytes(result_original)
            self.fixture.batch_path.write_bytes(batch_original)
            self.fixture.batch_sha = sha256_file(self.fixture.batch_path)


if __name__ == "__main__":
    unittest.main()
