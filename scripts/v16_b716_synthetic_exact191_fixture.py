#!/usr/bin/env python3
"""Create a hardened TEST-ONLY exact191 fixture from frozen b716 plans.

The fixture mirrors the 86b2077 exact191 execution/attempt/result lineage but
never imports or executes GeoTransformer.  It is consumable only through the
prepared builder's explicit test opt-in and cannot authorize a worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from safety.v16_b716_prepared_builder import (  # noqa: E402
    ALLOWLIST_SCHEMA, ATTEMPT_SCHEMA, AUTHORIZED_TASK_SCHEMA, AUTH_SCHEMA,
    BATCH_SCHEMA, CLEAN_SCHEMA, EXACT191_PAIR_SCHEMA, EXACT191_SCHEMA,
    EXPECTED_NEW_TYPED_FAILURE_COUNTS, OFFICIAL_RELEASE_SHA256,
    PREFLIGHT_SCHEMA, RESULT_SCHEMA, TASK_SCHEMA,
    create_only_deterministic_npz, create_only_json,
)
from safety.v16_b716_candidate_plan import (  # noqa: E402
    array_sha256 as execution_array_sha256,
)
from safety.v16_matched_region_colorpcr import (  # noqa: E402
    sha256_file, stable_json_sha256, verify_file,
)


CANDIDATE_SHA256 = (
    "774d4b49624e495412fcb72d1c79716d7b1b2b2840de72ce303ee8c70fd4ca68"
)


def _sealed(path: Path, value: dict) -> str:
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return create_only_json(path, value)


def _plain(path: Path, value: dict) -> str:
    return create_only_json(path, value)


def _file_row(path: Path, role: str, *, relative_to: Path | None = None,
              **extra) -> dict:
    shown = path.relative_to(relative_to).as_posix() if relative_to else str(path)
    return {"path": shown, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "role": role, **extra}


def _task_id(task: dict) -> str:
    return f"{task['short_id']}__{task['node_pair'][0]}_{task['node_pair'][1]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, default=(
        ROOT / "outputs/v16_b716_candidate_plan_fixed4_20260830/fixed4_manifest.json"))
    parser.add_argument("--candidate-manifest-sha256", default=CANDIDATE_SHA256)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    verify_file(args.candidate_manifest, args.candidate_manifest_sha256)
    candidate = json.loads(args.candidate_manifest.read_text())
    if (candidate.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256 or candidate.get("candidate_count") != 191
            or candidate.get("hypothesis_count") != 34):
        raise SystemExit("candidate manifest is not b716 fixed4")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("synthetic fixture output root must be empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidate_root = args.candidate_manifest.resolve().parent

    plans = []
    task_specs = []
    existing_entry_closure = []
    for pair_row in candidate["pairs"]:
        plan_path = candidate_root / pair_row["plan_path"]
        verify_file(plan_path, pair_row["plan_sha256"])
        plan = json.loads(plan_path.read_text())
        plans.append((pair_row, plan))
        for entry in plan["geot_entries"]:
            row = {
                "short_id": plan["short_id"],
                "candidate_index": int(entry["candidate_index"]),
                "node_pair": list(entry["node_pair"]),
                "object_pair": list(entry["object_pair"]),
            }
            if entry.get("origin") == "missing_execution_disabled":
                task_specs.append({"pair_id": plan["pair_id"], **row})
            else:
                existing_entry_closure.append({
                    **row, "entry_sha256": entry["entry_sha256"],
                    "source_cache_row": entry.get("source_cache_row"),
                    "status": entry["status"],
                })
    if len(task_specs) != 72 or len(existing_entry_closure) != 119:
        raise SystemExit("candidate plan is not exact 119+72")
    typed_new_keys = set()
    for ordinal, (_pair_row, plan) in enumerate(plans):
        missing = [
            (plan["short_id"], int(entry["candidate_index"]))
            for entry in plan["geot_entries"]
            if entry.get("origin") == "missing_execution_disabled"]
        typed_new_keys.update(
            missing[:EXPECTED_NEW_TYPED_FAILURE_COUNTS[ordinal]])
    if len(typed_new_keys) != 12:
        raise SystemExit("synthetic typed-new fixture is not exact12")

    execution_root = args.output_root / "synthetic_execution"
    prereg_path = execution_root / "preregister.json"
    transition = {
        "planned_task_state": "planned_disabled",
        "planned_execution_authorized": False,
        "authorized_view_schema": AUTHORIZED_TASK_SCHEMA,
        "authorized_state": "authorized_pending",
        "requires_disabled_false": True,
        "requires_real_execution_allowed_true": True,
        "planned_task_is_immutable": True,
    }
    cuda_uuid = "GPU-SYNTHETIC-NOT-EXECUTED"
    prereg = {
        "schema": "v16-b716-geot-backfill-preregister-v1",
        "frozen": True, "disabled": False,
        "execution_contract": {
            "real_execution_allowed": True,
            "authorization_derivation_contract": transition,
        },
        "cuda_hard_gate": {
            "required_services_checked": [],
            "baseline_service_process_count": 0,
            "baseline_service_identity": None,
        },
        "synthetic_test_fixture": True,
    }
    prereg_sha = _sealed(prereg_path, prereg)
    runtime_path = execution_root / "runtime" / "synthetic_runtime.py"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    with runtime_path.open("xb") as stream:
        stream.write(b"# synthetic non-executable runtime lineage\n")
    runtime_row = _file_row(
        runtime_path, "immutable_runtime_source_bundle:synthetic_runtime.py")
    runtime_bundle = [runtime_row]
    module_rows = [{
        "module": "synthetic_runtime", "path": runtime_row["path"],
        "bytes": runtime_row["bytes"], "sha256": runtime_row["sha256"],
    }]
    source_closure = sorted([
        _file_row(prereg_path, "frozen_backfill_preregistration"), runtime_row,
    ], key=lambda row: (row["path"], row["role"]))

    task_rows, task_values = [], []
    for spec in task_specs:
        task = {
            "schema": TASK_SCHEMA, "state": "planned_disabled",
            "execution_authorized": False,
            "execution_transition_contract": transition,
            "short_id": spec["short_id"], "pair_id": spec["pair_id"],
            "candidate_index": spec["candidate_index"],
            "node_pair": spec["node_pair"], "object_pair": spec["object_pair"],
            "synthetic_test_fixture": True,
        }
        task["task_sha256"] = stable_json_sha256(task)
        task_id = _task_id(task)
        task_path = execution_root / "tasks" / task_id / "task.json"
        task_file_sha = _plain(task_path, task)
        task_row = {
            "task_id": task_id, "short_id": task["short_id"],
            "node_pair": task["node_pair"], "task_sha256": task["task_sha256"],
            "path": task_path.relative_to(execution_root).as_posix(),
            "bytes": task_path.stat().st_size, "sha256": task_file_sha,
            "state": "planned_disabled",
        }
        task_values.append((task, task_path))
        task_rows.append(task_row)
    artifact_closure = [{**row, "role": "planned_task"} for row in task_rows]
    preflight = {
        "schema": PREFLIGHT_SCHEMA, "frozen": True, "disabled": False,
        "synthetic_test_fixture": True,
        "execution_derivation_contract": transition,
        "exact_batch_only": True, "key_selection_allowed": False,
        "result_based_selection_allowed": False, "official92_executed": False,
        "task_count": 72, "missing_key_count": 72, "tasks": task_rows,
        "task_closure_sha256": stable_json_sha256(task_rows),
        "source_closure": source_closure,
        "recursive_source_closure_sha256": stable_json_sha256(source_closure),
        "artifact_closure": artifact_closure,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifact_closure),
        "runtime_source_bundle": runtime_bundle,
        "immutable_runtime_source_bundle_sha256": stable_json_sha256(runtime_bundle),
        "runtime_module_entrypoints": module_rows,
        "runtime_module_entrypoint_closure_sha256": stable_json_sha256(module_rows),
    }
    preflight_path = execution_root / "preflight_manifest.json"
    preflight_sha = _sealed(preflight_path, preflight)
    preflight = json.loads(preflight_path.read_text())

    clean_path = execution_root / "clean_service_receipt.json"
    cuda_snapshot = {
        "uuid": cuda_uuid,
        "compute_processes": [],
        "synthetic_test_fixture": True,
    }
    clean_sha = _sealed(clean_path, {
        "schema": CLEAN_SCHEMA, "cuda_device_uuid": cuda_uuid,
        "clean": True,
        "services_checked": [],
        "compute_process_count": 0,
        "baseline_service_identity": None,
        "baseline_service_identity_sha256": None,
        "cuda_snapshot": cuda_snapshot,
        "cuda_snapshot_sha256": stable_json_sha256(cuda_snapshot),
        "synthetic_test_fixture": True,
    })
    authorization_path = execution_root / "authorization.json"
    authorization = {
        "schema": AUTH_SCHEMA, "authorized": True,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "missing_key_closure_sha256": stable_json_sha256(task_specs),
        "preregister_sha256": prereg_sha,
        "preflight_manifest_sha256": preflight_sha,
        "preflight_payload_sha256": preflight["payload_sha256"],
        "recursive_source_closure_sha256":
            preflight["recursive_source_closure_sha256"],
        "recursive_artifact_closure_sha256":
            preflight["recursive_artifact_closure_sha256"],
        "task_closure_sha256": preflight["task_closure_sha256"],
        "immutable_runtime_source_bundle_sha256":
            preflight["immutable_runtime_source_bundle_sha256"],
        "runtime_module_entrypoint_closure_sha256":
            preflight["runtime_module_entrypoint_closure_sha256"],
        "exact_batch_count": 72, "key_selection_allowed": False,
        "result_selection_allowed": False, "gt_allowed": False,
        "official92_allowed": False, "output_root": str(execution_root),
        "expires_utc": (datetime.now(timezone.utc)
                        + timedelta(days=3650)).isoformat(),
        "cuda_device_uuid": cuda_uuid,
        "clean_service_receipt_path": str(clean_path),
        "clean_service_receipt_sha256": clean_sha,
        "synthetic_test_fixture": True,
    }
    authorization_sha = _sealed(authorization_path, authorization)
    binding = {
        "authorization_sha256": authorization_sha,
        "preregister_sha256": prereg_sha,
        "preflight_manifest_sha256": preflight_sha,
        "preflight_payload_sha256": preflight["payload_sha256"],
        "recursive_source_closure_sha256":
            preflight["recursive_source_closure_sha256"],
        "recursive_artifact_closure_sha256":
            preflight["recursive_artifact_closure_sha256"],
        "task_closure_sha256": preflight["task_closure_sha256"],
        "immutable_runtime_source_bundle_sha256":
            preflight["immutable_runtime_source_bundle_sha256"],
        "runtime_module_entrypoint_closure_sha256":
            preflight["runtime_module_entrypoint_closure_sha256"],
        "cuda_device_uuid": cuda_uuid,
    }

    batch_rows, new_result_closure, lineage_by_candidate = [], [], {}
    for task, task_path in task_values:
        task_id = _task_id(task)
        directory = task_path.parent
        view = {
            "schema": AUTHORIZED_TASK_SCHEMA, "state": "authorized_pending",
            "execution_authorized": True, "planned_task_immutable": True,
            "planned_task_sha256": task["task_sha256"],
            "short_id": task["short_id"], "pair_id": task["pair_id"],
            "node_pair": task["node_pair"], "object_pair": task["object_pair"],
            "execution_binding": binding,
        }
        view_path = directory / "authorized_task_view.json"
        view_sha = _sealed(view_path, view)
        snapshot = {"uuid": cuda_uuid, "index": 0, "memory_used_mib": 0,
                    "utilization_percent": 0, "compute_processes": []}
        attempt = {
            "schema": ATTEMPT_SCHEMA, "task_sha256": task["task_sha256"],
            "authorized_task_view_sha256": view_sha, **binding,
            "cuda_snapshot": snapshot,
            "cuda_snapshot_sha256": stable_json_sha256(snapshot),
        }
        attempt_path = directory / "attempt_receipt.json"
        attempt_sha = _sealed(attempt_path, attempt)
        typed_failure = (
            task["short_id"], int(task["candidate_index"])) in typed_new_keys
        status = ("insufficient_post_voxel_points"
                  if typed_failure else "ok")
        result = {
            "schema": RESULT_SCHEMA, "task_sha256": task["task_sha256"],
            "short_id": task["short_id"], "pair_id": task["pair_id"],
            "node_pair": task["node_pair"], "object_pair": task["object_pair"],
            "status": status, "selector_eligible": False,
            "attempt_receipt_sha256": attempt_sha,
            "authorized_task_view_sha256": view_sha, **binding,
        }
        if typed_failure:
            result["failure"] = {
                "status": status,
                "detail": {"raw_points": 12, "post_voxel_points": 8,
                           "minimum_required": 16},
            }
            corr_sha = None
        else:
            base = float(task["candidate_index"] + 1)
            arrays = {
                "src_corr": np.asarray(
                    [[base, 0, 0], [base, 1, 0]], np.float32),
                "ref_corr": np.asarray(
                    [[base, 0, .1], [base, 1, .1]], np.float32),
                "scores": np.asarray([.9, .8], np.float32),
            }
            corr_path = directory / "correspondences.npz"
            corr_sha = create_only_deterministic_npz(corr_path, arrays)
            result["correspondences"] = {
                "path": "correspondences.npz",
                "bytes": corr_path.stat().st_size,
                "sha256": corr_sha,
                "arrays": {name: {"shape": list(value.shape),
                                    "dtype": str(value.dtype),
                                    "sha256": execution_array_sha256(value)}
                           for name, value in arrays.items()},
            }
        result_path = directory / "result.json"
        result_sha = _sealed(result_path, result)
        batch_rows.append({"task_id": task_id, "status": status,
                           "resumed": False,
                           "attempt_receipt_sha256": attempt_sha,
                           "result_sha256": result_sha})
        closure = {
            "short_id": task["short_id"],
            "candidate_index": task["candidate_index"],
            "node_pair": task["node_pair"], "object_pair": task["object_pair"],
            "task_sha256": task["task_sha256"],
            "authorized_task_view_sha256": view_sha,
            "attempt_sha256": attempt_sha, "result_sha256": result_sha,
            "status": status,
            "correspondence_sha256": corr_sha,
            "authorization_sha256": authorization_sha,
        }
        new_result_closure.append(closure)
        lineage_by_candidate[(task["short_id"], task["candidate_index"])] = closure
    attempt_closure = [{"task_id": row["task_id"],
                        "attempt_receipt_sha256": row["attempt_receipt_sha256"]}
                       for row in batch_rows]
    batch = {
        "schema": BATCH_SCHEMA, "exact_batch_count": 72,
        "selector_eligible": False, "result_based_selection_allowed": False,
        "execution_binding": binding, "results": batch_rows,
        "attempt_receipt_closure_sha256": stable_json_sha256(attempt_closure),
    }
    batch_path = execution_root / "batch_result.json"
    batch_sha = _sealed(batch_path, batch)

    pair_rows, artifacts = [], []
    ordered_keys, existing_keys, backfill_keys = [], [], []
    hypotheses_with_typed_failures = 0
    hypotheses_with_existing_typed_failures = 0
    for pair_row, plan in plans:
        records = plan["candidate_rank_records"]
        index_by_pair = {
            (int(row["source_index"]), int(row["reference_index"])): index
            for index, row in enumerate(records)
        }
        directory = args.output_root / "pairs" / plan["short_id"]
        entries_rows = []
        for entry in plan["geot_entries"]:
            candidate_key = {
                "short_id": plan["short_id"],
                "candidate_index": int(entry["candidate_index"]),
                "node_pair": list(entry["node_pair"]),
            }
            ordered_keys.append(candidate_key)
            key = (plan["short_id"], int(entry["candidate_index"]))
            if key in lineage_by_candidate:
                closure = lineage_by_candidate[key]
                status = closure["status"]
                origin = {"kind": "authorized_backfill", "status": status,
                          **{field: closure[field] for field in (
                              "task_sha256", "authorized_task_view_sha256",
                              "attempt_sha256", "result_sha256",
                              "correspondence_sha256", "authorization_sha256")}}
                if status != "ok":
                    origin["failure"] = {
                        "status": status,
                        "detail": {"raw_points": 12,
                                   "post_voxel_points": 8,
                                   "minimum_required": 16},
                    }
            else:
                existing_keys.append(candidate_key)
                origin = {"kind": "frozen_existing",
                          "entry_sha256": entry["entry_sha256"],
                          "source_cache_row": entry.get("source_cache_row"),
                          "status": entry["status"]}
                status = entry["status"]
            if key in lineage_by_candidate:
                backfill_keys.append(candidate_key)
            sealed = {"candidate_index": int(entry["candidate_index"]),
                      "node_pair": list(entry["node_pair"]),
                      "object_pair": list(entry["object_pair"]),
                      "status": status, "origin": origin,
                      "correspondences": {}}
            sealed["entry_sha256"] = stable_json_sha256(sealed)
            entries_rows.append(sealed)
        correspondences_path = directory / "exact191_correspondences.npz"
        corr_sha = create_only_deterministic_npz(correspondences_path, {
            "candidate_index": np.arange(len(records), dtype=np.int64),
            "synthetic_unexecuted": np.ones(len(records), np.uint8),
        })
        allowlist_path = directory / "frozen_hypothesis_allowlist.json"
        allow_rows = []
        for hypothesis in plan["hypotheses"]:
            indices = [index_by_pair[
                (int(row["source_index"]), int(row["reference_index"]))]
                for row in hypothesis["member_rank_records"]]
            existing_typed = [
                index for index in indices
                if entries_rows[index]["origin"]["kind"] == "frozen_existing"
                and entries_rows[index]["status"] != "ok"]
            new_typed = [
                index for index in indices
                if entries_rows[index]["origin"]["kind"]
                == "authorized_backfill"
                and entries_rows[index]["status"] != "ok"]
            typed = existing_typed + new_typed
            row = {
                "hypothesis_index": int(hypothesis["hypothesis_index"]),
                "hypothesis_sha256": hypothesis["hypothesis_sha256"],
                "member_candidate_indices": indices,
                "all_members_ok": not typed,
                "contains_typed_failure_members": bool(typed),
                "typed_failure_member_candidate_indices": typed,
                "existing_typed_failure_member_candidate_indices":
                    existing_typed,
                "new_typed_failure_member_candidate_indices": new_typed,
            }
            row["allowlist_entry_sha256"] = stable_json_sha256(row)
            allow_rows.append(row)
        pair_typed_hypotheses = sum(
            row["contains_typed_failure_members"] for row in allow_rows)
        pair_existing_typed_hypotheses = sum(
            bool(row["existing_typed_failure_member_candidate_indices"])
            for row in allow_rows)
        pair_existing_failed = sum(
            row["origin"]["kind"] == "frozen_existing"
            and row["status"] != "ok" for row in entries_rows)
        pair_new_failed = sum(
            row["origin"]["kind"] == "authorized_backfill"
            and row["status"] != "ok" for row in entries_rows)
        hypotheses_with_typed_failures += pair_typed_hypotheses
        hypotheses_with_existing_typed_failures += (
            pair_existing_typed_hypotheses)
        allowlist_sha = _sealed(allowlist_path, {
            "schema": ALLOWLIST_SCHEMA, "short_id": plan["short_id"],
            "pair_id": plan["pair_id"],
            "hypothesis_count": len(plan["hypotheses"]),
            "candidate_selection_allowed": False,
            "hypothesis_selection_allowed": False,
            "all_hypotheses_must_be_replayed": True,
            "typed_failure_members_visible_and_never_filtered": True,
            "hypotheses_with_typed_failure_members": pair_typed_hypotheses,
            "consumer_scope": "only_the_34_frozen_hypotheses_across_fixed4",
            "hypotheses": allow_rows,
        })
        entries_path = directory / "exact191_entries.json"
        entries_sha = _sealed(entries_path, {
            "schema": EXACT191_PAIR_SCHEMA, "short_id": plan["short_id"],
            "pair_id": plan["pair_id"], "candidate_count": len(records),
            "existing_count": pair_row["existing_reused"],
            "new_count": pair_row["missing_disabled"],
            "new_ok_count": pair_row["missing_disabled"] - pair_new_failed,
            "new_typed_failure_count": pair_new_failed,
            "existing_failed_count": pair_existing_failed,
            "hypotheses_with_typed_failure_members": pair_typed_hypotheses,
            "hypotheses_with_existing_typed_failure_members":
                pair_existing_typed_hypotheses,
            "entries": entries_rows, "correspondences": {},
            "hypothesis_allowlist": {},
        })
        paths = ((entries_path, entries_sha, "sealed_entries"),
                 (correspondences_path, corr_sha, "sealed_correspondences"),
                 (allowlist_path, allowlist_sha, "frozen_hypothesis_allowlist"))
        for path, digest, role in paths:
            artifacts.append({"path": path.relative_to(args.output_root).as_posix(),
                              "bytes": path.stat().st_size,
                              "sha256": digest, "role": role})
        pair_rows.append({
            "short_id": plan["short_id"], "pair_id": plan["pair_id"],
            "candidate_count": len(records),
            "existing_count": pair_row["existing_reused"],
            "new_count": pair_row["missing_disabled"],
            "new_ok_count": pair_row["missing_disabled"] - pair_new_failed,
            "new_typed_failure_count": pair_new_failed,
            "existing_failed_count": pair_existing_failed,
            "hypotheses_with_typed_failure_members": pair_typed_hypotheses,
            "hypotheses_with_existing_typed_failure_members":
                pair_existing_typed_hypotheses,
            "hypothesis_count": len(plan["hypotheses"]),
            "entries_path": entries_path.relative_to(args.output_root).as_posix(),
            "entries_bytes": entries_path.stat().st_size,
            "entries_sha256": entries_sha,
            "correspondences_path":
                correspondences_path.relative_to(args.output_root).as_posix(),
            "correspondences_bytes": correspondences_path.stat().st_size,
            "correspondences_sha256": corr_sha,
            "allowlist_path": allowlist_path.relative_to(args.output_root).as_posix(),
            "allowlist_bytes": allowlist_path.stat().st_size,
            "allowlist_sha256": allowlist_sha,
        })
    artifacts.sort(key=lambda row: row["path"])
    input_closure = sorted([
        _file_row(args.candidate_manifest.resolve(), "frozen_candidate_manifest"),
        _file_row(preflight_path, "authorized_preflight_manifest"),
        _file_row(prereg_path, "authorized_preregistration"),
        _file_row(authorization_path, "execution_authorization"),
        _file_row(batch_path, "exact72_batch_result"),
    ], key=lambda row: (row["path"], row["role"]))
    exact = {
        "schema": EXACT191_SCHEMA, "sealed": True,
        "synthetic_test_fixture": True,
        "candidate_count": 191, "existing_count": 119,
        "new_authorized_count": 72,
        "new_authorized_ok_count": 60,
        "new_authorized_typed_failure_count": 12,
        "existing_ok_count": int(candidate["geot_existing_ok"]),
        "existing_failed_count": int(candidate["geot_existing_failed"]),
        "typed_failure_existing_count": int(candidate["geot_existing_failed"]),
        "typed_failure_total_count": int(candidate["geot_existing_failed"]) + 12,
        "hypothesis_count": 34,
        "hypotheses_with_typed_failure_members": hypotheses_with_typed_failures,
        "hypotheses_with_existing_typed_failure_members":
            hypotheses_with_existing_typed_failures,
        "typed_failures_visible_and_never_filtered": True,
        "consumer_scope": "only_the_34_frozen_hypotheses_across_fixed4",
        "candidate_selection_allowed": False,
        "result_based_selection_allowed": False,
        "hypothesis_selection_allowed": False,
        "gt_allowed": False, "official92_allowed": False,
        "new_geot_execution_performed_by_merger": False,
        "b716_domain_only": True,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "fixed_hypothesis_distribution": [12, 8, 2, 12],
        "ordered_candidate_key_closure_sha256": stable_json_sha256(ordered_keys),
        "existing_candidate_key_closure_sha256": stable_json_sha256(existing_keys),
        "backfill_candidate_key_closure_sha256": stable_json_sha256(backfill_keys),
        "legacy_B_ep20_or_89ed_consumed": False,
        "candidate_manifest_sha256": args.candidate_manifest_sha256,
        "preflight_manifest_sha256": preflight_sha,
        "preregister_sha256": prereg_sha,
        "authorization_sha256": authorization_sha,
        "batch_result_sha256": batch_sha,
        "execution_binding": binding,
        "input_closure": input_closure,
        "recursive_input_closure_sha256": stable_json_sha256(input_closure),
        "existing_entry_closure": existing_entry_closure,
        "recursive_existing_entry_closure_sha256":
            stable_json_sha256(existing_entry_closure),
        "new_result_closure": new_result_closure,
        "recursive_new_result_closure_sha256":
            stable_json_sha256(new_result_closure),
        "pairs": pair_rows, "artifact_closure": artifacts,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifacts),
        "forbidden_inputs": ["GT", "official92", "result selection"],
    }
    path = args.output_root / "exact191_manifest.json"
    exact_sha = _sealed(path, exact)
    print(json.dumps({"test_fixture_only": True, "manifest": str(path),
                      "manifest_sha256": exact_sha}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
