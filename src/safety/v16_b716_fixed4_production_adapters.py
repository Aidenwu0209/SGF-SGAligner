"""Hash-bound production-stage adapter plans for the fixed4 DAG.

This module deliberately does not authorize or launch a stage.  It validates
an explicit input manifest, produces deterministic argv/contracts for the
reviewed production tools, and exposes pure frozen pair/aggregate gate calls.
The active execution boundary remains responsible for authorization,
immediate pre-launch revalidation, process isolation, and result sealing.
"""
from __future__ import annotations

import json
from itertools import combinations
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v13_dual_solver_runtime import (
    QUORUM as V13_QUORUM,
    REPEATS as V13_REPEATS,
    SCHEMA as V13_WORKER_SCHEMA,
    SUMMARY_SCHEMA as V13_RAW_SUMMARY_SCHEMA,
    array_sha256,
    sha256_file,
    stable_json_sha256,
    transform_distance,
    validate_se3,
)
from safety.v14_rigid_multihypothesis import (
    load_candidate_contract,
    verify_candidate_set_contract,
)
from safety.v15_safe_pose_cluster import select_unique_safe_pose_cluster
from safety.v16_b716_fixed4_stage_runners import (
    aggregate_gate_to_operational_fields,
    build_fixed4_aggregate_result,
    build_pair_gate_result,
    pair_gate_to_operational_fields,
)


INPUT_MANIFEST_SCHEMA = "v16-b716-fixed4-production-input-manifest-v1"
ADAPTER_CONTRACT_SCHEMA = "v16-b716-fixed4-production-adapter-contract-v1"
SLOT_EXPANSION_SCHEMA = "v16-b716-fixed4-production-slot-expansion-v1"
SLOT_RESULTS_SCHEMA = "v16-b716-fixed4-production-slot-results-v1"
V15_OUTCOME_SCHEMA = "v16-b716-fixed4-production-v15-outcome-v1"
MAX_CANDIDATE_SLOTS = 8
NORMAL_GATE_FAIL_RETURN_CODE = 2
SCIENCE_ROTATION_ABS_TOL_DEG = 1e-9
SCIENCE_TRANSLATION_ABS_TOL_M = 1e-12
SCIENCE_TRANSFORM_ABS_TOL = 1e-10
POLICY_FALSE = {
    "execution_authorized": False,
    "gt_allowed": False,
    "identity_fallback_allowed": False,
    "threshold_change_allowed": False,
    "result_selection_allowed": False,
    "reconstruction_authorized": False,
    "refusion_allowed": False,
}


class ProductionAdapterError(RuntimeError):
    """A production adapter manifest or derived plan failed closed."""


def _payload_valid(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    return value.get("payload_sha256") == stable_json_sha256(unsigned)


def _sha(value: Any, role: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ProductionAdapterError(f"invalid {role} SHA")
    return value


def _git_sha(value: Any, role: str) -> str:
    if (not isinstance(value, str) or len(value) not in {40, 64}
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ProductionAdapterError(f"invalid {role} git identity")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise ProductionAdapterError(f"{role} keys mismatch")


def _regular_file(path: Path, role: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProductionAdapterError(f"{role} missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ProductionAdapterError(f"{role} must be a non-symlink regular file")


def _verify_file_row(row: Mapping[str, Any], role: str) -> Path:
    _exact_keys(row, {"role", "path", "bytes", "sha256"}, role)
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        raise ProductionAdapterError(f"{role} path must be absolute")
    _regular_file(path, role)
    if (type(row.get("bytes")) is not int or row["bytes"] < 1
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != _sha(row.get("sha256"), role)):
        raise ProductionAdapterError(f"{role} bytes/SHA mismatch")
    return path


def _verify_directory_row(row: Mapping[str, Any], role: str) -> Path:
    _exact_keys(row, {"role", "path", "files", "closure_sha256"}, role)
    root = Path(str(row.get("path", "")))
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise ProductionAdapterError(f"{role} directory missing") from exc
    if not root.is_absolute() or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ProductionAdapterError(f"{role} must be an absolute real directory")
    files = row.get("files")
    if (not isinstance(files, list)
            or stable_json_sha256(files) != row.get("closure_sha256")):
        raise ProductionAdapterError(f"{role} directory closure mismatch")
    observed: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in [*dirnames, *filenames]:
            if stat.S_ISLNK((directory_path / name).lstat().st_mode):
                raise ProductionAdapterError(f"{role} directory contains symlink")
        for name in filenames:
            path = directory_path / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ProductionAdapterError(f"{role} contains non-regular file")
            observed.append({"path": str(path.relative_to(root)),
                             "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)})
    observed.sort(key=lambda item: item["path"])
    if files != observed:
        raise ProductionAdapterError(f"{role} directory is shallow or changed")
    return root


def _rows_by_role(rows: Any, role: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ProductionAdapterError(f"{role} rows missing")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("role"), str):
            raise ProductionAdapterError(f"{role} row {index} malformed")
        if row["role"] in result:
            raise ProductionAdapterError(f"duplicate {role} role")
        result[row["role"]] = row
    return result


def _output_paths(rows: Any, task_root: Path, *,
                  allow_existing: Sequence[str] = ()) -> dict[str, Path]:
    by_role = _rows_by_role(rows, "output")
    output_root = task_root / "production"
    result: dict[str, Path] = {}
    for role, row in by_role.items():
        _exact_keys(row, {"role", "path", "kind"}, f"output {role}")
        relative = Path(str(row.get("path", "")))
        if (relative.is_absolute() or ".." in relative.parts
                or row.get("kind") not in {"file", "directory"}):
            raise ProductionAdapterError(f"output {role} path/kind invalid")
        path = task_root / relative
        try:
            path.resolve().relative_to(output_root.resolve())
        except ValueError as exc:
            raise ProductionAdapterError(f"output {role} escapes production root") from exc
        if (path.exists() or path.is_symlink()) and role not in set(allow_existing):
            raise ProductionAdapterError(f"output {role} already exists")
        lowered = str(path).lower()
        if path.suffix.lower() == ".ply" or any(token in lowered for token in
                ("refusion", "reconstruction", "fused_map", "official92", "ground_truth")):
            raise ProductionAdapterError(f"output {role} is forbidden")
        result[role] = path
    return result


def load_bound_input_manifest(
    path: Path, expected_sha256: str, task: Mapping[str, Any], output_root: Path,
    *, allowed_existing_output_roles: Sequence[str] = (),
) -> dict[str, Any]:
    """Load and recursively verify every path in a stage input manifest."""
    path = Path(path)
    _regular_file(path, "production input manifest")
    if sha256_file(path) != _sha(expected_sha256, "input manifest"):
        raise ProductionAdapterError("production input manifest SHA mismatch")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ProductionAdapterError("production input manifest JSON invalid") from exc
    required = {"schema", "task_id", "task_payload_sha256", "stage",
        "file_inputs", "directory_inputs", "outputs", "parameters",
        *POLICY_FALSE.keys(), "payload_sha256"}
    if not isinstance(value, Mapping):
        raise ProductionAdapterError("production input manifest is not an object")
    _exact_keys(value, required, "production input manifest")
    if (not _payload_valid(value) or value.get("schema") != INPUT_MANIFEST_SCHEMA
            or value.get("task_id") != task.get("task_id")
            or value.get("task_payload_sha256") != task.get("payload_sha256")
            or value.get("stage") != task.get("stage")
            or any(value.get(key) is not expected
                   for key, expected in POLICY_FALSE.items())):
        raise ProductionAdapterError("production input manifest binding mismatch")
    files = _rows_by_role(value["file_inputs"], "file input")
    directories = _rows_by_role(value["directory_inputs"], "directory input")
    for role, row in files.items():
        _verify_file_row(row, f"file input {role}")
    for role, row in directories.items():
        _verify_directory_row(row, f"directory input {role}")
    task_root = Path(output_root).resolve() / "tasks" / str(task["task_id"])
    _output_paths(value["outputs"], task_root,
                  allow_existing=allowed_existing_output_roles)
    return dict(value)


def _require_roles(observed: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(observed) != expected:
        raise ProductionAdapterError(
            f"{role} roles mismatch missing={sorted(expected-set(observed))} "
            f"extra={sorted(set(observed)-expected)}")


def _require_output_shapes(rows: Any, expected: Mapping[str, tuple[str, str]]) -> None:
    by_role = _rows_by_role(rows, "output")
    for role, (kind, suffix) in expected.items():
        row = by_role[role]
        if row.get("kind") != kind or (suffix and not str(row.get("path", "")).endswith(suffix)):
            raise ProductionAdapterError(f"output {role} shape mismatch")


def _command(argv: Sequence[Any], outputs: Sequence[str], *, normal_rc: Sequence[int]) -> dict:
    return {"argv": [str(value) for value in argv],
            "declared_output_roles": list(outputs),
            "normal_return_codes": list(normal_rc),
            "shell": False, "environment_inherited": False}


def _color_contract(task: Mapping[str, Any], manifest: Mapping[str, Any],
                    task_root: Path) -> dict[str, Any]:
    files = _rows_by_role(manifest["file_inputs"], "file input")
    directories = _rows_by_role(manifest["directory_inputs"], "directory input")
    outputs = _output_paths(manifest["outputs"], task_root)
    _require_roles(files, {"sgaligner_python", "jojo_python",
        "sentinel_subprocess", "sentinel_worker",
        "corr_converter", "weights", "prepared_input", "extension"}, "color file")
    _require_roles(directories, {"colorpcr_repo"}, "color directory")
    _require_roles(outputs, {"sentinel_cache", "sentinel_evidence_dir",
        "exact_three_cache", "conversion_receipt"}, "color output")
    _require_output_shapes(manifest["outputs"], {
        "sentinel_cache": ("file", ".npz"),
        "sentinel_evidence_dir": ("directory", ""),
        "exact_three_cache": ("file", ".npz"),
        "conversion_receipt": ("file", ".json")})
    p = manifest.get("parameters")
    required = {"colorpcr_dependency_identity", "arm", "direction",
                "neighbor_limits", "sampling", "device"}
    if not isinstance(p, Mapping) or set(p) != required:
        raise ProductionAdapterError("color parameters mismatch")
    identity = p.get("colorpcr_dependency_identity")
    identity_keys = {"commit", "repo_closure_sha256", "python_tree_sha256",
                     "tracked_diff_sha256"}
    if not isinstance(identity, Mapping) or set(identity) != identity_keys:
        raise ProductionAdapterError("ColorPCR dependency identity mismatch")
    commit = _git_sha(identity.get("commit"), "ColorPCR dependency")
    python_tree = _sha(identity.get("python_tree_sha256"),
                       "ColorPCR Python tree")
    tracked_diff = _sha(identity.get("tracked_diff_sha256"),
                        "ColorPCR tracked diff")
    repo_closure = _sha(identity.get("repo_closure_sha256"),
                        "ColorPCR repository closure")
    if (p.get("arm") not in {"sgf_selected_union", "fullscan"}
            or p.get("direction") not in {"forward", "reverse"}
            or p.get("sampling") != "voxel10"
            or p.get("neighbor_limits") != [38, 36, 36, 38]
            or repo_closure != directories["colorpcr_repo"]["closure_sha256"]
            or str(task.get("pair_id", "")).count("_to_") != 1):
        raise ProductionAdapterError("color frozen parameter drift")
    sentinel = _command([
        files["sgaligner_python"]["path"], files["sentinel_subprocess"]["path"],
        "--python", files["jojo_python"]["path"],
        "--worker", files["sentinel_worker"]["path"],
        "--repo", directories["colorpcr_repo"]["path"],
        "--expected-commit", commit,
        "--weights", files["weights"]["path"], "--expected-weight-sha256", files["weights"]["sha256"],
        "--input", files["prepared_input"]["path"],
        "--expected-python-tree-sha256", python_tree,
        "--expected-tracked-diff-sha256", tracked_diff,
        "--extension", files["extension"]["path"],
        "--expected-extension-sha256", files["extension"]["sha256"],
        "--arm", p["arm"], "--direction", p["direction"],
        "--neighbor-limits", ",".join(str(value) for value in p["neighbor_limits"]),
        "--sampling", p["sampling"], "--device", p["device"],
        "--output", outputs["sentinel_cache"],
        "--evidence-dir", outputs["sentinel_evidence_dir"],
    ], ["sentinel_cache", "sentinel_evidence_dir"], normal_rc=[0])
    converter = _command([
        files["sgaligner_python"]["path"], files["corr_converter"]["path"],
        "--source", outputs["sentinel_cache"],
        "--prepared-input", files["prepared_input"]["path"],
        "--output", outputs["exact_three_cache"],
        "--receipt", outputs["conversion_receipt"],
        "--pair-id", task["pair_id"], "--arm", p["arm"],
        "--direction", p["direction"],
    ], ["exact_three_cache", "conversion_receipt"], normal_rc=[0])
    return {"commands": [sentinel, converter], "adapter_action": None}


def _pilot_contract(task: Mapping[str, Any], manifest: Mapping[str, Any],
                    task_root: Path) -> dict[str, Any]:
    files = _rows_by_role(manifest["file_inputs"], "file input")
    directories = _rows_by_role(manifest["directory_inputs"], "directory input")
    outputs = _output_paths(manifest["outputs"], task_root)
    _require_roles(files, {"python", "v14_builder", "v14_strict_runner",
        "forward_exact_three_cache", "reverse_exact_three_cache",
        "v13_preregister", "v14_preregister", "preflight_manifest",
        "pointdsc_checkpoint", "prepared_input"}, "pilot file")
    _require_roles(directories, {"pointdsc_root"}, "pilot directory")
    _require_roles(outputs, {"forward_candidate_dir", "reverse_candidate_dir",
        "candidate_set", "slot_root", "v15_outcome"}, "pilot output")
    _require_output_shapes(manifest["outputs"], {
        "forward_candidate_dir": ("directory", ""),
        "reverse_candidate_dir": ("directory", ""),
        "candidate_set": ("file", ".json"),
        "slot_root": ("directory", ""),
        "v15_outcome": ("file", ".json")})
    p = manifest.get("parameters")
    if (not isinstance(p, Mapping)
            or set(p) != {"pair_id", "arm", "max_candidate_slots", "device"}
            or p.get("pair_id") != task.get("pair_id")
            or p.get("arm") not in {"sgf_selected_union", "fullscan"}
            or p.get("max_candidate_slots") != MAX_CANDIDATE_SLOTS
            or p.get("device") != "cpu"):
        raise ProductionAdapterError("pilot frozen parameter drift")
    python = files["python"]["path"]; builder = files["v14_builder"]["path"]
    common = ["--pair-id", p["pair_id"], "--arm", p["arm"],
              "--preregister", files["v14_preregister"]["path"]]
    commands = [
        _command([python, builder, "build-direction", "--cache",
            files["forward_exact_three_cache"]["path"], "--output",
            outputs["forward_candidate_dir"], "--direction", "forward", *common],
            ["forward_candidate_dir"], normal_rc=[0]),
        _command([python, builder, "build-direction", "--cache",
            files["reverse_exact_three_cache"]["path"], "--output",
            outputs["reverse_candidate_dir"], "--direction", "reverse", *common],
            ["reverse_candidate_dir"], normal_rc=[0]),
        _command([python, builder, "pair-directions", "--forward-manifest",
            outputs["forward_candidate_dir"] / "manifest.json",
            "--reverse-manifest", outputs["reverse_candidate_dir"] / "manifest.json",
            "--output", outputs["candidate_set"], "--preregister",
            files["v14_preregister"]["path"]], ["candidate_set"], normal_rc=[0]),
    ]
    return {"commands": commands,
            "adapter_action": "expand_verified_candidate_set_then_v15"}


def _parent_gate_contract(task: Mapping[str, Any], manifest: Mapping[str, Any],
                          task_root: Path) -> dict[str, Any]:
    files = _rows_by_role(manifest["file_inputs"], "file input")
    directories = _rows_by_role(manifest["directory_inputs"], "directory input")
    outputs = _output_paths(manifest["outputs"], task_root)
    _require_roles(directories, set(), "gate directory")
    expected = {f"parent_result_{index}" for index, _ in enumerate(
        task.get("upstream_task_ids", ())) }
    _require_roles(files, expected, "gate parent-result file")
    _require_roles(outputs, {"core_gate_receipt"}, "gate output")
    _require_output_shapes(manifest["outputs"], {
        "core_gate_receipt": ("file", ".json")})
    p = manifest.get("parameters")
    if (not isinstance(p, Mapping)
            or set(p) != {"parent_task_ids", "parent_result_payload_sha256s"}
            or p.get("parent_task_ids") != task.get("upstream_task_ids")
            or not isinstance(p.get("parent_result_payload_sha256s"), list)
            or len(p["parent_result_payload_sha256s"]) != len(expected)):
        raise ProductionAdapterError("gate parent task order drift")
    expected_stage = ("bidirectional_multi_solver_pilot"
        if task.get("stage") == "v16_pair_hypothesis_cluster"
        else "v16_pair_hypothesis_cluster")
    parent_payloads = []
    parent_outcomes = []
    for index, parent_task_id in enumerate(p["parent_task_ids"]):
        path = _verify_file_row(files[f"parent_result_{index}"],
                                f"parent result {index}")
        try:
            value = json.loads(path.read_text())
        except Exception as exc:
            raise ProductionAdapterError(
                f"parent result {index} JSON invalid") from exc
        expected_payload_sha = _sha(
            p["parent_result_payload_sha256s"][index],
            f"parent result {index} payload")
        if (not isinstance(value, Mapping) or not _payload_valid(value)
                or value.get("payload_sha256") != expected_payload_sha
                or value.get("task_id") != parent_task_id
                or value.get("stage") != expected_stage):
            raise ProductionAdapterError(
                f"parent result {index} semantic binding mismatch")
        for key in ("gt_consumed", "official92_run", "thresholds_changed",
                    "result_selection_used", "default_checkpoint_replaced",
                    "refusion_run", "reconstruction_authorized"):
            if value.get(key) is not False:
                raise ProductionAdapterError(
                    f"parent result {index} policy binding mismatch")
        parent_payloads.append(expected_payload_sha)
        if expected_stage == "bidirectional_multi_solver_pilot":
            outcome = value.get("hypothesis_outcome")
            if (not isinstance(outcome, Mapping)
                    or set(outcome) != {"hypothesis_task_id", "gate_status",
                        "failure_class", "safe_transform",
                        "source_result_payload_sha256",
                        "measured_rotation_deg", "measured_translation_m",
                        "measurement_source_file_sha256",
                        "measurement_source_payload_sha256",
                        "measurement_candidate_slot",
                        "measurement_candidate_set_sha256",
                        "measurement_slot_results_payload_sha256",
                        "measurement_v15_decision_sha256"}
                    or outcome.get("hypothesis_task_id") != parent_task_id
                    or outcome.get("gate_status") not in {"PASS", "FAIL", "ABSTAIN"}):
                raise ProductionAdapterError(
                    f"parent result {index} hypothesis outcome mismatch")
            _sha(outcome.get("source_result_payload_sha256"),
                 f"parent result {index} V15 outcome")
            if outcome["gate_status"] in {"PASS", "FAIL"}:
                passed = outcome["gate_status"] == "PASS"
                rotation = outcome.get("measured_rotation_deg")
                translation = outcome.get("measured_translation_m")
                if value.get("status") != ("succeeded" if passed else "typed_failure") \
                        or outcome.get("failure_class") != (None if passed else
                            "FINITE_CONSENSUS_INCOMPATIBILITY") \
                        or not isinstance(outcome.get("measured_rotation_deg"),
                                          (int, float)) \
                        or not isinstance(outcome.get("measured_translation_m"),
                                          (int, float)) \
                        or float(rotation) < 0 or float(translation) < 0 \
                        or passed != (float(rotation) <= 5.0
                                      and float(translation) <= 0.10):
                    raise ProductionAdapterError(
                        f"parent result {index} finite status mismatch")
                for key in ("measurement_source_file_sha256",
                            "measurement_source_payload_sha256",
                            "measurement_candidate_set_sha256",
                            "measurement_slot_results_payload_sha256",
                            "measurement_v15_decision_sha256"):
                    _sha(outcome.get(key),
                         f"parent result {index} {key}")
                if type(outcome.get("measurement_candidate_slot")) is not int:
                    raise ProductionAdapterError(
                        f"parent result {index} measurement slot invalid")
                if passed:
                    try:
                        validate_se3(outcome.get("safe_transform"))
                    except Exception as exc:
                        raise ProductionAdapterError(
                            f"parent result {index} PASS transform invalid") from exc
                elif outcome.get("safe_transform") is not None:
                    raise ProductionAdapterError(
                        f"parent result {index} FAIL transform invalid")
            elif (value.get("status") != "typed_failure"
                  or not isinstance(outcome.get("failure_class"), str)
                  or outcome.get("safe_transform") is not None
                  or outcome.get("measured_rotation_deg") is not None
                  or outcome.get("measured_translation_m") is not None
                  or any(outcome.get(key) is not None for key in (
                    "measurement_source_file_sha256",
                    "measurement_source_payload_sha256",
                    "measurement_candidate_slot",
                    "measurement_candidate_set_sha256",
                    "measurement_slot_results_payload_sha256",
                    "measurement_v15_decision_sha256"))):
                raise ProductionAdapterError(
                    f"parent result {index} fail-closed outcome mismatch")
            parent_outcomes.append(dict(outcome))
        else:
            status = value.get("status"); decision = value.get("decision")
            transform = value.get("safe_cluster_transform")
            if status == "succeeded":
                if decision != "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER":
                    raise ProductionAdapterError(
                        f"parent result {index} pair decision mismatch")
                try:
                    validate_se3(transform)
                except Exception as exc:
                    raise ProductionAdapterError(
                        f"parent result {index} pair transform invalid") from exc
            elif (status != "typed_failure"
                  or decision not in {"NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER",
                    "PERMANENT_KNOWN_BAD_VETO"} or transform is not None):
                raise ProductionAdapterError(
                    f"parent result {index} pair fail-closed mismatch")
            parent_outcomes.append({"task_id": parent_task_id,
                "status": status, "decision": decision,
                "safe_cluster_transform": transform,
                "source_result_payload_sha256": expected_payload_sha})
    return {"commands": [], "verified_parent_result_payload_sha256s":
            parent_payloads, "verified_parent_outcomes": parent_outcomes,
            "adapter_action": (
        "invoke_frozen_pair_stage_runner" if task.get("stage") ==
        "v16_pair_hypothesis_cluster" else "invoke_frozen_aggregate_stage_runner")}


def build_stage_adapter_contract(
    task: Mapping[str, Any], manifest: Mapping[str, Any], output_root: Path,
) -> dict[str, Any]:
    """Build a deterministic, non-authorizing production stage contract."""
    stage = task.get("stage")
    task_root = Path(output_root).resolve() / "tasks" / str(task.get("task_id"))
    if manifest.get("task_id") != task.get("task_id") \
            or manifest.get("stage") != stage:
        raise ProductionAdapterError("task/manifest stage binding mismatch")
    if stage == "colorpcr_direction":
        body = _color_contract(task, manifest, task_root)
    elif stage == "bidirectional_multi_solver_pilot":
        body = _pilot_contract(task, manifest, task_root)
    elif stage in {"v16_pair_hypothesis_cluster", "fixed4_aggregate"}:
        body = _parent_gate_contract(task, manifest, task_root)
    else:
        raise ProductionAdapterError("unsupported fixed4 stage")
    value = {"schema": ADAPTER_CONTRACT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": stage,
        "input_manifest_payload_sha256": manifest["payload_sha256"],
        "file_input_closure_sha256": stable_json_sha256(manifest["file_inputs"]),
        "directory_input_closure_sha256": stable_json_sha256(
            manifest["directory_inputs"]),
        "output_contract_sha256": stable_json_sha256(manifest["outputs"]),
        **body, **POLICY_FALSE}
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def expand_verified_candidate_slots(
    task: Mapping[str, Any], manifest: Mapping[str, Any], candidate_set_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """Expand a sealed V14 set to exactly eight generated/absent slots."""
    task_root = Path(output_root).resolve() / "tasks" / str(task["task_id"])
    files = _rows_by_role(manifest["file_inputs"], "file input")
    directories = _rows_by_role(manifest["directory_inputs"], "directory input")
    outputs = _output_paths(
        manifest["outputs"], task_root,
        allow_existing=("forward_candidate_dir", "reverse_candidate_dir",
                        "candidate_set"))
    candidate_set = outputs["candidate_set"]
    _regular_file(candidate_set, "derived candidate set")
    if sha256_file(candidate_set) != _sha(candidate_set_sha256, "candidate set"):
        raise ProductionAdapterError("derived candidate set SHA mismatch")
    verified = verify_candidate_set_contract(candidate_set)
    count = verified["value"].get("candidate_count")
    if not isinstance(count, int) or not 0 <= count <= MAX_CANDIDATE_SLOTS:
        raise ProductionAdapterError("candidate set exceeds frozen slot budget")
    p = manifest["parameters"]
    slots = []
    for index in range(MAX_CANDIDATE_SLOTS):
        if index >= count:
            slots.append({"candidate_slot": index, "status": "typed_not_generated",
                          "failure_type": "CANDIDATE_SLOT_NOT_GENERATED",
                          "command": None})
            continue
        slot_output = outputs["slot_root"] / f"slot_{index:02d}"
        command = _command([
            files["python"]["path"], files["v14_strict_runner"]["path"],
            "--candidate-set", candidate_set, "--candidate-index", index,
            "--pair-id", p["pair_id"], "--arm", p["arm"],
            "--prepared-input", files["prepared_input"]["path"],
            "--v13-preregister", files["v13_preregister"]["path"],
            "--v14-preregister", files["v14_preregister"]["path"],
            "--preflight-manifest", files["preflight_manifest"]["path"],
            "--pointdsc-root", directories["pointdsc_root"]["path"],
            "--pointdsc-checkpoint", files["pointdsc_checkpoint"]["path"],
            "--output", slot_output, "--device", "cpu"],
            [], normal_rc=[0, NORMAL_GATE_FAIL_RETURN_CODE])
        slots.append({"candidate_slot": index, "status": "generated",
                      "failure_type": None, "command": command})
    value = {"schema": SLOT_EXPANSION_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "candidate_set_path": str(candidate_set),
        "candidate_set_sha256": candidate_set_sha256,
        "candidate_set_payload_sha256": verified["candidate_set_payload_sha256"],
        "candidate_count": count, "slots": slots,
        "slot_closure_sha256": stable_json_sha256(slots), **POLICY_FALSE}
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def classify_stage_returncode(returncode: int, *, summary_exists: bool) -> str:
    """Keep scientific rc=2 distinct from abnormal typed process failure."""
    if returncode == 0 and summary_exists:
        return "stage_succeeded"
    if returncode == NORMAL_GATE_FAIL_RETURN_CODE and summary_exists:
        return "normal_gate_failed"
    return "typed_process_failure"


def _json_file(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    """Read one immutable regular JSON file and detect read-time replacement."""
    _regular_file(path, role)
    before = sha256_file(path)
    if before != _sha(expected_sha256, role):
        raise ProductionAdapterError(f"{role} SHA mismatch")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ProductionAdapterError(f"{role} JSON invalid") from exc
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise ProductionAdapterError(f"{role} changed while reading")
    return value


def _science_number(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionAdapterError(f"{role} is not a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ProductionAdapterError(f"{role} is not a finite nonnegative number")
    return number


def _science_close(observed: Any, recomputed: float, *, role: str,
                   abs_tol: float) -> None:
    value = _science_number(observed, role)
    if not math.isclose(value, recomputed, rel_tol=0.0, abs_tol=abs_tol):
        raise ProductionAdapterError(f"{role} disagrees with worker evidence")


def _science_transform(value: Any, recomputed: np.ndarray, role: str) -> None:
    try:
        observed = validate_se3(value)
    except Exception as exc:
        raise ProductionAdapterError(f"{role} is not a valid SE(3)") from exc
    if not np.allclose(observed, recomputed, rtol=0.0,
                       atol=SCIENCE_TRANSFORM_ABS_TOL):
        raise ProductionAdapterError(f"{role} disagrees with worker evidence")


def _load_science_worker(path: Path, *, solver: str, direction: str,
                         repeat: int, expected_evidence_sha256: str,
                         raw: Mapping[str, Any]) -> dict[str, Any]:
    """Read one V13 worker atom and verify its own scientific payload hash."""
    _regular_file(path, f"{solver}/{direction}/{repeat} worker evidence")
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ProductionAdapterError("worker evidence JSON invalid") from exc
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise ProductionAdapterError("worker evidence changed while reading")
    evidence_sha = _sha(value.get("evidence_sha256"), "worker evidence")
    unsigned = {key: item for key, item in value.items()
                if key != "evidence_sha256"}
    if (evidence_sha != expected_evidence_sha256
            or evidence_sha != stable_json_sha256(unsigned)):
        raise ProductionAdapterError("worker evidence payload/SHA mismatch")
    expected_direction = ("source_to_reference" if direction == "forward"
                          else "reference_to_source")
    if (value.get("schema") != V13_WORKER_SCHEMA
            or value.get("solver") != solver
            or value.get("direction") != direction
            or value.get("repeat") != repeat
            or value.get("transform_direction") != expected_direction
            or value.get("unit") != "metre"
            or value.get("gt_free") is not True
            or value.get("gt_inputs") != []
            or value.get("fallback_used") is not False
            or type(value.get("known_bad_pair")) is not bool
            or type(value.get("correspondence_count")) is not int
            or value["correspondence_count"] < 40
            or not isinstance(value.get("diagnostics"), Mapping)
            or not isinstance(value.get("dependency"), Mapping)):
        raise ProductionAdapterError("worker evidence contract mismatch")
    cache_sha = raw.get("cache_sha256")
    correspondence_sha = raw.get("correspondence_sha256")
    if (not isinstance(cache_sha, Mapping)
            or not isinstance(correspondence_sha, Mapping)
            or value.get("cache_sha256") != cache_sha.get(direction)
            or value.get("correspondence_sha256") !=
                correspondence_sha.get(direction)
            or value.get("runtime_sha256") != raw.get("runtime_sha256")):
        raise ProductionAdapterError("worker/raw provenance binding mismatch")
    status = value.get("status")
    if status == "ok":
        if value.get("failure_type") is not None:
            raise ProductionAdapterError("successful worker has failure type")
        try:
            validate_se3(value.get("transform"))
        except Exception as exc:
            raise ProductionAdapterError(
                "successful worker transform is invalid") from exc
    elif status == "failed":
        if (value.get("transform") is not None
                or not isinstance(value.get("failure_type"), str)
                or not value["failure_type"]):
            raise ProductionAdapterError("failed worker is not fail-closed")
    else:
        raise ProductionAdapterError("worker status is invalid")
    return value


def _independent_worker_medoid(rows: Sequence[Mapping[str, Any]], *,
                               role: str) -> dict[str, Any]:
    """Recompute the frozen V13 q4 medoid without trusting its summary."""
    if len(rows) != V13_REPEATS:
        raise ProductionAdapterError(f"{role} worker group is not exact5")
    valid = [dict(row) for row in rows if row.get("status") == "ok"]
    cliques: list[tuple[int, ...]] = []
    for size in range(len(valid), 0, -1):
        for indices in combinations(range(len(valid)), size):
            compatible = True
            for left, right in combinations(indices, 2):
                rotation, translation = transform_distance(
                    valid[left]["transform"], valid[right]["transform"])
                if rotation > 5.0 or translation > 0.10:
                    compatible = False
                    break
            if compatible and not any(set(indices) < set(old)
                                      for old in cliques):
                cliques.append(indices)
    maximal = [clique for clique in cliques
               if not any(set(clique) < set(other) for other in cliques)]
    maximal.sort(key=lambda value: (-len(value), value))
    largest = len(maximal[0]) if maximal else 0
    winners = [value for value in maximal if len(value) == largest]
    rival = any(len(value) >= V13_QUORUM for value in maximal[1:])
    usable = (len(rows) == V13_REPEATS and largest >= V13_QUORUM
              and len(winners) == 1 and not rival)
    winning = winners[0] if usable else ()
    medoid_index = None
    if winning:
        def cost(index: int) -> tuple[float, int]:
            total = 0.0
            for other in winning:
                rotation, translation = transform_distance(
                    valid[index]["transform"], valid[other]["transform"])
                total += rotation / 5.0 + translation / 0.10
            return total, int(valid[index]["repeat"])
        medoid_index = min(winning, key=cost)
    return {
        "usable": usable,
        "requested": len(rows),
        "valid": len(valid),
        "clique_sizes": [len(value) for value in maximal],
        "winning_repeats": [int(valid[index]["repeat"])
                            for index in winning],
        "medoid_repeat": (int(valid[medoid_index]["repeat"])
                          if medoid_index is not None else None),
        "medoid_transform": (validate_se3(valid[medoid_index]["transform"])
                             if medoid_index is not None else None),
    }


def _verify_recorded_worker_gate(recorded: Any, recomputed: Mapping[str, Any],
                                 role: str) -> None:
    if not isinstance(recorded, Mapping):
        raise ProductionAdapterError(f"{role} gate is missing")
    for key in ("usable", "requested", "valid", "clique_sizes",
                "winning_repeats", "medoid_repeat"):
        if recorded.get(key) != recomputed.get(key):
            raise ProductionAdapterError(
                f"{role} gate disagrees with worker evidence")
    transform = recomputed.get("medoid_transform")
    if transform is None:
        if recorded.get("medoid_transform") is not None:
            raise ProductionAdapterError(f"{role} gate invented a medoid")
    else:
        _science_transform(recorded.get("medoid_transform"), transform,
                           f"{role} medoid transform")


def _verify_selected_slot_science(
    slot_root: Path, slot: int, raw: Mapping[str, Any], strict: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[float, float]:
    """Recompute the selected slot's cross-solver metric from worker atoms.

    Every worker transform is a column-vector SE(3).  Forward transforms map
    source to reference; reverse transforms map reference to source and are
    inverted only for direction-consistency checks.  The frozen cross-solver
    measurement compares the independently recomputed PointDSC-forward and
    pyGCRANSAC-forward q4 medoids.  Summary comparisons permit only numerical
    roundoff: 1e-9 degree and 1e-12 metre, with no relative tolerance.
    """
    worker_dir = slot_root / f"slot_{slot:02d}" / "raw" / "workers"
    expected_names = {f"{solver}_{direction}_{repeat}.json"
        for solver in ("pointdsc", "pygcransac")
        for direction in ("forward", "reverse")
        for repeat in range(V13_REPEATS)}
    try:
        observed_names = {path.name for path in worker_dir.iterdir()}
    except OSError as exc:
        raise ProductionAdapterError("selected worker evidence directory missing") from exc
    if observed_names != expected_names:
        raise ProductionAdapterError("selected worker evidence is not exact20")
    evidence_shas = raw.get("worker_evidence_sha256")
    if not isinstance(evidence_shas, Mapping):
        raise ProductionAdapterError("raw worker evidence digest map missing")
    workers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for solver in ("pointdsc", "pygcransac"):
        for direction in ("forward", "reverse"):
            group = []
            for repeat in range(V13_REPEATS):
                key = f"{solver}/{direction}/{repeat}"
                group.append(_load_science_worker(
                    worker_dir / f"{solver}_{direction}_{repeat}.json",
                    solver=solver, direction=direction, repeat=repeat,
                    expected_evidence_sha256=_sha(
                        evidence_shas.get(key), f"{key} worker evidence"),
                    raw=raw))
            workers[(solver, direction)] = group
    gates = {key: _independent_worker_medoid(rows,
             role=f"{key[0]}/{key[1]}")
             for key, rows in workers.items()}
    recorded_gates = raw.get("gates")
    if not isinstance(recorded_gates, Mapping) or set(recorded_gates) != {
            f"{solver}/{direction}" for solver in ("pointdsc", "pygcransac")
            for direction in ("forward", "reverse")}:
        raise ProductionAdapterError("raw gate matrix is incomplete")
    for (solver, direction), gate in gates.items():
        _verify_recorded_worker_gate(
            recorded_gates[f"{solver}/{direction}"], gate,
            f"{solver}/{direction}")
        if gate["usable"] is not True or gate["medoid_transform"] is None:
            raise ProductionAdapterError("selected V15 slot lacks four q4 medoids")

    direction_checks = raw.get("direction_checks")
    if not isinstance(direction_checks, Mapping) or set(direction_checks) != {
            "pointdsc", "pygcransac"}:
        raise ProductionAdapterError("raw direction checks are incomplete")
    for solver in ("pointdsc", "pygcransac"):
        rotation, translation = transform_distance(
            gates[(solver, "forward")]["medoid_transform"],
            np.linalg.inv(gates[(solver, "reverse")]["medoid_transform"]))
        recorded = direction_checks.get(solver)
        if not isinstance(recorded, Mapping):
            raise ProductionAdapterError("raw direction check is malformed")
        _science_close(recorded.get("rotation_deg"), rotation,
            role=f"{solver} direction rotation",
            abs_tol=SCIENCE_ROTATION_ABS_TOL_DEG)
        _science_close(recorded.get("translation_m"), translation,
            role=f"{solver} direction translation",
            abs_tol=SCIENCE_TRANSLATION_ABS_TOL_M)
        if recorded.get("usable") is not (rotation <= 5.0 and translation <= 0.10):
            raise ProductionAdapterError(
                f"{solver} direction usability disagrees with worker evidence")

    rotation, translation = transform_distance(
        gates[("pointdsc", "forward")]["medoid_transform"],
        gates[("pygcransac", "forward")]["medoid_transform"])
    recorded_cross = raw.get("cross_solver_check")
    if not isinstance(recorded_cross, Mapping):
        raise ProductionAdapterError("raw cross-solver check is missing")
    _science_close(recorded_cross.get("rotation_deg"), rotation,
        role="cross-solver rotation", abs_tol=SCIENCE_ROTATION_ABS_TOL_DEG)
    _science_close(recorded_cross.get("translation_m"), translation,
        role="cross-solver translation", abs_tol=SCIENCE_TRANSLATION_ABS_TOL_M)
    if recorded_cross.get("usable") is not (
            rotation <= 5.0 and translation <= 0.10):
        raise ProductionAdapterError(
            "cross-solver usability disagrees with worker evidence")

    medoid_safety = strict.get("medoid_safety")
    if not isinstance(medoid_safety, Mapping) or set(medoid_safety) != {
            f"{solver}/{direction}" for solver in ("pointdsc", "pygcransac")
            for direction in ("forward", "reverse")}:
        raise ProductionAdapterError("selected strict medoid matrix is incomplete")
    for key, gate in gates.items():
        name = f"{key[0]}/{key[1]}"
        row = medoid_safety[name]
        if not isinstance(row, Mapping):
            raise ProductionAdapterError("selected strict medoid row malformed")
        _science_transform(row.get("raw_transform"), gate["medoid_transform"],
                           f"{name} strict raw medoid")
        try:
            validate_se3(row.get("final_transform"))
        except Exception as exc:
            raise ProductionAdapterError(
                f"{name} strict final transform invalid") from exc

    realization = decision.get("selected_realization")
    selected_transform = decision.get("selected_transform")
    if (type(decision.get("selected_candidate_index")) is not int
            or decision.get("selected_candidate_index") != slot
            or decision.get("selected_candidate_index") !=
                strict.get("candidate_index")
            or decision.get("selected_candidate_sha256") !=
                strict.get("candidate_sha256")):
        raise ProductionAdapterError("selected V15 candidate binding mismatch")
    realization_map = {
        "pointdsc/forward": ("pointdsc/forward", False),
        "inverse(pointdsc/reverse)": ("pointdsc/reverse", True),
        "pygcransac/forward": ("pygcransac/forward", False),
        "inverse(pygcransac/reverse)": ("pygcransac/reverse", True),
    }
    if realization not in realization_map:
        raise ProductionAdapterError("selected V15 realization is invalid")
    strict_name, invert = realization_map[realization]
    expected_selected = validate_se3(
        medoid_safety[strict_name]["final_transform"])
    if invert:
        expected_selected = validate_se3(np.linalg.inv(expected_selected))
    _science_transform(selected_transform, expected_selected,
                       "selected V15 medoid transform")
    if decision.get("selected_transform_sha256") != array_sha256(
            validate_se3(selected_transform)):
        raise ProductionAdapterError("selected V15 transform SHA mismatch")
    return rotation, translation


def finalize_v15_from_slot_results(
    task: Mapping[str, Any], manifest: Mapping[str, Any],
    slot_expansion: Mapping[str, Any], slot_results_path: Path,
    slot_results_sha256: str, output_root: Path,
) -> dict[str, Any]:
    """Validate exact20 slot receipts and invoke the unchanged V15 selector.

    This is a pure sealing step: it never launches a solver and never authorizes
    the pair stage.  Absent slots are closed explicitly and generated slots must
    carry both an exact-20 raw summary and a strict V14 summary.
    """
    task_root = Path(output_root).resolve() / "tasks" / str(task["task_id"])
    outputs = _output_paths(manifest["outputs"], task_root,
        allow_existing=("forward_candidate_dir", "reverse_candidate_dir",
                        "candidate_set", "slot_root"))
    if (slot_expansion.get("schema") != SLOT_EXPANSION_SCHEMA
            or not _payload_valid(slot_expansion)
            or slot_expansion.get("task_id") != task.get("task_id")
            or slot_expansion.get("task_payload_sha256") != task.get("payload_sha256")
            or not isinstance(slot_expansion.get("slots"), list)
            or len(slot_expansion["slots"]) != MAX_CANDIDATE_SLOTS):
        raise ProductionAdapterError("slot expansion binding mismatch")
    results = _json_file(Path(slot_results_path), slot_results_sha256,
                         "slot results")
    required = {"schema", "task_id", "task_payload_sha256",
        "slot_expansion_payload_sha256", "candidate_set_sha256", "slots",
        *POLICY_FALSE.keys(), "payload_sha256"}
    _exact_keys(results, required, "slot results")
    if (not _payload_valid(results) or results.get("schema") != SLOT_RESULTS_SCHEMA
            or results.get("task_id") != task.get("task_id")
            or results.get("task_payload_sha256") != task.get("payload_sha256")
            or results.get("slot_expansion_payload_sha256")
                != slot_expansion.get("payload_sha256")
            or results.get("candidate_set_sha256")
                != slot_expansion.get("candidate_set_sha256")
            or any(results.get(key) is not expected
                   for key, expected in POLICY_FALSE.items())):
        raise ProductionAdapterError("slot results binding mismatch")
    rows = results.get("slots")
    if not isinstance(rows, list) or len(rows) != MAX_CANDIDATE_SLOTS:
        raise ProductionAdapterError("slot result closure is not exact-eight")
    evidence = []
    source_summaries: dict[int, dict[str, Any]] = {}
    expected_worker_keys = {f"{solver}/{direction}/{repeat}"
        for solver in ("pointdsc", "pygcransac")
        for direction in ("forward", "reverse") for repeat in range(5)}
    for index, (planned, row) in enumerate(zip(slot_expansion["slots"], rows)):
        if not isinstance(row, Mapping):
            raise ProductionAdapterError(f"slot result {index} malformed")
        _exact_keys(row, {"candidate_slot", "status", "returncode",
            "raw_summary_path", "raw_summary_sha256", "strict_summary_path",
            "strict_summary_sha256"}, f"slot result {index}")
        if row.get("candidate_slot") != index \
                or planned.get("candidate_slot") != index:
            raise ProductionAdapterError("slot result order drift")
        if planned.get("status") == "typed_not_generated":
            if (row.get("status") != "typed_not_generated"
                    or any(row.get(key) is not None for key in
                           ("returncode", "raw_summary_path", "raw_summary_sha256",
                            "strict_summary_path", "strict_summary_sha256"))):
                raise ProductionAdapterError(
                    "absent slot carries execution evidence")
            continue
        if planned.get("status") != "generated" \
                or row.get("status") not in {"stage_succeeded", "normal_gate_failed"}:
            raise ProductionAdapterError("generated slot result status invalid")
        expected_status = classify_stage_returncode(
            row.get("returncode"), summary_exists=True)
        if expected_status != row.get("status"):
            raise ProductionAdapterError("generated slot returncode/status mismatch")
        slot_root = outputs["slot_root"] / f"slot_{index:02d}"
        raw_path = Path(str(row.get("raw_summary_path", ""))).resolve()
        strict_path = Path(str(row.get("strict_summary_path", ""))).resolve()
        if (raw_path != (slot_root / "raw_summary.json").resolve()
                or strict_path != (slot_root / "summary.json").resolve()):
            raise ProductionAdapterError("slot result path is not canonical")
        raw = _json_file(raw_path, row.get("raw_summary_sha256"),
                         f"slot {index} raw summary")
        strict = _json_file(strict_path, row.get("strict_summary_sha256"),
                            f"slot {index} strict summary")
        worker_evidence = raw.get("worker_evidence_sha256", {})
        if (raw.get("schema") != V13_RAW_SUMMARY_SCHEMA
                or raw.get("worker_count") != 20
                or set(worker_evidence) != expected_worker_keys):
            raise ProductionAdapterError(
                f"slot {index} raw solver matrix is not exact20")
        for worker_key, digest in worker_evidence.items():
            _sha(digest, f"slot {index} worker {worker_key}")
        contract = load_candidate_contract(outputs["candidate_set"], index)
        evidence.append((contract, strict))
        source_summaries[index] = {
            "raw": raw, "raw_summary_sha256": row["raw_summary_sha256"],
            "strict": strict,
            "strict_summary_sha256": row["strict_summary_sha256"]}
    if len(evidence) != slot_expansion.get("candidate_count"):
        raise ProductionAdapterError("generated slot/evidence closure mismatch")
    decision = select_unique_safe_pose_cluster(
        evidence, known_bad=bool(task.get("known_bad")))
    # The active candidate preregisters exactly one measurement primitive that
    # is semantically identical to stage_runners.classify_finite_consensus:
    # the selected slot's PointDSC-forward versus pyGCRANSAC-forward q4-medoid
    # distance.  Never trust the derived raw_summary fields alone: recompute
    # them from the exact20 worker atoms, verify forward/reverse direction,
    # bind the selected V15 observed medoid to strict evidence, and only then
    # compare the recorded summary within explicit roundoff tolerances.
    observation = {"gate_status": "ABSTAIN", "measured_rotation_deg": None,
        "measured_translation_m": None,
        "measurement_source_file_sha256": None,
        "measurement_source_payload_sha256": None,
        "measurement_candidate_slot": None,
        "measurement_candidate_set_sha256": None,
        "measurement_slot_results_payload_sha256": None,
        "measurement_v15_decision_sha256": None}
    if decision.get("accepted") is True:
        selected = decision.get("selected_candidate_index")
        selected_source = source_summaries.get(selected)
        if selected_source is not None:
            raw = selected_source["raw"]
            rotation, translation = _verify_selected_slot_science(
                outputs["slot_root"], selected, raw,
                selected_source["strict"], decision)
            observation = {
                "gate_status": ("PASS" if rotation <= 5.0
                                and translation <= 0.10 else "FAIL"),
                "measured_rotation_deg": rotation,
                "measured_translation_m": translation,
                "measurement_source_file_sha256":
                    selected_source["raw_summary_sha256"],
                "measurement_source_payload_sha256": stable_json_sha256(raw),
                "measurement_candidate_slot": selected,
                "measurement_candidate_set_sha256":
                    slot_expansion["candidate_set_sha256"],
                "measurement_slot_results_payload_sha256":
                    results["payload_sha256"],
                "measurement_v15_decision_sha256":
                    stable_json_sha256(decision)}
    value = {"schema": V15_OUTCOME_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "candidate_set_sha256": slot_expansion["candidate_set_sha256"],
        "slot_expansion_payload_sha256": slot_expansion["payload_sha256"],
        "slot_results_payload_sha256": results["payload_sha256"],
        "v15_decision": decision, "gate_observation": observation,
        "downstream_authorized": False,
        **POLICY_FALSE}
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def run_frozen_pair_gate(
    payload: Mapping[str, Any], *,
    verified_parent_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Invoke the frozen pair gate after exact parent-derived partition checks."""
    replayed = [row.get("hypothesis_task_id") for row in verified_parent_outcomes]
    eligible = [row["hypothesis_task_id"] for row in verified_parent_outcomes
                if row.get("gate_status") in {"PASS", "FAIL"}]
    abstained = [row["hypothesis_task_id"] for row in verified_parent_outcomes
                 if row.get("gate_status") == "ABSTAIN"]
    if (payload.get("replayed_hypothesis_task_ids") != replayed
            or payload.get("eligible_hypothesis_task_ids") != eligible
            or payload.get("typed_abstention_hypothesis_task_ids") != abstained
            or [row.get("hypothesis_task_id") for row in
                payload.get("hypothesis_gate_results", ())] != eligible):
        raise ProductionAdapterError("pair core input is not parent-derived")
    return pair_gate_to_operational_fields(build_pair_gate_result(**dict(payload)))


def run_frozen_aggregate_gate(
    payload: Mapping[str, Any], *,
    verified_parent_outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Invoke the frozen aggregate gate after exact parent-result binding."""
    pair_results = payload.get("pair_results")
    if not isinstance(pair_results, Sequence) \
            or len(pair_results) != len(verified_parent_outcomes):
        raise ProductionAdapterError("aggregate parent count mismatch")
    for core, parent in zip(pair_results, verified_parent_outcomes):
        if (core.get("task_id") != parent.get("task_id")
                or core.get("decision") != parent.get("decision")
                or core.get("transform") != parent.get("safe_cluster_transform")):
            raise ProductionAdapterError(
                "aggregate core input is not parent-derived")
    return aggregate_gate_to_operational_fields(
        build_fixed4_aggregate_result(**dict(payload)))


def materialize_contract_create_only(root: Path, path: Path,
                                     value: Mapping[str, Any]) -> dict[str, Any]:
    """Write one immutable contract with O_EXCL; never overwrite/resume-drift."""
    root, path = Path(root).resolve(), Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProductionAdapterError("contract output escapes root") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    except FileExistsError as exc:
        raise ProductionAdapterError("contract output already exists") from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"path": str(path), "bytes": len(encoded), "sha256": sha256_file(path)}
