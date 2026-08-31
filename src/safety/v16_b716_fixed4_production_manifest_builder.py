"""Create-only builders for hash-bound fixed4 production manifests.

The builders are intentionally separate from dispatch and execution.  They
derive stage inputs from a sealed operational task, a reviewed asset manifest,
and canonical upstream RESULT-v5 files.  Caller-supplied runtime paths in a
task are never consumed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence
import uuid

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_active_production_wrapper import (
    ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
)
from safety.v16_b716_fixed4_execution_pilot import (
    ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA, POLICY_FALSE_FIELDS,
    PREFLIGHT_SCHEMA, RESULT_SCHEMA, TASK_SCHEMA,
)
from safety.v16_b716_fixed4_production_adapters import (
    INPUT_MANIFEST_SCHEMA, MAX_CANDIDATE_SLOTS,
    POLICY_FALSE as ADAPTER_POLICY_FALSE,
)
from safety.v16_b716_fixed4_subprocess_contract import (
    ACTIVE_PREFLIGHT_V2_SCHEMA,
    Fixed4SubprocessContractError, create_only_bytes_beneath,
    read_no_symlink_bytes,
)


PRODUCTION_ASSETS_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-production-assets-manifest-v1")
PRODUCTION_RUNTIME_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-production-runtime-manifest-v1")
PRODUCTION_MANIFEST_BUILD_RECEIPT_SCHEMA = (
    "v16-b716-fixed4-production-manifest-build-receipt-v2")
PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA = (
    "v16-b716-fixed4-production-manifest-transaction-commit-v2")
PRODUCTION_MANIFEST_TRANSACTION_DIRECTORY = "production_manifest_transactions"

_COLOR_ASSET_FILES = {
    "sgaligner_python", "jojo_python", "sentinel_subprocess",
    "sentinel_worker", "corr_converter", "weights", "extension",
}
_PILOT_ASSET_FILES = {
    "python", "v14_builder", "v14_strict_runner", "v13_preregister",
    "v14_preregister", "preflight_manifest", "pointdsc_checkpoint",
}
_STAGES = {
    "colorpcr_direction", "bidirectional_multi_solver_pilot",
    "v16_pair_hypothesis_cluster", "fixed4_aggregate",
}


class ProductionManifestBuilderError(RuntimeError):
    """A production manifest could not be derived without trusting input."""


def _payload_valid(value: Mapping[str, Any]) -> bool:
    return value.get("payload_sha256") == stable_json_sha256({
        key: item for key, item in value.items() if key != "payload_sha256"})


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise ProductionManifestBuilderError(f"{role} keys mismatch")


def _safe_task_id(value: Any) -> str:
    if (not isinstance(value, str) or not value
            or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                   for ch in value)
            or value in {".", ".."}):
        raise ProductionManifestBuilderError("task id is not path safe")
    return value


def _regular_file(path: Path, role: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ProductionManifestBuilderError(f"{role} path must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProductionManifestBuilderError(f"{role} missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ProductionManifestBuilderError(
            f"{role} must be a non-symlink regular file")
    return path


def _file_row(path: Path, role: str) -> dict[str, Any]:
    path = _regular_file(path, role)
    return {"role": role, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _verify_file_row(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    _exact_keys(row, {"role", "path", "bytes", "sha256"}, role)
    path = _regular_file(Path(str(row.get("path", ""))), role)
    if (row.get("role") != role or type(row.get("bytes")) is not int
            or row["bytes"] < 1 or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row.get("sha256")):
        raise ProductionManifestBuilderError(f"{role} file binding drift")
    return dict(row)


def _verify_asset_file_row(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    """Verify an asset row and canonicalize an allowed symlink to its target.

    Asset discovery may name an environment interpreter through a symlink.  A
    production input manifest never preserves that indirection: it records the
    resolved regular file and its bytes/SHA so the adapter still consumes only
    non-symlink paths.
    """
    _exact_keys(row, {"role", "path", "bytes", "sha256"}, role)
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        raise ProductionManifestBuilderError(f"{role} path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ProductionManifestBuilderError(f"{role} asset missing") from exc
    resolved = _regular_file(resolved, role)
    if (row.get("role") != role or type(row.get("bytes")) is not int
            or row["bytes"] < 1 or resolved.stat().st_size != row["bytes"]
            or sha256_file(resolved) != row.get("sha256")):
        raise ProductionManifestBuilderError(f"{role} asset binding drift")
    return {"role": role, "path": str(resolved), "bytes": row["bytes"],
            "sha256": row["sha256"]}


def _verify_directory_row(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    _exact_keys(row, {"role", "path", "files", "closure_sha256"}, role)
    root = Path(str(row.get("path", "")))
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise ProductionManifestBuilderError(f"{role} directory missing") from exc
    if (not root.is_absolute() or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode)):
        raise ProductionManifestBuilderError(
            f"{role} must be an absolute non-symlink directory")
    observed: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            if stat.S_ISLNK((base / name).lstat().st_mode):
                raise ProductionManifestBuilderError(
                    f"{role} directory contains symlink")
        for name in filenames:
            path = base / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ProductionManifestBuilderError(
                    f"{role} directory contains non-regular file")
            observed.append({"path": str(path.relative_to(root)),
                "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    observed.sort(key=lambda item: item["path"])
    if (row.get("role") != role or row.get("files") != observed
            or row.get("closure_sha256") != stable_json_sha256(observed)):
        raise ProductionManifestBuilderError(f"{role} closure drift")
    return dict(row)


def _rows_by_role(rows: Any, role: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ProductionManifestBuilderError(f"{role} rows missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if (not isinstance(row, Mapping) or not isinstance(row.get("role"), str)
                or row["role"] in result):
            raise ProductionManifestBuilderError(f"{role} rows malformed")
        result[row["role"]] = row
    return result


def _load_json(path: Path, expected_sha256: str | None, role: str) -> dict[str, Any]:
    path = _regular_file(path, role)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ProductionManifestBuilderError(f"{role} SHA drift")
    try:
        value = json.loads(read_no_symlink_bytes(path, role))
    except Exception as exc:
        raise ProductionManifestBuilderError(f"{role} JSON invalid") from exc
    if not isinstance(value, dict):
        raise ProductionManifestBuilderError(f"{role} must be an object")
    return value


def _validate_task_preflight(task: Mapping[str, Any], preflight: Mapping[str, Any],
                             output_root: Path) -> tuple[str, Path]:
    task_id = _safe_task_id(task.get("task_id"))
    stage = task.get("stage")
    if (not _payload_valid(task) or task.get("schema") != TASK_SCHEMA
            or stage not in _STAGES):
        raise ProductionManifestBuilderError("operational task binding invalid")
    if (not _payload_valid(preflight) or preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("output_root") != str(Path(output_root).resolve())):
        raise ProductionManifestBuilderError("active preflight/root binding invalid")
    active = preflight.get("active_subprocess_contract")
    if (not isinstance(active, Mapping)
            or active.get("schema") != ACTIVE_PREFLIGHT_V2_SCHEMA
            or active.get("runner_mode") != "active"
            or active.get("production_adapter_protocol_ready") is not True
            or active.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")):
        raise ProductionManifestBuilderError(
            "production-ready active preflight-v2 is required")
    identity = task.get("preflight_identity")
    if (not isinstance(identity, Mapping)
            or identity.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")):
        raise ProductionManifestBuilderError("task/preflight identity drift")
    descriptor = task.get("stage_runner_input_descriptor")
    if (not isinstance(descriptor, Mapping)
            or not _payload_valid(descriptor)
            or descriptor.get("schema") != ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA
            or descriptor.get("task_id") != task_id
            or descriptor.get("stage") != stage
            or descriptor.get("upstream_task_ids") != task.get("upstream_task_ids")
            or descriptor.get("derivation_policy")
                != "dispatcher_only_never_trust_task_runtime_paths"
            or descriptor.get("production_adapter_protocol_ready") is not True):
        raise ProductionManifestBuilderError(
            "production-ready active stage descriptor-v2 is required")
    execution_binding = task.get("execution_binding")
    if (not isinstance(execution_binding, Mapping)
            or execution_binding.get("runner_mode") != "active"
            or execution_binding.get("stage_implementation_status")
                != "production_adapter_ready"):
        raise ProductionManifestBuilderError(
            "production-ready execution binding is required")
    task_root = Path(output_root).resolve() / "tasks" / task_id
    return str(stage), task_root


def _validate_assets(value: Mapping[str, Any], stage: str) -> tuple[dict, dict, dict]:
    required = {"schema", "stage", "file_assets", "directory_assets",
                "stage_parameters", *POLICY_FALSE_FIELDS.keys(),
                "payload_sha256"}
    _exact_keys(value, required, "production assets manifest")
    if (not _payload_valid(value)
            or value.get("schema") != PRODUCTION_ASSETS_MANIFEST_SCHEMA
            or value.get("stage") != stage
            or any(value.get(key) is not False for key in POLICY_FALSE_FIELDS)):
        raise ProductionManifestBuilderError("production assets manifest invalid")
    files = _rows_by_role(value["file_assets"], "asset file")
    directories = _rows_by_role(value["directory_assets"], "asset directory")
    parameters = value.get("stage_parameters")
    if not isinstance(parameters, Mapping):
        raise ProductionManifestBuilderError("stage parameters missing")
    return files, directories, dict(parameters)


def _prepared_input(task: Mapping[str, Any]) -> dict[str, Any]:
    row = _file_row(Path(str(task.get("prepared_input_path", ""))),
                    "prepared_input")
    if row["sha256"] != task.get("prepared_input_sha256"):
        raise ProductionManifestBuilderError("prepared input task binding drift")
    return row


def _validate_parent_results(task: Mapping[str, Any], output_root: Path,
        upstream_results: Mapping[str, Mapping[str, Any]]) -> list[tuple[dict, Path]]:
    upstream = task.get("upstream_task_ids")
    if not isinstance(upstream, list) or set(upstream_results) != set(upstream):
        raise ProductionManifestBuilderError("parent result inventory mismatch")
    expected_stage = {
        "bidirectional_multi_solver_pilot": "colorpcr_direction",
        "v16_pair_hypothesis_cluster": "bidirectional_multi_solver_pilot",
        "fixed4_aggregate": "v16_pair_hypothesis_cluster",
    }.get(task["stage"])
    if expected_stage is None:
        if upstream or upstream_results:
            raise ProductionManifestBuilderError("source stage has parents")
        return []
    result: list[tuple[dict, Path]] = []
    for parent_id in upstream:
        _safe_task_id(parent_id)
        path = (Path(output_root).resolve() / "tasks" / parent_id / "result.json")
        observed = _load_json(path, None, f"parent result {parent_id}")
        supplied = upstream_results[parent_id]
        if (not isinstance(supplied, Mapping) or observed != dict(supplied)
                or not _payload_valid(observed)
                or observed.get("schema") != RESULT_SCHEMA
                or observed.get("task_id") != parent_id
                or observed.get("stage") != expected_stage):
            raise ProductionManifestBuilderError("parent RESULT-v5 binding drift")
        result.append((observed, path))
    return result


def load_canonical_upstream_results(task: Mapping[str, Any],
                                    output_root: Path) -> dict[str, dict[str, Any]]:
    """Load parents only from canonical task result paths."""
    result: dict[str, dict[str, Any]] = {}
    for parent in task.get("upstream_task_ids", []):
        _safe_task_id(parent)
        path = Path(output_root).resolve() / "tasks" / parent / "result.json"
        result[parent] = _load_json(path, None, f"parent result {parent}")
    return result


def _exact3_from_parent(parent: Mapping[str, Any], output_root: Path) -> tuple[str, dict]:
    rows = parent.get("output_artifacts")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ProductionManifestBuilderError("Color parent artifact inventory invalid")
    artifact_row = rows[0]
    if (not isinstance(artifact_row, Mapping)
            or set(artifact_row) < {"path", "bytes", "sha256"}):
        raise ProductionManifestBuilderError("Color parent artifact row invalid")
    artifact_path = Path(output_root).resolve() / str(artifact_row["path"])
    artifact = _load_json(
        artifact_path, str(artifact_row["sha256"]), "Color exact3 binding")
    if (not _payload_valid(artifact) or artifact.get("role") != "exact_three_cache"
            or artifact.get("task_id") != parent.get("task_id")):
        raise ProductionManifestBuilderError("Color exact3 binding invalid")
    source = _regular_file(Path(str(artifact.get("source_path", ""))),
                           "Color exact3 source")
    if sha256_file(source) != artifact.get("source_sha256"):
        raise ProductionManifestBuilderError("Color exact3 source drift")
    parent_task_path = (Path(output_root).resolve() / "tasks" /
                        str(parent["task_id"]) / "task.json")
    parent_task = _load_json(parent_task_path, None, "Color parent task")
    if (not _payload_valid(parent_task)
            or parent_task.get("payload_sha256") != parent.get("task_payload_sha256")
            or parent_task.get("stage") != "colorpcr_direction"
            or parent_task.get("direction") not in {"forward", "reverse"}):
        raise ProductionManifestBuilderError("Color parent direction binding invalid")
    return str(parent_task["direction"]), _file_row(
        source, f"{parent_task['direction']}_exact_three_cache")


def _output_rows(stage: str) -> list[dict[str, str]]:
    layouts = {
        "colorpcr_direction": (
            ("sentinel_cache", "production/sentinel_cache.npz", "file"),
            ("sentinel_evidence_dir", "production/sentinel_evidence", "directory"),
            ("exact_three_cache", "production/exact_three_cache.npz", "file"),
            ("conversion_receipt", "production/conversion_receipt.json", "file")),
        "bidirectional_multi_solver_pilot": (
            ("forward_candidate_dir", "production/forward_candidates", "directory"),
            ("reverse_candidate_dir", "production/reverse_candidates", "directory"),
            ("candidate_set", "production/candidate_set.json", "file"),
            ("slot_root", "production/slots", "directory"),
            ("v15_outcome", "production/v15_outcome.json", "file")),
        "v16_pair_hypothesis_cluster": (
            ("core_gate_receipt", "production/core_gate_receipt.json", "file"),),
        "fixed4_aggregate": (
            ("core_gate_receipt", "production/core_gate_receipt.json", "file"),),
    }
    return [{"role": role, "path": path, "kind": kind}
            for role, path, kind in layouts[stage]]


def build_production_input_manifest(*, task: Mapping[str, Any],
        preflight: Mapping[str, Any], output_root: Path,
        upstream_results: Mapping[str, Mapping[str, Any]],
        production_assets_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive a stage input manifest without reading task runtime paths."""
    stage, _task_root = _validate_task_preflight(task, preflight, output_root)
    assets, directories, stage_parameters = _validate_assets(
        production_assets_manifest, stage)
    parents = _validate_parent_results(task, output_root, upstream_results)
    files: list[dict[str, Any]] = []
    dirs: list[dict[str, Any]] = []
    parameters: dict[str, Any]
    if stage == "colorpcr_direction":
        if set(assets) != _COLOR_ASSET_FILES or set(directories) != {"colorpcr_repo"}:
            raise ProductionManifestBuilderError("Color asset roles mismatch")
        files = [_verify_asset_file_row(assets[role], role)
                 for role in sorted(_COLOR_ASSET_FILES)]
        files.append(_prepared_input(task))
        dirs = [_verify_directory_row(directories["colorpcr_repo"], "colorpcr_repo")]
        p = stage_parameters.get(stage)
        if (not isinstance(p, Mapping)
                or set(p) != {"colorpcr_dependency_identity", "arm", "device"}):
            raise ProductionManifestBuilderError("Color asset parameters mismatch")
        parameters = {"colorpcr_dependency_identity": dict(
            p["colorpcr_dependency_identity"]), "arm": p["arm"],
            "direction": task.get("direction"),
            "neighbor_limits": list(task.get("neighbor_limits", [])),
            "sampling": "voxel10", "device": p["device"]}
    elif stage == "bidirectional_multi_solver_pilot":
        if set(assets) != _PILOT_ASSET_FILES or set(directories) != {"pointdsc_root"}:
            raise ProductionManifestBuilderError("pilot asset roles mismatch")
        files = [_verify_asset_file_row(assets[role], role)
                 for role in sorted(_PILOT_ASSET_FILES)]
        exact3: dict[str, dict] = {}
        for parent, _ in parents:
            direction, row = _exact3_from_parent(parent, output_root)
            if direction in exact3:
                raise ProductionManifestBuilderError("duplicate Color direction parent")
            exact3[direction] = row
        if set(exact3) != {"forward", "reverse"}:
            raise ProductionManifestBuilderError("pilot lacks exact Color directions")
        files.extend([exact3["forward"], exact3["reverse"], _prepared_input(task)])
        dirs = [_verify_directory_row(directories["pointdsc_root"], "pointdsc_root")]
        p = stage_parameters.get(stage)
        if not isinstance(p, Mapping) or set(p) != {"arm"}:
            raise ProductionManifestBuilderError("pilot asset parameters mismatch")
        parameters = {"pair_id": task.get("pair_id"), "arm": p["arm"],
            "max_candidate_slots": MAX_CANDIDATE_SLOTS, "device": "cpu"}
    else:
        if assets or directories:
            raise ProductionManifestBuilderError("gate stage accepts no asset paths")
        if stage_parameters.get(stage) not in ({}, None):
            raise ProductionManifestBuilderError("gate stage accepts no asset parameters")
        files = []
        for index, (_parent, path) in enumerate(parents):
            files.append(_file_row(path, f"parent_result_{index}"))
        parameters = {"parent_task_ids": list(task["upstream_task_ids"]),
            "parent_result_payload_sha256s":
                [parent["payload_sha256"] for parent, _ in parents]}
    value = {"schema": INPUT_MANIFEST_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": stage,
        "file_inputs": files, "directory_inputs": dirs,
        "outputs": _output_rows(stage), "parameters": parameters,
        **ADAPTER_POLICY_FALSE}
    return _sealed(value)


def _validate_runtime_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {"schema", "interpreter", "runtime_dependency_files",
        "controlled_sys_path", "environment", "runner_source",
        "production_wrapper_cli", "validator_source",
        *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    _exact_keys(value, required, "production runtime manifest")
    if (not _payload_valid(value)
            or value.get("schema") != PRODUCTION_RUNTIME_MANIFEST_SCHEMA
            or any(value.get(key) is not False for key in POLICY_FALSE_FIELDS)):
        raise ProductionManifestBuilderError("production runtime manifest invalid")
    interpreter = value.get("interpreter")
    if not isinstance(interpreter, Mapping):
        raise ProductionManifestBuilderError("runtime interpreter missing")
    _exact_keys(interpreter, {"path", "realpath", "bytes", "sha256", "version"},
                "runtime interpreter")
    path = Path(str(interpreter["path"]))
    if not path.is_absolute():
        raise ProductionManifestBuilderError("runtime interpreter path not absolute")
    try:
        real = path.resolve(strict=True)
    except OSError as exc:
        raise ProductionManifestBuilderError("runtime interpreter missing") from exc
    real = _regular_file(real, "runtime interpreter realpath")
    if (str(real) != interpreter["realpath"] or real.is_symlink()
            or real.stat().st_size != interpreter["bytes"]
            or sha256_file(real) != interpreter["sha256"]
            or not isinstance(interpreter["version"], str)
            or not interpreter["version"]):
        raise ProductionManifestBuilderError("runtime interpreter drift")
    dependencies = value.get("runtime_dependency_files")
    if not isinstance(dependencies, list) or not dependencies:
        raise ProductionManifestBuilderError("runtime dependency closure empty")
    checked = []
    seen = set()
    for index, row in enumerate(dependencies):
        if not isinstance(row, Mapping):
            raise ProductionManifestBuilderError("runtime dependency row malformed")
        _exact_keys(row, {"path", "bytes", "sha256"},
                    f"runtime dependency {index}")
        dep = _regular_file(Path(str(row["path"])), f"runtime dependency {index}")
        if (str(dep) in seen or dep.stat().st_size != row["bytes"]
                or sha256_file(dep) != row["sha256"]):
            raise ProductionManifestBuilderError("runtime dependency drift")
        seen.add(str(dep)); checked.append(dict(row))
    runner = _verify_file_row(value["runner_source"], "runner_source")
    production_wrapper = _verify_file_row(
        value["production_wrapper_cli"], "production_wrapper_cli")
    validator = _verify_file_row(value["validator_source"], "validator_source")
    controlled = value.get("controlled_sys_path")
    if (not isinstance(controlled, list) or not controlled
            or any(not isinstance(item, str) or not Path(item).is_absolute()
                   or Path(item).is_symlink() or not Path(item).is_dir()
                   for item in controlled)):
        raise ProductionManifestBuilderError("controlled sys.path invalid")
    environment = value.get("environment")
    if (not isinstance(environment, Mapping)
            or environment.get("PYTHONNOUSERSITE") != "1"
            or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
            or environment.get("PYTHONPYCACHEPREFIX")
                != "/proc/v16-b716-fixed4-no-pyc"
            or environment.get("CUDA_CACHE_DISABLE") != "1"
            or "PYTHONPATH" in environment or "PYTHONHOME" in environment):
        raise ProductionManifestBuilderError("runtime environment polluted")
    return {"interpreter": dict(interpreter),
        "runtime_dependency_files": checked,
        "controlled_sys_path": list(controlled), "environment": dict(environment),
        "interpreter_path": str(path), "interpreter_sha256": interpreter["sha256"],
        "runner_source_path": runner["path"],
        "runner_source_sha256": runner["sha256"],
        "production_wrapper_path": production_wrapper["path"],
        "production_wrapper_sha256": production_wrapper["sha256"],
        "validator_source_path": validator["path"],
        "wrapper_source_sha256": validator["sha256"]}


def build_active_production_execution_manifest(*, task: Mapping[str, Any],
        production_input_manifest_path: Path,
        production_input_manifest: Mapping[str, Any], preflight: Mapping[str, Any],
        output_root: Path, upstream_results: Mapping[str, Mapping[str, Any]],
        runtime_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the production input document to an audited runtime closure."""
    stage, task_root = _validate_task_preflight(task, preflight, output_root)
    parents = _validate_parent_results(task, output_root, upstream_results)
    input_path = Path(production_input_manifest_path)
    transaction_root = input_path.parent
    expected_transactions_root = (task_root / "control" /
                                  PRODUCTION_MANIFEST_TRANSACTION_DIRECTORY)
    if (input_path.name != "production_input_manifest.json"
            or transaction_root.parent != expected_transactions_root
            or _safe_task_id(transaction_root.name) != transaction_root.name
            or not _payload_valid(production_input_manifest)):
        raise ProductionManifestBuilderError("production input canonical path invalid")
    if (production_input_manifest.get("schema") != INPUT_MANIFEST_SCHEMA
            or production_input_manifest.get("task_id") != task["task_id"]
            or production_input_manifest.get("task_payload_sha256")
                != task["payload_sha256"]
            or production_input_manifest.get("stage") != stage):
        raise ProductionManifestBuilderError("production input semantic drift")
    encoded_input = _manifest_bytes(production_input_manifest)
    input_sha256 = __import__("hashlib").sha256(encoded_input).hexdigest()
    if input_path.exists():
        observed_input = _load_json(
            input_path, input_sha256, "production input manifest")
        if observed_input != dict(production_input_manifest):
            raise ProductionManifestBuilderError("production input semantic drift")
    runtime = _validate_runtime_manifest(runtime_manifest)
    dependencies = runtime["runtime_dependency_files"]
    value = {"schema": ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "stage": stage, "production_input_manifest_path": str(input_path),
        "production_input_manifest_sha256": input_sha256,
        "production_input_manifest_payload_sha256":
            production_input_manifest["payload_sha256"],
        "interpreter": runtime["interpreter"],
        "runtime_dependency_files": dependencies,
        "runtime_dependency_closure_sha256": stable_json_sha256(dependencies),
        "controlled_sys_path": runtime["controlled_sys_path"],
        "environment": runtime["environment"],
        "parent_result_payload_sha256s":
            [parent["payload_sha256"] for parent, _ in parents],
        "runner_source_sha256": runtime["runner_source_sha256"],
        "wrapper_source_sha256": runtime["wrapper_source_sha256"],
        **POLICY_FALSE_FIELDS}
    return _sealed(value)


def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _write_manifest(root: Path, path: Path, value: Mapping[str, Any],
                    role: str) -> dict[str, Any]:
    encoded = _manifest_bytes(value)
    try:
        row, state = create_only_bytes_beneath(
            root, path, encoded, create_parents=True, resume_identical=False)
    except Fixed4SubprocessContractError as exc:
        raise ProductionManifestBuilderError(f"{role} create-only failed") from exc
    if state != "created":
        raise ProductionManifestBuilderError(f"{role} was not newly created")
    observed = _load_json(path, row["sha256"], role)
    if observed != dict(value):
        raise ProductionManifestBuilderError(f"{role} post-write mismatch")
    return {"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"],
            "payload_sha256": value["payload_sha256"]}


def _validate_committed_manifest_row(row: Any, transaction_root: Path,
                                     filename: str, role: str) -> dict[str, Any]:
    if (not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256", "payload_sha256"}):
        raise ProductionManifestBuilderError(f"{role} commit row malformed")
    path = Path(str(row.get("path", "")))
    if path != transaction_root / filename:
        raise ProductionManifestBuilderError(f"{role} commit path drift")
    value = _load_json(path, str(row.get("sha256", "")), role)
    if (type(row.get("bytes")) is not int or row["bytes"] < 1
            or path.stat().st_size != row["bytes"]
            or not _payload_valid(value)
            or value.get("payload_sha256") != row.get("payload_sha256")):
        raise ProductionManifestBuilderError(f"{role} commit binding drift")
    return value


def load_committed_production_manifest_transaction(*, commit_path: Path,
        task: Mapping[str, Any], preflight: Mapping[str, Any],
        output_root: Path) -> dict[str, Any]:
    """Load a manifest pair only when the final commit marker is complete.

    Interrupted attempts remain audit-visible in their unique transaction
    directory, but they are not consumer-visible because they have no valid
    commit marker.  A later create-only attempt can therefore retry safely.
    """
    _stage, task_root = _validate_task_preflight(task, preflight, output_root)
    commit_path = _regular_file(commit_path, "production transaction commit")
    transaction_root = commit_path.parent
    expected_transactions_root = (task_root / "control" /
                                  PRODUCTION_MANIFEST_TRANSACTION_DIRECTORY)
    if (commit_path.name != "COMMITTED.json"
            or transaction_root.parent != expected_transactions_root
            or _safe_task_id(transaction_root.name) != transaction_root.name):
        raise ProductionManifestBuilderError("production transaction path invalid")
    commit = _load_json(commit_path, None, "production transaction commit")
    required = {"schema", "transaction_id", "task_id", "task_payload_sha256",
        "stage", "created_at", "transaction_state",
        "production_input_manifest", "production_execution_manifest",
        *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    if (set(commit) != required or not _payload_valid(commit)
            or commit.get("schema")
                != PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA
            or commit.get("transaction_id") != transaction_root.name
            or commit.get("task_id") != task.get("task_id")
            or commit.get("task_payload_sha256") != task.get("payload_sha256")
            or commit.get("stage") != task.get("stage")
            or not isinstance(commit.get("created_at"), str)
            or not commit["created_at"].endswith("Z")
            or commit.get("transaction_state") != "COMMITTED"
            or any(commit.get(key) is not False for key in POLICY_FALSE_FIELDS)):
        raise ProductionManifestBuilderError("production transaction commit invalid")
    input_value = _validate_committed_manifest_row(
        commit["production_input_manifest"], transaction_root,
        "production_input_manifest.json", "production input manifest")
    execution_value = _validate_committed_manifest_row(
        commit["production_execution_manifest"], transaction_root,
        "production_execution_manifest.json", "production execution manifest")
    if (execution_value.get("production_input_manifest_path")
            != commit["production_input_manifest"]["path"]
            or execution_value.get("production_input_manifest_sha256")
                != commit["production_input_manifest"]["sha256"]
            or execution_value.get("production_input_manifest_payload_sha256")
                != commit["production_input_manifest"]["payload_sha256"]):
        raise ProductionManifestBuilderError(
            "production transaction cross-manifest binding drift")
    return {"commit": commit, "production_input_manifest": input_value,
            "production_execution_manifest": execution_value}


def materialize_production_manifests(*, task: Mapping[str, Any],
        preflight: Mapping[str, Any], output_root: Path,
        upstream_results: Mapping[str, Mapping[str, Any]],
        production_assets_manifest: Mapping[str, Any],
        runtime_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Publish one create-only pair, made consumer-visible by a final marker."""
    _stage, task_root = _validate_task_preflight(task, preflight, output_root)
    runtime = _validate_runtime_manifest(runtime_manifest)
    input_value = build_production_input_manifest(
        task=task, preflight=preflight, output_root=output_root,
        upstream_results=upstream_results,
        production_assets_manifest=production_assets_manifest)
    transaction_id = "tx-" + uuid.uuid4().hex
    transaction_root = (task_root / "control" /
        PRODUCTION_MANIFEST_TRANSACTION_DIRECTORY / transaction_id)
    input_path = transaction_root / "production_input_manifest.json"
    execution_value = build_active_production_execution_manifest(
        task=task, production_input_manifest_path=input_path,
        production_input_manifest=input_value, preflight=preflight,
        output_root=output_root, upstream_results=upstream_results,
        runtime_manifest=runtime_manifest)
    execution_path = transaction_root / "production_execution_manifest.json"
    input_row = _write_manifest(Path(output_root).resolve(), input_path,
                                input_value, "production input manifest")
    execution_row = _write_manifest(Path(output_root).resolve(), execution_path,
        execution_value, "production execution manifest")
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    commit_value = _sealed({
        "schema": PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA,
        "transaction_id": transaction_id, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "created_at": created_at, "transaction_state": "COMMITTED",
        "production_input_manifest": input_row,
        "production_execution_manifest": execution_row,
        **POLICY_FALSE_FIELDS})
    commit_path = transaction_root / "COMMITTED.json"
    commit_row = _write_manifest(Path(output_root).resolve(), commit_path,
                                 commit_value, "production transaction commit")
    loaded = load_committed_production_manifest_transaction(
        commit_path=commit_path, task=task, preflight=preflight,
        output_root=output_root)
    if (loaded["production_input_manifest"] != input_value
            or loaded["production_execution_manifest"] != execution_value):
        raise ProductionManifestBuilderError(
            "committed production transaction post-write mismatch")
    return _sealed({"schema": PRODUCTION_MANIFEST_BUILD_RECEIPT_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "stage": task["stage"], "transaction_id": transaction_id,
        "created_at": created_at, "transaction_state": "COMMITTED",
        "receipt_path": commit_row["path"],
        "receipt_sha256": commit_row["sha256"],
        "receipt_payload_sha256": commit_row["payload_sha256"],
        "transaction_commit": commit_row,
        "production_input_manifest": input_row,
        "production_execution_manifest": execution_row,
        "parent_result_payload_sha256s":
            list(execution_value["parent_result_payload_sha256s"]),
        "authorization_binding": {
            "transaction_id": transaction_id,
            "transaction_commit_path": commit_row["path"],
            "transaction_commit_sha256": commit_row["sha256"],
            "transaction_commit_payload_sha256": commit_row["payload_sha256"],
            "production_input_manifest_path": input_row["path"],
            "production_input_manifest_sha256": input_row["sha256"],
            "production_input_manifest_payload_sha256": input_row["payload_sha256"],
            "execution_manifest_path": execution_row["path"],
            "execution_manifest_sha256": execution_row["sha256"],
            "execution_manifest_payload_sha256": execution_row["payload_sha256"],
            "production_interpreter_path": runtime["interpreter_path"],
            "production_interpreter_sha256": runtime["interpreter_sha256"],
            "production_wrapper_path": runtime["production_wrapper_path"],
            "production_wrapper_sha256": runtime["production_wrapper_sha256"],
            "validator_source_path": runtime["validator_source_path"],
            "validator_source_sha256": runtime["wrapper_source_sha256"],
            "runner_source_path": runtime["runner_source_path"],
            "runner_source_sha256": runtime["runner_source_sha256"],
            "parent_result_payload_sha256s":
                list(execution_value["parent_result_payload_sha256s"]),
        },
        "create_only": True, "task_runtime_paths_trusted": False,
        **POLICY_FALSE_FIELDS})
