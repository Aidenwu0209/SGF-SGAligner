"""Fail-closed active wrapper for the frozen fixed4 production adapters.

This module is the only bridge from a hash-bound production input manifest to
an operational RESULT-v5 candidate.  It never signs an authorization and it
never releases ``result.json``.  The parent boundary must independently
validate the emitted candidate and the validation receipt before a create-only
release.

The implementation intentionally keeps process success separate from a normal
scientific gate failure (return code 2).  An abnormal process failure produces
only an attempt receipt; it cannot manufacture an operational result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from safety.v13_dual_solver_runtime import (
    sha256_file, stable_json_sha256, transform_distance, validate_se3,
)
from safety.v15_safe_pose_cluster import ROTATION_MAX_DEG, TRANSLATION_MAX_M
from safety.v16_b716_fixed4_execution_pilot import (
    ACTIVE_ADAPTER_VALIDATION_SCHEMA, EVIDENCE_RECEIPT_SCHEMA,
    HYPOTHESIS_OUTCOME_SCHEMA, POLICY_FALSE_FIELDS, RESULT_SCHEMA,
    SENTINEL_ATTEMPT_SCHEMA, SOLVER_ATTEMPT_SCHEMA,
    TYPED_FAILURE_REPLAY_SCHEMA, validate_runner_result,
)
from safety.v16_b716_fixed4_orchestrator_contract import FIXED_PAIR_ORDER
from safety.v16_b716_fixed4_production_adapters import (
    ADAPTER_CONTRACT_SCHEMA, NORMAL_GATE_FAIL_RETURN_CODE,
    ProductionAdapterError, build_stage_adapter_contract,
    classify_stage_returncode, expand_verified_candidate_slots,
    finalize_v15_from_slot_results, load_bound_input_manifest,
    materialize_contract_create_only, run_frozen_aggregate_gate,
    run_frozen_pair_gate,
)
from safety.v16_b716_fixed4_stage_runners import (
    AGGREGATE_GATE_SCHEMA, HYPOTHESIS_GATE_SCHEMA, PAIR_GATE_SCHEMA,
    POLICY_FALSE as CORE_POLICY_FALSE, build_pair_gate_result,
    classify_finite_consensus, validate_pair_gate_result,
)


ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-active-production-execution-manifest-v1")
ACTIVE_PRODUCTION_ATTEMPT_SCHEMA = (
    "v16-b716-fixed4-active-production-attempt-v1")
ACTIVE_PRODUCTION_WRAPPER_RESULT_SCHEMA = (
    "v16-b716-fixed4-active-production-wrapper-result-v1")
NORMAL_GATE_STATUS = "normal_gate_failed"
PROCESS_FAILURE_STATUS = "typed_process_failure"
CONTROLLED_PYTHON_LAUNCHER = r'''\
import json, runpy, sys
paths = json.loads(sys.argv[1])
script = sys.argv[2]
sys.path[:] = paths
sys.argv[:] = [script, *sys.argv[3:]]
runpy.run_path(script, run_name="__main__")
'''


class ActiveProductionWrapperError(RuntimeError):
    """A production task could not produce a fully validated RESULT-v5."""


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def _payload_valid(value: Mapping[str, Any]) -> bool:
    return value.get("payload_sha256") == stable_json_sha256({
        key: item for key, item in value.items() if key != "payload_sha256"})


def _exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise ActiveProductionWrapperError(f"{role} keys mismatch")


def _json(path: Path, expected_sha256: str | None, role: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ActiveProductionWrapperError(f"{role} path/SHA invalid")
    try:
        # Parse the same bytes whose digest is checked.  A separate hash and
        # read_text call leaves a substitution window between the two reads.
        encoded = path.read_bytes()
        if (expected_sha256 is not None
                and hashlib.sha256(encoded).hexdigest() != expected_sha256):
            raise ActiveProductionWrapperError(f"{role} path/SHA invalid")
        value = json.loads(encoded)
    except Exception as exc:
        raise ActiveProductionWrapperError(f"{role} JSON invalid") from exc
    if not isinstance(value, dict):
        raise ActiveProductionWrapperError(f"{role} must be an object")
    return value


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    path = Path(path).resolve(); root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ActiveProductionWrapperError("artifact escapes output root") from exc
    if path.is_symlink() or not path.is_file():
        raise ActiveProductionWrapperError("artifact must be a regular file")
    return {"path": str(relative), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _write_json(root: Path, path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = materialize_contract_create_only(root, path, value)
    return {"path": str(Path(path).resolve().relative_to(Path(root).resolve())),
            "bytes": receipt["bytes"], "sha256": receipt["sha256"]}


def _verify_execution_manifest(value: Mapping[str, Any], *, task: Mapping[str, Any],
                               path: Path) -> dict[str, Any]:
    required = {"schema", "task_id", "task_payload_sha256", "stage",
        "production_input_manifest_path", "production_input_manifest_sha256",
        "production_input_manifest_payload_sha256", "interpreter",
        "runtime_dependency_files", "runtime_dependency_closure_sha256",
        "controlled_sys_path", "environment", "parent_result_payload_sha256s",
        "runner_source_sha256", "wrapper_source_sha256",
        *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    _exact_keys(value, required, "active production execution manifest")
    interpreter = value.get("interpreter")
    if not isinstance(interpreter, Mapping):
        raise ActiveProductionWrapperError("production interpreter missing")
    _exact_keys(interpreter, {"path", "realpath", "bytes", "sha256", "version"},
                "production interpreter")
    executable = Path(str(interpreter.get("path", "")))
    real = Path(str(interpreter.get("realpath", "")))
    try:
        observed_real = executable.resolve(strict=True)
    except OSError as exc:
        raise ActiveProductionWrapperError("production interpreter missing") from exc
    if (not executable.is_absolute() or not real.is_absolute()
            or observed_real != real or real.is_symlink() or not real.is_file()
            or real.stat().st_size != interpreter.get("bytes")
            or sha256_file(real) != interpreter.get("sha256")
            or not isinstance(interpreter.get("version"), str)
            or not interpreter["version"]):
        raise ActiveProductionWrapperError("production interpreter identity drift")
    rows = value.get("runtime_dependency_files")
    if (not isinstance(rows, list) or not rows
            or stable_json_sha256(rows)
                != value.get("runtime_dependency_closure_sha256")):
        raise ActiveProductionWrapperError("runtime dependency closure invalid")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ActiveProductionWrapperError("runtime dependency row invalid")
        _exact_keys(row, {"path", "bytes", "sha256"},
                    f"runtime dependency {index}")
        dependency = Path(str(row.get("path", "")))
        if (not dependency.is_absolute() or dependency.is_symlink()
                or not dependency.is_file() or str(dependency) in seen
                or dependency.stat().st_size != row.get("bytes")
                or sha256_file(dependency) != row.get("sha256")):
            raise ActiveProductionWrapperError("runtime dependency SHA drift")
        seen.add(str(dependency))
    controlled = value.get("controlled_sys_path")
    if (not isinstance(controlled, list) or not controlled
            or any(not isinstance(item, str) or not Path(item).is_absolute()
                   for item in controlled)):
        raise ActiveProductionWrapperError("controlled sys.path invalid")
    environment = value.get("environment")
    if (not isinstance(environment, Mapping)
            or environment.get("PYTHONNOUSERSITE") != "1"
            or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
            or environment.get("PYTHONPYCACHEPREFIX")
                != "/proc/v16-b716-fixed4-no-pyc"
            or environment.get("CUDA_CACHE_DISABLE") != "1"
            or "PYTHONPATH" in environment or "PYTHONHOME" in environment
            or set(environment) - {"PATH", "LANG", "LC_ALL", "PYTHONNOUSERSITE",
                                     "PYTHONDONTWRITEBYTECODE",
                                     "PYTHONPYCACHEPREFIX", "CUDA_CACHE_DISABLE",
                                     "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES",
                                     "CUBLAS_WORKSPACE_CONFIG"}):
        raise ActiveProductionWrapperError("production environment is not sealed")
    if (not _payload_valid(value)
            or value.get("schema") != ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA
            or value.get("task_id") != task.get("task_id")
            or value.get("task_payload_sha256") != task.get("payload_sha256")
            or value.get("stage") != task.get("stage")
            or any(value.get(key) is not False for key in POLICY_FALSE_FIELDS)):
        raise ActiveProductionWrapperError("production execution binding mismatch")
    return dict(value)


def load_active_production_execution_manifest(
        path: Path, expected_sha256: str, *, task: Mapping[str, Any]) -> dict[str, Any]:
    value = _json(Path(path), expected_sha256, "production execution manifest")
    return _verify_execution_manifest(value, task=task, path=Path(path))


def _run_command(command: Mapping[str, Any], *, environment: Mapping[str, str],
                 controlled_sys_path: Sequence[str], python_path: str,
                 cwd: Path) -> dict[str, Any]:
    if (command.get("shell") is not False
            or command.get("environment_inherited") is not False
            or not isinstance(command.get("argv"), list)
            or not command["argv"]):
        raise ActiveProductionWrapperError("adapter command contract malformed")
    original = list(command["argv"])
    launched = original
    if (len(original) >= 2 and original[0] == python_path
            and original[1].endswith(".py")):
        launched = [python_path, "-I", "-S", "-B", "-X",
            "pycache_prefix=/proc/v16-b716-fixed4-no-pyc", "-c",
            CONTROLLED_PYTHON_LAUNCHER,
            json.dumps(list(controlled_sys_path), separators=(",", ":")),
            original[1], *original[2:]]
    scratch = cwd / "scratch"
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_environment = dict(environment)
    runtime_environment.update({"TMPDIR": str(scratch), "TMP": str(scratch),
        "TEMP": str(scratch), "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "JOBLIB_TEMP_FOLDER": str(scratch / "joblib"),
        "JOBLIB_MULTIPROCESSING": "0", "LOKY_MAX_CPU_COUNT": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    completed = subprocess.run(launched, cwd=cwd, env=runtime_environment,
                               capture_output=True, check=False)
    normal = command.get("normal_return_codes")
    if not isinstance(normal, list) or completed.returncode not in normal:
        status = PROCESS_FAILURE_STATUS
    else:
        status = (NORMAL_GATE_STATUS
                  if completed.returncode == NORMAL_GATE_FAIL_RETURN_CODE
                  else "stage_succeeded")
    return {"argv_sha256": stable_json_sha256(original),
        "launched_argv_sha256": stable_json_sha256(launched),
        "runtime_environment_sha256": stable_json_sha256(runtime_environment),
        "returncode": completed.returncode, "classification": status,
        "stdout_sha256": stable_json_sha256(list(completed.stdout)),
        "stderr_sha256": stable_json_sha256(list(completed.stderr))}


def _evidence_receipts(task: Mapping[str, Any], root: Path,
                       generated_slots: set[int] | None = None) -> list[dict[str, Any]]:
    task_root = root / "tasks" / str(task["task_id"])
    rows = []
    for node in task.get("evidence_nodes", ()):
        status = "consumed"
        stage = str(node.get("node_id", "")).split(".", 1)[0]
        if generated_slots is not None and stage in {"solver", "strict"}:
            try:
                slot = int(str(node["node_id"]).split(".")[4])
            except Exception as exc:
                raise ActiveProductionWrapperError(
                    "pilot evidence slot identity malformed") from exc
            status = "consumed" if slot in generated_slots else "typed_not_generated"
        document = _sealed({"schema": EVIDENCE_RECEIPT_SCHEMA,
            "operational_task_id": task["task_id"],
            "operational_task_payload_sha256": task["payload_sha256"],
            "node_id": node["node_id"],
            "node_payload_sha256": node["node_payload_sha256"],
            "status": status, **POLICY_FALSE_FIELDS})
        path = task_root / "evidence" / f"{int(node['ordinal']):05d}.json"
        row = _write_json(root, path, document)
        rows.append({**row, "node_id": node["node_id"],
                     "node_payload_sha256": node["node_payload_sha256"]})
    return rows


def _attempt_document(task: Mapping[str, Any], schema: str,
                      identity: Mapping[str, Any], *, status: str,
                      transform: Any, failure_type: str | None) -> dict[str, Any]:
    if status == "succeeded":
        validate_se3(transform)
        failure_type = None
    elif status == "typed_failure":
        transform = None
        if not isinstance(failure_type, str) or not failure_type:
            raise ActiveProductionWrapperError("typed failure lacks class")
    else:
        raise ActiveProductionWrapperError("attempt status invalid")
    return _sealed({"schema": schema, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "status": status,
        "transform": transform, "failure_type": failure_type,
        **identity, **POLICY_FALSE_FIELDS})


def _color_result(task: Mapping[str, Any], manifest: Mapping[str, Any],
                  root: Path, contract: Mapping[str, Any],
                  command_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(command_rows) != 2 or any(row["classification"] != "stage_succeeded"
                                     for row in command_rows):
        raise ActiveProductionWrapperError("ColorPCR process did not complete")
    outputs = {row["role"]: root / "tasks" / task["task_id"] / row["path"]
               for row in manifest["outputs"]}
    try:
        import numpy as np
        with np.load(outputs["sentinel_cache"], allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"].item()))
    except Exception as exc:
        raise ActiveProductionWrapperError("ColorPCR sentinel cache invalid") from exc
    paths = meta.get("sentinel_artifact_path", {})
    shas = meta.get("sentinel_artifact_sha256", {})
    attempts = []
    for sentinel in ("identity", "proper_nonzero"):
        artifact = Path(str(paths.get(sentinel, "")))
        if (not artifact.is_file() or artifact.is_symlink()
                or sha256_file(artifact) != shas.get(sentinel)):
            raise ActiveProductionWrapperError("ColorPCR sentinel evidence drift")
        with np.load(artifact, allow_pickle=False) as data:
            sentinel_meta = json.loads(str(data["meta_json"].item()))
        document = _attempt_document(task, SENTINEL_ATTEMPT_SCHEMA,
            {"sentinel": sentinel, "direction": task["direction"]},
            status="succeeded", transform=sentinel_meta.get("sentinel_transform"),
            failure_type=None)
        path = root / "tasks" / task["task_id"] / "attempts" / f"{sentinel}.json"
        row = _write_json(root, path, document)
        attempts.append({**row, "sentinel": sentinel,
                         "direction": task["direction"], "status": "succeeded"})
    exact3 = outputs["exact_three_cache"]
    if not exact3.is_file() or exact3.is_symlink():
        raise ActiveProductionWrapperError("exact-three output absent")
    artifact_document = _sealed({"schema": "v16-b716-fixed4-artifact-binding-v1",
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "role": "exact_three_cache", "source_path": str(exact3),
        "source_sha256": sha256_file(exact3), **POLICY_FALSE_FIELDS})
    artifact_path = root / "tasks" / task["task_id"] / "artifacts" / "exact3.json"
    artifact = _write_json(root, artifact_path, artifact_document)
    evidence = _evidence_receipts(task, root)
    return _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "succeeded", "typed_failure": None, **POLICY_FALSE_FIELDS,
        "output_artifacts": [artifact], "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "sentinel_attempts": attempts,
        "sentinel_attempt_closure_sha256": stable_json_sha256(attempts),
        "exact_three_cache_artifact_sha256": sha256_file(exact3)})


def _worker_attempts(task: Mapping[str, Any], root: Path, slot_root: Path,
                     slot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = []
    for path in sorted((slot_root / f"slot_{slot:02d}" / "raw" / "workers").glob("*.json")):
        source_rows.append(_json(path, None, "V13 worker evidence"))
    expected = [(solver, direction, repeat) for solver in ("pointdsc", "pygcransac")
                for direction in ("forward", "reverse") for repeat in range(5)]
    by_key = {(row.get("solver"), row.get("direction"), row.get("repeat")): row
              for row in source_rows}
    if set(by_key) != set(expected):
        raise ActiveProductionWrapperError("generated pilot slot is not exact20")
    attempts = []
    for solver, direction, repeat in expected:
        source = by_key[(solver, direction, repeat)]
        succeeded = source.get("status") == "ok"
        identity = {"candidate_slot": slot, "solver": solver,
                    "direction": direction, "repeat": repeat}
        document = _attempt_document(task, SOLVER_ATTEMPT_SCHEMA, identity,
            status="succeeded" if succeeded else "typed_failure",
            transform=source.get("transform"),
            failure_type=(None if succeeded else
                          str(source.get("failure_type") or "V13_SOLVER_FAILURE")))
        path = (root / "tasks" / task["task_id"] / "attempts" /
                f"c{slot}-{solver}-{direction}-{repeat}.json")
        row = _write_json(root, path, document)
        attempts.append({**row, **identity, "status": document["status"]})
    return attempts, source_rows


def _typed_failure_replay(task: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    rows = []
    for index in task.get("typed_failure_member_candidate_indices", ()):
        document = _attempt_document(task, TYPED_FAILURE_REPLAY_SCHEMA,
            {"candidate_index": index}, status="typed_failure", transform=None,
            failure_type="FROZEN_TYPED_FAILURE_REPLAY")
        path = root / "tasks" / task["task_id"] / "typed_failures" / f"{index}.json"
        row = _write_json(root, path, document)
        rows.append({**row, "candidate_index": index})
    return rows


def _first_valid_transform(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    for row in rows:
        if row.get("status") == "ok" and row.get("transform") is not None:
            return validate_se3(row["transform"]).tolist()
    raise ActiveProductionWrapperError("generated slot has no finite transform")


def _pilot_result(task: Mapping[str, Any], manifest: Mapping[str, Any], root: Path,
                  expansion: Mapping[str, Any], v15: Mapping[str, Any],
                  worker_sources: Mapping[int, Sequence[Mapping[str, Any]]],
                  attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decision = v15.get("v15_decision")
    if not isinstance(decision, Mapping):
        raise ActiveProductionWrapperError("V15 decision absent")
    selected = decision.get("selected_candidate_index")
    observation = v15.get("gate_observation")
    if not isinstance(observation, Mapping):
        raise ActiveProductionWrapperError("V15 gate observation absent")
    slots = []
    generated: set[int] = set()
    for planned in expansion["slots"]:
        slot = planned["candidate_slot"]
        if planned["status"] != "generated":
            slots.append({"candidate_slot": slot, "status": "typed_not_generated",
                "solver_rows_executed": 0,
                "failure_type": "CANDIDATE_SLOT_NOT_GENERATED"})
            continue
        transform = _first_valid_transform(worker_sources[slot])
        generated.add(slot)
        slots.append({"candidate_slot": slot, "status": "generated",
            "solver_rows_executed": 20, "transform": transform,
            "failure_type": None,
            "safe_vote": bool(decision.get("accepted") and selected == slot)})
    safe = [row for row in slots if row.get("safe_vote") is True]
    eligible = bool(task.get("safe_pose_vote_eligible"))
    success = (eligible and len(safe) == 1
               and observation.get("gate_status") == "PASS")
    if not eligible:
        gate, failure, transform = ("ABSTAIN",
            "TYPED_MEMBER_HYPOTHESIS_ABSTENTION", None)
        rotation = translation = measurement_source = None
    elif success:
        gate, failure = "PASS", None
        transform = validate_se3(decision.get("selected_transform")).tolist()
        rotation = observation.get("measured_rotation_deg")
        translation = observation.get("measured_translation_m")
        measurement_source = observation.get(
            "measurement_source_payload_sha256")
    elif eligible and observation.get("gate_status") == "FAIL":
        gate, failure, transform = ("FAIL",
            "FINITE_CONSENSUS_INCOMPATIBILITY", None)
        rotation = observation.get("measured_rotation_deg")
        translation = observation.get("measured_translation_m")
        measurement_source = observation.get(
            "measurement_source_payload_sha256")
    elif observation.get("gate_status") == "ABSTAIN":
        gate, failure, transform = ("ABSTAIN",
            "HYPOTHESIS_GATE_MEASUREMENT_SEMANTICS_UNBOUND", None)
        rotation = translation = measurement_source = None
    else:
        gate, failure, transform = "ABSTAIN", str(
            decision.get("reason") or "NO_UNIQUE_SAFE_POSE_CLUSTER").upper(), None
        rotation = translation = measurement_source = None
    v15_node = next((row for row in task.get("evidence_nodes", ())
                     if str(row.get("node_id", "")).startswith("v15.")), None)
    if v15_node is None:
        raise ActiveProductionWrapperError("pilot V15 evidence node missing")
    binding_keys = ("measurement_source_file_sha256",
        "measurement_candidate_slot", "measurement_candidate_set_sha256",
        "measurement_slot_results_payload_sha256",
        "measurement_v15_decision_sha256")
    measurement_binding = {key: observation.get(key) for key in binding_keys}
    outcome_document = _sealed({"schema": HYPOTHESIS_OUTCOME_SCHEMA,
        "hypothesis_task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "source_v15_node_id": v15_node["node_id"],
        "source_v15_node_payload_sha256": v15_node["node_payload_sha256"],
        "gate_status": gate, "failure_class": failure,
        "safe_transform": transform, "measured_rotation_deg": rotation,
        "measured_translation_m": translation,
        "measurement_source_payload_sha256": measurement_source,
        **measurement_binding,
        **POLICY_FALSE_FIELDS})
    outcome_path = root / "tasks" / task["task_id"] / "outcomes" / "v15.json"
    outcome_row = _write_json(root, outcome_path, outcome_document)
    outcome_receipt = {**outcome_row, "node_id": v15_node["node_id"],
        "node_payload_sha256": v15_node["node_payload_sha256"]}
    outcome = {"hypothesis_task_id": task["task_id"], "gate_status": gate,
        "failure_class": failure, "safe_transform": transform,
        "measured_rotation_deg": rotation,
        "measured_translation_m": translation,
        "measurement_source_payload_sha256": measurement_source,
        **measurement_binding,
        "source_result_payload_sha256": outcome_document["payload_sha256"]}
    typed = _typed_failure_replay(task, root)
    evidence = _evidence_receipts(task, root, generated)
    return _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "succeeded" if success else "typed_failure",
        "typed_failure": (None if success else {"type": failure, "transform": None}),
        **POLICY_FALSE_FIELDS, "output_artifacts": [],
        "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "candidate_slots": slots,
        "candidate_slot_closure_sha256": stable_json_sha256(slots),
        "solver_rows_executed": len(attempts), "solver_attempts": list(attempts),
        "solver_attempt_closure_sha256": stable_json_sha256(list(attempts)),
        "typed_failure_replay": typed,
        "typed_failure_replay_closure_sha256": stable_json_sha256(typed),
        "hypothesis_outcome": outcome,
        "hypothesis_outcome_receipt": outcome_receipt})


def _cluster_parent_passes(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [row for row in parents if row.get("gate_status") == "PASS"]
    if any(row.get("gate_status") == "FAIL" for row in parents):
        raise ActiveProductionWrapperError(
            "unbound finite FAIL measurement is forbidden")
    transforms = [validate_se3(row["safe_transform"]) for row in safe]
    if not transforms:
        return {"accepted": False, "reason": "no_safe_hypothesis"}
    compatible = True
    for left in range(len(transforms)):
        for right in range(left + 1, len(transforms)):
            rotation, translation = transform_distance(
                transforms[left], transforms[right])
            compatible &= (rotation <= ROTATION_MAX_DEG
                           and translation <= TRANSLATION_MAX_M)
    if not compatible:
        return {"accepted": False,
                "reason": "ambiguous_multiple_safe_hypothesis_pose_clusters"}
    scores = []
    for index, transform in enumerate(transforms):
        distances = [transform_distance(transform, other) for other in transforms]
        score = sum(r / ROTATION_MAX_DEG + t / TRANSLATION_MAX_M
                    for r, t in distances)
        scores.append((score, index))
    selected = transforms[min(scores)[1]].tolist()
    return {"accepted": True, "reason": "unique_safe_hypothesis_pose_cluster",
            "selected_transform": selected}


def _pair_result(task: Mapping[str, Any], root: Path,
                 contract: Mapping[str, Any]) -> dict[str, Any]:
    parents = contract.get("verified_parent_outcomes")
    if not isinstance(parents, list):
        raise ActiveProductionWrapperError("pair verified parent outcomes missing")
    gate_rows = []
    for row in parents:
        if row.get("gate_status") in {"PASS", "FAIL"}:
            gate_rows.append(classify_finite_consensus(
                hypothesis_task_id=row["hypothesis_task_id"],
                rotation_deg=row["measured_rotation_deg"],
                translation_m=row["measured_translation_m"],
                transform=(row["safe_transform"] if row["gate_status"] == "PASS"
                           else [[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])))
    eligible = [row["hypothesis_task_id"] for row in parents
                if row.get("gate_status") in {"PASS", "FAIL"}]
    abstained = [row["hypothesis_task_id"] for row in parents
                 if row.get("gate_status") == "ABSTAIN"]
    payload = {"task_id": task["task_id"], "pair_id": task["pair_id"],
        "replayed_hypothesis_task_ids": [row["hypothesis_task_id"] for row in parents],
        "eligible_hypothesis_task_ids": eligible,
        "typed_abstention_hypothesis_task_ids": abstained,
        "hypothesis_gate_results": gate_rows,
        "cluster_decision": _cluster_parent_passes(parents),
        "known_bad": bool(task["known_bad"])}
    operational = run_frozen_pair_gate(payload, verified_parent_outcomes=parents)
    evidence = _evidence_receipts(task, root)
    return _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        **operational, **POLICY_FALSE_FIELDS, "output_artifacts": [],
        "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence)})


def _operational_pair_to_core(task_id: str, pair_id: str,
                              value: Mapping[str, Any]) -> dict[str, Any]:
    passed = value.get("status") == "succeeded"
    failure = None if passed else value.get("typed_failure", {}).get("type")
    row = _sealed({"schema": PAIR_GATE_SCHEMA,
        "stage": "v16_pair_hypothesis_cluster", "task_id": task_id,
        "pair_id": pair_id, "execution_status": "succeeded",
        "gate_status": "PASS" if passed else "FAIL",
        "failure_class": failure, "decision": value.get("decision"),
        "transform": value.get("safe_cluster_transform"),
        "downstream_authorized": passed, "known_bad": pair_id == FIXED_PAIR_ORDER[-1],
        "replayed_hypothesis_task_ids": value.get("replayed_hypothesis_task_ids"),
        "eligible_hypothesis_task_ids": [
            *value.get("safe_vote_hypothesis_task_ids", ()),
            *value.get("gate_failed_hypothesis_task_ids", ())],
        "safe_vote_hypothesis_task_ids": value.get("safe_vote_hypothesis_task_ids"),
        "gate_failed_hypothesis_task_ids": value.get("gate_failed_hypothesis_task_ids"),
        "typed_abstention_hypothesis_task_ids":
            value.get("typed_abstention_hypothesis_task_ids"),
        "cluster_reason": value.get("decision"), **CORE_POLICY_FALSE})
    return validate_pair_gate_result(row)


def _aggregate_result(task: Mapping[str, Any], manifest: Mapping[str, Any],
                      root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    parent_files = {row["role"]: Path(row["path"])
                    for row in manifest["file_inputs"]}
    parent_values = [_json(parent_files[f"parent_result_{index}"], None,
                           f"aggregate parent {index}")
                     for index in range(4)]
    cores = [_operational_pair_to_core(task_id, pair_id, value)
             for task_id, pair_id, value in zip(
                 task["upstream_task_ids"], FIXED_PAIR_ORDER, parent_values)]
    payload = {"task_id": task["task_id"], "pair_results": cores,
        "expected_pair_ids": list(FIXED_PAIR_ORDER),
        "known_bad_pair_id": FIXED_PAIR_ORDER[-1]}
    verified = contract.get("verified_parent_outcomes")
    operational = run_frozen_aggregate_gate(
        payload, verified_parent_outcomes=verified)
    outcomes = [{"task_id": parent_id, "status": value["status"],
        "decision": value["decision"],
        "safe_cluster_transform": value["safe_cluster_transform"],
        "source_result_payload_sha256": value["payload_sha256"]}
        for parent_id, value in zip(task["upstream_task_ids"], parent_values)]
    evidence = _evidence_receipts(task, root)
    guard_sha = stable_json_sha256({"task_id": task["task_id"],
        "parent_result_payload_sha256s": [row["payload_sha256"]
                                          for row in parent_values]})
    return _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        **operational, **POLICY_FALSE_FIELDS, "output_artifacts": [],
        "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "pair_outcomes": outcomes,
        "pair_outcome_closure_sha256": stable_json_sha256(outcomes),
        "guard_audit_receipt_sha256": guard_sha})


def build_active_adapter_validation_receipt(
        *, task: Mapping[str, Any], candidate: Mapping[str, Any],
        candidate_path: Path, contract: Mapping[str, Any], contract_path: Path,
        execution_manifest: Mapping[str, Any], execution_manifest_path: Path,
        production_attempt: Mapping[str, Any], production_attempt_path: Path,
        output_rows: Sequence[Mapping[str, Any]], parent_payloads: Sequence[str],
        validator_source_sha256: str, runner_source_sha256: str,
        ) -> dict[str, Any]:
    validate_runner_result(task, candidate, Path(candidate_path).parents[3])
    stage_semantics = {"normal_gate_return_code": NORMAL_GATE_FAIL_RETURN_CODE,
        "normal_gate_distinct_from_process_failure": True,
        "parent_results_derived_not_task_reported": True,
        "operational_result_create_only_release_required": True}
    value = {"schema": ACTIVE_ADAPTER_VALIDATION_SCHEMA, "status": "PASS",
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "stage": task["stage"], "candidate_path": str(Path(candidate_path).resolve()),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_payload_sha256": candidate["payload_sha256"],
        "operational_result_schema": RESULT_SCHEMA,
        "parent_result_payload_sha256s": list(parent_payloads),
        "production_adapter_contract_path": str(Path(contract_path).resolve()),
        "production_adapter_contract_sha256": sha256_file(contract_path),
        "production_adapter_contract_payload_sha256": contract["payload_sha256"],
        "production_input_manifest_sha256":
            execution_manifest["production_input_manifest_sha256"],
        "production_input_manifest_payload_sha256":
            execution_manifest["production_input_manifest_payload_sha256"],
        "execution_manifest_path": str(Path(execution_manifest_path).resolve()),
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "execution_manifest_payload_sha256": execution_manifest["payload_sha256"],
        "production_attempt_path": str(Path(production_attempt_path).resolve()),
        "production_attempt_sha256": sha256_file(production_attempt_path),
        "production_attempt_payload_sha256": production_attempt["payload_sha256"],
        "output_artifact_rows": list(output_rows),
        "output_artifact_closure_sha256": stable_json_sha256(list(output_rows)),
        "validator_source_sha256": validator_source_sha256,
        "runner_source_sha256": runner_source_sha256,
        "stage_semantics": stage_semantics,
        "stage_semantics_sha256": stable_json_sha256(stage_semantics),
        **POLICY_FALSE_FIELDS}
    return _sealed(value)


def execute_active_production_wrapper(
        *, task: Mapping[str, Any], execution_manifest_path: Path,
        execution_manifest_sha256: str, output_root: Path,
        validator_source_sha256: str, runner_source_sha256: str,
        ) -> dict[str, Any]:
    """Execute one authorized task and emit candidate+validation, never result.json."""
    root = Path(output_root).resolve()
    task_root = root / "tasks" / str(task["task_id"])
    execution = load_active_production_execution_manifest(
        execution_manifest_path, execution_manifest_sha256, task=task)
    if (execution.get("wrapper_source_sha256") != validator_source_sha256
            or execution.get("runner_source_sha256") != runner_source_sha256):
        raise ActiveProductionWrapperError(
            "execution manifest source pin mismatch")
    input_path = Path(execution["production_input_manifest_path"])
    manifest = load_bound_input_manifest(
        input_path, execution["production_input_manifest_sha256"], task, root)
    if manifest.get("payload_sha256") != \
            execution["production_input_manifest_payload_sha256"]:
        raise ActiveProductionWrapperError("production input payload drift")
    contract = build_stage_adapter_contract(task, manifest, root)
    if contract.get("schema") != ADAPTER_CONTRACT_SCHEMA:
        raise ActiveProductionWrapperError("adapter contract schema drift")
    contract_path = task_root / "active" / "production_adapter_contract.json"
    materialize_contract_create_only(root, contract_path, contract)
    command_rows: list[dict[str, Any]] = []
    attempt_status = "succeeded"
    candidate: dict[str, Any]
    if task["stage"] == "colorpcr_direction":
        for command in contract["commands"]:
            row = _run_command(command, environment=execution["environment"],
                               controlled_sys_path=execution["controlled_sys_path"],
                               python_path=execution["interpreter"]["path"],
                               cwd=task_root)
            command_rows.append(row)
            if row["classification"] == PROCESS_FAILURE_STATUS:
                attempt_status = "typed_process_failure"; break
        if attempt_status != "succeeded":
            raise ActiveProductionWrapperError("ColorPCR process failure")
        candidate = _color_result(task, manifest, root, contract, command_rows)
    elif task["stage"] == "bidirectional_multi_solver_pilot":
        for command in contract["commands"]:
            row = _run_command(command, environment=execution["environment"],
                               controlled_sys_path=execution["controlled_sys_path"],
                               python_path=execution["interpreter"]["path"],
                               cwd=task_root); command_rows.append(row)
            if row["classification"] == PROCESS_FAILURE_STATUS:
                raise ActiveProductionWrapperError("V14 builder process failure")
        outputs = {row["role"]: task_root / row["path"] for row in manifest["outputs"]}
        expansion = expand_verified_candidate_slots(
            task, manifest, sha256_file(outputs["candidate_set"]), root)
        expansion_path = task_root / "active" / "slot_expansion.json"
        materialize_contract_create_only(root, expansion_path, expansion)
        slot_rows = []; attempts = []; worker_sources = {}
        for planned in expansion["slots"]:
            slot = planned["candidate_slot"]
            if planned["status"] != "generated":
                slot_rows.append({"candidate_slot": slot,
                    "status": "typed_not_generated", "returncode": None,
                    "raw_summary_path": None, "raw_summary_sha256": None,
                    "strict_summary_path": None, "strict_summary_sha256": None})
                continue
            row = _run_command(planned["command"],
                environment=execution["environment"],
                controlled_sys_path=execution["controlled_sys_path"],
                python_path=execution["interpreter"]["path"], cwd=task_root)
            command_rows.append(row)
            slot_dir = outputs["slot_root"] / f"slot_{slot:02d}"
            raw = slot_dir / "raw_summary.json"; strict = slot_dir / "summary.json"
            classification = classify_stage_returncode(
                row["returncode"], summary_exists=raw.is_file() and strict.is_file())
            if classification == PROCESS_FAILURE_STATUS:
                raise ActiveProductionWrapperError("strict runner process failure")
            slot_rows.append({"candidate_slot": slot, "status": classification,
                "returncode": row["returncode"], "raw_summary_path": str(raw.resolve()),
                "raw_summary_sha256": sha256_file(raw),
                "strict_summary_path": str(strict.resolve()),
                "strict_summary_sha256": sha256_file(strict)})
            rows, source = _worker_attempts(task, root, outputs["slot_root"], slot)
            attempts.extend(rows); worker_sources[slot] = source
        slot_document = _sealed({
            "schema": "v16-b716-fixed4-production-slot-results-v1",
            "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
            "slot_expansion_payload_sha256": expansion["payload_sha256"],
            "candidate_set_sha256": expansion["candidate_set_sha256"],
            "slots": slot_rows,
            "execution_authorized": False, "gt_allowed": False,
            "identity_fallback_allowed": False, "threshold_change_allowed": False,
            "result_selection_allowed": False, "reconstruction_authorized": False,
            "refusion_allowed": False})
        slot_path = task_root / "active" / "slot_results.json"
        materialize_contract_create_only(root, slot_path, slot_document)
        v15 = finalize_v15_from_slot_results(
            task, manifest, expansion, slot_path, sha256_file(slot_path), root)
        v15_path = task_root / "production" / "v15_outcome.json"
        materialize_contract_create_only(root, v15_path, v15)
        candidate = _pilot_result(task, manifest, root, expansion, v15,
                                  worker_sources, attempts)
    elif task["stage"] == "v16_pair_hypothesis_cluster":
        candidate = _pair_result(task, root, contract)
    elif task["stage"] == "fixed4_aggregate":
        candidate = _aggregate_result(task, manifest, root, contract)
    else:
        raise ActiveProductionWrapperError("unsupported production stage")
    parent_payloads = execution["parent_result_payload_sha256s"]
    parent_results = {}
    for parent, payload in zip(task.get("upstream_task_ids", ()), parent_payloads):
        parent_path = root / "tasks" / parent / "result.json"
        parent_results[parent] = _json(parent_path, None, "verified parent result")
        if parent_results[parent].get("payload_sha256") != payload:
            raise ActiveProductionWrapperError("parent result payload drift")
    validate_runner_result(task, candidate, root,
                           upstream_results=parent_results or None)
    candidate_path = task_root / "active" / "operational_result_candidate.json"
    materialize_contract_create_only(root, candidate_path, candidate)
    output_rows = []
    for path in sorted(p for p in (task_root / "production").rglob("*")
                       if p.is_file() and not p.is_symlink()):
        output_rows.append(_file_row(path, root))
    attempt = _sealed({"schema": ACTIVE_PRODUCTION_ATTEMPT_SCHEMA,
        "status": candidate["status"], "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "execution_manifest_payload_sha256": execution["payload_sha256"],
        "production_adapter_contract_sha256": sha256_file(contract_path),
        "production_adapter_contract_payload_sha256": contract["payload_sha256"],
        "parent_result_payload_sha256s": list(parent_payloads),
        "command_results": command_rows,
        "command_result_closure_sha256": stable_json_sha256(command_rows),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_payload_sha256": candidate["payload_sha256"],
        "normal_gate_failure_observed": any(
            row["classification"] == NORMAL_GATE_STATUS for row in command_rows),
        "process_failure_observed": False, **POLICY_FALSE_FIELDS})
    attempt_path = task_root / "active" / "production_attempt.json"
    materialize_contract_create_only(root, attempt_path, attempt)
    validation = build_active_adapter_validation_receipt(
        task=task, candidate=candidate, candidate_path=candidate_path,
        contract=contract, contract_path=contract_path,
        execution_manifest=execution,
        execution_manifest_path=execution_manifest_path,
        production_attempt=attempt, production_attempt_path=attempt_path,
        output_rows=output_rows, parent_payloads=parent_payloads,
        validator_source_sha256=validator_source_sha256,
        runner_source_sha256=runner_source_sha256)
    validation_path = task_root / "active" / "adapter_validation.json"
    materialize_contract_create_only(root, validation_path, validation)
    return _sealed({"schema": ACTIVE_PRODUCTION_WRAPPER_RESULT_SCHEMA,
        "task_id": task["task_id"], "stage": task["stage"],
        "candidate_path": str(candidate_path), "candidate_sha256": sha256_file(candidate_path),
        "candidate_payload_sha256": candidate["payload_sha256"],
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "validation_payload_sha256": validation["payload_sha256"],
        "attempt_path": str(attempt_path), "attempt_sha256": sha256_file(attempt_path),
        "attempt_payload_sha256": attempt["payload_sha256"],
        "operational_result_released": False,
        "process_failure_observed": False, **POLICY_FALSE_FIELDS})
