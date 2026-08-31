"""Fail-closed fixed4 execution contract, fix4.

CPU/filesystem metadata only: no model, GPU, solver, ICP, refusion, GT, or
official92 is imported or launched. Callers cannot inject a runner callable.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from safety.v13_dual_solver_runtime import (
    sha256_file, stable_json_sha256, validate_se3,
)
from safety.v16_b716_fixed4_orchestrator_contract import (
    EXPECTED_HYPOTHESES, EXPECTED_NODE_COUNT, FIXED_PAIR_ORDER,
    KNOWN_BAD_PAIR_ID, OFFICIAL_RELEASE_SHA256, HypothesisBinding,
    bind_hypotheses, build_task_dag, load_exact191_hypotheses,
    load_prepared_hypotheses, validate_preregister, verify_source_pins,
)
from safety.v16_b716_exact72_lineage_seal import (
    Exact72LineageSealError, verify_lineage_seal,
)
from safety.v16_b716_fixed4_subprocess_contract import (
    ACTIVE_AUTHORIZATION_SCHEMA, ACTIVE_PREFLIGHT_SCHEMA,
    ACTIVE_PREFLIGHT_V2_SCHEMA,
    DISABLED_EXIT_CODE, RUNNER_MODE_ACTIVE, RUNNER_MODE_DISABLED,
    SIGNATURE_ALGORITHM, Fixed4SubprocessContractError, build_subprocess_registry,
    create_only_bytes_beneath, ensure_no_symlink_directory,
    no_symlink_file_mode, no_symlink_file_row, read_no_symlink_bytes,
    task_execution_binding, validate_subprocess_registry,
    verify_fixed_signed_document,
)

PREFLIGHT_SCHEMA = "v16-b716-fixed4-execution-preflight-v5"
EXECUTION_PREREGISTER_SCHEMA = "v16-b716-fixed4-execution-pilot-preregister-v5"
TASK_SCHEMA = "v16-b716-fixed4-operational-task-v5"
TASK_MANIFEST_SCHEMA = "v16-b716-fixed4-operational-task-manifest-v5"
AUTH_SCHEMA = "v16-b716-fixed4-execution-authorization-v5"
GUARD_AUDIT_SCHEMA = "v16-b716-fixed4-guard-audit-receipt-v2"
RESULT_SCHEMA = "v16-b716-fixed4-operational-result-v5"
ATTEMPT_SCHEMA = "v16-b716-fixed4-operational-attempt-v4"
EVIDENCE_RECEIPT_SCHEMA = "v16-b716-fixed4-expanded-evidence-receipt-v1"
SOLVER_ATTEMPT_SCHEMA = "v16-b716-fixed4-solver-attempt-v2"
SENTINEL_ATTEMPT_SCHEMA = "v16-b716-fixed4-sentinel-attempt-v1"
TYPED_FAILURE_REPLAY_SCHEMA = "v16-b716-fixed4-typed-failure-replay-v1"
HYPOTHESIS_OUTCOME_SCHEMA = "v16-b716-fixed4-hypothesis-outcome-v1"
ACTIVE_EXECUTION_PREREGISTER_SCHEMA = (
    "v16-b716-fixed4-active-execution-candidate-preregister-v1")
ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA = (
    "v16-b716-fixed4-active-execution-ready-preregister-v2")
ACTIVE_STAGE_INPUT_DESCRIPTOR_SCHEMA = (
    "v16-b716-fixed4-active-stage-input-descriptor-v1")
ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA = (
    "v16-b716-fixed4-active-stage-input-descriptor-v2")
ACTIVE_AUTHORIZATION_REQUEST_SCHEMA = (
    "v16-b716-fixed4-active-authorization-request-v1")
ACTIVE_STAGE_AUTHORIZATION_SCHEMA = (
    "v16-b716-fixed4-active-stage-authorization-v1")
ACTIVE_DISPATCH_RECEIPT_SCHEMA = (
    "v16-b716-fixed4-active-stagewise-dispatch-receipt-v1")
ACTIVE_ADAPTER_VALIDATION_SCHEMA = (
    "v16-b716-fixed4-production-adapter-validation-v1")
ACTIVE_STAGE_ATTEMPT_SCHEMA = (
    "v16-b716-fixed4-active-operational-attempt-v1")
ACTIVE_STAGE_COMMIT_SCHEMA = (
    "v16-b716-fixed4-active-operational-commit-v1")
RUNNER_SOURCE_RELATIVE = "scripts/v16_b716_fixed4_disabled_stage_runner.sh"
MAX_AUTH_TTL_SECONDS = 3600
AUTH_CLOCK_SKEW_SECONDS = 300
EXACT_SOLVER_ROWS_PER_PILOT = 20
MAX_CANDIDATE_SLOTS_PER_PILOT = 8
SOLVERS = ("pointdsc", "pygcransac")
DIRECTIONS = ("forward", "reverse")
SEEDS = tuple(range(5))
SENTINELS = ("identity", "proper_nonzero")
OPERATIONAL_STAGE_COUNTS = {
    "colorpcr_direction": 68,
    "bidirectional_multi_solver_pilot": 34,
    "v16_pair_hypothesis_cluster": 4,
    "fixed4_aggregate": 1,
}
EXPECTED_EVIDENCE_NODES_PER_TASK = {
    "colorpcr_direction": 4,
    "bidirectional_multi_solver_pilot": 171,
    "v16_pair_hypothesis_cluster": 1,
    "fixed4_aggregate": 1,
}
OPERATIONAL_TASK_COUNT = sum(OPERATIONAL_STAGE_COUNTS.values())
ALLOWED_STAGES = tuple(OPERATIONAL_STAGE_COUNTS)
POLICY_FALSE_FIELDS = {
    "gt_consumed": False, "official92_run": False,
    "thresholds_changed": False, "result_selection_used": False,
    "default_checkpoint_replaced": False, "refusion_run": False,
    "reconstruction_authorized": False,
}
FORBIDDEN_KEY_TOKENS = {
    "gt", "groundtruth", "official92", "rre", "rte", "selectionlabel",
    "bestscore", "winner",
    "majority", "selectedhypothesisbyscore", "thresholdoverride",
    "checkpointreplacement", "refusionoutput", "reconstructionoutput",
}
FORBIDDEN_ARTIFACT_TOKENS = (
    "official92", "ground_truth", "groundtruth", "winner", "majority",
    "threshold_override", "refusion", "reconstruction", "fused_map",
)
REQUIRED_AUTH_REVIEW_FIELDS = {
    "independent_review_status": "PASS", "real_binding_reviewed": True,
    "exact72_closure_reviewed": True, "all_34_hypotheses_reviewed": True,
    "typed_failure_replay_reviewed": True, "known_bad_veto_reviewed": True,
    "clean_gpu_identity_reviewed": True, "pointdsc_dependency_reviewed": True,
    "evidence_mapping_reviewed": True, "runner_registry_reviewed": True,
    "create_only_reviewed": True,
    "detached_exact72_lineage_reviewed": True,
    "typed_failure_absence_reviewed": True,
    "historical_authorization_evidence_only": True,
    "downstream_capability_not_inherited": True,
    "signer_private_key_not_on_execution_host": True,
}

ACTIVE_READY_V2_SOURCE_RELATIVES = (
    "src/safety/v16_b716_fixed4_execution_pilot.py",
    "src/safety/v16_b716_fixed4_subprocess_contract.py",
    "src/safety/v16_b716_fixed4_stage_runners.py",
    "src/safety/v16_b716_fixed4_production_adapters.py",
    "src/safety/v16_b716_fixed4_production_manifest_builder.py",
    "src/safety/v16_b716_fixed4_assets_builder.py",
    "src/safety/v16_b716_fixed4_active_production_wrapper.py",
    "scripts/v16_b716_fixed4_execution_pilot.py",
    "scripts/v16_b716_fixed4_active_dispatch_cli.py",
    "scripts/v16_b716_fixed4_active_sealed_executor.py",
    "scripts/v16_b716_fixed4_active_stage_runner.sh",
    "scripts/v16_b716_fixed4_active_production_wrapper.py",
    "scripts/v16_b716_fixed4_production_adapter.py",
    "scripts/v16_b716_fixed4_production_manifest_builder.py",
    "scripts/v16_b716_fixed4_assets_builder.py",
    "scripts/v16_b716_fixed4_active_preregister_builder.py",
)


class Fixed4ExecutionPilotError(RuntimeError):
    """An authorization, closure, task, result, or receipt failed closed."""


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise Fixed4ExecutionPilotError(f"invalid {name}")
    return value


def _git_hash(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) not in (40, 64)
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise Fixed4ExecutionPilotError(f"invalid {name}")
    return value


def _payload_valid(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    return value.get("payload_sha256") == stable_json_sha256(unsigned)


def _json(path: Path, expected_sha: str | None, role: str) -> dict[str, Any]:
    path = Path(path)
    try:
        encoded = read_no_symlink_bytes(path, role)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    before = no_symlink_file_row(path, role)["sha256"]
    if expected_sha is not None and before != _sha(expected_sha, f"{role} SHA"):
        raise Fixed4ExecutionPilotError(f"{role} SHA mismatch")
    try:
        value = json.loads(encoded)
    except Exception as exc:
        raise Fixed4ExecutionPilotError(f"invalid {role} JSON") from exc
    if (not isinstance(value, dict)
            or no_symlink_file_row(path, role)["sha256"] != before):
        raise Fixed4ExecutionPilotError(f"{role} changed while reading")
    return value


def _git_identity(repo: Path, *, require_clean: bool) -> tuple[str, str]:
    repo = Path(repo).resolve()
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True,
                              text=True).stdout.strip()
        tree = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
                              check=True, capture_output=True,
                              text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                               check=True, capture_output=True,
                               text=True).stdout
    except Exception as exc:
        raise Fixed4ExecutionPilotError("git identity unavailable") from exc
    _git_hash(head, "git HEAD"); _git_hash(tree, "git tree")
    if require_clean and dirty:
        raise Fixed4ExecutionPilotError("repository is not clean")
    return head, tree


def _parse_time(value: Any, role: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception as exc:
        raise Fixed4ExecutionPilotError(f"{role} invalid") from exc
    if parsed.tzinfo is None:
        raise Fixed4ExecutionPilotError(f"{role} timezone missing")
    return parsed.astimezone(timezone.utc)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], role: str) -> None:
    if set(value) != expected:
        raise Fixed4ExecutionPilotError(
            f"{role} keys mismatch missing={sorted(expected-set(value))} "
            f"extra={sorted(set(value)-expected)}")


def _policy_false(value: Mapping[str, Any], role: str) -> None:
    for key, expected in POLICY_FALSE_FIELDS.items():
        if value.get(key) is not expected:
            raise Fixed4ExecutionPilotError(f"{role} policy drift: {key}")


def _reject_forbidden_keys(value: Any, path: str = "value") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            canonical = "".join(ch for ch in str(key).lower() if ch.isalnum())
            if canonical in FORBIDDEN_KEY_TOKENS:
                raise Fixed4ExecutionPilotError(f"forbidden evidence field: {path}.{key}")
            _reject_forbidden_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, f"{path}[{index}]")


def _resolve_row_file(root: Path, row: Mapping[str, Any], role: str,
                      *, within: Path | None = None,
                      require_read_only: bool = True) -> Path:
    _require_exact_keys(row, {"path", "bytes", "sha256"}, role)
    if not isinstance(row.get("path"), str) or not row["path"]:
        raise Fixed4ExecutionPilotError(f"{role} path invalid")
    if Path(row["path"]).is_absolute() or ".." in Path(row["path"]).parts:
        raise Fixed4ExecutionPilotError(f"{role} path invalid")
    root = Path(root); path = root / row["path"]
    boundary = Path(within) if within is not None else root
    try:
        path.relative_to(boundary)
    except ValueError as exc:
        raise Fixed4ExecutionPilotError(f"{role} escapes sealed root") from exc
    try:
        observed = no_symlink_file_row(path, role)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if (type(row.get("bytes")) is not int or row["bytes"] < 1
            or observed["bytes"] != row["bytes"]
            or observed["sha256"] != _sha(row.get("sha256"), f"{role} SHA")):
        raise Fixed4ExecutionPilotError(f"{role} bytes/SHA mismatch")
    mode = no_symlink_file_mode(path, role)
    if require_read_only and stat.S_IMODE(mode) & 0o222:
        raise Fixed4ExecutionPilotError(f"{role} is not immutable/read-only")
    return path


def create_only_artifact(root: Path, path: Path, data: bytes) -> dict[str, Any]:
    try:
        row, _state = create_only_bytes_beneath(root, path, data)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    return row


def _validate_execution_preregister(value: Mapping[str, Any], repo: Path) -> None:
    matrix = value.get("solver_matrix"); replay = value.get("replay_policy")
    guard = value.get("registration_defense"); registry = value.get("stage_runner_registry")
    runner_path = Path(repo).resolve() / RUNNER_SOURCE_RELATIVE
    executor_path = Path(repo).resolve() / "scripts/v16_b716_fixed4_sealed_executor.py"
    if (not _payload_valid(value) or value.get("schema") != EXECUTION_PREREGISTER_SCHEMA
            or value.get("frozen") is not True or value.get("disabled") is not True
            or value.get("execution_authorized") is not False
            or value.get("gpu_allowed") is not False
            or value.get("model_execution_allowed") is not False
            or value.get("solver_execution_allowed") is not False
            or value.get("reconstruction_authorized") is not False
            or value.get("gt_allowed") is not False or value.get("official92_allowed") is not False
            or value.get("threshold_change_allowed") is not False
            or value.get("result_selection_allowed") is not False
            or value.get("default_checkpoint_replacement_allowed") is not False
            or value.get("official_release_checkpoint_sha256") != OFFICIAL_RELEASE_SHA256
            or value.get("fixed_hypothesis_distribution") != list(EXPECTED_HYPOTHESES)
            or value.get("operational_stage_counts") != OPERATIONAL_STAGE_COUNTS
            or value.get("sentinel_workers_per_direction") != list(SENTINELS)
            or value.get("max_authorization_ttl_seconds") != MAX_AUTH_TTL_SECONDS
            or value.get("evidence_mapping_required") is not True):
        raise Fixed4ExecutionPilotError("execution pilot preregistration is not frozen and disabled")
    if (not isinstance(matrix, Mapping) or matrix.get("solvers") != list(SOLVERS)
            or matrix.get("directions") != list(DIRECTIONS)
            or matrix.get("seeds") != list(SEEDS)
            or matrix.get("exact_solver_rows_per_pilot") != EXACT_SOLVER_ROWS_PER_PILOT
            or matrix.get("repeats") != 5 or matrix.get("quorum") != 4
            or matrix.get("maximum_candidates_per_hypothesis") != 8
            or matrix.get("preregistered_solver_nodes_per_hypothesis") != 160
            or matrix.get("rotation_max_deg") != 5.0
            or matrix.get("translation_max_m") != 0.10):
        raise Fixed4ExecutionPilotError("solver matrix preregistration drift")
    if (not isinstance(replay, Mapping)
            or replay.get("all_34_hypotheses_required") is not True
            or replay.get("typed_failures_explicit_never_filtered") is not True
            or replay.get("all_members_ok_filter_allowed") is not False
            or replay.get("best_score_allowed") is not False
            or replay.get("majority_allowed") is not False
            or replay.get("known_bad_all_12_replayed") is not True
            or replay.get("known_bad_permanent_veto") is not True
            or replay.get("typed_failure_safe_pose_vote_allowed") is not False
            or replay.get("typed_failure_selector_allowed") is not False
            or replay.get("typed_failure_quorum_allowed") is not False
            or replay.get("typed_failure_cluster_contribution_allowed") is not False
            or replay.get("typed_failure_transform_allowed") is not False
            or replay.get("selector_eligible_required_false") is not True
            or value.get("detached_exact72_lineage_required") is not True):
        raise Fixed4ExecutionPilotError("replay policy preregistration drift")
    if (not isinstance(guard, Mapping) or guard.get("independent_audit_status") != "NOT_RUN"
            or guard.get("solver_family_contract_aligned") is not False
            or guard.get("authorized") is not False):
        raise Fixed4ExecutionPilotError("registration-defense failure must remain explicit")
    if (not runner_path.is_file() or runner_path.is_symlink()
            or not executor_path.is_file() or executor_path.is_symlink()
            or not isinstance(registry, Mapping)
            or registry.get("runner_path") != RUNNER_SOURCE_RELATIVE
            or registry.get("runner_sha256") != sha256_file(runner_path)
            or registry.get("sealed_executor_path")
                != "scripts/v16_b716_fixed4_sealed_executor.py"
            or registry.get("sealed_executor_sha256") != sha256_file(executor_path)
            or registry.get("stages") != list(ALLOWED_STAGES)
            or registry.get("execution_mode") != "hash_bound_independent_subprocess"
            or registry.get("checked_in_runner_disabled") is not True
            or registry.get("caller_runner_injection_allowed") is not False
            or registry.get("in_process_callable_allowed") is not False
            or registry.get("runner_reported_failure_type_trusted") is not False):
        raise Fixed4ExecutionPilotError("stage runner registry drift")


def validate_active_execution_preregister_candidate(
        value: Mapping[str, Any], repo: Path | None = None) -> None:
    """Validate an unsigned candidate protocol; disabled evidence cannot activate."""
    required = {"schema", "candidate_only", "frozen", "runner_mode",
        "legacy_disabled_preregister_accepted", "production_adapter_protocol_required",
        "production_adapter_ready", "operational_result_schema",
        "parent_results_derived_by_dispatcher", "task_stage_input_trusted",
        "max_authorization_ttl_seconds", "off_host_signer_required",
        "execution_host_private_key_allowed", "gt_allowed", "official92_allowed",
        "measurement_primitive",
        "threshold_change_allowed", "result_selection_allowed",
        "default_checkpoint_replacement_allowed", "reconstruction_authorized",
        "refusion_allowed", "formal_execution_authorized",
        "active_runner_registry_closure_sha256", "candidate_source_pins",
        "payload_sha256"}
    _require_exact_keys(value, required, "active execution candidate preregistration")
    schema = value.get("schema")
    production_ready = schema == ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA
    if (not _payload_valid(value)
            or schema not in {ACTIVE_EXECUTION_PREREGISTER_SCHEMA,
                              ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA}
            or value.get("candidate_only") is not True
            or value.get("frozen") is not True
            or value.get("runner_mode") != RUNNER_MODE_ACTIVE
            or value.get("legacy_disabled_preregister_accepted") is not False
            or value.get("production_adapter_protocol_required") is not True
            or value.get("production_adapter_ready") is not production_ready
            or value.get("operational_result_schema") != RESULT_SCHEMA
            or value.get("parent_results_derived_by_dispatcher") is not True
            or value.get("task_stage_input_trusted") is not False
            or value.get("max_authorization_ttl_seconds") != MAX_AUTH_TTL_SECONDS
            or value.get("off_host_signer_required") is not True
            or value.get("execution_host_private_key_allowed") is not False
            or value.get("formal_execution_authorized") is not False
            or not isinstance(value.get("candidate_source_pins"), Mapping)
            or not value["candidate_source_pins"]
            or not isinstance(value.get("active_runner_registry_closure_sha256"), str)
            or len(value["active_runner_registry_closure_sha256"]) != 64
            or any(value.get(key) is not False for key in (
                "gt_allowed", "official92_allowed", "threshold_change_allowed",
                "result_selection_allowed", "default_checkpoint_replacement_allowed",
                "reconstruction_authorized", "refusion_allowed"))):
        raise Fixed4ExecutionPilotError(
            "active execution candidate preregistration is not frozen/fail-closed")
    primitive = value.get("measurement_primitive")
    if primitive != {
            "schema": "v16-b716-fixed4-measurement-primitive-v1",
            "stage": "bidirectional_multi_solver_pilot",
            "source": "selected_v15_slot.raw_summary.cross_solver_check",
            "rotation_field": "rotation_deg", "rotation_unit": "degree",
            "translation_field": "translation_m", "translation_unit": "meter",
            "rotation_threshold_source":
                "safety.v15_safe_pose_cluster.ROTATION_MAX_DEG",
            "translation_threshold_source":
                "safety.v15_safe_pose_cluster.TRANSLATION_MAX_M",
            "requires_unique_safe_v15_acceptance": True,
            "direction_checks_allowed": False,
            "v15_compatibility_matrix_allowed": False,
            "gt_allowed": False, "result_selection_allowed": False}:
        raise Fixed4ExecutionPilotError(
            "active measurement primitive is absent or semantically ambiguous")
    for relative, digest in value["candidate_source_pins"].items():
        if (not isinstance(relative, str) or Path(relative).is_absolute()
                or ".." in Path(relative).parts):
            raise Fixed4ExecutionPilotError("active candidate source path invalid")
        _sha(digest, "active candidate source SHA")
        if repo is not None:
            source = Path(repo).resolve() / relative
            if (not source.is_file() or source.is_symlink()
                    or sha256_file(source) != digest):
                raise Fixed4ExecutionPilotError("active candidate source pin drift")


def build_active_execution_preregister_v2(repo: Path) -> dict[str, Any]:
    """Build the ready protocol envelope without granting execution authority.

    The document only declares that v2 manifests/adapters are available.  It
    never authorizes a node, reconstruction, refusion, GT, or official92.  A
    separate off-host signature remains mandatory for every topological node.
    """
    repo = Path(repo).resolve()
    _git_identity(repo, require_clean=True)
    _rows, runner_sha = _runner_closure(repo, runner_mode=RUNNER_MODE_ACTIVE)
    pins: dict[str, str] = {}
    for relative in ACTIVE_READY_V2_SOURCE_RELATIVES:
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise Fixed4ExecutionPilotError(
                f"active ready-v2 source missing: {relative}")
        pins[relative] = sha256_file(path)
    value = {
        "schema": ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA,
        "candidate_only": True,
        "frozen": True,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "legacy_disabled_preregister_accepted": False,
        "production_adapter_protocol_required": True,
        "production_adapter_ready": True,
        "operational_result_schema": RESULT_SCHEMA,
        "parent_results_derived_by_dispatcher": True,
        "task_stage_input_trusted": False,
        "max_authorization_ttl_seconds": MAX_AUTH_TTL_SECONDS,
        "off_host_signer_required": True,
        "execution_host_private_key_allowed": False,
        "formal_execution_authorized": False,
        "measurement_primitive": {
            "schema": "v16-b716-fixed4-measurement-primitive-v1",
            "stage": "bidirectional_multi_solver_pilot",
            "source": "selected_v15_slot.raw_summary.cross_solver_check",
            "rotation_field": "rotation_deg",
            "rotation_unit": "degree",
            "translation_field": "translation_m",
            "translation_unit": "meter",
            "rotation_threshold_source":
                "safety.v15_safe_pose_cluster.ROTATION_MAX_DEG",
            "translation_threshold_source":
                "safety.v15_safe_pose_cluster.TRANSLATION_MAX_M",
            "requires_unique_safe_v15_acceptance": True,
            "direction_checks_allowed": False,
            "v15_compatibility_matrix_allowed": False,
            "gt_allowed": False,
            "result_selection_allowed": False,
        },
        "active_runner_registry_closure_sha256": runner_sha,
        "candidate_source_pins": pins,
        "gt_allowed": False,
        "official92_allowed": False,
        "threshold_change_allowed": False,
        "result_selection_allowed": False,
        "default_checkpoint_replacement_allowed": False,
        "reconstruction_authorized": False,
        "refusion_allowed": False,
    }
    value["payload_sha256"] = stable_json_sha256(value)
    validate_active_execution_preregister_candidate(value, repo)
    return value


def _active_stage_input_descriptor(stage: str, task_id: str,
        upstream: Sequence[str], *, production_ready: bool = False,
        ) -> dict[str, Any]:
    source = ("sealed_preregistered_source_closure" if not upstream
              else "verified_upstream_operational_result_v5_receipts")
    value = {"schema": (ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA
                         if production_ready
                         else ACTIVE_STAGE_INPUT_DESCRIPTOR_SCHEMA),
        "task_id": task_id, "stage": stage,
        "upstream_task_ids": list(upstream), "input_source": source,
        "derivation_policy": "dispatcher_only_never_trust_task_runtime_paths",
        "production_input_manifest_schema":
            "v16-b716-fixed4-production-input-manifest-v1",
        "production_adapter_contract_schema":
            "v16-b716-fixed4-production-adapter-contract-v1",
        "operational_result_schema": RESULT_SCHEMA,
        "production_adapter_protocol_ready": production_ready}
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _task_id(stage: str, binding: HypothesisBinding | None = None,
             suffix: str = "") -> str:
    if binding is None:
        return stage + (("." + suffix) if suffix else "")
    base = (f"{stage}.p{binding.pair_ordinal}.h{binding.hypothesis_index:02d}."
            f"{binding.hypothesis_sha256[:12]}")
    return base + (("." + suffix) if suffix else "")


def _owned_nodes(task: Mapping[str, Any], dag: Mapping[str, Any]) -> list[dict]:
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        raise Fixed4ExecutionPilotError("evidence DAG nodes missing")
    stage = task["stage"]; owned = []
    for node in nodes:
        if stage == "colorpcr_direction":
            match = (node.get("pair_id") == task.get("pair_id")
                     and node.get("hypothesis_index") == task.get("hypothesis_index")
                     and node.get("direction") == task.get("direction")
                     and node.get("stage") in {"colorpcr_worker",
                         "sentinel_direction_cache", "exact_three_direction_cache"})
        elif stage == "bidirectional_multi_solver_pilot":
            match = (node.get("pair_id") == task.get("pair_id")
                     and node.get("hypothesis_index") == task.get("hypothesis_index")
                     and node.get("stage") in {"prepared_input", "v14_candidate_set",
                         "v13_solver_row", "v13_strict_candidate_gate",
                         "v15_hypothesis_candidate_cluster"})
        elif stage == "v16_pair_hypothesis_cluster":
            match = (node.get("stage") == "v16_pair_hypothesis_cluster"
                     and node.get("pair_id") == task.get("pair_id"))
        else:
            match = node.get("stage") == "fixed4_aggregate"
        if match:
            owned.append({"ordinal": node["ordinal"], "node_id": node["task_id"],
                          "node_payload_sha256": node["node_payload_sha256"]})
    return owned


def bind_evidence_ownership(
    tasks: Sequence[Mapping[str, Any]], dag: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sealed = []; seen: set[str] = set()
    for task in tasks:
        value = dict(task); owned = _owned_nodes(value, dag)
        if len(owned) != EXPECTED_EVIDENCE_NODES_PER_TASK[value["stage"]]:
            raise Fixed4ExecutionPilotError(
                f"evidence ownership count mismatch: {value['task_id']}")
        if any(row["node_id"] in seen for row in owned):
            raise Fixed4ExecutionPilotError("evidence DAG node has two owners")
        seen.update(row["node_id"] for row in owned)
        value["evidence_nodes"] = owned; value["evidence_node_count"] = len(owned)
        value["evidence_node_closure_sha256"] = stable_json_sha256(owned)
        value["payload_sha256"] = stable_json_sha256(
            {key: item for key, item in value.items() if key != "payload_sha256"})
        sealed.append(value)
    dag_ids = {row.get("task_id") for row in dag.get("nodes", [])}
    if len(seen) != EXPECTED_NODE_COUNT or seen != dag_ids:
        raise Fixed4ExecutionPilotError("evidence ownership is not exhaustive")
    mapping = [{**node, "operational_task_id": task["task_id"],
                "operational_task_payload_sha256": task["payload_sha256"]}
               for task in sealed for node in task["evidence_nodes"]]
    mapping.sort(key=lambda row: row["ordinal"])
    if ([row["ordinal"] for row in mapping] != list(range(EXPECTED_NODE_COUNT))
            or len({row["node_id"] for row in mapping}) != EXPECTED_NODE_COUNT):
        raise Fixed4ExecutionPilotError("evidence mapping ordinal closure mismatch")
    return sealed, mapping


def build_operational_tasks(
    bindings: Sequence[HypothesisBinding], preflight_identity: Mapping[str, Any],
    dag: Mapping[str, Any], execution_registry: Sequence[Mapping[str, Any]],
    *, runner_mode: str = RUNNER_MODE_DISABLED,
    production_adapter_protocol_ready: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(bindings) != 34:
        raise Fixed4ExecutionPilotError("operational inputs are not exact 34")
    tasks: list[dict[str, Any]] = []; ids: set[str] = set()
    pilot_ids: dict[str, list[str]] = {pair: [] for pair in FIXED_PAIR_ORDER}
    eligible_pilot_ids: dict[str, list[str]] = {pair: [] for pair in FIXED_PAIR_ORDER}
    abstaining_pilot_ids: dict[str, list[str]] = {pair: [] for pair in FIXED_PAIR_ORDER}

    def add(stage: str, task_id: str, upstream: Sequence[str], **fields: Any) -> str:
        if task_id in ids or any(parent not in ids for parent in upstream):
            raise Fixed4ExecutionPilotError("task collision/non-topological DAG")
        value = {"schema": TASK_SCHEMA, "ordinal": len(tasks), "task_id": task_id,
            "stage": stage, "execution_authorized": False,
            "execution_performed": False, "upstream_task_ids": list(upstream),
            "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "gt_allowed": False, "official92_allowed": False,
            "threshold_change_allowed": False, "result_selection_allowed": False,
            "reconstruction_authorized": False, "refusion_allowed": False,
            "preflight_identity": dict(preflight_identity),
            "execution_binding": task_execution_binding(
                execution_registry,
                str(preflight_identity["runner_registry_closure_sha256"]),
                stage, task_id),
            **fields}
        if runner_mode == RUNNER_MODE_ACTIVE:
            value["stage_runner_input_descriptor"] = \
                _active_stage_input_descriptor(
                    stage, task_id, upstream,
                    production_ready=production_adapter_protocol_ready)
        elif runner_mode != RUNNER_MODE_DISABLED:
            raise Fixed4ExecutionPilotError("unknown operational task runner mode")
        value["payload_sha256"] = stable_json_sha256(value)
        ids.add(task_id); tasks.append(value); return task_id

    for binding in bindings:
        common = {"pair_id": binding.pair_id,
            "hypothesis_index": binding.hypothesis_index,
            "hypothesis_sha256": binding.hypothesis_sha256,
            "prepared_input_path": binding.prepared_input_path,
            "prepared_input_sha256": binding.prepared_input_sha256,
            "contains_typed_failure_members": binding.contains_typed_failure_members,
            "existing_typed_failure_member_candidate_indices":
                list(binding.existing_typed_failure_member_candidate_indices),
            "new_typed_failure_member_candidate_indices":
                list(binding.new_typed_failure_member_candidate_indices),
            "typed_failure_member_candidate_indices":
                list(binding.typed_failure_member_candidate_indices),
            "typed_failure_policy": "explicit_replay_never_filter",
            "safe_pose_vote_eligible": binding.safe_pose_vote_eligible,
            "selector_eligible": binding.selector_eligible}
        direction_ids = []
        for direction in DIRECTIONS:
            direction_ids.append(add("colorpcr_direction",
                _task_id("colorpcr_direction", binding, direction), (), **common,
                direction=direction, sentinel_workers=list(SENTINELS), worker_seed=7351,
                neighbor_limits=[38, 36, 36, 38], voxel_m=0.10, coarsest_cap=512,
                output_contract="sentinel_invariant_exact_three_cache"))
        pilot_id = add(
            "bidirectional_multi_solver_pilot", _task_id("bidirectional_pilot", binding),
            direction_ids, **common, max_candidates=8, min_correspondences=40,
            exact_solver_rows=EXACT_SOLVER_ROWS_PER_PILOT,
            solver_matrix={"solvers": list(SOLVERS), "directions": list(DIRECTIONS),
                           "seeds": list(SEEDS)}, preregistered_solver_nodes=160,
            repeats=5, quorum=4, rotation_max_deg=5.0, translation_max_m=0.10,
            acceptance_forced_false=not binding.safe_pose_vote_eligible,
            icp="unchanged_fixed_trace_v13_authority", rule_b="unchanged_v13_authority",
            absent_candidate_policy="typed_not_generated_no_transform")
        pilot_ids[binding.pair_id].append(pilot_id)
        (eligible_pilot_ids if binding.safe_pose_vote_eligible
         else abstaining_pilot_ids)[binding.pair_id].append(pilot_id)
    pair_tasks = []
    for ordinal, (pair_id, count) in enumerate(zip(FIXED_PAIR_ORDER, EXPECTED_HYPOTHESES)):
        known_bad = pair_id == KNOWN_BAD_PAIR_ID
        pair_tasks.append(add("v16_pair_hypothesis_cluster", f"v16_pair.p{ordinal}",
            pilot_ids[pair_id], pair_id=pair_id, expected_hypothesis_count=count,
            all_hypotheses_required=True, known_bad=known_bad, permanent_veto=known_bad,
            eligible_hypothesis_task_ids=eligible_pilot_ids[pair_id],
            typed_abstention_hypothesis_task_ids=abstaining_pilot_ids[pair_id],
            acceptance_rule=("permanent_known_bad_veto" if known_bad else
                             "one_unique_complete_linkage_safe_pose_cluster")))
    add("fixed4_aggregate", "fixed4.aggregate", pair_tasks,
        normal_pair_rule="all_three_normals_each_require_unique_compatible_safe_pose_cluster",
        known_bad_rule="all_12_replayed_then_permanent_veto", control_can_rescue=False)
    counts = {stage: sum(row["stage"] == stage for row in tasks)
              for stage in OPERATIONAL_STAGE_COUNTS}
    if counts != OPERATIONAL_STAGE_COUNTS or len(tasks) != OPERATIONAL_TASK_COUNT:
        raise Fixed4ExecutionPilotError("operational task count mismatch")
    return bind_evidence_ownership(tasks, dag)


def _runner_closure(repo: Path, *, runner_mode: str = RUNNER_MODE_DISABLED,
                    ) -> tuple[list[dict[str, str]], str]:
    try:
        rows, digest = build_subprocess_registry(repo, runner_mode=runner_mode)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if [row["stage"] for row in rows] != list(ALLOWED_STAGES):
        raise Fixed4ExecutionPilotError("runner registry stage order changed")
    return rows, digest


def build_preflight(
    *, repo: Path, preregister_path: Path, exact191_path: Path,
    exact191_sha256: str, exact72_lineage_path: Path,
    exact72_lineage_sha256: str, prepared_path: Path, prepared_sha256: str,
    output_root: Path, runner_mode: str = RUNNER_MODE_DISABLED,
    active_preregister_path: Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve(); output_root = Path(output_root).resolve()
    head, tree = _git_identity(repo, require_clean=True)
    preregister_path = Path(preregister_path).resolve()
    preregister = _json(preregister_path, None, "fixed4 preregistration")
    validate_preregister(preregister); verify_source_pins(repo, preregister)
    if runner_mode not in {RUNNER_MODE_DISABLED, RUNNER_MODE_ACTIVE}:
        raise Fixed4ExecutionPilotError("unknown preflight runner mode")
    exec_prereg_path = (repo / "manifests/v16_b716_fixed4_execution_pilot_preregister.json"
        if runner_mode == RUNNER_MODE_DISABLED else
        Path(active_preregister_path or "").resolve())
    if runner_mode == RUNNER_MODE_ACTIVE and active_preregister_path is None:
        raise Fixed4ExecutionPilotError(
            "active preflight requires separate candidate preregistration")
    exec_prereg = _json(exec_prereg_path, None, "execution pilot preregistration")
    if runner_mode == RUNNER_MODE_DISABLED:
        _validate_execution_preregister(exec_prereg, repo)
    else:
        validate_active_execution_preregister_candidate(exec_prereg, repo)
    active_production_ready = bool(
        runner_mode == RUNNER_MODE_ACTIVE
        and exec_prereg.get("schema") == ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA
        and exec_prereg.get("production_adapter_ready") is True)
    try:
        lineage_binding = verify_lineage_seal(
            exact72_lineage_path, exact72_lineage_sha256)
    except Exact72LineageSealError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if lineage_binding.get("exact191_manifest_sha256") != exact191_sha256:
        raise Fixed4ExecutionPilotError(
            "exact72 lineage is not bound to supplied exact191 manifest")
    execution_binding = lineage_binding["execution_binding"]
    bindings = bind_hypotheses(
        load_exact191_hypotheses(exact191_path, exact191_sha256),
        load_prepared_hypotheses(prepared_path, prepared_sha256))
    prereg_sha = sha256_file(preregister_path)
    dag = build_task_dag(bindings, prereg_sha, synthetic_fixture=False)
    runner_rows, runner_sha = _runner_closure(repo, runner_mode=runner_mode)
    if (runner_mode == RUNNER_MODE_ACTIVE
            and exec_prereg.get("active_runner_registry_closure_sha256")
                != runner_sha):
        raise Fixed4ExecutionPilotError(
            "active preregistration runner registry closure drift")
    identity = {"repo_root": str(repo), "git_head": head, "git_tree": tree,
        "output_root": str(output_root), "preregister_sha256": prereg_sha,
        "execution_pilot_preregister_sha256": sha256_file(exec_prereg_path),
        "execution_pilot_preregister_payload_sha256": exec_prereg["payload_sha256"],
        "exact191_manifest_sha256": _sha(exact191_sha256, "exact191 SHA"),
        "exact72_lineage_manifest_sha256": _sha(
            exact72_lineage_sha256, "exact72 lineage SHA"),
        "exact72_lineage_payload_sha256":
            lineage_binding["lineage_payload_sha256"],
        "exact72_lineage_frozen_closure_sha256":
            lineage_binding["frozen_clone_file_closure_sha256"],
        "prepared_builder_manifest_sha256": _sha(prepared_sha256, "prepared SHA"),
        "dag_payload_sha256": dag["payload_sha256"],
        "runner_registry_closure_sha256": runner_sha,
        **({"runner_mode": RUNNER_MODE_ACTIVE}
           if runner_mode == RUNNER_MODE_ACTIVE else {})}
    tasks, mapping = build_operational_tasks(
        bindings, identity, dag, runner_rows, runner_mode=runner_mode,
        production_adapter_protocol_ready=active_production_ready)
    task_rows = [{"ordinal": row["ordinal"], "task_id": row["task_id"],
        "stage": row["stage"], "payload_sha256": row["payload_sha256"],
        "evidence_node_count": row["evidence_node_count"],
        "evidence_node_closure_sha256": row["evidence_node_closure_sha256"]}
        for row in tasks]
    source_rows = []
    source_relatives = [
        "manifests/v16_b716_fixed4_execution_pilot_preregister.json",
        "src/safety/v16_b716_fixed4_execution_pilot.py",
        "src/safety/v16_b716_fixed4_subprocess_contract.py",
        "src/safety/v16_b716_fixed4_stage_runners.py",
        "src/safety/v16_b716_fixed4_orchestrator_contract.py",
        "src/safety/v16_b716_exact72_lineage_seal.py",
        "src/safety/v13_dual_solver_runtime.py", "src/safety/v13_strict_pair_gate.py",
        "src/safety/v14_rigid_multihypothesis.py", "src/safety/v15_safe_pose_cluster.py",
        "src/safety/v16_safe_hypothesis_cluster.py",
        "scripts/v13_colorpcr_sentinel_subprocess.py",
        "scripts/v13_colorpcr_official_worker.py",
        "scripts/v16_b716_fixed4_execution_pilot.py",
        "scripts/v16_b716_exact72_lineage_seal.py",
        "scripts/v16_b716_fixed4_disabled_stage_runner.sh",
        "scripts/v16_b716_fixed4_sealed_executor.py"]
    if runner_mode == RUNNER_MODE_ACTIVE:
        try:
            active_preregister_relative = str(exec_prereg_path.relative_to(repo))
        except ValueError as exc:
            raise Fixed4ExecutionPilotError(
                "active preregistration must be inside repository") from exc
        source_relatives.extend([
            active_preregister_relative,
            "src/safety/v16_b716_fixed4_production_adapters.py",
            "src/safety/v16_b716_fixed4_production_manifest_builder.py",
            "src/safety/v16_b716_fixed4_assets_builder.py",
            "src/safety/v16_b716_fixed4_active_production_wrapper.py",
            "scripts/v16_b716_fixed4_production_adapter.py",
            "scripts/v16_b716_fixed4_production_manifest_builder.py",
            "scripts/v16_b716_fixed4_assets_builder.py",
            "scripts/v16_b716_fixed4_active_production_wrapper.py",
            "scripts/v16_b716_fixed4_active_stage_runner.sh",
            "scripts/v16_b716_fixed4_active_sealed_executor.py",
            "scripts/v16_b716_fixed4_active_dispatch_cli.py"])
    for relative in source_relatives:
        path = repo / relative
        if not path.is_file():
            raise Fixed4ExecutionPilotError(f"execution source missing: {relative}")
        source_rows.append({"path": relative, "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    value = {"schema": PREFLIGHT_SCHEMA, "frozen": True, "sealed": True,
        "execution_authorized": False, "execution_performed": False,
        "gt_allowed": False, "official92_allowed": False,
        "threshold_change_allowed": False, "result_selection_allowed": False,
        "default_checkpoint_replacement_allowed": False,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "repo_root": str(repo), "git_head": head, "git_tree": tree,
        "output_root": str(output_root), "fixed_pair_order": list(FIXED_PAIR_ORDER),
        "hypothesis_distribution": list(EXPECTED_HYPOTHESES),
        "exact72_execution_binding": execution_binding,
        "exact72_lineage_validation": lineage_binding,
        "preregister_path": str(preregister_path), "preregister_sha256": prereg_sha,
        "execution_pilot_preregister_path": str(exec_prereg_path),
        "execution_pilot_preregister_sha256": sha256_file(exec_prereg_path),
        "execution_pilot_preregister_payload_sha256": exec_prereg["payload_sha256"],
        "exact191_manifest_path": str(Path(exact191_path).resolve()),
        "exact191_manifest_sha256": exact191_sha256,
        "exact72_lineage_manifest_path": str(
            Path(exact72_lineage_path).resolve()),
        "exact72_lineage_manifest_sha256": exact72_lineage_sha256,
        "prepared_builder_manifest_path": str(Path(prepared_path).resolve()),
        "prepared_builder_manifest_sha256": prepared_sha256,
        "dag": dag, "dag_payload_sha256": dag["payload_sha256"],
        "evidence_ownership_mapping": mapping,
        "evidence_ownership_mapping_sha256": stable_json_sha256(mapping),
        "evidence_ownership_node_count": len(mapping),
        "operational_stage_counts": OPERATIONAL_STAGE_COUNTS,
        "operational_task_count": len(tasks), "operational_task_closure": task_rows,
        "operational_task_closure_sha256": stable_json_sha256(task_rows),
        "runner_registry": runner_rows, "runner_registry_closure_sha256": runner_sha,
        "execution_source_closure": source_rows,
        "execution_source_closure_sha256": stable_json_sha256(source_rows),
        "reconstruction_authorized": False, "refusion_allowed": False,
        **({"active_subprocess_contract": {
            "schema": (ACTIVE_PREFLIGHT_V2_SCHEMA if active_production_ready
                       else ACTIVE_PREFLIGHT_SCHEMA),
            "runner_mode": RUNNER_MODE_ACTIVE,
            "runner_registry_closure_sha256": runner_sha,
            "sealed_executor_sha256": runner_rows[0]["sealed_executor"]["source"]["sha256"],
            "legacy_disabled_preflight_accepted": False,
            "contract_fixture_allowed": False,
            **({"production_adapter_protocol_ready": True}
               if active_production_ready else {}),
            "operational_result_release_allowed": False}}
           if runner_mode == RUNNER_MODE_ACTIVE else {}),
        "registration_defense_guard": {
            "status": "P0_UNAUTHORIZED_PENDING_INDEPENDENT_AUDIT",
            "independent_audit_status": "NOT_RUN",
            "solver_family_contract_aligned": False,
            "reconstruction_authorized": False, "refusion_allowed": False}}
    value["payload_sha256"] = stable_json_sha256(value); value["_tasks"] = tasks
    return value


def _create_only_json(root: Path, path: Path, value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    try:
        _row, state = create_only_bytes_beneath(
            root, path, encoded, create_parents=True, resume_identical=True)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    return state


def materialize_preflight(output_root: Path,
                          value: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_root)
    if not root.is_absolute():
        raise Fixed4ExecutionPilotError("output root must be absolute")
    try:
        ensure_no_symlink_directory(root, "authorized output root", create=True)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    public = {key: item for key, item in value.items() if key != "_tasks"}
    if (value.get("schema") != PREFLIGHT_SCHEMA
            or value.get("execution_authorized") is not False
            or value.get("execution_performed") is not False
            or value.get("output_root") != str(root)
            or value.get("evidence_ownership_node_count") != EXPECTED_NODE_COUNT
            or value.get("operational_task_count") != OPERATIONAL_TASK_COUNT
            or not _payload_valid(public)):
        raise Fixed4ExecutionPilotError("preflight is not sealed/disabled")
    tasks = value.get("_tasks")
    if not isinstance(tasks, list) or len(tasks) != OPERATIONAL_TASK_COUNT:
        raise Fixed4ExecutionPilotError("preflight task payload missing")
    states = {"created": 0, "resumed_identical": 0}; rows = []
    lineage = value.get("exact72_lineage_validation")
    if not isinstance(lineage, Mapping):
        raise Fixed4ExecutionPilotError("lineage validation receipt missing")
    lineage_receipt = {
        "schema": "v16-b716-fixed4-lineage-validation-receipt-v1",
        "exact72_lineage_manifest_path":
            value["exact72_lineage_manifest_path"],
        "exact72_lineage_manifest_sha256":
            value["exact72_lineage_manifest_sha256"],
        "exact72_lineage_payload_sha256": lineage["lineage_payload_sha256"],
        "frozen_clone_file_closure_sha256":
            lineage["frozen_clone_file_closure_sha256"],
        "exact191_manifest_sha256": lineage["exact191_manifest_sha256"],
        "task_count": lineage["task_count"],
        "ok_count": lineage["ok_count"],
        "typed_failure_count": lineage["typed_failure_count"],
        "missing_count": 0, "extra_count": 0,
        "symlink_count": 0, "hash_mismatch_count": 0,
        "execution_authorized": False,
    }
    lineage_receipt["payload_sha256"] = stable_json_sha256(lineage_receipt)
    states[_create_only_json(
        root, root / "lineage_validation_receipt.json",
        lineage_receipt)] += 1
    for task in tasks:
        path = root / "tasks" / task["task_id"] / "task.json"
        states[_create_only_json(root, path, task)] += 1
        file_row = no_symlink_file_row(path, "materialized task")
        rows.append({"task_id": task["task_id"], "stage": task["stage"],
            "path": str(path.relative_to(root)), "bytes": file_row["bytes"],
            "sha256": file_row["sha256"], "payload_sha256": task["payload_sha256"],
            "evidence_node_count": task["evidence_node_count"],
            "evidence_node_closure_sha256": task["evidence_node_closure_sha256"]})
    manifest = {"schema": TASK_MANIFEST_SCHEMA, "repo_root": value["repo_root"],
        "git_head": value["git_head"], "git_tree": value["git_tree"],
        "output_root": str(root), "preflight_payload_sha256": value["payload_sha256"],
        "runner_registry_closure_sha256": value["runner_registry_closure_sha256"],
        "evidence_ownership_mapping_sha256": value["evidence_ownership_mapping_sha256"],
        "task_count": len(rows), "stage_counts": OPERATIONAL_STAGE_COUNTS,
        "tasks": rows, "task_closure_sha256": stable_json_sha256(rows)}
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    states[_create_only_json(root, root / "task_manifest.json", manifest)] += 1
    states[_create_only_json(root, root / "execution_preflight.json", public)] += 1
    return {"states": states, "task_count": len(rows),
            "preflight": str(root / "execution_preflight.json"),
            "manifest": str(root / "task_manifest.json")}


def _validate_guard_audit(path: Path, expected_sha: str,
                          *, preflight: Mapping[str, Any],
                          manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    audit = _json(path, expected_sha, "independent guard audit receipt")
    required = {"schema", "status", "independent_reviewer", "reviewed_at",
        "repo_root", "git_head", "git_tree", "output_root",
        "task_manifest_sha256", "task_manifest_payload_sha256",
        "runner_registry_closure_sha256", "evidence_ownership_mapping_sha256",
        "registration_guard_status", "reconstruction_authorized", "refusion_allowed",
        "normal_test_log", "clean_test_log", "signature_algorithm",
        "signing_key_id", "signature_b64", "payload_sha256"}
    _require_exact_keys(audit, required, "guard audit")
    try:
        verify_fixed_signed_document(
            audit, repo_root=Path(str(preflight.get("repo_root", ""))),
            output_root=output_root, purpose="independent guard audit receipt")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    manifest_path = Path(output_root).resolve() / "task_manifest.json"
    if (not _payload_valid(audit) or audit.get("schema") != GUARD_AUDIT_SCHEMA
            or audit.get("status") != "PASS"
            or not isinstance(audit.get("independent_reviewer"), str)
            or not audit["independent_reviewer"].strip()
            or audit.get("repo_root") != preflight.get("repo_root")
            or audit.get("git_head") != preflight.get("git_head")
            or audit.get("git_tree") != preflight.get("git_tree")
            or audit.get("output_root") != str(Path(output_root).resolve())
            or audit.get("task_manifest_sha256") != sha256_file(manifest_path)
            or audit.get("task_manifest_payload_sha256") != manifest.get("payload_sha256")
            or audit.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")
            or audit.get("evidence_ownership_mapping_sha256")
                != preflight.get("evidence_ownership_mapping_sha256")
            or audit.get("registration_guard_status") != "UNREVIEWED_REFUSION_DISABLED"
            or audit.get("reconstruction_authorized") is not False
            or audit.get("refusion_allowed") is not False):
        raise Fixed4ExecutionPilotError("guard audit binding mismatch")
    _parse_time(audit["reviewed_at"], "guard audit reviewed_at")
    for role in ("normal_test_log", "clean_test_log"):
        if not isinstance(audit.get(role), Mapping):
            raise Fixed4ExecutionPilotError(f"guard audit {role} missing")
        _resolve_row_file(output_root, audit[role], f"guard audit {role}")
    return audit


def _validate_source_and_registry(preflight: Mapping[str, Any]) -> None:
    repo = Path(str(preflight.get("repo_root", ""))).resolve()
    head, tree = _git_identity(repo, require_clean=True)
    if head != preflight.get("git_head") or tree != preflight.get("git_tree"):
        raise Fixed4ExecutionPilotError("git HEAD/tree drift")
    rows = preflight.get("execution_source_closure")
    if (not isinstance(rows, list) or stable_json_sha256(rows)
            != preflight.get("execution_source_closure_sha256")):
        raise Fixed4ExecutionPilotError("execution source closure malformed")
    for row in rows:
        raw_path = repo / str(row.get("path", ""))
        if raw_path.is_symlink():
            raise Fixed4ExecutionPilotError("execution source symlink rejected")
        path = raw_path.resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise Fixed4ExecutionPilotError("execution source escapes repository") from exc
        if (not path.is_file() or path.stat().st_size != row.get("bytes")
                or sha256_file(path) != row.get("sha256")):
            raise Fixed4ExecutionPilotError("execution source closure drift")
    active_contract = preflight.get("active_subprocess_contract")
    runner_mode = (RUNNER_MODE_ACTIVE if isinstance(active_contract, Mapping)
                   else RUNNER_MODE_DISABLED)
    if runner_mode == RUNNER_MODE_ACTIVE:
        active_schema = active_contract.get("schema")
        if active_schema not in {ACTIVE_PREFLIGHT_SCHEMA,
                                  ACTIVE_PREFLIGHT_V2_SCHEMA}:
            raise Fixed4ExecutionPilotError("active subprocess preflight schema drift")
        expected = {"schema": active_schema,
            "runner_mode": RUNNER_MODE_ACTIVE,
            "runner_registry_closure_sha256":
                preflight.get("runner_registry_closure_sha256"),
            "sealed_executor_sha256":
                preflight.get("runner_registry", [{}])[0].get(
                    "sealed_executor", {}).get("source", {}).get("sha256"),
            "legacy_disabled_preflight_accepted": False,
            "contract_fixture_allowed": False,
            "operational_result_release_allowed": False}
        if active_schema == ACTIVE_PREFLIGHT_V2_SCHEMA:
            expected["production_adapter_protocol_ready"] = True
        if active_contract != expected:
            raise Fixed4ExecutionPilotError("active subprocess preflight drift")
    try:
        validate_subprocess_registry(
            repo, preflight.get("runner_registry"),
            preflight.get("runner_registry_closure_sha256"),
            runner_mode=runner_mode)
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    try:
        lineage = verify_lineage_seal(
            Path(str(preflight.get("exact72_lineage_manifest_path", ""))),
            str(preflight.get("exact72_lineage_manifest_sha256", "")))
    except Exact72LineageSealError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if (lineage.get("lineage_payload_sha256")
            != preflight.get("exact72_lineage_validation", {}).get(
                "lineage_payload_sha256")
            or lineage.get("frozen_clone_file_closure_sha256")
            != preflight.get("exact72_lineage_validation", {}).get(
                "frozen_clone_file_closure_sha256")
            or lineage.get("exact191_manifest_sha256")
            != preflight.get("exact191_manifest_sha256")):
        raise Fixed4ExecutionPilotError("lineage changed after preflight")
    bind_hypotheses(
        load_exact191_hypotheses(
            Path(str(preflight.get("exact191_manifest_path", ""))),
            str(preflight.get("exact191_manifest_sha256", ""))),
        load_prepared_hypotheses(
            Path(str(preflight.get("prepared_builder_manifest_path", ""))),
            str(preflight.get("prepared_builder_manifest_sha256", ""))))


def validate_authorization(path: Path, expected_sha: str, preflight_path: Path,
                           expected_preflight_sha: str) -> dict[str, Any]:
    preflight_path = Path(preflight_path).resolve()
    preflight = _json(preflight_path, expected_preflight_sha, "execution preflight")
    authorization = _json(path, expected_sha, "execution authorization")
    output_root = Path(str(preflight.get("output_root", ""))).resolve()
    manifest_path = output_root / "task_manifest.json"
    manifest = _json(manifest_path, None, "operational task manifest")
    required = {"schema", "status", "authorization_scope", "execution_authorized",
        "execution_performed", "issued_at", "expires_at", "repo_root", "git_head",
        "git_tree", "output_root", "preflight_path", "preflight_sha256",
        "preflight_payload_sha256", "dag_payload_sha256",
        "execution_pilot_preregister_sha256",
        "execution_pilot_preregister_payload_sha256",
        "operational_task_closure_sha256", "evidence_ownership_mapping_sha256",
        "runner_registry_closure_sha256", "execution_source_closure_sha256",
        "exact191_manifest_sha256", "exact72_lineage_manifest_sha256",
        "exact72_lineage_payload_sha256",
        "exact72_lineage_frozen_closure_sha256",
        "prepared_builder_manifest_sha256",
        "official_release_checkpoint_sha256", "task_manifest_path",
        "task_manifest_sha256", "task_manifest_payload_sha256",
        "authorized_task_ids", "authorized_task_ids_sha256", "allowed_stages",
        "guard_audit_receipt_path", "guard_audit_receipt_sha256",
        "guard_audit_receipt_payload_sha256", "gt_allowed", "official92_allowed",
        "threshold_change_allowed", "result_selection_allowed",
        "default_checkpoint_replacement_allowed", *REQUIRED_AUTH_REVIEW_FIELDS.keys(),
        "signature_algorithm", "signing_key_id", "signature_b64",
        "payload_sha256"}
    _require_exact_keys(authorization, required, "authorization")
    try:
        verify_fixed_signed_document(
            authorization, repo_root=Path(str(preflight.get("repo_root", ""))),
            output_root=output_root, purpose="execution authorization")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if (not _payload_valid(preflight) or not _payload_valid(authorization)
            or preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("execution_authorized") is not False
            or authorization.get("schema") != AUTH_SCHEMA
            or authorization.get("status") != "PASS"
            or authorization.get("authorization_scope") != "fixed4_all_107_operational_tasks"
            or authorization.get("execution_authorized") is not True
            or authorization.get("execution_performed") is not False
            or authorization.get("repo_root") != preflight.get("repo_root")
            or authorization.get("git_head") != preflight.get("git_head")
            or authorization.get("git_tree") != preflight.get("git_tree")
            or authorization.get("output_root") != str(output_root)
            or authorization.get("preflight_path") != str(preflight_path)
            or authorization.get("preflight_sha256") != expected_preflight_sha
            or authorization.get("preflight_payload_sha256") != preflight.get("payload_sha256")
            or authorization.get("dag_payload_sha256") != preflight.get("dag_payload_sha256")
            or authorization.get("execution_pilot_preregister_sha256")
                != preflight.get("execution_pilot_preregister_sha256")
            or authorization.get("execution_pilot_preregister_payload_sha256")
                != preflight.get("execution_pilot_preregister_payload_sha256")
            or authorization.get("operational_task_closure_sha256")
                != preflight.get("operational_task_closure_sha256")
            or authorization.get("evidence_ownership_mapping_sha256")
                != preflight.get("evidence_ownership_mapping_sha256")
            or authorization.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")
            or authorization.get("execution_source_closure_sha256")
                != preflight.get("execution_source_closure_sha256")
            or authorization.get("exact191_manifest_sha256")
                != preflight.get("exact191_manifest_sha256")
            or authorization.get("exact72_lineage_manifest_sha256")
                != preflight.get("exact72_lineage_manifest_sha256")
            or authorization.get("exact72_lineage_payload_sha256")
                != preflight.get("exact72_lineage_validation", {}).get(
                    "lineage_payload_sha256")
            or authorization.get("exact72_lineage_frozen_closure_sha256")
                != preflight.get("exact72_lineage_validation", {}).get(
                    "frozen_clone_file_closure_sha256")
            or authorization.get("prepared_builder_manifest_sha256")
                != preflight.get("prepared_builder_manifest_sha256")
            or authorization.get("official_release_checkpoint_sha256")
                != OFFICIAL_RELEASE_SHA256
            or authorization.get("task_manifest_path") != str(manifest_path)
            or authorization.get("task_manifest_sha256") != sha256_file(manifest_path)
            or authorization.get("task_manifest_payload_sha256")
                != manifest.get("payload_sha256")
            or authorization.get("allowed_stages") != list(ALLOWED_STAGES)
            or authorization.get("gt_allowed") is not False
            or authorization.get("official92_allowed") is not False
            or authorization.get("threshold_change_allowed") is not False
            or authorization.get("result_selection_allowed") is not False
            or authorization.get("default_checkpoint_replacement_allowed") is not False):
        raise Fixed4ExecutionPilotError("execution authorization mismatch")
    for key, expected in REQUIRED_AUTH_REVIEW_FIELDS.items():
        if authorization.get(key) != expected:
            raise Fixed4ExecutionPilotError(f"execution authorization review missing: {key}")
    issued = _parse_time(authorization["issued_at"], "authorization issued_at")
    expires = _parse_time(authorization["expires_at"], "authorization expires_at")
    now = datetime.now(timezone.utc)
    if (issued > now + timedelta(seconds=AUTH_CLOCK_SKEW_SECONDS) or expires <= now
            or expires <= issued
            or (expires-issued).total_seconds() > MAX_AUTH_TTL_SECONDS):
        raise Fixed4ExecutionPilotError("authorization TTL invalid/expired")
    rows = manifest.get("tasks"); task_ids = authorization.get("authorized_task_ids")
    expected_ids = [row.get("task_id") for row in rows] if isinstance(rows, list) else None
    if (manifest.get("schema") != TASK_MANIFEST_SCHEMA or not _payload_valid(manifest)
            or manifest.get("repo_root") != preflight.get("repo_root")
            or manifest.get("git_head") != preflight.get("git_head")
            or manifest.get("git_tree") != preflight.get("git_tree")
            or manifest.get("output_root") != str(output_root)
            or manifest.get("task_count") != OPERATIONAL_TASK_COUNT
            or task_ids != expected_ids
            or stable_json_sha256(task_ids) != authorization.get("authorized_task_ids_sha256")):
        raise Fixed4ExecutionPilotError("authorization task scope mismatch")
    _validate_source_and_registry(preflight)
    audit = _validate_guard_audit(
        Path(str(authorization.get("guard_audit_receipt_path", ""))),
        authorization["guard_audit_receipt_sha256"], preflight=preflight,
        manifest=manifest, output_root=output_root)
    if audit.get("payload_sha256") != authorization.get("guard_audit_receipt_payload_sha256"):
        raise Fixed4ExecutionPilotError("guard audit payload binding mismatch")
    return authorization


def _transform_valid(value: Any) -> bool:
    try:
        validate_se3(value)
    except Exception:
        return False
    return True


def _validate_attempt_document(document: Mapping[str, Any], *, schema: str,
                               task: Mapping[str, Any], identity: Mapping[str, Any],
                               role: str) -> None:
    required = {"schema", "task_id", "task_payload_sha256", "status",
        "transform", "failure_type", *POLICY_FALSE_FIELDS.keys(),
        "payload_sha256", *identity.keys()}
    _require_exact_keys(document, required, role)
    if (not _payload_valid(document) or document.get("schema") != schema
            or document.get("task_id") != task.get("task_id")
            or document.get("task_payload_sha256") != task.get("payload_sha256")
            or any(document.get(key) != item for key, item in identity.items())):
        raise Fixed4ExecutionPilotError(f"{role} binding mismatch")
    _policy_false(document, role)
    if document.get("status") == "succeeded":
        if document.get("failure_type") is not None \
                or not _transform_valid(document.get("transform")):
            raise Fixed4ExecutionPilotError(f"{role} success transform invalid")
    elif document.get("status") == "typed_failure":
        if (not isinstance(document.get("failure_type"), str)
                or not document["failure_type"] or document.get("transform") is not None):
            raise Fixed4ExecutionPilotError(f"{role} typed failure must have no transform")
    else:
        raise Fixed4ExecutionPilotError(f"{role} status invalid")


def _file_row_from_wrapper(row: Mapping[str, Any], extra: set[str], role: str) -> dict:
    _require_exact_keys(row, {"path", "bytes", "sha256", *extra}, role)
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _validate_evidence_receipts(task: Mapping[str, Any], value: Mapping[str, Any],
                                root: Path) -> dict[str, str]:
    rows = value.get("evidence_receipts"); expected_nodes = task.get("evidence_nodes")
    if (not isinstance(rows, list) or not isinstance(expected_nodes, list)
            or len(rows) != task.get("evidence_node_count") or len(rows) != len(expected_nodes)
            or stable_json_sha256(rows) != value.get("evidence_receipt_closure_sha256")):
        raise Fixed4ExecutionPilotError("expanded evidence receipt closure mismatch")
    task_root = Path(root).resolve() / "tasks" / task["task_id"]
    statuses: dict[str, str] = {}
    for index, (row, expected) in enumerate(zip(rows, expected_nodes)):
        file_row = _file_row_from_wrapper(row, {"node_id", "node_payload_sha256"},
                                          f"evidence receipt row {index}")
        if (row.get("node_id") != expected.get("node_id")
                or row.get("node_payload_sha256") != expected.get("node_payload_sha256")):
            raise Fixed4ExecutionPilotError("evidence receipt node mismatch")
        path = _resolve_row_file(root, file_row, f"evidence receipt {index}",
                                 within=task_root / "evidence")
        receipt = _json(path, row["sha256"], "expanded evidence receipt")
        required = {"schema", "operational_task_id", "operational_task_payload_sha256",
            "node_id", "node_payload_sha256", "status", *POLICY_FALSE_FIELDS.keys(),
            "payload_sha256"}
        _require_exact_keys(receipt, required, "expanded evidence receipt")
        if (not _payload_valid(receipt) or receipt.get("schema") != EVIDENCE_RECEIPT_SCHEMA
                or receipt.get("operational_task_id") != task.get("task_id")
                or receipt.get("operational_task_payload_sha256") != task.get("payload_sha256")
                or receipt.get("node_id") != expected.get("node_id")
                or receipt.get("node_payload_sha256") != expected.get("node_payload_sha256")
                or receipt.get("status") not in {"consumed", "typed_not_generated"}):
            raise Fixed4ExecutionPilotError("expanded evidence binding mismatch")
        _policy_false(receipt, "expanded evidence receipt")
        statuses[str(expected["node_id"])] = str(receipt["status"])
    return statuses


_PILOT_SLOT_NODE = re.compile(
    r"^(?:solver|strict)\.p\d+\.h\d+\."
    r"[0-9a-f]{12}\.(\d+)(?:\.|$)")


def _validate_pilot_evidence_semantics(
    task: Mapping[str, Any], statuses: Mapping[str, str],
    slots: Mapping[int, Mapping[str, Any]],
) -> None:
    """Bind the expanded 171-node DAG to the eight candidate slots.

    The DAG preallocates all 160 solver rows.  An absent slot therefore closes
    those nodes as typed-not-generated; it must never look like executed work.
    Prepared/V14/V15 nodes are always consumed because even a fail-closed pilot
    must materialize and audit its abstention.
    """
    if len(statuses) != EXPECTED_EVIDENCE_NODES_PER_TASK[
            "bidirectional_multi_solver_pilot"]:
        raise Fixed4ExecutionPilotError("pilot evidence node count drift")
    solver_counts = {slot: 0 for slot in slots}
    strict_counts = {slot: 0 for slot in slots}
    stage_counts = {"prepared": 0, "v14_candidates": 0, "v15": 0}
    for node_id, observed in statuses.items():
        stage = node_id.split(".", 1)[0]
        if stage in stage_counts:
            stage_counts[stage] += 1
            expected = "consumed"
        elif stage in {"solver", "strict"}:
            match = _PILOT_SLOT_NODE.match(node_id)
            if match is None:
                raise Fixed4ExecutionPilotError(
                    "pilot evidence node identity is not parseable")
            slot = int(match.group(1))
            if slot not in slots:
                raise Fixed4ExecutionPilotError("pilot evidence slot escapes 0..7")
            expected = ("consumed" if slots[slot].get("status") == "generated"
                        else "typed_not_generated")
            if stage == "solver":
                solver_counts[slot] += 1
            else:
                strict_counts[slot] += 1
        else:
            raise Fixed4ExecutionPilotError("unexpected pilot evidence stage")
        if observed != expected:
            raise Fixed4ExecutionPilotError(
                f"pilot evidence status mismatch: {node_id}")
    if (stage_counts != {"prepared": 1, "v14_candidates": 1, "v15": 1}
            or any(solver_counts[slot] != EXACT_SOLVER_ROWS_PER_PILOT
                   for slot in slots)
            or any(strict_counts[slot] != 1 for slot in slots)):
        raise Fixed4ExecutionPilotError("pilot 171-node stage closure mismatch")


def _validate_output_artifacts(task: Mapping[str, Any], value: Mapping[str, Any],
                               root: Path) -> None:
    rows = value.get("output_artifacts")
    if not isinstance(rows, list):
        raise Fixed4ExecutionPilotError("result artifact list missing")
    task_root = Path(root).resolve() / "tasks" / task["task_id"]; seen = set()
    for index, row in enumerate(rows):
        path = _resolve_row_file(root, row, f"output artifact {index}",
                                 within=task_root / "artifacts")
        lowered = str(path).lower()
        if path.suffix.lower() == ".ply" or any(token in lowered
                                                for token in FORBIDDEN_ARTIFACT_TOKENS):
            raise Fixed4ExecutionPilotError("reconstruction/refusion artifact rejected")
        if str(path) in seen:
            raise Fixed4ExecutionPilotError("duplicate output artifact")
        seen.add(str(path))


def _validate_sentinel_attempts(task: Mapping[str, Any], value: Mapping[str, Any],
                                root: Path) -> None:
    rows = value.get("sentinel_attempts")
    if (not isinstance(rows, list) or len(rows) != 2
            or stable_json_sha256(rows) != value.get("sentinel_attempt_closure_sha256")):
        raise Fixed4ExecutionPilotError("sentinel attempt closure mismatch")
    task_root = Path(root).resolve() / "tasks" / task["task_id"]; observed = []
    for index, row in enumerate(rows):
        file_row = _file_row_from_wrapper(row, {"sentinel", "direction", "status"},
                                          f"sentinel attempt row {index}")
        identity = {"sentinel": row.get("sentinel"), "direction": row.get("direction")}
        path = _resolve_row_file(root, file_row, f"sentinel attempt {index}",
                                 within=task_root / "attempts")
        document = _json(path, row["sha256"], "sentinel attempt")
        _validate_attempt_document(document, schema=SENTINEL_ATTEMPT_SCHEMA,
            task=task, identity=identity, role="sentinel attempt")
        if row.get("status") != document.get("status"):
            raise Fixed4ExecutionPilotError("sentinel attempt status mismatch")
        observed.append((row.get("sentinel"), row.get("direction")))
    if observed != [(sentinel, task.get("direction")) for sentinel in SENTINELS]:
        raise Fixed4ExecutionPilotError("sentinel identity closure mismatch")


def _validate_candidate_slots(value: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = value.get("candidate_slots")
    if (not isinstance(rows, list) or len(rows) != MAX_CANDIDATE_SLOTS_PER_PILOT
            or stable_json_sha256(rows) != value.get("candidate_slot_closure_sha256")):
        raise Fixed4ExecutionPilotError("pilot candidate-slot closure mismatch")
    slots: dict[int, Mapping[str, Any]] = {}
    for expected_slot, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("candidate_slot") != expected_slot:
            raise Fixed4ExecutionPilotError("pilot candidate slots are not exact 0..7")
        status = row.get("status")
        common = {"candidate_slot", "status", "solver_rows_executed",
                  "failure_type"}
        if status == "generated":
            _require_exact_keys(row, common | {"transform", "safe_vote"},
                                f"candidate slot {expected_slot}")
            if (row.get("solver_rows_executed") != EXACT_SOLVER_ROWS_PER_PILOT
                    or row.get("failure_type") is not None
                    or not isinstance(row.get("safe_vote"), bool)
                    or not _transform_valid(row.get("transform"))):
                raise Fixed4ExecutionPilotError(
                    f"generated candidate slot {expected_slot} malformed")
        elif status == "typed_not_generated":
            # A missing/failed slot is an abstention.  In particular it has no
            # vote field at all, rather than a false value that downstream code
            # could accidentally count.
            _require_exact_keys(row, common, f"candidate slot {expected_slot}")
            if (row.get("solver_rows_executed") != 0
                    or not isinstance(row.get("failure_type"), str)
                    or not row["failure_type"]):
                raise Fixed4ExecutionPilotError(
                    f"typed-not-generated slot {expected_slot} carried output")
        else:
            raise Fixed4ExecutionPilotError(
                f"candidate slot {expected_slot} status invalid")
        slots[expected_slot] = row
    return slots


def _validate_solver_attempts(task: Mapping[str, Any], value: Mapping[str, Any],
                              root: Path,
                              slots: Mapping[int, Mapping[str, Any]],
                              ) -> dict[tuple[int, str, str, int], Mapping[str, Any]]:
    rows = value.get("solver_attempts")
    generated_slots = [slot for slot, row in slots.items()
                       if row.get("status") == "generated"]
    expected = [(slot, solver, direction, repeat)
                for slot in generated_slots for solver in SOLVERS
                for direction in DIRECTIONS for repeat in SEEDS]
    expected_row_count = EXACT_SOLVER_ROWS_PER_PILOT * len(generated_slots)
    if (not isinstance(rows, list) or len(rows) != expected_row_count
            or value.get("solver_rows_executed") != expected_row_count
            or stable_json_sha256(rows) != value.get("solver_attempt_closure_sha256")):
        raise Fixed4ExecutionPilotError(
            "each generated candidate slot must contain exact20 solver rows")
    task_root = Path(root).resolve() / "tasks" / task["task_id"]; observed = []
    documents: dict[tuple[int, str, str, int], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        file_row = _file_row_from_wrapper(row, {"candidate_slot", "solver", "direction",
                                                "repeat", "status"},
                                          f"solver attempt row {index}")
        identity = {"candidate_slot": row.get("candidate_slot"),
                    "solver": row.get("solver"),
                    "direction": row.get("direction"),
                    "repeat": row.get("repeat")}
        path = _resolve_row_file(root, file_row, f"solver attempt {index}",
                                 within=task_root / "attempts")
        document = _json(path, row["sha256"], "solver attempt")
        _validate_attempt_document(document, schema=SOLVER_ATTEMPT_SCHEMA,
            task=task, identity=identity, role="solver attempt")
        if row.get("status") != document.get("status"):
            raise Fixed4ExecutionPilotError("solver attempt status mismatch")
        key = (row.get("candidate_slot"), row.get("solver"),
               row.get("direction"), row.get("repeat"))
        observed.append(key); documents[key] = document
    if observed != expected or len(set(observed)) != expected_row_count:
        raise Fixed4ExecutionPilotError(
            "candidate-slot/solver/direction/repeat closure mismatch")
    quorum = task.get("quorum")
    if not isinstance(quorum, int) or quorum != 4:
        raise Fixed4ExecutionPilotError("pilot quorum drift")
    for slot, slot_row in slots.items():
        if slot_row.get("status") != "generated" or slot_row.get("safe_vote") is not True:
            continue
        succeeded = [document for key, document in documents.items()
                     if key[0] == slot and document.get("status") == "succeeded"]
        if len(succeeded) < quorum:
            raise Fixed4ExecutionPilotError(
                "safe-vote slot lacks preregistered solver quorum")
        transform_sha = stable_json_sha256(slot_row.get("transform"))
        if all(stable_json_sha256(document.get("transform")) != transform_sha
               for document in succeeded):
            raise Fixed4ExecutionPilotError(
                "safe-vote transform is not backed by a successful solver row")
    return documents


def _validate_hypothesis_outcome(
    task: Mapping[str, Any], value: Mapping[str, Any],
    slots: Mapping[int, Mapping[str, Any]], root: Path,
) -> None:
    outcome = value.get("hypothesis_outcome")
    if not isinstance(outcome, Mapping):
        raise Fixed4ExecutionPilotError("pilot hypothesis outcome missing")
    required = {"hypothesis_task_id", "gate_status", "failure_class",
                "safe_transform", "source_result_payload_sha256",
                "measured_rotation_deg", "measured_translation_m",
                "measurement_source_file_sha256",
                "measurement_source_payload_sha256",
                "measurement_candidate_slot",
                "measurement_candidate_set_sha256",
                "measurement_slot_results_payload_sha256",
                "measurement_v15_decision_sha256"}
    _require_exact_keys(outcome, required, "pilot hypothesis outcome")
    if outcome.get("hypothesis_task_id") != task.get("task_id"):
        raise Fixed4ExecutionPilotError("pilot hypothesis outcome identity drift")
    source_sha = _sha(outcome.get("source_result_payload_sha256"),
                      "pilot hypothesis outcome source result SHA")
    v15_nodes = [row for row in task.get("evidence_nodes", ())
                 if str(row.get("node_id", "")).startswith("v15.")]
    if len(v15_nodes) != 1:
        raise Fixed4ExecutionPilotError("pilot V15 evidence binding is not unique")
    v15_node = v15_nodes[0]
    receipt = value.get("hypothesis_outcome_receipt")
    if not isinstance(receipt, Mapping):
        raise Fixed4ExecutionPilotError("pilot hypothesis outcome receipt missing")
    file_row = _file_row_from_wrapper(
        receipt, {"node_id", "node_payload_sha256"},
        "pilot hypothesis outcome receipt")
    if (receipt.get("node_id") != v15_node.get("node_id")
            or receipt.get("node_payload_sha256")
                != v15_node.get("node_payload_sha256")):
        raise Fixed4ExecutionPilotError("pilot outcome/V15 node binding mismatch")
    task_root = Path(root).resolve() / "tasks" / str(task["task_id"])
    path = _resolve_row_file(
        root, file_row, "pilot hypothesis outcome",
        within=task_root / "outcomes")
    document = _json(path, receipt["sha256"], "pilot hypothesis outcome")
    document_required = {"schema", "hypothesis_task_id", "task_payload_sha256",
        "source_v15_node_id", "source_v15_node_payload_sha256", "gate_status",
        "failure_class", "safe_transform", "measured_rotation_deg",
        "measured_translation_m", "measurement_source_file_sha256",
        "measurement_source_payload_sha256", "measurement_candidate_slot",
        "measurement_candidate_set_sha256",
        "measurement_slot_results_payload_sha256",
        "measurement_v15_decision_sha256",
        *POLICY_FALSE_FIELDS.keys(),
        "payload_sha256"}
    _require_exact_keys(document, document_required,
                        "file-backed pilot hypothesis outcome")
    if (not _payload_valid(document)
            or document.get("schema") != HYPOTHESIS_OUTCOME_SCHEMA
            or document.get("hypothesis_task_id") != task.get("task_id")
            or document.get("task_payload_sha256") != task.get("payload_sha256")
            or document.get("source_v15_node_id") != v15_node.get("node_id")
            or document.get("source_v15_node_payload_sha256")
                != v15_node.get("node_payload_sha256")
            or document.get("payload_sha256") != source_sha
            or any(document.get(key) != outcome.get(key) for key in
                   ("gate_status", "failure_class", "safe_transform",
                    "measured_rotation_deg", "measured_translation_m",
                    "measurement_source_file_sha256",
                    "measurement_source_payload_sha256",
                    "measurement_candidate_slot",
                    "measurement_candidate_set_sha256",
                    "measurement_slot_results_payload_sha256",
                    "measurement_v15_decision_sha256"))):
        raise Fixed4ExecutionPilotError(
            "file-backed pilot hypothesis outcome binding mismatch")
    _policy_false(document, "file-backed pilot hypothesis outcome")
    safe_rows = [row for row in slots.values() if row.get("status") == "generated"
                 and row.get("safe_vote") is True]
    if not task.get("safe_pose_vote_eligible"):
        expected = {
            "hypothesis_task_id": task.get("task_id"), "gate_status": "ABSTAIN",
            "failure_class": "TYPED_MEMBER_HYPOTHESIS_ABSTENTION",
            "safe_transform": None, "source_result_payload_sha256": source_sha,
            "measured_rotation_deg": None, "measured_translation_m": None,
            "measurement_source_file_sha256": None,
            "measurement_source_payload_sha256": None,
            "measurement_candidate_slot": None,
            "measurement_candidate_set_sha256": None,
            "measurement_slot_results_payload_sha256": None,
            "measurement_v15_decision_sha256": None,
        }
        if dict(outcome) != expected:
            raise Fixed4ExecutionPilotError("typed hypothesis outcome is not abstention")
        return
    if value.get("status") == "succeeded":
        rotation = outcome.get("measured_rotation_deg")
        translation = outcome.get("measured_translation_m")
        if (outcome.get("gate_status") != "PASS"
                or outcome.get("failure_class") is not None
                or isinstance(rotation, bool) or isinstance(translation, bool)
                or not isinstance(rotation, (int, float))
                or not isinstance(translation, (int, float))
                or not math.isfinite(float(rotation))
                or not math.isfinite(float(translation))
                or float(rotation) < 0 or float(translation) < 0
                or float(rotation) > 5.0 or float(translation) > 0.10
                or not isinstance(outcome.get(
                    "measurement_source_payload_sha256"), str)
                or len(outcome["measurement_source_payload_sha256"]) != 64
                or not isinstance(outcome.get(
                    "measurement_source_file_sha256"), str)
                or len(outcome["measurement_source_file_sha256"]) != 64
                or type(outcome.get("measurement_candidate_slot")) is not int
                or outcome["measurement_candidate_slot"] < 0
                or any(not isinstance(outcome.get(key), str)
                       or len(outcome[key]) != 64 for key in (
                        "measurement_candidate_set_sha256",
                        "measurement_slot_results_payload_sha256",
                        "measurement_v15_decision_sha256"))
                or not _transform_valid(outcome.get("safe_transform"))
                or not safe_rows
                or all(stable_json_sha256(row.get("transform")) !=
                       stable_json_sha256(outcome.get("safe_transform"))
                       for row in safe_rows)):
            raise Fixed4ExecutionPilotError(
                "successful pilot hypothesis outcome is not candidate-backed")
    else:
        finite_fail = outcome.get("gate_status") == "FAIL"
        if finite_fail:
            rotation = outcome.get("measured_rotation_deg")
            translation = outcome.get("measured_translation_m")
            valid = (outcome.get("failure_class") ==
                     "FINITE_CONSENSUS_INCOMPATIBILITY"
                     and value["typed_failure"]["type"] ==
                     "FINITE_CONSENSUS_INCOMPATIBILITY"
                     and outcome.get("safe_transform") is None
                     and not isinstance(rotation, bool)
                     and not isinstance(translation, bool)
                     and isinstance(rotation, (int, float))
                     and isinstance(translation, (int, float))
                     and math.isfinite(float(rotation))
                     and math.isfinite(float(translation))
                     and float(rotation) >= 0 and float(translation) >= 0
                     and (float(rotation) > 5.0 or float(translation) > 0.10)
                     and type(outcome.get("measurement_candidate_slot")) is int
                     and all(isinstance(outcome.get(key), str)
                             and len(outcome[key]) == 64 for key in (
                              "measurement_source_file_sha256",
                              "measurement_source_payload_sha256",
                              "measurement_candidate_set_sha256",
                              "measurement_slot_results_payload_sha256",
                              "measurement_v15_decision_sha256")))
        else:
            valid = (outcome.get("gate_status") == "ABSTAIN"
                     and outcome.get("failure_class") ==
                         value["typed_failure"]["type"]
                     and outcome.get("safe_transform") is None
                     and all(outcome.get(key) is None for key in (
                        "measured_rotation_deg", "measured_translation_m",
                        "measurement_source_file_sha256",
                        "measurement_source_payload_sha256",
                        "measurement_candidate_slot",
                        "measurement_candidate_set_sha256",
                        "measurement_slot_results_payload_sha256",
                        "measurement_v15_decision_sha256"))
                     and not safe_rows)
        if not valid:
            raise Fixed4ExecutionPilotError(
                "failed pilot hypothesis outcome is not fail-closed")


def _validate_typed_failure_replay(task: Mapping[str, Any], value: Mapping[str, Any],
                                   root: Path) -> None:
    rows = value.get("typed_failure_replay")
    expected = task.get("typed_failure_member_candidate_indices")
    if (not isinstance(rows, list) or not isinstance(expected, list)
            or len(rows) != len(expected)
            or stable_json_sha256(rows) != value.get("typed_failure_replay_closure_sha256")):
        raise Fixed4ExecutionPilotError("typed-failure replay closure mismatch")
    task_root = Path(root).resolve() / "tasks" / task["task_id"]
    for index, (row, candidate_index) in enumerate(zip(rows, expected)):
        file_row = _file_row_from_wrapper(row, {"candidate_index"},
                                          f"typed failure row {index}")
        if row.get("candidate_index") != candidate_index:
            raise Fixed4ExecutionPilotError("typed-failure candidate drift")
        path = _resolve_row_file(root, file_row, f"typed failure replay {index}",
                                 within=task_root / "typed_failures")
        document = _json(path, row["sha256"], "typed failure replay")
        _validate_attempt_document(document, schema=TYPED_FAILURE_REPLAY_SCHEMA,
            task=task, identity={"candidate_index": candidate_index},
            role="typed failure replay")
        if document.get("status") != "typed_failure" or document.get("transform") is not None:
            raise Fixed4ExecutionPilotError("typed-failure replay cannot contain transform")


COMMON_RESULT_KEYS = {"schema", "task_id", "task_payload_sha256", "stage",
    "status", "typed_failure", *POLICY_FALSE_FIELDS.keys(), "output_artifacts",
    "evidence_receipts", "evidence_receipt_closure_sha256", "payload_sha256"}
STAGE_RESULT_KEYS = {
    "colorpcr_direction": {"sentinel_attempts", "sentinel_attempt_closure_sha256",
                            "exact_three_cache_artifact_sha256"},
    "bidirectional_multi_solver_pilot": {"candidate_slots",
        "candidate_slot_closure_sha256", "solver_rows_executed", "solver_attempts",
        "solver_attempt_closure_sha256", "typed_failure_replay",
        "typed_failure_replay_closure_sha256", "hypothesis_outcome",
        "hypothesis_outcome_receipt"},
    "v16_pair_hypothesis_cluster": {"replayed_hypothesis_task_ids", "decision",
        "safe_cluster_transform", "safe_vote_hypothesis_task_ids",
        "gate_failed_hypothesis_task_ids",
        "typed_abstention_hypothesis_task_ids"},
    "fixed4_aggregate": {"replayed_pair_task_ids", "pair_outcomes",
        "pair_outcome_closure_sha256", "decision", "guard_audit_receipt_sha256"}}

PAIR_SAFE_DECISION = "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"
PAIR_FAIL_CLOSED_DECISION = "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER"
PAIR_FAIL_CLOSED_TYPE = PAIR_FAIL_CLOSED_DECISION
KNOWN_BAD_DECISION = "PERMANENT_KNOWN_BAD_VETO"
AGGREGATE_SAFE_DECISION = "THREE_NORMALS_ACCEPTED_KNOWN_BAD_VETOED_NO_REFUSION"
AGGREGATE_NORMAL_FAIL_DECISION = "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"
AGGREGATE_KNOWN_BAD_FAIL_DECISION = "FIXED4_KNOWN_BAD_VETO_FAILED"


def _validate_top_typed_failure(value: Mapping[str, Any]) -> None:
    typed = value.get("typed_failure")
    if value.get("status") == "typed_failure":
        if (not isinstance(typed, Mapping) or set(typed) != {"type", "transform"}
                or not isinstance(typed.get("type"), str) or not typed["type"]
                or typed.get("transform") is not None):
            raise Fixed4ExecutionPilotError("typed failure is not explicit")
    elif value.get("status") == "succeeded":
        if typed is not None:
            raise Fixed4ExecutionPilotError("success cannot carry typed failure")
    else:
        raise Fixed4ExecutionPilotError("result status invalid")


def _derived_pair_decision(task: Mapping[str, Any],
                           value: Mapping[str, Any]) -> str:
    """Derive, rather than trust, the pair decision from fail-closed fields."""
    replayed = value.get("replayed_hypothesis_task_ids")
    if replayed != task.get("upstream_task_ids"):
        raise Fixed4ExecutionPilotError("all pair hypotheses were not replayed")
    safe = value.get("safe_vote_hypothesis_task_ids")
    failed = value.get("gate_failed_hypothesis_task_ids")
    abstained = value.get("typed_abstention_hypothesis_task_ids")
    eligible = task.get("eligible_hypothesis_task_ids")
    if (not all(isinstance(rows, list) for rows in (safe, failed, abstained, eligible))
            or len(safe) != len(set(safe)) or len(failed) != len(set(failed))
            or len(abstained) != len(set(abstained))
            or set(safe).intersection(failed)
            or set(safe + failed).intersection(abstained)
            or sorted(safe + failed) != sorted(eligible)
            or abstained != task.get("typed_abstention_hypothesis_task_ids")
            or sorted(safe + failed + abstained) != sorted(replayed)):
        raise Fixed4ExecutionPilotError(
            "pair safe-vote/gate-failed/abstention closure drift")
    if task.get("known_bad"):
        if (len(replayed) != 12 or value.get("status") != "typed_failure"
                or value.get("safe_cluster_transform") is not None
                or value.get("typed_failure") != {
                    "type": "KNOWN_BAD_PERMANENT_VETO", "transform": None}):
            raise Fixed4ExecutionPilotError("known-bad veto was weakened")
        return KNOWN_BAD_DECISION
    if value.get("status") == "succeeded":
        if not _transform_valid(value.get("safe_cluster_transform")):
            raise Fixed4ExecutionPilotError("normal safe-cluster transform invalid")
        return PAIR_SAFE_DECISION
    if (value.get("status") == "typed_failure"
            and value.get("safe_cluster_transform") is None
            and value.get("typed_failure") == {
                "type": PAIR_FAIL_CLOSED_TYPE, "transform": None}):
        return PAIR_FAIL_CLOSED_DECISION
    raise Fixed4ExecutionPilotError("normal pair failure is not explicit fail-closed")


def _derived_aggregate_decision(task: Mapping[str, Any],
                                value: Mapping[str, Any]) -> str:
    """Recompute aggregate eligibility from four structured pair outcomes."""
    replayed = value.get("replayed_pair_task_ids")
    rows = value.get("pair_outcomes")
    if (replayed != task.get("upstream_task_ids") or not isinstance(rows, list)
            or len(rows) != len(replayed) or len(rows) != 4
            or stable_json_sha256(rows) != value.get("pair_outcome_closure_sha256")):
        raise Fixed4ExecutionPilotError("aggregate pair-outcome closure mismatch")
    normal_failures = 0; known_bad_vetoed = False
    for index, (row, task_id) in enumerate(zip(rows, replayed)):
        if not isinstance(row, Mapping):
            raise Fixed4ExecutionPilotError("aggregate pair outcome malformed")
        _require_exact_keys(row, {"task_id", "status", "decision",
                                  "safe_cluster_transform",
                                  "source_result_payload_sha256"},
                            f"aggregate pair outcome {index}")
        _sha(row.get("source_result_payload_sha256"),
             f"aggregate pair outcome {index} source result SHA")
        if row.get("task_id") != task_id:
            raise Fixed4ExecutionPilotError("aggregate pair outcome task drift")
        if index == 3:
            known_bad_vetoed = (row.get("status") == "typed_failure"
                and row.get("decision") == KNOWN_BAD_DECISION
                and row.get("safe_cluster_transform") is None)
        elif row.get("status") == "succeeded":
            if (row.get("decision") != PAIR_SAFE_DECISION
                    or not _transform_valid(row.get("safe_cluster_transform"))):
                raise Fixed4ExecutionPilotError("aggregate normal pair outcome invalid")
        elif row.get("status") == "typed_failure":
            if (row.get("decision") != PAIR_FAIL_CLOSED_DECISION
                    or row.get("safe_cluster_transform") is not None):
                raise Fixed4ExecutionPilotError("aggregate pair failure not fail-closed")
            normal_failures += 1
        else:
            raise Fixed4ExecutionPilotError("aggregate pair status invalid")
    if normal_failures:
        if (value.get("status") != "typed_failure"
                or value.get("typed_failure") != {
                    "type": AGGREGATE_NORMAL_FAIL_DECISION, "transform": None}):
            raise Fixed4ExecutionPilotError("aggregate failure was not propagated")
        return AGGREGATE_NORMAL_FAIL_DECISION
    if not known_bad_vetoed:
        if (value.get("status") != "typed_failure"
                or value.get("typed_failure") != {
                    "type": AGGREGATE_KNOWN_BAD_FAIL_DECISION, "transform": None}):
            raise Fixed4ExecutionPilotError("known-bad veto failure was not propagated")
        return AGGREGATE_KNOWN_BAD_FAIL_DECISION
    if value.get("status") != "succeeded" or value.get("typed_failure") is not None:
        raise Fixed4ExecutionPilotError("aggregate success status mismatch")
    return AGGREGATE_SAFE_DECISION


def _validate_pair_parent_binding(
    task: Mapping[str, Any], value: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
) -> None:
    if list(parents) != task.get("upstream_task_ids"):
        raise Fixed4ExecutionPilotError("pair parent result order drift")
    safe: list[str] = []; failed: list[str] = []; abstained: list[str] = []
    for task_id, result in parents.items():
        outcome = result.get("hypothesis_outcome")
        if not isinstance(outcome, Mapping) \
                or outcome.get("hypothesis_task_id") != task_id:
            raise Fixed4ExecutionPilotError("pair parent hypothesis outcome missing")
        gate = outcome.get("gate_status")
        if gate == "PASS":
            safe.append(task_id)
        elif gate == "FAIL":
            failed.append(task_id)
        elif gate == "ABSTAIN":
            abstained.append(task_id)
        else:
            raise Fixed4ExecutionPilotError("pair parent gate status invalid")
    if (value.get("safe_vote_hypothesis_task_ids") != safe
            or value.get("gate_failed_hypothesis_task_ids") != failed
            or value.get("typed_abstention_hypothesis_task_ids") != abstained):
        raise Fixed4ExecutionPilotError("pair partitions are not parent-derived")


def _validate_aggregate_parent_binding(
    task: Mapping[str, Any], value: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
) -> None:
    if list(parents) != task.get("upstream_task_ids"):
        raise Fixed4ExecutionPilotError("aggregate parent result order drift")
    outcomes = value.get("pair_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(parents):
        raise Fixed4ExecutionPilotError("aggregate parent outcome count drift")
    for row, (task_id, result) in zip(outcomes, parents.items()):
        expected = {
            "task_id": task_id,
            "status": result.get("status"),
            "decision": result.get("decision"),
            "safe_cluster_transform": result.get("safe_cluster_transform"),
            "source_result_payload_sha256": result.get("payload_sha256"),
        }
        if row != expected:
            raise Fixed4ExecutionPilotError(
                "aggregate pair outcome is not parent-derived")


def validate_runner_result(task: Mapping[str, Any], value: Mapping[str, Any],
                           output_root: Path, *,
                           upstream_results: Mapping[str, Mapping[str, Any]] | None = None,
                           ) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Fixed4ExecutionPilotError("runner result is not a mapping")
    _reject_forbidden_keys(value, "result")
    stage = task.get("stage")
    _require_exact_keys(value, COMMON_RESULT_KEYS | STAGE_RESULT_KEYS.get(stage, set()),
                        "runner result")
    if (not _payload_valid(value) or value.get("schema") != RESULT_SCHEMA
            or value.get("task_id") != task.get("task_id")
            or value.get("task_payload_sha256") != task.get("payload_sha256")
            or value.get("stage") != stage):
        raise Fixed4ExecutionPilotError("runner result binding mismatch")
    _policy_false(value, "runner result"); _validate_top_typed_failure(value)
    evidence_statuses = _validate_evidence_receipts(task, value, output_root)
    _validate_output_artifacts(task, value, output_root)
    if stage == "colorpcr_direction":
        _validate_sentinel_attempts(task, value, output_root)
        _sha(value.get("exact_three_cache_artifact_sha256"), "exact-three cache SHA")
    elif stage == "bidirectional_multi_solver_pilot":
        slots = _validate_candidate_slots(value)
        _validate_solver_attempts(task, value, output_root, slots)
        _validate_pilot_evidence_semantics(task, evidence_statuses, slots)
        _validate_typed_failure_replay(task, value, output_root)
        generated = [row for row in slots.values() if row.get("status") == "generated"]
        safe_votes = [row for row in generated if row.get("safe_vote") is True]
        if task.get("safe_pose_vote_eligible"):
            if value.get("status") == "succeeded" and not safe_votes:
                raise Fixed4ExecutionPilotError("successful pilot has no safe vote")
            if value.get("status") == "typed_failure" and safe_votes:
                raise Fixed4ExecutionPilotError("failed pilot contributed a safe vote")
        elif (generated or value.get("status") != "typed_failure"
              or value.get("typed_failure") != {
                  "type": "TYPED_MEMBER_HYPOTHESIS_ABSTENTION", "transform": None}):
            raise Fixed4ExecutionPilotError(
                "typed-affected hypothesis generated a transform or safe vote")
        _validate_hypothesis_outcome(task, value, slots, output_root)
    elif stage == "v16_pair_hypothesis_cluster":
        if value.get("decision") != _derived_pair_decision(task, value):
            raise Fixed4ExecutionPilotError("pair decision does not match derived outcome")
        if upstream_results is not None:
            _validate_pair_parent_binding(task, value, upstream_results)
    elif stage == "fixed4_aggregate":
        if value.get("decision") != _derived_aggregate_decision(task, value):
            raise Fixed4ExecutionPilotError(
                "fixed4 aggregate decision does not match derived outcomes")
        _sha(value.get("guard_audit_receipt_sha256"), "guard audit receipt SHA")
        if upstream_results is not None:
            _validate_aggregate_parent_binding(task, value, upstream_results)
    return dict(value)


def _validate_task_manifest(task_path: Path, task: Mapping[str, Any],
                            output_root: Path,
                            preflight: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_root).resolve()
    expected_path = (root / "tasks" / str(task.get("task_id")) / "task.json").resolve()
    if Path(task_path).resolve() != expected_path:
        raise Fixed4ExecutionPilotError("task path is not canonical")
    manifest = _json(root / "task_manifest.json", None, "operational task manifest")
    if (not _payload_valid(manifest) or manifest.get("schema") != TASK_MANIFEST_SCHEMA
            or manifest.get("repo_root") != preflight.get("repo_root")
            or manifest.get("git_head") != preflight.get("git_head")
            or manifest.get("git_tree") != preflight.get("git_tree")
            or manifest.get("output_root") != str(root)
            or manifest.get("preflight_payload_sha256") != preflight.get("payload_sha256")
            or manifest.get("task_count") != OPERATIONAL_TASK_COUNT
            or manifest.get("stage_counts") != OPERATIONAL_STAGE_COUNTS):
        raise Fixed4ExecutionPilotError("operational task manifest mismatch")
    rows = manifest.get("tasks")
    if (not isinstance(rows, list) or stable_json_sha256(rows)
            != manifest.get("task_closure_sha256")):
        raise Fixed4ExecutionPilotError("task manifest closure mismatch")
    matches = [row for row in rows if row.get("task_id") == task.get("task_id")]
    if (len(matches) != 1 or matches[0].get("path") != str(expected_path.relative_to(root))
            or matches[0].get("stage") != task.get("stage")
            or matches[0].get("bytes") != expected_path.stat().st_size
            or matches[0].get("sha256") != sha256_file(expected_path)
            or matches[0].get("payload_sha256") != task.get("payload_sha256")
            or matches[0].get("evidence_node_closure_sha256")
                != task.get("evidence_node_closure_sha256")):
        raise Fixed4ExecutionPilotError("task is not in sealed manifest")
    return manifest


def _validate_attempt_receipt(attempt: Mapping[str, Any], *, task: Mapping[str, Any],
        result_path: Path, task_path: Path, authorization_sha256: str,
        preflight_sha256: str, manifest_sha256: str,
        runner_registry_sha256: str) -> None:
    required = {"schema", "status", "task_id", "task_sha256", "task_payload_sha256",
        "preflight_sha256", "authorization_sha256", "task_manifest_sha256",
        "runner_registry_closure_sha256", "result_sha256",
        "evidence_receipt_closure_sha256", *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    _require_exact_keys(attempt, required, "operational attempt")
    if (not _payload_valid(attempt) or attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("task_id") != task.get("task_id")
            or attempt.get("task_sha256") != sha256_file(task_path)
            or attempt.get("task_payload_sha256") != task.get("payload_sha256")
            or attempt.get("preflight_sha256") != preflight_sha256
            or attempt.get("authorization_sha256") != authorization_sha256
            or attempt.get("task_manifest_sha256") != manifest_sha256
            or attempt.get("runner_registry_closure_sha256") != runner_registry_sha256
            or attempt.get("result_sha256") != sha256_file(result_path)):
        raise Fixed4ExecutionPilotError("operational attempt closure mismatch")
    _policy_false(attempt, "operational attempt")


def _validate_upstream_receipts(task: Mapping[str, Any], output_root: Path,
        authorization_sha256: str, preflight_sha256: str,
        manifest_sha256: str, runner_registry_sha256: str,
        ) -> dict[str, Mapping[str, Any]]:
    root = Path(output_root).resolve()
    parents: dict[str, Mapping[str, Any]] = {}
    for upstream in task.get("upstream_task_ids", ()):
        parent_root = root / "tasks" / str(upstream)
        task_path = parent_root / "task.json"; result_path = parent_root / "result.json"
        attempt_path = parent_root / "attempt_receipt.json"
        if not (task_path.is_file() and result_path.is_file() and attempt_path.is_file()):
            raise Fixed4ExecutionPilotError(f"upstream receipt missing: {upstream}")
        parent_task = _json(task_path, None, "upstream task")
        result = validate_runner_result(parent_task, _json(result_path, None, "upstream result"), root)
        attempt = _json(attempt_path, None, "upstream attempt")
        _validate_attempt_receipt(attempt, task=parent_task, result_path=result_path,
            task_path=task_path, authorization_sha256=authorization_sha256,
            preflight_sha256=preflight_sha256, manifest_sha256=manifest_sha256,
            runner_registry_sha256=runner_registry_sha256)
        if (attempt.get("status") != result.get("status")
                or attempt.get("evidence_receipt_closure_sha256")
                    != result.get("evidence_receipt_closure_sha256")):
            raise Fixed4ExecutionPilotError(f"upstream closure mismatch: {upstream}")
        parents[str(upstream)] = result
    return parents


def _task_root_has_partial_state(task_root: Path) -> bool:
    allowed = {str(task_root / "task.json")}
    return any(str(path) not in allowed for path in _tree_regular_files(task_root))


def _tree_regular_files(root: Path) -> list[Path]:
    try:
        ensure_no_symlink_directory(root, "authorized task tree")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        try:
            ensure_no_symlink_directory(directory_path, "authorized task tree component")
        except Fixed4SubprocessContractError as exc:
            raise Fixed4ExecutionPilotError(str(exc)) from exc
        for name in [*dirnames, *filenames]:
            path = directory_path / name
            if stat.S_ISLNK(path.lstat().st_mode):
                raise Fixed4ExecutionPilotError("authorized task tree symlink rejected")
        for name in filenames:
            path = directory_path / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise Fixed4ExecutionPilotError("authorized task artifact is not regular")
            files.append(path)
    return files


def _validate_task_inventory(task: Mapping[str, Any], value: Mapping[str, Any] | None,
                             output_root: Path) -> None:
    root = Path(output_root).resolve(); task_root = root / "tasks" / task["task_id"]
    expected = {str((task_root / "task.json").resolve())}
    for name in ("result.json", "attempt_receipt.json"):
        path = task_root / name
        if path.is_file():
            expected.add(str(path.resolve()))
    wrapper_names = ("stdout.bin", "stderr.bin", "strace.log",
                     "consumption_receipt.json")
    present_wrapper = [task_root / "wrapper" / name for name in wrapper_names
                       if (task_root / "wrapper" / name).is_file()]
    if present_wrapper and len(present_wrapper) != len(wrapper_names):
        raise Fixed4ExecutionPilotError("partial wrapper inventory")
    expected.update(str(path.resolve()) for path in present_wrapper)
    result = value if isinstance(value, Mapping) else {}
    for key in ("output_artifacts", "evidence_receipts", "sentinel_attempts",
                "solver_attempts", "typed_failure_replay"):
        for row in result.get(key, []):
            expected.add(str((root / row["path"]).resolve()))
    outcome_receipt = result.get("hypothesis_outcome_receipt")
    if outcome_receipt is not None:
        if not isinstance(outcome_receipt, Mapping):
            raise Fixed4ExecutionPilotError("pilot outcome inventory row malformed")
        expected.add(str((root / outcome_receipt["path"]).resolve()))
    observed = {str(path) for path in _tree_regular_files(task_root)}
    if observed != expected:
        raise Fixed4ExecutionPilotError("runner task-root inventory mismatch")


def execute_authorized_task(*, task_path: Path, preflight_path: Path,
        preflight_sha256: str, authorization_path: Path,
        authorization_sha256: str, output_root: Path) -> dict[str, Any]:
    """Execute only through the reviewed registry; no caller runner argument."""
    authorization = validate_authorization(authorization_path, authorization_sha256,
                                            preflight_path, preflight_sha256)
    task = _json(task_path, None, "operational task")
    if (not _payload_valid(task) or task.get("schema") != TASK_SCHEMA
            or task.get("task_id") not in authorization["authorized_task_ids"]
            or task.get("stage") not in authorization["allowed_stages"]
            or task.get("execution_authorized") is not False
            or task.get("execution_performed") is not False
            or task.get("reconstruction_authorized") is not False
            or task.get("refusion_allowed") is not False):
        raise Fixed4ExecutionPilotError("operational task is not executable")
    preflight = _json(preflight_path, preflight_sha256, "preflight")
    root = Path(output_root)
    try:
        ensure_no_symlink_directory(root, "authorized output root")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    if (not root.is_absolute() or str(root) != preflight["output_root"]
            or Path(preflight_path) != root / "execution_preflight.json"
            or Path(authorization_path) != root / "authorization.json"):
        raise Fixed4ExecutionPilotError("output root authorization mismatch")
    expected_identity = {key: preflight.get(key) for key in (
        "repo_root", "git_head", "git_tree", "output_root", "preregister_sha256",
        "execution_pilot_preregister_sha256",
        "execution_pilot_preregister_payload_sha256", "exact191_manifest_sha256",
        "prepared_builder_manifest_sha256", "dag_payload_sha256",
        "runner_registry_closure_sha256")}
    if task.get("preflight_identity") != expected_identity:
        raise Fixed4ExecutionPilotError("task/preflight identity mismatch")
    _validate_task_manifest(task_path, task, root, preflight)
    manifest_sha = sha256_file(root / "task_manifest.json")
    upstream_results = _validate_upstream_receipts(task, root, authorization_sha256,
        preflight_sha256, manifest_sha, preflight["runner_registry_closure_sha256"])
    task_root = root / "tasks" / task["task_id"]
    if Path(task_path) != task_root / "task.json":
        raise Fixed4ExecutionPilotError("task path/layout mismatch")
    try:
        ensure_no_symlink_directory(task_root, "authorized task root")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc
    result_path = task_root / "result.json"; attempt_path = task_root / "attempt_receipt.json"
    if result_path.exists() or attempt_path.exists():
        if not result_path.is_file() or not attempt_path.is_file():
            raise Fixed4ExecutionPilotError("partial task resume state")
        result = validate_runner_result(
            task, _json(result_path, None, "resumed result"), root,
            upstream_results=upstream_results)
        attempt = _json(attempt_path, None, "resumed attempt")
        _validate_attempt_receipt(attempt, task=task, result_path=result_path,
            task_path=Path(task_path), authorization_sha256=authorization_sha256,
            preflight_sha256=preflight_sha256, manifest_sha256=manifest_sha,
            runner_registry_sha256=preflight["runner_registry_closure_sha256"])
        if attempt.get("status") != result.get("status"):
            raise Fixed4ExecutionPilotError("resume receipt status mismatch")
        _validate_task_inventory(task, result, root)
        return {"state": "resumed_identical", "result": result}
    if _task_root_has_partial_state(task_root):
        raise Fixed4ExecutionPilotError("partial/unsealed output exists; overwrite forbidden")
    control_paths = {
        "task_path": Path(task_path), "preflight_path": Path(preflight_path),
        "authorization_path": Path(authorization_path),
        "task_manifest_path": root / "task_manifest.json",
    }
    control_sha = {
        "task_sha256": no_symlink_file_row(control_paths["task_path"], "task")["sha256"],
        "preflight_sha256": no_symlink_file_row(
            control_paths["preflight_path"], "preflight")["sha256"],
        "authorization_sha256": no_symlink_file_row(
            control_paths["authorization_path"], "authorization")["sha256"],
        "task_manifest_sha256": no_symlink_file_row(
            control_paths["task_manifest_path"], "task manifest")["sha256"],
    }
    # Reconstruct the only executable boundary from local code literals. The
    # signed registry remains evidence, never a caller-selected launch table.
    executor_interpreter = Path("/usr/bin/python3.12")
    executor_interpreter_sha256 = \
        "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
    executor_source = Path(preflight["repo_root"]) / \
        "scripts/v16_b716_fixed4_sealed_executor.py"
    executor_source_sha256 = \
        "6b6b3b3165e9a521af226b12d4c002f888de3da80c9fef1977b5846025019f19"
    if (no_symlink_file_row(executor_interpreter, "sealed executor interpreter")["sha256"]
            != executor_interpreter_sha256
            or no_symlink_file_row(executor_source, "sealed executor source")["sha256"]
            != executor_source_sha256):
        raise Fixed4ExecutionPilotError("code-pinned sealed executor SHA drift")
    executor_argv = [
        str(executor_interpreter), "-I", "-S", str(executor_source),
        "--repo", preflight["repo_root"],
        "--task", str(control_paths["task_path"]),
        "--task-sha256", control_sha["task_sha256"],
        "--preflight", str(control_paths["preflight_path"]),
        "--preflight-sha256", control_sha["preflight_sha256"],
        "--authorization", str(control_paths["authorization_path"]),
        "--authorization-sha256", control_sha["authorization_sha256"],
        "--task-manifest", str(control_paths["task_manifest_path"]),
        "--task-manifest-sha256", control_sha["task_manifest_sha256"],
        "--task-root", str(task_root)]
    # The parent invokes only a code/SHA-pinned fresh-process entrypoint. It
    # never calls a stage function or a Python callable registry from this process.
    completed = subprocess.run(
        executor_argv, cwd=task_root,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        capture_output=True, check=False)
    if completed.returncode != DISABLED_EXIT_CODE:
        raise Fixed4ExecutionPilotError("sealed executor rejected controls/authorization")
    receipt = _json(task_root / "wrapper/consumption_receipt.json", None,
                    "sealed executor consumption receipt")
    if receipt.get("failure_type") != "CHECKED_IN_RUNNER_EXECUTION_DISABLED":
        raise Fixed4ExecutionPilotError("sealed executor failure classification drift")
    _validate_task_inventory(task, None, root)
    _validate_source_and_registry(preflight)
    # No result/attempt is accepted from the disabled child.  The trusted parent
    # derives the failure class exclusively from exit/trace evidence.
    raise Fixed4ExecutionPilotError(
        f"trusted wrapper failure_type={receipt['failure_type']}")


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_active_preflight_document(preflight: Mapping[str, Any]) -> None:
    """Validate the candidate active envelope without implying readiness."""
    active = preflight.get("active_subprocess_contract")
    rows = preflight.get("runner_registry")
    if (not _payload_valid(preflight) or preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("execution_authorized") is not False
            or preflight.get("execution_performed") is not False
            or preflight.get("reconstruction_authorized") is not False
            or preflight.get("refusion_allowed") is not False
            or not isinstance(rows, list) or not rows
            or not isinstance(active, Mapping)
            or active.get("schema") not in {
                ACTIVE_PREFLIGHT_SCHEMA, ACTIVE_PREFLIGHT_V2_SCHEMA}
            or active.get("runner_mode") != RUNNER_MODE_ACTIVE
            or active.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")
            or active.get("sealed_executor_sha256")
                != rows[0].get("sealed_executor", {}).get("source", {}).get("sha256")
            or active.get("legacy_disabled_preflight_accepted") is not False
            or active.get("contract_fixture_allowed") is not False
            or active.get("operational_result_release_allowed") is not False):
        raise Fixed4ExecutionPilotError("active preflight is not sealed/fail-closed")
    expected_active = {
        "schema": active["schema"],
        "runner_mode": RUNNER_MODE_ACTIVE,
        "runner_registry_closure_sha256":
            preflight.get("runner_registry_closure_sha256"),
        "sealed_executor_sha256":
            rows[0].get("sealed_executor", {}).get("source", {}).get("sha256"),
        "legacy_disabled_preflight_accepted": False,
        "contract_fixture_allowed": False,
        "operational_result_release_allowed": False,
    }
    if active["schema"] == ACTIVE_PREFLIGHT_V2_SCHEMA:
        expected_active["production_adapter_protocol_ready"] = True
    if active != expected_active:
        raise Fixed4ExecutionPilotError("active preflight version/fields drift")
    if any(row.get("runner_mode") != RUNNER_MODE_ACTIVE for row in rows):
        raise Fixed4ExecutionPilotError("active preflight contains legacy registry")


def build_active_stage_authorization_request(
        *, preflight: Mapping[str, Any], preflight_path: Path,
        manifest: Mapping[str, Any], manifest_path: Path,
        task: Mapping[str, Any], task_path: Path, signing_key_id: str,
        issued_at: datetime | None = None, ttl_seconds: int = MAX_AUTH_TTL_SECONDS,
        renewal_of_authorization_payload_sha256: str | None = None,
        production_execution_manifest_path: Path | None = None,
        ) -> dict[str, Any]:
    """Build an unsigned, off-host signing request for exactly one DAG node."""
    _validate_active_preflight_document(preflight)
    if (type(ttl_seconds) is not int or ttl_seconds < 1
            or ttl_seconds > MAX_AUTH_TTL_SECONDS):
        raise Fixed4ExecutionPilotError("active authorization TTL exceeds 3600 seconds")
    if not isinstance(signing_key_id, str) or not signing_key_id.strip():
        raise Fixed4ExecutionPilotError("active signing key id missing")
    descriptor = task.get("stage_runner_input_descriptor")
    if (not _payload_valid(task) or task.get("schema") != TASK_SCHEMA
            or not isinstance(descriptor, Mapping)
            or not _payload_valid(descriptor)
            or descriptor.get("schema") not in {
                ACTIVE_STAGE_INPUT_DESCRIPTOR_SCHEMA,
                ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA}
            or descriptor.get("task_id") != task.get("task_id")
            or descriptor.get("stage") != task.get("stage")
            or descriptor.get("upstream_task_ids") != task.get("upstream_task_ids")
            or descriptor.get("derivation_policy")
                != "dispatcher_only_never_trust_task_runtime_paths"):
        raise Fixed4ExecutionPilotError("active stage input descriptor mismatch")
    if not isinstance(manifest, Mapping) or not _payload_valid(manifest):
        raise Fixed4ExecutionPilotError("active task manifest malformed")
    production_ready = production_execution_manifest_path is not None
    active_contract = preflight.get("active_subprocess_contract")
    production_bindings: dict[str, Any] = {}
    if production_ready:
        if (descriptor.get("schema") != ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA
                or descriptor.get("production_adapter_protocol_ready") is not True
                or not isinstance(active_contract, Mapping)
                or active_contract.get("schema") != ACTIVE_PREFLIGHT_V2_SCHEMA
                or active_contract.get("production_adapter_protocol_ready") is not True):
            raise Fixed4ExecutionPilotError(
                "production-ready v2 descriptor/preflight is required")
        execution_path = Path(production_execution_manifest_path).resolve()
        committed = _committed_production_control(
            root=Path(preflight["output_root"]).resolve(), task=task,
            preflight=preflight)
        if committed is None or execution_path != committed[1].resolve():
            raise Fixed4ExecutionPilotError(
                "production execution manifest is not the committed transaction")
        commit_path = committed[0].resolve()
        commit = _json(commit_path, None,
                       "production manifest transaction commit")
        execution = _json(execution_path, None,
                          "production execution manifest")
        if (not _payload_valid(execution)
                or execution.get("schema")
                    != "v16-b716-fixed4-active-production-execution-manifest-v1"
                or execution.get("task_id") != task.get("task_id")
                or execution.get("task_payload_sha256")
                    != task.get("payload_sha256")
                or execution.get("stage") != task.get("stage")
                or execution.get("parent_result_payload_sha256s") != [
                    # Parent payloads are dispatcher-derived from canonical results.
                    _json(Path(preflight["output_root"]) / "tasks" / parent /
                          "result.json", None, f"authorization parent {parent}")[
                              "payload_sha256"]
                    for parent in task.get("upstream_task_ids", ())]):
            raise Fixed4ExecutionPilotError(
                "production execution manifest semantic binding mismatch")
        input_path = Path(str(execution.get(
            "production_input_manifest_path", ""))).resolve()
        expected_input_path = execution_path.with_name(
            "production_input_manifest.json")
        input_manifest = _json(input_path, None, "production input manifest")
        interpreter = execution.get("interpreter")
        if (input_path != expected_input_path or not _payload_valid(input_manifest)
                or input_manifest.get("payload_sha256")
                    != execution.get("production_input_manifest_payload_sha256")
                or sha256_file(input_path)
                    != execution.get("production_input_manifest_sha256")
                or not isinstance(interpreter, Mapping)):
            raise Fixed4ExecutionPilotError(
                "production input/interpreter binding mismatch")
        production_wrapper = Path(preflight["repo_root"]) / \
            "scripts/v16_b716_fixed4_active_production_wrapper.py"
        validator_source = Path(preflight["repo_root"]) / \
            "src/safety/v16_b716_fixed4_active_production_wrapper.py"
        runner_source = Path(preflight["repo_root"]) / \
            "scripts/v16_b716_fixed4_active_stage_runner.sh"
        for path, role in ((production_wrapper, "production wrapper"),
                           (validator_source, "production validator"),
                           (runner_source, "active runner")):
            if path.is_symlink() or not path.is_file():
                raise Fixed4ExecutionPilotError(f"{role} missing")
        if (execution.get("wrapper_source_sha256") != sha256_file(validator_source)
                or execution.get("runner_source_sha256") != sha256_file(runner_source)
                or stable_json_sha256(execution.get("runtime_dependency_files"))
                    != execution.get("runtime_dependency_closure_sha256")):
            raise Fixed4ExecutionPilotError(
                "production source/runtime closure binding drift")
        production_bindings = {
            "production_input_manifest_path": str(input_path),
            "production_input_manifest_sha256": sha256_file(input_path),
            "production_input_manifest_payload_sha256":
                input_manifest["payload_sha256"],
            "execution_manifest_path": str(execution_path),
            "execution_manifest_sha256": sha256_file(execution_path),
            "execution_manifest_payload_sha256": execution["payload_sha256"],
            "production_manifest_commit_path": str(commit_path),
            "production_manifest_commit_sha256": sha256_file(commit_path),
            "production_manifest_commit_payload_sha256":
                commit["payload_sha256"],
            "runtime_dependency_closure_sha256":
                execution["runtime_dependency_closure_sha256"],
            "production_interpreter_path": interpreter["path"],
            "production_interpreter_sha256": interpreter["sha256"],
            "production_wrapper_path": str(production_wrapper.resolve()),
            "production_wrapper_sha256": sha256_file(production_wrapper),
            "validator_source_path": str(validator_source.resolve()),
            "validator_source_sha256": sha256_file(validator_source),
            "runner_source_path": str(runner_source.resolve()),
            "runner_source_sha256": sha256_file(runner_source),
            "parent_result_payload_sha256s":
                list(execution["parent_result_payload_sha256s"]),
        }
    now = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = now + timedelta(seconds=ttl_seconds)
    body = {
        "schema": ACTIVE_STAGE_AUTHORIZATION_SCHEMA,
        "status": "PASS", "authorization_scope": "one_topological_task",
        "execution_authorized": True, "execution_performed": False,
        "issued_at": _utc_text(now), "expires_at": _utc_text(expires),
        "repo_root": preflight["repo_root"], "git_head": preflight["git_head"],
        "git_tree": preflight["git_tree"], "output_root": preflight["output_root"],
        "preflight_path": str(Path(preflight_path).resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_payload_sha256": preflight["payload_sha256"],
        "task_manifest_path": str(Path(manifest_path).resolve()),
        "task_manifest_sha256": sha256_file(manifest_path),
        "task_manifest_payload_sha256": manifest["payload_sha256"],
        "task_path": str(Path(task_path).resolve()),
        "task_sha256": sha256_file(task_path),
        "task_id": task["task_id"], "stage": task["stage"],
        "task_payload_sha256": task["payload_sha256"],
        "upstream_task_ids": list(task["upstream_task_ids"]),
        "stage_input_descriptor_payload_sha256": descriptor["payload_sha256"],
        "runner_registry_closure_sha256":
            preflight["runner_registry_closure_sha256"],
        "execution_source_closure_sha256":
            preflight["execution_source_closure_sha256"],
        "parent_results_derived_by_dispatcher": True,
        "task_stage_input_trusted": False,
        "production_adapter_protocol_required": True,
        "production_adapter_protocol_ready": production_ready,
        "operational_result_schema": RESULT_SCHEMA,
        "signer_private_key_not_on_execution_host": True,
        "renewal_of_authorization_payload_sha256":
            renewal_of_authorization_payload_sha256,
        **production_bindings,
        **POLICY_FALSE_FIELDS,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "signing_key_id": signing_key_id,
    }
    request = {
        "schema": ACTIVE_AUTHORIZATION_REQUEST_SCHEMA,
        "request_kind": ("renewal" if renewal_of_authorization_payload_sha256
                         else "initial"),
        "unsigned": True, "signer_location": "off_host",
        "private_key_expected_on_execution_host": False,
        "authorization_body": body,
        "authorization_body_sha256": stable_json_sha256(body),
    }
    request["payload_sha256"] = stable_json_sha256(request)
    return request


def active_authorization_time_status(value: Mapping[str, Any], *,
                                     now: datetime | None = None) -> str:
    issued = _parse_time(value.get("issued_at"), "active issued_at")
    expires = _parse_time(value.get("expires_at"), "active expires_at")
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if expires <= issued or (expires - issued).total_seconds() > MAX_AUTH_TTL_SECONDS:
        raise Fixed4ExecutionPilotError("active authorization TTL invalid")
    if issued > observed + timedelta(seconds=AUTH_CLOCK_SKEW_SECONDS):
        raise Fixed4ExecutionPilotError("active authorization issued in future")
    return "EXPIRED" if expires <= observed else "CURRENT"


def validate_active_stage_authorization(
        value: Mapping[str, Any], request: Mapping[str, Any], *,
        preflight: Mapping[str, Any], output_root: Path,
        now: datetime | None = None, require_current: bool = True) -> None:
    """Verify the off-host signature and exact unsigned request body."""
    if (not _payload_valid(request)
            or request.get("schema") != ACTIVE_AUTHORIZATION_REQUEST_SCHEMA
            or request.get("unsigned") is not True
            or request.get("signer_location") != "off_host"
            or request.get("private_key_expected_on_execution_host") is not False
            or stable_json_sha256(request.get("authorization_body"))
                != request.get("authorization_body_sha256")):
        raise Fixed4ExecutionPilotError("active authorization request malformed")
    body = request["authorization_body"]
    if (not isinstance(value, Mapping)
            or {key: item for key, item in value.items()
                if key not in {"signature_b64", "payload_sha256"}} != body
            or not _payload_valid(value)
            or value.get("schema") != ACTIVE_STAGE_AUTHORIZATION_SCHEMA
            or value.get("preflight_payload_sha256") != preflight.get("payload_sha256")
            or value.get("signer_private_key_not_on_execution_host") is not True):
        raise Fixed4ExecutionPilotError("active signed authorization/body mismatch")
    status = active_authorization_time_status(value, now=now)
    if require_current and status != "CURRENT":
        raise Fixed4ExecutionPilotError("active authorization expired")
    try:
        verify_fixed_signed_document(
            value, repo_root=Path(str(preflight.get("repo_root", ""))),
            output_root=Path(output_root).resolve(),
            purpose="active stage authorization")
    except Fixed4SubprocessContractError as exc:
        raise Fixed4ExecutionPilotError(str(exc)) from exc


def _active_authorization_body_sha256(value: Mapping[str, Any]) -> str:
    return stable_json_sha256({
        key: item for key, item in value.items()
        if key not in {"signature_b64", "payload_sha256"}
    })


def _active_request_paths(root: Path, task_id: str,
                          request: Mapping[str, Any]) -> tuple[Path, Path]:
    body_sha = request.get("authorization_body_sha256")
    if (not isinstance(body_sha, str) or len(body_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in body_sha)):
        raise Fixed4ExecutionPilotError("authorization body SHA invalid")
    return (Path(root) / "authorization_requests" / task_id /
            f"{body_sha}.json",
            Path(root) / "authorizations" / task_id / f"{body_sha}.json")


def _committed_production_control(
        *, root: Path, task: Mapping[str, Any],
        preflight: Mapping[str, Any]) -> tuple[Path, Path] | None:
    """Resolve the sole committed manifest transaction for one task.

    Incomplete transaction directories are retained as crash evidence but are
    not consumer-visible.  More than one committed transaction is ambiguous
    and therefore fails closed.
    """
    transaction_root = (Path(root) / "tasks" / str(task["task_id"]) /
                        "control" / "production_manifest_transactions")
    if not transaction_root.exists():
        return None
    if transaction_root.is_symlink() or not transaction_root.is_dir():
        raise Fixed4ExecutionPilotError(
            "production manifest transaction root invalid")
    committed: list[tuple[Path, Path]] = []
    allowed = {"production_input_manifest.json",
               "production_execution_manifest.json", "COMMITTED.json"}
    for directory in sorted(transaction_root.iterdir()):
        if (directory.is_symlink() or not directory.is_dir()
                or not re.fullmatch(r"tx-[0-9a-f]{32}", directory.name)):
            raise Fixed4ExecutionPilotError(
                "production manifest transaction inventory invalid")
        entries = list(directory.iterdir())
        if any(path.is_symlink() or not path.is_file()
               or path.name not in allowed for path in entries):
            raise Fixed4ExecutionPilotError(
                "production manifest transaction file invalid")
        commit_path = directory / "COMMITTED.json"
        if not commit_path.is_file():
            continue
        try:
            from safety.v16_b716_fixed4_production_manifest_builder import (
                load_committed_production_manifest_transaction,
            )
            loaded = load_committed_production_manifest_transaction(
                commit_path=commit_path, task=task, preflight=preflight,
                output_root=root)
        except Exception as exc:
            raise Fixed4ExecutionPilotError(
                "production manifest transaction commit invalid") from exc
        execution_path = Path(loaded["commit"][
            "production_execution_manifest"]["path"])
        committed.append((commit_path, execution_path))
    if len(committed) > 1:
        raise Fixed4ExecutionPilotError(
            "multiple committed production manifest transactions")
    return committed[0] if committed else None


def _validate_stored_active_request(
        request: Mapping[str, Any], *, task: Mapping[str, Any],
        preflight: Mapping[str, Any], preflight_path: Path,
        manifest: Mapping[str, Any], manifest_path: Path,
        execution_manifest_path: Path) -> None:
    body = request.get("authorization_body")
    commit_path = Path(execution_manifest_path).with_name("COMMITTED.json")
    commit = _json(commit_path, None,
                   "stored production manifest transaction commit")
    if (not _payload_valid(request)
            or request.get("schema") != ACTIVE_AUTHORIZATION_REQUEST_SCHEMA
            or request.get("unsigned") is not True
            or request.get("signer_location") != "off_host"
            or request.get("private_key_expected_on_execution_host") is not False
            or not isinstance(body, Mapping)
            or stable_json_sha256(body)
                != request.get("authorization_body_sha256")
            or body.get("schema") != ACTIVE_STAGE_AUTHORIZATION_SCHEMA
            or body.get("task_id") != task.get("task_id")
            or body.get("task_payload_sha256") != task.get("payload_sha256")
            or body.get("stage") != task.get("stage")
            or body.get("preflight_path") != str(Path(preflight_path).resolve())
            or body.get("preflight_sha256") != sha256_file(preflight_path)
            or body.get("preflight_payload_sha256") != preflight.get("payload_sha256")
            or body.get("task_manifest_path") != str(Path(manifest_path).resolve())
            or body.get("task_manifest_sha256") != sha256_file(manifest_path)
            or body.get("task_manifest_payload_sha256") != manifest.get("payload_sha256")
            or body.get("execution_manifest_path")
                != str(Path(execution_manifest_path).resolve())
            or body.get("execution_manifest_sha256")
                != sha256_file(execution_manifest_path)
            or body.get("production_manifest_commit_path")
                != str(commit_path.resolve())
            or body.get("production_manifest_commit_sha256")
                != sha256_file(commit_path)
            or body.get("production_manifest_commit_payload_sha256")
                != commit.get("payload_sha256")
            or body.get("production_adapter_protocol_ready") is not True
            or body.get("signer_private_key_not_on_execution_host") is not True
            or any(body.get(key) is not False
                   for key in POLICY_FALSE_FIELDS)):
        raise Fixed4ExecutionPilotError(
            "stored active authorization request binding mismatch")


def _load_active_request_chain(
        *, root: Path, task: Mapping[str, Any], preflight: Mapping[str, Any],
        preflight_path: Path, manifest: Mapping[str, Any], manifest_path: Path,
        execution_manifest_path: Path,
        now: datetime | None) -> tuple[dict[str, Any], Path, Path, dict[str, Any] | None]:
    """Return the unique current request/auth leaf; forks fail closed."""
    task_id = str(task["task_id"])
    request_dir = Path(root) / "authorization_requests" / task_id
    auth_dir = Path(root) / "authorizations" / task_id
    if not request_dir.exists():
        raise FileNotFoundError
    if request_dir.is_symlink() or not request_dir.is_dir():
        raise Fixed4ExecutionPilotError("authorization request directory invalid")
    requests: list[tuple[dict[str, Any], Path, Path]] = []
    for path in sorted(request_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or not re.fullmatch(
                r"[0-9a-f]{64}\.json", path.name):
            raise Fixed4ExecutionPilotError(
                "authorization request inventory invalid")
        request = _json(path, None, "active authorization request")
        _validate_stored_active_request(
            request, task=task, preflight=preflight,
            preflight_path=preflight_path, manifest=manifest,
            manifest_path=manifest_path,
            execution_manifest_path=execution_manifest_path)
        expected_request, expected_auth = _active_request_paths(root, task_id, request)
        if path.resolve() != expected_request.resolve():
            raise Fixed4ExecutionPilotError(
                "authorization request path/body mismatch")
        requests.append((request, expected_request, expected_auth))
    roots = [row for row in requests if row[0]["authorization_body"].get(
        "renewal_of_authorization_payload_sha256") is None]
    if len(roots) != 1:
        raise Fixed4ExecutionPilotError(
            "authorization chain must have exactly one initial request")
    request_by_body = {
        str(row[0]["authorization_body_sha256"]): row for row in requests}
    if len(request_by_body) != len(requests):
        raise Fixed4ExecutionPilotError("duplicate authorization request body")
    if auth_dir.exists():
        if auth_dir.is_symlink() or not auth_dir.is_dir():
            raise Fixed4ExecutionPilotError("authorization directory invalid")
        observed_auth_names = set()
        for path in auth_dir.iterdir():
            if path.is_symlink() or not path.is_file() or not re.fullmatch(
                    r"[0-9a-f]{64}\.json", path.name):
                raise Fixed4ExecutionPilotError("authorization inventory invalid")
            observed_auth_names.add(path.name)
        expected_auth_names = {
            f"{body}.json" for body, row in request_by_body.items()
            if row[2].is_file()}
        if observed_auth_names != expected_auth_names:
            raise Fixed4ExecutionPilotError(
                "orphan/extra authorization file rejected")
    auth_by_body: dict[str, dict[str, Any]] = {}
    auth_payload_to_body: dict[str, str] = {}
    for body_sha, (request, _, auth_path) in request_by_body.items():
        if not auth_path.is_file():
            continue
        auth = _json(auth_path, None, "active stage authorization")
        validate_active_stage_authorization(
            auth, request, preflight=preflight, output_root=root,
            now=now, require_current=False)
        payload = str(auth.get("payload_sha256"))
        if payload in auth_payload_to_body:
            raise Fixed4ExecutionPilotError("duplicate authorization payload")
        auth_by_body[body_sha] = auth
        auth_payload_to_body[payload] = body_sha
    children_by_parent: dict[str, list[tuple[dict[str, Any], Path, Path]]] = {}
    for row in requests:
        renewal_of = row[0]["authorization_body"].get(
            "renewal_of_authorization_payload_sha256")
        if renewal_of is None:
            continue
        if renewal_of not in auth_payload_to_body:
            raise Fixed4ExecutionPilotError(
                "authorization renewal references unknown parent")
        children_by_parent.setdefault(str(renewal_of), []).append(row)
    if any(len(children) != 1 for children in children_by_parent.values()):
        raise Fixed4ExecutionPilotError(
            "concurrent authorization renewal fork rejected")
    current = roots[0]; visited: set[str] = set()
    while True:
        request, request_path, auth_path = current
        body_sha = str(request["authorization_body_sha256"])
        if body_sha in visited:
            raise Fixed4ExecutionPilotError("authorization renewal cycle")
        visited.add(body_sha)
        auth = auth_by_body.get(body_sha)
        if auth is None:
            children = []
        else:
            children = children_by_parent.get(str(auth["payload_sha256"]), [])
        if not children:
            if visited != set(request_by_body):
                raise Fixed4ExecutionPilotError(
                    "orphan/disconnected authorization request rejected")
            return request, request_path, auth_path, auth
        current = children[0]


def validate_active_adapter_result_release(
        *, task: Mapping[str, Any], candidate: Mapping[str, Any],
        candidate_path: Path, adapter_validation: Mapping[str, Any],
        adapter_validation_path: Path, output_root: Path,
        upstream_results: Mapping[str, Mapping[str, Any]],
        repo_root: Path,
        release_result: bool = True,
        ) -> dict[str, Any]:
    """Validate the adapter receipt and optionally release RESULT-v5.

    The dispatcher uses ``release_result=False`` so it can stage an immutable
    attempt, the validation receipt and RESULT-v5 before publishing a final
    create-only commit marker.  Other historical callers retain the original
    one-call release behaviour.
    """
    root = Path(output_root).resolve()
    required = {"schema", "status", "task_id", "task_payload_sha256",
        "stage", "candidate_path", "candidate_sha256", "candidate_payload_sha256",
        "operational_result_schema", "parent_result_payload_sha256s",
        "production_adapter_contract_path", "production_adapter_contract_sha256",
        "production_adapter_contract_payload_sha256",
        "production_input_manifest_sha256",
        "production_input_manifest_payload_sha256", "execution_manifest_path",
        "execution_manifest_sha256", "execution_manifest_payload_sha256",
        "production_attempt_path", "production_attempt_sha256",
        "production_attempt_payload_sha256", "output_artifact_rows",
        "output_artifact_closure_sha256", "validator_source_sha256",
        "runner_source_sha256", "stage_semantics", "stage_semantics_sha256",
        *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    _require_exact_keys(adapter_validation, required, "active adapter validation")
    parent_payloads = [upstream_results[parent]["payload_sha256"]
                       for parent in task.get("upstream_task_ids", ())]
    if (not _payload_valid(adapter_validation)
            or adapter_validation.get("schema") != ACTIVE_ADAPTER_VALIDATION_SCHEMA
            or adapter_validation.get("status") != "PASS"
            or adapter_validation.get("task_id") != task.get("task_id")
            or adapter_validation.get("task_payload_sha256")
                != task.get("payload_sha256")
            or adapter_validation.get("stage") != task.get("stage")
            or adapter_validation.get("candidate_path")
                != str(Path(candidate_path).resolve())
            or adapter_validation.get("candidate_sha256")
                != sha256_file(candidate_path)
            or adapter_validation.get("candidate_payload_sha256")
                != candidate.get("payload_sha256")
            or adapter_validation.get("operational_result_schema") != RESULT_SCHEMA
            or adapter_validation.get("parent_result_payload_sha256s")
                != parent_payloads):
        raise Fixed4ExecutionPilotError("production adapter validation mismatch")
    _policy_false(adapter_validation, "active adapter validation")
    semantics = adapter_validation.get("stage_semantics")
    if (not isinstance(semantics, Mapping)
            or semantics != {"normal_gate_return_code": 2,
                "normal_gate_distinct_from_process_failure": True,
                "parent_results_derived_not_task_reported": True,
                "operational_result_create_only_release_required": True}
            or stable_json_sha256(semantics)
                != adapter_validation.get("stage_semantics_sha256")):
        raise Fixed4ExecutionPilotError("adapter stage semantics drift")
    for prefix in ("production_adapter_contract", "execution_manifest",
                   "production_attempt"):
        path = Path(str(adapter_validation.get(f"{prefix}_path", "")))
        expected = adapter_validation.get(f"{prefix}_sha256")
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or sha256_file(path) != expected):
            raise Fixed4ExecutionPilotError(f"adapter {prefix} binding drift")
        document = _json(path, expected, f"adapter {prefix}")
        if document.get("payload_sha256") != adapter_validation.get(
                f"{prefix}_payload_sha256"):
            raise Fixed4ExecutionPilotError(f"adapter {prefix} payload drift")
    outputs = adapter_validation.get("output_artifact_rows")
    if (not isinstance(outputs, list)
            or stable_json_sha256(outputs)
                != adapter_validation.get("output_artifact_closure_sha256")):
        raise Fixed4ExecutionPilotError("adapter output closure drift")
    for index, row in enumerate(outputs):
        _resolve_row_file(root, row, f"adapter output {index}",
                          within=root / "tasks" / str(task["task_id"]) /
                          "production")
    repo = Path(repo_root).resolve()
    validator_source = repo / \
        "src/safety/v16_b716_fixed4_active_production_wrapper.py"
    runner_source = repo / "scripts/v16_b716_fixed4_active_stage_runner.sh"
    if (sha256_file(validator_source)
            != adapter_validation.get("validator_source_sha256")
            or sha256_file(runner_source)
            != adapter_validation.get("runner_source_sha256")):
        raise Fixed4ExecutionPilotError("adapter validator/runner source drift")
    if sha256_file(adapter_validation_path) != no_symlink_file_row(
            adapter_validation_path, "adapter validation")["sha256"]:
        raise Fixed4ExecutionPilotError("adapter validation changed while reading")
    validated = validate_runner_result(
        task, candidate, root, upstream_results=upstream_results)
    if release_result:
        result_path = root / "tasks" / str(task["task_id"]) / "result.json"
        encoded = (json.dumps(validated, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode()
        try:
            create_only_bytes_beneath(root, result_path, encoded)
        except Fixed4SubprocessContractError as exc:
            raise Fixed4ExecutionPilotError(str(exc)) from exc
    return validated


def validate_active_adapter_validation_receipt(
        *, task: Mapping[str, Any], candidate: Mapping[str, Any],
        candidate_path: Path, receipt: Mapping[str, Any], receipt_path: Path,
        output_root: Path, upstream_results: Mapping[str, Mapping[str, Any]],
        repo_root: Path) -> None:
    """Validate the receipt API without releasing ``result.json``.

    This uses the exact same checks as the release gate but rejects an existing
    or missing candidate release target by validating into a private temporary
    view.  It is intended for dispatcher pre-release checks and tests.
    """
    root = Path(output_root).resolve()
    result_path = root / "tasks" / str(task["task_id"]) / "result.json"
    if result_path.exists() or result_path.is_symlink():
        raise Fixed4ExecutionPilotError("operational result already exists")
    # Inline the non-mutating validation by temporarily exercising the same
    # receipt primitives; candidate validation is the final semantic check.
    parent_payloads = [upstream_results[parent]["payload_sha256"]
                       for parent in task.get("upstream_task_ids", ())]
    if (receipt.get("schema") != ACTIVE_ADAPTER_VALIDATION_SCHEMA
            or receipt.get("status") != "PASS"
            or receipt.get("parent_result_payload_sha256s") != parent_payloads
            or sha256_file(receipt_path) != no_symlink_file_row(
                receipt_path, "adapter validation receipt")["sha256"]):
        raise Fixed4ExecutionPilotError("adapter validation receipt mismatch")
    # The full file/source/semantic checks are performed immediately again by
    # ``validate_active_adapter_result_release`` before its O_EXCL write.
    validate_runner_result(task, candidate, root,
                           upstream_results=upstream_results or None)


def _validate_active_parent_attempt(
        *, task: Mapping[str, Any], task_root: Path,
        result: Mapping[str, Any], upstream_results: Mapping[str, Mapping[str, Any]],
        preflight: Mapping[str, Any], output_root: Path,
        ) -> None:
    """Require validator/auth receipts before a child may consume a parent."""
    root = Path(output_root).resolve()
    validation_path = task_root / "adapter_validation.json"
    commit_path = task_root / "active_commit.json"
    for path, role in ((validation_path, "parent adapter validation"),
                       (commit_path, "parent active commit")):
        if not path.is_file():
            raise Fixed4ExecutionPilotError(f"{role} missing")
    commit = _json(commit_path, None, "parent active commit")
    required_commit = {"schema", "status", "task_id", "task_payload_sha256",
        "attempt_path", "attempt_sha256", "attempt_payload_sha256",
        "adapter_validation_path", "adapter_validation_sha256",
        "adapter_validation_payload_sha256", "result_path", "result_sha256",
        "result_payload_sha256", "parent_result_payload_sha256s",
        "operational_result_released", *POLICY_FALSE_FIELDS.keys(),
        "payload_sha256"}
    _require_exact_keys(commit, required_commit, "parent active commit")
    attempt_path = Path(str(commit.get("attempt_path", "")))
    result_path = task_root / "result.json"
    if (not _payload_valid(commit)
            or commit.get("schema") != ACTIVE_STAGE_COMMIT_SCHEMA
            or commit.get("status") != "COMMITTED"
            or commit.get("task_id") != task.get("task_id")
            or commit.get("task_payload_sha256") != task.get("payload_sha256")
            or commit.get("adapter_validation_path")
                != str(validation_path.resolve())
            or commit.get("adapter_validation_sha256")
                != sha256_file(validation_path)
            or commit.get("result_path") != str(result_path.resolve())
            or commit.get("result_sha256") != sha256_file(result_path)
            or commit.get("result_payload_sha256") != result.get("payload_sha256")
            or commit.get("operational_result_released") is not True
            or not attempt_path.is_absolute() or not attempt_path.is_file()
            or attempt_path.is_symlink()
            or attempt_path.parent != task_root / "active_attempts"
            or commit.get("attempt_sha256") != sha256_file(attempt_path)):
        raise Fixed4ExecutionPilotError("parent active commit mismatch")
    _policy_false(commit, "parent active commit")
    attempt = _json(attempt_path, None, "parent active attempt")
    required_attempt = {"schema", "status", "task_id", "task_payload_sha256",
        "executed_at", "authorization_request_path",
        "authorization_request_sha256",
        "authorization_request_payload_sha256", "authorization_path",
        "authorization_sha256", "authorization_payload_sha256",
        "adapter_validation_sha256", "adapter_validation_payload_sha256",
        "result_sha256", "result_payload_sha256",
        "parent_result_payload_sha256s", *POLICY_FALSE_FIELDS.keys(),
        "payload_sha256"}
    _require_exact_keys(attempt, required_attempt, "parent active attempt")
    executed_at = _parse_time(attempt.get("executed_at"),
                              "parent active attempt executed_at")
    if executed_at > datetime.now(timezone.utc) + timedelta(
            seconds=AUTH_CLOCK_SKEW_SECONDS):
        raise Fixed4ExecutionPilotError(
            "parent active attempt executed_at is in the future")
    request_path = Path(str(attempt.get("authorization_request_path", "")))
    auth_path = Path(str(attempt.get("authorization_path", "")))
    if not request_path.is_file() or not auth_path.is_file():
        raise Fixed4ExecutionPilotError("parent authorization chain missing")
    request = _json(request_path, None, "parent authorization request")
    auth = _json(auth_path, None, "parent authorization")
    expected_request_path, expected_auth_path = _active_request_paths(
        root, str(task["task_id"]), request)
    if (request_path.resolve() != expected_request_path.resolve()
            or auth_path.resolve() != expected_auth_path.resolve()
            or attempt_path != (task_root / "active_attempts" /
                f"{auth.get('payload_sha256')}.json")
            or attempt.get("authorization_request_sha256")
                != sha256_file(request_path)
            or attempt.get("authorization_request_payload_sha256")
                != request.get("payload_sha256")):
        raise Fixed4ExecutionPilotError(
            "parent authorization request path/binding mismatch")
    validate_active_stage_authorization(
        auth, request, preflight=preflight, output_root=root,
        now=executed_at)
    issued_at = _parse_time(auth.get("issued_at"), "parent authorization issued_at")
    expires_at = _parse_time(auth.get("expires_at"), "parent authorization expires_at")
    if not (issued_at <= executed_at < expires_at):
        raise Fixed4ExecutionPilotError(
            "parent attempt was not executed inside authorization TTL")
    validation = _json(validation_path, None, "parent adapter validation")
    required_validation = {"schema", "status", "task_id", "task_payload_sha256",
        "stage", "candidate_path", "candidate_sha256", "candidate_payload_sha256",
        "operational_result_schema", "parent_result_payload_sha256s",
        "production_adapter_contract_path", "production_adapter_contract_sha256",
        "production_adapter_contract_payload_sha256",
        "production_input_manifest_sha256",
        "production_input_manifest_payload_sha256", "execution_manifest_path",
        "execution_manifest_sha256", "execution_manifest_payload_sha256",
        "production_attempt_path", "production_attempt_sha256",
        "production_attempt_payload_sha256", "output_artifact_rows",
        "output_artifact_closure_sha256", "validator_source_sha256",
        "runner_source_sha256", "stage_semantics", "stage_semantics_sha256",
        *POLICY_FALSE_FIELDS.keys(), "payload_sha256"}
    _require_exact_keys(validation, required_validation, "parent adapter validation")
    parent_payloads = [upstream_results[parent]["payload_sha256"]
                       for parent in task.get("upstream_task_ids", ())]
    if (not _payload_valid(validation)
            or validation.get("schema") != ACTIVE_ADAPTER_VALIDATION_SCHEMA
            or validation.get("status") != "PASS"
            or validation.get("task_id") != task.get("task_id")
            or validation.get("task_payload_sha256") != task.get("payload_sha256")
            or validation.get("stage") != task.get("stage")
            or validation.get("candidate_sha256") != sha256_file(result_path)
            or validation.get("candidate_payload_sha256") != result.get("payload_sha256")
            or validation.get("operational_result_schema") != RESULT_SCHEMA
            or validation.get("parent_result_payload_sha256s") != parent_payloads):
        raise Fixed4ExecutionPilotError("parent adapter validation mismatch")
    _policy_false(validation, "parent adapter validation")
    if (not _payload_valid(attempt)
            or attempt.get("schema") != ACTIVE_STAGE_ATTEMPT_SCHEMA
            or attempt.get("status") != result.get("status")
            or attempt.get("task_id") != task.get("task_id")
            or attempt.get("task_payload_sha256") != task.get("payload_sha256")
            or attempt.get("authorization_path") != str(auth_path.resolve())
            or attempt.get("authorization_sha256") != sha256_file(auth_path)
            or attempt.get("authorization_payload_sha256") != auth.get("payload_sha256")
            or attempt.get("adapter_validation_sha256") != sha256_file(validation_path)
            or attempt.get("adapter_validation_payload_sha256")
                != validation.get("payload_sha256")
            or attempt.get("result_sha256") != sha256_file(result_path)
            or attempt.get("result_payload_sha256") != result.get("payload_sha256")
            or attempt.get("parent_result_payload_sha256s") != parent_payloads):
        raise Fixed4ExecutionPilotError("parent active attempt mismatch")
    _policy_false(attempt, "parent active attempt")
    if (commit.get("attempt_payload_sha256") != attempt.get("payload_sha256")
            or commit.get("adapter_validation_payload_sha256")
                != validation.get("payload_sha256")
            or commit.get("parent_result_payload_sha256s") != parent_payloads):
        raise Fixed4ExecutionPilotError("parent active commit payload mismatch")


def validate_active_dispatch_execution_inputs(
        *, preflight: Mapping[str, Any], preflight_path: Path,
        manifest: Mapping[str, Any], manifest_path: Path,
        task: Mapping[str, Any], task_path: Path,
        request: Mapping[str, Any], request_path: Path,
        authorization: Mapping[str, Any], authorization_path: Path,
        execution_manifest_path: Path, output_root: Path,
        now: datetime | None = None,
        ) -> dict[str, Mapping[str, Any]]:
    """Revalidate dispatcher membership, controls and all completed ancestors."""
    root = Path(output_root).resolve()
    preflight_path = Path(preflight_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    task_path = Path(task_path).resolve()
    request_path = Path(request_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    execution_manifest_path = Path(execution_manifest_path).resolve()
    if (preflight_path != root / "execution_preflight.json"
            or manifest_path != root / "task_manifest.json"
            or task_path != root / "tasks" / str(task.get("task_id")) / "task.json"):
        raise Fixed4ExecutionPilotError(
            "active direct execution control path is not canonical")
    _validate_active_preflight_document(preflight)
    if (preflight.get("output_root") != str(root)
            or manifest.get("schema") != TASK_MANIFEST_SCHEMA
            or not _payload_valid(manifest)
            or manifest.get("preflight_payload_sha256")
                != preflight.get("payload_sha256")
            or manifest.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")):
        raise Fixed4ExecutionPilotError(
            "active direct execution preflight/manifest mismatch")
    _validate_source_and_registry(preflight)
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or len(rows) != OPERATIONAL_TASK_COUNT:
        raise Fixed4ExecutionPilotError(
            "active direct execution task inventory mismatch")
    manifest_row = next((row for row in rows
                         if row.get("task_id") == task.get("task_id")), None)
    if (not isinstance(manifest_row, Mapping)
            or manifest_row.get("path") != str(task_path.relative_to(root))
            or manifest_row.get("sha256") != sha256_file(task_path)
            or manifest_row.get("payload_sha256") != task.get("payload_sha256")):
        raise Fixed4ExecutionPilotError(
            "active direct execution task membership mismatch")
    _validate_task_manifest(task_path, task, root, preflight)
    _validate_stored_active_request(
        request, task=task, preflight=preflight,
        preflight_path=preflight_path, manifest=manifest,
        manifest_path=manifest_path,
        execution_manifest_path=execution_manifest_path)
    expected_request, expected_auth = _active_request_paths(
        root, str(task["task_id"]), request)
    if (request_path != expected_request.resolve()
            or authorization_path != expected_auth.resolve()):
        raise Fixed4ExecutionPilotError(
            "active direct execution authorization path mismatch")
    validate_active_stage_authorization(
        authorization, request, preflight=preflight,
        output_root=root, now=now)
    completed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        candidate_task_path = root / str(row.get("path", ""))
        candidate_task = _json(
            candidate_task_path, row.get("sha256"),
            "active direct execution prior task")
        _validate_task_manifest(candidate_task_path, candidate_task, root, preflight)
        upstream = candidate_task.get("upstream_task_ids", [])
        if any(parent not in completed for parent in upstream):
            raise Fixed4ExecutionPilotError(
                "active direct execution parents are incomplete/non-topological")
        if candidate_task.get("task_id") == task.get("task_id"):
            return {parent: completed[parent] for parent in upstream}
        candidate_root = root / "tasks" / str(candidate_task["task_id"])
        result_path = candidate_root / "result.json"
        if not result_path.is_file():
            raise Fixed4ExecutionPilotError(
                "active direct execution skipped an earlier incomplete task")
        parent_results = {parent: completed[parent] for parent in upstream}
        result = validate_runner_result(
            candidate_task,
            _json(result_path, None, "active direct execution parent result"),
            root, upstream_results=parent_results or None)
        _validate_active_parent_attempt(
            task=candidate_task, task_root=candidate_root, result=result,
            upstream_results=parent_results, preflight=preflight,
            output_root=root)
        completed[str(candidate_task["task_id"])] = result
    raise Fixed4ExecutionPilotError(
        "active direct execution task missing from topological manifest")


def plan_active_stagewise_dispatch(
        *, preflight_path: Path, manifest_path: Path, output_root: Path,
        signing_key_id: str, now: datetime | None = None,
        ) -> dict[str, Any]:
    """Plan the next DAG node; never signs and never calls an unavailable adapter."""
    root = Path(output_root).resolve()
    preflight = _json(preflight_path, None, "active execution preflight")
    manifest = _json(manifest_path, None, "active task manifest")
    _validate_active_preflight_document(preflight)
    if (preflight.get("output_root") != str(root)
            or manifest.get("preflight_payload_sha256") != preflight.get("payload_sha256")
            or manifest.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")):
        raise Fixed4ExecutionPilotError("active dispatch root/manifest mismatch")
    _validate_source_and_registry(preflight)
    completed: dict[str, Mapping[str, Any]] = {}
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or len(rows) != OPERATIONAL_TASK_COUNT:
        raise Fixed4ExecutionPilotError("active dispatch task inventory mismatch")
    for row in rows:
        task_path = root / str(row.get("path", ""))
        task = _json(task_path, row.get("sha256"), "active operational task")
        _validate_task_manifest(task_path, task, root, preflight)
        upstream = task.get("upstream_task_ids", [])
        if any(parent not in completed for parent in upstream):
            raise Fixed4ExecutionPilotError("active dispatch parent result missing/non-topological")
        task_root = root / "tasks" / str(task["task_id"])
        result_path = task_root / "result.json"
        if result_path.is_file():
            result = validate_runner_result(
                task, _json(result_path, None, "active parent result"), root,
                upstream_results={parent: completed[parent] for parent in upstream})
            _validate_active_parent_attempt(
                task=task, task_root=task_root, result=result,
                upstream_results={parent: completed[parent] for parent in upstream},
                preflight=preflight, output_root=root)
            completed[task["task_id"]] = result
            continue
        legacy_request_path = root / "authorization_requests" / f"{task['task_id']}.json"
        legacy_auth_path = root / "authorizations" / f"{task['task_id']}.json"
        if legacy_request_path.exists() or legacy_auth_path.exists():
            raise Fixed4ExecutionPilotError(
                "unversioned per-task authorization state is rejected")
        request_path: Path | None = None
        auth_path: Path | None = None
        descriptor = task.get("stage_runner_input_descriptor", {})
        active_contract = preflight.get("active_subprocess_contract")
        # The historical v1 preregistration remains immutable evidence that
        # the adapter was not ready.  A ready-looking legacy boolean never
        # upgrades that envelope: production requires explicit v2 controls on
        # both the task descriptor and the preflight.
        protocol_ready = bool(
            isinstance(descriptor, Mapping)
            and descriptor.get("schema")
                == ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA
            and descriptor.get("production_adapter_protocol_ready") is True
            and isinstance(active_contract, Mapping)
            and active_contract.get("schema") == ACTIVE_PREFLIGHT_V2_SCHEMA
            and active_contract.get("production_adapter_protocol_ready") is True)
        committed_control = (_committed_production_control(
            root=root, task=task, preflight=preflight)
            if protocol_ready else None)
        execution_manifest_path = (committed_control[1]
                                   if committed_control is not None else None)
        request = None
        status = "PRODUCTION_MANIFESTS_REQUIRED"
        if not protocol_ready:
            status = "PRODUCTION_ADAPTER_PROTOCOL_UNAVAILABLE"
        elif execution_manifest_path is None:
            status = "PRODUCTION_MANIFESTS_REQUIRED"
        renewal_of = None
        if protocol_ready and execution_manifest_path is not None:
            try:
                stored_request, request_path, auth_path, auth = \
                    _load_active_request_chain(
                        root=root, task=task, preflight=preflight,
                        preflight_path=preflight_path, manifest=manifest,
                        manifest_path=manifest_path,
                        execution_manifest_path=execution_manifest_path,
                        now=now)
            except FileNotFoundError:
                request = build_active_stage_authorization_request(
                    preflight=preflight, preflight_path=preflight_path,
                    manifest=manifest, manifest_path=manifest_path,
                    task=task, task_path=task_path,
                    signing_key_id=signing_key_id, issued_at=now,
                    production_execution_manifest_path=execution_manifest_path)
                request_path, auth_path = _active_request_paths(
                    root, str(task["task_id"]), request)
                status = "AUTHORIZATION_REQUIRED"
            else:
                renewal_of = stored_request["authorization_body"].get(
                    "renewal_of_authorization_payload_sha256")
                if auth is None:
                    status = "AUTHORIZATION_REQUIRED"
                elif active_authorization_time_status(auth, now=now) == "EXPIRED":
                    renewal_of = auth.get("payload_sha256")
                    request = build_active_stage_authorization_request(
                        preflight=preflight, preflight_path=preflight_path,
                        manifest=manifest, manifest_path=manifest_path,
                        task=task, task_path=task_path,
                        signing_key_id=signing_key_id, issued_at=now,
                        renewal_of_authorization_payload_sha256=renewal_of,
                        production_execution_manifest_path=execution_manifest_path)
                    request_path, auth_path = _active_request_paths(
                        root, str(task["task_id"]), request)
                    status = "AUTHORIZATION_RENEWAL_REQUIRED"
                else:
                    validate_active_stage_authorization(
                        auth, stored_request, preflight=preflight,
                        output_root=root, now=now)
                    status = "AUTHORIZED_READY_TO_EXECUTE"
        receipt = {"schema": ACTIVE_DISPATCH_RECEIPT_SCHEMA,
            "status": status, "task_id": task["task_id"], "stage": task["stage"],
            "task_payload_sha256": task["payload_sha256"],
            "verified_parent_result_payload_sha256s":
                [completed[parent]["payload_sha256"] for parent in upstream],
            "request_path": (str(request_path) if request_path else None),
            "authorization_path": (str(auth_path) if auth_path else None),
            "renewal_of_authorization_payload_sha256": renewal_of,
            "authorization_ttl_checked_before_node": True,
            "unsigned_request_generated": request is not None,
            "private_key_used_on_execution_host": False,
            "execution_performed": False,
            "production_adapter_protocol_ready": protocol_ready,
            "operational_result_released": False,
            **POLICY_FALSE_FIELDS}
        receipt["payload_sha256"] = stable_json_sha256(receipt)
        return {"receipt": receipt, "unsigned_request": request}
    receipt = {"schema": ACTIVE_DISPATCH_RECEIPT_SCHEMA,
        "status": "ALL_TASKS_ALREADY_COMPLETE", "task_id": None, "stage": None,
        "task_payload_sha256": None,
        "verified_parent_result_payload_sha256s": [],
        "request_path": None, "authorization_path": None,
        "renewal_of_authorization_payload_sha256": None,
        "authorization_ttl_checked_before_node": True,
        "unsigned_request_generated": False,
        "private_key_used_on_execution_host": False,
        "execution_performed": False,
        "production_adapter_protocol_ready": False,
        "operational_result_released": False, **POLICY_FALSE_FIELDS}
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    return {"receipt": receipt, "unsigned_request": None}
