"""Hash-bound fixed4 stage runner primitives.

This module defines the missing stage-result semantics between the existing
V13--V16 algorithms. Process execution and scientific gate status are
separate; finite incompatibility is auditable and can never authorize a
downstream stage. No GT, threshold override, ranking, identity fallback,
reconstruction, or refusion path exists here.

Current operational task manifests do not carry ``stage_runner_input``. The
registered wrappers therefore remain fail closed until a future independently
reviewed preflight hash-binds those inputs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from safety.v13_dual_solver_runtime import stable_json_sha256, validate_se3
from safety.v15_safe_pose_cluster import ROTATION_MAX_DEG, TRANSLATION_MAX_M
from safety.v16_safe_hypothesis_cluster import (
    select_unique_safe_hypothesis_pose_cluster,
)


HYPOTHESIS_GATE_SCHEMA = "v16-b716-fixed4-hypothesis-gate-result-v1"
PAIR_GATE_SCHEMA = "v16-b716-fixed4-pair-gate-result-v1"
AGGREGATE_GATE_SCHEMA = "v16-b716-fixed4-aggregate-gate-result-v1"
EXECUTION_SUCCEEDED = "succeeded"
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
FINITE_CONSENSUS_INCOMPATIBILITY = "FINITE_CONSENSUS_INCOMPATIBILITY"
NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER = (
    "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER"
)
KNOWN_BAD_PERMANENT_VETO = "KNOWN_BAD_PERMANENT_VETO"
FIXED4_NORMAL_PAIR_CONSENSUS_FAILED = (
    "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"
)
FIXED4_KNOWN_BAD_VETO_FAILED = "FIXED4_KNOWN_BAD_VETO_FAILED"

POLICY_FALSE = {
    "gt_consumed": False,
    "thresholds_changed": False,
    "identity_fallback_used": False,
    "result_selection_used": False,
    "reconstruction_authorized": False,
    "refusion_run": False,
}


class RegisteredStageRunnerUnavailable(RuntimeError):
    """The reviewed registry has no executable implementation for this stage."""


class StageGateResultError(RuntimeError):
    """A stage result is malformed, unbound, or attempts to fail open."""


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def _payload_valid(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    return value.get("payload_sha256") == stable_json_sha256(unsigned)


def _exact_ids(values: Sequence[Any], name: str) -> list[str]:
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))):
        raise StageGateResultError(f"{name} must contain unique non-empty ids")
    return list(values)


def _validated_transform(value: Any, name: str) -> list[list[float]]:
    if value is None:
        raise StageGateResultError(f"{name} is required")
    try:
        return validate_se3(value).tolist()
    except Exception as exc:
        raise StageGateResultError(f"{name} is not finite SE3") from exc


def _finite_nonnegative(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0):
        raise StageGateResultError(f"{name} must be finite and non-negative")
    return float(value)


def _compatible(rotation_deg: float, translation_m: float) -> bool:
    rotation_ok = (
        rotation_deg <= ROTATION_MAX_DEG
        or math.isclose(rotation_deg, ROTATION_MAX_DEG,
                        rel_tol=0.0, abs_tol=1e-9)
    )
    translation_ok = (
        translation_m <= TRANSLATION_MAX_M
        or math.isclose(translation_m, TRANSLATION_MAX_M,
                        rel_tol=0.0, abs_tol=1e-12)
    )
    return rotation_ok and translation_ok


def classify_finite_consensus(
    *, hypothesis_task_id: str, rotation_deg: float,
    translation_m: float, transform: Any,
) -> dict[str, Any]:
    """Classify one completed finite cross-solver compatibility check."""
    if not isinstance(hypothesis_task_id, str) or not hypothesis_task_id:
        raise StageGateResultError("hypothesis task id is invalid")
    rotation = _finite_nonnegative(rotation_deg, "rotation")
    translation = _finite_nonnegative(translation_m, "translation")
    compatible = _compatible(rotation, translation)
    accepted_transform = (
        _validated_transform(transform, "compatible transform")
        if compatible else None
    )
    value = {
        "schema": HYPOTHESIS_GATE_SCHEMA,
        "hypothesis_task_id": hypothesis_task_id,
        "execution_status": EXECUTION_SUCCEEDED,
        "gate_status": GATE_PASS if compatible else GATE_FAIL,
        "failure_class": (
            None if compatible else FINITE_CONSENSUS_INCOMPATIBILITY
        ),
        "decision": (
            "FINITE_CONSENSUS_COMPATIBLE" if compatible
            else "CROSS_SOLVER_INCOMPATIBLE"
        ),
        "transform": accepted_transform,
        "downstream_authorized": compatible,
        "measured_rotation_deg": rotation,
        "measured_translation_m": translation,
        "threshold_rotation_deg": float(ROTATION_MAX_DEG),
        "threshold_translation_m": float(TRANSLATION_MAX_M),
        **POLICY_FALSE,
    }
    return _sealed(value)


def validate_hypothesis_gate_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or not _payload_valid(value)
            or value.get("schema") != HYPOTHESIS_GATE_SCHEMA
            or value.get("execution_status") != EXECUTION_SUCCEEDED
            or any(value.get(key) is not False for key in POLICY_FALSE)):
        raise StageGateResultError("hypothesis gate result binding invalid")
    rotation = _finite_nonnegative(
        value.get("measured_rotation_deg"), "rotation")
    translation = _finite_nonnegative(
        value.get("measured_translation_m"), "translation")
    if (value.get("threshold_rotation_deg") != float(ROTATION_MAX_DEG)
            or value.get("threshold_translation_m")
            != float(TRANSLATION_MAX_M)):
        raise StageGateResultError("frozen compatibility thresholds changed")
    compatible = _compatible(rotation, translation)
    if compatible:
        if (value.get("gate_status") != GATE_PASS
                or value.get("failure_class") is not None
                or value.get("decision") != "FINITE_CONSENSUS_COMPATIBLE"
                or value.get("downstream_authorized") is not True):
            raise StageGateResultError("compatible hypothesis verdict invalid")
        _validated_transform(value.get("transform"), "compatible transform")
    elif (value.get("gate_status") != GATE_FAIL
          or value.get("failure_class") != FINITE_CONSENSUS_INCOMPATIBILITY
          or value.get("decision") != "CROSS_SOLVER_INCOMPATIBLE"
          or value.get("transform") is not None
          or value.get("downstream_authorized") is not False):
        raise StageGateResultError("incompatible hypothesis did not fail closed")
    return dict(value)


def evaluate_v16_pair_cluster(
    evidence: Sequence[Mapping[str, Any]], *,
    expected_hypothesis_count: int, known_bad: bool,
) -> dict[str, Any]:
    """Use the existing V16 complete-linkage authority unchanged."""
    return select_unique_safe_hypothesis_pose_cluster(
        evidence,
        expected_hypothesis_count=expected_hypothesis_count,
        known_bad=known_bad,
    )


def build_pair_gate_result(
    *, task_id: str, pair_id: str,
    replayed_hypothesis_task_ids: Sequence[str],
    eligible_hypothesis_task_ids: Sequence[str],
    typed_abstention_hypothesis_task_ids: Sequence[str],
    hypothesis_gate_results: Sequence[Mapping[str, Any]],
    cluster_decision: Mapping[str, Any], known_bad: bool,
) -> dict[str, Any]:
    """Build one pair receipt with exact vote/fail/abstain closure."""
    if not isinstance(task_id, str) or not task_id:
        raise StageGateResultError("pair task id is invalid")
    if not isinstance(pair_id, str) or not pair_id:
        raise StageGateResultError("pair id is invalid")
    replayed = _exact_ids(replayed_hypothesis_task_ids, "replayed hypotheses")
    eligible = _exact_ids(eligible_hypothesis_task_ids, "eligible hypotheses")
    abstained = _exact_ids(
        typed_abstention_hypothesis_task_ids, "typed abstentions")
    if (set(eligible).intersection(abstained)
            or sorted(eligible + abstained) != sorted(replayed)):
        raise StageGateResultError("eligible/typed-abstention closure mismatch")
    rows = [validate_hypothesis_gate_result(row)
            for row in hypothesis_gate_results]
    by_id = {row.get("hypothesis_task_id"): row for row in rows}
    if (len(by_id) != len(rows) or set(by_id) != set(eligible)):
        raise StageGateResultError("eligible gate-result closure mismatch")
    safe_vote = [task for task in eligible
                 if by_id[task]["gate_status"] == GATE_PASS]
    gate_failed = [task for task in eligible
                   if by_id[task]["gate_status"] == GATE_FAIL]
    if sorted(safe_vote + gate_failed) != sorted(eligible):
        raise StageGateResultError("safe-vote/gate-failed closure mismatch")

    if known_bad:
        gate_status, failure_class = GATE_FAIL, KNOWN_BAD_PERMANENT_VETO
        decision, transform, downstream = (
            "PERMANENT_KNOWN_BAD_VETO", None, False)
    else:
        accepted = cluster_decision.get("accepted")
        reason = cluster_decision.get("reason")
        if not isinstance(accepted, bool):
            raise StageGateResultError("V16 cluster decision missing verdict")
        if accepted:
            if reason != "unique_safe_hypothesis_pose_cluster":
                raise StageGateResultError("accepted V16 cluster reason invalid")
            transform = _validated_transform(
                cluster_decision.get("selected_transform"),
                "selected safe-cluster transform")
            safe_transforms = [by_id[task].get("transform") for task in safe_vote]
            if (not safe_transforms
                    or all(stable_json_sha256(candidate) != stable_json_sha256(transform)
                           for candidate in safe_transforms)):
                raise StageGateResultError(
                    "selected safe-cluster transform is not parent-evidence-backed")
            gate_status, failure_class = GATE_PASS, None
            decision = "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"
            downstream = True
        else:
            transform = None
            gate_status = GATE_FAIL
            failure_class = NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER
            decision = "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER"
            downstream = False
    return _sealed({
        "schema": PAIR_GATE_SCHEMA,
        "stage": "v16_pair_hypothesis_cluster",
        "task_id": task_id,
        "pair_id": pair_id,
        "execution_status": EXECUTION_SUCCEEDED,
        "gate_status": gate_status,
        "failure_class": failure_class,
        "decision": decision,
        "transform": transform,
        "downstream_authorized": downstream,
        "known_bad": bool(known_bad),
        "replayed_hypothesis_task_ids": replayed,
        "eligible_hypothesis_task_ids": eligible,
        "safe_vote_hypothesis_task_ids": safe_vote,
        "gate_failed_hypothesis_task_ids": gate_failed,
        "typed_abstention_hypothesis_task_ids": abstained,
        "cluster_reason": cluster_decision.get("reason"),
        **POLICY_FALSE,
    })


def validate_pair_gate_result(value: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or not _payload_valid(value)
            or value.get("schema") != PAIR_GATE_SCHEMA
            or value.get("execution_status") != EXECUTION_SUCCEEDED
            or value.get("stage") != "v16_pair_hypothesis_cluster"):
        raise StageGateResultError("pair gate result binding invalid")
    replayed = _exact_ids(
        value.get("replayed_hypothesis_task_ids"), "replayed hypotheses")
    eligible = _exact_ids(
        value.get("eligible_hypothesis_task_ids"), "eligible hypotheses")
    safe_vote = _exact_ids(
        value.get("safe_vote_hypothesis_task_ids"), "safe votes")
    gate_failed = _exact_ids(
        value.get("gate_failed_hypothesis_task_ids"), "gate failures")
    abstained = _exact_ids(
        value.get("typed_abstention_hypothesis_task_ids"), "typed abstentions")
    if (set(eligible).intersection(abstained)
            or sorted(eligible + abstained) != sorted(replayed)
            or set(safe_vote).intersection(gate_failed)
            or sorted(safe_vote + gate_failed) != sorted(eligible)):
        raise StageGateResultError("pair hypothesis partitions are not exact")
    if value.get("known_bad"):
        valid = (
            value.get("gate_status") == GATE_FAIL
            and value.get("failure_class") == KNOWN_BAD_PERMANENT_VETO
            and value.get("decision") == "PERMANENT_KNOWN_BAD_VETO"
            and value.get("transform") is None
            and value.get("downstream_authorized") is False)
    elif value.get("gate_status") == GATE_PASS:
        valid = (
            value.get("failure_class") is None
            and value.get("decision") ==
            "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"
            and value.get("downstream_authorized") is True)
        if valid:
            _validated_transform(value.get("transform"),
                                 "pair safe-cluster transform")
    else:
        valid = (
            value.get("gate_status") == GATE_FAIL
            and value.get("failure_class") ==
            NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER
            and value.get("decision") ==
            "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER"
            and value.get("transform") is None
            and value.get("downstream_authorized") is False)
    if not valid or any(value.get(key) is not False for key in POLICY_FALSE):
        raise StageGateResultError("pair gate result did not fail closed")
    return dict(value)


def build_fixed4_aggregate_result(
    *, task_id: str, pair_results: Sequence[Mapping[str, Any]],
    expected_pair_ids: Sequence[str], known_bad_pair_id: str,
) -> dict[str, Any]:
    """Aggregate three normal pairs and one permanent known-bad veto."""
    pair_ids = _exact_ids(expected_pair_ids, "fixed4 pair ids")
    if len(pair_ids) != 4 or known_bad_pair_id != pair_ids[-1]:
        raise StageGateResultError("fixed4 identity/order invalid")
    rows = [validate_pair_gate_result(row) for row in pair_results]
    if [row.get("pair_id") for row in rows] != pair_ids:
        raise StageGateResultError("fixed4 pair replay order mismatch")
    failed_pairs = [row["pair_id"] for row in rows[:-1]
                    if row["gate_status"] != GATE_PASS]
    known_bad = rows[-1]
    known_bad_vetoed = (
        known_bad["gate_status"] == GATE_FAIL
        and known_bad["failure_class"] == KNOWN_BAD_PERMANENT_VETO
        and known_bad["decision"] == "PERMANENT_KNOWN_BAD_VETO"
        and known_bad["transform"] is None)
    if failed_pairs:
        gate_status = GATE_FAIL
        failure_class = FIXED4_NORMAL_PAIR_CONSENSUS_FAILED
        decision = "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"
        downstream = False
    elif not known_bad_vetoed:
        gate_status = GATE_FAIL
        failure_class = FIXED4_KNOWN_BAD_VETO_FAILED
        decision = "FIXED4_KNOWN_BAD_VETO_FAILED"
        downstream = False
    else:
        gate_status, failure_class = GATE_PASS, None
        decision = "THREE_NORMALS_ACCEPTED_KNOWN_BAD_VETOED_NO_REFUSION"
        downstream = True
    return _sealed({
        "schema": AGGREGATE_GATE_SCHEMA,
        "stage": "fixed4_aggregate",
        "task_id": task_id,
        "execution_status": EXECUTION_SUCCEEDED,
        "gate_status": gate_status,
        "failure_class": failure_class,
        "decision": decision,
        "transform": None,
        "downstream_authorized": downstream,
        "replayed_pair_task_ids": [row["task_id"] for row in rows],
        "replayed_pair_ids": pair_ids,
        "failed_normal_pair_ids": failed_pairs,
        "known_bad_pair_id": known_bad_pair_id,
        "known_bad_vetoed": known_bad_vetoed,
        **POLICY_FALSE,
    })


def validate_fixed4_aggregate_result(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or not _payload_valid(value)
            or value.get("schema") != AGGREGATE_GATE_SCHEMA
            or value.get("stage") != "fixed4_aggregate"
            or value.get("execution_status") != EXECUTION_SUCCEEDED
            or value.get("transform") is not None
            or any(value.get(key) is not False for key in POLICY_FALSE)):
        raise StageGateResultError("fixed4 aggregate result binding invalid")
    pair_ids = _exact_ids(value.get("replayed_pair_ids"), "fixed4 pair ids")
    task_ids = _exact_ids(
        value.get("replayed_pair_task_ids"), "fixed4 pair task ids")
    failed = _exact_ids(
        value.get("failed_normal_pair_ids"), "failed normal pairs")
    if (len(pair_ids) != 4 or len(task_ids) != 4
            or not set(failed) <= set(pair_ids[:-1])):
        raise StageGateResultError("fixed4 aggregate replay closure invalid")
    if value.get("gate_status") == GATE_PASS:
        valid = (
            not failed and value.get("known_bad_vetoed") is True
            and value.get("failure_class") is None
            and value.get("decision") ==
            "THREE_NORMALS_ACCEPTED_KNOWN_BAD_VETOED_NO_REFUSION"
            and value.get("downstream_authorized") is True)
    elif failed:
        valid = (
            value.get("gate_status") == GATE_FAIL
            and value.get("failure_class") == FIXED4_NORMAL_PAIR_CONSENSUS_FAILED
            and value.get("decision") ==
            "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"
            and value.get("downstream_authorized") is False)
    else:
        valid = (
            value.get("gate_status") == GATE_FAIL
            and value.get("known_bad_vetoed") is False
            and value.get("failure_class") == FIXED4_KNOWN_BAD_VETO_FAILED
            and value.get("decision") == "FIXED4_KNOWN_BAD_VETO_FAILED"
            and value.get("downstream_authorized") is False)
    if not valid:
        raise StageGateResultError("fixed4 aggregate result did not fail closed")
    return dict(value)


def pair_gate_to_operational_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Map reviewed pair-core semantics into the sole operational-v5 vocabulary."""
    row = validate_pair_gate_result(value)
    passed = row["gate_status"] == GATE_PASS
    return {
        "status": "succeeded" if passed else "typed_failure",
        "typed_failure": (None if passed else {
            "type": row["failure_class"], "transform": None}),
        "replayed_hypothesis_task_ids": row["replayed_hypothesis_task_ids"],
        "safe_vote_hypothesis_task_ids": row["safe_vote_hypothesis_task_ids"],
        "gate_failed_hypothesis_task_ids": row["gate_failed_hypothesis_task_ids"],
        "typed_abstention_hypothesis_task_ids":
            row["typed_abstention_hypothesis_task_ids"],
        "decision": row["decision"],
        "safe_cluster_transform": row["transform"],
    }


def aggregate_gate_to_operational_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Map reviewed fixed4-core semantics into operational-v5 status fields."""
    row = validate_fixed4_aggregate_result(value)
    passed = row["gate_status"] == GATE_PASS
    return {
        "status": "succeeded" if passed else "typed_failure",
        "typed_failure": (None if passed else {
            "type": row["failure_class"], "transform": None}),
        "decision": row["decision"],
        "replayed_pair_task_ids": row["replayed_pair_task_ids"],
    }


def dispatch_stage_result(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pure explicit dispatcher for future reviewed wrappers and tests."""
    if stage == "v16_pair_hypothesis_cluster":
        return build_pair_gate_result(**dict(payload))
    if stage == "fixed4_aggregate":
        return build_fixed4_aggregate_result(**dict(payload))
    raise RegisteredStageRunnerUnavailable(
        f"reviewed stage-result dispatcher unavailable for: {stage}")


def _disabled(stage: str) -> Callable[[Mapping[str, Any], Path], Mapping[str, Any]]:
    def runner(task: Mapping[str, Any], task_root: Path) -> Mapping[str, Any]:
        del task, task_root
        raise RegisteredStageRunnerUnavailable(
            f"registered stage runner remains execution-disabled: {stage}")
    runner.__name__ = f"run_{stage}"
    runner.__qualname__ = runner.__name__
    return runner


def _hash_bound_dispatch(
    stage: str,
) -> Callable[[Mapping[str, Any], Path], Mapping[str, Any]]:
    def runner(task: Mapping[str, Any], task_root: Path) -> Mapping[str, Any]:
        del task_root
        payload = task.get("stage_runner_input")
        if not isinstance(payload, Mapping):
            raise RegisteredStageRunnerUnavailable(
                f"hash-bound stage_runner_input absent for: {stage}")
        return dispatch_stage_result(stage, payload)
    runner.__name__ = f"run_{stage}"
    runner.__qualname__ = runner.__name__
    return runner


run_colorpcr_direction = _disabled("colorpcr_direction")
run_bidirectional_multi_solver_pilot = _disabled(
    "bidirectional_multi_solver_pilot")
run_v16_pair_hypothesis_cluster = _hash_bound_dispatch(
    "v16_pair_hypothesis_cluster")
run_fixed4_aggregate = _hash_bound_dispatch("fixed4_aggregate")

STAGE_RUNNER_REGISTRY: Mapping[
    str, Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
] = {
    "colorpcr_direction": run_colorpcr_direction,
    "bidirectional_multi_solver_pilot": run_bidirectional_multi_solver_pilot,
    "v16_pair_hypothesis_cluster": run_v16_pair_hypothesis_cluster,
    "fixed4_aggregate": run_fixed4_aggregate,
}


def get_registered_stage_runner(
    stage: str,
) -> Callable[[Mapping[str, Any], Path], Mapping[str, Any]]:
    try:
        return STAGE_RUNNER_REGISTRY[stage]
    except KeyError as exc:
        raise RegisteredStageRunnerUnavailable(
            f"no reviewed runner registered for stage: {stage}") from exc


def registry_descriptor(source_sha256: str) -> list[dict[str, str]]:
    """Return the canonical registry identity bound by preflight/auth."""
    return [{
        "stage": stage,
        "module": __name__,
        "callable": runner.__name__,
        "source_path": "src/safety/v16_b716_fixed4_stage_runners.py",
        "source_sha256": source_sha256,
    } for stage, runner in STAGE_RUNNER_REGISTRY.items()]
