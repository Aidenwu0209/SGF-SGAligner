from __future__ import annotations

from copy import deepcopy

import pytest

from safety.v13_dual_solver_runtime import stable_json_sha256
from safety.v16_b716_fixed4_stage_runners import (
    FINITE_CONSENSUS_INCOMPATIBILITY,
    FIXED4_NORMAL_PAIR_CONSENSUS_FAILED,
    GATE_FAIL,
    GATE_PASS,
    NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER,
    RegisteredStageRunnerUnavailable,
    StageGateResultError,
    build_fixed4_aggregate_result,
    build_pair_gate_result,
    aggregate_gate_to_operational_fields,
    classify_finite_consensus,
    dispatch_stage_result,
    get_registered_stage_runner,
    pair_gate_to_operational_fields,
    validate_fixed4_aggregate_result,
    validate_hypothesis_gate_result,
    validate_pair_gate_result,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _hypothesis(task_id: str, *, passed: bool):
    return classify_finite_consensus(
        hypothesis_task_id=task_id,
        rotation_deg=1.0 if passed else 5.181,
        translation_m=0.01 if passed else 0.143,
        transform=IDENTITY,
    )


def _cluster(*, accepted: bool):
    if accepted:
        return {
            "accepted": True,
            "reason": "unique_safe_hypothesis_pose_cluster",
            "selected_transform": IDENTITY,
        }
    return {
        "accepted": False,
        "reason": "ambiguous_multiple_safe_hypothesis_pose_clusters",
    }


def _pair(task_id: str, pair_id: str, *, passed: bool, known_bad: bool = False):
    return build_pair_gate_result(
        task_id=task_id,
        pair_id=pair_id,
        replayed_hypothesis_task_ids=[f"{task_id}.h0", f"{task_id}.h1"],
        eligible_hypothesis_task_ids=[f"{task_id}.h0"],
        typed_abstention_hypothesis_task_ids=[f"{task_id}.h1"],
        hypothesis_gate_results=[_hypothesis(f"{task_id}.h0", passed=True)],
        cluster_decision=_cluster(accepted=passed),
        known_bad=known_bad,
    )


def test_cross_solver_5181deg_0143m_is_scientific_gate_fail():
    result = _hypothesis("pilot.h0", passed=False)
    assert result["execution_status"] == "succeeded"
    assert result["gate_status"] == GATE_FAIL
    assert result["failure_class"] == FINITE_CONSENSUS_INCOMPATIBILITY
    assert result["decision"] == "CROSS_SOLVER_INCOMPATIBLE"
    assert result["transform"] is None
    assert result["downstream_authorized"] is False
    validate_hypothesis_gate_result(result)


def test_frozen_thresholds_cannot_be_relabelled_or_widened():
    result = _hypothesis("pilot.h0", passed=False)
    tampered = deepcopy(result)
    tampered["threshold_rotation_deg"] = 6.0
    unsigned = {key: value for key, value in tampered.items()
                if key != "payload_sha256"}
    tampered["payload_sha256"] = stable_json_sha256(unsigned)
    with pytest.raises(StageGateResultError, match="thresholds changed"):
        validate_hypothesis_gate_result(tampered)


def test_pair_splits_eligible_safe_vote_gate_failed_and_typed_abstention():
    result = build_pair_gate_result(
        task_id="pair.p0",
        pair_id="p0",
        replayed_hypothesis_task_ids=["h0", "h1", "h2"],
        eligible_hypothesis_task_ids=["h0", "h1"],
        typed_abstention_hypothesis_task_ids=["h2"],
        hypothesis_gate_results=[
            _hypothesis("h0", passed=True),
            _hypothesis("h1", passed=False),
        ],
        cluster_decision=_cluster(accepted=True),
        known_bad=False,
    )
    assert result["safe_vote_hypothesis_task_ids"] == ["h0"]
    assert result["gate_failed_hypothesis_task_ids"] == ["h1"]
    assert result["typed_abstention_hypothesis_task_ids"] == ["h2"]
    assert result["gate_status"] == GATE_PASS
    validate_pair_gate_result(result)


def test_normal_pair_cluster_failure_is_auditable_and_unauthorized():
    result = build_pair_gate_result(
        task_id="pair.p0",
        pair_id="p0",
        replayed_hypothesis_task_ids=["h0"],
        eligible_hypothesis_task_ids=["h0"],
        typed_abstention_hypothesis_task_ids=[],
        hypothesis_gate_results=[_hypothesis("h0", passed=False)],
        cluster_decision=_cluster(accepted=False),
        known_bad=False,
    )
    assert result["execution_status"] == "succeeded"
    assert result["gate_status"] == GATE_FAIL
    assert result["failure_class"] == NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER
    assert result["transform"] is None
    assert result["downstream_authorized"] is False
    validate_pair_gate_result(result)


def test_fixed4_aggregate_preserves_normal_failure_receipt():
    pairs = [
        _pair("pair.0", "p0", passed=True),
        _pair("pair.1", "p1", passed=False),
        _pair("pair.2", "p2", passed=True),
        _pair("pair.3", "p3", passed=False, known_bad=True),
    ]
    result = build_fixed4_aggregate_result(
        task_id="fixed4.aggregate",
        pair_results=pairs,
        expected_pair_ids=["p0", "p1", "p2", "p3"],
        known_bad_pair_id="p3",
    )
    assert result["gate_status"] == GATE_FAIL
    assert result["failure_class"] == FIXED4_NORMAL_PAIR_CONSENSUS_FAILED
    assert result["failed_normal_pair_ids"] == ["p1"]
    assert result["transform"] is None
    assert result["downstream_authorized"] is False
    validate_fixed4_aggregate_result(result)


def test_fixed4_pass_does_not_authorize_reconstruction_or_refusion():
    pairs = [
        _pair("pair.0", "p0", passed=True),
        _pair("pair.1", "p1", passed=True),
        _pair("pair.2", "p2", passed=True),
        _pair("pair.3", "p3", passed=False, known_bad=True),
    ]
    result = dispatch_stage_result("fixed4_aggregate", {
        "task_id": "fixed4.aggregate",
        "pair_results": pairs,
        "expected_pair_ids": ["p0", "p1", "p2", "p3"],
        "known_bad_pair_id": "p3",
    })
    assert result["gate_status"] == GATE_PASS
    assert result["downstream_authorized"] is True
    assert result["reconstruction_authorized"] is False
    assert result["refusion_run"] is False
    validate_fixed4_aggregate_result(result)


def test_core_pair_and_aggregate_map_to_operational_v5_vocabulary():
    normal_fail = _pair("pair.0", "p0", passed=False)
    pair_fields = pair_gate_to_operational_fields(normal_fail)
    assert pair_fields["status"] == "typed_failure"
    assert pair_fields["typed_failure"] == {
        "type": "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER", "transform": None}
    assert set(pair_fields) == {
        "status", "typed_failure", "replayed_hypothesis_task_ids",
        "safe_vote_hypothesis_task_ids", "gate_failed_hypothesis_task_ids",
        "typed_abstention_hypothesis_task_ids", "decision",
        "safe_cluster_transform"}

    aggregate = build_fixed4_aggregate_result(
        task_id="fixed4.aggregate",
        pair_results=[normal_fail, _pair("pair.1", "p1", passed=True),
                      _pair("pair.2", "p2", passed=True),
                      _pair("pair.3", "p3", passed=False, known_bad=True)],
        expected_pair_ids=["p0", "p1", "p2", "p3"],
        known_bad_pair_id="p3")
    aggregate_fields = aggregate_gate_to_operational_fields(aggregate)
    assert aggregate_fields["status"] == "typed_failure"
    assert aggregate_fields["typed_failure"] == {
        "type": "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED", "transform": None}


def test_registered_pair_runner_requires_future_hash_bound_input(tmp_path):
    runner = get_registered_stage_runner("v16_pair_hypothesis_cluster")
    with pytest.raises(RegisteredStageRunnerUnavailable,
                       match="hash-bound stage_runner_input absent"):
        runner({"stage": "v16_pair_hypothesis_cluster"}, tmp_path)


def test_dispatch_rejects_unimplemented_solver_stage():
    with pytest.raises(RegisteredStageRunnerUnavailable):
        dispatch_stage_result("bidirectional_multi_solver_pilot", {})
