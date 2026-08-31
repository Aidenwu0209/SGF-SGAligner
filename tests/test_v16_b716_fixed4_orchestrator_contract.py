import json
from pathlib import Path

import pytest

from safety.v13_dual_solver_runtime import sha256_file
from safety.v16_b716_fixed4_orchestrator_contract import (
    EXPECTED_NODE_COUNT,
    EXPECTED_STAGE_COUNTS,
    FIXED_PAIR_ORDER,
    KNOWN_BAD_PAIR_ID,
    Fixed4OrchestratorContractError,
    bind_hypotheses,
    build_task_dag,
    materialize_planning_receipts,
    synthetic_fixture_bindings,
    validate_preregister,
)


REPO = Path(__file__).resolve().parents[1]
PREREGISTER = REPO / "manifests/v16_b716_fixed4_orchestrator_preregister.json"


def _preregister():
    return json.loads(PREREGISTER.read_text())


def _dag():
    preregister_sha = sha256_file(PREREGISTER)
    return build_task_dag(
        synthetic_fixture_bindings(), preregister_sha,
        synthetic_fixture=True), preregister_sha


def test_historical_preregister_is_well_formed_disabled_and_source_pinned():
    """Validate the frozen record, not an unrelated later worktree state."""
    value = _preregister()
    validate_preregister(value)
    assert value["schema"] == "v16-b716-fixed4-orchestrator-preregister-v2"
    assert value["frozen"] is True
    assert value["disabled"] is True
    assert value["execution_authorized"] is False
    assert value["gpu_allowed"] is False
    assert value["model_execution_allowed"] is False
    assert value["solver_execution_allowed"] is False
    assert value["official92_allowed"] is False
    assert value["gt_allowed"] is False
    pins = value["source_sha256"]
    assert pins
    assert all(isinstance(relative, str) and relative for relative in pins)
    assert all(isinstance(digest, str) and len(digest) == 64
               and set(digest) <= set("0123456789abcdef")
               for digest in pins.values())
    assert value["reviewed_real_bindings"]["authorization_ready"] is False
    assert value["unresolved_p0"]


def test_historical_preregister_cannot_authorize_active_execution():
    """Any active v2 needs fresh pins from its reviewed candidate commit."""
    value = _preregister()
    for field in ("execution_authorized", "gpu_allowed",
                  "model_execution_allowed", "solver_execution_allowed"):
        changed = json.loads(json.dumps(value))
        changed[field] = True
        with pytest.raises(Fixed4OrchestratorContractError,
                           match="frozen and disabled"):
            validate_preregister(changed)


def test_exact_dag_counts_and_complete_hypothesis_replay():
    dag, _sha = _dag()
    assert dag["node_count"] == EXPECTED_NODE_COUNT == 6091
    assert dag["stage_counts"] == EXPECTED_STAGE_COUNTS
    assert dag["stage_counts"]["colorpcr_worker"] == 136
    assert dag["stage_counts"]["sentinel_direction_cache"] == 68
    assert dag["stage_counts"]["exact_three_direction_cache"] == 68
    assert dag["stage_counts"]["v13_solver_row"] == 5440
    prepared = [row for row in dag["nodes"]
                if row["stage"] == "prepared_input"]
    assert [(row["pair_id"], row["hypothesis_index"]) for row in prepared] == [
        (pair_id, index)
        for pair_id, count in zip(FIXED_PAIR_ORDER, (12, 8, 2, 12))
        for index in range(count)
    ]
    assert sum(row["contains_typed_failure_members"] for row in prepared) == 10
    assert sum(bool(row["existing_typed_failure_member_candidate_indices"])
               for row in prepared) == 8
    assert sum(row["safe_pose_vote_eligible"] for row in prepared) == 24
    assert sum(row["selector_eligible"] for row in prepared) == 0
    assert all(row["typed_failure_policy"] == "explicit_replay_never_filter"
               for row in prepared if row["contains_typed_failure_members"])
    assert dag["all_members_ok_filter_allowed"] is False
    assert dag["result_selection_allowed"] is False
    assert dag["typed_failure_hypotheses_safe_vote_count"] == 0
    assert dag["selector_eligible_hypothesis_count"] == 0


def test_known_bad_all_twelve_replayed_then_permanent_veto():
    dag, _sha = _dag()
    bad_prepared = [row for row in dag["nodes"]
                    if row["stage"] == "prepared_input"
                    and row["pair_id"] == KNOWN_BAD_PAIR_ID]
    assert len(bad_prepared) == 12
    bad_v16 = [row for row in dag["nodes"]
               if row["stage"] == "v16_pair_hypothesis_cluster"
               and row["pair_id"] == KNOWN_BAD_PAIR_ID]
    assert len(bad_v16) == 1
    assert bad_v16[0]["permanent_veto"] is True
    assert bad_v16[0]["expected_hypothesis_count"] == 12
    assert len(bad_v16[0]["upstream_task_ids"]) == 12


def test_normal_pairs_require_unique_compatible_safe_cluster():
    dag, _sha = _dag()
    rows = [row for row in dag["nodes"]
            if row["stage"] == "v16_pair_hypothesis_cluster"]
    for row in rows[:-1]:
        assert row["acceptance_rule"] == (
            "one_unique_complete_linkage_safe_hypothesis_pose_cluster")
        assert row["best_score_forbidden"] is True
        assert row["majority_forbidden"] is True
    aggregate = dag["nodes"][-1]
    assert aggregate["normal_pair_rule"] == (
        "all_three_normals_each_require_unique_compatible_safe_pose_cluster")
    assert aggregate["known_bad_rule"] == (
        "all_12_replayed_then_permanent_veto")


def test_create_only_receipts_resume_identically_and_reject_tamper(tmp_path):
    dag, preregister_sha = _dag()
    first = materialize_planning_receipts(tmp_path, dag, preregister_sha)
    assert first["receipt_count"] == EXPECTED_NODE_COUNT
    assert first["states"] == {"created": EXPECTED_NODE_COUNT,
                                "resumed_identical": 0}
    second = materialize_planning_receipts(tmp_path, dag, preregister_sha)
    assert second["states"] == {"created": 0,
                                 "resumed_identical": EXPECTED_NODE_COUNT}
    target = tmp_path / first["receipts"][0]["path"]
    target.chmod(0o644)
    value = json.loads(target.read_text())
    value["execution_authorized"] = True
    target.write_text(json.dumps(value))
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="existing receipt differs"):
        materialize_planning_receipts(tmp_path, dag, preregister_sha)


def test_preregister_cannot_silently_authorize_real_inputs():
    value = _preregister()
    value["reviewed_real_bindings"]["authorization_ready"] = True
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="P0 blockers"):
        validate_preregister(value)


def test_reordering_or_dropping_hypothesis_fails_closed():
    rows = synthetic_fixture_bindings()
    preregister_sha = sha256_file(PREREGISTER)
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="ordered exact"):
        build_task_dag(rows[:-1], preregister_sha, synthetic_fixture=True)
    swapped = list(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="ordered exact"):
        build_task_dag(swapped, preregister_sha, synthetic_fixture=True)


def test_exact_and_prepared_typed_failures_are_cross_bound():
    rows = synthetic_fixture_bindings()
    exact = {pair: [] for pair in FIXED_PAIR_ORDER}
    prepared = {pair: [] for pair in FIXED_PAIR_ORDER}
    for row in rows:
        exact[row.pair_id].append({
            "hypothesis_index": row.hypothesis_index,
            "hypothesis_sha256": row.hypothesis_sha256,
            "contains_typed_failure_members":
                row.contains_typed_failure_members,
            "existing_typed_failure_member_candidate_indices":
                list(row.existing_typed_failure_member_candidate_indices),
            "new_typed_failure_member_candidate_indices":
                list(row.new_typed_failure_member_candidate_indices),
            "typed_failure_member_candidate_indices":
                list(row.typed_failure_member_candidate_indices),
        })
        prepared[row.pair_id].append({
            "hypothesis_index": row.hypothesis_index,
            "hypothesis_sha256": row.hypothesis_sha256,
            "prepared_input_path": row.prepared_input_path,
            "prepared_input_sha256": row.prepared_input_sha256,
            "contains_typed_failure_members":
                row.contains_typed_failure_members,
            "existing_typed_failure_member_candidate_indices":
                list(row.existing_typed_failure_member_candidate_indices),
            "new_typed_failure_member_candidate_indices":
                list(row.new_typed_failure_member_candidate_indices),
            "typed_failure_member_candidate_indices":
                list(row.typed_failure_member_candidate_indices),
            "safe_pose_vote_eligible": row.safe_pose_vote_eligible,
            "selector_eligible": row.selector_eligible,
        })
    assert len(bind_hypotheses(exact, prepared)) == 34
    prepared[FIXED_PAIR_ORDER[0]][0][
        "typed_failure_member_candidate_indices"] = []
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="typed-failure binding"):
        bind_hypotheses(exact, prepared)


def test_typed_hypothesis_cannot_be_promoted_to_safe_vote_or_selector():
    rows = synthetic_fixture_bindings()
    exact = {pair: [] for pair in FIXED_PAIR_ORDER}
    prepared = {pair: [] for pair in FIXED_PAIR_ORDER}
    for row in rows:
        common = {
            "hypothesis_index": row.hypothesis_index,
            "hypothesis_sha256": row.hypothesis_sha256,
            "contains_typed_failure_members": row.contains_typed_failure_members,
            "existing_typed_failure_member_candidate_indices":
                list(row.existing_typed_failure_member_candidate_indices),
            "new_typed_failure_member_candidate_indices":
                list(row.new_typed_failure_member_candidate_indices),
            "typed_failure_member_candidate_indices":
                list(row.typed_failure_member_candidate_indices),
        }
        exact[row.pair_id].append(dict(common))
        prepared[row.pair_id].append({
            **common,
            "prepared_input_path": row.prepared_input_path,
            "prepared_input_sha256": row.prepared_input_sha256,
            "safe_pose_vote_eligible": row.safe_pose_vote_eligible,
            "selector_eligible": False,
        })
    typed = next((pair, index) for pair, values in prepared.items()
                 for index, value in enumerate(values)
                 if value["contains_typed_failure_members"])
    pair, index = typed
    prepared[pair][index]["safe_pose_vote_eligible"] = True
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="typed-failure binding"):
        bind_hypotheses(exact, prepared)
    prepared[pair][index]["safe_pose_vote_eligible"] = False
    prepared[pair][index]["selector_eligible"] = True
    with pytest.raises(Fixed4OrchestratorContractError,
                       match="typed-failure binding"):
        bind_hypotheses(exact, prepared)
