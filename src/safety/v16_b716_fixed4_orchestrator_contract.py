"""Execution-disabled contract for the final b716 fixed4 replay.

The module builds a deterministic task graph and create-only planning
receipts.  It never imports or executes ColorPCR, GeoTransformer, PointDSC,
pygcransac, ICP, a GPU runtime, or official92.  Real evidence can only be
planned after the sealed exact191 and prepared-builder manifests are bound by
path and SHA in a later reviewed preregistration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256


PREREGISTER_SCHEMA = "v16-b716-fixed4-orchestrator-preregister-v2"
DAG_SCHEMA = "v16-b716-fixed4-orchestrator-dag-v1"
RECEIPT_SCHEMA = "v16-b716-fixed4-planned-stage-receipt-v1"
RECEIPT_MANIFEST_SCHEMA = "v16-b716-fixed4-receipt-manifest-v1"
EXACT191_SCHEMA = "v16-b716-exact191-merged-manifest-v1"
EXACT191_PAIR_SCHEMA = "v16-b716-exact191-pair-v1"
ALLOWLIST_SCHEMA = "v16-b716-frozen-hypothesis-allowlist-v1"
BUILDER_SCHEMA = "v16-b716-matched-region-prepared-builder-v2"
PAIR_BUILDER_SCHEMA = "v16-b716-matched-region-prepared-pair-v2"
PREPARED_SCHEMA = "v16-b716-matched-region-prepared-input-v2"
OFFICIAL_RELEASE_SHA256 = (
    "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e"
)
FIXED_PAIR_ORDER = (
    "09582205-e2c2-2de1-9475-1cdac7639e60_to_"
    "0958220d-e2c2-2de1-9710-c37018da1883",
    "68bae76c-3567-2f7c-827d-373035a2d942_to_"
    "68bae76e-3567-2f7c-82bd-a09641695364",
    "f38169cf-378c-2a65-855f-05d491a3f26e_to_"
    "f38169c7-378c-2a65-8543-3c7481e856fe",
    "6a36052f-fa53-2915-9400-831b60c63077_to_"
    "6a36052d-fa53-2915-9764-30d81b2cc2b5",
)
EXPECTED_HYPOTHESES = (12, 8, 2, 12)
EXPECTED_BY_PAIR = dict(zip(FIXED_PAIR_ORDER, EXPECTED_HYPOTHESES))
EXPECTED_EXISTING_TYPED_FAILURE_COUNT = 16
EXPECTED_NEW_TYPED_FAILURE_COUNT = 12
EXPECTED_TYPED_FAILURE_COUNT = 28
EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES = 8
EXPECTED_TYPED_FAILURE_HYPOTHESES = 10
KNOWN_BAD_PAIR_ID = FIXED_PAIR_ORDER[-1]
DIRECTIONS = ("forward", "reverse")
SENTINELS = ("identity", "proper_nonzero")
SOLVERS = ("pointdsc", "pygcransac")
REPEATS = tuple(range(5))
MAX_CANDIDATES_PER_HYPOTHESIS = 8
EXPECTED_STAGE_COUNTS = {
    "prepared_input": 34,
    "colorpcr_worker": 136,
    "sentinel_direction_cache": 68,
    "exact_three_direction_cache": 68,
    "v14_candidate_set": 34,
    "v13_solver_row": 5440,
    "v13_strict_candidate_gate": 272,
    "v15_hypothesis_candidate_cluster": 34,
    "v16_pair_hypothesis_cluster": 4,
    "fixed4_aggregate": 1,
}
EXPECTED_NODE_COUNT = sum(EXPECTED_STAGE_COUNTS.values())


class Fixed4OrchestratorContractError(RuntimeError):
    """A frozen identity, receipt, or fail-closed policy is malformed."""


@dataclass(frozen=True)
class HypothesisBinding:
    pair_id: str
    pair_ordinal: int
    hypothesis_index: int
    hypothesis_sha256: str
    prepared_input_path: str
    prepared_input_sha256: str
    contains_typed_failure_members: bool
    existing_typed_failure_member_candidate_indices: tuple[int, ...]
    new_typed_failure_member_candidate_indices: tuple[int, ...]
    typed_failure_member_candidate_indices: tuple[int, ...]
    safe_pose_vote_eligible: bool
    selector_eligible: bool


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise Fixed4OrchestratorContractError(f"invalid {name}")
    return value


def _payload_valid(value: Mapping[str, Any]) -> bool:
    expected = value.get("payload_sha256")
    if not isinstance(expected, str):
        return False
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    return expected == stable_json_sha256(unsigned)


def _load_bound_json(path: Path, expected_sha256: str,
                     name: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or sha256_file(path) != _sha(expected_sha256, name):
        raise Fixed4OrchestratorContractError(f"{name} path/SHA mismatch")
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise Fixed4OrchestratorContractError(
            f"{name} is not valid JSON") from exc
    if (not isinstance(value, dict) or sha256_file(path) != before
            or not _payload_valid(value)):
        raise Fixed4OrchestratorContractError(
            f"{name} changed or has invalid payload SHA")
    return value


def _resolve_bound_file(root: Path, relative: Any, expected_bytes: Any,
                        expected_sha256: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise Fixed4OrchestratorContractError(f"{name} path missing")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Fixed4OrchestratorContractError(
            f"{name} escapes sealed root") from exc
    size_bound = expected_bytes is not None
    if size_bound and (type(expected_bytes) is not int or expected_bytes < 1):
        raise Fixed4OrchestratorContractError(f"{name} byte count invalid")
    if (not path.is_file()
            or (size_bound and path.stat().st_size != expected_bytes)
            or sha256_file(path) != _sha(expected_sha256, f"{name} SHA")):
        raise Fixed4OrchestratorContractError(f"{name} bytes/SHA mismatch")
    return path


def _typed_failure_partition(
    value: Mapping[str, Any], name: str, *, require_all_members_ok: bool,
    require_safe_vote: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    existing = value.get("existing_typed_failure_member_candidate_indices")
    new = value.get("new_typed_failure_member_candidate_indices")
    combined = value.get("typed_failure_member_candidate_indices")
    if (not isinstance(existing, list) or not isinstance(new, list)
            or not isinstance(combined, list)):
        raise Fixed4OrchestratorContractError(
            f"{name} typed-failure partition missing")
    for label, rows in (("existing", existing), ("new", new),
                        ("combined", combined)):
        if (len(rows) != len(set(rows))
                or any(type(item) is not int or item < 0 for item in rows)):
            raise Fixed4OrchestratorContractError(
                f"{name} {label} typed-failure indices invalid")
    if (set(existing).intersection(new)
            or combined != existing + new
            or value.get("contains_typed_failure_members") is not bool(combined)):
        raise Fixed4OrchestratorContractError(
            f"{name} typed-failure partition mismatch")
    if (require_all_members_ok
            and value.get("all_members_ok") is not (not combined)):
        raise Fixed4OrchestratorContractError(
            f"{name} all-members-ok mismatch")
    if (require_safe_vote
            and value.get("safe_pose_vote_eligible") is not (not combined)):
        raise Fixed4OrchestratorContractError(
            f"{name} typed hypothesis is not fail-closed for safe voting")
    return tuple(existing), tuple(new), tuple(combined)


def validate_preregister(value: Mapping[str, Any]) -> None:
    if (value.get("schema") != PREREGISTER_SCHEMA
            or value.get("frozen") is not True
            or value.get("disabled") is not True
            or value.get("execution_authorized") is not False
            or value.get("gpu_allowed") is not False
            or value.get("model_execution_allowed") is not False
            or value.get("solver_execution_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("gt_allowed") is not False
            or value.get("default_checkpoint_replacement_allowed") is not False
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or value.get("fixed_pair_order") != list(FIXED_PAIR_ORDER)
            or value.get("expected_hypothesis_distribution")
            != list(EXPECTED_HYPOTHESES)
            or value.get("expected_stage_counts") != EXPECTED_STAGE_COUNTS
            or value.get("expected_node_count") != EXPECTED_NODE_COUNT):
        raise Fixed4OrchestratorContractError(
            "orchestrator preregistration is not frozen and disabled")
    if (value.get("expected_existing_typed_failure_count")
            != EXPECTED_EXISTING_TYPED_FAILURE_COUNT
            or value.get("expected_new_typed_failure_count")
            != EXPECTED_NEW_TYPED_FAILURE_COUNT
            or value.get("expected_typed_failure_total_count")
            != EXPECTED_TYPED_FAILURE_COUNT
            or value.get("expected_existing_typed_failure_hypothesis_count")
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or value.get("expected_all_typed_failure_hypothesis_count")
            != EXPECTED_TYPED_FAILURE_HYPOTHESES):
        raise Fixed4OrchestratorContractError(
            "typed-failure preregistration counts drifted")
    rules = value.get("selection_rules")
    if (not isinstance(rules, Mapping)
            or rules.get("all_34_hypotheses_replayed") is not True
            or rules.get("typed_failure_members_explicit_never_filtered") is not True
            or rules.get("all_members_ok_filter_forbidden") is not True
            or rules.get("best_score_forbidden") is not True
            or rules.get("majority_forbidden") is not True
            or rules.get("result_based_selection_forbidden") is not True
            or rules.get("known_bad_all_12_replayed") is not True
            or rules.get("known_bad_permanent_veto") is not True
            or rules.get("typed_failure_safe_pose_vote_allowed") is not False
            or rules.get("typed_failure_selector_allowed") is not False
            or rules.get("typed_failure_quorum_allowed") is not False
            or rules.get("typed_failure_cluster_contribution_allowed") is not False
            or rules.get("typed_failure_aggregate_acceptance_allowed") is not False
            or rules.get("selector_eligible_required_false") is not True
            or rules.get("normal_acceptance")
            != "each_normal_requires_one_unique_complete_linkage_safe_pose_cluster"):
        raise Fixed4OrchestratorContractError("selection rules are not fail-closed")
    bindings = value.get("reviewed_real_bindings")
    p0 = value.get("unresolved_p0")
    if (not isinstance(bindings, Mapping) or not isinstance(p0, list) or not p0
            or bindings.get("authorization_ready") is not False
            or bindings.get("exact191_manifest_path") is not None
            or bindings.get("exact191_manifest_sha256") is not None
            or bindings.get("exact72_lineage_manifest_path") is not None
            or bindings.get("exact72_lineage_manifest_sha256") is not None
            or bindings.get("prepared_builder_manifest_path") is not None
            or bindings.get("prepared_builder_manifest_sha256") is not None):
        raise Fixed4OrchestratorContractError(
            "unreviewed real inputs must remain explicit P0 blockers")
    pins = value.get("source_sha256")
    if not isinstance(pins, Mapping) or not pins:
        raise Fixed4OrchestratorContractError("source SHA closure is missing")
    for relative, digest in pins.items():
        if not isinstance(relative, str) or not relative:
            raise Fixed4OrchestratorContractError("source pin path invalid")
        _sha(digest, f"source pin {relative}")


def verify_source_pins(repo: Path, preregister: Mapping[str, Any]) -> None:
    validate_preregister(preregister)
    repo = Path(repo).resolve()
    for relative, expected in preregister["source_sha256"].items():
        path = (repo / relative).resolve()
        try:
            path.relative_to(repo)
        except ValueError as exc:
            raise Fixed4OrchestratorContractError(
                "source pin escapes repository") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise Fixed4OrchestratorContractError(
                f"source pin mismatch: {relative}")


def load_exact191_hypotheses(
    manifest_path: Path, manifest_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    value = _load_bound_json(manifest_path, manifest_sha256,
                             "exact191 manifest")
    if (value.get("schema") != EXACT191_SCHEMA
            or value.get("sealed") is not True
            or value.get("provenance_sealed") is not True
            or value.get("candidate_count") != 191
            or value.get("existing_count") != 119
            or value.get("new_authorized_count") != 72
            or value.get("typed_failure_existing_count")
            != EXPECTED_EXISTING_TYPED_FAILURE_COUNT
            or value.get("new_authorized_typed_failure_count")
            != EXPECTED_NEW_TYPED_FAILURE_COUNT
            or value.get("typed_failure_total_count")
            != EXPECTED_TYPED_FAILURE_COUNT
            or value.get("hypothesis_count") != 34
            or value.get("hypotheses_with_existing_typed_failure_members")
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or value.get("hypotheses_with_typed_failure_members")
            != EXPECTED_TYPED_FAILURE_HYPOTHESES
            or value.get("typed_failures_visible_and_never_filtered") is not True
            or value.get("fixed_hypothesis_distribution")
            != list(EXPECTED_HYPOTHESES)
            or value.get("candidate_selection_allowed") is not False
            or value.get("result_based_selection_allowed") is not False
            or value.get("hypothesis_selection_allowed") is not False
            or value.get("gt_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256):
        raise Fixed4OrchestratorContractError("exact191 contract mismatch")
    rows = value.get("pairs")
    if (not isinstance(rows, list)
            or [row.get("pair_id") for row in rows] != list(FIXED_PAIR_ORDER)):
        raise Fixed4OrchestratorContractError("exact191 fixed4 order mismatch")
    root = manifest_path.parent
    result: dict[str, list[dict[str, Any]]] = {}
    existing_typed_hypotheses = 0
    typed_hypotheses = 0
    existing_typed_members = 0
    new_typed_members = 0
    for ordinal, row in enumerate(rows):
        pair_id = FIXED_PAIR_ORDER[ordinal]
        entries_path = _resolve_bound_file(
            root, row.get("entries_path"), row.get("entries_bytes"),
            row.get("entries_sha256"), f"entries {pair_id}")
        entries = _load_bound_json(
            entries_path, row["entries_sha256"], f"entries {pair_id}")
        entry_rows = entries.get("entries")
        if (entries.get("schema") != EXACT191_PAIR_SCHEMA
                or entries.get("pair_id") != pair_id
                or entries.get("short_id") != row.get("short_id")
                or not isinstance(entry_rows, list)
                or entries.get("candidate_count") != len(entry_rows)
                or entries.get("candidate_count") != row.get("candidate_count")
                or any(not isinstance(item, Mapping) for item in entry_rows)):
            raise Fixed4OrchestratorContractError(
                "exact191 pair entry closure mismatch")
        entry_indices = [item.get("candidate_index") for item in entry_rows]
        if (any(type(index) is not int for index in entry_indices)
                or len(set(entry_indices)) != len(entry_indices)):
            raise Fixed4OrchestratorContractError(
                "exact191 pair candidate identity mismatch")
        existing_failure_indices = {
            item["candidate_index"] for item in entry_rows
            if item.get("status") == "insufficient_post_voxel_points"
            and isinstance(item.get("origin"), Mapping)
            and item["origin"].get("kind") == "frozen_existing"}
        new_failure_indices = {
            item["candidate_index"] for item in entry_rows
            if item.get("status") == "insufficient_post_voxel_points"
            and isinstance(item.get("origin"), Mapping)
            and item["origin"].get("kind") == "authorized_backfill"}
        if (len(existing_failure_indices) != row.get("existing_failed_count")
                or len(new_failure_indices) != row.get("new_typed_failure_count")
                or entries.get("existing_failed_count")
                    != row.get("existing_failed_count")
                or entries.get("new_typed_failure_count")
                    != row.get("new_typed_failure_count")
                or entries.get("existing_count") != row.get("existing_count")
                or entries.get("new_count") != row.get("new_count")
                or entries.get("new_ok_count") != row.get("new_ok_count")
                or len(existing_failure_indices | new_failure_indices)
                    != sum(item.get("status")
                           == "insufficient_post_voxel_points"
                           for item in entry_rows)):
            raise Fixed4OrchestratorContractError(
                "exact191 pair typed-result count mismatch")
        path = _resolve_bound_file(
            root, row.get("allowlist_path"), row.get("allowlist_bytes"),
            row.get("allowlist_sha256"), f"allowlist {pair_id}")
        allowlist = _load_bound_json(
            path, row["allowlist_sha256"], f"allowlist {pair_id}")
        hypotheses = allowlist.get("hypotheses")
        if (allowlist.get("schema") != ALLOWLIST_SCHEMA
                or allowlist.get("pair_id") != pair_id
                or allowlist.get("hypothesis_count") != EXPECTED_HYPOTHESES[ordinal]
                or allowlist.get("all_hypotheses_must_be_replayed") is not True
                or allowlist.get(
                    "typed_failure_members_visible_and_never_filtered") is not True
                or not isinstance(hypotheses, list)
                or len(hypotheses) != EXPECTED_HYPOTHESES[ordinal]):
            raise Fixed4OrchestratorContractError("allowlist contract mismatch")
        checked = []
        for index, hypothesis in enumerate(hypotheses):
            if hypothesis.get("hypothesis_index") != index:
                raise Fixed4OrchestratorContractError(
                    "exact191 hypothesis order mismatch")
            existing, new, failed = _typed_failure_partition(
                hypothesis, "exact191 hypothesis",
                require_all_members_ok=True, require_safe_vote=False)
            if (not set(existing).issubset(existing_failure_indices)
                    or not set(new).issubset(new_failure_indices)):
                raise Fixed4OrchestratorContractError(
                    "exact191 hypothesis typed-result provenance mismatch")
            _sha(hypothesis.get("hypothesis_sha256"), "hypothesis SHA")
            existing_typed_hypotheses += int(bool(existing))
            typed_hypotheses += int(bool(failed))
            checked.append(dict(hypothesis))
        if (row.get("hypotheses_with_existing_typed_failure_members")
                != sum(bool(item.get(
                    "existing_typed_failure_member_candidate_indices"))
                    for item in checked)
                or row.get("hypotheses_with_typed_failure_members")
                != sum(item["contains_typed_failure_members"]
                       for item in checked)
                or entries.get("hypotheses_with_existing_typed_failure_members")
                    != row.get("hypotheses_with_existing_typed_failure_members")
                or entries.get("hypotheses_with_typed_failure_members")
                    != row.get("hypotheses_with_typed_failure_members")):
            raise Fixed4OrchestratorContractError(
                "exact191 pair typed-failure closure mismatch")
        existing_typed_members += len(existing_failure_indices)
        new_typed_members += len(new_failure_indices)
        result[pair_id] = checked
    if (existing_typed_hypotheses
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or typed_hypotheses != EXPECTED_TYPED_FAILURE_HYPOTHESES
            or existing_typed_members != EXPECTED_EXISTING_TYPED_FAILURE_COUNT
            or new_typed_members != EXPECTED_NEW_TYPED_FAILURE_COUNT):
        raise Fixed4OrchestratorContractError(
            "exact191 typed-failure hypothesis closure changed")
    return result


def load_prepared_hypotheses(
    manifest_path: Path, manifest_sha256: str,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    value = _load_bound_json(manifest_path, manifest_sha256,
                             "prepared builder manifest")
    if (value.get("schema") != BUILDER_SCHEMA
            or value.get("sealed") is not True
            or value.get("cpu_only") is not True
            or value.get("worker_execution_authorized") is not False
            or value.get("registration_executed") is not False
            or value.get("official92_executed") is not False
            or value.get("gt_consumed") is not False
            or value.get("fallback_used") is not False
            or value.get("geot_result_filtering_used") is not False
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or value.get("pair_count") != 4
            or value.get("hypothesis_count") != 34
            or value.get("hypothesis_distribution")
            != list(EXPECTED_HYPOTHESES)
            or value.get("existing_typed_failure_count")
            != EXPECTED_EXISTING_TYPED_FAILURE_COUNT
            or value.get("new_typed_failure_count")
            != EXPECTED_NEW_TYPED_FAILURE_COUNT
            or value.get("typed_failure_total_count")
            != EXPECTED_TYPED_FAILURE_COUNT
            or value.get("hypotheses_with_existing_typed_failure_members")
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or value.get("hypotheses_with_typed_failure_members")
            != EXPECTED_TYPED_FAILURE_HYPOTHESES
            or value.get("typed_failures_visible_and_never_filtered") is not True):
        raise Fixed4OrchestratorContractError(
            "prepared builder contract mismatch")
    rows = value.get("pairs")
    if (not isinstance(rows, list)
            or [row.get("pair_id") for row in rows] != list(FIXED_PAIR_ORDER)):
        raise Fixed4OrchestratorContractError("prepared fixed4 order mismatch")
    root = manifest_path.parent
    result: dict[str, list[dict[str, Any]]] = {}
    paths: set[str] = set()
    existing_typed_hypotheses = 0
    typed_hypotheses = 0
    existing_typed_members = 0
    new_typed_members = 0
    for ordinal, row in enumerate(rows):
        pair_id = FIXED_PAIR_ORDER[ordinal]
        pair_path = _resolve_bound_file(
            root, row.get("pair_manifest_path"),
            None,
            row.get("pair_manifest_sha256"), f"prepared pair {pair_id}")
        pair = _load_bound_json(
            pair_path, row["pair_manifest_sha256"], f"prepared pair {pair_id}")
        artifacts = pair.get("hypotheses")
        if (pair.get("schema") != PAIR_BUILDER_SCHEMA
                or pair.get("pair_id") != pair_id
                or pair.get("expected_hypothesis_count")
                != EXPECTED_HYPOTHESES[ordinal]
                or pair.get("hypothesis_count") != EXPECTED_HYPOTHESES[ordinal]
                or pair.get("all_hypotheses_replayed") is not True
                or pair.get(
                    "typed_failure_members_visible_and_never_filtered") is not True
                or pair.get("geot_result_filtering_used") is not False
                or pair.get("worker_execution_authorized") is not False
                or not isinstance(artifacts, list)
                or len(artifacts) != EXPECTED_HYPOTHESES[ordinal]):
            raise Fixed4OrchestratorContractError(
                "prepared pair manifest mismatch")
        checked = []
        for index, artifact in enumerate(artifacts):
            if artifact.get("hypothesis_index") != index:
                raise Fixed4OrchestratorContractError(
                    "prepared hypothesis order mismatch")
            hypothesis_sha = _sha(artifact.get("hypothesis_sha256"),
                                  "prepared hypothesis SHA")
            if (artifact.get(
                    "typed_failure_members_visible_and_never_filtered") is not True
                    or artifact.get("selector_eligible") is not False):
                raise Fixed4OrchestratorContractError(
                    "prepared typed-failure metadata mismatch")
            existing, new, failed = _typed_failure_partition(
                artifact, "prepared hypothesis",
                require_all_members_ok=False, require_safe_vote=True)
            contains = bool(failed)
            prepared_path = _resolve_bound_file(
                root, artifact.get("prepared_input_path"),
                None,
                artifact.get("prepared_input_sha256"), "prepared NPZ")
            evidence_path = _resolve_bound_file(
                root, artifact.get("evidence_path"),
                None,
                artifact.get("evidence_sha256"), "prepared evidence")
            evidence = _load_bound_json(
                evidence_path, artifact["evidence_sha256"],
                "prepared hypothesis evidence")
            if (evidence.get("schema") != PREPARED_SCHEMA
                    or evidence.get("pair_id") != pair_id
                    or evidence.get("hypothesis_index") != index
                    or evidence.get("hypothesis_sha256") != hypothesis_sha
                    or evidence.get("prepared_input_sha256")
                    != artifact.get("prepared_input_sha256")
                    or evidence.get("worker_execution_authorized") is not False
                    or evidence.get("registration_executed") is not False
                    or evidence.get("geot_result_filtering_used") is not False
                    or evidence.get(
                        "typed_failure_members_visible_and_never_filtered") is not True
                    or evidence.get("contains_typed_failure_members") is not contains
                    or evidence.get("existing_typed_failure_member_candidate_indices")
                    != list(existing)
                    or evidence.get("new_typed_failure_member_candidate_indices")
                    != list(new)
                    or evidence.get("typed_failure_member_candidate_indices")
                    != list(failed)
                    or evidence.get("safe_pose_vote_eligible") is not (not contains)
                    or evidence.get("selector_eligible") is not False):
                raise Fixed4OrchestratorContractError(
                    "prepared hypothesis evidence mismatch")
            key = str(prepared_path)
            if key in paths:
                raise Fixed4OrchestratorContractError(
                    "prepared hypothesis path collision")
            paths.add(key)
            existing_typed_hypotheses += int(bool(existing))
            typed_hypotheses += int(contains)
            checked.append({
                "hypothesis_index": index,
                "hypothesis_sha256": hypothesis_sha,
                "prepared_input_path": key,
                "prepared_input_sha256": artifact["prepared_input_sha256"],
                "contains_typed_failure_members": contains,
                "existing_typed_failure_member_candidate_indices":
                    list(existing),
                "new_typed_failure_member_candidate_indices": list(new),
                "typed_failure_member_candidate_indices": list(failed),
                "safe_pose_vote_eligible": not contains,
                "selector_eligible": False,
            })
        if (pair.get("hypotheses_with_existing_typed_failure_members")
                != sum(bool(row[
                    "existing_typed_failure_member_candidate_indices"])
                    for row in checked)
                or pair.get("hypotheses_with_new_typed_failure_members")
                != sum(bool(row[
                    "new_typed_failure_member_candidate_indices"])
                    for row in checked)
                or pair.get("hypotheses_with_typed_failure_members") != sum(
                    row["contains_typed_failure_members"] for row in checked)
                or type(pair.get("existing_typed_failure_count")) is not int
                or type(pair.get("new_typed_failure_count")) is not int
                or pair.get("typed_failure_total_count")
                    != pair["existing_typed_failure_count"]
                       + pair["new_typed_failure_count"]):
            raise Fixed4OrchestratorContractError(
                "prepared pair typed-failure count mismatch")
        existing_typed_members += pair["existing_typed_failure_count"]
        new_typed_members += pair["new_typed_failure_count"]
        result[pair_id] = checked
    if (len(paths) != 34
            or existing_typed_hypotheses
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or typed_hypotheses != EXPECTED_TYPED_FAILURE_HYPOTHESES
            or existing_typed_members != EXPECTED_EXISTING_TYPED_FAILURE_COUNT
            or new_typed_members != EXPECTED_NEW_TYPED_FAILURE_COUNT):
        raise Fixed4OrchestratorContractError(
            "prepared hypothesis/typed-failure closure mismatch")
    return result


def bind_hypotheses(
    exact: Mapping[str, Sequence[Mapping[str, Any]]],
    prepared: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[HypothesisBinding]:
    bindings: list[HypothesisBinding] = []
    for ordinal, pair_id in enumerate(FIXED_PAIR_ORDER):
        exact_rows = exact.get(pair_id)
        prepared_rows = prepared.get(pair_id)
        if (not isinstance(exact_rows, Sequence)
                or not isinstance(prepared_rows, Sequence)
                or len(exact_rows) != EXPECTED_HYPOTHESES[ordinal]
                or len(prepared_rows) != EXPECTED_HYPOTHESES[ordinal]):
            raise Fixed4OrchestratorContractError(
                "exact/prepared hypothesis distribution mismatch")
        for index, (exact_row, prepared_row) in enumerate(
                zip(exact_rows, prepared_rows)):
            if (exact_row.get("hypothesis_index") != index
                    or prepared_row.get("hypothesis_index") != index
                    or exact_row.get("hypothesis_sha256")
                    != prepared_row.get("hypothesis_sha256")):
                raise Fixed4OrchestratorContractError(
                    "exact/prepared hypothesis identity mismatch")
            existing = tuple(exact_row.get(
                "existing_typed_failure_member_candidate_indices", ()))
            new = tuple(exact_row.get(
                "new_typed_failure_member_candidate_indices", ()))
            failed = tuple(exact_row.get(
                "typed_failure_member_candidate_indices", ()))
            prepared_existing = tuple(prepared_row.get(
                "existing_typed_failure_member_candidate_indices", ()))
            prepared_new = tuple(prepared_row.get(
                "new_typed_failure_member_candidate_indices", ()))
            prepared_failed = tuple(prepared_row.get(
                "typed_failure_member_candidate_indices", ()))
            if (prepared_row.get("contains_typed_failure_members")
                    is not bool(failed)
                    or prepared_existing != existing
                    or prepared_new != new
                    or prepared_failed != failed
                    or prepared_row.get("safe_pose_vote_eligible")
                    is not (not failed)
                    or prepared_row.get("selector_eligible") is not False):
                raise Fixed4OrchestratorContractError(
                    "exact/prepared typed-failure binding mismatch")
            bindings.append(HypothesisBinding(
                pair_id=pair_id,
                pair_ordinal=ordinal,
                hypothesis_index=index,
                hypothesis_sha256=_sha(exact_row["hypothesis_sha256"],
                                       "bound hypothesis SHA"),
                prepared_input_path=str(prepared_row["prepared_input_path"]),
                prepared_input_sha256=_sha(
                    prepared_row["prepared_input_sha256"],
                    "bound prepared input SHA"),
                contains_typed_failure_members=bool(failed),
                existing_typed_failure_member_candidate_indices=existing,
                new_typed_failure_member_candidate_indices=new,
                typed_failure_member_candidate_indices=failed,
                safe_pose_vote_eligible=not failed,
                selector_eligible=False,
            ))
    if len(bindings) != 34:
        raise Fixed4OrchestratorContractError("bound closure is not exact 34")
    return bindings


def synthetic_fixture_bindings() -> list[HypothesisBinding]:
    """Deterministic metadata-only fixture; no model or solver is imported."""
    existing_typed = {(0, 0), (0, 7), (1, 1), (1, 6), (2, 0),
                      (3, 2), (3, 7), (3, 11)}
    new_typed = {(1, 2), (3, 3)}
    rows = []
    for ordinal, pair_id in enumerate(FIXED_PAIR_ORDER):
        for index in range(EXPECTED_HYPOTHESES[ordinal]):
            hypothesis_sha = stable_json_sha256({
                "fixture": True, "pair_id": pair_id,
                "hypothesis_index": index,
            })
            existing = ((index,) if (ordinal, index) in existing_typed else ())
            new = ((100 + index,) if (ordinal, index) in new_typed else ())
            failed = existing + new
            rows.append(HypothesisBinding(
                pair_id=pair_id, pair_ordinal=ordinal,
                hypothesis_index=index, hypothesis_sha256=hypothesis_sha,
                prepared_input_path=(
                    f"/synthetic/fixed4/p{ordinal}/h{index:02d}-{hypothesis_sha[:16]}.npz"),
                prepared_input_sha256=stable_json_sha256({
                    "fixture_input": hypothesis_sha}),
                contains_typed_failure_members=bool(failed),
                existing_typed_failure_member_candidate_indices=existing,
                new_typed_failure_member_candidate_indices=new,
                typed_failure_member_candidate_indices=failed,
                safe_pose_vote_eligible=not failed,
                selector_eligible=False,
            ))
    return rows


def _task_id(stage: str, binding: HypothesisBinding | None = None,
             *parts: Any) -> str:
    prefix = stage
    if binding is not None:
        prefix += (f".p{binding.pair_ordinal}.h{binding.hypothesis_index:02d}."
                   f"{binding.hypothesis_sha256[:12]}")
    if parts:
        prefix += "." + ".".join(str(value) for value in parts)
    return prefix


def build_task_dag(bindings: Sequence[HypothesisBinding],
                   preregister_sha256: str,
                   *, synthetic_fixture: bool) -> dict[str, Any]:
    _sha(preregister_sha256, "preregister SHA")
    if (len(bindings) != 34
            or [(row.pair_id, row.hypothesis_index) for row in bindings]
            != [(pair_id, index)
                for pair_id, count in zip(FIXED_PAIR_ORDER, EXPECTED_HYPOTHESES)
                for index in range(count)]
            or sum(bool(row.existing_typed_failure_member_candidate_indices)
                   for row in bindings)
            != EXPECTED_EXISTING_TYPED_FAILURE_HYPOTHESES
            or sum(row.contains_typed_failure_members for row in bindings)
            != EXPECTED_TYPED_FAILURE_HYPOTHESES
            or any(row.safe_pose_vote_eligible
                   is row.contains_typed_failure_members for row in bindings)
            or any(row.selector_eligible for row in bindings)):
        raise Fixed4OrchestratorContractError(
            "DAG input is not ordered exact 12/8/2/12")
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()

    def add(stage: str, task_id: str, upstream: Sequence[str],
            **fields: Any) -> str:
        if task_id in node_ids or any(parent not in node_ids for parent in upstream):
            raise Fixed4OrchestratorContractError(
                "DAG task collision or non-topological dependency")
        node = {
            "ordinal": len(nodes), "task_id": task_id, "stage": stage,
            "state": "planned_disabled", "execution_authorized": False,
            "execution_performed": False, "upstream_task_ids": list(upstream),
            **fields,
        }
        node["node_payload_sha256"] = stable_json_sha256(node)
        node_ids.add(task_id)
        nodes.append(node)
        return task_id

    v15_ids: dict[str, list[str]] = {pair: [] for pair in FIXED_PAIR_ORDER}
    for binding in bindings:
        common = {
            "pair_id": binding.pair_id,
            "hypothesis_index": binding.hypothesis_index,
            "hypothesis_sha256": binding.hypothesis_sha256,
            "contains_typed_failure_members":
                binding.contains_typed_failure_members,
            "existing_typed_failure_member_candidate_indices":
                list(binding.existing_typed_failure_member_candidate_indices),
            "new_typed_failure_member_candidate_indices":
                list(binding.new_typed_failure_member_candidate_indices),
            "typed_failure_member_candidate_indices":
                list(binding.typed_failure_member_candidate_indices),
            "safe_pose_vote_eligible": binding.safe_pose_vote_eligible,
            "selector_eligible": binding.selector_eligible,
            "typed_failure_policy": (
                "explicit_replay_never_filter" if
                binding.contains_typed_failure_members else "none"),
        }
        prepared_id = add(
            "prepared_input", _task_id("prepared", binding), (), **common,
            prepared_input_path=binding.prepared_input_path,
            prepared_input_sha256=binding.prepared_input_sha256,
            all_hypotheses_must_be_replayed=True,
        )
        exact3_ids = []
        for direction in DIRECTIONS:
            worker_ids = []
            for sentinel in SENTINELS:
                worker_ids.append(add(
                    "colorpcr_worker",
                    _task_id("colorpcr", binding, direction, sentinel),
                    (prepared_id,), **common, direction=direction,
                    sentinel=sentinel, device="gpu_future_only",
                    random_seed=7351, model_execution_allowed=False,
                ))
            sentinel_id = add(
                "sentinel_direction_cache",
                _task_id("sentinel_cache", binding, direction), worker_ids,
                **common, direction=direction,
                required_invariant_outputs=[
                    "ref_corr_points", "src_corr_points", "corr_scores",
                    "estimated_transform"],
                estimated_transform_discarded=True,
            )
            exact3_ids.append(add(
                "exact_three_direction_cache",
                _task_id("exact3_cache", binding, direction), (sentinel_id,),
                **common, direction=direction,
                exact_array_keys=["src_corr", "ref_corr", "scores"],
                result_selection_allowed=False,
            ))
        candidate_set_id = add(
            "v14_candidate_set", _task_id("v14_candidates", binding),
            exact3_ids, **common, max_candidates=8,
            min_input_correspondences=40, max_input_correspondences=1000,
            residual_threshold_m=0.10, diagnostic_pose_not_authoritative=True,
        )
        strict_ids = []
        for candidate_slot in range(MAX_CANDIDATES_PER_HYPOTHESIS):
            solver_ids = []
            for solver in SOLVERS:
                for direction in DIRECTIONS:
                    for repeat in REPEATS:
                        solver_ids.append(add(
                            "v13_solver_row",
                            _task_id("solver", binding, candidate_slot,
                                     solver, direction, repeat),
                            (candidate_set_id,), **common,
                            candidate_slot=candidate_slot, solver=solver,
                            direction=direction, repeat=repeat,
                            permutation_seed=(
                                0 if repeat == 0 else
                                "sha256(cache_input|direction|repeat|v13)"),
                            candidate_slot_may_be_absent=True,
                            absent_slot_policy="typed_not_generated_no_transform",
                        ))
            strict_ids.append(add(
                "v13_strict_candidate_gate",
                _task_id("strict", binding, candidate_slot), solver_ids,
                **common, candidate_slot=candidate_slot,
                repeats=5, quorum=4, rotation_max_deg=5.0,
                translation_max_m=0.10,
                icp_seed_forward=42, icp_seed_reverse=43,
                rule_b="unchanged_v13_authority",
                icp="unchanged_fixed_trace_v13_authority",
            ))
        v15_id = add(
            "v15_hypothesis_candidate_cluster", _task_id("v15", binding),
            strict_ids, **common,
            selection_rule=(
                "typed_failure_replay_permanent_not_safe_vote" if
                binding.contains_typed_failure_members else
                "one_unique_complete_linkage_safe_pose_cluster"),
            acceptance_forced_false=binding.contains_typed_failure_members,
            best_score_forbidden=True, majority_forbidden=True,
        )
        v15_ids[binding.pair_id].append(v15_id)

    pair_cluster_ids = []
    for ordinal, pair_id in enumerate(FIXED_PAIR_ORDER):
        known_bad = pair_id == KNOWN_BAD_PAIR_ID
        pair_cluster_ids.append(add(
            "v16_pair_hypothesis_cluster", f"v16_pair.p{ordinal}",
            v15_ids[pair_id], pair_id=pair_id,
            expected_hypothesis_count=EXPECTED_HYPOTHESES[ordinal],
            all_hypotheses_replayed=True,
            typed_failure_hypotheses_replayed_not_safe_voters=True,
            known_bad=known_bad,
            permanent_veto=known_bad,
            acceptance_rule=(
                "permanent_known_bad_veto" if known_bad else
                "one_unique_complete_linkage_safe_hypothesis_pose_cluster"),
            result_selection_allowed=False, best_score_forbidden=True,
            majority_forbidden=True,
        ))
    add(
        "fixed4_aggregate", "fixed4.aggregate", pair_cluster_ids,
        primary_only=True, control_can_rescue=False,
        normal_pair_rule=(
            "all_three_normals_each_require_unique_compatible_safe_pose_cluster"),
        known_bad_rule="all_12_replayed_then_permanent_veto",
        official92_allowed=False, gt_allowed=False,
        default_checkpoint_replacement_allowed=False,
    )
    stage_counts = {stage: 0 for stage in EXPECTED_STAGE_COUNTS}
    for node in nodes:
        stage_counts[node["stage"]] += 1
    if stage_counts != EXPECTED_STAGE_COUNTS or len(nodes) != EXPECTED_NODE_COUNT:
        raise Fixed4OrchestratorContractError("DAG count closure mismatch")
    dag = {
        "schema": DAG_SCHEMA, "frozen": True, "disabled": True,
        "execution_authorized": False,
        "synthetic_fixture": bool(synthetic_fixture),
        "preregister_sha256": preregister_sha256,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "pair_order": list(FIXED_PAIR_ORDER),
        "hypothesis_distribution": list(EXPECTED_HYPOTHESES),
        "hypothesis_count": 34, "typed_failure_hypothesis_count": sum(
            row.contains_typed_failure_members for row in bindings),
        "historical_typed_failure_hypothesis_count": sum(
            bool(row.existing_typed_failure_member_candidate_indices)
            for row in bindings),
        "typed_failure_hypotheses_safe_vote_count": sum(
            row.contains_typed_failure_members and row.safe_pose_vote_eligible
            for row in bindings),
        "selector_eligible_hypothesis_count": sum(
            row.selector_eligible for row in bindings),
        "stage_counts": stage_counts, "node_count": len(nodes),
        "result_selection_allowed": False,
        "all_members_ok_filter_allowed": False,
        "nodes": nodes,
    }
    dag["payload_sha256"] = stable_json_sha256(dag)
    return dag


def receipt_relative_path(node: Mapping[str, Any]) -> Path:
    ordinal = node.get("ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise Fixed4OrchestratorContractError("receipt node ordinal invalid")
    digest = stable_json_sha256(str(node.get("task_id")))[:20]
    return Path("receipts") / str(node.get("stage")) / f"{ordinal:05d}-{digest}.json"


def expected_receipt(node: Mapping[str, Any], dag: Mapping[str, Any],
                     preregister_sha256: str) -> dict[str, Any]:
    value = {
        "schema": RECEIPT_SCHEMA,
        "state": "planned_disabled",
        "execution_authorized": False,
        "execution_performed": False,
        "task_id": node.get("task_id"),
        "stage": node.get("stage"),
        "ordinal": node.get("ordinal"),
        "node_payload_sha256": node.get("node_payload_sha256"),
        "upstream_task_ids": node.get("upstream_task_ids"),
        "dag_payload_sha256": dag.get("payload_sha256"),
        "preregister_sha256": preregister_sha256,
        "output_artifacts": [],
        "typed_failure_policy": node.get("typed_failure_policy", "not_applicable"),
    }
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def write_create_only_receipt(path: Path, expected: Mapping[str, Any]) -> str:
    """Create once, or accept an exactly identical prior planning receipt."""
    path = Path(path)
    encoded = (json.dumps(expected, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        try:
            observed = json.loads(path.read_text())
        except Exception as exc:
            raise Fixed4OrchestratorContractError(
                "existing receipt is unreadable") from exc
        if observed != expected or not _payload_valid(observed):
            raise Fixed4OrchestratorContractError(
                "existing receipt differs from frozen plan")
        return "resumed_identical"
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return "created"


def materialize_planning_receipts(
    output_root: Path, dag: Mapping[str, Any], preregister_sha256: str,
) -> dict[str, Any]:
    if (dag.get("schema") != DAG_SCHEMA or dag.get("disabled") is not True
            or dag.get("execution_authorized") is not False
            or not _payload_valid(dag)
            or dag.get("node_count") != EXPECTED_NODE_COUNT
            or dag.get("stage_counts") != EXPECTED_STAGE_COUNTS):
        raise Fixed4OrchestratorContractError("DAG is not a disabled exact plan")
    output_root = Path(output_root)
    states = {"created": 0, "resumed_identical": 0}
    rows = []
    for node in dag["nodes"]:
        relative = receipt_relative_path(node)
        expected = expected_receipt(node, dag, preregister_sha256)
        state = write_create_only_receipt(output_root / relative, expected)
        states[state] += 1
        rows.append({
            "ordinal": node["ordinal"], "task_id": node["task_id"],
            "path": relative.as_posix(),
            "sha256": sha256_file(output_root / relative),
        })
    manifest = {
        "schema": RECEIPT_MANIFEST_SCHEMA,
        "disabled": True, "execution_performed": False,
        "dag_payload_sha256": dag["payload_sha256"],
        "preregister_sha256": preregister_sha256,
        "receipt_count": len(rows), "stage_counts": EXPECTED_STAGE_COUNTS,
        "states": states, "receipts": rows,
    }
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    return manifest
