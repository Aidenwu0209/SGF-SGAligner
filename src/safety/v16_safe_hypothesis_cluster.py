"""Closed V16 hypothesis-level safe-pose consensus.

This module is deliberately execution-agnostic. It consumes already sealed
V15 candidate-level decisions, never launches ColorPCR or a registration
solver, and never changes a V13/V14/V15 threshold.
"""
from __future__ import annotations

from itertools import combinations
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v13_dual_solver_runtime import (
    array_sha256,
    stable_json_sha256,
    transform_distance,
    validate_se3,
)
from safety.v15_safe_pose_cluster import (
    ROTATION_MAX_DEG,
    SCHEMA as V15_DECISION_SCHEMA,
    TRANSLATION_MAX_M,
)


SCHEMA = "v16-safe-hypothesis-pose-cluster-decision-v1"
AGGREGATE_SCHEMA = "v16-fixed4-safe-hypothesis-aggregate-v1"
EVIDENCE_SCHEMA = "v16-hypothesis-safe-pose-evidence-v1"
PRIMARY_ARM = "sgf_selected_union"
CONTROL_ARM = "fullscan"
MAX_HYPOTHESES = 12

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
EXPECTED_HYPOTHESIS_COUNTS = dict(zip(FIXED_PAIR_ORDER, (12, 8, 2, 12)))
KNOWN_BAD_PAIR_ID = FIXED_PAIR_ORDER[-1]
CANONICAL_REALIZATIONS = (
    "pointdsc/forward",
    "inverse(pointdsc/reverse)",
    "pygcransac/forward",
    "inverse(pygcransac/reverse)",
)


class SafeHypothesisClusterError(RuntimeError):
    """Malformed, incomplete, colliding, or unbound hypothesis evidence."""


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise SafeHypothesisClusterError(f"invalid {name}")
    return value


def _path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SafeHypothesisClusterError(f"invalid {name}")
    path = Path(value)
    if not path.is_absolute():
        raise SafeHypothesisClusterError(f"{name} must be absolute")
    return str(path)


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


def _distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    try:
        rotation, translation = transform_distance(left, right)
    except Exception as exc:
        raise SafeHypothesisClusterError("invalid SE3 distance input") from exc
    if not math.isfinite(rotation) or not math.isfinite(translation):
        raise SafeHypothesisClusterError("invalid SE3 distance output")
    return rotation, translation


def hypothesis_output_relative_path(
    pair_id: str, hypothesis_index: int, hypothesis_sha256: str,
) -> str:
    """Return a deterministic collision-resistant primary output path."""
    if pair_id.count("_to_") != 1 or "/" in pair_id or ".." in pair_id:
        raise SafeHypothesisClusterError("malformed pair id")
    if not isinstance(hypothesis_index, int) or hypothesis_index < 0:
        raise SafeHypothesisClusterError("invalid hypothesis index")
    hypothesis_sha256 = _sha(hypothesis_sha256, "hypothesis SHA")
    return str(PurePosixPath(
        "pairs", pair_id, PRIMARY_ARM, "hypotheses",
        f"h{hypothesis_index:02d}-{hypothesis_sha256[:16]}",
    ))


def _extract_accepted_observations(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if row.get("schema") != EVIDENCE_SCHEMA:
        raise SafeHypothesisClusterError("hypothesis evidence schema mismatch")
    pair_id = str(row.get("pair_id", ""))
    if pair_id not in EXPECTED_HYPOTHESIS_COUNTS:
        raise SafeHypothesisClusterError("pair is outside frozen fixed4")
    if row.get("arm") != PRIMARY_ARM:
        raise SafeHypothesisClusterError("hypothesis evidence is not primary")
    index = row.get("hypothesis_index")
    if not isinstance(index, int) or index < 0:
        raise SafeHypothesisClusterError("hypothesis index is malformed")
    hypothesis_sha = _sha(row.get("hypothesis_sha256"), "hypothesis SHA")
    prepared_path = _path(row.get("prepared_input_path"),
                          "prepared input path")
    prepared_sha = _sha(row.get("prepared_input_sha256"),
                        "prepared input SHA")
    expected_relative = hypothesis_output_relative_path(
        pair_id, index, hypothesis_sha)
    if row.get("output_relative_path") != expected_relative:
        raise SafeHypothesisClusterError("hypothesis output path mismatch")

    decision = row.get("candidate_decision")
    if (not isinstance(decision, Mapping)
            or decision.get("schema") != V15_DECISION_SCHEMA
            or not isinstance(decision.get("accepted"), bool)
            or decision.get("gt_consumed") is not False
            or decision.get("fallback_used") is not False):
        raise SafeHypothesisClusterError("V15 decision binding is malformed")
    identity = {
        "pair_id": pair_id,
        "hypothesis_index": index,
        "hypothesis_sha256": hypothesis_sha,
        "prepared_input_path": prepared_path,
        "prepared_input_sha256": prepared_sha,
        "output_relative_path": expected_relative,
    }
    if not decision["accepted"]:
        return identity, []
    if decision.get("reason") != "unique_safe_pose_cluster":
        raise SafeHypothesisClusterError(
            "accepted V15 decision lacks unique-cluster reason")

    candidates = decision.get("pose_realizations")
    if not isinstance(candidates, list) or not candidates:
        raise SafeHypothesisClusterError(
            "accepted V15 decision has no realizations")
    observations: list[dict[str, Any]] = []
    candidate_shas = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise SafeHypothesisClusterError("candidate realization malformed")
        candidate_sha = _sha(candidate.get("candidate_sha256"),
                             "candidate SHA")
        candidate_shas.append(candidate_sha)
        candidate_index = candidate.get("candidate_index")
        if not isinstance(candidate_index, int) or candidate_index < 0:
            raise SafeHypothesisClusterError("candidate index malformed")
        realizations = candidate.get("realizations")
        if not isinstance(realizations, list) or len(realizations) != 4:
            raise SafeHypothesisClusterError(
                "candidate must expose four canonical realizations")
        names = [value.get("realization") for value in realizations
                 if isinstance(value, Mapping)]
        if tuple(names) != CANONICAL_REALIZATIONS:
            raise SafeHypothesisClusterError(
                "canonical realization order mismatch")
        for ordinal, realization in enumerate(realizations):
            if (realization.get("ordinal") != ordinal
                    or realization.get("realization")
                    != CANONICAL_REALIZATIONS[ordinal]):
                raise SafeHypothesisClusterError(
                    "canonical realization ordinal mismatch")
            try:
                transform = validate_se3(realization.get("transform"))
            except Exception as exc:
                raise SafeHypothesisClusterError(
                    "invalid final SE3 realization") from exc
            transform_sha = _sha(realization.get("transform_sha256"),
                                 "transform SHA")
            if array_sha256(transform) != transform_sha:
                raise SafeHypothesisClusterError(
                    "final transform SHA mismatch")
            observations.append({
                "hypothesis_sha256": hypothesis_sha,
                "hypothesis_index": index,
                "candidate_sha256": candidate_sha,
                "candidate_index": candidate_index,
                "realization": CANONICAL_REALIZATIONS[ordinal],
                "ordinal": ordinal,
                "transform": transform,
                "transform_sha256": transform_sha,
            })
    if len(candidate_shas) != len(set(candidate_shas)):
        raise SafeHypothesisClusterError("duplicate candidate SHA in hypothesis")
    for left, right in combinations(observations, 2):
        if not _compatible(*_distance(left["transform"], right["transform"])):
            raise SafeHypothesisClusterError(
                "accepted V15 hypothesis is not complete-linkage safe")
    return identity, observations


def _maximal_cliques(names: Sequence[str],
                     compatible_edges: set[tuple[str, str]]) -> list[list[str]]:
    if not names or len(names) > MAX_HYPOTHESES:
        raise SafeHypothesisClusterError("hypothesis clique budget violated")
    cliques: list[tuple[str, ...]] = []
    for mask in range(1, 1 << len(names)):
        members = tuple(names[index] for index in range(len(names))
                        if mask & (1 << index))
        if all(tuple(sorted(pair)) in compatible_edges
               for pair in combinations(members, 2)):
            cliques.append(members)
    maximal = [
        clique for clique in cliques
        if not any(set(clique) < set(other) for other in cliques)
    ]
    return [list(value) for value in sorted(set(maximal))]


def _observed_medoid(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = []
    for row in observations:
        distances = [_distance(row["transform"], other["transform"])
                     for other in observations]
        normalized = [
            max(rotation / ROTATION_MAX_DEG,
                translation / TRANSLATION_MAX_M)
            for rotation, translation in distances
        ]
        scores.append((
            max(normalized),
            sum(normalized),
            row["hypothesis_sha256"],
            row["candidate_sha256"],
            row["ordinal"],
            row,
        ))
    maximum, total, _hsha, _csha, _ordinal, selected = min(scores)
    return {
        "hypothesis_sha256": selected["hypothesis_sha256"],
        "hypothesis_index": selected["hypothesis_index"],
        "candidate_sha256": selected["candidate_sha256"],
        "candidate_index": selected["candidate_index"],
        "realization": selected["realization"],
        "transform": selected["transform"].tolist(),
        "transform_sha256": selected["transform_sha256"],
        "maximum_normalized_distance": maximum,
        "sum_normalized_distance": total,
        "observed_transform_count": len(observations),
    }


def select_unique_safe_hypothesis_pose_cluster(
    evidence: Sequence[Mapping[str, Any]], *,
    expected_hypothesis_count: int,
    known_bad: bool,
) -> dict[str, Any]:
    """Accept all and only one equivalent cluster of safe hypotheses.

    Unsafe hypotheses do not vote. Every safe hypothesis participates, so a
    majority can never hide even one incompatible safe pose.
    """
    if (not isinstance(expected_hypothesis_count, int)
            or expected_hypothesis_count < 1
            or expected_hypothesis_count > MAX_HYPOTHESES):
        raise SafeHypothesisClusterError("expected hypothesis count invalid")
    if len(evidence) != expected_hypothesis_count:
        raise SafeHypothesisClusterError("hypothesis evidence count mismatch")
    parsed = [_extract_accepted_observations(row) for row in evidence]
    identities = [identity for identity, _observations in parsed]
    pair_ids = {identity["pair_id"] for identity in identities}
    if len(pair_ids) != 1:
        raise SafeHypothesisClusterError("hypotheses span multiple pairs")
    pair_id = next(iter(pair_ids))
    if EXPECTED_HYPOTHESIS_COUNTS[pair_id] != expected_hypothesis_count:
        raise SafeHypothesisClusterError("frozen pair hypothesis count mismatch")
    if sorted(identity["hypothesis_index"] for identity in identities) \
            != list(range(expected_hypothesis_count)):
        raise SafeHypothesisClusterError("hypothesis indices are not exact")
    hypothesis_shas = [identity["hypothesis_sha256"] for identity in identities]
    if len(hypothesis_shas) != len(set(hypothesis_shas)):
        raise SafeHypothesisClusterError("duplicate hypothesis SHA")
    paths = [identity["prepared_input_path"] for identity in identities]
    output_paths = [identity["output_relative_path"] for identity in identities]
    if len(paths) != len(set(paths)) or len(output_paths) != len(set(output_paths)):
        raise SafeHypothesisClusterError("hypothesis output/input path collision")

    if known_bad != (pair_id == KNOWN_BAD_PAIR_ID):
        raise SafeHypothesisClusterError("known-bad identity flag mismatch")
    if known_bad:
        safe_candidate_count = 0
        hypotheses_with_evidence = 0
        for row in evidence:
            decision = row["candidate_decision"]
            if decision.get("accepted") is True:
                raise SafeHypothesisClusterError(
                    "known-bad hypothesis bypassed candidate-level veto")
            count = decision.get(
                "strict_geometry_safe_count_before_veto", 0)
            if not isinstance(count, int) or count < 0:
                raise SafeHypothesisClusterError(
                    "known-bad pre-veto count malformed")
            safe_candidate_count += count
            hypotheses_with_evidence += int(count > 0)
        return {
            "schema": SCHEMA,
            "pair_id": pair_id,
            "arm": PRIMARY_ARM,
            "accepted": False,
            "reason": "known_bad_veto",
            "expected_hypothesis_count": expected_hypothesis_count,
            "executed_hypothesis_count": len(evidence),
            "strict_geometry_safe_candidate_count_before_veto":
                safe_candidate_count,
            "hypotheses_with_pre_veto_geometry_safe_evidence":
                hypotheses_with_evidence,
            "gt_consumed": False,
            "fallback_used": False,
        }

    safe = [
        {"identity": identity, "observations": observations}
        for identity, observations in parsed if observations
    ]
    if not safe:
        return {
            "schema": SCHEMA,
            "pair_id": pair_id,
            "arm": PRIMARY_ARM,
            "accepted": False,
            "reason": "no_safe_hypothesis",
            "expected_hypothesis_count": expected_hypothesis_count,
            "executed_hypothesis_count": len(evidence),
            "safe_hypothesis_count": 0,
            "pose_cluster_count": 0,
            "pose_clusters": [],
            "compatibility_matrix": [],
            "gt_consumed": False,
            "fallback_used": False,
        }
    safe.sort(key=lambda value: value["identity"]["hypothesis_sha256"])
    matrix = []
    edges: set[tuple[str, str]] = set()
    for left, right in combinations(safe, 2):
        distances = [
            _distance(a["transform"], b["transform"])
            for a in left["observations"] for b in right["observations"]
        ]
        row = {
            "left_hypothesis_sha256":
                left["identity"]["hypothesis_sha256"],
            "right_hypothesis_sha256":
                right["identity"]["hypothesis_sha256"],
            "cross_realization_comparisons": len(distances),
            "max_rotation_deg": max(value[0] for value in distances),
            "max_translation_m": max(value[1] for value in distances),
            "compatible": all(_compatible(*value) for value in distances),
        }
        matrix.append(row)
        if row["compatible"]:
            edges.add(tuple(sorted((
                row["left_hypothesis_sha256"],
                row["right_hypothesis_sha256"],
            ))))
    names = [value["identity"]["hypothesis_sha256"] for value in safe]
    cliques = _maximal_cliques(names, edges)
    one_cluster = len(cliques) == 1 and cliques[0] == names
    base = {
        "schema": SCHEMA,
        "pair_id": pair_id,
        "arm": PRIMARY_ARM,
        "expected_hypothesis_count": expected_hypothesis_count,
        "executed_hypothesis_count": len(evidence),
        "safe_hypothesis_count": len(safe),
        "safe_hypothesis_sha256": names,
        "pose_cluster_count": len(cliques),
        "pose_clusters": cliques,
        "compatibility_matrix": matrix,
        "gt_consumed": False,
        "fallback_used": False,
    }
    if not one_cluster:
        return {
            **base,
            "accepted": False,
            "reason": "ambiguous_multiple_safe_hypothesis_pose_clusters",
        }
    observations = [
        observation
        for hypothesis in safe for observation in hypothesis["observations"]
    ]
    medoid = _observed_medoid(observations)
    return {
        **base,
        "accepted": True,
        "reason": "unique_safe_hypothesis_pose_cluster",
        "selected_hypothesis_sha256": medoid["hypothesis_sha256"],
        "selected_hypothesis_index": medoid["hypothesis_index"],
        "selected_candidate_sha256": medoid["candidate_sha256"],
        "selected_candidate_index": medoid["candidate_index"],
        "selected_realization": medoid["realization"],
        "selected_transform": medoid["transform"],
        "selected_transform_sha256": medoid["transform_sha256"],
        "medoid_score": {
            "maximum_normalized_distance":
                medoid["maximum_normalized_distance"],
            "sum_normalized_distance": medoid["sum_normalized_distance"],
            "observed_transform_count": medoid[
                "observed_transform_count"],
        },
    }


def aggregate_fixed4_research_v16(
    primary_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate primary rows while preserving control-no-rescue semantics."""
    if [row.get("pair_id") for row in primary_rows] != list(FIXED_PAIR_ORDER):
        raise SafeHypothesisClusterError("primary fixed4 order mismatch")
    if [row.get("pair_id") for row in control_rows] != list(FIXED_PAIR_ORDER):
        raise SafeHypothesisClusterError("control fixed4 order mismatch")
    for row in primary_rows:
        if (row.get("schema") != SCHEMA or row.get("arm") != PRIMARY_ARM
                or not isinstance(row.get("accepted"), bool)):
            raise SafeHypothesisClusterError("primary row malformed")
    for row in control_rows:
        if (row.get("arm") != CONTROL_ARM
                or not isinstance(row.get("accepted"), bool)):
            raise SafeHypothesisClusterError("control row malformed")
    if primary_rows[-1]["accepted"]:
        raise SafeHypothesisClusterError("known-bad primary bypassed veto")
    if control_rows[-1]["accepted"]:
        raise SafeHypothesisClusterError("known-bad control bypassed veto")
    normal_primary = {
        row["pair_id"]: bool(row["accepted"]) for row in primary_rows[:-1]
    }
    controls = {
        row["pair_id"]: bool(row["accepted"]) for row in control_rows[:-1]
    }
    failures = [pair for pair, accepted in normal_primary.items()
                if not accepted]
    value = {
        "schema": AGGREGATE_SCHEMA,
        "safe": not failures,
        "reason": ("all_normal_primary_safe_and_known_bad_vetoed"
                   if not failures else "normal_primary_failed"),
        "primary_arm": PRIMARY_ARM,
        "control_arm": CONTROL_ARM,
        "control_can_rescue": False,
        "normal_primary_safe": normal_primary,
        "normal_primary_failures": failures,
        "control_safe_diagnostic": controls,
        "known_bad_pair_id": KNOWN_BAD_PAIR_ID,
        "known_bad_veto_by_arm": {
            PRIMARY_ARM: True,
            CONTROL_ARM: True,
        },
        "gt_consumed": False,
        "fallback_used": False,
        "official92_run": False,
    }
    return {**value, "payload_sha256": stable_json_sha256(value)}
