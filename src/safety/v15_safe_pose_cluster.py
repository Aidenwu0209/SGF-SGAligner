"""V15 deterministic complete-linkage selection over V13-strict final poses.

This module never reruns a solver and never weakens a V13/V14 gate.  It only
corrects the V14 category error that treated multiple independently supported
candidates for one equivalent final pose as multiple poses.
"""
from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v13_dual_solver_runtime import (
    array_sha256, transform_distance, validate_se3,
)
from safety.v14_rigid_multihypothesis import (
    CANDIDATE_SCHEMA, STRICT_AUTHORITY, STRICT_SCHEMA,
)


SCHEMA = "v15-safe-pose-cluster-decision-v1"
AGGREGATE_SCHEMA = "v15-fixed4-safe-pose-cluster-aggregate-v1"
ROTATION_MAX_DEG = 5.0
TRANSLATION_MAX_M = 0.10
REALIZATIONS = (
    ("pointdsc/forward", False),
    ("pointdsc/reverse", True),
    ("pygcransac/forward", False),
    ("pygcransac/reverse", True),
)


class SafePoseClusterError(RuntimeError):
    """Malformed, unbound, or internally inconsistent evidence."""


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise SafePoseClusterError(f"invalid {name}")
    return value


def _path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SafePoseClusterError(f"invalid {name}")
    return value


def _bound_evidence(
    contract: Mapping[str, Any], strict: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate identity binding; unlike V14, malformed evidence never skips."""
    candidate = contract.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("schema") != CANDIDATE_SCHEMA:
        raise SafePoseClusterError("candidate binding schema mismatch")
    if strict.get("schema") != STRICT_SCHEMA:
        raise SafePoseClusterError("strict binding schema mismatch")
    if strict.get("gate_authority") != STRICT_AUTHORITY:
        raise SafePoseClusterError("strict binding authority mismatch")
    if not isinstance(strict.get("safe"), bool) \
            or not isinstance(strict.get("strict_geometry_safe_before_veto"), bool):
        raise SafePoseClusterError("strict binding safety flags are malformed")
    if strict.get("gt_consumed") is not False \
            or strict.get("fallback_used") is not False:
        raise SafePoseClusterError("GT/fallback evidence is forbidden")

    index = contract.get("candidate_index")
    if not isinstance(index, int) or index < 0:
        raise SafePoseClusterError("candidate binding index mismatch")
    candidate_sha = _sha(candidate.get("candidate_sha256"), "candidate SHA")
    candidate_set_sha = _sha(contract.get("candidate_set_sha256"),
                             "candidate set SHA")
    candidate_set_path = _path(contract.get("candidate_set_path"),
                               "candidate set path")
    pair_id = _path(candidate.get("pair_id"), "pair id")
    arm = _path(candidate.get("arm"), "arm")
    cache_sha = {
        direction: _sha(candidate.get(
            f"{direction}_candidate_cache_sha256"),
            f"{direction} cache SHA")
        for direction in ("forward", "reverse")}
    receipt_sha = {
        direction: _sha(candidate.get(
            f"{direction}_candidate_receipt_sha256"),
            f"{direction} receipt SHA")
        for direction in ("forward", "reverse")}
    cache_path = {
        direction: _path(candidate.get(
            f"{direction}_candidate_cache_path"),
            f"{direction} cache path")
        for direction in ("forward", "reverse")}
    receipt_path = {
        direction: _path(candidate.get(
            f"{direction}_candidate_receipt_path"),
            f"{direction} receipt path")
        for direction in ("forward", "reverse")}
    expected = {
        "candidate_sha256": candidate_sha,
        "candidate_index": index,
        "candidate_set_path": candidate_set_path,
        "candidate_set_sha256": candidate_set_sha,
        "pair_id": pair_id,
        "arm": arm,
        "cache_sha256": cache_sha,
        "candidate_cache_path": cache_path,
        "candidate_receipt_sha256": receipt_sha,
        "candidate_receipt_path": receipt_path,
    }
    mismatch = [key for key, value in expected.items()
                if strict.get(key) != value]
    if mismatch:
        raise SafePoseClusterError(
            f"candidate/strict binding mismatch: {sorted(mismatch)}")
    return dict(candidate), dict(strict)


def _canonical_realizations(strict: Mapping[str, Any]) -> list[dict[str, Any]]:
    medoids = strict.get("medoid_safety")
    if not isinstance(medoids, Mapping) or set(medoids) != {
            name for name, _invert in REALIZATIONS}:
        raise SafePoseClusterError("strict-safe realization set is incomplete")
    values = []
    for ordinal, (name, invert) in enumerate(REALIZATIONS):
        row = medoids.get(name)
        if not isinstance(row, Mapping) or row.get("final_transform") is None:
            raise SafePoseClusterError("strict-safe realization is missing")
        try:
            transform = validate_se3(row["final_transform"])
            if invert:
                transform = validate_se3(np.linalg.inv(transform))
        except Exception as exc:
            raise SafePoseClusterError("invalid SE3 realization") from exc
        values.append({
            "realization": (name if not invert else f"inverse({name})"),
            "ordinal": ordinal,
            "transform": transform,
            "transform_sha256": array_sha256(transform),
        })
    distances = [_distance(a["transform"], b["transform"])
                 for a, b in combinations(values, 2)]
    if distances and not all(_compatible(*distance) for distance in distances):
        raise SafePoseClusterError("strict-safe realization spread exceeds V14 thresholds")
    return values


def _distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        rotation, translation = transform_distance(a, b)
    except Exception as exc:
        raise SafePoseClusterError("invalid SE3 distance input") from exc
    if not math.isfinite(rotation) or not math.isfinite(translation):
        raise SafePoseClusterError("invalid SE3 distance output")
    return rotation, translation


def _compatible(rotation_deg: float, translation_m: float) -> bool:
    # isclose only absorbs transcendental roundoff at the exact frozen boundary.
    rotation_ok = (rotation_deg <= ROTATION_MAX_DEG
                   or math.isclose(rotation_deg, ROTATION_MAX_DEG,
                                   rel_tol=0.0, abs_tol=1e-9))
    translation_ok = (translation_m <= TRANSLATION_MAX_M
                      or math.isclose(translation_m, TRANSLATION_MAX_M,
                                      rel_tol=0.0, abs_tol=1e-12))
    return rotation_ok and translation_ok


def _pair_compatibility(left: Mapping[str, Any],
                        right: Mapping[str, Any]) -> dict[str, Any]:
    distances = [_distance(a["transform"], b["transform"])
                 for a in left["realizations"]
                 for b in right["realizations"]]
    max_rotation = max(value[0] for value in distances)
    max_translation = max(value[1] for value in distances)
    return {
        "left_candidate_sha256": left["candidate_sha256"],
        "right_candidate_sha256": right["candidate_sha256"],
        "cross_realization_comparisons": len(distances),
        "max_rotation_deg": max_rotation,
        "max_translation_m": max_translation,
        "compatible": all(_compatible(*value) for value in distances),
    }


def _maximal_cliques(candidates: Sequence[Mapping[str, Any]],
                     matrix: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """Exact order-independent maximal-clique enumeration; n is frozen <= 8."""
    names = [str(value["candidate_sha256"]) for value in candidates]
    if len(names) > 8:
        raise SafePoseClusterError("safe candidate count exceeds frozen V14 budget")
    edge = {tuple(sorted((row["left_candidate_sha256"],
                          row["right_candidate_sha256"])))
            for row in matrix if row["compatible"]}
    cliques = []
    for mask in range(1, 1 << len(names)):
        members = [names[index] for index in range(len(names))
                   if mask & (1 << index)]
        if all(tuple(sorted(pair)) in edge for pair in combinations(members, 2)):
            cliques.append(tuple(members))
    maximal = [clique for clique in cliques
               if not any(set(clique) < set(other) for other in cliques)]
    return [list(value) for value in sorted(set(maximal))]


def _observed_medoid(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations = []
    for candidate in candidates:
        for realization in candidate["realizations"]:
            observations.append({
                **realization,
                "candidate_sha256": candidate["candidate_sha256"],
                "candidate_index": candidate["candidate_index"],
            })
    scores = []
    for row in observations:
        distances = [_distance(row["transform"], other["transform"])
                     for other in observations]
        normalized = [max(rotation / ROTATION_MAX_DEG,
                          translation / TRANSLATION_MAX_M)
                      for rotation, translation in distances]
        scores.append((max(normalized), sum(normalized),
                       row["candidate_sha256"], row["ordinal"], row))
    maximum, total, _sha_value, _ordinal, selected = min(scores)
    return {
        "candidate_sha256": selected["candidate_sha256"],
        "candidate_index": selected["candidate_index"],
        "realization": selected["realization"],
        "transform": selected["transform"].tolist(),
        "transform_sha256": selected["transform_sha256"],
        "maximum_normalized_distance": maximum,
        "sum_normalized_distance": total,
        "observed_transform_count": len(observations),
    }


def select_unique_safe_pose_cluster(
    evidence: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], *,
    known_bad: bool,
) -> dict[str, Any]:
    """Accept one complete-linkage cluster, never merely one raw candidate."""
    bound = [_bound_evidence(contract, strict)
             for contract, strict in evidence]
    candidate_shas = [candidate["candidate_sha256"]
                      for candidate, _strict in bound]
    if len(candidate_shas) != len(set(candidate_shas)):
        raise SafePoseClusterError("duplicate candidate SHA in one row")

    if known_bad:
        pre_veto = []
        for candidate, strict in bound:
            if strict["strict_geometry_safe_before_veto"]:
                realizations = _canonical_realizations(strict)
                pre_veto.append({
                    "candidate_sha256": candidate["candidate_sha256"],
                    "candidate_index": strict["candidate_index"],
                    "realization_sha256": [row["transform_sha256"]
                                           for row in realizations],
                })
        return {
            "schema": SCHEMA, "accepted": False, "reason": "known_bad_veto",
            "strict_geometry_safe_count_before_veto": len(pre_veto),
            "strict_geometry_safe_candidates_before_veto": sorted(
                pre_veto, key=lambda value: value["candidate_sha256"]),
            "gt_consumed": False, "fallback_used": False,
        }

    candidates = []
    for candidate, strict in bound:
        if not strict["safe"]:
            continue
        if not strict["strict_geometry_safe_before_veto"]:
            raise SafePoseClusterError("safe candidate lacks pre-veto geometry safety")
        candidates.append({
            "candidate_sha256": candidate["candidate_sha256"],
            "candidate_index": strict["candidate_index"],
            "realizations": _canonical_realizations(strict),
        })
    candidates.sort(key=lambda value: value["candidate_sha256"])
    if not candidates:
        return {
            "schema": SCHEMA, "accepted": False,
            "reason": "no_safe_candidate", "safe_candidate_count": 0,
            "pose_cluster_count": 0, "pose_clusters": [],
            "compatibility_matrix": [], "pose_realizations": [],
            "gt_consumed": False, "fallback_used": False,
        }

    matrix = [_pair_compatibility(left, right)
              for left, right in combinations(candidates, 2)]
    cliques = _maximal_cliques(candidates, matrix)
    realizations = [{
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_index": candidate["candidate_index"],
        "realizations": [{
            "realization": row["realization"],
            "ordinal": row["ordinal"],
            "transform": row["transform"].tolist(),
            "transform_sha256": row["transform_sha256"],
        } for row in candidate["realizations"]],
    } for candidate in candidates]
    all_names = [value["candidate_sha256"] for value in candidates]
    one_cluster = len(cliques) == 1 and cliques[0] == all_names
    base = {
        "schema": SCHEMA,
        "safe_candidate_count": len(candidates),
        "pose_cluster_count": len(cliques),
        "pose_clusters": cliques,
        "compatibility_matrix": matrix,
        "pose_realizations": realizations,
        "gt_consumed": False, "fallback_used": False,
    }
    if not one_cluster:
        return {
            **base, "accepted": False,
            "reason": "ambiguous_multiple_safe_pose_clusters",
        }
    medoid = _observed_medoid(candidates)
    return {
        **base, "accepted": True, "reason": "unique_safe_pose_cluster",
        "pose_cluster_member_sha256": all_names,
        "selected_candidate_sha256": medoid["candidate_sha256"],
        "selected_candidate_index": medoid["candidate_index"],
        "selected_realization": medoid["realization"],
        "selected_transform": medoid["transform"],
        "selected_transform_sha256": medoid["transform_sha256"],
        "medoid_score": {
            "maximum_normalized_distance": medoid[
                "maximum_normalized_distance"],
            "sum_normalized_distance": medoid["sum_normalized_distance"],
            "observed_transform_count": medoid["observed_transform_count"],
        },
    }


def aggregate_fixed4_research_v15(
    rows: Sequence[Mapping[str, Any]], preregister: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve V14 primary-only fixed4 semantics under the corrected selector."""
    if (preregister.get("schema") != "v15-safe-pose-cluster-preregister-v1"
            or preregister.get("selection_rule") !=
            "exactly_one_complete_linkage_strict_safe_pose_cluster_else_reject"):
        raise SafePoseClusterError("V15 aggregate preregistration mismatch")
    pair_order = [str(value) for value in preregister.get("fixed_pair_order", ())]
    primary = str(preregister.get("primary_arm", ""))
    control = str(preregister.get("control_arm", ""))
    known_bad = str(preregister.get("known_bad_pair_id", ""))
    expected = [(pair_id, arm) for pair_id in pair_order
                for arm in (primary, control)]
    actual = [(str(row.get("pair_id")), str(row.get("arm"))) for row in rows]
    if (len(pair_order) != 4 or len(set(pair_order)) != 4
            or known_bad != pair_order[-1] or actual != expected):
        raise SafePoseClusterError("V15 aggregate requires exact ordered 4x2 rows")
    by_key = {(row["pair_id"], row["arm"]): row["decision"] for row in rows}
    normal = pair_order[:-1]
    primary_safe = {pair_id: bool(by_key[(pair_id, primary)].get("accepted"))
                    for pair_id in normal}
    control_safe = {pair_id: bool(by_key[(pair_id, control)].get("accepted"))
                    for pair_id in normal}
    veto = {arm: by_key[(known_bad, arm)].get("reason") == "known_bad_veto"
            for arm in (primary, control)}
    ambiguous = [{"pair_id": pair_id, "arm": arm}
                 for pair_id, arm in expected
                 if by_key[(pair_id, arm)].get("reason") ==
                 "ambiguous_multiple_safe_pose_clusters"]
    safe = all(primary_safe.values()) and all(veto.values()) and not ambiguous
    return {
        "schema": AGGREGATE_SCHEMA,
        "safe": safe,
        "reason": ("fixed4_research_gate_pass" if safe else
                   "ambiguous_safe_pose_clusters" if ambiguous else
                   "known_bad_veto_failed" if not all(veto.values()) else
                   "normal_primary_failed"),
        "primary_arm": primary, "control_arm": control,
        "control_can_rescue": False,
        "normal_primary_safe": primary_safe,
        "normal_primary_failures": [pair_id for pair_id in normal
                                    if not primary_safe[pair_id]],
        "control_safe_diagnostic": control_safe,
        "known_bad_pair_id": known_bad,
        "known_bad_veto_by_arm": veto,
        "ambiguous_rows": ambiguous,
        "replay_only": True, "independent_gate": False,
        "gt_consumed": False, "official92_run": False,
    }
