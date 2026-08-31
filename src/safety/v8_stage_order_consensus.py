"""V8 GT-free stage-order consensus over repeated rigid transforms.

Unlike V7, geometry is clustered before the unchanged Rule-B decision is
applied to the observed medoid.  Raw transforms are deliberately diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class V8Config:
    repeats: int = 5
    quorum: int = 4
    max_rotation_deg: float = 5.0
    max_translation_m: float = 0.10


def transform_distance(a: Any, b: Any) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("transforms must be 4x4")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("transforms must be finite")
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    return rotation, translation


def _compatible(a: Mapping[str, Any], b: Mapping[str, Any],
                config: V8Config) -> bool:
    dr, dt = transform_distance(a["transform"], b["transform"])
    return dr <= config.max_rotation_deg and dt <= config.max_translation_m


def maximal_cliques(records: Sequence[Mapping[str, Any]],
                    config: V8Config) -> list[tuple[int, ...]]:
    """Enumerate maximal complete-linkage sets (K is exactly five)."""
    indices = tuple(range(len(records)))
    candidates: list[tuple[int, ...]] = []
    for size in range(len(indices), 0, -1):
        for subset in combinations(indices, size):
            if all(_compatible(records[i], records[j], config)
                   for i, j in combinations(subset, 2)):
                candidates.append(subset)
    maximal = [
        clique for clique in candidates
        if not any(set(clique) < set(other) for other in candidates)
    ]
    return sorted(set(maximal), key=lambda row: (-len(row), row))


def _medoid(records: Sequence[Mapping[str, Any]],
            clique: tuple[int, ...], config: V8Config) -> int:
    def score(index: int) -> tuple[float, str, int]:
        total = 0.0
        for other in clique:
            dr, dt = transform_distance(
                records[index]["transform"], records[other]["transform"])
            total += dr / config.max_rotation_deg
            total += dt / config.max_translation_m
        return (total, str(records[index].get("stable_signature", "")), index)
    return min(clique, key=score)


def cluster_direction(records: Sequence[Mapping[str, Any]],
                      config: V8Config) -> dict[str, Any]:
    """Cluster every finite final transform before any Rule-B filtering."""
    reasons: list[str] = []
    if len(records) != config.repeats:
        reasons.append("repeat_count_mismatch")
    valid: list[dict[str, Any]] = []
    for original_index, row in enumerate(records):
        try:
            transform = np.asarray(row.get("transform"), dtype=np.float64)
            finite = transform.shape == (4, 4) and np.isfinite(transform).all()
        except (TypeError, ValueError):
            finite = False
        if row.get("status") != "ok" or not finite:
            continue
        item = dict(row)
        item["transform"] = transform
        item["_original_index"] = original_index
        valid.append(item)
    if len(valid) != len(records):
        reasons.append("invalid_run_present")
    cliques = maximal_cliques(valid, config) if valid else []
    largest = len(cliques[0]) if cliques else 0
    winners = [clique for clique in cliques if len(clique) == largest]
    if largest < config.quorum:
        reasons.append("consensus_quorum_not_met")
    if len(winners) != 1:
        reasons.append("largest_clique_not_unique")
    winning = winners[0] if len(winners) == 1 else ()
    medoid = _medoid(valid, winning, config) if winning else None
    return {
        "usable": not reasons,
        "rejection_reasons": reasons,
        "requested": len(records),
        "valid": len(valid),
        "clique_sizes": [len(row) for row in cliques],
        "winning_original_indices": [
            valid[index]["_original_index"] for index in winning],
        "medoid_original_index": (
            valid[medoid]["_original_index"] if medoid is not None else None),
    }


def cross_final_agreement(forward: Sequence[Mapping[str, Any]],
                          reverse_inverted: Sequence[Mapping[str, Any]],
                          forward_members: Sequence[int],
                          reverse_members: Sequence[int],
                          config: V8Config) -> dict[str, Any]:
    """Maximum bipartite agreement between the two winning final cliques."""
    edges: dict[int, list[tuple[int, float, float]]] = {}
    for left in sorted(forward_members):
        for right in sorted(reverse_members):
            try:
                dr, dt = transform_distance(
                    forward[left]["transform"],
                    reverse_inverted[right]["transform"])
            except ValueError:
                continue
            if dr <= config.max_rotation_deg and dt <= config.max_translation_m:
                edges.setdefault(left, []).append((right, dr, dt))
    for values in edges.values():
        values.sort(key=lambda row: (row[1], row[2], row[0]))
    matched: dict[int, tuple[int, float, float]] = {}

    def augment(left: int, seen: set[int]) -> bool:
        for right, dr, dt in edges.get(left, []):
            if right in seen:
                continue
            seen.add(right)
            previous = matched.get(right)
            if previous is None or augment(previous[0], seen):
                matched[right] = (left, dr, dt)
                return True
        return False

    for left in sorted(edges):
        augment(left, set())
    selected = sorted(
        ((left, right, dr, dt)
         for right, (left, dr, dt) in matched.items()),
        key=lambda row: (row[0], row[1]))
    return {
        "usable": len(selected) >= config.quorum,
        "agreement_count": len(selected),
        "matches": [
            {"forward": left, "reverse": right,
             "rotation_deg": dr, "translation_m": dt}
            for left, right, dr, dt in selected],
        "rejection_reasons": (
            [] if len(selected) >= config.quorum
            else ["cross_final_quorum_not_met"]),
    }


def fixed_trace_gate(worker: Mapping[str, Any]) -> dict[str, Any]:
    """Require fixed-correspondence ICP evidence for a fresh V8 run.

    Legacy V7 evidence lacks these fields and is explicitly non-qualifying.
    """
    trace = worker.get("icp", {}).get("trace", [])
    available = bool(trace) and all(
        "fixed_correspondence_rmse_before_m" in step
        and "fixed_correspondence_rmse_after_m" in step
        for step in trace)
    non_increasing = available and all(
        np.isfinite(step["fixed_correspondence_rmse_before_m"])
        and np.isfinite(step["fixed_correspondence_rmse_after_m"])
        and step["fixed_correspondence_rmse_after_m"]
        <= step["fixed_correspondence_rmse_before_m"] + 1e-12
        for step in trace)
    last = trace[-1] if trace else {}
    stable = bool(last) and last.get("update_rotation_deg", float("inf")) <= .25 \
        and last.get("update_translation_m", float("inf")) <= .005
    reasons = []
    if not available:
        reasons.append("fixed_correspondence_trace_missing")
    elif not non_increasing:
        reasons.append("fixed_correspondence_rmse_increased")
    if not stable:
        reasons.append("icp_last_update_too_large")
    return {
        "available": available,
        "usable": not reasons,
        "fixed_rmse_non_increasing": non_increasing,
        "last_update_stable": stable,
        "rejection_reasons": reasons,
    }


def evaluate_stage_order(
    workers: Sequence[Mapping[str, Any]],
    config: V8Config,
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
    *,
    require_fixed_trace: bool = True,
) -> dict[str, Any]:
    """Run the frozen V8 ordering without GT or labels."""
    forward = sorted(
        (row for row in workers if row.get("direction") == "forward"),
        key=lambda row: row.get("replicate"))
    reverse = sorted(
        (row for row in workers if row.get("direction") == "reverse"),
        key=lambda row: row.get("replicate"))
    if len(forward) != config.repeats or len(reverse) != config.repeats:
        raise ValueError("V8 requires exactly five forward and five reverse workers")

    def records(rows: Sequence[Mapping[str, Any]], field: str,
                invert: bool = False) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            transform = np.asarray(row[field], dtype=np.float64)
            if invert:
                transform = np.linalg.inv(transform)
            output.append({
                "status": row.get("status"),
                "transform": transform,
                "stable_signature": row.get("permutation_provenance_sha256"),
            })
        return output

    f_final = records(forward, "final_transform")
    r_final = records(reverse, "final_transform", invert=True)
    f_cluster = cluster_direction(f_final, config)
    r_cluster = cluster_direction(r_final, config)
    f_index = f_cluster["medoid_original_index"]
    r_index = r_cluster["medoid_original_index"]

    def decision(row: Mapping[str, Any] | None) -> dict[str, Any]:
        if row is None:
            return {"usable": False, "rejection_reasons": ["missing_medoid"]}
        violations = list(rule_b(dict(row.get("rule_b_features", {}))))
        recorded = list(row.get("decision", {}).get("rejection_reasons", []))
        if violations != recorded or bool(not violations) is not bool(
                row.get("rule_b_accepted")):
            violations.append("recorded_rule_b_mismatch")
        return {"usable": not violations, "rejection_reasons": violations}

    f_rule = decision(forward[f_index] if f_index is not None else None)
    r_rule = decision(reverse[r_index] if r_index is not None else None)
    f_trace = fixed_trace_gate(forward[f_index]) if f_index is not None else {
        "available": False, "usable": False,
        "rejection_reasons": ["missing_forward_medoid"]}
    r_trace = fixed_trace_gate(reverse[r_index]) if r_index is not None else {
        "available": False, "usable": False,
        "rejection_reasons": ["missing_reverse_medoid"]}
    cross = cross_final_agreement(
        f_final, r_final, f_cluster["winning_original_indices"],
        r_cluster["winning_original_indices"], config)
    raw_diagnostic = {
        "forward": cluster_direction(records(forward, "raw_transform"), config),
        "reverse_inverted": cluster_direction(
            records(reverse, "raw_transform", invert=True), config),
    }
    trace_ok = all(row.get("usable") for row in (f_trace, r_trace))
    usable = all((f_cluster["usable"], r_cluster["usable"],
                  f_rule["usable"], r_rule["usable"], cross["usable"]))
    if require_fixed_trace:
        usable = usable and trace_ok
    selected = None
    if f_index is not None:
        selected = {
            "forward_replicate": int(forward[f_index]["replicate"]),
            "raw_transform": forward[f_index]["raw_transform"],
            "final_transform": forward[f_index]["final_transform"],
            "worker_evidence_sha256": forward[f_index]["evidence_sha256"],
        }
    return {
        "config": {
            "repeats": config.repeats, "quorum": config.quorum,
            "max_rotation_deg": config.max_rotation_deg,
            "max_translation_m": config.max_translation_m,
        },
        "stage_order": [
            "all_finite_final_transforms", "directional_complete_linkage",
            "observed_medoid", "unchanged_rule_b", "cross_final_consensus",
            "fixed_correspondence_trace_if_fresh"],
        "usable_for_reconstruction": bool(usable),
        "fresh_v8_qualified": bool(usable and trace_ok),
        "directional_final_consensus": {
            "forward": f_cluster, "reverse_inverted": r_cluster},
        "medoid_rule_b": {"forward": f_rule, "reverse": r_rule},
        "medoid_fixed_trace": {"forward": f_trace, "reverse": r_trace},
        "cross_final": cross,
        "raw_consensus_diagnostic_only": raw_diagnostic,
        "selected_observed_forward_medoid": selected,
    }
