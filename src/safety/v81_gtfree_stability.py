"""Label-free V8.1 stability aggregation for repeated rigid registration.

The production candidate is intentionally conservative and fixed before any
selection labels are loaded:

* pool two outer runs into ten forward and ten true-reverse workers;
* require a unique complete-linkage component of at least nine per direction;
* require at least nine forward/reverse cross-final matches;
* apply the unchanged Rule-B and fixed-trace gates to the observed medoids;
* reject an isolated lucky medoid unless at least five members of each winning
  component independently pass the same safety gates;
* require all twenty single-worker jackknife replays to retain q=8 geometry,
  safe medoids, and tightly bounded medoid drift.

No GT loader, label, RRE/RTE, strict/relaxed result, or posthoc evaluator is
imported here.  The caller remains responsible for validating the immutable
worker evidence chain before invoking this pure aggregation layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from safety.v8_stage_order_consensus import (
    V8Config,
    cluster_direction,
    cross_final_agreement,
    evaluate_stage_order,
    fixed_trace_gate,
    transform_distance,
)


@dataclass(frozen=True)
class V81Config:
    outer_repeats: int = 2
    repeats_per_outer_direction: int = 5
    pooled_quorum: int = 9
    jackknife_quorum: int = 8
    max_rotation_deg: float = 5.0
    max_translation_m: float = 0.10
    min_member_safety_votes: int = 5
    max_jackknife_medoid_rotation_deg: float = 1.0
    max_jackknife_medoid_translation_m: float = 0.02

    @property
    def pooled_repeats(self) -> int:
        return self.outer_repeats * self.repeats_per_outer_direction


def _finite_transform(worker: Mapping[str, Any], *, invert: bool) -> Any:
    try:
        transform = np.asarray(worker.get("final_transform"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return None
        if invert:
            transform = np.linalg.inv(transform)
        if not np.isfinite(transform).all():
            return None
        return transform
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _validate_and_flatten(
    outer_workers: Sequence[Sequence[Mapping[str, Any]]],
    config: V81Config,
) -> tuple[list[Mapping[str, Any]], list[str]]:
    reasons: list[str] = []
    if len(outer_workers) != config.outer_repeats:
        return [], ["outer_repeat_count_mismatch"]
    flattened: list[Mapping[str, Any]] = []
    worker_identities: list[tuple[int, str, int, str]] = []
    expected_replicates = list(range(config.repeats_per_outer_direction))
    for outer_index, workers in enumerate(outer_workers):
        if not isinstance(workers, Sequence):
            reasons.append(f"outer_{outer_index:02d}_not_a_sequence")
            continue
        rows = list(workers)
        for direction in ("forward", "reverse"):
            directional = sorted(
                (row for row in rows if row.get("direction") == direction),
                key=lambda row: row.get("replicate", -1),
            )
            if [row.get("replicate") for row in directional] \
                    != expected_replicates:
                reasons.append(
                    f"outer_{outer_index:02d}_{direction}_shape_mismatch")
            flattened.extend(directional)
            worker_identities.extend((
                outer_index, direction, int(row.get("replicate", -1)),
                str(row.get("evidence_sha256", "")))
                for row in directional)
        if len(rows) != 2 * config.repeats_per_outer_direction:
            reasons.append(f"outer_{outer_index:02d}_worker_count_mismatch")
    if len(worker_identities) != 2 * config.pooled_repeats:
        reasons.append("pooled_worker_count_mismatch")
    if any(not row[3] for row in worker_identities):
        reasons.append("worker_evidence_identity_missing")
    elif len(set(worker_identities)) != len(worker_identities):
        reasons.append("worker_execution_identity_reused")
    return flattened, sorted(set(reasons))


def _records(workers: Sequence[Mapping[str, Any]],
             direction: str) -> list[dict[str, Any]]:
    rows = sorted(
        (row for row in workers if row.get("direction") == direction),
        key=lambda row: (
            int(row.get("_outer_repeat", 0)),
            int(row.get("replicate", -1)),
            str(row.get("evidence_sha256", "")),
        ),
    )
    return [{
        "status": row.get("status"),
        "transform": _finite_transform(row, invert=direction == "reverse"),
        "stable_signature": str(row.get("evidence_sha256", "")),
        "worker": row,
    } for row in rows]


def _worker_safety(
    worker: Mapping[str, Any],
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
) -> dict[str, Any]:
    violations = list(rule_b(dict(worker.get("rule_b_features", {}))))
    recorded = list(worker.get("decision", {}).get("rejection_reasons", []))
    if violations != recorded or bool(not violations) is not bool(
            worker.get("rule_b_accepted")):
        violations.append("recorded_rule_b_mismatch")
    trace = fixed_trace_gate(worker)
    reasons = list(violations)
    reasons.extend(trace["rejection_reasons"])
    return {
        "usable": not reasons,
        "rule_b_rejection_reasons": violations,
        "fixed_trace": trace,
        "rejection_reasons": reasons,
    }


def _pool_geometry(
    forward: Sequence[Mapping[str, Any]],
    reverse: Sequence[Mapping[str, Any]],
    *,
    quorum: int,
    max_rotation_deg: float,
    max_translation_m: float,
) -> dict[str, Any]:
    forward_config = V8Config(
        repeats=len(forward), quorum=quorum,
        max_rotation_deg=max_rotation_deg,
        max_translation_m=max_translation_m)
    reverse_config = V8Config(
        repeats=len(reverse), quorum=quorum,
        max_rotation_deg=max_rotation_deg,
        max_translation_m=max_translation_m)
    cross_config = V8Config(
        repeats=1, quorum=quorum,
        max_rotation_deg=max_rotation_deg,
        max_translation_m=max_translation_m)
    forward_cluster = cluster_direction(forward, forward_config)
    reverse_cluster = cluster_direction(reverse, reverse_config)
    cross = cross_final_agreement(
        forward, reverse,
        forward_cluster["winning_original_indices"],
        reverse_cluster["winning_original_indices"], cross_config)
    return {
        "usable": bool(forward_cluster["usable"]
                       and reverse_cluster["usable"] and cross["usable"]),
        "forward": forward_cluster,
        "reverse_inverted": reverse_cluster,
        "cross_final": cross,
    }


def _pool_decision(
    forward: Sequence[Mapping[str, Any]],
    reverse: Sequence[Mapping[str, Any]],
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
    *,
    quorum: int,
    config: V81Config,
    require_member_vote: bool,
) -> dict[str, Any]:
    geometry = _pool_geometry(
        forward, reverse, quorum=quorum,
        max_rotation_deg=config.max_rotation_deg,
        max_translation_m=config.max_translation_m)
    f_members = geometry["forward"]["winning_original_indices"]
    r_members = geometry["reverse_inverted"]["winning_original_indices"]
    f_index = geometry["forward"]["medoid_original_index"]
    r_index = geometry["reverse_inverted"]["medoid_original_index"]
    f_safety = [_worker_safety(forward[index]["worker"], rule_b)
                for index in f_members]
    r_safety = [_worker_safety(reverse[index]["worker"], rule_b)
                for index in r_members]
    f_votes = sum(row["usable"] for row in f_safety)
    r_votes = sum(row["usable"] for row in r_safety)
    f_medoid = (_worker_safety(forward[f_index]["worker"], rule_b)
                if f_index is not None else {
                    "usable": False, "rejection_reasons": ["missing_medoid"]})
    r_medoid = (_worker_safety(reverse[r_index]["worker"], rule_b)
                if r_index is not None else {
                    "usable": False, "rejection_reasons": ["missing_medoid"]})
    vote_ok = (not require_member_vote or (
        f_votes >= config.min_member_safety_votes
        and r_votes >= config.min_member_safety_votes))
    reasons: list[str] = []
    if not geometry["usable"]:
        reasons.append("pooled_geometry_unusable")
    if not f_medoid["usable"]:
        reasons.append("forward_medoid_safety_failed")
    if not r_medoid["usable"]:
        reasons.append("reverse_medoid_safety_failed")
    if require_member_vote and f_votes < config.min_member_safety_votes:
        reasons.append("forward_member_safety_vote_not_met")
    if require_member_vote and r_votes < config.min_member_safety_votes:
        reasons.append("reverse_member_safety_vote_not_met")
    return {
        "usable": not reasons and vote_ok,
        "rejection_reasons": reasons,
        "geometry": geometry,
        "medoid_safety": {"forward": f_medoid, "reverse": r_medoid},
        "member_safety": {
            "minimum_votes": (config.min_member_safety_votes
                              if require_member_vote else None),
            "forward_pass": f_votes,
            "forward_members": len(f_members),
            "reverse_pass": r_votes,
            "reverse_members": len(r_members),
        },
        "medoid_indices": {"forward": f_index, "reverse": r_index},
    }


def _jackknife(
    forward: Sequence[Mapping[str, Any]],
    reverse: Sequence[Mapping[str, Any]],
    full: Mapping[str, Any],
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
    config: V81Config,
) -> dict[str, Any]:
    f_full = full["medoid_indices"]["forward"]
    r_full = full["medoid_indices"]["reverse"]
    if f_full is None or r_full is None:
        return {"usable": False, "scenario_count": 0,
                "passed": 0, "scenarios": [],
                "rejection_reasons": ["full_pool_medoid_missing"]}
    f_anchor = forward[f_full]["transform"]
    r_anchor = reverse[r_full]["transform"]
    scenarios = []
    for side in ("forward", "reverse"):
        for drop in range(config.pooled_repeats):
            f_rows = [row for index, row in enumerate(forward)
                      if side != "forward" or index != drop]
            r_rows = [row for index, row in enumerate(reverse)
                      if side != "reverse" or index != drop]
            result = _pool_decision(
                f_rows, r_rows, rule_b,
                quorum=config.jackknife_quorum,
                config=config, require_member_vote=False)
            reasons = list(result["rejection_reasons"])
            f_index = result["medoid_indices"]["forward"]
            r_index = result["medoid_indices"]["reverse"]
            f_drift = r_drift = (float("inf"), float("inf"))
            if f_index is not None:
                f_drift = transform_distance(
                    f_anchor, f_rows[f_index]["transform"])
            if r_index is not None:
                r_drift = transform_distance(
                    r_anchor, r_rows[r_index]["transform"])
            for direction, drift in (("forward", f_drift),
                                     ("reverse", r_drift)):
                if drift[0] > config.max_jackknife_medoid_rotation_deg:
                    reasons.append(f"{direction}_jackknife_rotation_drift")
                if drift[1] > config.max_jackknife_medoid_translation_m:
                    reasons.append(f"{direction}_jackknife_translation_drift")
            scenarios.append({
                "dropped_direction": side,
                "dropped_index": drop,
                "usable": not reasons,
                "rejection_reasons": reasons,
                "forward_medoid_drift": {
                    "rotation_deg": f_drift[0], "translation_m": f_drift[1]},
                "reverse_medoid_drift": {
                    "rotation_deg": r_drift[0], "translation_m": r_drift[1]},
            })
    passed = sum(row["usable"] for row in scenarios)
    return {
        "usable": passed == len(scenarios),
        "scenario_count": len(scenarios),
        "passed": passed,
        "scenarios": scenarios,
        "rejection_reasons": ([] if passed == len(scenarios)
                              else ["single_worker_jackknife_failed"]),
    }


def evaluate_v81_stability(
    outer_workers: Sequence[Sequence[Mapping[str, Any]]],
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
    config: V81Config = V81Config(),
) -> dict[str, Any]:
    """Evaluate the single pre-registered V8.1 GT-free aggregation policy."""
    flattened, shape_reasons = _validate_and_flatten(outer_workers, config)
    tagged: list[dict[str, Any]] = []
    for outer_index, workers in enumerate(outer_workers):
        for worker in workers:
            row = dict(worker)
            row["_outer_repeat"] = outer_index
            tagged.append(row)
    forward = _records(tagged, "forward")
    reverse = _records(tagged, "reverse")
    if shape_reasons:
        return {
            "usable_for_reconstruction": False,
            "rejection_reasons": shape_reasons,
            "pool": None, "jackknife": None,
            "selected_observed_forward_medoid": None,
            "config": config.__dict__,
        }
    full = _pool_decision(
        forward, reverse, rule_b, quorum=config.pooled_quorum,
        config=config, require_member_vote=True)
    jackknife = _jackknife(forward, reverse, full, rule_b, config)
    reasons = list(full["rejection_reasons"])
    reasons.extend(jackknife["rejection_reasons"])
    selected = None
    f_index = full["medoid_indices"]["forward"]
    if f_index is not None:
        worker = forward[f_index]["worker"]
        selected = {
            "outer_repeat": int(worker["_outer_repeat"]),
            "replicate": int(worker["replicate"]),
            "raw_transform": worker["raw_transform"],
            "final_transform": worker["final_transform"],
            "worker_evidence_sha256": worker["evidence_sha256"],
        }
    return {
        "usable_for_reconstruction": not reasons,
        "rejection_reasons": reasons,
        "config": {
            **config.__dict__, "pooled_repeats": config.pooled_repeats},
        "stage_order": [
            "pool_two_outers", "directional_complete_linkage_q9",
            "cross_final_q9", "observed_medoid_rule_b_and_fixed_trace",
            "cluster_member_safety_vote_5", "twenty_way_jackknife_q8",
            "bounded_jackknife_medoid_drift"],
        "pool": full,
        "jackknife": jackknife,
        "selected_observed_forward_medoid": selected,
    }


def compare_gtfree_policies(
    outer_workers: Sequence[Sequence[Mapping[str, Any]]],
    rule_b: Callable[[Mapping[str, Any]], Sequence[str]],
    config: V81Config = V81Config(),
) -> dict[str, Any]:
    """Return label-free diagnostics; never use this to choose posthoc."""
    tagged = []
    outer_v8 = []
    for outer_index, workers in enumerate(outer_workers):
        outer_v8.append(evaluate_stage_order(
            workers, V8Config(), rule_b, require_fixed_trace=True)[
                "usable_for_reconstruction"])
        for worker in workers:
            row = dict(worker)
            row["_outer_repeat"] = outer_index
            tagged.append(row)
    forward = _records(tagged, "forward")
    reverse = _records(tagged, "reverse")
    rows = {"outer_v8": outer_v8, "dual_outer_unanimity": all(outer_v8)}
    for quorum in (8, 9):
        medoid = _pool_decision(
            forward, reverse, rule_b, quorum=quorum,
            config=config, require_member_vote=False)
        vote = _pool_decision(
            forward, reverse, rule_b, quorum=quorum,
            config=config, require_member_vote=True)
        rows[f"pooled_q{quorum}_medoid"] = medoid["usable"]
        rows[f"pooled_q{quorum}_member_vote"] = vote["usable"]
    rows["v81_recommended"] = evaluate_v81_stability(
        outer_workers, rule_b, config)["usable_for_reconstruction"]
    return rows
