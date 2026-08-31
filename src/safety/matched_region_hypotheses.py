"""Deterministic GT-free matched-region hypotheses for official SGAligner.

This module is adapter-only.  It groups frozen mutual cross-graph node
candidates into one-to-one, locally connected, approximately rigid object
combinations.  It then exposes canonical surface unions for one independent
official GeoTransformer call in each direction.  It deliberately does not
import labels, GT transforms, posthoc evaluation, Rule-B, or checkpoint code.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MatchedRegionConfig:
    local_neighbors: int = 4
    absolute_distance_tolerance_m: float = 0.10
    relative_distance_tolerance: float = 0.20
    min_members: int = 3
    max_members: int = 6
    beam_width: int = 64
    max_hypotheses: int = 12


class MatchedRegionError(RuntimeError):
    """The GT-free candidate/surface contract is malformed."""


def _surface(value: Any, index: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0
            or not np.isfinite(points).all()):
        raise MatchedRegionError(f"surface {index} is not finite Nx3")
    return np.ascontiguousarray(points)


def _candidate_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        int(row.get("worst_cross_rank", 1)),
        int(row.get("rank_sum", 2)),
        int(row["source_index"]),
        int(row["reference_index"]),
    )


def _canonical_candidates(
    candidates: Sequence[Mapping[str, Any]], src_count: int,
) -> list[dict[str, int]]:
    rows: dict[tuple[int, int], dict[str, int]] = {}
    for raw in candidates:
        source = int(raw["source_index"])
        reference = int(raw["reference_index"])
        if not (0 <= source < src_count <= reference):
            raise MatchedRegionError("candidate is not source-to-reference")
        row = {
            "source_index": source,
            "reference_index": reference,
            "forward_cross_rank": int(raw.get("forward_cross_rank", 1)),
            "reverse_cross_rank": int(raw.get("reverse_cross_rank", 1)),
            "worst_cross_rank": int(raw.get("worst_cross_rank", 1)),
            "rank_sum": int(raw.get("rank_sum", 2)),
        }
        key = (source, reference)
        if key in rows and rows[key] != row:
            raise MatchedRegionError("duplicate candidate has conflicting ranks")
        rows[key] = row
    return sorted(rows.values(), key=_candidate_key)


def _local_edges(
    surfaces: Mapping[int, Any], src_count: int,
    explicit_edges: Sequence[Sequence[int]], neighbors: int,
) -> tuple[set[tuple[int, int]], dict[int, np.ndarray]]:
    if neighbors < 1:
        raise MatchedRegionError("local_neighbors must be positive")
    arrays = {int(index): _surface(points, int(index))
              for index, points in surfaces.items()}
    if not arrays:
        raise MatchedRegionError("no registration surfaces")
    centroids = {index: value.astype(np.float64).mean(axis=0)
                 for index, value in arrays.items()}
    edges: set[tuple[int, int]] = set()
    for raw in explicit_edges:
        if len(raw) != 2:
            raise MatchedRegionError("explicit edge must have two endpoints")
        left, right = (int(raw[0]), int(raw[1]))
        if left == right or left not in arrays or right not in arrays:
            continue
        if (left < src_count) != (right < src_count):
            continue
        edges.add(tuple(sorted((left, right))))
    for side in (sorted(i for i in arrays if i < src_count),
                 sorted(i for i in arrays if i >= src_count)):
        for index in side:
            ranked = sorted(
                (float(np.linalg.norm(centroids[index] - centroids[other])),
                 other)
                for other in side if other != index)
            for _distance, other in ranked[:neighbors]:
                edges.add(tuple(sorted((index, other))))
    return edges, centroids


def _pair_compatible(
    left: Mapping[str, int], right: Mapping[str, int],
    edges: set[tuple[int, int]], centroids: Mapping[int, np.ndarray],
    config: MatchedRegionConfig,
) -> tuple[bool, bool]:
    src_left, ref_left = left["source_index"], left["reference_index"]
    src_right, ref_right = right["source_index"], right["reference_index"]
    if src_left == src_right or ref_left == ref_right:
        return False, False
    src_distance = float(np.linalg.norm(
        centroids[src_left] - centroids[src_right]))
    ref_distance = float(np.linalg.norm(
        centroids[ref_left] - centroids[ref_right]))
    tolerance = max(
        config.absolute_distance_tolerance_m,
        config.relative_distance_tolerance * max(src_distance, ref_distance),
    )
    distance_consistent = abs(src_distance - ref_distance) <= tolerance
    local = (tuple(sorted((src_left, src_right))) in edges
             and tuple(sorted((ref_left, ref_right))) in edges)
    return distance_consistent, local


def generate_matched_region_hypotheses(
    candidates: Sequence[Mapping[str, Any]],
    surfaces: Mapping[int, Any],
    src_count: int,
    *,
    explicit_edges: Sequence[Sequence[int]] = (),
    config: MatchedRegionConfig = MatchedRegionConfig(),
) -> list[dict[str, Any]]:
    """Build a bounded deterministic set of local multi-object hypotheses.

    A state is one-to-one, pairwise distance-consistent, and connected through
    local edges on both scans.  Beam pruning depends only on frozen ranks and
    canonical node indices, never a registration result.
    """
    if not (3 <= config.min_members <= config.max_members):
        raise MatchedRegionError("member bounds must satisfy 3 <= min <= max")
    if config.beam_width < 1 or config.max_hypotheses < 1:
        raise MatchedRegionError("resource limits must be positive")
    rows = _canonical_candidates(candidates, src_count)
    edges, centroids = _local_edges(
        surfaces, src_count, explicit_edges, config.local_neighbors)
    missing = sorted({value for row in rows
                      for value in (row["source_index"],
                                    row["reference_index"])
                      if value not in centroids})
    if missing:
        raise MatchedRegionError(
            f"candidate surfaces missing for indices {missing}")
    compatibility: dict[tuple[int, int], tuple[bool, bool]] = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            compatibility[(left, right)] = _pair_compatible(
                rows[left], rows[right], edges, centroids, config)

    def can_extend(state: tuple[int, ...], new: int) -> bool:
        distance_ok = []
        local_links = []
        for old in state:
            result = compatibility[(min(old, new), max(old, new))]
            distance_ok.append(result[0])
            local_links.append(result[1])
        return all(distance_ok) and any(local_links)

    def state_key(state: tuple[int, ...]) -> tuple[Any, ...]:
        members = [rows[index] for index in state]
        return (
            sum(row["worst_cross_rank"] for row in members),
            sum(row["rank_sum"] for row in members),
            tuple((row["source_index"], row["reference_index"])
                  for row in members),
        )

    frontier = [(index,) for index in range(len(rows))]
    eligible: set[tuple[int, ...]] = set()
    for target_size in range(2, config.max_members + 1):
        expanded: set[tuple[int, ...]] = set()
        for state in frontier:
            for new in range(state[-1] + 1, len(rows)):
                if can_extend(state, new):
                    expanded.add(state + (new,))
        frontier = sorted(expanded, key=state_key)[:config.beam_width]
        if target_size >= config.min_members:
            eligible.update(frontier)
        if not frontier:
            break

    ordered = sorted(eligible, key=lambda state: (
        -len(state), *state_key(state)))[:config.max_hypotheses]
    output = []
    for hypothesis_index, state in enumerate(ordered):
        members = [rows[index] for index in state]
        member_pairs = [[row["source_index"], row["reference_index"]]
                        for row in members]
        payload = {
            "members": member_pairs,
            "member_rank_records": members,
            "member_count": len(members),
        }
        payload["hypothesis_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload["hypothesis_index"] = hypothesis_index
        output.append(payload)
    return output


def union_hypothesis_surfaces(
    hypothesis: Mapping[str, Any], surfaces: Mapping[int, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical full object-surface unions; no correspondence mixing."""
    members = [tuple(int(value) for value in pair)
               for pair in hypothesis.get("members", [])]
    if len(members) < 3 or len({a for a, _ in members}) != len(members) \
            or len({b for _, b in members}) != len(members):
        raise MatchedRegionError("hypothesis is not a 3+ member one-to-one set")
    canonical = sorted(members)
    source = np.concatenate([_surface(surfaces[a], a)
                             for a, _ in canonical], axis=0)
    reference = np.concatenate([_surface(surfaces[b], b)
                                for _, b in canonical], axis=0)
    return np.ascontiguousarray(source), np.ascontiguousarray(reference)


def full_scene_union(
    surfaces: Mapping[int, Any], src_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic-only full-scene object union, never a selectable arm."""
    source_ids = sorted(int(index) for index in surfaces if int(index) < src_count)
    reference_ids = sorted(int(index) for index in surfaces if int(index) >= src_count)
    if not source_ids or not reference_ids:
        raise MatchedRegionError("both scan sides require registration surfaces")
    return (
        np.ascontiguousarray(np.concatenate(
            [_surface(surfaces[index], index) for index in source_ids])),
        np.ascontiguousarray(np.concatenate(
            [_surface(surfaces[index], index) for index in reference_ids])),
    )


def run_independent_bidirectional_geot(
    source: np.ndarray,
    reference: np.ndarray,
    runner: Callable[[np.ndarray, np.ndarray], tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Call the unchanged GeoT path twice; reverse is never inferred by inverse."""
    source = _surface(source, -1)
    reference = _surface(reference, -2)
    forward_status, forward = runner(source, reference)
    reverse_status, reverse = runner(reference, source)
    return {
        "forward": {"status": str(forward_status), "output": forward},
        "reverse": {"status": str(reverse_status), "output": reverse},
        "both_ok": forward_status == "ok" and reverse_status == "ok",
    }


def unique_safe_hypothesis(
    rows: Sequence[Mapping[str, Any]], *, known_bad: bool = False,
) -> dict[str, Any]:
    """Mechanically gate already-computed frozen Rule-B/q4 evidence."""
    safe = [row for row in rows if (
        row.get("forward_status") == "ok"
        and row.get("reverse_status") == "ok"
        and row.get("cross_direction_consistent") is True
        and row.get("rule_b_safe") is True
        and row.get("q4_stable") is True
    )]
    if known_bad:
        return {"accepted": False, "reason": "known_bad_veto",
                "safe_hypotheses": len(safe)}
    if len(safe) != 1:
        return {"accepted": False,
                "reason": ("no_unique_safe_hypothesis" if not safe
                           else "multiple_safe_hypotheses"),
                "safe_hypotheses": len(safe)}
    return {"accepted": True, "reason": "unique_safe_hypothesis",
            "safe_hypotheses": 1,
            "hypothesis_sha256": safe[0].get("hypothesis_sha256")}
