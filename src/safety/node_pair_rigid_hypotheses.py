"""Deterministic GT-free rigid hypotheses from per-node GeoT matches.

This module deliberately stops before RegistrationDecision.  It turns each
node-pair correspondence set into an independently auditable rigid estimate,
then partitions compatible transforms into disjoint complete-linkage modes.
It imports no GT loader and does not change any official SGAligner component.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v8_stage_order_consensus import transform_distance


@dataclass(frozen=True)
class NodePairHypothesisConfig:
    max_correspondences: int = 256
    ransac_trials: int = 64
    inlier_threshold_m: float = 0.05
    min_inliers: int = 6
    max_rotation_deg: float = 5.0
    max_translation_m: float = 0.10
    min_cluster_members: int = 3


class NodePairHypothesisError(RuntimeError):
    """A cached correspondence or deterministic solve is malformed."""


def _seed(context: str) -> int:
    return int.from_bytes(hashlib.sha256(context.encode()).digest()[:8], "big")


def rigid_kabsch(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (source.ndim != 2 or source.shape != reference.shape
            or source.shape[1:] != (3,) or len(source) < 3
            or not np.isfinite(source).all()
            or not np.isfinite(reference).all()):
        raise NodePairHypothesisError("Kabsch requires finite Nx3 pairs")
    centre_source = source.mean(axis=0)
    centre_reference = reference.mean(axis=0)
    covariance = ((source - centre_source).T
                  @ (reference - centre_reference))
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= 1e-10:
        raise NodePairHypothesisError("degenerate rigid sample")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = centre_reference - rotation @ centre_source
    if not np.isfinite(transform).all():
        raise NodePairHypothesisError("nonfinite rigid transform")
    return transform


def residuals(source: np.ndarray, reference: np.ndarray,
              transform: np.ndarray) -> np.ndarray:
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    return np.linalg.norm(moved - reference, axis=1)


def estimate_node_pair(
    source: Any,
    reference: Any,
    scores: Any,
    *,
    seed_context: str,
    config: NodePairHypothesisConfig = NodePairHypothesisConfig(),
) -> dict[str, Any]:
    """Estimate one rigid transform with deterministic minimal-set RANSAC."""
    source = np.asarray(source, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (source.ndim != 2 or source.shape != reference.shape
            or source.shape[1:] != (3,) or scores.shape != (len(source),)
            or not np.isfinite(source).all()
            or not np.isfinite(reference).all()
            or not np.isfinite(scores).all()):
        raise NodePairHypothesisError("malformed node-pair correspondences")
    order = np.argsort(-scores, kind="stable")[:config.max_correspondences]
    source = source[order]
    reference = reference[order]
    scores = scores[order]
    if len(source) < config.min_inliers:
        return {"status": "insufficient_correspondences",
                "correspondence_count": int(len(source))}
    rng = np.random.default_rng(_seed(seed_context))
    best: tuple[tuple[int, float, float, tuple[int, int, int]], np.ndarray] | None = None
    for _ in range(config.ransac_trials):
        sample = tuple(sorted(int(i) for i in rng.choice(
            len(source), 3, replace=False)))
        try:
            transform = rigid_kabsch(source[list(sample)], reference[list(sample)])
        except NodePairHypothesisError:
            continue
        distance = residuals(source, reference, transform)
        inliers = distance <= config.inlier_threshold_m
        count = int(inliers.sum())
        if count < config.min_inliers:
            continue
        quality = (
            count,
            float(scores[inliers].sum()),
            -float(np.median(distance[inliers])),
            tuple(-value for value in sample),
        )
        if best is None or quality > best[0]:
            best = (quality, inliers)
    if best is None:
        return {"status": "ransac_no_model",
                "correspondence_count": int(len(source))}
    inliers = best[1]
    try:
        transform = rigid_kabsch(source[inliers], reference[inliers])
        for _ in range(2):
            distance = residuals(source, reference, transform)
            inliers = distance <= config.inlier_threshold_m
            if int(inliers.sum()) < config.min_inliers:
                raise NodePairHypothesisError("refinement lost inlier support")
            transform = rigid_kabsch(source[inliers], reference[inliers])
    except NodePairHypothesisError as exc:
        return {"status": "refinement_failed", "reason": str(exc),
                "correspondence_count": int(len(source))}
    distance = residuals(source, reference, transform)
    inliers = distance <= config.inlier_threshold_m
    count = int(inliers.sum())
    return {
        "status": "ok",
        "transform": transform,
        "correspondence_count": int(len(source)),
        "inliers_5cm": count,
        "inlier_ratio_5cm": float(count / len(source)),
        "weighted_inlier_support": float(scores[inliers].sum()),
        "median_inlier_residual_m": float(np.median(distance[inliers])),
        "seed_context_sha256": hashlib.sha256(seed_context.encode()).hexdigest(),
    }


def estimate_cache_node_pairs(
    cached: Mapping[str, Any],
    *,
    direction: str,
    config: NodePairHypothesisConfig = NodePairHypothesisConfig(),
) -> dict[str, Any]:
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be forward or reverse")
    pair_id = str(cached.get("pair_id", ""))
    members = [(int(a), int(b)) for a, b in cached.get("_members", [])]
    estimates, failures = [], []
    for source_index, reference_index in members:
        entry = cached.get("geot", {}).get((source_index, reference_index))
        if not isinstance(entry, Mapping) or entry.get("status") != "ok":
            failures.append({"node_pair": [source_index, reference_index],
                             "status": (entry or {}).get("status", "missing")})
            continue
        source = entry.get("src_corr")
        reference = entry.get("ref_corr")
        if direction == "reverse":
            source, reference = reference, source
        estimate = estimate_node_pair(
            source, reference, entry.get("scores"),
            seed_context=(f"v9-node-pair|{pair_id}|{source_index}|"
                          f"{reference_index}|{direction}"),
            config=config)
        row = {"node_pair_original": [source_index, reference_index],
               "direction": direction, **estimate}
        if estimate["status"] == "ok":
            estimates.append(row)
        else:
            failures.append(row)
    estimates.sort(key=lambda row: (
        -row["inliers_5cm"], -row["weighted_inlier_support"],
        tuple(row["node_pair_original"])))
    return {"estimates": estimates, "failures": failures,
            "requested": len(members)}


def compatible(a: Mapping[str, Any], b: Mapping[str, Any],
               config: NodePairHypothesisConfig) -> bool:
    rotation, translation = transform_distance(a["transform"], b["transform"])
    return (rotation <= config.max_rotation_deg
            and translation <= config.max_translation_m)


def _candidate_clique(seed: int, remaining: set[int],
                      compatibility: np.ndarray,
                      order: Sequence[int]) -> tuple[int, ...]:
    clique = [seed]
    for index in order:
        if index != seed and index in remaining and all(
                compatibility[index, member] for member in clique):
            clique.append(index)
    return tuple(sorted(clique))


def disjoint_complete_linkage_modes(
    estimates: Sequence[Mapping[str, Any]],
    config: NodePairHypothesisConfig = NodePairHypothesisConfig(),
) -> dict[str, Any]:
    """Greedily extract deterministic, mutually exclusive rigid cliques."""
    rows = list(estimates)
    key = lambda index: (
        -int(rows[index]["inliers_5cm"]),
        -float(rows[index]["weighted_inlier_support"]),
        tuple(rows[index]["node_pair_original"]),
    )
    remaining = set(range(len(rows)))
    compatibility = np.eye(len(rows), dtype=bool)
    distances: dict[tuple[int, int], tuple[float, float]] = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            distance = transform_distance(
                rows[left]["transform"], rows[right]["transform"])
            distances[(left, right)] = distance
            compatibility[left, right] = compatibility[right, left] = (
                distance[0] <= config.max_rotation_deg
                and distance[1] <= config.max_translation_m)
    modes = []
    while remaining:
        order = sorted(remaining, key=key)
        candidates = [_candidate_clique(
            seed, remaining, compatibility, order) for seed in order]
        winning = min(candidates, key=lambda clique: (
            -len(clique),
            -sum(int(rows[index]["inliers_5cm"]) for index in clique),
            -sum(float(rows[index]["weighted_inlier_support"])
                 for index in clique),
            tuple(tuple(rows[index]["node_pair_original"])
                  for index in clique),
        ))
        remaining.difference_update(winning)
        member_rows = [rows[index] for index in winning]
        def medoid_score(index: int) -> tuple[float, tuple]:
            total = 0.0
            for other in winning:
                if index == other:
                    continue
                pair = (min(index, other), max(index, other))
                rotation, translation = distances[pair]
                total += rotation / config.max_rotation_deg
                total += translation / config.max_translation_m
            return total, key(index)
        medoid = min(winning, key=lambda index: (
            medoid_score(index)))
        modes.append({
            "member_count": len(winning),
            "members": [row["node_pair_original"] for row in member_rows],
            "member_indices": list(winning),
            "total_inliers_5cm": sum(
                int(row["inliers_5cm"]) for row in member_rows),
            "medoid_transform": rows[medoid]["transform"],
            "medoid_node_pair": rows[medoid]["node_pair_original"],
            "eligible": len(winning) >= config.min_cluster_members,
        })
    return {
        "modes": modes,
        "eligible_modes": [row for row in modes if row["eligible"]],
        "estimate_count": len(rows),
        "assigned_once": sum(row["member_count"] for row in modes) == len(rows),
    }


def cross_direction_mode_matches(
    forward_modes: Sequence[Mapping[str, Any]],
    reverse_modes: Sequence[Mapping[str, Any]],
    config: NodePairHypothesisConfig = NodePairHypothesisConfig(),
) -> list[dict[str, Any]]:
    matches = []
    for forward_index, forward in enumerate(forward_modes):
        for reverse_index, reverse in enumerate(reverse_modes):
            try:
                inverted = np.linalg.inv(np.asarray(
                    reverse["medoid_transform"], dtype=np.float64))
                rotation, translation = transform_distance(
                    forward["medoid_transform"], inverted)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if (rotation <= config.max_rotation_deg
                    and translation <= config.max_translation_m):
                matches.append({
                    "forward_mode": forward_index,
                    "reverse_mode": reverse_index,
                    "rotation_deg": rotation,
                    "translation_m": translation,
                })
    return sorted(matches, key=lambda row: (
        row["rotation_deg"], row["translation_m"],
        row["forward_mode"], row["reverse_mode"]))
