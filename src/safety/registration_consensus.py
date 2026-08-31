"""GT-free complete-linkage consensus for repeated rigid registration."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class ConsensusConfig:
    repeats: int = 5
    quorum: int = 4
    max_rotation_deg: float = 2.5
    max_translation_m: float = 0.05


def transform_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """SO(3) geodesic angle and world-frame translation L2."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("consensus transforms must be 4x4")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("consensus transforms must be finite")
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    return rotation, translation


def _compatible(a: dict, b: dict, config: ConsensusConfig) -> bool:
    dr, dt = transform_distance(a["transform"], b["transform"])
    return dr <= config.max_rotation_deg and dt <= config.max_translation_m


def _maximal_cliques(records: list[dict],
                     config: ConsensusConfig) -> list[tuple[int, ...]]:
    """Enumerate maximal complete-linkage sets; K is fixed at five."""
    indices = tuple(range(len(records)))
    cliques: list[tuple[int, ...]] = []
    for size in range(len(indices), 0, -1):
        for subset in combinations(indices, size):
            if all(_compatible(records[i], records[j], config)
                   for i, j in combinations(subset, 2)):
                if not any(set(subset) < set(existing)
                           for existing in cliques):
                    cliques.append(subset)
    maximal = [
        clique for clique in cliques
        if not any(set(clique) < set(other) for other in cliques)
    ]
    return sorted(maximal, key=lambda c: (-len(c), c))


def _medoid(records: list[dict], clique: tuple[int, ...]) -> int:
    def score(index: int) -> tuple[float, str]:
        total = 0.0
        for other in clique:
            dr, dt = transform_distance(
                records[index]["transform"], records[other]["transform"])
            total += dr / 5.0 + dt / 0.20
        return total, str(records[index].get("stable_signature", index))
    return min(clique, key=score)


def evaluate_direction(records: list[dict], config: ConsensusConfig) -> dict:
    """Evaluate one direction. Records contain no GT-derived fields."""
    reasons: list[str] = []
    if len(records) != config.repeats:
        reasons.append("repeat_count_mismatch")
    valid: list[dict] = []
    for index, record in enumerate(records):
        transform = record.get("transform")
        try:
            finite = (
                transform is not None
                and np.asarray(transform).shape == (4, 4)
                and np.isfinite(np.asarray(transform, dtype=float)).all()
            )
        except (TypeError, ValueError):
            finite = False
        if not finite or record.get("status") != "ok":
            continue
        copy = dict(record)
        copy["_original_index"] = index
        valid.append(copy)
    if len(valid) != len(records):
        reasons.append("invalid_run_present")
    accepted = [
        record for record in valid
        if record.get("rule_b_accepted") is True
    ]
    if len(accepted) < config.quorum:
        reasons.append("rule_b_quorum_not_met")
    cliques = _maximal_cliques(accepted, config) if accepted else []
    largest = len(cliques[0]) if cliques else 0
    equally_largest = [c for c in cliques if len(c) == largest]
    if largest < config.quorum:
        reasons.append("consensus_quorum_not_met")
    if len(equally_largest) != 1:
        reasons.append("largest_clique_not_unique")
    if any(len(c) >= 2 for c in cliques[1:]):
        reasons.append("rival_cluster_present")
    winning = equally_largest[0] if len(equally_largest) == 1 else ()
    if winning and len(winning) != len(accepted):
        reasons.append("accepted_run_outside_consensus")
    medoid = _medoid(accepted, winning) if winning else None
    return {
        "usable": not reasons,
        "rejection_reasons": reasons,
        "requested": len(records),
        "valid": len(valid),
        "rule_b_accepted": len(accepted),
        "clique_sizes": [len(c) for c in cliques],
        "winning_original_indices": [
            accepted[i]["_original_index"] for i in winning],
        "medoid_original_index": (
            accepted[medoid]["_original_index"]
            if medoid is not None else None),
        "config": {
            "repeats": config.repeats,
            "quorum": config.quorum,
            "max_rotation_deg": config.max_rotation_deg,
            "max_translation_m": config.max_translation_m,
        },
    }


def cross_direction_agreement(
    forward: list[dict],
    reverse_inverted: list[dict],
    config: ConsensusConfig,
) -> dict:
    """Require at least quorum independent forward/reverse agreements."""
    pairs = []
    for i, fwd in enumerate(forward):
        if fwd.get("status") != "ok" or not fwd.get("rule_b_accepted"):
            continue
        for j, rev in enumerate(reverse_inverted):
            if rev.get("status") != "ok" or not rev.get("rule_b_accepted"):
                continue
            try:
                dr, dt = transform_distance(
                    fwd["transform"], rev["transform"])
            except ValueError:
                continue
            if (dr <= config.max_rotation_deg
                    and dt <= config.max_translation_m):
                pairs.append((i, j, dr, dt))
    by_left: dict[int, list[tuple[int, float, float]]] = {}
    for i, j, dr, dt in pairs:
        by_left.setdefault(i, []).append((j, dr, dt))
    for values in by_left.values():
        values.sort(key=lambda value: (value[1], value[2], value[0]))
    right_match: dict[int, tuple[int, float, float]] = {}

    def augment(left: int, seen: set[int]) -> bool:
        for right, dr, dt in by_left.get(left, []):
            if right in seen:
                continue
            seen.add(right)
            previous = right_match.get(right)
            if previous is None or augment(previous[0], seen):
                right_match[right] = (left, dr, dt)
                return True
        return False

    for left in sorted(by_left):
        augment(left, set())
    selected = sorted(
        ((left, right, dr, dt)
         for right, (left, dr, dt) in right_match.items()),
        key=lambda value: (value[0], value[1]),
    )
    usable = len(selected) >= config.quorum
    return {
        "usable": usable,
        "agreement_count": len(selected),
        "matches": [
            {"forward": i, "reverse": j, "rotation_deg": dr,
             "translation_m": dt}
            for i, j, dr, dt in selected
        ],
        "rejection_reasons": [] if usable else [
            "cross_direction_quorum_not_met"],
    }
