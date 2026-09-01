"""PAGOR/G3Reg-inspired multi-hypothesis registration with TEASER witness.

This module is deliberately GT-free.  It receives already matched metric
correspondences, creates independent solver-family hypotheses, and releases a
pose only when a unique complete-linkage cross-family cluster survives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import stable_json_sha256, validate_se3


HYPOTHESIS_SCHEMA = "relative_pose_hypothesis.v1"
DECISION_SCHEMA = "registration_decision.v2"


@dataclass(frozen=True)
class RobustPoseConfig:
    compatibility_thresholds_m: tuple[float, ...] = (0.05, 0.10, 0.20)
    minimum_separation_m: float = 0.04
    minimum_support: int = 6
    maximum_correspondences: int = 1000
    clique_seed_budget: int = 64
    deterministic_ransac_pool: int = 32
    deterministic_ransac_triangle_budget: int = 256
    residual_threshold_m: float = 0.10
    consensus_rotation_deg: float = 2.5
    consensus_translation_m: float = 0.05
    minimum_solver_families: int = 2
    pygcransac_repeats: int = 5
    pygcransac_quorum: int = 4
    minimum_overlap: float = 0.10
    maximum_icp_update_translation_m: float = 0.20
    maximum_icp_update_rotation_deg: float = 10.0
    maximum_bidirectional_translation_m: float = 0.20
    maximum_bidirectional_rotation_deg: float = 5.0
    maximum_cycle_translation_m: float = 0.20
    maximum_cycle_rotation_deg: float = 5.0
    minimum_spatial_extent_m: float = 2.0
    minimum_spatial_second_axis_m: float = 0.10


def _points(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite Nx3")
    return np.ascontiguousarray(array)


def _proper_kabsch(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(source) < 3 or source.shape != reference.shape:
        raise ValueError("Kabsch requires aligned 3+ point sets")
    source_mean, reference_mean = source.mean(0), reference.mean(0)
    covariance = (source - source_mean).T @ (reference - reference_mean)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= max(1e-12, singular[0] * 1e-8):
        raise ValueError("degenerate rigid support")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = reference_mean - rotation @ source_mean
    return validate_se3(transform)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = validate_se3(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_distance(left: object, right: object) -> tuple[float, float]:
    left, right = validate_se3(left, "left"), validate_se3(right, "right")
    delta = np.linalg.inv(left) @ right
    cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return (
        float(np.degrees(np.arccos(cosine))),
        float(np.linalg.norm(delta[:3, 3])),
    )


def compatibility_graph(
    source: object,
    reference: object,
    threshold_m: float,
    minimum_separation_m: float = 0.04,
) -> np.ndarray:
    source, reference = _points(source, "source"), _points(reference, "reference")
    if source.shape != reference.shape:
        raise ValueError("correspondence arrays must have equal shape")
    source_distance = np.linalg.norm(source[:, None] - source[None, :], axis=2)
    reference_distance = np.linalg.norm(
        reference[:, None] - reference[None, :], axis=2,
    )
    graph = (
        (source_distance >= minimum_separation_m)
        & (reference_distance >= minimum_separation_m)
        & (np.abs(source_distance - reference_distance) <= threshold_m)
    )
    np.fill_diagonal(graph, False)
    return graph


def greedy_max_clique(graph: np.ndarray, seed_budget: int = 64) -> np.ndarray:
    graph = np.asarray(graph, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("compatibility graph must be square")
    count = len(graph)
    if not count:
        return np.empty(0, dtype=np.int64)
    degrees = graph.sum(axis=1)
    seeds = sorted(range(count), key=lambda i: (-int(degrees[i]), i))[:seed_budget]
    best: tuple[int, ...] = ()
    for seed in seeds:
        clique = [seed]
        available = np.flatnonzero(graph[seed]).tolist()
        while available:
            indices = np.asarray(available, dtype=np.int64)
            induced = graph[np.ix_(indices, indices)].sum(axis=1)
            node = min(
                zip(available, induced.tolist()),
                key=lambda item: (-int(item[1]), -int(degrees[item[0]]), item[0]),
            )[0]
            clique.append(int(node))
            available = [
                value for value in available
                if value != node and graph[node, value]
            ]
        canonical = tuple(sorted(clique))
        if len(canonical) > len(best) or (
            len(canonical) == len(best) and canonical < best
        ):
            best = canonical
    return np.asarray(best, dtype=np.int64)


def _refine(
    source: np.ndarray,
    reference: np.ndarray,
    initial: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    transform = initial
    support = np.zeros(len(source), dtype=bool)
    for _ in range(4):
        residual = np.linalg.norm(
            transform_points(source, transform) - reference, axis=1,
        )
        support = residual <= threshold_m
        if int(support.sum()) < 3:
            break
        transform = _proper_kabsch(source[support], reference[support])
    return transform, support


def _transform_sha256(transform: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(transform, dtype=np.float64).tobytes(),
    ).hexdigest()


def _hypothesis(
    *,
    family: str,
    solver: str,
    transform: np.ndarray,
    support_count: int,
    correspondence_count: int,
    threshold_m: float,
    certificate: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    transform = validate_se3(transform)
    unsigned = {
        "schema": HYPOTHESIS_SCHEMA,
        "solver_family": family,
        "solver": solver,
        "matrix_convention": "T_reference_source_m",
        "transform": transform.tolist(),
        "transform_sha256": _transform_sha256(transform),
        "support_count": int(support_count),
        "correspondence_count": int(correspondence_count),
        "noise_bound_m": float(threshold_m),
        "estimate_scaling": False,
        "certificate": dict(certificate or {}),
        "gt_consumed": False,
    }
    return {**unsigned, "hypothesis_sha256": stable_json_sha256(unsigned)}


def compatibility_hypotheses(
    source: object,
    reference: object,
    config: RobustPoseConfig = RobustPoseConfig(),
) -> list[dict[str, Any]]:
    source, reference = _points(source, "source"), _points(reference, "reference")
    if source.shape != reference.shape:
        raise ValueError("correspondence arrays must have equal shape")
    source, reference = source[:config.maximum_correspondences], reference[:config.maximum_correspondences]
    values = []
    for threshold in config.compatibility_thresholds_m:
        graph = compatibility_graph(
            source, reference, threshold, config.minimum_separation_m,
        )
        clique = greedy_max_clique(graph, config.clique_seed_budget)
        if len(clique) < config.minimum_support:
            continue
        try:
            initial = _proper_kabsch(source[clique], reference[clique])
            transform, support = _refine(
                source, reference, initial, config.residual_threshold_m,
            )
        except ValueError:
            continue
        if int(support.sum()) < config.minimum_support:
            continue
        values.append(_hypothesis(
            family="compatibility_graph",
            solver=f"pagor_clique_{threshold:.2f}m",
            transform=transform,
            support_count=int(support.sum()),
            correspondence_count=len(source),
            threshold_m=threshold,
            certificate={
                "clique_size": int(len(clique)),
                "compatibility_edge_count": int(np.triu(graph, 1).sum()),
            },
        ))
    return values


def deterministic_ransac_hypothesis(
    source: object,
    reference: object,
    config: RobustPoseConfig = RobustPoseConfig(),
) -> dict[str, Any] | None:
    """Bounded lexicographic triangle RANSAC with no random state."""
    source, reference = _points(source, "source"), _points(reference, "reference")
    if source.shape != reference.shape:
        raise ValueError("correspondence arrays must have equal shape")
    count = min(len(source), config.maximum_correspondences)
    source, reference = source[:count], reference[:count]
    pool = min(count, config.deterministic_ransac_pool)
    best = None
    considered = 0
    for seed in combinations(range(pool), 3):
        if considered >= config.deterministic_ransac_triangle_budget:
            break
        considered += 1
        try:
            initial = _proper_kabsch(source[list(seed)], reference[list(seed)])
            refined, support = _refine(
                source, reference, initial, config.residual_threshold_m,
            )
        except ValueError:
            continue
        support_count = int(support.sum())
        if support_count < config.minimum_support:
            continue
        residual = np.linalg.norm(
            transform_points(source[support], refined) - reference[support],
            axis=1,
        )
        score = (
            -support_count,
            float(np.median(residual)),
            tuple(seed),
        )
        if best is None or score < best[0]:
            best = (score, refined, support_count, seed)
    if best is None:
        return None
    _score, transform, support_count, seed = best
    return _hypothesis(
        family="deterministic_ransac",
        solver="bounded_lexicographic_triangle_ransac",
        transform=transform,
        support_count=support_count,
        correspondence_count=count,
        threshold_m=config.residual_threshold_m,
        certificate={
            "seed_indices": list(seed),
            "triangles_considered": considered,
            "random_state_used": False,
        },
    )


def pygcransac_hypothesis(
    source: object,
    reference: object,
    *,
    threshold_m: float = 0.05,
) -> tuple[dict[str, Any] | None, str | None]:
    source, reference = _points(source, "source"), _points(reference, "reference")
    try:
        import pygcransac
    except ImportError:
        return None, "pygcransac_unavailable"
    correspondences = np.concatenate([source, reference], axis=1)
    minimum = correspondences.min(axis=0)
    shifted = correspondences - minimum
    try:
        row_transform, _ = pygcransac.findRigidTransform(
            np.ascontiguousarray(shifted, dtype=np.float64),
            probabilities=[], threshold=threshold_m, neighborhood_size=4.0,
            sampler=1, min_iters=1000, max_iters=10000,
            spatial_coherence_weight=0.0, use_space_partitioning=True,
            neighborhood=0, conf=0.999, use_sprt=False,
        )
    except (RuntimeError, ValueError) as exc:
        return None, f"pygcransac_failed:{type(exc).__name__}:{exc}"
    if not isinstance(row_transform, np.ndarray) or row_transform.shape != (4, 4):
        return None, "pygcransac_malformed_transform"
    t1 = np.eye(4)
    t1[3, :3] = -minimum[:3]
    t2_inv = np.eye(4)
    t2_inv[3, :3] = minimum[3:]
    transform = validate_se3((t1 @ row_transform @ t2_inv).T)
    residual = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
    support = int((residual <= threshold_m).sum())
    return _hypothesis(
        family="pygcransac", solver="pygcransac",
        transform=transform, support_count=support,
        correspondence_count=len(source), threshold_m=threshold_m,
    ), None


def repeated_solver_consensus(
    hypotheses: Sequence[Mapping[str, Any]],
    config: RobustPoseConfig = RobustPoseConfig(),
    *,
    quorum: int | None = None,
) -> dict[str, Any] | None:
    """Return one observed medoid only after a unique repeat consensus."""
    quorum = config.pygcransac_quorum if quorum is None else int(quorum)
    if len(hypotheses) < quorum:
        return None
    parsed = [dict(row) for row in hypotheses]
    compatible = np.zeros((len(parsed), len(parsed)), dtype=bool)
    for left in range(len(parsed)):
        compatible[left, left] = True
        for right in range(left + 1, len(parsed)):
            rotation, translation = transform_distance(
                parsed[left]["transform"], parsed[right]["transform"],
            )
            value = (
                rotation <= config.consensus_rotation_deg
                and translation <= config.consensus_translation_m
            )
            compatible[left, right] = compatible[right, left] = value
    cliques = []
    indices = tuple(range(len(parsed)))
    for size in range(1, len(indices) + 1):
        for subset in combinations(indices, size):
            if all(compatible[left, right] for left, right in combinations(subset, 2)):
                cliques.append(subset)
    maximal = [
        value for value in cliques
        if not any(set(value) < set(other) for other in cliques)
    ]
    eligible = sorted(
        {value for value in maximal if len(value) >= quorum},
        key=lambda value: (-len(value), value),
    )
    if len(eligible) != 1:
        return None
    winning = eligible[0]
    if any(len(value) >= 2 for value in maximal if value != winning):
        return None
    scores = []
    for index in winning:
        total = sum(
            rotation / config.consensus_rotation_deg
            + translation / config.consensus_translation_m
            for rotation, translation in (
                transform_distance(
                    parsed[index]["transform"], parsed[other]["transform"],
                ) for other in winning
            )
        )
        scores.append((total, str(parsed[index]["hypothesis_sha256"]), index))
    selected = dict(parsed[min(scores)[2]])
    selected["certificate"] = {
        **dict(selected.get("certificate", {})),
        "repeat_count": len(parsed),
        "repeat_quorum": quorum,
        "winning_repeat_indices": list(winning),
        "repeat_transform_sha256": [
            parsed[index]["transform_sha256"] for index in range(len(parsed))
        ],
    }
    unsigned = dict(selected)
    unsigned.pop("hypothesis_sha256", None)
    selected["hypothesis_sha256"] = stable_json_sha256(unsigned)
    return selected


def teaser_hypotheses(
    source: object,
    reference: object,
    config: RobustPoseConfig = RobustPoseConfig(),
) -> tuple[list[dict[str, Any]], str | None]:
    source, reference = _points(source, "source"), _points(reference, "reference")
    try:
        import teaserpp_python
    except ImportError:
        return [], "teaserpp_python_unavailable"
    values = []
    for noise_bound in config.compatibility_thresholds_m:
        params = teaserpp_python.RobustRegistrationSolver.Params()
        params.cbar2 = 1.0
        params.noise_bound = float(noise_bound)
        params.estimate_scaling = False
        params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        params.rotation_gnc_factor = 1.4
        params.rotation_max_iterations = 100
        params.rotation_cost_threshold = 1e-12
        try:
            solver = teaserpp_python.RobustRegistrationSolver(params)
            solver.solve(source.T, reference.T)
            solution = solver.getSolution()
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(solution.rotation, dtype=np.float64)
            transform[:3, 3] = np.asarray(solution.translation, dtype=np.float64).reshape(3)
            transform = validate_se3(transform)
        except Exception as exc:  # optional backend boundary
            return values, f"teaserpp_failed:{type(exc).__name__}:{exc}"
        residual = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
        support = int((residual <= noise_bound).sum())
        if support >= config.minimum_support:
            values.append(_hypothesis(
                family="teaserpp", solver=f"teaserpp_gnc_tls_{noise_bound:.2f}m",
                transform=transform, support_count=support,
                correspondence_count=len(source), threshold_m=noise_bound,
                certificate={"rotation_algorithm": "GNC_TLS"},
            ))
    return values, None


def generate_hypotheses(
    source: object,
    reference: object,
    config: RobustPoseConfig = RobustPoseConfig(),
    *,
    include_pygcransac: bool = True,
    include_teaser: bool = True,
) -> dict[str, Any]:
    source, reference = _points(source, "source"), _points(reference, "reference")
    if source.shape != reference.shape:
        raise ValueError("correspondence arrays must have equal shape")
    hypotheses = compatibility_hypotheses(source, reference, config)
    witness_hypotheses = []
    deterministic = deterministic_ransac_hypothesis(
        source, reference, config,
    )
    if deterministic is not None:
        hypotheses.append(deterministic)
    unavailable = []
    if include_pygcransac:
        repeated, repeat_errors = [], []
        for _ in range(config.pygcransac_repeats):
            value, reason = pygcransac_hypothesis(source, reference)
            if value is not None:
                repeated.append(value)
            if reason:
                repeat_errors.append(reason)
        value = repeated_solver_consensus(repeated, config)
        if value is not None:
            value["witness_only"] = True
            witness_hypotheses.append(value)
        else:
            unavailable.append(
                "pygcransac_repeat_consensus_failed:"
                f"valid={len(repeated)}/{config.pygcransac_repeats}:"
                f"errors={repeat_errors}"
            )
    if include_teaser:
        values, reason = teaser_hypotheses(source, reference, config)
        hypotheses.extend(values)
        if reason:
            unavailable.append(reason)
    return {
        "schema": "robust_pose_hypothesis_set.v1",
        "config": asdict(config),
        "config_sha256": stable_json_sha256(asdict(config)),
        "correspondence_count": len(source),
        "hypotheses": hypotheses,
        "witness_hypotheses": witness_hypotheses,
        "unavailable_backends": unavailable,
        "gt_consumed": False,
    }


def _compatible(left: Mapping[str, Any], right: Mapping[str, Any], config: RobustPoseConfig) -> bool:
    rotation, translation = transform_distance(left["transform"], right["transform"])
    return (
        rotation <= config.consensus_rotation_deg
        and translation <= config.consensus_translation_m
    )


def _complete_linkage_clusters(
    hypotheses: Sequence[Mapping[str, Any]], config: RobustPoseConfig,
) -> list[tuple[int, ...]]:
    if len(hypotheses) > 12:
        raise ValueError("hypothesis budget exceeds exhaustive consensus limit")
    cliques = []
    indices = tuple(range(len(hypotheses)))
    for size in range(1, len(indices) + 1):
        for subset in combinations(indices, size):
            if all(
                _compatible(hypotheses[left], hypotheses[right], config)
                for left, right in combinations(subset, 2)
            ):
                cliques.append(subset)
    maximal = [
        clique for clique in cliques
        if not any(set(clique) < set(other) for other in cliques)
    ]
    return sorted(set(maximal), key=lambda value: (-len(value), value))


def select_cross_solver_consensus(
    hypotheses: Sequence[Mapping[str, Any]],
    config: RobustPoseConfig = RobustPoseConfig(),
) -> dict[str, Any]:
    parsed = []
    for row in hypotheses:
        if row.get("schema") != HYPOTHESIS_SCHEMA or row.get("gt_consumed") is not False:
            raise ValueError("hypothesis schema or GT contract mismatch")
        copy = dict(row)
        copy["transform"] = validate_se3(row["transform"])
        parsed.append(copy)
    clusters = _complete_linkage_clusters(parsed, config) if parsed else []
    eligible = [
        cluster for cluster in clusters
        if len({parsed[index]["solver_family"] for index in cluster})
        >= config.minimum_solver_families
    ]
    base = {
        "schema": "robust_pose_consensus.v1",
        "accepted": False,
        "hypothesis_count": len(parsed),
        "eligible_cluster_count": len(eligible),
        "clusters": [list(value) for value in clusters],
        "gt_consumed": False,
    }
    if len(eligible) != 1:
        return {
            **base,
            "reason": "no_cross_solver_cluster" if not eligible else "ambiguous_cross_solver_clusters",
        }
    winning = eligible[0]
    scores = []
    for index in winning:
        distances = [transform_distance(parsed[index]["transform"], parsed[other]["transform"]) for other in winning]
        score = sum(
            rotation / config.consensus_rotation_deg
            + translation / config.consensus_translation_m
            for rotation, translation in distances
        )
        non_deterministic = int(
            parsed[index]["solver_family"] in {"pygcransac", "teaserpp"}
        )
        family_preference = {
            "deterministic_ransac": 0,
            "compatibility_graph": 1,
            "teaserpp": 2,
            "pygcransac": 3,
        }.get(parsed[index]["solver_family"], 9)
        scores.append((
            non_deterministic, family_preference, score,
            str(parsed[index]["hypothesis_sha256"]), index,
        ))
    selected_index = min(scores)[4]
    selected = parsed[selected_index]
    return {
        **base,
        "accepted": True,
        "reason": "unique_cross_solver_complete_linkage_cluster",
        "winning_indices": list(winning),
        "solver_families": sorted({parsed[index]["solver_family"] for index in winning}),
        "selected_index": selected_index,
        "selected_hypothesis_sha256": selected["hypothesis_sha256"],
        "selected_transform": selected["transform"].tolist(),
    }


def spatial_support(points: object) -> tuple[float, float]:
    points = _points(points, "spatial support")
    if len(points) < 3:
        return 0.0, 0.0
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return float(singular[0]), float(singular[1])


def decide_registration_v2(
    consensus: Mapping[str, Any],
    metrics: Mapping[str, object],
    config: RobustPoseConfig = RobustPoseConfig(),
) -> dict[str, Any]:
    forbidden = sorted(
        key for key in metrics
        if key.lower().startswith("gt") or "ground_truth" in key.lower()
    )
    if forbidden:
        raise ValueError(f"GT fields are forbidden in registration decision: {forbidden}")
    required = {
        "spatial_extent_m", "spatial_second_axis_m", "icp_update_translation_m",
        "icp_update_rotation_deg", "bidirectional_translation_m",
        "bidirectional_rotation_deg", "cycle_translation_m", "cycle_rotation_deg",
        "overlap_ratio",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"registration decision metrics missing: {missing}")
    numeric = {key: float(metrics[key]) for key in required}
    if not np.isfinite(list(numeric.values())).all():
        raise ValueError("registration decision metrics must be finite")
    reasons = []
    if consensus.get("accepted") is not True:
        reasons.append(str(consensus.get("reason", "consensus_rejected")))
    checks = (
        (numeric["spatial_extent_m"] >= config.minimum_spatial_extent_m, "spatial_extent_too_small"),
        (numeric["spatial_second_axis_m"] >= config.minimum_spatial_second_axis_m, "spatial_second_axis_too_small"),
        (numeric["icp_update_translation_m"] <= config.maximum_icp_update_translation_m, "icp_translation_update_too_large"),
        (numeric["icp_update_rotation_deg"] <= config.maximum_icp_update_rotation_deg, "icp_rotation_update_too_large"),
        (numeric["bidirectional_translation_m"] <= config.maximum_bidirectional_translation_m, "bidirectional_translation_inconsistent"),
        (numeric["bidirectional_rotation_deg"] <= config.maximum_bidirectional_rotation_deg, "bidirectional_rotation_inconsistent"),
        (numeric["cycle_translation_m"] <= config.maximum_cycle_translation_m, "cycle_translation_inconsistent"),
        (numeric["cycle_rotation_deg"] <= config.maximum_cycle_rotation_deg, "cycle_rotation_inconsistent"),
        (numeric["overlap_ratio"] >= config.minimum_overlap, "overlap_too_small"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    accepted = not reasons
    unsigned = {
        "schema": DECISION_SCHEMA,
        "status": "accepted" if accepted else "rejected",
        "usable_for_reconstruction": accepted,
        "rejection_reasons": reasons,
        "selected_transform": consensus.get("selected_transform") if accepted else None,
        "consensus": dict(consensus),
        "metrics": numeric,
        "config": asdict(config),
        "gt_consumed": False,
        "fallback_used": False,
    }
    return {**unsigned, "decision_sha256": stable_json_sha256(unsigned)}
