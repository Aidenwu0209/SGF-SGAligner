"""Fix-2 safety module: corr-independent RegistrationDecision evidence.

Real surface overlap, real segment ICP, real bidirectional consistency,
node-pair evidence — every feature carries source/unit provenance.  The
three pre-registered candidate rules A/B/C are evaluated here; GT never
enters any feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class SurfaceEvidence:
    overlap_5cm: float
    overlap_10cm: float
    symmetric_trimmed_chamfer_m: float
    median_residual_m: float
    p90_residual_m: float
    n_src_points: int
    n_ref_points: int
    seed: int
    source: str = "full world object surfaces, corr-independent"
    units: str = "metres"


def surface_evidence(
    src_surface: np.ndarray,
    ref_surface: np.ndarray,
    transform: np.ndarray,
    *,
    seed: int = 42,
    max_points: int = 20000,
    trim: float = 0.9,
) -> SurfaceEvidence:
    """Bidirectional NN validation on FULL matched surfaces.

    Surfaces are the union of the matched objects' complete registration
    points (never GeoT corr points).  Deterministic subsample to
    max_points with the recorded seed.
    """
    rng = np.random.default_rng(seed)
    src = np.asarray(src_surface, dtype=np.float64)
    ref = np.asarray(ref_surface, dtype=np.float64)
    if len(src) > max_points:
        src = src[rng.choice(len(src), max_points, replace=False)]
    if len(ref) > max_points:
        ref = ref[rng.choice(len(ref), max_points, replace=False)]
    moved = src @ transform[:3, :3].T + transform[:3, 3]
    d_sr = cKDTree(ref).query(moved, k=1)[0]
    d_rs = cKDTree(moved).query(ref, k=1)[0]
    k = max(int(len(d_sr) * trim), 1)
    trimmed_src = np.sort(d_sr)[:k]
    k_r = max(int(len(d_rs) * trim), 1)
    trimmed_ref = np.sort(d_rs)[:k_r]
    return SurfaceEvidence(
        overlap_5cm=float(np.mean(d_sr <= 0.05)),
        overlap_10cm=float(np.mean(d_sr <= 0.10)),
        symmetric_trimmed_chamfer_m=float(
            (trimmed_src.mean() + trimmed_ref.mean()) / 2
        ),
        median_residual_m=float(np.median(d_sr)),
        p90_residual_m=float(np.percentile(d_sr, 90)),
        n_src_points=int(len(src)),
        n_ref_points=int(len(ref)),
        seed=int(seed),
    )


@dataclass
class SegmentIcpResult:
    transform: np.ndarray
    converged: bool
    fitness: float
    rmse_m: float
    update_translation_m: float
    update_rotation_deg: float
    iterations_run: int
    source: str = "deterministic NN-ICP on matched-object surface union"
    units: str = "metres / degrees"


def segment_icp(
    src_surface: np.ndarray,
    ref_surface: np.ndarray,
    initial: np.ndarray,
    *,
    threshold: float = 0.20,
    max_iterations: int = 30,
    seed: int = 42,
    max_points: int = 30000,
) -> SegmentIcpResult:
    """Deterministic point-to-point ICP on the matched surface union."""
    rng = np.random.default_rng(seed)
    src = np.asarray(src_surface, dtype=np.float64)
    ref = np.asarray(ref_surface, dtype=np.float64)
    if len(src) > max_points:
        src = src[rng.choice(len(src), max_points, replace=False)]
    if len(ref) > max_points:
        ref = ref[rng.choice(len(ref), max_points, replace=False)]
    tree = cKDTree(ref)
    transform = np.asarray(initial, dtype=np.float64).copy()
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        moved = src @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1)
        keep = distances <= threshold
        if int(keep.sum()) < 10:
            break
        a = src[keep]
        b = ref[indices[keep]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = cb - rotation @ ca
    moved = src @ transform[:3, :3].T + transform[:3, 3]
    distances, _ = tree.query(moved, k=1)
    keep = distances <= threshold
    fitness = float(keep.mean()) if len(keep) else 0.0
    rmse = (
        float(np.sqrt(np.mean(distances[keep] ** 2)))
        if int(keep.sum()) else float("inf")
    )
    update_t = float(
        np.linalg.norm(transform[:3, 3] - np.asarray(initial)[:3, 3])
    )
    cos_r = (
        np.trace(
            transform[:3, :3].T @ np.asarray(initial)[:3, :3]
        ) - 1
    ) / 2
    update_r = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
    converged = bool(fitness > 0.0 and np.isfinite(rmse))
    return SegmentIcpResult(
        transform=transform,
        converged=converged,
        fitness=fitness,
        rmse_m=rmse,
        update_translation_m=update_t,
        update_rotation_deg=update_r,
        iterations_run=iterations,
    )


def transform_discrepancy(
    t_sr: np.ndarray, t_rs: np.ndarray
) -> tuple[float, float]:
    """Rotation/translation gap between T_sr and inverse(T_rs)."""
    composed = t_sr @ t_rs  # ~identity when consistent
    cos_r = (np.trace(composed[:3, :3]) - 1) / 2
    rotation_gap = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
    translation_gap = float(np.linalg.norm(composed[:3, 3]))
    return rotation_gap, translation_gap


# ------------------------------------------------------------------ rules
# Pre-registered thresholds (round values, frozen on calibration only;
# selection chooses among rule candidates A/B/C, never threshold tuning
# on fixed12).

RULE_THRESHOLDS = {
    "min_overlap_10cm": 0.10,
    "max_median_residual_m": 0.10,
    "max_symmetric_trimmed_chamfer_m": 0.10,
    "max_icp_update_translation_m": 0.20,
    "max_icp_update_rotation_deg": 10.0,
    "min_icp_fitness": 0.30,
    "max_bidirectional_rotation_deg": 5.0,
    "max_bidirectional_translation_m": 0.20,
    "min_node_pair_success_ratio": 0.50,
    "min_ransac_inliers": 6,
    "min_spatial_extent_m": 1.0,
}


def evaluate_rule_a(features: dict) -> dict:
    """A: independent surface + real ICP."""
    violations = []
    if features["overlap_10cm"] < RULE_THRESHOLDS["min_overlap_10cm"]:
        violations.append("surface_overlap_10cm_below_min")
    if features["median_residual_m"] > RULE_THRESHOLDS["max_median_residual_m"]:
        violations.append("surface_median_residual_above_max")
    if (
        features["symmetric_trimmed_chamfer_m"]
        > RULE_THRESHOLDS["max_symmetric_trimmed_chamfer_m"]
    ):
        violations.append("surface_chamfer_above_max")
    if not features["icp_converged"]:
        violations.append("segment_icp_not_converged")
    if (
        features["icp_update_translation_m"]
        > RULE_THRESHOLDS["max_icp_update_translation_m"]
    ):
        violations.append("icp_translation_update_above_max")
    if (
        features["icp_update_rotation_deg"]
        > RULE_THRESHOLDS["max_icp_update_rotation_deg"]
    ):
        violations.append("icp_rotation_update_above_max")
    if features["icp_fitness"] < RULE_THRESHOLDS["min_icp_fitness"]:
        violations.append("icp_fitness_below_min")
    if features["ransac_inliers"] < RULE_THRESHOLDS["min_ransac_inliers"]:
        violations.append("ransac_inliers_below_min")
    if features["spatial_extent_m"] < RULE_THRESHOLDS["min_spatial_extent_m"]:
        violations.append("spatial_extent_below_min")
    return violations


def evaluate_rule_b(features: dict) -> dict:
    """B: A + bidirectional consistency (must exist)."""
    violations = evaluate_rule_a(features)
    if not features.get("bidirectional_available"):
        violations.append("bidirectional_unavailable")
        return violations
    if (
        features["bidirectional_rotation_deg"]
        > RULE_THRESHOLDS["max_bidirectional_rotation_deg"]
    ):
        violations.append("bidirectional_rotation_above_max")
    if (
        features["bidirectional_translation_m"]
        > RULE_THRESHOLDS["max_bidirectional_translation_m"]
    ):
        violations.append("bidirectional_translation_above_max")
    return violations


def evaluate_rule_c(features: dict) -> dict:
    """C: B + node-pair success ratio / spatial support."""
    violations = evaluate_rule_b(features)
    if (
        features["node_pair_success_ratio"]
        < RULE_THRESHOLDS["min_node_pair_success_ratio"]
    ):
        violations.append("node_pair_success_ratio_below_min")
    return violations


RULE_EVALUATORS = {"A": evaluate_rule_a, "B": evaluate_rule_b,
                   "C": evaluate_rule_c}
