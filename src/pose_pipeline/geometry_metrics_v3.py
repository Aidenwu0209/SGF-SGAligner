"""Independent no-GT geometry diagnostics; never a promotion gate.

V2 is preserved verbatim.  This module removes the world-grid layer test and
checks the support of supplied plane hypotheses.  Close parallel surfaces can
also be real furniture: geometric proximity is not proof of reconstruction
error.  RGB-D visibility validation and independent threshold calibration are
still required before these diagnostics may authorize a trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeometryDiagnosticConfig:
    minimum_normal_separation_m: float = 0.03
    maximum_normal_separation_m: float = 0.12
    maximum_tangent_separation_m: float = 0.02
    maximum_parallel_normal_angle_deg: float = 15.0
    plane_support_band_m: float = 0.15
    maximum_plane_normal_angle_deg: float = 15.0
    maximum_plane_offset_m: float = 0.20
    maximum_plane_centroid_distance_m: float = 2.0
    projected_support_distance_m: float = 0.04
    minimum_projected_overlap_fraction: float = 0.50
    minimum_shared_support_points: int = 300
    minimum_valid_normal_fraction: float = 0.80
    minimum_layer_fraction_for_relative_change: float = 0.01
    minimum_thickness_for_relative_change_m: float = 0.002
    query_batch_size: int = 256

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) or value <= 0
               for value in values.values()):
            raise ValueError("Diagnostic thresholds must be finite and positive")
        if self.minimum_normal_separation_m >= self.maximum_normal_separation_m:
            raise ValueError("Normal separation interval must be increasing")
        if max(self.maximum_parallel_normal_angle_deg,
               self.maximum_plane_normal_angle_deg) >= 90.0:
            raise ValueError("Normal angle limits must be below 90 degrees")
        for name in ("minimum_projected_overlap_fraction",
                     "minimum_valid_normal_fraction",
                     "minimum_layer_fraction_for_relative_change"):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        for name in ("minimum_shared_support_points", "query_batch_size"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")


def _valid_cloud(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("Points and normals must have matching N x 3 shapes")
    lengths = np.linalg.norm(normals, axis=1)
    finite_points = np.isfinite(points).all(axis=1)
    valid = finite_points & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    count = int(valid.sum())
    return points[valid], normals[valid] / lengths[valid, None], {
        "input_points": int(len(points)),
        "finite_points": int(finite_points.sum()),
        "valid_normal_points": count,
        "valid_normal_fraction": count / max(int(len(points)), 1),
    }


def candidate_layer_pairs(
    points: np.ndarray, normals: np.ndarray, *,
    config: GeometryDiagnosticConfig | None = None,
) -> dict:
    """Count points having a close, parallel surface along their normal.

    Euclidean distances and normal dot products make the predicate invariant
    to a common rigid transform (apart from floating-point threshold ties).
    No world-axis voxel grid or finite-k nearest-neighbor truncation is used.
    The denominator is valid-normal points, so this is a point-weighted
    diagnostic and is not invariant to resampling or unequal surface density.
    """
    from scipy.spatial import cKDTree

    config = config or GeometryDiagnosticConfig()
    points, normals, coverage = _valid_cloud(points, normals)
    flagged = np.zeros(len(points), dtype=bool)
    pair_count = 0
    if len(points):
        tree = cKDTree(points)
        radius = math.hypot(config.maximum_normal_separation_m,
                            config.maximum_tangent_separation_m)
        cosine = math.cos(math.radians(config.maximum_parallel_normal_angle_deg))
        epsilon = 1e-10
        for start in range(0, len(points), config.query_batch_size):
            stop = min(start + config.query_batch_size, len(points))
            neighborhoods = tree.query_ball_point(points[start:stop], radius + epsilon)
            sizes = np.asarray([len(item) for item in neighborhoods], dtype=np.int64)
            if not sizes.sum():
                continue
            source = np.repeat(np.arange(start, stop), sizes)
            target = np.concatenate(neighborhoods).astype(np.int64)
            # Visit each unordered pair once; the predicate is symmetric.
            unique = target > source
            source, target = source[unique], target[unique]
            delta = points[target] - points[source]
            length2 = np.einsum("ij,ij->i", delta, delta)
            normal_dot = np.abs(np.einsum("ij,ij->i", normals[source], normals[target]))
            axial_source = np.abs(np.einsum("ij,ij->i", delta, normals[source]))
            axial_target = np.abs(np.einsum("ij,ij->i", delta, normals[target]))
            tangent_source2 = np.maximum(length2 - axial_source ** 2, 0.0)
            tangent_target2 = np.maximum(length2 - axial_target ** 2, 0.0)
            minimum = config.minimum_normal_separation_m - epsilon
            maximum = config.maximum_normal_separation_m + epsilon
            tangent2 = config.maximum_tangent_separation_m ** 2 + epsilon
            valid = (
                (normal_dot >= cosine - epsilon)
                & (axial_source >= minimum) & (axial_source <= maximum)
                & (axial_target >= minimum) & (axial_target <= maximum)
                & (tangent_source2 <= tangent2) & (tangent_target2 <= tangent2)
            )
            flagged[source[valid]] = True
            flagged[target[valid]] = True
            pair_count += int(valid.sum())
    reliable = (len(points) >= config.minimum_shared_support_points
                and coverage["valid_normal_fraction"] >= config.minimum_valid_normal_fraction)
    return {
        "schema": "candidate_layer_pairs.v1",
        **coverage,
        "status": "measured" if reliable else "inconclusive",
        "candidate_pair_count": pair_count,
        "points_with_candidate_layer_pair": int(flagged.sum()),
        "point_fraction_with_candidate_layer_pair": (
            float(flagged.mean()) if len(points) else None
        ),
        "error_truth_established": False,
        "gt_consumed": False,
    }


def _plane_frame(plane: dict) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(plane.get("normal"), dtype=np.float64)
    centre = np.asarray(plane.get("centroid_m"), dtype=np.float64)
    if normal.shape != (3,) or centre.shape != (3,):
        raise ValueError("Plane hypotheses require normal and centroid_m vectors")
    if not np.isfinite(normal).all() or not np.isfinite(centre).all():
        raise ValueError("Plane hypotheses must be finite")
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        raise ValueError("Plane normal must be nonzero")
    return normal / length, centre


def _plane_support(points: np.ndarray, normals: np.ndarray, plane: dict,
                   config: GeometryDiagnosticConfig) -> np.ndarray:
    normal, centre = _plane_frame(plane)
    offset = (points - centre) @ normal
    parallel = np.abs(normals @ normal) >= math.cos(
        math.radians(config.maximum_plane_normal_angle_deg)
    ) - 1e-10
    return points[parallel & (np.abs(offset) <= config.plane_support_band_m + 1e-10)]


def compare_plane_support(
    baseline_points: np.ndarray, baseline_normals: np.ndarray,
    candidate_points: np.ndarray, candidate_normals: np.ndarray,
    baseline_plane: dict, candidate_plane: dict, *,
    config: GeometryDiagnosticConfig | None = None,
) -> dict:
    """Verify projected support overlap and measure uncensored ROI thickness.

    Plane hypotheses locate broad bands only.  All normal-compatible points
    in the bands and the common projected support participate in quantiles;
    RANSAC's 15 mm inlier mask is never reused for thickness.  This projected
    ROI is NOT an RGB-D visibility mask and does not prove object identity.
    The reported thickness is the normal-coordinate spread of this support
    band; nearby real structures can contribute, so it is not established
    physical-plane thickness or a ground-truth reconstruction error.
    """
    from scipy.spatial import cKDTree

    config = config or GeometryDiagnosticConfig()
    bp, bn, _ = _valid_cloud(baseline_points, baseline_normals)
    cp, cn, _ = _valid_cloud(candidate_points, candidate_normals)
    normal_b, centre_b = _plane_frame(baseline_plane)
    normal_c, centre_c = _plane_frame(candidate_plane)
    normal_angle = math.degrees(math.acos(float(np.clip(
        abs(normal_b @ normal_c), 0.0, 1.0,
    ))))
    displacement = centre_c - centre_b
    centroid_distance = float(np.linalg.norm(displacement))
    normal_offset = max(abs(float(displacement @ normal_b)),
                        abs(float(displacement @ normal_c)))
    result = {
        "baseline_plane_id": str(baseline_plane.get("plane_id", "unknown")),
        "candidate_plane_id": str(candidate_plane.get("plane_id", "unknown")),
        "status": "inconclusive",
        "normal_angle_deg": normal_angle,
        "centroid_distance_m": centroid_distance,
        "symmetric_normal_offset_m": normal_offset,
        "common_rgbd_visibility_established": False,
    }
    if (normal_angle > config.maximum_plane_normal_angle_deg
            or normal_offset > config.maximum_plane_offset_m
            or centroid_distance > config.maximum_plane_centroid_distance_m):
        return {**result, "reason": "plane_pose_incompatible"}
    support_b = _plane_support(bp, bn, baseline_plane, config)
    support_c = _plane_support(cp, cn, candidate_plane, config)
    result.update(baseline_band_points=len(support_b), candidate_band_points=len(support_c))
    if min(len(support_b), len(support_c)) < config.minimum_shared_support_points:
        return {**result, "reason": "insufficient_plane_band_support"}
    # Both supports are projected into the SAME baseline plane.  Doing this
    # in 3-D avoids introducing an arbitrary in-plane world grid or basis.
    projection_b = support_b - np.outer((support_b - centre_b) @ normal_b, normal_b)
    projection_c = support_c - np.outer((support_c - centre_b) @ normal_b, normal_b)
    distance_b = cKDTree(projection_c).query(projection_b, k=1)[0]
    distance_c = cKDTree(projection_b).query(projection_c, k=1)[0]
    shared_b = distance_b <= config.projected_support_distance_m + 1e-10
    shared_c = distance_c <= config.projected_support_distance_m + 1e-10
    overlap_b, overlap_c = float(shared_b.mean()), float(shared_c.mean())
    result.update(
        baseline_projected_overlap_fraction=overlap_b,
        candidate_projected_overlap_fraction=overlap_c,
        baseline_shared_support_points=int(shared_b.sum()),
        candidate_shared_support_points=int(shared_c.sum()),
    )
    if (min(overlap_b, overlap_c) < config.minimum_projected_overlap_fraction
            or min(int(shared_b.sum()), int(shared_c.sum())) < config.minimum_shared_support_points):
        return {**result, "reason": "projected_support_does_not_overlap_sufficiently"}
    residual_b = (support_b[shared_b] - centre_b) @ normal_b
    residual_c = (support_c[shared_c] - centre_c) @ normal_c
    thickness_b = float(np.ptp(np.percentile(residual_b, [10, 90])))
    thickness_c = float(np.ptp(np.percentile(residual_c, [10, 90])))
    baseline_measurable = thickness_b >= config.minimum_thickness_for_relative_change_m
    return {
        **result,
        "status": "measured" if baseline_measurable else "inconclusive",
        "reason": ("projected_support_verified" if baseline_measurable
                   else "baseline_thickness_below_relative_resolution_floor"),
        "baseline_thickness_p90_p10_m": thickness_b,
        "candidate_thickness_p90_p10_m": thickness_c,
        "thickness_candidate_over_baseline": (
            thickness_c / thickness_b if baseline_measurable else None
        ),
    }


def compare_no_gt_geometry_v3(
    baseline_points: np.ndarray, baseline_normals: np.ndarray,
    candidate_points: np.ndarray, candidate_normals: np.ndarray, *,
    baseline_planes: list[dict], candidate_planes: list[dict],
    config: GeometryDiagnosticConfig | None = None,
) -> dict:
    """Return independent diagnostics with no trajectory acceptance authority."""
    config = config or GeometryDiagnosticConfig()
    before = candidate_layer_pairs(baseline_points, baseline_normals, config=config)
    after = candidate_layer_pairs(candidate_points, candidate_normals, config=config)
    bf = before["point_fraction_with_candidate_layer_pair"]
    cf = after["point_fraction_with_candidate_layer_pair"]
    layer_comparable = (
        before["status"] == after["status"] == "measured"
        and bf is not None and cf is not None
        and bf >= config.minimum_layer_fraction_for_relative_change
    )
    pairs = [compare_plane_support(
        baseline_points, baseline_normals, candidate_points, candidate_normals,
        left, right, config=config,
    ) for left in baseline_planes for right in candidate_planes]
    measured = [row for row in pairs if row["status"] == "measured"]
    measured.sort(key=lambda row: (
        -min(row["baseline_shared_support_points"], row["candidate_shared_support_points"]),
        row["normal_angle_deg"], row["baseline_plane_id"], row["candidate_plane_id"],
    ))
    # One-to-one assignment prevents reusing the exact same hypothesis.  A
    # detector can still emit different hypotheses for one physical support;
    # no aggregate scene score is formed from these potentially overlapping ROIs.
    selected, used_b, used_c = [], set(), set()
    for row in measured:
        b_id, c_id = row["baseline_plane_id"], row["candidate_plane_id"]
        if b_id not in used_b and c_id not in used_c:
            selected.append(row)
            used_b.add(b_id)
            used_c.add(c_id)
    return {
        "schema": "geometry_comparison.v3",
        "status": "measured" if layer_comparable and selected else "inconclusive",
        "diagnostic_only": True,
        "usable_for_promotion": False,
        "gt_consumed": False,
        "config": asdict(config),
        "candidate_layer_pairs": {
            "baseline": before, "candidate": after,
            "relative_change_status": "measured" if layer_comparable else "inconclusive",
            "candidate_over_baseline": cf / bf if layer_comparable else None,
        },
        "selected_plane_support_matches": selected,
        "all_plane_pair_diagnostics": pairs,
        "limitations": [
            "Close parallel surfaces can be genuine scene structure; no error truth is established.",
            "Layer fraction is point weighted and remains sensitive to surface sampling density.",
            "Supplied plane hypotheses may change with the detector or pose correction.",
            "Reported thickness is support-band normal spread and can include nearby real structures.",
            "Projected common support is not an RGB-D common-visibility mask.",
            "No GT-free improvement threshold has been independently calibrated.",
        ],
    }


def read_ply_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    vertices = PlyData.read(Path(path))["vertex"].data
    required = {"x", "y", "z", "nx", "ny", "nz"}
    if not required.issubset(set(vertices.dtype.names or ())):
        raise ValueError("V3 diagnostics require positions and actual surface normals")
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    normals = np.column_stack([vertices[name] for name in ("nx", "ny", "nz")])
    return points, normals
