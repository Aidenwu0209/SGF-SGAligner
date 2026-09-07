"""No-GT reconstruction geometry metrics for Orbbec safety gates."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def _ply_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    return np.column_stack([
        vertex[name] for name in ("x", "y", "z")
    ]).astype(np.float32)


def _plane_metrics(points: np.ndarray, normals: np.ndarray) -> list[dict]:
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    length = np.linalg.norm(normals, axis=1)
    horizontal = (
        np.isfinite(length)
        & (length > 0.5)
        & (np.abs(normals[:, 1]) / np.maximum(length, 1e-9)
           >= math.cos(math.radians(25.0)))
    )
    candidates = points[horizontal]
    if len(candidates) < 500:
        return []
    bins = np.floor(candidates[:, 1] / 0.02).astype(np.int32)
    unique, counts = np.unique(bins, return_counts=True)
    selected_heights, results = [], []
    rng = np.random.default_rng(20260901)
    for index in np.argsort(counts)[::-1]:
        height = (float(unique[index]) + 0.5) * 0.02
        if any(abs(height - prior) < 0.08 for prior in selected_heights):
            continue
        selected_heights.append(height)
        band = candidates[np.abs(candidates[:, 1] - height) <= 0.045]
        if len(band) < 500:
            continue
        if len(band) > 50_000:
            band = band[rng.choice(len(band), 50_000, replace=False)]
        model = RANSACRegressor(
            estimator=LinearRegression(), min_samples=200,
            residual_threshold=0.015, max_trials=100,
            random_state=20260901,
        )
        model.fit(band[:, (0, 2)], band[:, 1])
        inliers = model.inlier_mask_
        if inliers is None or int(inliers.sum()) < 300:
            continue
        coefficients = model.estimator_.coef_
        normal = np.array([-coefficients[0], 1.0, -coefficients[1]])
        normal /= np.linalg.norm(normal)
        inlier_points = band[inliers]
        residual = inlier_points[:, 1] - model.predict(inlier_points[:, (0, 2)])
        centroid = np.mean(inlier_points, axis=0)
        plane_key = np.r_[np.round(normal, 3), np.round(centroid, 2)]
        results.append({
            "plane_id": "plane_" + hashlib.sha256(
                np.ascontiguousarray(plane_key, dtype=np.float64).tobytes(),
            ).hexdigest()[:12],
            "points": int(inliers.sum()),
            "inlier_ratio": float(np.mean(inliers)),
            "tilt_from_gravity_deg": math.degrees(math.acos(float(np.clip(abs(normal[1]), 0, 1)))),
            "thickness_p90_p10_m": float(np.percentile(residual, 90) - np.percentile(residual, 10)),
            "span_x_m": float(np.ptp(inlier_points[:, 0])),
            "span_z_m": float(np.ptp(inlier_points[:, 2])),
            "normal": normal.astype(float).tolist(),
            "centroid_m": centroid.astype(float).tolist(),
        })
        if len(results) >= 4:
            break
    return sorted(results, key=lambda item: item["points"], reverse=True)


def _layer_conflict(points: np.ndarray) -> float:
    cell = np.floor(points[:, (0, 2)] / 0.03).astype(np.int32)
    height = np.floor(points[:, 1] / 0.02).astype(np.int32)
    order = np.lexsort((height, cell[:, 1], cell[:, 0]))
    cell, height = cell[order], height[order]
    unique, starts = np.unique(cell, axis=0, return_index=True)
    if not len(unique):
        return 0.0
    ends = np.r_[starts[1:], len(height)]
    conflicts = 0
    for start, stop in zip(starts, ends):
        levels = np.unique(height[start:stop])
        separation = levels[:, None] - levels[None, :]
        conflicts += int(np.any((separation >= 2) & (separation <= 6)))
    return conflicts / len(unique)


def ply_geometry_metrics(path: Path) -> dict:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float32)
    names = set(vertex.dtype.names or ())
    normals = (
        np.column_stack([vertex[name] for name in ("nx", "ny", "nz")]).astype(np.float32)
        if {"nx", "ny", "nz"} <= names else np.zeros_like(points)
    )
    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    points, normals = points[finite], normals[finite]
    if not len(points):
        raise ValueError("PLY contains no finite points")
    return {
        "schema": "no_gt_geometry_metrics.v1",
        "source": str(Path(path).resolve()),
        "vertices": int(len(points)),
        "occupied_voxels_2cm": int(len(np.unique(np.floor(points / 0.02).astype(np.int32), axis=0))),
        "bbox_extent_m": np.ptp(points, axis=0).astype(float).tolist(),
        "robust_extent_p99_p01_m": (
            np.percentile(points, 99, axis=0)
            - np.percentile(points, 1, axis=0)
        ).astype(float).tolist(),
        "near_parallel_layer_conflict_ratio": _layer_conflict(points),
        "horizontal_planes": _plane_metrics(points, normals),
        "gt_consumed": False,
    }


def render_fixed_comparison_views(
    baseline_path: Path, candidate_path: Path, output_path: Path,
    *, maximum_points: int = 80_000,
) -> dict:
    """Render shared-axis top/side views for reproducible human inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clouds = [_ply_points(Path(baseline_path)), _ply_points(Path(candidate_path))]
    prepared = []
    for points in clouds:
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            raise ValueError("comparison cloud contains no finite points")
        if len(points) > maximum_points:
            indices = np.linspace(0, len(points) - 1, maximum_points, dtype=np.int64)
            points = points[indices]
        prepared.append(points)
    combined = np.concatenate(prepared, axis=0)
    bounds = {
        "x": [float(np.percentile(combined[:, 0], 0.5)), float(np.percentile(combined[:, 0], 99.5))],
        "y": [float(np.percentile(combined[:, 1], 0.5)), float(np.percentile(combined[:, 1], 99.5))],
        "z": [float(np.percentile(combined[:, 2], 0.5)), float(np.percentile(combined[:, 2], 99.5))],
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=160)
    for column, (label, points) in enumerate(zip(("baseline", "candidate"), prepared)):
        axes[0, column].scatter(points[:, 0], points[:, 2], c=points[:, 1], s=0.12, cmap="viridis")
        axes[0, column].set(xlim=bounds["x"], ylim=bounds["z"], title=f"{label} top (x-z)", xlabel="x [m]", ylabel="z [m]")
        axes[1, column].scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.12, cmap="plasma")
        axes[1, column].set(xlim=bounds["x"], ylim=bounds["y"], title=f"{label} side (x-y)", xlabel="x [m]", ylabel="y [m]")
        for row in (0, 1):
            axes[row, column].set_aspect("equal", adjustable="box")
            axes[row, column].grid(alpha=0.15)
    figure.suptitle("Fixed-view pose backend A/B (shared axes)")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return {
        "schema": "fixed_ply_comparison_view.v1",
        "path": str(output_path.resolve()),
        "baseline": str(Path(baseline_path).resolve()),
        "candidate": str(Path(candidate_path).resolve()),
        "shared_bounds_m": bounds,
        "maximum_points_per_cloud": maximum_points,
    }


def compare_no_gt_geometry(baseline: dict, candidate: dict) -> dict:
    before = float(baseline["near_parallel_layer_conflict_ratio"])
    after = float(candidate["near_parallel_layer_conflict_ratio"])
    baseline_plane = max(
        baseline.get("horizontal_planes", []),
        key=lambda row: int(row.get("points", 0)), default=None,
    )
    candidate_plane = max(
        candidate.get("horizontal_planes", []),
        key=lambda row: int(row.get("points", 0)), default=None,
    )
    point_ratio = candidate["vertices"] / max(baseline["vertices"], 1)
    bbox_ratio = (
        np.asarray(candidate["bbox_extent_m"], dtype=float)
        / np.maximum(np.asarray(baseline["bbox_extent_m"], dtype=float), 1e-9)
    )
    conflict_improvement = (before - after) / max(before, 1e-12)
    gates = {
        "layer_conflict_improves_10pct": conflict_improvement >= 0.10,
        "layer_conflict_not_worse_10pct": after <= before * 1.10,
        "point_count_at_least_80pct": point_ratio >= 0.80,
        "every_bbox_axis_at_least_80pct": bool(np.all(bbox_ratio >= 0.80)),
    }
    thickness_improvement = None
    tilt_delta = None
    if baseline_plane and candidate_plane:
        thickness_before = float(baseline_plane["thickness_p90_p10_m"])
        thickness_after = float(candidate_plane["thickness_p90_p10_m"])
        thickness_improvement = (
            thickness_before - thickness_after
        ) / max(thickness_before, 1e-12)
        tilt_delta = float(candidate_plane["tilt_from_gravity_deg"] - baseline_plane["tilt_from_gravity_deg"])
        gates["ground_tilt_regression_at_most_2deg"] = tilt_delta <= 2.0
    else:
        gates["ground_tilt_regression_at_most_2deg"] = False
    return {
        "schema": "no_gt_geometry_comparison.v1",
        "passes_scene_safety": all(value for key, value in gates.items() if key != "layer_conflict_improves_10pct"),
        "passes_scene_improvement": all(gates.values()),
        "gates": gates,
        "layer_conflict_improvement_fraction": conflict_improvement,
        "dominant_plane_thickness_improvement_fraction": thickness_improvement,
        "ground_tilt_delta_deg": tilt_delta,
        "point_count_ratio": point_ratio,
        "bbox_axis_ratios": bbox_ratio.tolist(),
        "gt_consumed": False,
    }


def _match_physical_plane(
    baseline_planes: list[dict], candidate_planes: list[dict],
) -> tuple[dict | None, dict | None, dict | None]:
    if not baseline_planes or not candidate_planes:
        return None, None, None
    candidates = []
    for left in baseline_planes:
        left_normal = np.asarray(left.get("normal", [0.0, 1.0, 0.0]), dtype=float)
        left_centroid = np.asarray(left.get("centroid_m", [0.0, 0.0, 0.0]), dtype=float)
        for right in candidate_planes:
            right_normal = np.asarray(right.get("normal", [0.0, 1.0, 0.0]), dtype=float)
            right_centroid = np.asarray(right.get("centroid_m", [0.0, 0.0, 0.0]), dtype=float)
            normal_angle = math.degrees(math.acos(float(np.clip(
                abs(np.dot(left_normal, right_normal))
                / max(np.linalg.norm(left_normal) * np.linalg.norm(right_normal), 1e-12),
                0.0, 1.0,
            ))))
            height_delta = abs(float(left_centroid[1] - right_centroid[1]))
            horizontal_delta = float(np.linalg.norm(
                left_centroid[(0, 2),] - right_centroid[(0, 2),],
            ))
            score = normal_angle / 10.0 + height_delta / 0.10 + horizontal_delta / 2.0
            candidates.append((
                score, -min(int(left.get("points", 0)), int(right.get("points", 0))),
                str(left.get("plane_id", "")), str(right.get("plane_id", "")),
                left, right, normal_angle, height_delta, horizontal_delta,
            ))
    candidates.sort(key=lambda row: row[:4])
    best = candidates[0]
    if best[6] > 15.0 or best[7] > 0.20:
        return None, None, None
    match_id = "matched_plane_" + hashlib.sha256(
        f"{best[2]}|{best[3]}".encode(),
    ).hexdigest()[:12]
    return best[4], best[5], {
        "matched_plane_id": match_id,
        "baseline_plane_id": best[2],
        "candidate_plane_id": best[3],
        "normal_angle_deg": best[6],
        "centroid_height_delta_m": best[7],
        "centroid_horizontal_delta_m": best[8],
    }


def compare_no_gt_geometry_v2(
    baseline: dict, candidate: dict, *,
    admitted_frame_sha256: str | None = None,
    common_visibility_mask_sha256: str | None = None,
    minimum_occupied_voxel_ratio: float = 0.80,
    minimum_each_robust_extent_ratio: float = 0.85,
    maximum_matched_plane_tilt_regression_deg: float = 2.0,
    maximum_thickness_ratio: float = 1.10,
    maximum_layer_conflict_ratio: float = 1.10,
) -> dict:
    baseline_extent = np.asarray(
        baseline.get("robust_extent_p99_p01_m", baseline["bbox_extent_m"]),
        dtype=float,
    )
    candidate_extent = np.asarray(
        candidate.get("robust_extent_p99_p01_m", candidate["bbox_extent_m"]),
        dtype=float,
    )
    extent_ratio = candidate_extent / np.maximum(baseline_extent, 1e-9)
    voxel_ratio = float(candidate["occupied_voxels_2cm"]) / max(
        int(baseline["occupied_voxels_2cm"]), 1,
    )
    before_conflict = float(baseline["near_parallel_layer_conflict_ratio"])
    after_conflict = float(candidate["near_parallel_layer_conflict_ratio"])
    left, right, plane_match = _match_physical_plane(
        list(baseline.get("horizontal_planes", [])),
        list(candidate.get("horizontal_planes", [])),
    )
    tilt_delta = None
    thickness_ratio = None
    if left is not None and right is not None:
        tilt_delta = float(
            right["tilt_from_gravity_deg"] - left["tilt_from_gravity_deg"]
        )
        thickness_ratio = float(right["thickness_p90_p10_m"]) / max(
            float(left["thickness_p90_p10_m"]), 1e-9,
        )
    gates = {
        "occupied_voxels_above_minimum_ratio": (
            voxel_ratio >= minimum_occupied_voxel_ratio
        ),
        "all_robust_extents_above_minimum_ratio": bool(np.all(
            extent_ratio >= minimum_each_robust_extent_ratio
        )),
        "same_physical_plane_matched": plane_match is not None,
        "matched_plane_tilt_regression_within_limit": (
            tilt_delta is not None
            and tilt_delta <= maximum_matched_plane_tilt_regression_deg
        ),
        "matched_plane_thickness_within_ratio": (
            thickness_ratio is not None
            and thickness_ratio <= maximum_thickness_ratio
        ),
        "layer_conflict_within_ratio": (
            after_conflict <= before_conflict * maximum_layer_conflict_ratio + 1e-12
        ),
    }
    improvement_gates = {
        "matched_plane_thickness_improves_10pct": (
            thickness_ratio is not None and thickness_ratio <= 0.90
        ),
        "layer_conflict_improves_10pct": (
            after_conflict <= before_conflict * 0.90 + 1e-12
        ),
    }
    passes_safety = all(gates.values())
    return {
        "schema": "geometry_comparison.v2",
        "passes_scene_safety": passes_safety,
        "passes_scene_improvement": (
            passes_safety and all(improvement_gates.values())
        ),
        "gates": gates,
        "improvement_gates": improvement_gates,
        "occupied_voxel_ratio": voxel_ratio,
        "robust_extent_axis_ratios": extent_ratio.tolist(),
        "matched_plane": plane_match,
        "matched_plane_tilt_delta_deg": tilt_delta,
        "matched_plane_thickness_ratio": thickness_ratio,
        "layer_conflict_ratio": {
            "baseline": before_conflict,
            "candidate": after_conflict,
            "candidate_over_baseline": after_conflict / max(before_conflict, 1e-12),
        },
        "admitted_frame_sha256": admitted_frame_sha256,
        "common_visibility_mask_sha256": common_visibility_mask_sha256,
        "safety_thresholds": {
            "minimum_occupied_voxel_ratio": minimum_occupied_voxel_ratio,
            "minimum_each_robust_extent_ratio": minimum_each_robust_extent_ratio,
            "maximum_matched_plane_tilt_regression_deg": (
                maximum_matched_plane_tilt_regression_deg
            ),
            "maximum_thickness_ratio": maximum_thickness_ratio,
            "maximum_layer_conflict_ratio": maximum_layer_conflict_ratio,
        },
        "gt_consumed": False,
    }
