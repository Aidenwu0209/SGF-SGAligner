"""No-GT reconstruction geometry metrics for Orbbec safety gates."""

from __future__ import annotations

import json
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
        results.append({
            "points": int(inliers.sum()),
            "inlier_ratio": float(np.mean(inliers)),
            "tilt_from_gravity_deg": math.degrees(math.acos(float(np.clip(abs(normal[1]), 0, 1)))),
            "thickness_p90_p10_m": float(np.percentile(residual, 90) - np.percentile(residual, 10)),
            "span_x_m": float(np.ptp(inlier_points[:, 0])),
            "span_z_m": float(np.ptp(inlier_points[:, 2])),
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
    input_vertices = int(len(points))
    points, normals = points[finite], normals[finite]
    if not len(points):
        raise ValueError("PLY contains no finite points")
    return {
        "schema": "no_gt_geometry_metrics.v1",
        "source": str(Path(path).resolve()),
        "input_vertices": input_vertices,
        "vertices": int(len(points)),
        "all_vertices_finite": bool(np.all(finite)),
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
