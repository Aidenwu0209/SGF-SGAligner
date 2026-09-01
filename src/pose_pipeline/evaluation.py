"""Evaluation-only trajectory metrics and paired bootstrap summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import PoseRecord, load_trajectory, sha256_file, validate_se3
from .robust_backend import transform_distance


def trajectory_metrics(
    estimate: Sequence[PoseRecord], reference: Sequence[PoseRecord], *,
    _include_sim3: bool = True,
) -> dict:
    estimate_by_id = {row.frame_id: row for row in estimate}
    reference_by_id = {row.frame_id: row for row in reference}
    if not set(reference_by_id) <= set(estimate_by_id):
        raise ValueError("reference contains frames missing from estimate")
    frame_ids = sorted(reference_by_id)
    if not frame_ids:
        raise ValueError("estimate/reference have no evaluable frames")
    absolute_rotation, absolute_translation = [], []
    relative_rotation, relative_translation = [], []
    for frame_id in frame_ids:
        rotation, translation = transform_distance(
            estimate_by_id[frame_id].t_world_camera,
            reference_by_id[frame_id].t_world_camera,
        )
        absolute_rotation.append(rotation)
        absolute_translation.append(translation)
    for left, right in zip(frame_ids, frame_ids[1:]):
        estimate_delta = np.linalg.inv(
            estimate_by_id[left].t_world_camera,
        ) @ estimate_by_id[right].t_world_camera
        reference_delta = np.linalg.inv(
            reference_by_id[left].t_world_camera,
        ) @ reference_by_id[right].t_world_camera
        rotation, translation = transform_distance(estimate_delta, reference_delta)
        relative_rotation.append(rotation)
        relative_translation.append(translation)

    def describe(values: Sequence[float]) -> dict:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {
                "count": 0,
                "available": False,
                "median": None,
                "mean": None,
                "rmse": None,
                "p95": None,
                "max": None,
            }
        return {
            "count": int(len(array)),
            "available": True,
            "median": float(np.median(array)),
            "mean": float(np.mean(array)),
            "rmse": float(np.sqrt(np.mean(array ** 2))),
            "p95": float(np.percentile(array, 95)),
            "max": float(np.max(array)),
        }

    metric_result = {
        "schema": "pose_trajectory_evaluation.v1",
        "frame_count": len(frame_ids),
        "estimate_frame_count": len(estimate_by_id),
        "evaluation_coverage": len(frame_ids) / len(estimate_by_id),
        "excluded_estimate_frame_count": len(estimate_by_id) - len(frame_ids),
        "absolute_translation_m": describe(absolute_translation),
        "absolute_rotation_deg": describe(absolute_rotation),
        "relative_translation_m": describe(relative_translation),
        "relative_rotation_deg": describe(relative_rotation),
        "gt_role": "evaluation_only",
    }
    if not _include_sim3:
        return metric_result
    if len(frame_ids) >= 2:
        estimate_xyz = np.stack([
            estimate_by_id[frame_id].t_world_camera[:3, 3] for frame_id in frame_ids
        ])
        reference_xyz = np.stack([
            reference_by_id[frame_id].t_world_camera[:3, 3] for frame_id in frame_ids
        ])
        source_center = estimate_xyz.mean(axis=0)
        target_center = reference_xyz.mean(axis=0)
        source_zero = estimate_xyz - source_center
        target_zero = reference_xyz - target_center
        covariance = target_zero.T @ source_zero / len(frame_ids)
        left, singular, right_t = np.linalg.svd(covariance)
        sign = np.ones(3)
        if np.linalg.det(left @ right_t) < 0:
            sign[-1] = -1.0
        alignment_rotation = left @ np.diag(sign) @ right_t
        se3_translation = target_center - alignment_rotation @ source_center

        def aligned_metrics(scale: float, translation: np.ndarray) -> dict:
            aligned = []
            for frame_id in frame_ids:
                source_pose = estimate_by_id[frame_id].t_world_camera
                pose = np.eye(4)
                pose[:3, :3] = alignment_rotation @ source_pose[:3, :3]
                pose[:3, 3] = scale * alignment_rotation @ source_pose[:3, 3] + translation
                aligned.append(PoseRecord(
                    frame_id=frame_id,
                    timestamp_us=estimate_by_id[frame_id].timestamp_us,
                    t_world_camera=validate_se3(pose),
                    source="evaluation_only_aligned",
                ))
            return trajectory_metrics(
                aligned, [reference_by_id[value] for value in frame_ids],
                _include_sim3=False,
            )

        se3_aligned = aligned_metrics(1.0, se3_translation)
        metric_se3 = {
            "available": True,
            "scale_fixed_to_one": True,
            "global_rotation": alignment_rotation.tolist(),
            "global_translation": se3_translation.tolist(),
            "absolute_translation_m": se3_aligned["absolute_translation_m"],
            "absolute_rotation_deg": se3_aligned["absolute_rotation_deg"],
            "relative_translation_m": se3_aligned["relative_translation_m"],
            "relative_rotation_deg": se3_aligned["relative_rotation_deg"],
        }
        variance = float(np.mean(np.sum(source_zero ** 2, axis=1)))
        if variance <= 1e-12:
            sim3 = {
                "available": False,
                "reason": "estimate_translation_variance_too_small",
            }
        else:
            scale = float(np.sum(singular * sign) / variance)
            translation = target_center - scale * alignment_rotation @ source_center
            sim3_aligned = aligned_metrics(scale, translation)
            sim3 = {
                "available": True,
                "scale_reference_units_per_estimate_unit": scale,
                "global_rotation": alignment_rotation.tolist(),
                "global_translation": translation.tolist(),
                "absolute_translation_m": sim3_aligned["absolute_translation_m"],
                "absolute_rotation_deg": sim3_aligned["absolute_rotation_deg"],
                "relative_translation_m": sim3_aligned["relative_translation_m"],
                "relative_rotation_deg": sim3_aligned["relative_rotation_deg"],
            }
    else:
        metric_se3 = {"available": False, "reason": "at_least_two_frames_required"}
        sim3 = {"available": False, "reason": "at_least_two_frames_required"}
    metric_result["metric_se3"] = metric_se3
    metric_result["sim3_alignment"] = sim3
    metric_result["metric_conclusion_must_not_use_sim3"] = True
    return metric_result


def evaluate_trajectory_files(
    estimate_path: Path, reference_path: Path, output_path: Path,
) -> dict:
    estimate, _ = load_trajectory(estimate_path)
    reference, _ = load_trajectory(reference_path)
    value = trajectory_metrics(estimate, reference)
    value["inputs"] = {
        "estimate": {"path": str(Path(estimate_path).resolve()), "sha256": sha256_file(estimate_path)},
        "reference": {"path": str(Path(reference_path).resolve()), "sha256": sha256_file(reference_path)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return value


def paired_bootstrap_improvement(
    baseline: Sequence[float], candidate: Sequence[float], *,
    samples: int = 10_000, seed: int = 42,
) -> dict:
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("paired bootstrap inputs must be equal non-empty vectors")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired bootstrap inputs must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    before = baseline[indices].mean(axis=1)
    after = candidate[indices].mean(axis=1)
    improvement = (before - after) / np.maximum(before, 1e-12)
    observed = float((baseline.mean() - candidate.mean()) / max(baseline.mean(), 1e-12))
    return {
        "observed_fraction": observed,
        "ci95_fraction": [float(np.percentile(improvement, 2.5)), float(np.percentile(improvement, 97.5))],
        "samples": samples,
        "seed": seed,
        "passes_10pct_and_positive_ci": bool(
            observed >= 0.10 and np.percentile(improvement, 2.5) > 0.0
        ),
    }


def reconstruction_surface_metrics(
    estimate_cloud: Path, reference_surface: Path,
    estimate_world_to_dataset_world: object,
    *, voxel_m: float = 0.03, threshold_m: float = 0.05,
) -> dict:
    """Evaluation-only symmetric surface distances in the dataset frame."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    alignment = validate_se3(
        estimate_world_to_dataset_world, "reconstruction evaluation alignment",
    )
    estimate = o3d.io.read_point_cloud(str(estimate_cloud))
    reference = o3d.io.read_point_cloud(str(reference_surface))
    if not estimate.has_points() or not reference.has_points():
        raise ValueError("estimate/reference reconstruction cloud is empty")
    estimate.transform(alignment)
    estimate = estimate.voxel_down_sample(voxel_m)
    reference = reference.voxel_down_sample(voxel_m)
    estimate_points = np.asarray(estimate.points, dtype=np.float64)
    reference_points = np.asarray(reference.points, dtype=np.float64)
    estimate_to_reference = cKDTree(reference_points).query(
        estimate_points, k=1, workers=-1,
    )[0]
    reference_to_estimate = cKDTree(estimate_points).query(
        reference_points, k=1, workers=-1,
    )[0]
    precision = float(np.mean(estimate_to_reference <= threshold_m))
    recall = float(np.mean(reference_to_estimate <= threshold_m))
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "schema": "reconstruction_surface_evaluation.v1",
        "estimate_points": len(estimate_points),
        "reference_points": len(reference_points),
        "voxel_m": voxel_m,
        "threshold_m": threshold_m,
        "estimate_to_reference_mean_m": float(np.mean(estimate_to_reference)),
        "reference_to_estimate_mean_m": float(np.mean(reference_to_estimate)),
        "symmetric_chamfer_mean_m": float(
            0.5 * (np.mean(estimate_to_reference) + np.mean(reference_to_estimate))
        ),
        "symmetric_chamfer_rmse_m": float(np.sqrt(
            0.5 * (
                np.mean(estimate_to_reference ** 2)
                + np.mean(reference_to_estimate ** 2)
            )
        )),
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "gt_role": "evaluation_only",
    }


def scannet_reference_trajectory(
    scene: Path, frame_ids: Sequence[int], timestamps_us: Mapping[int, int],
) -> list[PoseRecord]:
    scene = Path(scene).resolve()
    selected_ids, poses = [], []
    for frame_id in frame_ids:
        try:
            matrix = validate_se3(
                np.loadtxt(scene / "pose" / f"{frame_id}.txt"),
                f"ScanNet GT frame {frame_id}",
            )
        except ValueError:
            continue
        selected_ids.append(frame_id)
        poses.append(matrix)
    if not poses:
        raise ValueError("ScanNet sequence has no finite evaluation poses")
    origin = poses[0]
    return [PoseRecord(
        frame_id=frame_id,
        timestamp_us=int(timestamps_us[frame_id]),
        t_world_camera=validate_se3(np.linalg.inv(origin) @ pose),
        valid=True,
        source="ScanNet_GT_evaluation_only",
    ) for frame_id, pose in zip(selected_ids, poses)]


def scan3r_reference_trajectory(
    sequence: Path, frame_ids: Sequence[int], timestamps_us: Mapping[int, int],
    *, input_rotated_ccw: bool = True,
) -> list[PoseRecord]:
    """Load 3RScan camera poses in the evaluation-only process.

    The public adapter rotates the RGB-D images counter-clockwise by default.
    The fixed post-rotation below changes only the camera coordinate basis; it
    does not use any estimated or ground-truth motion during inference.
    """
    sequence = Path(sequence).resolve()
    camera_rotated_to_original = np.eye(4, dtype=np.float64)
    if input_rotated_ccw:
        camera_rotated_to_original[:3, :3] = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
    selected_ids, poses = [], []
    for frame_id in frame_ids:
        try:
            matrix = validate_se3(
                np.loadtxt(sequence / f"frame-{frame_id:06d}.pose.txt"),
                f"3RScan GT frame {frame_id}",
            )
        except ValueError:
            continue
        selected_ids.append(frame_id)
        poses.append(validate_se3(matrix @ camera_rotated_to_original))
    if not poses:
        raise ValueError("3RScan sequence has no finite evaluation poses")
    origin = poses[0]
    return [PoseRecord(
        frame_id=frame_id,
        timestamp_us=int(timestamps_us[frame_id]),
        t_world_camera=validate_se3(np.linalg.inv(origin) @ pose),
        valid=True,
        source="3RScan_GT_evaluation_only",
    ) for frame_id, pose in zip(selected_ids, poses)]
