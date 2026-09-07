"""Evaluation-only trajectory metrics and paired bootstrap summaries."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    PoseRecord, load_manifest, load_trajectory, sha256_file,
    stable_json_sha256, validate_se3,
)
from .robust_backend import transform_distance


def trajectory_metrics(
    estimate: Sequence[PoseRecord], reference: Sequence[PoseRecord],
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

    return {
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


def build_scannet_common_observed_surface(
    scene: Path, manifest_path: Path, baseline_trajectory_path: Path,
    candidate_trajectory_path: Path, reference_surface: Path,
    output_path: Path, *, pixel_stride: int = 8,
    visibility_distance_m: float = 0.08,
) -> dict:
    """Cull the sealed reference surface to RGB-D support common to both arms.

    This function belongs to the evaluation process: it opens ScanNet GT poses
    and never feeds any result back into inference.
    """
    import cv2
    import open3d as o3d
    from scipy.spatial import cKDTree

    if pixel_stride < 1 or visibility_distance_m <= 0.0:
        raise ValueError("invalid common-observation sampling config")
    scene = Path(scene).resolve()
    manifest = load_manifest(manifest_path)
    if manifest.dataset != "scannet":
        raise ValueError("common observed surface currently supports ScanNet")
    baseline, _ = load_trajectory(baseline_trajectory_path)
    candidate, _ = load_trajectory(candidate_trajectory_path)
    common_ids = sorted(
        {row.frame_id for row in baseline}
        & {row.frame_id for row in candidate}
        & {frame.frame_id for frame in manifest.frames}
    )
    if not common_ids:
        raise ValueError("baseline/candidate have no common admitted frames")
    frame_by_id = {frame.frame_id: frame for frame in manifest.frames}
    observed = []
    gt_pose_hashes = []
    for frame_id in common_ids:
        frame = frame_by_id[frame_id]
        pose_path = scene / "pose" / f"{frame_id}.txt"
        try:
            truth = validate_se3(
                np.loadtxt(pose_path), f"ScanNet GT frame {frame_id}",
            )
        except ValueError:
            continue
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(f"invalid ScanNet depth frame {frame_id}")
        vv, uu = np.mgrid[
            0:depth.shape[0]:pixel_stride,
            0:depth.shape[1]:pixel_stride,
        ]
        z = depth[::pixel_stride, ::pixel_stride].astype(np.float64) / manifest.depth_scale
        valid = np.isfinite(z) & (z >= 0.30) & (z <= 4.50)
        fx, fy, cx, cy = frame.intrinsics
        camera = np.column_stack([
            (uu[valid] - cx) * z[valid] / fx,
            (vv[valid] - cy) * z[valid] / fy,
            z[valid],
        ])
        world = camera @ truth[:3, :3].T + truth[:3, 3]
        observed.append(world)
        gt_pose_hashes.append({
            "frame_id": frame_id,
            "pose_sha256": sha256_file(pose_path),
            "depth_sha256": sha256_file(frame.depth_path),
        })
    if not observed:
        raise ValueError("common admitted frames have no finite GT poses")
    observed_points = np.concatenate(observed, axis=0)
    observed_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(observed_points),
    ).voxel_down_sample(0.03)
    observed_points = np.asarray(observed_cloud.points, dtype=np.float64)
    reference = o3d.io.read_point_cloud(str(reference_surface))
    if not reference.has_points():
        mesh = o3d.io.read_triangle_mesh(str(reference_surface))
        reference = o3d.geometry.PointCloud(mesh.vertices)
    reference = reference.voxel_down_sample(0.02)
    reference_points = np.asarray(reference.points, dtype=np.float64)
    distances = cKDTree(observed_points).query(
        reference_points, k=1, workers=-1,
    )[0]
    visible_mask = distances <= visibility_distance_m
    visible_points = reference_points[visible_mask]
    if len(visible_points) < 500:
        raise RuntimeError(
            f"common visibility mask retained only {len(visible_points)} points"
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(visible_points),
    )
    if not o3d.io.write_point_cloud(str(output_path), output_cloud, write_ascii=False):
        raise RuntimeError("failed to write common observed ScanNet surface")
    mask_hash = hashlib.sha256(
        np.ascontiguousarray(visible_mask, dtype=np.uint8).tobytes(),
    ).hexdigest()
    return {
        "schema": "scannet_common_observed_surface.v1",
        "surface": str(output_path),
        "surface_sha256": sha256_file(output_path),
        "reference_surface": str(Path(reference_surface).resolve()),
        "reference_surface_sha256": sha256_file(reference_surface),
        "admitted_frame_count": len(common_ids),
        "evaluable_frame_count": len(gt_pose_hashes),
        "admitted_frame_sha256": stable_json_sha256(common_ids),
        "evaluation_input_sha256": stable_json_sha256(gt_pose_hashes),
        "common_visibility_mask_sha256": mask_hash,
        "reference_point_count": len(reference_points),
        "visible_reference_point_count": len(visible_points),
        "pixel_stride": pixel_stride,
        "visibility_distance_m": visibility_distance_m,
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
