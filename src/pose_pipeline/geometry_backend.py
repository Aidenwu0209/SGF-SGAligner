"""Geometry bootstrap for sequence-level sparse submap validation.

SGAligner point correspondences use :mod:`robust_backend` directly.  This
module supplies a separately labelled FPFH provider so sequence experiments
can test the pose graph and refusion chain before semantic subgraphs exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .robust_backend import (
    RobustPoseConfig,
    decide_registration_v2,
    generate_hypotheses,
    select_cross_solver_consensus,
    spatial_support,
    transform_distance,
    transform_points,
)


@dataclass(frozen=True)
class GeometryBootstrapConfig:
    voxel_m: float = 0.08
    normal_radius_m: float = 0.20
    feature_radius_m: float = 0.40
    feature_ratio: float = 0.97
    maximum_correspondences: int = 700
    icp_distance_m: float = 0.15
    verification_distance_m: float = 0.10


def _cloud_and_fpfh(points: np.ndarray, config: GeometryBootstrapConfig):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(config.voxel_m)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=config.normal_radius_m, max_nn=40,
    ))
    cloud.normalize_normals()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=config.feature_radius_m, max_nn=100,
        ),
    )
    xyz = np.ascontiguousarray(np.asarray(cloud.points), dtype=np.float64)
    descriptors = np.ascontiguousarray(np.asarray(feature.data).T, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(descriptors).all(axis=1)
    return xyz[finite], descriptors[finite]


def fpfh_correspondences(
    source: np.ndarray,
    reference: np.ndarray,
    config: GeometryBootstrapConfig = GeometryBootstrapConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    source_xyz, source_features = _cloud_and_fpfh(source, config)
    reference_xyz, reference_features = _cloud_and_fpfh(reference, config)
    if len(source_features) < 3 or len(reference_features) < 3:
        raise ValueError("insufficient FPFH features")
    tree = cKDTree(reference_features)
    distances, indices = tree.query(source_features, k=2, workers=-1)
    reverse = cKDTree(source_features).query(
        reference_features, k=1, workers=-1,
    )[1]
    source_indices = np.arange(len(source_features), dtype=np.int64)
    reference_indices = indices[:, 0].astype(np.int64)
    ratio = distances[:, 0] / np.maximum(distances[:, 1], 1e-12)
    keep = (
        (ratio <= config.feature_ratio)
        & (reverse[reference_indices] == source_indices)
    )
    source_indices, reference_indices = source_indices[keep], reference_indices[keep]
    scores = distances[keep, 0]
    order = np.lexsort((reference_indices, source_indices, scores))
    order = order[:config.maximum_correspondences]
    return (
        source_xyz,
        reference_xyz,
        np.ascontiguousarray(source_xyz[source_indices[order]]),
        np.ascontiguousarray(reference_xyz[reference_indices[order]]),
    )


def _icp(
    source: np.ndarray, reference: np.ndarray, initial: np.ndarray,
    distance_m: float,
) -> np.ndarray:
    import open3d as o3d

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    reference_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=distance_m * 2.0, max_nn=40,
    ))
    reference_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=distance_m * 2.0, max_nn=40,
    ))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, reference_cloud, distance_m, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def dense_verification(
    source: np.ndarray, reference: np.ndarray, transform: np.ndarray,
    distance_m: float,
) -> dict[str, float]:
    from scipy.spatial import cKDTree

    moved = transform_points(source, transform)
    forward = cKDTree(reference).query(moved, k=1, workers=-1)[0]
    reverse = cKDTree(moved).query(reference, k=1, workers=-1)[0]
    forward_overlap = float(np.mean(forward <= distance_m))
    reverse_overlap = float(np.mean(reverse <= distance_m))
    inside = np.r_[forward[forward <= distance_m], reverse[reverse <= distance_m]]
    return {
        "forward_overlap": forward_overlap,
        "reverse_overlap": reverse_overlap,
        "minimum_overlap": min(forward_overlap, reverse_overlap),
        "trimmed_rmse_m": float(np.sqrt(np.mean(inside ** 2))) if len(inside) else 1_000_000.0,
    }


def _one_direction(
    source: np.ndarray,
    reference: np.ndarray,
    robust_config: RobustPoseConfig,
    geometry_config: GeometryBootstrapConfig,
) -> dict[str, Any]:
    source_xyz, reference_xyz, source_corr, reference_corr = fpfh_correspondences(
        source, reference, geometry_config,
    )
    hypothesis_set = generate_hypotheses(
        source_corr, reference_corr, robust_config,
    )
    consensus = select_cross_solver_consensus(
        hypothesis_set["hypotheses"], robust_config,
    )
    result = {
        "provider": "geometry_bootstrap_fpfh",
        "hypothesis_set": hypothesis_set,
        "consensus": consensus,
        "accepted": False,
    }
    if consensus["accepted"] is not True:
        result["reason"] = consensus["reason"]
        return result
    initial = np.asarray(consensus["selected_transform"], dtype=np.float64)
    refined = _icp(
        source_xyz, reference_xyz, initial, geometry_config.icp_distance_m,
    )
    update_rotation, update_translation = transform_distance(initial, refined)
    verification = dense_verification(
        source_xyz, reference_xyz, refined,
        geometry_config.verification_distance_m,
    )
    extent, second = spatial_support(source_corr)
    result.update({
        "accepted": True,
        "reason": "cross_solver_consensus_and_icp",
        "transform": refined.tolist(),
        "icp_update_rotation_deg": update_rotation,
        "icp_update_translation_m": update_translation,
        "verification": verification,
        "spatial_extent_m": extent,
        "spatial_second_axis_m": second,
        "correspondence_count": len(source_corr),
    })
    return result


def register_submaps_bidirectional(
    source: np.ndarray,
    reference: np.ndarray,
    robust_config: RobustPoseConfig = RobustPoseConfig(),
    geometry_config: GeometryBootstrapConfig = GeometryBootstrapConfig(),
) -> dict[str, Any]:
    forward = _one_direction(source, reference, robust_config, geometry_config)
    reverse = _one_direction(reference, source, robust_config, geometry_config)
    base = {
        "schema": "submap_registration.v1",
        "correspondence_provider": "geometry_bootstrap_fpfh",
        "robust_config": asdict(robust_config),
        "geometry_config": asdict(geometry_config),
        "forward": forward,
        "reverse": reverse,
        "accepted": False,
        "gt_consumed": False,
    }
    if not forward["accepted"] or not reverse["accepted"]:
        return {**base, "reason": "direction_failed"}
    forward_transform = np.asarray(forward["transform"], dtype=np.float64)
    reverse_transform = np.asarray(reverse["transform"], dtype=np.float64)
    cycle_rotation, cycle_translation = transform_distance(
        forward_transform, np.linalg.inv(reverse_transform),
    )
    metrics = {
        "spatial_extent_m": forward["spatial_extent_m"],
        "spatial_second_axis_m": forward["spatial_second_axis_m"],
        "icp_update_translation_m": forward["icp_update_translation_m"],
        "icp_update_rotation_deg": forward["icp_update_rotation_deg"],
        "bidirectional_translation_m": cycle_translation,
        "bidirectional_rotation_deg": cycle_rotation,
        "cycle_translation_m": cycle_translation,
        "cycle_rotation_deg": cycle_rotation,
        "overlap_ratio": forward["verification"]["minimum_overlap"],
    }
    decision = decide_registration_v2(
        forward["consensus"], metrics, robust_config,
    )
    accepted = decision["usable_for_reconstruction"]
    return {
        **base,
        "accepted": accepted,
        "reason": "registration_decision_v2_pass" if accepted else "registration_decision_v2_reject",
        "transform": forward_transform.tolist() if accepted else None,
        "decision": decision,
    }
