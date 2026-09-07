"""Geometry bootstrap for sequence-level sparse submap validation.

SGAligner point correspondences use :mod:`robust_backend` directly.  This
module supplies a separately labelled FPFH provider so sequence experiments
can test the pose graph and refusion chain before semantic subgraphs exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .contracts import stable_json_sha256, validate_se3

from .robust_backend import (
    RobustPoseConfig,
    _hypothesis,
    decide_registration_v2,
    decide_registration_v3,
    generate_hypotheses,
    select_cross_solver_consensus,
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
    correspondence_policy: str = "baseline_mutual_fpfh"
    spatial_bin_count: int = 4
    regeneration_pool_multiplier: int = 4
    regeneration_compatibility_m: float = 0.10
    decision_version: int = 2

    def __post_init__(self) -> None:
        if self.correspondence_policy not in {
            "baseline_mutual_fpfh", "spatial_balanced_v2",
        }:
            raise ValueError("unsupported correspondence policy")
        if self.spatial_bin_count < 1:
            raise ValueError("spatial_bin_count must be positive")
        if self.regeneration_pool_multiplier < 1:
            raise ValueError("regeneration_pool_multiplier must be positive")
        if self.regeneration_compatibility_m <= 0.0:
            raise ValueError("regeneration compatibility must be positive")
        if self.decision_version not in {2, 3}:
            raise ValueError("decision_version must be 2 or 3")


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


def _spatial_bin_keys(points: np.ndarray, count: int) -> np.ndarray:
    minimum = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - minimum, 1e-9)
    normalized = np.clip((points - minimum) / extent, 0.0, 1.0 - 1e-12)
    return np.floor(normalized * count).astype(np.int32)


def spatial_balanced_fpfh_correspondences(
    source: np.ndarray,
    reference: np.ndarray,
    config: GeometryBootstrapConfig = GeometryBootstrapConfig(
        correspondence_policy="spatial_balanced_v2", decision_version=3,
    ),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Regenerate one-way FPFH matches and select spatially diverse support.

    This is an independent deterministic implementation of two transferable
    ideas from recent registration work: expand beyond mutual-only matches,
    then rank well-distributed support ahead of dense local clusters.
    """
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
    scale = max(float(np.median(distances[:, 0])), 1e-12)
    confidence = (
        np.clip(1.0 - ratio, 0.0, 1.0)
        * np.exp(-distances[:, 0] / scale)
    )
    mutual = reverse[reference_indices] == source_indices
    seed_mask = mutual & (ratio <= config.feature_ratio)
    seed_source = source_indices[seed_mask]
    seed_reference = reference_indices[seed_mask]
    if len(seed_source) < 3:
        raise ValueError("insufficient mutual FPFH seeds for regeneration")

    pool_limit = config.maximum_correspondences * config.regeneration_pool_multiplier
    pool_order = np.lexsort((reference_indices, source_indices, -confidence))
    pool_order = pool_order[:pool_limit]
    candidate_source = source_indices[pool_order]
    candidate_reference = reference_indices[pool_order]
    candidate_confidence = confidence[pool_order]
    accepted = []
    seed_source_points = source_xyz[seed_source]
    seed_reference_points = reference_xyz[seed_reference]
    required_support = min(3, len(seed_source))
    for ordinal, (source_index, reference_index) in enumerate(zip(
        candidate_source, candidate_reference,
    )):
        source_distance = np.linalg.norm(
            seed_source_points - source_xyz[source_index], axis=1,
        )
        reference_distance = np.linalg.norm(
            seed_reference_points - reference_xyz[reference_index], axis=1,
        )
        compatible = (
            np.abs(source_distance - reference_distance)
            <= config.regeneration_compatibility_m
        )
        if int(compatible.sum()) >= required_support:
            accepted.append(ordinal)
    if len(accepted) < 3:
        raise ValueError("correspondence regeneration produced insufficient support")
    accepted = np.asarray(accepted, dtype=np.int64)
    candidate_source = candidate_source[accepted]
    candidate_reference = candidate_reference[accepted]
    candidate_confidence = candidate_confidence[accepted]

    source_bins = _spatial_bin_keys(source_xyz[candidate_source], config.spatial_bin_count)
    reference_bins = _spatial_bin_keys(
        reference_xyz[candidate_reference], config.spatial_bin_count,
    )
    buckets: dict[tuple[int, ...], list[int]] = {}
    for index in range(len(candidate_source)):
        key = tuple(source_bins[index].tolist() + reference_bins[index].tolist())
        buckets.setdefault(key, []).append(index)
    for rows in buckets.values():
        rows.sort(key=lambda index: (
            -float(candidate_confidence[index]),
            int(candidate_source[index]),
            int(candidate_reference[index]),
        ))
    selected = []
    keys = sorted(buckets)
    while len(selected) < config.maximum_correspondences:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) >= config.maximum_correspondences:
                    break
        if not progressed:
            break
    selected = np.asarray(selected, dtype=np.int64)
    source_corr = np.ascontiguousarray(source_xyz[candidate_source[selected]])
    reference_corr = np.ascontiguousarray(
        reference_xyz[candidate_reference[selected]],
    )
    selected_confidence = np.ascontiguousarray(
        candidate_confidence[selected], dtype=np.float64,
    )
    evidence = {
        "schema": "fpfh_correspondence_evidence.v2",
        "policy": "spatial_balanced_v2",
        "mutual_seed_count": int(len(seed_source)),
        "regenerated_pool_count": int(len(pool_order)),
        "compatibility_survivor_count": int(len(accepted)),
        "selected_count": int(len(selected)),
        "occupied_source_reference_bins": int(len({
            tuple(source_bins[index].tolist() + reference_bins[index].tolist())
            for index in selected
        })),
        "mean_confidence": float(np.mean(selected_confidence)),
        "gt_consumed": False,
    }
    return (
        source_xyz, reference_xyz, source_corr, reference_corr,
        selected_confidence, evidence,
    )


def _hypothesis_distribution_quality(
    hypothesis: dict[str, Any], source: np.ndarray, reference: np.ndarray,
    threshold_m: float,
) -> dict[str, float]:
    transform = np.asarray(hypothesis["transform"], dtype=np.float64)
    residual = np.linalg.norm(transform_points(source, transform) - reference, axis=1)
    inliers = residual <= threshold_m
    if int(inliers.sum()) < 3:
        return {"score": 0.0, "inlier_ratio": 0.0, "coverage": 0.0, "rmse_m": 1e6}
    values = source[inliers]
    all_extent = np.maximum(np.ptp(source, axis=0), 1e-9)
    coverage = float(np.prod(np.clip(np.ptp(values, axis=0) / all_extent, 0.0, 1.0)) ** (1.0 / 3.0))
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    inlier_ratio = float(np.mean(inliers))
    score = inlier_ratio * max(coverage, 1e-6) / (1.0 + rmse / threshold_m)
    return {
        "score": score,
        "inlier_ratio": inlier_ratio,
        "coverage": coverage,
        "rmse_m": rmse,
    }


def _select_distribution_aware_consensus(
    hypotheses: list[dict[str, Any]], source: np.ndarray,
    reference: np.ndarray, robust_config: RobustPoseConfig,
) -> dict[str, Any]:
    consensus = select_cross_solver_consensus(hypotheses, robust_config)
    if consensus.get("accepted") is not True:
        return consensus
    candidate_indices = consensus["winning_indices"]
    qualities = {
        int(index): _hypothesis_distribution_quality(
            hypotheses[int(index)], source, reference,
            robust_config.residual_threshold_m,
        )
        for index in candidate_indices
    }
    selected_index = min(
        qualities,
        key=lambda index: (
            -qualities[index]["score"],
            str(hypotheses[index]["hypothesis_sha256"]),
        ),
    )
    selected = hypotheses[selected_index]
    return {
        **consensus,
        "reason": "unique_cross_solver_cluster_distribution_ranked",
        "selected_index": selected_index,
        "selected_hypothesis_sha256": selected["hypothesis_sha256"],
        "selected_transform": selected["transform"],
        "distribution_quality": {
            str(index): quality for index, quality in qualities.items()
        },
    }


def _point_information(
    source: np.ndarray, reference: np.ndarray, transform: np.ndarray,
    distance_m: float,
) -> tuple[np.ndarray, float, int]:
    from scipy.spatial import cKDTree

    moved = transform_points(source, transform)
    reference_tree = cKDTree(reference)
    distances, indices = reference_tree.query(moved, k=1, workers=-1)
    keep = distances <= distance_m
    if int(keep.sum()) < 6:
        information = np.eye(6, dtype=np.float64) * 1e-6
        return information, 1.0 / np.finfo(np.float64).eps, int(keep.sum())
    points = moved[keep]
    matches = reference[indices[keep]]
    neighbour_count = min(12, len(reference))
    neighbour_indices = reference_tree.query(
        matches, k=neighbour_count, workers=-1,
    )[1]
    rows = []
    for point, neighbours in zip(points, neighbour_indices):
        local = reference[np.atleast_1d(neighbours)]
        centred = local - local.mean(axis=0)
        covariance = centred.T @ centred / max(len(local), 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        rows.append(np.r_[np.cross(point, normal), normal])
    jacobian = np.stack(rows, axis=0)
    hessian = jacobian.T @ jacobian / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    largest = max(float(eigenvalues[-1]), 1e-12)
    numeric_floor = largest * np.finfo(np.float64).eps
    condition = float(largest / max(float(eigenvalues[0]), numeric_floor))
    regularization_floor = largest * 1e-6
    clipped = np.maximum(eigenvalues, regularization_floor)
    information = eigenvectors @ np.diag(clipped / largest) @ eigenvectors.T
    return np.ascontiguousarray(information), condition, int(keep.sum())


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
    visual_hypothesis: dict | None = None,
) -> dict[str, Any]:
    if geometry_config.correspondence_policy == "spatial_balanced_v2":
        (
            source_xyz, reference_xyz, source_corr, reference_corr,
            correspondence_confidence, correspondence_evidence,
        ) = spatial_balanced_fpfh_correspondences(
            source, reference, geometry_config,
        )
    else:
        source_xyz, reference_xyz, source_corr, reference_corr = fpfh_correspondences(
            source, reference, geometry_config,
        )
        correspondence_confidence = np.ones(len(source_corr), dtype=np.float64)
        correspondence_evidence = {
            "schema": "fpfh_correspondence_evidence.v1",
            "policy": "baseline_mutual_fpfh",
            "selected_count": int(len(source_corr)),
            "mean_confidence": 1.0,
            "gt_consumed": False,
        }
    hypothesis_set = generate_hypotheses(
        source_corr, reference_corr, robust_config,
        include_pygcransac=(
            geometry_config.correspondence_policy != "spatial_balanced_v2"
        ),
    )
    if visual_hypothesis is not None:
        # An RGB-D estimate is one independent family, never two votes merely
        # because it was solved in both directions. The original unique-clique
        # and minimum-family requirements remain unchanged.
        hypothesis_set["hypotheses"] = [
            *hypothesis_set["hypotheses"], visual_hypothesis,
        ]
    consensus = (
        _select_distribution_aware_consensus(
            hypothesis_set["hypotheses"], source_corr, reference_corr,
            robust_config,
        )
        if geometry_config.correspondence_policy == "spatial_balanced_v2"
        else select_cross_solver_consensus(
            hypothesis_set["hypotheses"], robust_config,
        )
    )
    result = {
        "provider": "geometry_bootstrap_fpfh",
        "hypothesis_set": hypothesis_set,
        "consensus": consensus,
        "correspondence_evidence": correspondence_evidence,
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
    singular = np.linalg.svd(
        source_corr - source_corr.mean(axis=0), compute_uv=False,
    )
    extent = float(singular[0]) if len(singular) else 0.0
    second = float(singular[1]) if len(singular) > 1 else 0.0
    third = float(singular[2]) if len(singular) > 2 else 0.0
    if geometry_config.decision_version == 3:
        # The legacy extent/second-axis thresholds were calibrated on raw
        # singular values.  The new third-axis field is explicitly metric, so
        # only it is normalized to an RMS spatial scale in metres.
        third /= np.sqrt(max(len(source_corr) - 1, 1))
    information, condition, information_support = _point_information(
        source_xyz, reference_xyz, refined,
        geometry_config.verification_distance_m,
    )
    result.update({
        "accepted": True,
        "reason": "cross_solver_consensus_and_icp",
        "transform": refined.tolist(),
        "icp_update_rotation_deg": update_rotation,
        "icp_update_translation_m": update_translation,
        "verification": verification,
        "spatial_extent_m": extent,
        "spatial_second_axis_m": second,
        "spatial_third_axis_m": third,
        "correspondence_count": len(source_corr),
        "correspondence_confidence": float(np.mean(correspondence_confidence)),
        "information_matrix": information.tolist(),
        "information_condition_number": condition,
        "information_support_count": information_support,
    })
    return result


def register_submaps_bidirectional(
    source: np.ndarray,
    reference: np.ndarray,
    robust_config: RobustPoseConfig = RobustPoseConfig(),
    geometry_config: GeometryBootstrapConfig = GeometryBootstrapConfig(),
    *, visual_evidence: dict | None = None,
) -> dict[str, Any]:
    visual_forward, visual_reverse = None, None
    if visual_evidence is not None:
        if (
            visual_evidence.get("schema") != "rgbd_visual_loop_estimate.v1"
            or visual_evidence.get("accepted") is not True
            or visual_evidence.get("gt_consumed") is not False
            or robust_config.minimum_solver_families < 2
        ):
            raise ValueError("visual consensus requires an accepted GT-free witness and two solver families")
        certificate = {"visual_evidence_sha256": stable_json_sha256(visual_evidence)}
        hypotheses = []
        for direction in ("forward", "reverse"):
            witness = visual_evidence[direction]
            transform = validate_se3(witness["transform"])
            if direction == "reverse":
                transform = np.linalg.inv(transform)
            hypotheses.append(_hypothesis(
                family="rgbd_pnp", solver="bidirectional_sift_epnp_ransac",
                transform=transform, support_count=int(witness["inliers"]),
                correspondence_count=int(witness["depth_correspondences"]),
                threshold_m=geometry_config.verification_distance_m,
                certificate=certificate,
            ))
        visual_forward, visual_reverse = hypotheses
    forward = _one_direction(source, reference, robust_config, geometry_config, visual_forward)
    reverse = _one_direction(reference, source, robust_config, geometry_config, visual_reverse)
    base = {
        "schema": (
            "submap_registration.v2"
            if geometry_config.decision_version == 3 else "submap_registration.v1"
        ),
        "correspondence_provider": (
            "geometry_bootstrap_fpfh_spatial_balanced_v2"
            if geometry_config.correspondence_policy == "spatial_balanced_v2"
            else "geometry_bootstrap_fpfh"
        ),
        "robust_config": asdict(robust_config),
        "geometry_config": asdict(geometry_config),
        "forward": forward,
        "reverse": reverse,
        "accepted": False,
        "gt_consumed": False,
        "visual_consensus_witness": visual_evidence is not None,
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
        "forward_overlap": forward["verification"]["forward_overlap"],
        "reverse_overlap": forward["verification"]["reverse_overlap"],
        "trimmed_rmse_m": forward["verification"]["trimmed_rmse_m"],
        "spatial_third_axis_m": forward["spatial_third_axis_m"],
        "correspondence_confidence": forward["correspondence_confidence"],
        "information_matrix": forward["information_matrix"],
        "information_condition_number": forward["information_condition_number"],
    }
    decision = (
        decide_registration_v3(forward["consensus"], metrics, robust_config)
        if geometry_config.decision_version == 3
        else decide_registration_v2(forward["consensus"], metrics, robust_config)
    )
    accepted = decision["usable_for_reconstruction"]
    return {
        **base,
        "accepted": accepted,
        "reason": (
            f"registration_decision_v{geometry_config.decision_version}_pass"
            if accepted else
            f"registration_decision_v{geometry_config.decision_version}_reject"
        ),
        "transform": forward_transform.tolist() if accepted else None,
        "decision": decision,
        "information_matrix": forward["information_matrix"],
        "edge_confidence": float(
            forward["correspondence_confidence"]
            * forward["verification"]["minimum_overlap"]
            / (1.0 + forward["verification"]["trimmed_rmse_m"]
               / geometry_config.verification_distance_m)
        ),
    }
