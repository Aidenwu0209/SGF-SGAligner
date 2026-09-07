"""Create-only baseline/candidate sequence runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from .contracts import (
    bind_manifest_trajectory,
    load_manifest,
    load_trajectory,
    sha256_file,
    stable_json_sha256,
    write_trajectory,
)
from .appearance import AppearanceConfig, extract_anchor_descriptors
from .bounded_backend import BoundedBackendConfig, optimize_bounded_trajectory
from .visual_verification import (
    VisualVerificationConfig, estimate_rgbd_loop, compare_visual_registration,
)
from .geometry_backend import (
    GeometryBootstrapConfig,
    register_submaps_bidirectional,
)
from .pose_graph import (
    CorrectionAuditConfig,
    LoopWeightConfig,
    PoseGraphEdge,
    PoseGraphOptimizationConfig,
    audit_corrected_trajectory,
    loop_edge_weight,
    optimize_pose_graph,
    propagate_anchor_corrections,
)
from .robust_backend import RobustPoseConfig
from .submaps import (
    LoopProposalConfig,
    SubmapConfig,
    build_submap,
    propose_loop_pairs,
    save_submap,
    select_anchor_ordinals,
)
from .telemetry import collect_resource_telemetry
from .depth_first_loops import DepthFirstPipelineConfig


@dataclass(frozen=True)
class PrecommitGeometryConfig:
    enabled: bool = False
    require_scene_improvement: bool = False
    frame_stride: int = 8
    voxel_length_m: float = 0.02
    sdf_trunc_m: float = 0.08
    depth_trunc_m: float = 4.50
    minimum_occupied_voxel_ratio: float = 0.80
    minimum_each_robust_extent_ratio: float = 0.85
    maximum_matched_plane_tilt_regression_deg: float = 2.0
    maximum_thickness_ratio: float = 1.10
    maximum_layer_conflict_ratio: float = 1.10
    improvement_policy: str = "legacy_planes"

    def __post_init__(self) -> None:
        if self.improvement_policy not in {"legacy_planes", "postfit_depth_relative"}:
            raise ValueError("unsupported geometry improvement policy")
        if self.frame_stride < 1:
            raise ValueError("precommit refusion frame stride must be positive")
        positive = (
            self.voxel_length_m, self.sdf_trunc_m, self.depth_trunc_m,
            self.minimum_occupied_voxel_ratio,
            self.minimum_each_robust_extent_ratio,
            self.maximum_matched_plane_tilt_regression_deg,
            self.maximum_thickness_ratio, self.maximum_layer_conflict_ratio,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("precommit geometry parameters must be finite and positive")


def precommit_geometry_decision(
    geometry: dict, config: PrecommitGeometryConfig,
) -> tuple[bool, str | None]:
    if not geometry.get("passes_scene_safety", False):
        return False, "precommit_geometry_gate_failed"
    if (
        config.require_scene_improvement
        and not geometry.get("passes_scene_improvement", False)
    ):
        return False, "precommit_geometry_improvement_gate_failed"
    return True, None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _copy_verified(source: Path, destination: Path) -> str:
    """Commit a trajectory only after checking the actual copied bytes."""
    with Path(source).open("rb") as reader, Path(destination).open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    digest = sha256_file(destination)
    if digest != sha256_file(source):
        raise RuntimeError("committed trajectory hash differs from source")
    return digest


def _retain_candidate_noop(
    *, output_dir: Path, manifest, trajectory, trajectory_payload: dict,
    frame_count: int, reason: str, source_trajectory_path: Path | None = None,
    run_started: float | None = None,
    stage_runtime: dict[str, float] | None = None,
) -> dict[str, Any]:
    if source_trajectory_path is None:
        write_trajectory(
            output_dir / "trajectory.json", trajectory,
            sequence_id=manifest.sequence_id, arm="candidate",
            metadata={
                "source_trajectory_sha256": trajectory_payload["payload_sha256"],
                "backend_correction": False,
                "fail_closed_action": "retain_original_dpv_trajectory",
            },
        )
    else:
        _copy_verified(source_trajectory_path, output_dir / "trajectory.json")
    result = {
        "schema": (
            "pose_pipeline_run.v2"
            if source_trajectory_path is not None else "pose_pipeline_run.v1"
        ),
        "arm": "candidate",
        "sequence_id": manifest.sequence_id,
        "frame_count": frame_count,
        "accepted": False,
        "reason": reason,
        "accepted_loop_count": 0,
        "corrected_trajectory_written": True,
        "backend_correction_applied": False,
        "identity_fallback_used": False,
        "gt_consumed": False,
        "fail_closed_byte_identical_to_source": (
            source_trajectory_path is not None
            and sha256_file(source_trajectory_path) == sha256_file(output_dir / "trajectory.json")
        ),
        "committed_trajectory_sha256": sha256_file(output_dir / "trajectory.json"),
    }
    if run_started is not None:
        telemetry_path = output_dir / "resource_telemetry.json"
        _write_json(
            telemetry_path,
            collect_resource_telemetry(
                run_started, stages=stage_runtime or {},
                work_items=frame_count, work_item_name="admitted_frame",
            ),
        )
        result["resource_telemetry_path"] = telemetry_path.name
    _write_json(output_dir / "run_result.json", result)
    return result


def run_sequence(
    *,
    arm: str,
    manifest_path: Path,
    trajectory_path: Path,
    output_dir: Path,
    submap_config: SubmapConfig = SubmapConfig(),
    proposal_config: LoopProposalConfig = LoopProposalConfig(),
    robust_config: RobustPoseConfig = RobustPoseConfig(),
    geometry_config: GeometryBootstrapConfig = GeometryBootstrapConfig(),
    loop_weight_config: LoopWeightConfig = LoopWeightConfig(),
    appearance_config: AppearanceConfig = AppearanceConfig(),
    pose_graph_config: PoseGraphOptimizationConfig = PoseGraphOptimizationConfig(),
    correction_config: CorrectionAuditConfig = CorrectionAuditConfig(),
    precommit_geometry_config: PrecommitGeometryConfig = PrecommitGeometryConfig(),
    visual_config: VisualVerificationConfig = VisualVerificationConfig(),
    bounded_config: BoundedBackendConfig = BoundedBackendConfig(),
    depth_first_config: DepthFirstPipelineConfig = DepthFirstPipelineConfig(),
) -> dict[str, Any]:
    if arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")
    if precommit_geometry_config.require_scene_improvement and not precommit_geometry_config.enabled:
        raise ValueError("geometry improvement requires the geometry gate to be enabled")
    if precommit_geometry_config.improvement_policy == "postfit_depth_relative" and not depth_first_config.enabled:
        raise ValueError("postfit improvement requires depth-first recovery")
    if depth_first_config.enabled and (
        not bounded_config.enabled or bounded_config.maximum_loop_degree != 4
        or bounded_config.correction_scaling_policy != "smooth_local"
        or not precommit_geometry_config.enabled
    ):
        raise ValueError("depth-first recovery requires full geometry checks and degree-4 smooth-local backend")
    unified_mode = visual_config.enabled or bounded_config.enabled
    if unified_mode and not (
        visual_config.enabled and bounded_config.enabled
        and precommit_geometry_config.enabled
        and precommit_geometry_config.require_scene_improvement
        and precommit_geometry_config.frame_stride == 1
        and correction_config.maximum_absolute_correction_translation_m is not None
        and correction_config.maximum_absolute_correction_rotation_deg is not None
        and pose_graph_config.robustifier == "huber"
        and correction_config.propagation == "legacy_slerp_linear"
    ):
        raise ValueError(
            "unified backend requires PnP, bounded Huber, absolute limits, "
            "legacy propagation and full-frame geometry safety plus improvement"
        )
    output_dir = Path(output_dir).resolve()
    run_started = time.perf_counter()
    stage_runtime: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(manifest_path)
    trajectory, trajectory_payload = load_trajectory(trajectory_path)
    v2_mode = bool(
        proposal_config.policy != "distance_topk"
        or geometry_config.decision_version == 3
        or pose_graph_config.robustifier != "huber"
        or correction_config.propagation != "legacy_slerp_linear"
        or correction_config.maximum_absolute_correction_translation_m is not None
        or correction_config.maximum_absolute_correction_rotation_deg is not None
        or precommit_geometry_config.enabled
        or unified_mode
    )
    bound = bind_manifest_trajectory(
        manifest, trajectory, allow_manifest_superset=not unified_mode,
    )
    _write_json(output_dir / "input_binding.json", {
        "schema": "pose_pipeline_input_binding.v1",
        "manifest_sha256": sha256_file(manifest_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "admitted_frame_sha256": stable_json_sha256([
            frame.frame_id for frame, _pose in bound
        ]),
        "visual_config": asdict(visual_config),
        "bounded_config": asdict(bounded_config),
        "correction_config": asdict(correction_config),
        "precommit_geometry_config": asdict(precommit_geometry_config),
        "gt_consumed": False,
    })
    if arm == "baseline":
        write_trajectory(
            output_dir / "trajectory.json", trajectory,
            sequence_id=manifest.sequence_id, arm="baseline",
            metadata={
                "source_trajectory_sha256": trajectory_payload["payload_sha256"],
                "backend_correction": False,
            },
        )
        result = {
            "schema": "pose_pipeline_run.v1",
            "arm": "baseline",
            "sequence_id": manifest.sequence_id,
            "frame_count": len(bound),
            "accepted_loop_count": 0,
            "corrected_trajectory_written": True,
            "gt_consumed": False,
        }
        telemetry_path = output_dir / "resource_telemetry.json"
        _write_json(
            telemetry_path, collect_resource_telemetry(
                run_started, stages=stage_runtime,
                work_items=len(bound), work_item_name="admitted_frame",
            ),
        )
        result["resource_telemetry_path"] = telemetry_path.name
        _write_json(output_dir / "run_result.json", result)
        return result

    if len(bound) < 2:
        _write_json(output_dir / "loop_evidence.json", {
            "schema": "pose_pipeline_loop_evidence.v1",
            "sequence_id": manifest.sequence_id,
            "correspondence_provider": "geometry_bootstrap_fpfh",
            "proposal_count": 0,
            "pre_sparsification_accepted_loop_count": 0,
            "submap_config": asdict(submap_config),
            "proposal_config": asdict(proposal_config),
            "robust_config": asdict(robust_config),
            "geometry_config": asdict(geometry_config),
            "loop_weight_config": asdict(loop_weight_config),
            "anchors": [],
            "evidence": [],
            "rejection_reason": "fewer_than_two_valid_frontend_poses",
            "gt_consumed": False,
        })
        return _retain_candidate_noop(
            output_dir=output_dir, manifest=manifest,
            trajectory=trajectory, trajectory_payload=trajectory_payload,
            frame_count=len(bound),
            reason="insufficient_valid_poses_for_sparse_backend",
            source_trajectory_path=trajectory_path if v2_mode else None,
            run_started=run_started, stage_runtime=stage_runtime,
        )

    anchors = select_anchor_ordinals(len(bound), submap_config.anchor_stride)
    stage_started = time.perf_counter()
    submaps = []
    anchor_rows = []
    for anchor_index, ordinal in enumerate(anchors):
        try:
            submap = build_submap(
                bound, ordinal, manifest.depth_scale, submap_config,
            )
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            _write_json(output_dir / "loop_evidence.json", {
                "schema": "pose_pipeline_loop_evidence.v1",
                "sequence_id": manifest.sequence_id,
                "correspondence_provider": "geometry_bootstrap_fpfh",
                "proposal_count": 0,
                "pre_sparsification_accepted_loop_count": 0,
                "submap_config": asdict(submap_config),
                "proposal_config": asdict(proposal_config),
                "robust_config": asdict(robust_config),
                "geometry_config": asdict(geometry_config),
                "loop_weight_config": asdict(loop_weight_config),
                "anchors": anchor_rows,
                "evidence": [],
                "rejection_reason": "submap_construction_failed",
                "failure": {
                    "anchor_index": anchor_index,
                    "anchor_ordinal": ordinal,
                    "error": f"{type(error).__name__}: {error}",
                },
                "gt_consumed": False,
            })
            return _retain_candidate_noop(
                output_dir=output_dir, manifest=manifest,
                trajectory=trajectory, trajectory_payload=trajectory_payload,
                frame_count=len(bound), reason="submap_construction_failed",
                source_trajectory_path=trajectory_path if v2_mode else None,
                run_started=run_started, stage_runtime=stage_runtime,
            )
        path = output_dir / "submaps" / (
            f"anchor_{anchor_index:03d}_frame_{submap.anchor_frame_id:06d}.npz"
        )
        save_submap(path, submap, submap_config)
        submaps.append(submap)
        anchor_rows.append({
            "anchor_index": anchor_index,
            "anchor_ordinal": ordinal,
            "anchor_frame_id": submap.anchor_frame_id,
            "source_frame_ids": list(submap.source_frame_ids),
            "point_count": len(submap.points),
            "points_sha256": submap.points_sha256,
            "path": str(path),
        })
    stage_runtime["submap_construction"] = time.perf_counter() - stage_started
    appearance_descriptors = None
    appearance_metadata = None
    if proposal_config.policy == "hybrid36":
        stage_started = time.perf_counter()
        try:
            appearance_descriptors, appearance_metadata = extract_anchor_descriptors(
                bound, anchors, appearance_config,
                output_dir / "appearance" / "anchor_descriptors.npz",
            )
            _write_json(
                output_dir / "appearance" / "anchor_descriptors.json",
                appearance_metadata,
            )
        except (ValueError, RuntimeError, OSError) as error:
            _write_json(output_dir / "loop_evidence.json", {
                "schema": "pose_pipeline_loop_evidence.v2",
                "sequence_id": manifest.sequence_id,
                "proposal_config": asdict(proposal_config),
                "appearance_config": asdict(appearance_config),
                "proposal_count": 0,
                "evidence": [],
                "rejection_reason": "appearance_extraction_failed",
                "failure": f"{type(error).__name__}: {error}",
                "gt_consumed": False,
            })
            return _retain_candidate_noop(
                output_dir=output_dir, manifest=manifest,
                trajectory=trajectory, trajectory_payload=trajectory_payload,
                frame_count=len(bound), reason="appearance_extraction_failed",
                source_trajectory_path=trajectory_path if v2_mode else None,
                run_started=run_started, stage_runtime=stage_runtime,
            )
        stage_runtime["appearance_descriptor"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    proposals = propose_loop_pairs(
        bound, anchors, proposal_config,
        appearance_descriptors=appearance_descriptors,
    )
    stage_runtime["loop_proposal"] = time.perf_counter() - stage_started
    evidence, loop_edges = [], []
    stage_started = time.perf_counter()
    for proposal in proposals:
        source_index = int(proposal["source_anchor_index"])
        target_index = int(proposal["target_anchor_index"])
        visual_estimate = None
        if visual_config.enabled:
            visual_estimate = estimate_rgbd_loop(
                bound[anchors[source_index]][0], bound[anchors[target_index]][0],
                manifest.depth_scale, config=visual_config,
            )
        try:
            if visual_config.enabled and not visual_estimate["accepted"]:
                registration = {
                    "accepted": False, "reason": "independent_visual_estimate_rejected",
                    "gt_consumed": False,
                }
            else:
                registration = register_submaps_bidirectional(
                    submaps[source_index].points,
                    submaps[target_index].points,
                    robust_config,
                    geometry_config,
                    **({"visual_evidence": visual_estimate} if visual_config.enabled else {}),
                )
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            registration = {
                "schema": "submap_registration.v1",
                "correspondence_provider": "geometry_bootstrap_fpfh",
                "accepted": False,
                "reason": "registration_exception_fail_closed",
                "error": f"{type(error).__name__}: {error}",
                "gt_consumed": False,
            }
        visual = None
        if registration["accepted"] and visual_config.enabled:
            visual = compare_visual_registration(
                visual_estimate,
                np.asarray(registration["transform"], dtype=np.float64),
                config=visual_config,
            )
        edge_verified = bool(
            registration["accepted"]
            and (not visual_config.enabled or visual["accepted"])
        )
        evidence.append({
            **proposal, "registration": registration,
            "visual_estimate": visual_estimate,
            "visual_verification": visual, "edge_verified": edge_verified,
            "source_submap_sha256": submaps[source_index].points_sha256,
            "target_submap_sha256": submaps[target_index].points_sha256,
        })
        if edge_verified:
            overlap = registration["forward"]["verification"]["minimum_overlap"]
            loop_edges.append(PoseGraphEdge(
                source=source_index,
                target=target_index,
                source_to_target=np.asarray(registration["transform"], dtype=np.float64),
                kind="robust_submap_loop",
                weight=loop_edge_weight(
                    overlap, source_index, target_index, len(anchors),
                    loop_weight_config,
                ),
                provenance=(
                    "spatial_balanced_fpfh+pagor+teaser"
                    if geometry_config.decision_version == 3
                    else "geometry_bootstrap_fpfh+pagor+pygcransac+teaser_witness"
                ),
                information=np.asarray(
                    registration.get("information_matrix", np.eye(6)),
                    dtype=np.float64,
                ),
                confidence=float(registration.get("edge_confidence", 1.0)),
            ))
    stage_runtime["registration"] = time.perf_counter() - stage_started
    _write_json(output_dir / "loop_evidence.json", {
        "schema": (
            "pose_pipeline_loop_evidence.v2"
            if proposal_config.policy == "hybrid36"
            or geometry_config.decision_version == 3
            else "pose_pipeline_loop_evidence.v1"
        ),
        "sequence_id": manifest.sequence_id,
        "correspondence_provider": "geometry_bootstrap_fpfh",
        "proposal_count": len(proposals),
        "pre_sparsification_accepted_loop_count": len(loop_edges),
        "submap_config": asdict(submap_config),
        "proposal_config": asdict(proposal_config),
        "robust_config": asdict(robust_config),
        "geometry_config": asdict(geometry_config),
        "loop_weight_config": asdict(loop_weight_config),
        "visual_config": asdict(visual_config),
        "bounded_config": asdict(bounded_config),
        "appearance_config": asdict(appearance_config),
        "appearance_descriptor_cache": appearance_metadata,
        "anchors": anchor_rows,
        "evidence": evidence,
        "gt_consumed": False,
    })
    depth_first_edges = []
    if depth_first_config.enabled:
        from .depth_first_loops import build_depth_first_edges
        stage_started = time.perf_counter()
        loop_edges, depth_first_edges, depth_first_evidence = build_depth_first_edges(
            manifest, trajectory, anchors, evidence, loop_edges,
            robust_config, geometry_config,
        )
        _write_json(output_dir / "depth_first_evidence.json", depth_first_evidence)
        stage_runtime["depth_first_recovery"] = time.perf_counter() - stage_started
    if not loop_edges:
        return _retain_candidate_noop(
            output_dir=output_dir, manifest=manifest,
            trajectory=trajectory, trajectory_payload=trajectory_payload,
            frame_count=len(bound), reason="no_verified_loop",
            source_trajectory_path=trajectory_path if v2_mode else None,
            run_started=run_started, stage_runtime=stage_runtime,
        )
    initial_anchors = [trajectory[index].t_world_camera for index in anchors]
    stage_started = time.perf_counter()
    bounded_result = None
    try:
        if bounded_config.enabled:
            corrected, bounded_result = optimize_bounded_trajectory(
                trajectory, anchors, loop_edges, config=bounded_config,
                optimization_config=pose_graph_config,
                correction_config=correction_config,
            )
            _write_json(output_dir / "bounded_backend.json", bounded_result)
            optimization = bounded_result["pose_graph"]
            stage_runtime["pose_graph"] = time.perf_counter() - stage_started
            if not bounded_result["success"] or bounded_result.get("no_op", False):
                return _retain_candidate_noop(
                    output_dir=output_dir, manifest=manifest,
                    trajectory=trajectory, trajectory_payload=trajectory_payload,
                    frame_count=len(bound), reason=(
                        "all_loops_rejected_by_influence"
                        if bounded_result.get("no_op", False) else "bounded_backend_rejected"
                    ),
                    source_trajectory_path=trajectory_path,
                    run_started=run_started, stage_runtime=stage_runtime,
                )
        else:
            optimized, optimization = optimize_pose_graph(
                initial_anchors, loop_edges,
                optimization_config=pose_graph_config,
            )
    except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as error:
        _write_json(output_dir / "backend_failure.json", {
            "error": f"{type(error).__name__}: {error}", "gt_consumed": False,
        })
        return _retain_candidate_noop(
            output_dir=output_dir, manifest=manifest,
            trajectory=trajectory, trajectory_payload=trajectory_payload,
            frame_count=len(bound), reason="pose_graph_exception_fail_closed",
            source_trajectory_path=trajectory_path if v2_mode else None,
            run_started=run_started, stage_runtime=stage_runtime,
        )
    stage_runtime["pose_graph"] = time.perf_counter() - stage_started
    _write_json(output_dir / "pose_graph_result.json", optimization)
    if not optimization["success"]:
        return _retain_candidate_noop(
            output_dir=output_dir, manifest=manifest,
            trajectory=trajectory, trajectory_payload=trajectory_payload,
            frame_count=len(bound), reason="pose_graph_failed",
            source_trajectory_path=trajectory_path if v2_mode else None,
            run_started=run_started, stage_runtime=stage_runtime,
        )
    stage_started = time.perf_counter()
    if bounded_result is None:
        corrected = propagate_anchor_corrections(
            trajectory, anchors, optimized,
            propagation=correction_config.propagation,
        )
    correction_audit = audit_corrected_trajectory(
        trajectory, corrected, correction_config,
    )
    stage_runtime["correction_propagation_and_audit"] = (
        time.perf_counter() - stage_started
    )
    _write_json(output_dir / "correction_audit.json", correction_audit)
    postfit_depth = None
    if depth_first_config.enabled:
        from .postfit_depth import audit_postfit_depth
        postfit_depth = audit_postfit_depth(
            manifest, trajectory, corrected, anchors, depth_first_edges,
        )
        _write_json(output_dir / "postfit_depth_audit.json", postfit_depth)
    if not correction_audit["passes"]:
        return _retain_candidate_noop(
            output_dir=output_dir, manifest=manifest,
            trajectory=trajectory, trajectory_payload=trajectory_payload,
            frame_count=len(bound), reason="correction_audit_failed",
            source_trajectory_path=trajectory_path if v2_mode else None,
            run_started=run_started, stage_runtime=stage_runtime,
        )
    proposed_path = (
        output_dir / "trajectory.proposed.json"
        if precommit_geometry_config.enabled else output_dir / "trajectory.json"
    )
    write_trajectory(
        proposed_path, corrected,
        sequence_id=manifest.sequence_id, arm="candidate",
        metadata={
            "source_trajectory_sha256": trajectory_payload["payload_sha256"],
            "pose_graph_result_sha256": stable_json_sha256(optimization),
            "correspondence_provider": "geometry_bootstrap_fpfh",
        },
    )
    precommit_geometry = None
    if precommit_geometry_config.enabled:
        from reconstruction.rgbd_refusion import (
            FullRefusionRequest, run_full_rgbd_refusion,
        )
        from .geometry_metrics import (
            compare_no_gt_geometry_v2, ply_geometry_metrics,
        )

        stage_started = time.perf_counter()
        selected_ids = [
            frame.frame_id
            for index, (frame, _pose) in enumerate(bound)
            if index % precommit_geometry_config.frame_stride == 0
        ]
        if bound[-1][0].frame_id not in selected_ids:
            selected_ids.append(bound[-1][0].frame_id)
        frame_hash = stable_json_sha256(selected_ids)
        try:
            baseline_refusion = run_full_rgbd_refusion(FullRefusionRequest(
                manifest=manifest_path,
                trajectory=trajectory_path,
                output_dir=output_dir / "precommit_refusion" / "baseline",
                fused_frame_ids=tuple(selected_ids),
                voxel_length_m=precommit_geometry_config.voxel_length_m,
                sdf_trunc_m=precommit_geometry_config.sdf_trunc_m,
                depth_trunc_m=precommit_geometry_config.depth_trunc_m,
            ))
            candidate_refusion = run_full_rgbd_refusion(FullRefusionRequest(
                manifest=manifest_path,
                trajectory=proposed_path,
                output_dir=output_dir / "precommit_refusion" / "candidate",
                fused_frame_ids=tuple(selected_ids),
                voxel_length_m=precommit_geometry_config.voxel_length_m,
                sdf_trunc_m=precommit_geometry_config.sdf_trunc_m,
                depth_trunc_m=precommit_geometry_config.depth_trunc_m,
            ))
            precommit_geometry = compare_no_gt_geometry_v2(
                ply_geometry_metrics(Path(baseline_refusion["cloud"])),
                ply_geometry_metrics(Path(candidate_refusion["cloud"])),
                admitted_frame_sha256=frame_hash,
                minimum_occupied_voxel_ratio=(
                    precommit_geometry_config.minimum_occupied_voxel_ratio
                ),
                minimum_each_robust_extent_ratio=(
                    precommit_geometry_config.minimum_each_robust_extent_ratio
                ),
                maximum_matched_plane_tilt_regression_deg=(
                    precommit_geometry_config.maximum_matched_plane_tilt_regression_deg
                ),
                maximum_thickness_ratio=(
                    precommit_geometry_config.maximum_thickness_ratio
                ),
                maximum_layer_conflict_ratio=(
                    precommit_geometry_config.maximum_layer_conflict_ratio
                ),
            )
        except (ValueError, RuntimeError, OSError, ImportError) as error:
            precommit_geometry = {
                "schema": "geometry_comparison.v2",
                "passes_scene_safety": False,
                "reason": "precommit_refusion_exception_fail_closed",
                "error": f"{type(error).__name__}: {error}",
                "admitted_frame_sha256": frame_hash,
                "gt_consumed": False,
            }
        if precommit_geometry_config.improvement_policy == "postfit_depth_relative":
            from .postfit_depth import relative_improvement_decision
            improvement = relative_improvement_decision(postfit_depth)
            precommit_geometry["legacy_passes_scene_improvement"] = precommit_geometry.get("passes_scene_improvement", False)
            precommit_geometry["improvement_policy"] = "postfit_depth_relative"
            precommit_geometry["postfit_relative_improvement"] = improvement
            precommit_geometry["passes_scene_improvement"] = bool(
                precommit_geometry.get("passes_scene_safety", False) and improvement["passes"]
            )
        _write_json(
            output_dir / "precommit_refusion" / "geometry_comparison.json",
            precommit_geometry,
        )
        stage_runtime["precommit_sparse_refusion"] = (
            time.perf_counter() - stage_started
        )
        geometry_accepted, rejection_reason = precommit_geometry_decision(
            precommit_geometry, precommit_geometry_config,
        )
        if not geometry_accepted:
            _copy_verified(trajectory_path, output_dir / "trajectory.json")
            result = {
                "schema": "pose_pipeline_run.v2",
                "arm": "candidate",
                "sequence_id": manifest.sequence_id,
                "frame_count": len(bound),
                "accepted": False,
                "reason": str(rejection_reason),
                "accepted_loop_count": optimization["accepted_loop_edge_count"],
                "corrected_trajectory_written": True,
                "proposed_trajectory_retained": str(proposed_path),
                "backend_correction_applied": False,
                "fail_closed_byte_identical_to_source": True,
                "precommit_geometry": precommit_geometry,
                "precommit_geometry_improvement_required": (
                    precommit_geometry_config.require_scene_improvement
                ),
                "identity_fallback_used": False,
                "gt_consumed": False,
            }
            telemetry_path = output_dir / "resource_telemetry.json"
            _write_json(
                telemetry_path,
                collect_resource_telemetry(
                    run_started, stages=stage_runtime,
                    work_items=len(bound), work_item_name="admitted_frame",
                ),
            )
            result["resource_telemetry_path"] = telemetry_path.name
            _write_json(output_dir / "run_result.json", result)
            return result
        _copy_verified(proposed_path, output_dir / "trajectory.json")
    result = {
        "schema": (
            "pose_pipeline_run.v2"
            if precommit_geometry_config.enabled
            or pose_graph_config.robustifier == "adaptive_gnc"
            else "pose_pipeline_run.v1"
        ),
        "arm": "candidate",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(bound),
        "accepted": True,
        "reason": "verified_sparse_loops_optimized",
        "accepted_loop_count": optimization["accepted_loop_edge_count"],
        "corrected_trajectory_written": True,
        "backend_correction_applied": True,
        "correction_audit": correction_audit,
        "precommit_geometry": precommit_geometry,
        "precommit_geometry_improvement_required": (
            precommit_geometry_config.require_scene_improvement
        ),
        "identity_fallback_used": False,
        "committed_trajectory_sha256": sha256_file(output_dir / "trajectory.json"),
        "gt_consumed": False,
    }
    telemetry_path = output_dir / "resource_telemetry.json"
    _write_json(
        telemetry_path, collect_resource_telemetry(
            run_started, stages=stage_runtime,
            work_items=len(bound), work_item_name="admitted_frame",
        ),
    )
    result["resource_telemetry_path"] = telemetry_path.name
    _write_json(output_dir / "run_result.json", result)
    return result
