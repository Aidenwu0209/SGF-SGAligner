"""Create-only baseline/candidate sequence runner."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    bind_manifest_trajectory,
    load_manifest,
    load_trajectory,
    stable_json_sha256,
    write_trajectory,
)
from .geometry_backend import (
    GeometryBootstrapConfig,
    register_submaps_bidirectional,
)
from .pose_graph import (
    LoopWeightConfig,
    PoseGraphEdge,
    loop_edge_weight,
    optimize_pose_graph,
    propagate_anchor_corrections,
)
from .robust_backend import RobustPoseConfig
from .submaps import (
    AdaptiveAnchorConfig,
    LoopProposalConfig,
    SubmapConfig,
    build_submap,
    propose_loop_pairs,
    save_submap,
    select_adaptive_anchor_ordinals,
    select_anchor_ordinals,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _retain_candidate_noop(
    *, output_dir: Path, manifest, trajectory, trajectory_payload: dict,
    frame_count: int, reason: str,
) -> dict[str, Any]:
    write_trajectory(
        output_dir / "trajectory.json", trajectory,
        sequence_id=manifest.sequence_id, arm="candidate",
        metadata={
            "source_trajectory_sha256": trajectory_payload["payload_sha256"],
            "backend_correction": False,
            "fail_closed_action": "retain_original_dpv_trajectory",
        },
    )
    result = {
        "schema": "pose_pipeline_run.v1",
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
    }
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
    adaptive_anchor_config: AdaptiveAnchorConfig | None = None,
) -> dict[str, Any]:
    if arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest(manifest_path)
    trajectory, trajectory_payload = load_trajectory(trajectory_path)
    bound = bind_manifest_trajectory(
        manifest, trajectory, allow_manifest_superset=True,
    )
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
        write_trajectory(
            output_dir / "trajectory.json", trajectory,
            sequence_id=manifest.sequence_id, arm="candidate",
            metadata={
                "source_trajectory_sha256": trajectory_payload["payload_sha256"],
                "backend_correction": False,
                "fail_closed_action": "retain_original_dpv_trajectory",
            },
        )
        result = {
            "schema": "pose_pipeline_run.v1",
            "arm": "candidate",
            "sequence_id": manifest.sequence_id,
            "frame_count": len(bound),
            "accepted": False,
            "reason": "insufficient_valid_poses_for_sparse_backend",
            "accepted_loop_count": 0,
            "corrected_trajectory_written": True,
            "backend_correction_applied": False,
            "identity_fallback_used": False,
            "gt_consumed": False,
        }
        _write_json(output_dir / "run_result.json", result)
        return result

    if adaptive_anchor_config is None:
        anchors = select_anchor_ordinals(len(bound), submap_config.anchor_stride)
        anchor_selection = {
            "mode": "fixed_stride",
            "anchor_stride": submap_config.anchor_stride,
            "anchors": anchors,
            "evidence": None,
        }
    else:
        try:
            anchors, selection_evidence = select_adaptive_anchor_ordinals(
                bound, manifest.depth_scale,
                adaptive_anchor_config, submap_config,
            )
        except (OSError, ValueError, RuntimeError) as error:
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
                "anchor_selection": {
                    "mode": "adaptive_metric_rgbd_reprojection",
                    "config": asdict(adaptive_anchor_config),
                },
                "anchors": [],
                "evidence": [],
                "rejection_reason": "adaptive_anchor_selection_failed",
                "failure": f"{type(error).__name__}: {error}",
                "gt_consumed": False,
            })
            return _retain_candidate_noop(
                output_dir=output_dir, manifest=manifest,
                trajectory=trajectory, trajectory_payload=trajectory_payload,
                frame_count=len(bound), reason="adaptive_anchor_selection_failed",
            )
        anchor_selection = {
            "mode": "adaptive_metric_rgbd_reprojection",
            "config": asdict(adaptive_anchor_config),
            "anchors": anchors,
            "evidence": selection_evidence,
        }
    submaps = []
    anchor_rows = []
    for anchor_index, ordinal in enumerate(anchors):
        try:
            submap = build_submap(
                bound, ordinal, manifest.depth_scale, submap_config,
            )
        except (ValueError, RuntimeError) as error:
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
                "anchor_selection": anchor_selection,
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
    proposals = propose_loop_pairs(bound, anchors, proposal_config)
    evidence, loop_edges = [], []
    for proposal in proposals:
        source_index = int(proposal["source_anchor_index"])
        target_index = int(proposal["target_anchor_index"])
        try:
            registration = register_submaps_bidirectional(
                submaps[source_index].points,
                submaps[target_index].points,
                robust_config,
                geometry_config,
            )
        except (ValueError, RuntimeError) as error:
            registration = {
                "schema": "submap_registration.v1",
                "correspondence_provider": "geometry_bootstrap_fpfh",
                "accepted": False,
                "reason": "registration_exception_fail_closed",
                "error": f"{type(error).__name__}: {error}",
                "gt_consumed": False,
            }
        evidence.append({**proposal, "registration": registration})
        if registration["accepted"]:
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
                provenance="geometry_bootstrap_fpfh+pagor+pygcransac+teaser_witness",
            ))
    _write_json(output_dir / "loop_evidence.json", {
        "schema": "pose_pipeline_loop_evidence.v1",
        "sequence_id": manifest.sequence_id,
        "correspondence_provider": "geometry_bootstrap_fpfh",
        "proposal_count": len(proposals),
        "pre_sparsification_accepted_loop_count": len(loop_edges),
        "submap_config": asdict(submap_config),
        "proposal_config": asdict(proposal_config),
        "robust_config": asdict(robust_config),
        "geometry_config": asdict(geometry_config),
        "loop_weight_config": asdict(loop_weight_config),
        "anchor_selection": anchor_selection,
        "anchors": anchor_rows,
        "evidence": evidence,
        "gt_consumed": False,
    })
    if not loop_edges:
        write_trajectory(
            output_dir / "trajectory.json", trajectory,
            sequence_id=manifest.sequence_id, arm="candidate",
            metadata={
                "source_trajectory_sha256": trajectory_payload["payload_sha256"],
                "backend_correction": False,
                "fail_closed_action": "retain_original_dpv_trajectory",
            },
        )
        result = {
            "schema": "pose_pipeline_run.v1",
            "arm": "candidate",
            "sequence_id": manifest.sequence_id,
            "frame_count": len(bound),
            "accepted": False,
            "reason": "no_verified_loop",
            "accepted_loop_count": 0,
            "corrected_trajectory_written": True,
            "backend_correction_applied": False,
            "identity_fallback_used": False,
            "gt_consumed": False,
        }
        _write_json(output_dir / "run_result.json", result)
        return result
    initial_anchors = [trajectory[index].t_world_camera for index in anchors]
    optimized, optimization = optimize_pose_graph(initial_anchors, loop_edges)
    _write_json(output_dir / "pose_graph_result.json", optimization)
    if not optimization["success"]:
        write_trajectory(
            output_dir / "trajectory.json", trajectory,
            sequence_id=manifest.sequence_id, arm="candidate",
            metadata={
                "source_trajectory_sha256": trajectory_payload["payload_sha256"],
                "backend_correction": False,
                "fail_closed_action": "retain_original_dpv_trajectory",
            },
        )
        result = {
            "schema": "pose_pipeline_run.v1",
            "arm": "candidate",
            "sequence_id": manifest.sequence_id,
            "frame_count": len(bound),
            "accepted": False,
            "reason": "pose_graph_failed",
            "accepted_loop_count": 0,
            "corrected_trajectory_written": True,
            "backend_correction_applied": False,
            "identity_fallback_used": False,
            "gt_consumed": False,
        }
        _write_json(output_dir / "run_result.json", result)
        return result
    corrected = propagate_anchor_corrections(trajectory, anchors, optimized)
    write_trajectory(
        output_dir / "trajectory.json", corrected,
        sequence_id=manifest.sequence_id, arm="candidate",
        metadata={
            "source_trajectory_sha256": trajectory_payload["payload_sha256"],
            "pose_graph_result_sha256": stable_json_sha256(optimization),
            "correspondence_provider": "geometry_bootstrap_fpfh",
        },
    )
    result = {
        "schema": "pose_pipeline_run.v1",
        "arm": "candidate",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(bound),
        "accepted": True,
        "reason": "verified_sparse_loops_optimized",
        "accepted_loop_count": optimization["accepted_loop_edge_count"],
        "corrected_trajectory_written": True,
        "backend_correction_applied": True,
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    _write_json(output_dir / "run_result.json", result)
    return result
