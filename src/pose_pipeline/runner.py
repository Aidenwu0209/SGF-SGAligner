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
from .pose_graph import PoseGraphEdge, optimize_pose_graph, propagate_anchor_corrections
from .robust_backend import RobustPoseConfig
from .submaps import (
    LoopProposalConfig,
    SubmapConfig,
    build_submap,
    propose_loop_pairs,
    save_submap,
    select_anchor_ordinals,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


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

    anchors = select_anchor_ordinals(len(bound), submap_config.anchor_stride)
    submaps = []
    anchor_rows = []
    for anchor_index, ordinal in enumerate(anchors):
        submap = build_submap(
            bound, ordinal, manifest.depth_scale, submap_config,
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
        registration = register_submaps_bidirectional(
            submaps[source_index].points,
            submaps[target_index].points,
            robust_config,
            geometry_config,
        )
        evidence.append({**proposal, "registration": registration})
        if registration["accepted"]:
            overlap = registration["forward"]["verification"]["minimum_overlap"]
            loop_edges.append(PoseGraphEdge(
                source=source_index,
                target=target_index,
                source_to_target=np.asarray(registration["transform"], dtype=np.float64),
                kind="robust_submap_loop",
                weight=float(np.clip(overlap / 0.35, 0.7, 1.5)),
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
