#!/usr/bin/env python3
"""Run an adaptive-anchor backend and compare its final geometry to DPV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import (
    bind_manifest_trajectory,
    load_manifest,
    load_trajectory,
    sha256_file,
)
from pose_pipeline.geometry_metrics import (
    compare_no_gt_geometry,
    ply_geometry_metrics,
    render_fixed_comparison_views,
)
from pose_pipeline.runner import run_sequence
from pose_pipeline.submaps import AdaptiveAnchorConfig
from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--baseline-refusion-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    trajectory, trajectory_payload = load_trajectory(args.trajectory)
    bound = bind_manifest_trajectory(
        manifest, trajectory, allow_manifest_superset=True,
    )
    baseline_trajectory, baseline_trajectory_payload = load_trajectory(
        args.baseline_trajectory,
    )
    baseline_bound = bind_manifest_trajectory(
        manifest, baseline_trajectory, allow_manifest_superset=True,
    )
    same_input_poses = (
        len(bound) == len(baseline_bound)
        and all(
            first[0].frame_id == second[0].frame_id
            and np.array_equal(
                first[1].t_world_camera, second[1].t_world_camera,
            )
            for first, second in zip(bound, baseline_bound)
        )
    )
    if not same_input_poses:
        raise ValueError("baseline refusion trajectory does not match A/B input poses")
    baseline_refusion = json.loads(args.baseline_refusion_result.read_text())
    expected_baseline = {
        "schema": "rgbd_full_refusion.v1",
        "status": "completed",
        "sequence_id": manifest.sequence_id,
        "integrated_frame_count": len(bound),
        "trajectory_pose_count": len(bound),
        "manifest_sha256": sha256_file(args.manifest),
        "trajectory_payload_sha256": baseline_trajectory_payload["payload_sha256"],
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    mismatches = {
        key: {"expected": value, "actual": baseline_refusion.get(key)}
        for key, value in expected_baseline.items()
        if baseline_refusion.get(key) != value
    }
    if mismatches:
        raise ValueError(f"baseline refusion contract mismatch: {mismatches}")
    baseline_cloud = Path(baseline_refusion["cloud"])
    if not baseline_cloud.is_file():
        raise FileNotFoundError(f"baseline cloud missing: {baseline_cloud}")
    if sha256_file(baseline_cloud) != baseline_refusion["cloud_sha256"]:
        raise ValueError("baseline cloud SHA mismatch")

    args.output.mkdir(parents=True, exist_ok=False)
    candidate = run_sequence(
        arm="candidate",
        manifest_path=args.manifest,
        trajectory_path=args.trajectory,
        output_dir=args.output / "candidate",
        adaptive_anchor_config=AdaptiveAnchorConfig(),
    )
    candidate_refusion = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=args.manifest,
        trajectory=args.output / "candidate" / "trajectory.json",
        output_dir=args.output / "candidate_refusion",
    ))
    baseline_geometry = ply_geometry_metrics(baseline_cloud)
    candidate_cloud = Path(candidate_refusion["cloud"])
    candidate_geometry = ply_geometry_metrics(candidate_cloud)
    comparison = compare_no_gt_geometry(baseline_geometry, candidate_geometry)
    view = render_fixed_comparison_views(
        baseline_cloud,
        candidate_cloud,
        args.output / "baseline_candidate_fixed_views.png",
    )
    _write_json(args.output / "baseline_geometry.json", baseline_geometry)
    _write_json(args.output / "candidate_geometry.json", candidate_geometry)
    _write_json(args.output / "comparison.json", comparison)
    _write_json(args.output / "baseline_candidate_fixed_views.json", view)
    summary = {
        "schema": "adaptive_anchor_backend_ab.v1",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(bound),
        "candidate": candidate,
        "same_input_pose_matrices": same_input_poses,
        "input_trajectory_payload_sha256": trajectory_payload["payload_sha256"],
        "baseline_trajectory_payload_sha256": (
            baseline_trajectory_payload["payload_sha256"]
        ),
        "baseline_refusion": baseline_refusion,
        "candidate_refusion": candidate_refusion,
        "geometry": comparison,
        "clouds_byte_identical": (
            baseline_refusion["cloud_sha256"]
            == candidate_refusion["cloud_sha256"]
        ),
        "promotion_gate": {
            "backend_correction_applied": bool(
                candidate.get("backend_correction_applied", False)
            ),
            "passes_scene_safety": comparison["passes_scene_safety"],
            "passes_scene_improvement": comparison["passes_scene_improvement"],
        },
        "promote": (
            bool(candidate.get("backend_correction_applied", False))
            and comparison["passes_scene_safety"]
            and comparison["passes_scene_improvement"]
        ),
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
