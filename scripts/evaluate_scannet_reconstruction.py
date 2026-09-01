#!/usr/bin/env python3
"""Add evaluation-only ScanNet mesh metrics to completed A/B outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import load_trajectory, sha256_file, validate_se3
from pose_pipeline.evaluation import paired_bootstrap_improvement, reconstruction_surface_metrics


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def alignment_for(scene: Path, trajectory_path: Path) -> np.ndarray:
    trajectory, _ = load_trajectory(trajectory_path)
    for pose in trajectory:
        try:
            truth = validate_se3(
                np.loadtxt(scene / "pose" / f"{pose.frame_id}.txt"),
                f"ScanNet GT frame {pose.frame_id}",
            )
        except ValueError:
            continue
        return validate_se3(truth @ np.linalg.inv(pose.t_world_camera))
    raise ValueError(f"no finite evaluation pose for {scene.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-results", type=Path, required=True)
    parser.add_argument("--rgbd-root", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scene_output in sorted(args.sequence_results.glob("scene*")):
        if not scene_output.is_dir():
            continue
        scene_id = scene_output.name
        scene = args.rgbd_root / scene_id
        mesh = args.mesh_root / scene_id / f"{scene_id}_vh_clean_2.ply"
        row = {"scene_id": scene_id, "status": "failed"}
        if not mesh.is_file():
            row.update({"status": "missing_reference_mesh", "mesh": str(mesh)})
            rows.append(row)
            continue
        try:
            for arm in ("baseline", "candidate"):
                trajectory = scene_output / arm / "trajectory.json"
                cloud = scene_output / f"{arm}_refusion" / "refused.ply"
                metrics = reconstruction_surface_metrics(
                    cloud, mesh, alignment_for(scene, trajectory),
                )
                metrics["estimate_cloud_sha256"] = sha256_file(cloud)
                metrics["reference_mesh_sha256"] = sha256_file(mesh)
                write_json(
                    scene_output / "evaluation" / f"{arm}_reconstruction.json",
                    metrics,
                )
                row[f"{arm}_chamfer_rmse_m"] = metrics["symmetric_chamfer_rmse_m"]
                row[f"{arm}_fscore"] = metrics["fscore"]
            row["status"] = "completed"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    complete = [row for row in rows if row["status"] == "completed"]
    summary = {
        "schema": "scannet_reconstruction_evaluation_matrix.v1",
        "scene_count": len(rows),
        "completed_count": len(complete),
        "missing_mesh_count": sum(row["status"] == "missing_reference_mesh" for row in rows),
        "chamfer_bootstrap": paired_bootstrap_improvement(
            [row["baseline_chamfer_rmse_m"] for row in complete],
            [row["candidate_chamfer_rmse_m"] for row in complete],
        ) if complete else None,
        "fscore_error_bootstrap": paired_bootstrap_improvement(
            [1.0 - row["baseline_fscore"] for row in complete],
            [1.0 - row["candidate_fscore"] for row in complete],
        ) if complete else None,
        "rows": rows,
        "gt_role": "evaluation_only",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
