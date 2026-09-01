#!/usr/bin/env python3
"""Run create-only full Orbbec baseline/candidate/refusion A/B jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pose_pipeline.adapters import orbbec_manifest
from pose_pipeline.contracts import (
    load_legacy_tcw_mm, sha256_file, write_input_sha256_audit,
    write_manifest, write_trajectory,
)
from pose_pipeline.geometry_metrics import (
    compare_no_gt_geometry, ply_geometry_metrics, render_fixed_comparison_views,
)
from pose_pipeline.runner import run_sequence
from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_scene(scene_id: str, journal: Path, legacy_trajectory: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest = orbbec_manifest(journal)
    manifest_path = output / "manifest.json"
    write_manifest(manifest_path, manifest)
    input_audit = write_input_sha256_audit(
        output / "inference_inputs.sha256.jsonl", manifest,
    )
    trajectory = load_legacy_tcw_mm(
        legacy_trajectory,
        allowed_frame_ids={frame.frame_id for frame in manifest.frames},
    )
    trajectory_path = output / "dpv_trajectory.json"
    write_trajectory(
        trajectory_path, trajectory, sequence_id=scene_id, arm="baseline",
        metadata={
            "import_format": "T_cw_row_major_translation_mm",
            "source_trajectory_sha256": sha256_file(legacy_trajectory),
            "source_journal_sha256": sha256_file(journal),
        },
    )
    baseline = run_sequence(
        arm="baseline", manifest_path=manifest_path,
        trajectory_path=trajectory_path, output_dir=output / "baseline",
    )
    candidate = run_sequence(
        arm="candidate", manifest_path=manifest_path,
        trajectory_path=trajectory_path, output_dir=output / "candidate",
    )
    if not candidate.get("corrected_trajectory_written"):
        return {
            "scene_id": scene_id, "status": "candidate_rejected",
            "frames": len(trajectory), "candidate": candidate,
        }
    baseline_refusion = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest_path,
        trajectory=output / "baseline" / "trajectory.json",
        output_dir=output / "baseline_refusion",
    ))
    candidate_refusion = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest_path,
        trajectory=output / "candidate" / "trajectory.json",
        output_dir=output / "candidate_refusion",
    ))
    baseline_geometry = ply_geometry_metrics(Path(baseline_refusion["cloud"]))
    candidate_geometry = ply_geometry_metrics(Path(candidate_refusion["cloud"]))
    write_json(output / "baseline_geometry.json", baseline_geometry)
    write_json(output / "candidate_geometry.json", candidate_geometry)
    comparison = compare_no_gt_geometry(baseline_geometry, candidate_geometry)
    write_json(output / "comparison.json", comparison)
    view = render_fixed_comparison_views(
        Path(baseline_refusion["cloud"]), Path(candidate_refusion["cloud"]),
        output / "baseline_candidate_fixed_views.png",
    )
    write_json(output / "baseline_candidate_fixed_views.json", view)
    return {
        "scene_id": scene_id,
        "status": "completed",
        "frames": len(trajectory),
        "accepted_loops": candidate.get("accepted_loop_count", 0),
        "input_records_sha256": input_audit["records_sha256"],
        **comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene", action="append", nargs=3,
        metavar=("ID", "JOURNAL", "TRAJECTORY"), required=True,
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scene_id, journal, trajectory in args.scene:
        try:
            rows.append(run_scene(
                scene_id, Path(journal), Path(trajectory),
                args.output / scene_id,
            ))
        except Exception as exc:  # retain failed outputs for diagnosis
            rows.append({
                "scene_id": scene_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
    completed = [row for row in rows if row["status"] == "completed"]
    summary = {
        "schema": "orbbec_pose_matrix.v1",
        "scene_count": len(rows),
        "completed_count": len(completed),
        "improved_count": sum(bool(row.get("passes_scene_improvement")) for row in completed),
        "safe_count": sum(bool(row.get("passes_scene_safety")) for row in completed),
        "passes_4_of_5_improvement": (
            len(rows) == 5 and len(completed) == 5
            and sum(bool(row.get("passes_scene_improvement")) for row in completed) >= 4
        ),
        "rows": rows,
        "gt_consumed": False,
    }
    write_json(args.output / "summary.json", summary)
    with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        fields = [
            "scene_id", "status", "frames", "accepted_loops",
            "passes_scene_safety", "passes_scene_improvement",
            "layer_conflict_improvement_fraction",
            "dominant_plane_thickness_improvement_fraction",
            "ground_tilt_delta_deg", "point_count_ratio", "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
