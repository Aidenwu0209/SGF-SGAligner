#!/usr/bin/env python3
"""Create-only full-frame TSDF refusion for an Orbbec validation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_pipeline.geometry_metrics import ply_geometry_metrics  # noqa: E402
from reconstruction.rgbd_refusion import (  # noqa: E402
    FullRefusionRequest,
    run_full_rgbd_refusion,
)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    matrix = json.loads((run_root / "summary.json").read_text())
    rows = []
    for result in matrix["results"]:
        sequence_id = result["sequence_id"]
        refusion = run_full_rgbd_refusion(FullRefusionRequest(
            manifest=Path(result["manifest"]),
            trajectory=run_root / sequence_id / "frontend" / "trajectory.json",
            output_dir=output / sequence_id,
        ))
        geometry = ply_geometry_metrics(Path(refusion["cloud"]))
        _write_json(output / sequence_id / "geometry.json", geometry)
        rows.append({
            "sequence_id": sequence_id,
            "status": refusion["status"],
            "requested_frame_count": refusion["requested_frame_count"],
            "integrated_frame_count": refusion["integrated_frame_count"],
            "point_count": refusion["point_count"],
            "cloud": refusion["cloud"],
            "cloud_sha256": refusion["cloud_sha256"],
            "occupied_voxels_2cm": geometry["occupied_voxels_2cm"],
            "bbox_extent_m": geometry["bbox_extent_m"],
            "near_parallel_layer_conflict_ratio": geometry[
                "near_parallel_layer_conflict_ratio"
            ],
            "identity_fallback_used": False,
            "gt_consumed": False,
        })
    summary = {
        "schema": "orbbec_validation_full_refusion.v1",
        "sequence_count": len(rows),
        "total_integrated_frames": sum(row["integrated_frame_count"] for row in rows),
        "total_points": sum(row["point_count"] for row in rows),
        "all_completed": len(rows) == 6 and all(row["status"] == "completed" for row in rows),
        "identity_fallback_used": False,
        "gt_consumed": False,
        "rows": rows,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["all_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
