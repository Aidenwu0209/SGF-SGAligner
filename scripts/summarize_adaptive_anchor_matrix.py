#!/usr/bin/env python3
"""Create a compact cross-scene verdict from adaptive-anchor evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENES = (
    "fast_turn",
    "leave_and_return",
    "sgf_parameter_control",
    "slow_table_loop",
    "small_motion",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-dir", type=Path, required=True)
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed_payload = json.loads(args.fixed_summary.read_text())
    fixed_by_scene = {
        row["scene_id"]: row for row in fixed_payload["rows"]
    }
    rows = []
    for scene in SCENES:
        schedule = json.loads((args.schedule_dir / f"{scene}.json").read_text())
        backend = json.loads((args.backend_dir / scene / "summary.json").read_text())
        fixed = fixed_by_scene[scene]
        rows.append({
            "sequence_id": scene,
            "frame_count": backend["frame_count"],
            "fixed_anchor_count": schedule["fixed"]["anchor_count"],
            "adaptive_anchor_count": schedule["adaptive"]["anchor_count"],
            "anchor_count_multiplier": schedule["comparison"][
                "anchor_count_multiplier"
            ],
            "p95_median_flow_reduction_fraction": schedule["comparison"][
                "p95_median_flow_reduction_fraction"
            ],
            "passes_anchor_schedule_gate": schedule["comparison"][
                "passes_anchor_schedule_gate"
            ],
            "fixed_accepted_loop_count": fixed["accepted_loops"],
            "adaptive_accepted_loop_count": backend["candidate"][
                "accepted_loop_count"
            ],
            "backend_correction_applied": backend["candidate"].get(
                "backend_correction_applied", False,
            ),
            "layer_conflict_improvement_fraction": backend["geometry"][
                "layer_conflict_improvement_fraction"
            ],
            "dominant_plane_thickness_improvement_fraction": backend[
                "geometry"
            ]["dominant_plane_thickness_improvement_fraction"],
            "point_count_ratio": backend["geometry"]["point_count_ratio"],
            "passes_scene_safety": backend["geometry"]["passes_scene_safety"],
            "passes_scene_improvement": backend["geometry"][
                "passes_scene_improvement"
            ],
            "promote": backend["promote"],
            "candidate_cloud_sha256": backend["candidate_refusion"][
                "cloud_sha256"
            ],
            "identity_fallback_used": backend["identity_fallback_used"],
            "gt_consumed": backend["gt_consumed"],
        })
    completed = len(rows)
    schedule_passed = sum(row["passes_anchor_schedule_gate"] for row in rows)
    safe = sum(row["passes_scene_safety"] for row in rows)
    improved = sum(row["passes_scene_improvement"] for row in rows)
    promoted = sum(row["promote"] for row in rows)
    payload = {
        "schema": "adaptive_anchor_orbbec5_matrix.v1",
        "scene_count": len(SCENES),
        "completed_count": completed,
        "total_frame_count": sum(row["frame_count"] for row in rows),
        "schedule_gate_passed_count": schedule_passed,
        "safe_count": safe,
        "improved_count": improved,
        "individual_promotion_count": promoted,
        "matrix_gate": {
            "all_five_completed": completed == 5,
            "all_five_schedule_gates_passed": schedule_passed == 5,
            "all_five_safe": safe == 5,
            "at_least_four_of_five_improved": improved >= 4,
        },
        "promote_to_production": (
            completed == 5
            and schedule_passed == 5
            and safe == 5
            and improved >= 4
        ),
        "rows": rows,
        "identity_fallback_used": any(
            row["identity_fallback_used"] for row in rows
        ),
        "gt_consumed": any(row["gt_consumed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
