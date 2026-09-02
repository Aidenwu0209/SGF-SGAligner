#!/usr/bin/env python3
"""Evaluation-only comparison of DPV and G-CVO pair deltas against dataset pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import load_trajectory, sha256_file, validate_se3
from pose_pipeline.robust_backend import transform_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--gcvo-result", type=Path, required=True)
    parser.add_argument("--evaluation-pose-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, _ = load_trajectory(args.baseline_trajectory)
    baseline_by_id = {row.frame_id: row.t_world_camera for row in baseline}
    candidate = json.loads(args.gcvo_result.read_text())
    rows = []
    for pair in candidate["rows"]:
        source_id, target_id = pair["source_frame_id"], pair["target_frame_id"]
        gt_source = validate_se3(np.loadtxt(args.evaluation_pose_root / f"{source_id}.txt"), "GT source")
        gt_target = validate_se3(np.loadtxt(args.evaluation_pose_root / f"{target_id}.txt"), "GT target")
        gt_delta = np.linalg.inv(gt_source) @ gt_target
        baseline_delta = np.linalg.inv(baseline_by_id[source_id]) @ baseline_by_id[target_id]
        gcvo_delta = validate_se3(pair["T_source_target"], "G-CVO delta")
        baseline_rotation, baseline_translation = transform_distance(gt_delta, baseline_delta)
        gcvo_rotation, gcvo_translation = transform_distance(gt_delta, gcvo_delta)
        disagreement_rotation, disagreement_translation = transform_distance(baseline_delta, gcvo_delta)
        rows.append({
            "source_frame_id": source_id,
            "target_frame_id": target_id,
            "baseline_translation_error_m": baseline_translation,
            "gcvo_translation_error_m": gcvo_translation,
            "baseline_rotation_error_deg": baseline_rotation,
            "gcvo_rotation_error_deg": gcvo_rotation,
            "gcvo_translation_wins": gcvo_translation < baseline_translation,
            "gcvo_rotation_wins": gcvo_rotation < baseline_rotation,
            "baseline_disagreement_translation_m": disagreement_translation,
            "baseline_disagreement_rotation_deg": disagreement_rotation,
        })
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)
    summary = {
        "pair_count": len(rows),
        "translation_win_count": sum(row["gcvo_translation_wins"] for row in rows),
        "rotation_win_count": sum(row["gcvo_rotation_wins"] for row in rows),
        "both_win_count": sum(row["gcvo_translation_wins"] and row["gcvo_rotation_wins"] for row in rows),
        "baseline_translation_rmse_m": float(np.sqrt(np.mean(values("baseline_translation_error_m") ** 2))),
        "gcvo_translation_rmse_m": float(np.sqrt(np.mean(values("gcvo_translation_error_m") ** 2))),
        "baseline_rotation_rmse_deg": float(np.sqrt(np.mean(values("baseline_rotation_error_deg") ** 2))),
        "gcvo_rotation_rmse_deg": float(np.sqrt(np.mean(values("gcvo_rotation_error_deg") ** 2))),
    }
    output = {
        "schema": "gcvo_pair_window_evaluation.v1",
        "summary": summary,
        "rows": rows,
        "inputs": {
            "baseline_sha256": sha256_file(args.baseline_trajectory),
            "gcvo_result_sha256": sha256_file(args.gcvo_result),
        },
        "gt_role": "evaluation_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
