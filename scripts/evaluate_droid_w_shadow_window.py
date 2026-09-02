#!/usr/bin/env python3
"""Evaluation-only comparison of DROID-W and the production frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_pipeline.contracts import load_manifest, load_trajectory, sha256_file
from pose_pipeline.evaluation import trajectory_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    frame_ids = {frame.frame_id for frame in manifest.frames}
    baseline, _ = load_trajectory(args.baseline)
    candidate, _ = load_trajectory(args.candidate)
    reference, _ = load_trajectory(args.reference)
    baseline = [row for row in baseline if row.frame_id in frame_ids]
    candidate = [row for row in candidate if row.frame_id in frame_ids]
    reference = [row for row in reference if row.frame_id in frame_ids]
    expected = len(manifest.frames)
    if len(baseline) != expected or len(candidate) != expected:
        raise ValueError("baseline/candidate do not completely cover the shadow window")
    if len(reference) < 2:
        raise ValueError("evaluation reference has fewer than two valid poses")

    value = {
        "schema": "droid_w_shadow_window_evaluation.v1",
        "sequence_id": manifest.sequence_id,
        "window_frame_count": expected,
        "reference_frame_count": len(reference),
        "baseline": trajectory_metrics(baseline, reference),
        "candidate": trajectory_metrics(candidate, reference),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "manifest": args.manifest,
                "baseline": args.baseline,
                "candidate": args.candidate,
                "reference": args.reference,
            }.items()
        },
        "gt_role": "evaluation_only",
        "promotion_requires_full_sequence_and_refusion": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "baseline_relative_translation_rmse_m":
            value["baseline"]["relative_translation_m"]["rmse"],
        "candidate_relative_translation_rmse_m":
            value["candidate"]["relative_translation_m"]["rmse"],
        "baseline_relative_rotation_rmse_deg":
            value["baseline"]["relative_rotation_deg"]["rmse"],
        "candidate_relative_rotation_rmse_deg":
            value["candidate"]["relative_rotation_deg"]["rmse"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
