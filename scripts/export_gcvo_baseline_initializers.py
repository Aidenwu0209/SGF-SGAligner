#!/usr/bin/env python3
"""Export DPV relative poses as create-only G-CVO pair initializers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import load_trajectory, sha256_file, stable_json_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    rows, payload = load_trajectory(args.trajectory)
    selected = rows[args.start:args.start + args.count]
    if len(selected) != args.count or args.count < 2:
        raise ValueError("initializer window is invalid or incomplete")
    if args.output_dir.exists():
        raise FileExistsError(f"create-only output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    audit_rows = []
    for source, target in zip(selected, selected[1:]):
        delta = np.linalg.inv(source.t_world_camera) @ target.t_world_camera
        path = args.output_dir / f"frame_{source.frame_id}_to_{target.frame_id}.txt"
        with path.open("x", encoding="utf-8") as stream:
            for matrix_row in delta:
                stream.write(" ".join(f"{value:.12g}" for value in matrix_row) + "\n")
        audit_rows.append({
            "source_frame_id": source.frame_id,
            "target_frame_id": target.frame_id,
            "initializer": str(path),
            "initializer_sha256": sha256_file(path),
        })
    unsigned = {
        "schema": "gcvo_baseline_initializers.v1",
        "trajectory_payload_sha256": payload["payload_sha256"],
        "start_ordinal": args.start,
        "count": args.count,
        "rows": audit_rows,
        "gt_consumed": False,
    }
    with (args.output_dir / "initializer_audit.json").open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
