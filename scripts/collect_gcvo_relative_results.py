#!/usr/bin/env python3
"""Collect provider text outputs into the no-GT G-CVO result contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import load_manifest, sha256_file, stable_json_sha256, validate_se3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--provider-dir", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--gcvo-commit", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    frames = manifest.frames[args.start:args.start + args.count]
    if len(frames) != args.count or args.count < 2 or len(args.gcvo_commit) != 40:
        raise ValueError("result window or provider commit is invalid")
    rows = []
    for source, target in zip(frames, frames[1:]):
        stem = f"frame_{source.frame_id}_to_{target.frame_id}"
        trajectory_path = args.provider_dir / f"{stem}.txt"
        log_path = args.provider_dir / f"{stem}.log"
        raw = np.loadtxt(trajectory_path, dtype=np.float64, ndmin=2)
        if raw.shape != (2, 12):
            raise ValueError(f"bad G-CVO provider trajectory: {trajectory_path}")
        matrix = np.eye(4)
        matrix[:3] = raw[1].reshape(3, 4)
        matrix = validate_se3(matrix, stem)
        log = log_path.read_text()
        match = re.search(r"iters=(\d+) sec=([0-9.eE+-]+) code=(-?\d+)", log)
        if match is None or int(match.group(3)) != 0:
            raise ValueError(f"G-CVO provider did not report success: {log_path}")
        rows.append({
            "source_frame_id": source.frame_id,
            "target_frame_id": target.frame_id,
            "T_source_target": matrix.tolist(),
            "iterations": int(match.group(1)),
            "registration_seconds": float(match.group(2)),
            "provider_trajectory_sha256": sha256_file(trajectory_path),
            "provider_log_sha256": sha256_file(log_path),
        })
    unsigned = {
        "schema": "gcvo_relative_refinement.v1",
        "sequence_id": manifest.sequence_id,
        "matrix_convention": "p_source=T_source_target@p_target",
        "gcvo_commit": args.gcvo_commit,
        "gcvo_config_sha256": sha256_file(args.provider_config),
        "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
        "rows": rows,
        "gt_consumed": False,
        "identity_fallback_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
