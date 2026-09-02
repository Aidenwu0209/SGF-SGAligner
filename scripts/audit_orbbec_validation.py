#!/usr/bin/env python3
"""Independently audit a create-only Orbbec validation output."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_pipeline.contracts import (  # noqa: E402
    load_manifest,
    load_trajectory,
    sha256_file,
    validate_se3,
)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def _rotation_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _audit_sequence(sequence_dir: Path, result: dict) -> dict:
    manifest = load_manifest(Path(result["manifest"]))
    poses, payload = load_trajectory(sequence_dir / "frontend" / "trajectory.json")
    expected_ids = [frame.frame_id for frame in manifest.frames]
    actual_ids = [pose.frame_id for pose in poses]
    matrices = [validate_se3(pose.t_world_camera, f"frame {pose.frame_id}") for pose in poses]
    translations = []
    rotations = []
    for previous, current in zip(matrices, matrices[1:]):
        delta = np.linalg.inv(previous) @ current
        translations.append(float(np.linalg.norm(delta[:3, 3])))
        rotations.append(_rotation_deg(delta[:3, :3]))
    source_counts = Counter(pose.source for pose in poses)
    complete = expected_ids == actual_ids
    return {
        "sequence_id": manifest.sequence_id,
        "input_frames": len(expected_ids),
        "trajectory_frames": len(actual_ids),
        "frame_ids_exact": complete,
        "missing_frame_ids": sorted(set(expected_ids) - set(actual_ids)),
        "extra_frame_ids": sorted(set(actual_ids) - set(expected_ids)),
        "all_se3_valid": True,
        "max_step_translation_m": max(translations, default=0.0),
        "p99_step_translation_m": float(np.percentile(translations, 99)) if translations else 0.0,
        "max_step_rotation_deg": max(rotations, default=0.0),
        "p99_step_rotation_deg": float(np.percentile(rotations, 99)) if rotations else 0.0,
        "source_counts": dict(sorted(source_counts.items())),
        "stable_pose_sha256_q1e7": payload["stable_pose_sha256_q1e7"],
        "identity_fallback_used": payload["identity_fallback_used"],
        "gt_consumed": bool(result["gt_consumed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    summary = json.loads((run_root / "summary.json").read_text())
    sequences = []
    for result in summary["results"]:
        sequences.append(_audit_sequence(run_root / result["sequence_id"], result))
    total_input = sum(row["input_frames"] for row in sequences)
    total_poses = sum(row["trajectory_frames"] for row in sequences)
    audit = {
        "schema": "orbbec_validation_independent_audit.v1",
        "base_commit": args.base_commit,
        "loop_closure": summary["loop_closure"],
        "sequence_count": len(sequences),
        "total_input_frames": total_input,
        "total_trajectory_frames": total_poses,
        "all_frame_ids_exact": all(row["frame_ids_exact"] for row in sequences),
        "all_se3_valid": all(row["all_se3_valid"] for row in sequences),
        "identity_fallback_used": any(row["identity_fallback_used"] for row in sequences),
        "gt_consumed": any(row["gt_consumed"] for row in sequences),
        "artifacts": {
            "worker": {"path": str(args.worker.resolve()), "sha256": sha256_file(args.worker)},
            "metric_config": {
                "path": str(args.metric_config.resolve()),
                "sha256": sha256_file(args.metric_config),
            },
            "runner": {"path": str(args.runner.resolve()), "sha256": sha256_file(args.runner)},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "nvidia_smi": _command_output([
                "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader",
            ]),
        },
        "sequences": sequences,
    }
    passed = (
        len(sequences) == 6
        and total_input == total_poses
        and audit["all_frame_ids_exact"]
        and audit["all_se3_valid"]
        and not audit["identity_fallback_used"]
        and not audit["gt_consumed"]
        and summary["all_trajectories_complete"]
    )
    audit["passed"] = passed
    _write_json(run_root / "independent_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
