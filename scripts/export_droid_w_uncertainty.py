#!/usr/bin/env python3
"""Convert DROID-W keyframe uncertainty into a full manifest sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import load_manifest, sha256_file


def droid_uncertainty_to_dynamic(value: np.ndarray) -> np.ndarray:
    """Match the official UD-BA weighting: dynamic = 1 - w_uncer."""
    scaled = np.maximum(45.0 * np.asarray(value, dtype=np.float32) - 35.0, 0.1)
    static_weight = np.clip(1.0 / scaled, 0.0, 1.0)
    return np.clip(1.0 - static_weight, 0.0, 1.0)


def interpolate_keyframes(
    keyframe_ordinals: np.ndarray, keyframe_values: np.ndarray, frame_count: int,
) -> np.ndarray:
    order = np.argsort(keyframe_ordinals, kind="stable")
    times = np.asarray(keyframe_ordinals, dtype=np.float64)[order]
    values = np.asarray(keyframe_values, dtype=np.float32)[order]
    if len(times) < 2 or len(np.unique(times)) != len(times):
        raise ValueError("DROID-W needs at least two unique uncertainty keyframes")
    if times[0] < 0 or times[-1] >= frame_count:
        raise ValueError("DROID-W keyframe ordinal escapes manifest")
    output = np.empty((frame_count, *values.shape[1:]), dtype=np.float32)
    for ordinal in range(frame_count):
        right = int(np.searchsorted(times, ordinal, side="left"))
        if right == 0:
            output[ordinal] = values[0]
        elif right == len(times):
            output[ordinal] = values[-1]
        else:
            left = right - 1
            alpha = float((ordinal - times[left]) / (times[right] - times[left]))
            output[ordinal] = (1.0 - alpha) * values[left] + alpha * values[right]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--droid-video", type=Path, required=True)
    parser.add_argument("--droid-commit", required=True)
    parser.add_argument("--droid-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if len(args.droid_commit) != 40:
        raise ValueError("DROID-W commit must be a full SHA")
    with np.load(args.droid_video) as video:
        timestamps = np.asarray(video["timestamps"], dtype=np.float64)
        raw_uncertainty = np.asarray(video["uncertainties"], dtype=np.float32)
    if raw_uncertainty.ndim != 3 or len(timestamps) != len(raw_uncertainty):
        raise ValueError("DROID-W video uncertainty is malformed")
    ordinals = np.rint(timestamps).astype(np.int64)
    if not np.allclose(timestamps, ordinals, atol=1e-5):
        raise ValueError("DROID-W timestamps are not manifest ordinals")
    keyframe_dynamic = droid_uncertainty_to_dynamic(raw_uncertainty)
    dynamic = interpolate_keyframes(ordinals, keyframe_dynamic, len(manifest.frames))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"create-only output exists: {args.output}")
    np.savez_compressed(
        args.output,
        frame_ids=np.asarray([frame.frame_id for frame in manifest.frames], dtype=np.int64),
        dynamic_uncertainty=dynamic,
        provider=np.asarray("DROID-W-UD-BA"),
        model_commit=np.asarray(args.droid_commit),
        checkpoint_sha256=np.asarray(sha256_file(args.droid_checkpoint)),
        source_video_sha256=np.asarray(sha256_file(args.droid_video)),
        temporal_interpolation=np.asarray("linear_keyframe_ordinal_nearest_edges"),
        uncertainty_conversion=np.asarray("1-clip(1/max(45*u-35,0.1),0,1)"),
        gt_consumed=np.asarray(False),
    )


if __name__ == "__main__":
    main()
