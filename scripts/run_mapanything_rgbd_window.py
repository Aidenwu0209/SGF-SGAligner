#!/usr/bin/env python3
"""Run one official MapAnything RGB-D window without exposing GT poses.

This script intentionally lives outside the lightweight adapter process.  The
heavy model environment writes a small, auditable NPZ boundary artifact that
``pose_pipeline adapt-mapanything`` can consume later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-ids", default=",".join(str(i) for i in range(8)))
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--intrinsic", type=Path)
    parser.add_argument(
        "--pose-trajectory", type=Path,
        help="Optional pose_trajectory.v1 used as OpenCV cam2world conditioning",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--cuda-alloc-conf",
        help="Optional PYTORCH_CUDA_ALLOC_CONF override; leave unset on Jetson",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_path(directory: Path, frame_id: int, suffixes: tuple[str, ...]) -> Path:
    for suffix in suffixes:
        candidate = directory / f"{frame_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing frame {frame_id} in {directory}")


def load_intrinsic(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic must be a finite 3x3 or 4x4 matrix")
    return matrix


def prepare_raw_views(
    root: Path,
    frame_ids: list[int],
    intrinsic: np.ndarray,
    depth_scale: float,
    pose_by_frame: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth scale must be finite and positive")
    views = []
    for frame_id in frame_ids:
        color_path = frame_path(root / "color", frame_id, (".jpg", ".png", ".jpeg"))
        depth_path = frame_path(root / "depth", frame_id, (".png", ".tiff", ".tif"))
        image = Image.open(color_path).convert("RGB")
        depth = np.asarray(Image.open(depth_path), dtype=np.float32) / depth_scale
        if depth.ndim != 2:
            raise ValueError(f"depth frame {frame_id} is not single-channel")
        if depth.shape != image.size[::-1]:
            depth = np.asarray(
                Image.fromarray(depth).resize(image.size, resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            )
        view = {
            "img": image,
            "intrinsics": intrinsic.copy(),
            "depth_z": depth,
            # Do not materialize ``is_metric_scale`` here.  The official
            # preprocessor defaults an omitted value to scalar ``True``;
            # passing a one-element NumPy array is later interpreted as an
            # integer index by the official model and breaks a valid window.
        }
        if pose_by_frame is not None:
            view["camera_poses"] = pose_by_frame[frame_id].copy()
        views.append(view)
    return views


def load_pose_conditioning(path: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from pose_pipeline.contracts import load_trajectory

    rows, _ = load_trajectory(path)
    available = {row.frame_id: row.t_world_camera for row in rows}
    missing = [frame_id for frame_id in frame_ids if frame_id not in available]
    if missing:
        raise ValueError(f"pose conditioning misses requested frames: {missing}")
    return {frame_id: available[frame_id] for frame_id in frame_ids}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frame_ids = [int(value) for value in args.frame_ids.split(",") if value.strip()]
    if len(frame_ids) not in {8, 16} or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("MapAnything window must contain 8 or 16 unique frame IDs")
    input_root = args.input_root.resolve()
    checkpoint = args.checkpoint.resolve()
    intrinsic_path = (
        args.intrinsic.resolve()
        if args.intrinsic
        else input_root / "intrinsic" / "intrinsic_color.txt"
    )
    forbidden_pose = input_root / "pose"
    if forbidden_pose.exists():
        raise ValueError(f"inference root must not expose GT pose directory: {forbidden_pose}")

    if args.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", args.cuda_alloc_conf)
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    intrinsic = load_intrinsic(intrinsic_path)
    pose_by_frame = (
        load_pose_conditioning(args.pose_trajectory.resolve(), frame_ids)
        if args.pose_trajectory is not None else None
    )
    raw_views = prepare_raw_views(
        input_root, frame_ids, intrinsic, args.depth_scale, pose_by_frame,
    )
    processed_views = preprocess_inputs(raw_views, resolution_set=518, verbose=True)

    started = time.perf_counter()
    model_started = time.perf_counter()
    model = MapAnything.from_pretrained(str(checkpoint)).to(args.device).eval()
    model_load_seconds = time.perf_counter() - model_started
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        predictions = model.infer(
            processed_views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype=args.amp_dtype,
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
            use_multiview_confidence=False,
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=pose_by_frame is None,
            ignore_depth_scale_inputs=False,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

    def stack(name: str) -> np.ndarray:
        values = []
        for prediction in predictions:
            value = prediction[name]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if value.shape[0] == 1:
                value = value[0]
            values.append(value)
        return np.stack(values)

    poses = stack("camera_poses").astype(np.float64)
    if poses.shape != (len(frame_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"invalid camera pose output shape {poses.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(
            stream,
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            camera_poses=poses,
            depth_z=stack("depth_z").astype(np.float32),
            confidence=stack("conf").astype(np.float32),
            pts3d=stack("pts3d").astype(np.float32),
            metric_scaling_factor=stack("metric_scaling_factor").astype(np.float32),
        )
    runtime = {
        "schema": "mapanything_official_window_runtime.v1",
        "status": "completed",
        "input_root": str(input_root),
        "frame_ids": frame_ids,
        "gt_consumed": False,
        "input_mode": (
            "conditioned_on_dpv_pose" if pose_by_frame is not None
            else "independent_rgb_intrinsics_depth"
        ),
        "pose_trajectory": (
            str(args.pose_trajectory.resolve())
            if args.pose_trajectory is not None else None
        ),
        "pose_trajectory_sha256": (
            sha256_file(args.pose_trajectory.resolve())
            if args.pose_trajectory is not None else None
        ),
        "resolution_set": 518,
        "memory_efficient_inference": True,
        "minibatch_size": 1,
        "amp_dtype": args.amp_dtype,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if args.device.startswith("cuda") else 0
        ),
        "output_sha256": sha256_file(args.output),
    }
    runtime_path = args.output.with_suffix(".runtime.json")
    with runtime_path.open("x", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
