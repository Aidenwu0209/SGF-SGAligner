"""Replay a GT-free manifest through the existing DPV-SLAM socket worker."""

from __future__ import annotations

import json
from pathlib import Path
import socket
import struct
import time

import numpy as np

from .contracts import (
    FrameRecord,
    PoseRecord,
    SequenceManifest,
    load_manifest,
    write_manifest,
    write_trajectory,
)
from .depth_filter import (
    DepthFilterAccumulator,
    DepthFilterConfig,
    apply_depth_filter,
)


REQUEST = struct.Struct("<8sIIQQddddIII")
RESPONSE = struct.Struct("<8s6i21dI")
REQUEST_MAGIC = b"XFREQ01\0"
RESPONSE_MAGIC = b"XFRSP01\0"


def _pad_rgbd_to_multiple(
    color: np.ndarray, depth: np.ndarray, multiple: int = 16,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Pad only the bottom/right borders without changing pixel coordinates.

    DPV requires both image dimensions to be divisible by 16.  Zero padding
    preserves the original principal point, unlike resizing or centered
    padding, and zero depth keeps padded pixels outside metric estimation.
    """
    if multiple < 1:
        raise ValueError("padding multiple must be positive")
    if color.ndim != 3 or color.shape[2] != 3 or depth.ndim != 2:
        raise ValueError("RGB-D padding requires HxWx3 color and HxW depth")
    if color.shape[:2] != depth.shape:
        raise ValueError("RGB-D padding requires aligned image sizes")
    height, width = depth.shape
    pad_bottom = (-height) % multiple
    pad_right = (-width) % multiple
    if pad_bottom or pad_right:
        color = np.pad(
            color, ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="constant", constant_values=0,
        )
        depth = np.pad(
            depth, ((0, pad_bottom), (0, pad_right)),
            mode="constant", constant_values=0,
        )
    return color, depth, {
        "original_width": int(width),
        "original_height": int(height),
        "padded_width": int(width + pad_right),
        "padded_height": int(height + pad_bottom),
        "pad_right_px": int(pad_right),
        "pad_bottom_px": int(pad_bottom),
    }


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = []
    while count:
        chunk = connection.recv(count)
        if not chunk:
            raise EOFError("DPV-SLAM worker closed the socket")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _read_frame(
    frame: FrameRecord, depth_scale: float = 1000.0,
    depth_filter_config: DepthFilterConfig = DepthFilterConfig(),
    *, return_filter_stats: bool = False,
):
    import cv2

    color = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise FileNotFoundError(f"missing RGB-D frame {frame.frame_id}")
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"bad depth frame {frame.frame_id}: {depth.shape}/{depth.dtype}")
    fx, fy, cx, cy = frame.intrinsics
    if frame.rotate_ccw:
        old_width = depth.shape[1]
        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fx, fy, cx, cy = fy, fx, cy, old_width - 1.0 - cx
    height, width = depth.shape
    if color.shape[:2] != depth.shape:
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    depth, filter_stats = apply_depth_filter(
        depth, depth_scale, depth_filter_config,
    )
    color, depth, preprocessing = _pad_rgbd_to_multiple(color, depth)
    preprocessing["depth_filter"] = filter_stats.as_dict()
    result = (
        np.ascontiguousarray(color, dtype=np.uint8),
        np.ascontiguousarray(depth.astype("<u2", copy=False)),
        (float(fx), float(fy), float(cx), float(cy)),
        preprocessing,
    )
    return result + (filter_stats,) if return_filter_stats else result


def replay_manifest(
    *, manifest_path: Path, socket_path: Path, output_dir: Path,
    timeout_s: float = 30.0,
    depth_filter_config: DepthFilterConfig = DepthFilterConfig(),
) -> dict:
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records, poses, valid_frames = [], [], []
    depth_filter_audit = DepthFilterAccumulator(depth_filter_config)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_s)
    connection.connect(str(socket_path))
    started = time.monotonic()
    try:
        for frame in manifest.frames:
            color, depth, intrinsics, preprocessing, filter_stats = _read_frame(
                frame, manifest.depth_scale, depth_filter_config,
                return_filter_stats=True,
            )
            depth_filter_audit.update(frame.frame_id, filter_stats)
            height, width = depth.shape
            header = REQUEST.pack(
                REQUEST_MAGIC, width, height, frame.frame_id,
                frame.timestamp_us, *intrinsics, 0, color.nbytes, depth.nbytes,
            )
            frame_started = time.monotonic()
            connection.sendall(header)
            connection.sendall(color.tobytes(order="C"))
            connection.sendall(depth.tobytes(order="C"))
            values = RESPONSE.unpack(_recv_exact(connection, RESPONSE.size))
            if values[0] != RESPONSE_MAGIC:
                raise RuntimeError(f"bad DPV response magic: {values[0]!r}")
            reason_length = int(values[28])
            reason = _recv_exact(connection, reason_length).decode(
                "utf-8", "replace",
            ) if reason_length else ""
            t_camera_world = np.asarray(
                values[7:23], dtype=np.float64,
            ).reshape(4, 4)
            valid = bool(
                bool(values[1]) and bool(np.isfinite(t_camera_world).all())
            )
            row = {
                "frame_id": frame.frame_id,
                "timestamp_us": frame.timestamp_us,
                "valid": valid,
                "initialized": bool(values[2]),
                "correspondences": int(values[3]),
                "inliers": int(values[4]),
                "keyframe": bool(values[5]),
                "imu_used": bool(values[6]),
                "translation_m": float(values[23]),
                "rotation_deg": float(values[24]),
                "inlier_ratio": float(values[25]),
                "reprojection_rmse_px": float(values[26]),
                "depth_inlier_ratio": float(values[27]),
                "preprocessing": preprocessing,
                "reason": reason,
                "latency_ms": (time.monotonic() - frame_started) * 1000.0,
            }
            records.append(row)
            if valid:
                t_world_camera = np.linalg.inv(t_camera_world)
                poses.append(PoseRecord(
                    frame_id=frame.frame_id,
                    timestamp_us=frame.timestamp_us,
                    t_world_camera=t_world_camera,
                    valid=True,
                    source="DPV-SLAM",
                ))
                valid_frames.append(frame)
    finally:
        connection.close()
    with (output_dir / "responses.jsonl").open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    if not poses:
        raise RuntimeError("DPV-SLAM produced no valid pose")
    tracked = SequenceManifest(
        dataset=manifest.dataset,
        sequence_id=manifest.sequence_id,
        root=manifest.root,
        depth_scale=manifest.depth_scale,
        frames=tuple(valid_frames),
        source=f"{manifest.source}+dpv_valid_only",
    )
    write_manifest(output_dir / "tracked_manifest.json", tracked)
    write_trajectory(
        output_dir / "trajectory.json", poses,
        sequence_id=manifest.sequence_id, arm="baseline",
        metadata={
            "frontend": "DPV-SLAM",
            "raw_frame_count": len(records),
            "depth_filter_parameters_sha256": (
                depth_filter_config.parameters_sha256
            ),
        },
    )
    latencies = [row["latency_ms"] for row in records]
    summary = {
        "schema": "dpv_manifest_replay.v2",
        "sequence_id": manifest.sequence_id,
        "input_frame_count": len(records),
        "valid_pose_count": len(poses),
        "coverage": len(poses) / len(records),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "runtime_s": time.monotonic() - started,
        "depth_filter": depth_filter_audit.summary(),
        "dimension_padding": {
            "method": "zero_pad_bottom_right_to_multiple_16",
            "principal_point_adjusted": False,
            "frames_padded": sum(
                bool(row["preprocessing"]["pad_right_px"])
                or bool(row["preprocessing"]["pad_bottom_px"])
                for row in records
            ),
        },
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary
