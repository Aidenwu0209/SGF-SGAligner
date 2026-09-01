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


REQUEST = struct.Struct("<8sIIQQddddIII")
RESPONSE = struct.Struct("<8s6i21dI")
REQUEST_MAGIC = b"XFREQ01\0"
RESPONSE_MAGIC = b"XFRSP01\0"


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = []
    while count:
        chunk = connection.recv(count)
        if not chunk:
            raise EOFError("DPV-SLAM worker closed the socket")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def _read_frame(frame: FrameRecord):
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
    return (
        np.ascontiguousarray(color, dtype=np.uint8),
        np.ascontiguousarray(depth.astype("<u2", copy=False)),
        (float(fx), float(fy), float(cx), float(cy)),
    )


def replay_manifest(
    *, manifest_path: Path, socket_path: Path, output_dir: Path,
    timeout_s: float = 30.0,
) -> dict:
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records, poses, valid_frames = [], [], []
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(timeout_s)
    connection.connect(str(socket_path))
    started = time.monotonic()
    try:
        for frame in manifest.frames:
            color, depth, intrinsics = _read_frame(frame)
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
        metadata={"frontend": "DPV-SLAM", "raw_frame_count": len(records)},
    )
    latencies = [row["latency_ms"] for row in records]
    summary = {
        "schema": "dpv_manifest_replay.v1",
        "sequence_id": manifest.sequence_id,
        "input_frame_count": len(records),
        "valid_pose_count": len(poses),
        "coverage": len(poses) / len(records),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "runtime_s": time.monotonic() - started,
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary
