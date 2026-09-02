#!/usr/bin/env python3
"""Export an audited manifest window as PointXYZRGB PCDs for G-CVO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pose_pipeline.contracts import load_manifest, sha256_file, stable_json_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="manifest ordinal")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--maximum-depth-m", type=float, default=5.0)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.output_dir.exists():
        raise FileExistsError(f"create-only output already exists: {args.output_dir}")
    selected = manifest.frames[args.start:args.start + args.count]
    if len(selected) != args.count or args.count < 2 or args.stride < 1:
        raise ValueError("requested export window is invalid or incomplete")
    args.output_dir.mkdir(parents=True)
    rows = []
    for ordinal, frame in enumerate(selected):
        color = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise ValueError(f"failed to read frame {frame.frame_id}")
        if frame.rotate_ccw:
            color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
            depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if color.shape[:2] != depth.shape:
            color = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        fx, fy, cx, cy = frame.intrinsics
        v, u = np.mgrid[0:depth.shape[0]:args.stride, 0:depth.shape[1]:args.stride]
        z = depth[::args.stride, ::args.stride].astype(np.float64) / manifest.depth_scale
        valid = np.isfinite(z) & (z > 0.10) & (z <= args.maximum_depth_m)
        z, u, v = z[valid], u[valid], v[valid]
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        colors = color[::args.stride, ::args.stride][valid][:, ::-1].astype(np.float64) / 255.0
        if len(points) < 500:
            raise ValueError(f"frame {frame.frame_id} has too few valid RGB-D points")
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        output = args.output_dir / f"{ordinal:06d}_frame_{frame.frame_id}.pcd"
        if not o3d.io.write_point_cloud(str(output), cloud, write_ascii=False, compressed=False):
            raise RuntimeError(f"failed to write {output}")
        rows.append({
            "ordinal": ordinal,
            "frame_id": frame.frame_id,
            "timestamp_us": frame.timestamp_us,
            "point_count": len(points),
            "pcd": str(output),
            "pcd_sha256": sha256_file(output),
        })
    unsigned = {
        "schema": "gcvo_rgbd_pcd_export.v1",
        "sequence_id": manifest.sequence_id,
        "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
        "start_ordinal": args.start,
        "count": args.count,
        "sampling_stride": args.stride,
        "maximum_depth_m": args.maximum_depth_m,
        "rows": rows,
        "gt_consumed": False,
    }
    with (args.output_dir / "export_audit.json").open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
