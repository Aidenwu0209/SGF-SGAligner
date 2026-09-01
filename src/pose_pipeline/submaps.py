"""Independent local RGB-D submaps and GT-free revisit proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import FrameRecord, PoseRecord, stable_json_sha256


@dataclass(frozen=True)
class SubmapConfig:
    anchor_stride: int = 80
    half_window: int = 20
    frame_stride: int = 4
    pixel_stride: int = 4
    minimum_depth_m: float = 0.30
    maximum_depth_m: float = 4.50
    voxel_m: float = 0.06
    maximum_points: int = 30_000


@dataclass(frozen=True)
class LoopProposalConfig:
    minimum_anchor_gap: int = 4
    maximum_initial_distance_m: float = 2.25
    maximum_pairs: int = 36


@dataclass(frozen=True)
class Submap:
    anchor_ordinal: int
    anchor_frame_id: int
    source_frame_ids: tuple[int, ...]
    points: np.ndarray
    points_sha256: str


def config_sha256(value: object) -> str:
    return stable_json_sha256(asdict(value))


def select_anchor_ordinals(count: int, stride: int) -> list[int]:
    if count < 2 or stride < 1:
        raise ValueError("invalid anchor selection request")
    anchors = list(range(0, count, stride))
    if anchors[-1] != count - 1:
        anchors.append(count - 1)
    return anchors


def _read_depth(frame: FrameRecord) -> np.ndarray:
    import cv2

    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"frame {frame.frame_id} depth is not uint16 HxW")
    if frame.rotate_ccw:
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return depth


def depth_points(
    frame: FrameRecord, depth_scale: float, config: SubmapConfig,
) -> np.ndarray:
    depth = _read_depth(frame)
    height, width = depth.shape
    vv, uu = np.mgrid[
        0:height:config.pixel_stride, 0:width:config.pixel_stride,
    ]
    z = depth[::config.pixel_stride, ::config.pixel_stride].astype(np.float64) / depth_scale
    valid = (
        np.isfinite(z)
        & (z >= config.minimum_depth_m)
        & (z <= config.maximum_depth_m)
    )
    fx, fy, cx, cy = frame.intrinsics
    if frame.rotate_ccw:
        # (u, v) -> (v, width - 1 - u) after CCW rotation.
        old_width = height
        fx, fy, cx, cy = fy, fx, cy, old_width - 1.0 - cx
    z = z[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    return np.ascontiguousarray(np.column_stack([x, y, z]))


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _points_sha256(points: np.ndarray) -> str:
    value = np.ascontiguousarray(points, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_submap(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    anchor_ordinal: int,
    depth_scale: float,
    config: SubmapConfig = SubmapConfig(),
) -> Submap:
    import open3d as o3d

    if not 0 <= anchor_ordinal < len(bound):
        raise IndexError("anchor ordinal outside bound trajectory")
    start = max(0, anchor_ordinal - config.half_window)
    stop = min(len(bound), anchor_ordinal + config.half_window + 1)
    selected = list(range(start, stop, config.frame_stride))
    if anchor_ordinal not in selected:
        selected.append(anchor_ordinal)
    selected.sort()
    anchor_pose = bound[anchor_ordinal][1].t_world_camera
    pieces, frame_ids = [], []
    for ordinal in selected:
        frame, pose = bound[ordinal]
        points = depth_points(frame, depth_scale, config)
        current_to_anchor = np.linalg.inv(anchor_pose) @ pose.t_world_camera
        pieces.append(_transform(points, current_to_anchor))
        frame_ids.append(frame.frame_id)
    points = np.concatenate(pieces, axis=0)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    points = np.asarray(cloud.voxel_down_sample(config.voxel_m).points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > config.maximum_points:
        indices = np.linspace(
            0, len(points) - 1, config.maximum_points, dtype=np.int64,
        )
        points = points[indices]
    if len(points) < 500:
        raise ValueError(f"anchor {anchor_ordinal} produced only {len(points)} points")
    points = np.ascontiguousarray(points)
    return Submap(
        anchor_ordinal=anchor_ordinal,
        anchor_frame_id=bound[anchor_ordinal][0].frame_id,
        source_frame_ids=tuple(frame_ids),
        points=points,
        points_sha256=_points_sha256(points),
    )


def save_submap(path: Path, submap: Submap, config: SubmapConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            points=submap.points,
            anchor_ordinal=np.asarray(submap.anchor_ordinal, dtype=np.int64),
            anchor_frame_id=np.asarray(submap.anchor_frame_id, dtype=np.int64),
            source_frame_ids=np.asarray(submap.source_frame_ids, dtype=np.int64),
            points_sha256=np.asarray(submap.points_sha256),
            config_sha256=np.asarray(config_sha256(config)),
        )


def propose_loop_pairs(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    anchors: Sequence[int],
    config: LoopProposalConfig = LoopProposalConfig(),
) -> list[dict]:
    centres = [bound[ordinal][1].t_world_camera[:3, 3] for ordinal in anchors]
    proposals = []
    for source in range(len(anchors)):
        for target in range(source + config.minimum_anchor_gap, len(anchors)):
            distance = float(np.linalg.norm(centres[source] - centres[target]))
            if distance > config.maximum_initial_distance_m:
                continue
            proposals.append({
                "source_anchor_index": source,
                "target_anchor_index": target,
                "source_ordinal": int(anchors[source]),
                "target_ordinal": int(anchors[target]),
                "source_frame_id": int(bound[anchors[source]][0].frame_id),
                "target_frame_id": int(bound[anchors[target]][0].frame_id),
                "initial_centre_distance_m": distance,
                "frame_gap": int(abs(
                    bound[anchors[target]][0].frame_id
                    - bound[anchors[source]][0].frame_id
                )),
            })
    proposals.sort(key=lambda row: (
        row["initial_centre_distance_m"], -row["frame_gap"],
        row["source_frame_id"], row["target_frame_id"],
    ))
    return proposals[:config.maximum_pairs]
