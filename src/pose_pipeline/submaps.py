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
    policy: str = "distance_topk"
    minimum_anchor_gap: int = 4
    maximum_initial_distance_m: float = 2.25
    maximum_pairs: int = 36
    appearance_mutual_top_k: int = 2
    maximum_pairs_per_anchor: int = 4
    temporal_bin_count: int = 3

    def __post_init__(self) -> None:
        if self.policy not in {"distance_topk", "hybrid36"}:
            raise ValueError("loop proposal policy must be distance_topk or hybrid36")
        if self.minimum_anchor_gap < 1:
            raise ValueError("minimum anchor gap must be positive")
        if (
            not np.isfinite(self.maximum_initial_distance_m)
            or self.maximum_initial_distance_m <= 0.0
        ):
            raise ValueError("maximum initial distance must be finite and positive")
        if self.maximum_pairs < 1:
            raise ValueError("maximum loop pairs must be positive")
        if self.appearance_mutual_top_k < 1:
            raise ValueError("appearance mutual top-k must be positive")
        if self.maximum_pairs_per_anchor < 1:
            raise ValueError("maximum pairs per anchor must be positive")
        if self.temporal_bin_count < 1:
            raise ValueError("temporal bin count must be positive")


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
    *,
    appearance_descriptors: np.ndarray | None = None,
) -> list[dict]:
    if config.policy == "hybrid36":
        return _propose_hybrid_pairs(
            bound, anchors, config, appearance_descriptors,
        )
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


def _mutual_appearance_pairs(
    descriptors: np.ndarray, minimum_gap: int, top_k: int,
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("appearance descriptors must be finite anchor_count x D")
    length = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(length <= 1e-12):
        raise ValueError("appearance descriptors must have non-zero norm")
    values = values / length
    similarity = values @ values.T
    neighbours: list[set[int]] = []
    for source in range(len(values)):
        eligible = [
            target for target in range(len(values))
            if abs(target - source) >= minimum_gap
        ]
        eligible.sort(key=lambda target: (-float(similarity[source, target]), target))
        neighbours.append(set(eligible[:top_k]))
    pairs = {
        (source, target)
        for source in range(len(values))
        for target in neighbours[source]
        if source < target and source in neighbours[target]
    }
    return similarity, pairs


def _propose_hybrid_pairs(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    anchors: Sequence[int],
    config: LoopProposalConfig,
    appearance_descriptors: np.ndarray | None,
) -> list[dict]:
    if appearance_descriptors is None:
        raise ValueError("hybrid36 requires appearance descriptors")
    if len(appearance_descriptors) != len(anchors):
        raise ValueError("appearance descriptor count must match anchors")
    similarity, appearance_pairs = _mutual_appearance_pairs(
        appearance_descriptors,
        config.minimum_anchor_gap,
        config.appearance_mutual_top_k,
    )
    centres = [bound[ordinal][1].t_world_camera[:3, 3] for ordinal in anchors]
    candidates = []
    denominator = max(1, len(anchors) - 1)
    for source in range(len(anchors)):
        for target in range(source + config.minimum_anchor_gap, len(anchors)):
            distance = float(np.linalg.norm(centres[source] - centres[target]))
            distance_eligible = distance <= config.maximum_initial_distance_m
            appearance_eligible = (source, target) in appearance_pairs
            if not distance_eligible and not appearance_eligible:
                continue
            sources = []
            if distance_eligible:
                sources.append("trajectory_distance")
            if appearance_eligible:
                sources.append("clip_mutual_topk")
            distance_score = max(
                0.0, 1.0 - distance / config.maximum_initial_distance_m,
            )
            appearance_score = float((similarity[source, target] + 1.0) / 2.0)
            span_fraction = (target - source) / denominator
            combined = (
                max(distance_score, appearance_score)
                + (0.15 if len(sources) == 2 else 0.0)
                + 0.05 * span_fraction
            )
            temporal_bin = min(
                config.temporal_bin_count - 1,
                int(span_fraction * config.temporal_bin_count),
            )
            candidates.append({
                "schema": "loop_proposal.v2",
                "proposal_policy": "hybrid36",
                "proposal_sources": sources,
                "source_anchor_index": source,
                "target_anchor_index": target,
                "source_ordinal": int(anchors[source]),
                "target_ordinal": int(anchors[target]),
                "source_frame_id": int(bound[anchors[source]][0].frame_id),
                "target_frame_id": int(bound[anchors[target]][0].frame_id),
                "initial_centre_distance_m": distance,
                "appearance_cosine_similarity": float(similarity[source, target]),
                "distance_score": distance_score,
                "appearance_score": appearance_score,
                "combined_score": combined,
                "span_fraction": span_fraction,
                "temporal_bin": temporal_bin,
                "frame_gap": int(abs(
                    bound[anchors[target]][0].frame_id
                    - bound[anchors[source]][0].frame_id
                )),
                "ranking_reason": "union_score_then_temporal_diversity",
            })
    buckets: list[list[dict]] = [[] for _ in range(config.temporal_bin_count)]
    for row in candidates:
        buckets[int(row["temporal_bin"])].append(row)
    for bucket in buckets:
        bucket.sort(key=lambda row: (
            -float(row["combined_score"]),
            -int(len(row["proposal_sources"])),
            -int(row["frame_gap"]),
            int(row["source_frame_id"]),
            int(row["target_frame_id"]),
        ))
    selected: list[dict] = []
    degrees = [0] * len(anchors)
    while len(selected) < config.maximum_pairs:
        progressed = False
        for bucket in reversed(buckets):
            while bucket:
                row = bucket.pop(0)
                source = int(row["source_anchor_index"])
                target = int(row["target_anchor_index"])
                if (
                    degrees[source] >= config.maximum_pairs_per_anchor
                    or degrees[target] >= config.maximum_pairs_per_anchor
                ):
                    continue
                selected.append(row)
                degrees[source] += 1
                degrees[target] += 1
                progressed = True
                break
            if len(selected) >= config.maximum_pairs:
                break
        if not progressed:
            break
    return selected
