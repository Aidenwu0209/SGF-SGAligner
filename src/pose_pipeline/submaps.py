"""Independent local RGB-D submaps and GT-free revisit proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

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
class AdaptiveAnchorConfig:
    """GT-free geometric anchor policy for metric RGB-D trajectories.

    Flow is induced by reprojecting the current sensor-depth samples into the
    latest accepted anchor using the supplied ``T_world_camera`` poses.  The
    maximum gap remains a hard bound, so a quiet or textureless sequence does
    not silently lose temporal coverage.
    """

    # Match the default submap half-window: denser anchors would mostly build
    # redundant local clouds while multiplying registration cost.
    minimum_gap: int = 20
    maximum_gap: int = 80
    pixel_stride: int = 8
    flow_threshold_px: float = 24.0
    minimum_overlap_fraction: float = 0.35
    translation_threshold_m: float = 0.25
    rotation_threshold_deg: float = 12.0
    minimum_valid_depth_samples: int = 256

    def __post_init__(self) -> None:
        if self.minimum_gap < 1:
            raise ValueError("adaptive minimum gap must be positive")
        if self.maximum_gap < self.minimum_gap:
            raise ValueError("adaptive maximum gap must be >= minimum gap")
        if self.pixel_stride < 1:
            raise ValueError("adaptive pixel stride must be positive")
        positive = {
            "flow threshold": self.flow_threshold_px,
            "translation threshold": self.translation_threshold_m,
            "rotation threshold": self.rotation_threshold_deg,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"adaptive {name} must be finite and positive")
        if not 0.0 <= self.minimum_overlap_fraction <= 1.0:
            raise ValueError("adaptive minimum overlap fraction must be in [0, 1]")
        if self.minimum_valid_depth_samples < 1:
            raise ValueError("adaptive minimum valid depth samples must be positive")


@dataclass(frozen=True)
class LoopProposalConfig:
    minimum_anchor_gap: int = 4
    maximum_initial_distance_m: float = 2.25
    maximum_pairs: int = 36

    def __post_init__(self) -> None:
        if self.minimum_anchor_gap < 1:
            raise ValueError("minimum anchor gap must be positive")
        if (
            not np.isfinite(self.maximum_initial_distance_m)
            or self.maximum_initial_distance_m <= 0.0
        ):
            raise ValueError("maximum initial distance must be finite and positive")
        if self.maximum_pairs < 1:
            raise ValueError("maximum loop pairs must be positive")


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


def _intrinsics_for_loaded_depth(
    frame: FrameRecord, depth_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    height, _ = depth_shape
    fx, fy, cx, cy = frame.intrinsics
    if frame.rotate_ccw:
        # The loaded image has already been rotated.  Its height equals the
        # original width used by the pinhole CCW transform.
        original_width = height
        fx, fy, cx, cy = fy, fx, cy, original_width - 1.0 - cx
    return float(fx), float(fy), float(cx), float(cy)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def geometric_reprojection_stats(
    anchor: tuple[FrameRecord, PoseRecord],
    current: tuple[FrameRecord, PoseRecord],
    depth_scale: float,
    config: AdaptiveAnchorConfig = AdaptiveAnchorConfig(),
    depth_config: SubmapConfig = SubmapConfig(),
    *,
    anchor_depth: np.ndarray | None = None,
    current_depth: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure metric pose-induced image motion without RGB or ground truth."""

    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth scale must be finite and positive")
    anchor_frame, anchor_pose = anchor
    current_frame, current_pose = current
    anchor_depth = _read_depth(anchor_frame) if anchor_depth is None else anchor_depth
    current_depth = _read_depth(current_frame) if current_depth is None else current_depth
    if (
        anchor_depth.ndim != 2
        or current_depth.ndim != 2
        or anchor_depth.dtype != np.uint16
        or current_depth.dtype != np.uint16
    ):
        raise ValueError("adaptive anchor depth inputs must be uint16 HxW")

    height, width = current_depth.shape
    stride = config.pixel_stride
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    z = current_depth[::stride, ::stride].astype(np.float64) / depth_scale
    valid = (
        np.isfinite(z)
        & (z >= depth_config.minimum_depth_m)
        & (z <= depth_config.maximum_depth_m)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count < config.minimum_valid_depth_samples:
        raise ValueError(
            f"frame {current_frame.frame_id} has only {valid_count} valid "
            "depth samples for adaptive anchor selection"
        )

    current_fx, current_fy, current_cx, current_cy = _intrinsics_for_loaded_depth(
        current_frame, current_depth.shape,
    )
    z_valid = z[valid]
    u_valid = uu[valid].astype(np.float64)
    v_valid = vv[valid].astype(np.float64)
    current_points = np.column_stack([
        (u_valid - current_cx) * z_valid / current_fx,
        (v_valid - current_cy) * z_valid / current_fy,
        z_valid,
    ])

    current_to_anchor = (
        np.linalg.inv(anchor_pose.t_world_camera) @ current_pose.t_world_camera
    )
    # ``einsum`` avoids spurious Accelerate/BLAS floating-point warnings seen
    # for small matrices on macOS while preserving the row-point convention.
    anchor_points = np.einsum(
        "ij,nj->ni", current_to_anchor[:3, :3], current_points,
    ) + current_to_anchor[:3, 3]
    anchor_z = anchor_points[:, 2]
    projectable = np.isfinite(anchor_points).all(axis=1) & (anchor_z > 1e-6)
    projectable_count = int(np.count_nonzero(projectable))
    anchor_fx, anchor_fy, anchor_cx, anchor_cy = _intrinsics_for_loaded_depth(
        anchor_frame, anchor_depth.shape,
    )
    anchor_height, anchor_width = anchor_depth.shape

    median_flow_px: float | None = None
    p90_flow_px: float | None = None
    in_bounds_count = 0
    if projectable_count:
        projected = anchor_points[projectable]
        projected_u = anchor_fx * projected[:, 0] / projected[:, 2] + anchor_cx
        projected_v = anchor_fy * projected[:, 1] / projected[:, 2] + anchor_cy
        source_u = u_valid[projectable]
        source_v = v_valid[projectable]
        flow = np.hypot(projected_u - source_u, projected_v - source_v)
        finite_flow = flow[np.isfinite(flow)]
        if len(finite_flow):
            median_flow_px = float(np.median(finite_flow))
            p90_flow_px = float(np.quantile(finite_flow, 0.90))
        in_bounds_count = int(np.count_nonzero(
            (projected_u >= 0.0)
            & (projected_u < anchor_width)
            & (projected_v >= 0.0)
            & (projected_v < anchor_height)
        ))

    relative_translation_m = float(np.linalg.norm(current_to_anchor[:3, 3]))
    relative_rotation_deg = _rotation_angle_deg(current_to_anchor[:3, :3])
    return {
        "valid_depth_sample_count": valid_count,
        "projectable_sample_count": projectable_count,
        "in_bounds_sample_count": in_bounds_count,
        "projectable_fraction": projectable_count / valid_count,
        "in_bounds_fraction": in_bounds_count / valid_count,
        "median_flow_px": median_flow_px,
        "p90_flow_px": p90_flow_px,
        "relative_translation_m": relative_translation_m,
        "relative_rotation_deg": relative_rotation_deg,
    }


def select_adaptive_anchor_ordinals(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    depth_scale: float,
    config: AdaptiveAnchorConfig = AdaptiveAnchorConfig(),
    depth_config: SubmapConfig = SubmapConfig(),
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select anchors and return per-frame, create-only-friendly evidence."""

    if len(bound) < 2:
        raise ValueError("adaptive anchor selection requires at least two frames")
    anchors = [0]
    evidence: list[dict[str, Any]] = [{
        "ordinal": 0,
        "frame_id": int(bound[0][0].frame_id),
        "anchor_ordinal_before_selection": None,
        "gap_from_anchor": 0,
        "selected": True,
        "triggers": ["initial"],
        "stats": None,
    }]
    anchor_ordinal = 0
    anchor_depth = _read_depth(bound[anchor_ordinal][0])
    for ordinal in range(1, len(bound)):
        current_depth = _read_depth(bound[ordinal][0])
        stats = geometric_reprojection_stats(
            bound[anchor_ordinal], bound[ordinal], depth_scale,
            config, depth_config,
            anchor_depth=anchor_depth, current_depth=current_depth,
        )
        gap = ordinal - anchor_ordinal
        triggers: list[str] = []
        if gap >= config.maximum_gap:
            triggers.append("maximum_gap")
        if gap >= config.minimum_gap:
            if (
                stats["median_flow_px"] is not None
                and stats["median_flow_px"] >= config.flow_threshold_px
            ):
                triggers.append("geometric_flow")
            if stats["in_bounds_fraction"] <= config.minimum_overlap_fraction:
                triggers.append("low_overlap")
            if stats["relative_translation_m"] >= config.translation_threshold_m:
                triggers.append("translation")
            if stats["relative_rotation_deg"] >= config.rotation_threshold_deg:
                triggers.append("rotation")
        if ordinal == len(bound) - 1:
            triggers.append("endpoint")
        selected = bool(triggers)
        evidence.append({
            "ordinal": ordinal,
            "frame_id": int(bound[ordinal][0].frame_id),
            "anchor_ordinal_before_selection": anchor_ordinal,
            "anchor_frame_id_before_selection": int(
                bound[anchor_ordinal][0].frame_id
            ),
            "gap_from_anchor": gap,
            "selected": selected,
            "triggers": triggers,
            "stats": stats,
        })
        if selected:
            anchors.append(ordinal)
            anchor_ordinal = ordinal
            anchor_depth = current_depth
    return anchors, evidence


def audit_anchor_schedule(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    anchors: Sequence[int],
    depth_scale: float,
    config: AdaptiveAnchorConfig = AdaptiveAnchorConfig(),
    depth_config: SubmapConfig = SubmapConfig(),
) -> list[dict[str, Any]]:
    """Evaluate every frame against its preceding anchor under one schedule."""

    normalized = [int(value) for value in anchors]
    if (
        len(bound) < 2
        or len(normalized) < 2
        or normalized[0] != 0
        or normalized[-1] != len(bound) - 1
        or normalized != sorted(set(normalized))
    ):
        raise ValueError("anchor audit requires sorted unique endpoints")
    rows: list[dict[str, Any]] = []
    for anchor_index, (start, stop) in enumerate(zip(normalized, normalized[1:])):
        anchor_depth = _read_depth(bound[start][0])
        for ordinal in range(start + 1, stop + 1):
            current_depth = _read_depth(bound[ordinal][0])
            rows.append({
                "anchor_index": anchor_index,
                "anchor_ordinal": start,
                "anchor_frame_id": int(bound[start][0].frame_id),
                "ordinal": ordinal,
                "frame_id": int(bound[ordinal][0].frame_id),
                "gap_from_anchor": ordinal - start,
                "is_next_anchor": ordinal == stop,
                "stats": geometric_reprojection_stats(
                    bound[start], bound[ordinal], depth_scale,
                    config, depth_config,
                    anchor_depth=anchor_depth, current_depth=current_depth,
                ),
            })
    return rows


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
