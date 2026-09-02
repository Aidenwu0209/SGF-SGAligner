#!/usr/bin/env python3
"""DPV-SLAM live pose worker for SceneGraphFusion.

SceneGraphFusion sends aligned BGR/depth frames over a Unix socket. The
official DPVO implementation estimates monocular motion and runs DPV-SLAM's
proximity loop-closure/global-BA backend (paper mechanism 1) by default.
The aligned RGB-D depth turns monocular translation into metres and rejects
geometrically inconsistent updates before they corrupt the fused map.

The wire format intentionally stays compatible with the existing learned-pose
socket adapter in libLiveRGBD.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import signal
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


REQUEST = struct.Struct("<8sIIQQddddIII")
IMU_SAMPLE = struct.Struct("<QB7xddd")
RESPONSE = struct.Struct("<8s6i21dI")
REQUEST_MAGIC = b"XFREQ01\0"
RESPONSE_MAGIC = b"XFRSP01\0"


@dataclasses.dataclass
class Options:
    min_depth_m: float = 0.20
    max_depth_m: float = 5.00
    min_scale_samples: int = 12
    min_scale_keyframes: int = 3
    max_scale_keyframes: int = 16
    max_scale_history: int = 7
    scale_warmup_measurements: int = 21
    scale_warmup_min_keyframes: int = 14
    scale_warmup_max_mad_fraction: float = 0.03
    scale_warmup_max_p90_p10_ratio: float = 1.05
    min_scale_update_motion_m: float = 0.012
    max_scale_update_rotation_deg: float = 5.0
    max_scale_update_fraction: float = 0.15
    proximity_gauge_rebase_ratio: float = 1.50
    proximity_gauge_rebase_window_frames: int = 12
    proximity_gauge_rebase_min_samples: int = 128
    depth_stride: int = 8
    depth_tolerance_m: float = 0.08
    min_depth_correspondences: int = 120
    min_depth_inlier_ratio: float = 0.42
    max_depth_inlier_rmse_m: float = 0.055
    max_linear_speed_mps: float = 0.60
    max_angular_speed_dps: float = 60.0
    base_translation_gate_m: float = 0.020
    base_rotation_gate_deg: float = 1.5
    max_translation_gate_m: float = 0.08
    max_rotation_gate_deg: float = 6.0
    max_recovery_translation_m: float = 0.10
    max_recovery_rotation_deg: float = 6.0
    trusted_motion_min_depth_correspondences: int = 1000
    trusted_motion_min_depth_inlier_ratio: float = 0.95
    trusted_motion_max_depth_rmse_m: float = 0.055
    trusted_motion_max_linear_speed_mps: float = 1.20
    trusted_motion_max_translation_m: float = 0.12
    max_imu_rotation_error_deg: float = 2.5
    max_imu_supported_rotation_deg: float = 10.0
    fast_rotation_min_depth_inlier_ratio: float = 0.85
    fast_rotation_max_depth_rmse_m: float = 0.025
    depth_history_keyframes: int = 32
    gravity_window_us: int = 750_000
    gravity_min_samples: int = 12
    gravity_min_accel_mps2: float = 7.0
    gravity_max_accel_mps2: float = 12.5
    gravity_max_direction_p90_deg: float = 5.0
    gravity_max_initial_tilt_deg: float = 65.0
    # Capture gravity while the camera is still, before visual initialization.
    # DPV-SLAM can need several seconds of parallax; using acceleration from
    # that later moving interval can reject or bias the map origin.
    gravity_bootstrap_min_samples: int = 24
    gravity_bootstrap_min_window_us: int = 400_000
    gravity_bootstrap_max_rotation_deg: float = 3.0
    gravity_bootstrap_min_magnitude_mps2: float = 9.2
    gravity_bootstrap_max_magnitude_mps2: float = 10.4
    gravity_bootstrap_max_direction_p90_deg: float = 2.5
    gravity_nominal_mps2: float = 9.80665
    gravity_source_p90_weight_mps2_per_deg: float = 0.02
    online_gravity_correction: bool = False
    online_gravity_min_residual_deg: float = 0.5
    online_gravity_max_residual_deg: float = 10.0
    online_gravity_gain: float = 0.15
    online_gravity_max_step_deg: float = 0.5
    # Disabled by default so existing live runs retain their current behavior.
    # When enabled, a rejected long-baseline pose is not compared forever with
    # the last fused RGB-D frame. Consecutive, strongly depth-supported local
    # DPVO steps build a provisional trajectory and atomically resume mapping.
    local_recovery_consecutive_frames: int = 0
    local_recovery_min_correspondences: int = 400
    local_recovery_min_inlier_ratio: float = 0.75
    local_recovery_max_depth_rmse_m: float = 0.040
    local_recovery_max_step_translation_m: float = 0.12
    local_recovery_max_step_rotation_deg: float = 12.0
    local_recovery_max_gap_translation_m: float = 3.0
    local_recovery_max_gap_rotation_deg: float = 179.0
    # Apply the rigid correction observed at the last accepted DPVO anchor
    # when proximity global BA changes it. This updates future live poses; it
    # deliberately does not pretend to non-rigidly reintegrate old surfels.
    apply_global_ba_anchor_correction: bool = False
    global_ba_max_anchor_translation_m: float = 0.75
    global_ba_max_anchor_rotation_deg: float = 45.0


@dataclasses.dataclass
class FrameRequest:
    width: int
    height: int
    frame_index: int
    timestamp_us: int
    fx: float
    fy: float
    cx: float
    cy: float
    imu: list[tuple[int, int, float, float, float]]
    color_bgr: np.ndarray
    depth_mm: np.ndarray


@dataclasses.dataclass
class Estimate:
    valid: bool = False
    initialized: bool = False
    pose_cw_m: np.ndarray = dataclasses.field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    correspondences: int = 0
    inliers: int = 0
    keyframe: bool = False
    imu_used: bool = False
    translation_m: float = 0.0
    rotation_deg: float = 0.0
    inlier_ratio: float = 0.0
    reprojection_rmse_px: float = 0.0
    depth_inlier_ratio: float = 0.0
    reason: str = ""


@dataclasses.dataclass
class DepthQuality:
    correspondences: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    inlier_rmse_m: float = math.inf


@dataclasses.dataclass
class GravityEstimate:
    valid: bool = False
    direction_camera: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    samples: int = 0
    median_magnitude_mps2: float = 0.0
    direction_p90_deg: float = math.inf
    reason: str = ""


class MetricScaleFilter:
    """Track DPVO's current gauge without freezing it at the startup gauge.

    DPV-SLAM's global BA is free to renormalize its monocular trajectory and
    inverse depths.  The RGB-D measurements therefore form a rolling gauge,
    not a permanent anchor.  Observations are allowed to accumulate while a
    pose is rejected so that the tracker can recover from a genuine gauge
    change; the candidate scale is committed only after RGB-D validation.
    """

    def __init__(
        self,
        history_size: int,
        max_update_fraction: float,
    ) -> None:
        self.observations: deque[float] = deque(maxlen=history_size)
        self.max_update_fraction = max_update_fraction
        self.value: Optional[float] = None

    def observe(self, measurement: float, informative: bool) -> tuple[float, bool]:
        """Return a rate-limited candidate without committing it."""
        if not math.isfinite(measurement) or measurement <= 0.0:
            if self.value is None:
                raise ValueError("metric scale measurement must be positive and finite")
            return self.value, False
        if self.value is None:
            return measurement, True
        if not informative:
            return self.value, False

        self.observations.append(measurement)
        target = float(np.median(np.asarray(self.observations, dtype=np.float64)))
        lower = self.value * (1.0 - self.max_update_fraction)
        upper = self.value * (1.0 + self.max_update_fraction)
        return float(np.clip(target, lower, upper)), True

    def commit(self, candidate: float) -> float:
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError("metric scale candidate must be positive and finite")
        self.value = candidate
        if not self.observations:
            self.observations.append(candidate)
        return self.value

    def rebase(self, measurement: float) -> float:
        """Atomically replace an obsolete monocular gauge after global BA."""
        if not math.isfinite(measurement) or measurement <= 0.0:
            raise ValueError("metric scale rebase must be positive and finite")
        self.observations.clear()
        self.observations.append(measurement)
        self.value = measurement
        return self.value

    def update(self, measurement: float, informative: bool) -> tuple[float, bool]:
        candidate, observed = self.observe(measurement, informative)
        if observed:
            self.commit(candidate)
        return self.value, observed


def assess_scale_stability(
    measurements: list[float], options: Options
) -> tuple[Optional[float], float, float]:
    """Return a stable DPVO gauge plus robust spread diagnostics."""
    if len(measurements) < options.scale_warmup_measurements:
        return None, math.inf, math.inf
    values = np.asarray(
        measurements[-options.scale_warmup_measurements :], dtype=np.float64
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < options.scale_warmup_measurements:
        return None, math.inf, math.inf
    median = float(np.median(values))
    mad_fraction = float(np.median(np.abs(values - median)) / median)
    p10 = float(np.percentile(values, 10.0))
    p90_p10_ratio = float(np.percentile(values, 90.0) / max(p10, 1e-9))
    if (
        mad_fraction > options.scale_warmup_max_mad_fraction
        or p90_p10_ratio > options.scale_warmup_max_p90_p10_ratio
    ):
        return None, mad_fraction, p90_p10_ratio
    return median, mad_fraction, p90_p10_ratio


def recv_exact(connection: socket.socket, count: int) -> Optional[bytes]:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_request(connection: socket.socket) -> Optional[FrameRequest]:
    raw_header = recv_exact(connection, REQUEST.size)
    if raw_header is None:
        return None
    (
        magic,
        width,
        height,
        frame_index,
        timestamp_us,
        fx,
        fy,
        cx,
        cy,
        imu_count,
        color_bytes,
        depth_bytes,
    ) = REQUEST.unpack(raw_header)
    if magic != REQUEST_MAGIC:
        raise RuntimeError(f"bad request magic: {magic!r}")
    if (
        width <= 0
        or height <= 0
        or width > 4096
        or height > 4096
        or color_bytes != width * height * 3
        or depth_bytes != width * height * 2
        or imu_count > 10000
    ):
        raise RuntimeError("invalid frame dimensions or payload lengths")

    imu = []
    for _ in range(imu_count):
        raw_sample = recv_exact(connection, IMU_SAMPLE.size)
        if raw_sample is None:
            raise EOFError("incomplete IMU payload")
        imu.append(IMU_SAMPLE.unpack(raw_sample))
    raw_color = recv_exact(connection, color_bytes)
    raw_depth = recv_exact(connection, depth_bytes)
    if raw_color is None or raw_depth is None:
        raise EOFError("incomplete image payload")
    return FrameRequest(
        width=width,
        height=height,
        frame_index=frame_index,
        timestamp_us=timestamp_us,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        imu=imu,
        color_bgr=np.frombuffer(raw_color, np.uint8).reshape(height, width, 3).copy(),
        depth_mm=np.frombuffer(raw_depth, "<u2").reshape(height, width).copy(),
    )


def send_estimate(connection: socket.socket, estimate: Estimate) -> None:
    reason = estimate.reason.encode("utf-8", errors="replace")[:65536]
    pose = np.asarray(estimate.pose_cw_m, np.float64).reshape(16).tolist()
    header = RESPONSE.pack(
        RESPONSE_MAGIC,
        int(estimate.valid),
        int(estimate.initialized),
        int(estimate.correspondences),
        int(estimate.inliers),
        int(estimate.keyframe),
        int(estimate.imu_used),
        *pose,
        float(estimate.translation_m),
        float(estimate.rotation_deg),
        float(estimate.inlier_ratio),
        float(estimate.reprojection_rmse_px),
        float(estimate.depth_inlier_ratio),
        len(reason),
    )
    connection.sendall(header)
    if reason:
        connection.sendall(reason)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate one gyro sample in camera-pose convention."""
    rotation, _ = cv2.Rodrigues((-omega * dt).astype(np.float64))
    return rotation


def angle_between_vectors_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return math.inf
    cosine = float(np.clip((first @ second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest rotation which maps ``source`` onto ``target``."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine <= 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        # This branch is not expected for gravity because the nearest signed
        # Y axis is selected, but keep the helper complete and deterministic.
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(source @ basis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        rotation, _ = cv2.Rodrigues((axis * math.pi).astype(np.float64))
        return rotation
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * (
        (1.0 - cosine) / (sine * sine)
    )


def gravity_pose_rotation_cw(
    direction_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build T_cw rotation whose nearest signed Y axis follows gravity."""
    direction = np.asarray(direction_camera, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    world_axis = np.array(
        [0.0, 1.0 if direction[1] >= 0.0 else -1.0, 0.0],
        dtype=np.float64,
    )
    tilt_deg = angle_between_vectors_deg(world_axis, direction)
    return rotation_between_vectors(world_axis, direction), world_axis, tilt_deg


def estimate_gravity_direction(
    history: list[tuple[int, np.ndarray, np.ndarray]],
    current_rotation_cw: np.ndarray,
    options: Options,
) -> GravityEstimate:
    """Estimate current-camera gravity from recent, gyro-transported accel."""
    if not history:
        return GravityEstimate(reason="no camera-axis accelerometer samples")
    latest_timestamp_us = history[-1][0]
    cutoff_us = latest_timestamp_us - options.gravity_window_us
    transported = []
    for timestamp_us, acceleration, sample_rotation_cw in history:
        if timestamp_us < cutoff_us:
            continue
        value = np.asarray(acceleration, dtype=np.float64)
        magnitude = float(np.linalg.norm(value))
        if (
            not np.all(np.isfinite(value))
            or magnitude < options.gravity_min_accel_mps2
            or magnitude > options.gravity_max_accel_mps2
        ):
            continue
        current_from_sample = current_rotation_cw @ sample_rotation_cw.T
        transported.append(current_from_sample @ value)
    if len(transported) < options.gravity_min_samples:
        return GravityEstimate(
            samples=len(transported),
            reason=(
                f"recent accelerometer samples {len(transported)} "
                f"< {options.gravity_min_samples}"
            ),
        )

    values = np.asarray(transported, dtype=np.float64)
    magnitudes = np.linalg.norm(values, axis=1)
    directions = values / magnitudes[:, None]
    seed = np.median(directions, axis=0)
    seed /= np.linalg.norm(seed)
    angles = np.degrees(
        np.arccos(np.clip(directions @ seed, -1.0, 1.0))
    )
    median_angle = float(np.median(angles))
    angle_mad = float(np.median(np.abs(angles - median_angle)))
    robust_limit_deg = max(
        2.0, min(8.0, median_angle + 3.5 * 1.4826 * angle_mad)
    )
    inliers = angles <= robust_limit_deg
    if int(np.count_nonzero(inliers)) < options.gravity_min_samples:
        return GravityEstimate(
            samples=int(np.count_nonzero(inliers)),
            median_magnitude_mps2=float(np.median(magnitudes)),
            reason="accelerometer direction has too few robust inliers",
        )
    direction = np.median(directions[inliers], axis=0)
    direction /= np.linalg.norm(direction)
    refined_angles = np.degrees(
        np.arccos(np.clip(directions[inliers] @ direction, -1.0, 1.0))
    )
    p90_deg = float(np.percentile(refined_angles, 90.0))
    valid = p90_deg <= options.gravity_max_direction_p90_deg
    return GravityEstimate(
        valid=valid,
        direction_camera=direction,
        samples=int(np.count_nonzero(inliers)),
        median_magnitude_mps2=float(np.median(magnitudes[inliers])),
        direction_p90_deg=p90_deg,
        reason=(
            ""
            if valid
            else (
                f"accelerometer direction p90 {p90_deg:.2f}deg > "
                f"{options.gravity_max_direction_p90_deg:.2f}deg"
            )
        ),
    )


def imu_supported_rotation_gate(
    visual_rotation_deg: float,
    imu_rotation_deg: float,
    gyro_samples_seen: int,
    base_gate_deg: float,
    options: Options,
) -> tuple[float, bool]:
    """Relax the angular gate only when gyro magnitude supports the visual turn."""
    consistent = (
        gyro_samples_seen > 0
        and abs(visual_rotation_deg - imu_rotation_deg)
        <= options.max_imu_rotation_error_deg
    )
    if not consistent:
        return base_gate_deg, False
    supported_gate = min(
        options.max_imu_supported_rotation_deg,
        imu_rotation_deg + options.max_imu_rotation_error_deg,
    )
    return max(base_gate_deg, supported_gate), True


def scale_update_is_informative(
    translation_m: float, rotation_deg: float, options: Options
) -> bool:
    """Only learn metric gauge from frames with useful parallax and modest turn."""
    return (
        math.isfinite(translation_m)
        and math.isfinite(rotation_deg)
        and translation_m >= options.min_scale_update_motion_m
        and rotation_deg <= options.max_scale_update_rotation_deg
    )


def proximity_gauge_rebase_is_supported(
    committed_scale: float,
    measured_scale: float,
    proximity_event_active: bool,
    scale_samples: int,
    scale_keyframes: int,
    options: Options,
) -> bool:
    """Allow a large scale jump only immediately after proximity global BA.

    The patch-scale measurement is already a robust median over several
    RGB-D-backed keyframes.  The resulting pose must still pass the normal
    dense RGB-D and continuity checks before the new gauge is committed.
    """
    if (
        not proximity_event_active
        or not math.isfinite(committed_scale)
        or not math.isfinite(measured_scale)
        or committed_scale <= 0.0
        or measured_scale <= 0.0
        or scale_samples < options.proximity_gauge_rebase_min_samples
        or scale_keyframes < options.min_scale_keyframes
    ):
        return False
    ratio = max(
        committed_scale / measured_scale,
        measured_scale / committed_scale,
    )
    return ratio >= options.proximity_gauge_rebase_ratio


def trusted_translation_gate(
    quality: DepthQuality,
    imu_consistent: bool,
    dt: float,
    normal_gate_m: float,
    options: Options,
) -> tuple[float, bool]:
    """Relax translation only when dense RGB-D and the gyro support the motion."""
    trusted = (
        imu_consistent
        and quality.correspondences
        >= options.trusted_motion_min_depth_correspondences
        and quality.inlier_ratio
        >= options.trusted_motion_min_depth_inlier_ratio
        and quality.inlier_rmse_m <= options.trusted_motion_max_depth_rmse_m
    )
    if not trusted:
        return normal_gate_m, False
    supported_gate = min(
        options.trusted_motion_max_translation_m,
        options.base_translation_gate_m
        + options.trusted_motion_max_linear_speed_mps * max(dt, 0.0),
    )
    return max(normal_gate_m, supported_gate), True


def depth_consistency(
    previous_depth_mm: np.ndarray,
    current_depth_mm: np.ndarray,
    transform_current_previous_m: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    options: Options,
) -> DepthQuality:
    """Validate a previous-camera to current-camera transform with RGB-D."""
    height, width = previous_depth_mm.shape
    offset = max(1, options.depth_stride // 2)
    vv, uu = np.mgrid[offset:height:options.depth_stride, offset:width:options.depth_stride]
    z = previous_depth_mm[vv, uu].astype(np.float64) * 0.001
    valid = (z >= options.min_depth_m) & (z <= options.max_depth_m)
    if not np.any(valid):
        return DepthQuality()

    fx, fy, cx, cy = intrinsics
    z = z[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    points = np.vstack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    rotation = transform_current_previous_m[:3, :3]
    translation = transform_current_previous_m[:3, 3:4]
    transformed = rotation @ points + translation
    predicted_z = transformed[2]
    front = predicted_z > options.min_depth_m
    if not np.any(front):
        return DepthQuality()
    transformed = transformed[:, front]
    predicted_z = predicted_z[front]
    projected_u = np.rint(fx * transformed[0] / predicted_z + cx).astype(np.int32)
    projected_v = np.rint(fy * transformed[1] / predicted_z + cy).astype(np.int32)
    inside = (
        (projected_u >= 0)
        & (projected_u < width)
        & (projected_v >= 0)
        & (projected_v < height)
        & (predicted_z <= options.max_depth_m)
    )
    if not np.any(inside):
        return DepthQuality()
    projected_u = projected_u[inside]
    projected_v = projected_v[inside]
    predicted_z = predicted_z[inside]
    observed_z = current_depth_mm[projected_v, projected_u].astype(np.float64) * 0.001
    observed_valid = (
        (observed_z >= options.min_depth_m) & (observed_z <= options.max_depth_m)
    )
    residual = np.abs(observed_z[observed_valid] - predicted_z[observed_valid])
    correspondences = int(residual.size)
    if correspondences == 0:
        return DepthQuality()
    inlier_mask = residual <= options.depth_tolerance_m
    inliers = int(np.count_nonzero(inlier_mask))
    ratio = inliers / correspondences
    rmse = (
        float(np.sqrt(np.mean(np.square(residual[inlier_mask]))))
        if inliers
        else math.inf
    )
    return DepthQuality(correspondences, inliers, ratio, rmse)


class DPVSLAMMetricTracker:
    def __init__(
        self,
        dpvo_root: Path,
        network_path: Path,
        config_path: Path,
        options: Options,
        loop_closure: bool,
        gravity_align: bool,
        finalized_trajectory_path: Optional[Path] = None,
    ) -> None:
        sys.path.insert(0, str(dpvo_root))
        from dpvo.config import cfg
        from dpvo.dpvo import DPVO

        self.DPVO = DPVO
        self.cfg = cfg.clone()
        self.cfg.merge_from_file(str(config_path))
        self.cfg.LOOP_CLOSURE = loop_closure
        self.cfg.CLASSIC_LOOP_CLOSURE = False
        self.network_path = network_path
        self.options = options
        self.gravity_align = gravity_align
        self.finalized_trajectory_path = finalized_trajectory_path
        self.finalized_trajectory_stream = None
        self.finalized_frame_ids: set[int] = set()
        if finalized_trajectory_path is not None:
            finalized_trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            self.finalized_trajectory_stream = finalized_trajectory_path.open(
                "x", encoding="utf-8"
            )
        self.slam = None
        self.input_shape: Optional[tuple[int, int]] = None
        self.depth_by_counter: dict[int, np.ndarray] = {}
        self.scale_filter = MetricScaleFilter(
            options.max_scale_history,
            options.max_scale_update_fraction,
        )
        self.scale_warmup: deque[float] = deque(
            maxlen=options.scale_warmup_measurements
        )
        self.last_accepted_counter: Optional[int] = None
        self.last_accepted_depth: Optional[np.ndarray] = None
        self.last_accepted_timestamp_us: Optional[int] = None
        self.last_output_pose = np.eye(4, dtype=np.float64)
        self.imu_rotation_cw = np.eye(3, dtype=np.float64)
        self.last_accepted_imu_rotation_cw = np.eye(3, dtype=np.float64)
        self.last_gyro_timestamp_us: Optional[int] = None
        self.gyro_samples_seen = 0
        self.accel_history: deque[tuple[int, np.ndarray, np.ndarray]] = deque(
            maxlen=4096
        )
        self.accel_samples_seen = 0
        self.bootstrap_gravity: Optional[GravityEstimate] = None
        self.bootstrap_gravity_rotation_cw: Optional[np.ndarray] = None
        self.gravity_world_axis: Optional[np.ndarray] = None
        self.gravity_alignment_applied = False
        self.accepted = 0
        self.rejected = 0
        self.last_proximity_epoch = -1000
        self.pending_gauge_rebase_epoch: Optional[int] = None
        self.pending_gauge_rebase_until_counter = -1
        self.recovery_counter: Optional[int] = None
        self.recovery_raw_pose: Optional[np.ndarray] = None
        self.recovery_depth: Optional[np.ndarray] = None
        self.recovery_timestamp_us: Optional[int] = None
        self.recovery_imu_rotation_cw: Optional[np.ndarray] = None
        self.recovery_output_pose: Optional[np.ndarray] = None
        self.recovery_streak = 0
        self.recovery_gap_translation_m = 0.0
        self.recovery_gap_rotation_deg = 0.0
        self.recovery_finalized_buffer: list[tuple[int, int, np.ndarray]] = []
        self.frame_metadata_by_counter: dict[int, tuple[int, int]] = {}
        self.pending_anchor_correction_raw: Optional[np.ndarray] = None
        self.global_reference_counter: Optional[int] = None
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info(
            "CONFIG network=%s config=%s loop_closure=%s gravity_align=%s "
            "patches=%d buffer=%d",
            network_path,
            config_path,
            loop_closure,
            gravity_align,
            self.cfg.PATCHES_PER_FRAME,
            self.cfg.BUFFER_SIZE,
        )

    def close(self) -> None:
        if self.finalized_trajectory_stream is not None:
            self.finalized_trajectory_stream.close()
            self.finalized_trajectory_stream = None

    def _emit_finalized_pose(
        self,
        frame_id: int,
        timestamp_us: int,
        pose_cw_m: np.ndarray,
        *,
        source: str,
        finalized_at_frame: int,
    ) -> None:
        if self.finalized_trajectory_stream is None or frame_id in self.finalized_frame_ids:
            return
        pose = np.asarray(pose_cw_m, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("finalized pose must be a finite 4x4 matrix")
        row = {
            "schema": "dpv_finalized_pose.v1",
            "frame_id": int(frame_id),
            "timestamp_us": int(timestamp_us),
            "T_camera_world_m": pose.reshape(-1).tolist(),
            "source": source,
            "finalized_at_frame": int(finalized_at_frame),
            "identity_fallback_used": False,
            "gt_consumed": False,
        }
        self.finalized_trajectory_stream.write(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        )
        self.finalized_trajectory_stream.flush()
        self.finalized_frame_ids.add(frame_id)

    def _emit_warmup_backfill(
        self,
        origin_counter: int,
        origin_raw_pose: np.ndarray,
        metric_scale: float,
        origin_output_pose: np.ndarray,
        finalized_at_frame: int,
    ) -> int:
        emitted = 0
        inverse_origin = np.linalg.inv(origin_raw_pose)
        for counter, (frame_id, timestamp_us) in sorted(
            self.frame_metadata_by_counter.items()
        ):
            if counter > origin_counter:
                continue
            try:
                raw_pose = self._resolve_pose_matrix(counter)
            except (KeyError, RecursionError):
                continue
            delta = raw_pose @ inverse_origin
            delta[:3, 3] *= metric_scale
            pose = delta @ origin_output_pose
            if not np.isfinite(pose).all():
                continue
            before = len(self.finalized_frame_ids)
            self._emit_finalized_pose(
                frame_id,
                timestamp_us,
                pose,
                source="DPV-SLAM:warmup_backfill",
                finalized_at_frame=finalized_at_frame,
            )
            emitted += len(self.finalized_frame_ids) - before
        logging.info(
            "WARMUP_BACKFILL finalized_at=%d emitted=%d candidates=%d",
            finalized_at_frame,
            emitted,
            len(self.frame_metadata_by_counter),
        )
        return emitted

    def _update_imu(
        self, samples: list[tuple[int, int, float, float, float]]
    ) -> None:
        for timestamp_us, kind, x, y, z in sorted(samples, key=lambda item: item[0]):
            if kind == 0:
                acceleration = np.array([x, y, z], dtype=np.float64)
                magnitude = float(np.linalg.norm(acceleration))
                if (
                    np.all(np.isfinite(acceleration))
                    and self.options.gravity_min_accel_mps2 <= magnitude
                    <= self.options.gravity_max_accel_mps2
                ):
                    self.accel_history.append(
                        (
                            timestamp_us,
                            acceleration,
                            self.imu_rotation_cw.copy(),
                        )
                    )
                    self.accel_samples_seen += 1
                continue
            if kind != 1:
                continue
            if (
                self.last_gyro_timestamp_us is not None
                and timestamp_us > self.last_gyro_timestamp_us
            ):
                dt = (timestamp_us - self.last_gyro_timestamp_us) * 1e-6
                omega = np.array([x, y, z], dtype=np.float64)
                if (
                    0.0 < dt <= 0.1
                    and np.all(np.isfinite(omega))
                    and np.linalg.norm(omega) < 20.0
                ):
                    self.imu_rotation_cw = (
                        rotation_from_omega(omega, dt) @ self.imu_rotation_cw
                    )
                    self.gyro_samples_seen += 1
            self.last_gyro_timestamp_us = timestamp_us
        self._maybe_capture_bootstrap_gravity()

    def _maybe_capture_bootstrap_gravity(self) -> None:
        if self.bootstrap_gravity is not None or not self.gravity_align:
            return
        if len(self.accel_history) < self.options.gravity_bootstrap_min_samples:
            return
        latest_timestamp_us = self.accel_history[-1][0]
        cutoff_us = latest_timestamp_us - self.options.gravity_window_us
        recent = [item for item in self.accel_history if item[0] >= cutoff_us]
        if len(recent) < self.options.gravity_bootstrap_min_samples:
            return
        if (
            recent[-1][0] - recent[0][0]
            < self.options.gravity_bootstrap_min_window_us
        ):
            return
        interval_rotation_deg = rotation_angle_deg(
            recent[-1][2] @ recent[0][2].T
        )
        if interval_rotation_deg > self.options.gravity_bootstrap_max_rotation_deg:
            return
        gravity = estimate_gravity_direction(
            recent, self.imu_rotation_cw, self.options
        )
        if (
            not gravity.valid
            or gravity.samples < self.options.gravity_bootstrap_min_samples
            or gravity.median_magnitude_mps2
            < self.options.gravity_bootstrap_min_magnitude_mps2
            or gravity.median_magnitude_mps2
            > self.options.gravity_bootstrap_max_magnitude_mps2
            or gravity.direction_p90_deg
            > self.options.gravity_bootstrap_max_direction_p90_deg
        ):
            return
        self.bootstrap_gravity = gravity
        self.bootstrap_gravity_rotation_cw = self.imu_rotation_cw.copy()
        logging.info(
            "GRAVITY_BOOTSTRAP_CAPTURED samples=%d magnitude=%.3fmps2 "
            "p90=%.2fdeg interval_rotation=%.2fdeg direction=[%.5f %.5f %.5f]",
            gravity.samples,
            gravity.median_magnitude_mps2,
            gravity.direction_p90_deg,
            interval_rotation_deg,
            *gravity.direction_camera,
        )

    def _gravity_estimate(self) -> GravityEstimate:
        return estimate_gravity_direction(
            list(self.accel_history), self.imu_rotation_cw, self.options
        )

    def _gravity_quality_score(self, gravity: GravityEstimate) -> float:
        if not gravity.valid:
            return math.inf
        return abs(
            gravity.median_magnitude_mps2 - self.options.gravity_nominal_mps2
        ) + (
            self.options.gravity_source_p90_weight_mps2_per_deg
            * gravity.direction_p90_deg
        )

    def _initialize_gravity_aligned_origin(self) -> tuple[np.ndarray, str]:
        origin_pose = np.eye(4, dtype=np.float64)
        if not self.gravity_align:
            logging.info("GRAVITY_ALIGNMENT applied=0 reason=disabled")
            return origin_pose, "gravity_alignment=disabled"

        gravity_source = "current"
        gravity = self._gravity_estimate()
        current_score = self._gravity_quality_score(gravity)
        bootstrap_score = math.inf
        if (
            self.bootstrap_gravity is not None
            and self.bootstrap_gravity_rotation_cw is not None
        ):
            direction_camera = (
                self.imu_rotation_cw
                @ self.bootstrap_gravity_rotation_cw.T
                @ self.bootstrap_gravity.direction_camera
            )
            direction_camera /= np.linalg.norm(direction_camera)
            bootstrap_gravity = dataclasses.replace(
                self.bootstrap_gravity,
                direction_camera=direction_camera,
                reason="",
                valid=True,
            )
            bootstrap_score = self._gravity_quality_score(bootstrap_gravity)
            if bootstrap_score < current_score:
                gravity = bootstrap_gravity
                gravity_source = "bootstrap"
        logging.info(
            "GRAVITY_SOURCE_SELECTED source=%s current_score=%.4f "
            "bootstrap_score=%.4f",
            gravity_source,
            current_score,
            bootstrap_score,
        )
        if not gravity.valid:
            logging.warning(
                "GRAVITY_ALIGNMENT applied=0 accel_seen=%d recent=%d reason=%s",
                self.accel_samples_seen,
                gravity.samples,
                gravity.reason,
            )
            return origin_pose, f"gravity_alignment=unavailable({gravity.reason})"

        rotation_cw, world_axis, tilt_deg = gravity_pose_rotation_cw(
            gravity.direction_camera
        )
        if tilt_deg > self.options.gravity_max_initial_tilt_deg:
            reason = (
                f"initial tilt {tilt_deg:.2f}deg > "
                f"{self.options.gravity_max_initial_tilt_deg:.2f}deg"
            )
            logging.warning(
                "GRAVITY_ALIGNMENT applied=0 accel_seen=%d recent=%d reason=%s",
                self.accel_samples_seen,
                gravity.samples,
                reason,
            )
            return origin_pose, f"gravity_alignment=unavailable({reason})"

        origin_pose[:3, :3] = rotation_cw
        self.gravity_world_axis = world_axis
        self.gravity_alignment_applied = True
        logging.info(
            "GRAVITY_ALIGNMENT applied=1 source=%s tilt=%.2fdeg "
            "direction=[%.5f %.5f %.5f] "
            "world_axis=[%.0f %.0f %.0f] samples=%d magnitude=%.3fmps2 p90=%.2fdeg",
            gravity_source,
            tilt_deg,
            *gravity.direction_camera,
            *world_axis,
            gravity.samples,
            gravity.median_magnitude_mps2,
            gravity.direction_p90_deg,
        )
        return (
            origin_pose,
            f"gravity_alignment=1 source={gravity_source} tilt={tilt_deg:.2f}deg "
            f"samples={gravity.samples} p90={gravity.direction_p90_deg:.2f}deg",
        )

    def _gravity_residual_deg(self, pose_cw_m: np.ndarray) -> Optional[float]:
        if not self.gravity_alignment_applied or self.gravity_world_axis is None:
            return None
        gravity = self._gravity_estimate()
        if not gravity.valid:
            return None
        predicted_camera = pose_cw_m[:3, :3] @ self.gravity_world_axis
        return angle_between_vectors_deg(predicted_camera, gravity.direction_camera)

    def _apply_online_gravity_constraint(
        self, pose_cw_m: np.ndarray
    ) -> tuple[np.ndarray, bool, Optional[float], Optional[float]]:
        if (
            not self.options.online_gravity_correction
            or not self.gravity_alignment_applied
            or self.gravity_world_axis is None
        ):
            return pose_cw_m, False, None, None
        gravity = self._gravity_estimate()
        if not gravity.valid:
            return pose_cw_m, False, None, None
        predicted_camera = pose_cw_m[:3, :3] @ self.gravity_world_axis
        residual_deg = angle_between_vectors_deg(
            predicted_camera, gravity.direction_camera
        )
        if (
            not math.isfinite(residual_deg)
            or residual_deg < self.options.online_gravity_min_residual_deg
            or residual_deg > self.options.online_gravity_max_residual_deg
        ):
            return pose_cw_m, False, residual_deg, residual_deg
        full_correction = rotation_between_vectors(
            predicted_camera, gravity.direction_camera
        )
        rotation_vector, _ = cv2.Rodrigues(full_correction)
        full_angle = float(np.linalg.norm(rotation_vector))
        if full_angle <= 1e-12:
            return pose_cw_m, False, residual_deg, residual_deg
        step_deg = min(
            self.options.online_gravity_max_step_deg,
            residual_deg * self.options.online_gravity_gain,
        )
        correction_vector = (
            rotation_vector.reshape(3)
            * math.radians(step_deg)
            / full_angle
        )
        correction, _ = cv2.Rodrigues(correction_vector.astype(np.float64))
        corrected = pose_cw_m.copy()
        corrected[:3, :3] = correction @ corrected[:3, :3]
        corrected[:3, 3] = correction @ corrected[:3, 3]
        corrected_prediction = corrected[:3, :3] @ self.gravity_world_axis
        corrected_residual_deg = angle_between_vectors_deg(
            corrected_prediction, gravity.direction_camera
        )
        return corrected, True, residual_deg, corrected_residual_deg

    def _clear_local_recovery(self) -> None:
        self.recovery_counter = None
        self.recovery_raw_pose = None
        self.recovery_depth = None
        self.recovery_timestamp_us = None
        self.recovery_imu_rotation_cw = None
        self.recovery_output_pose = None
        self.recovery_streak = 0
        self.recovery_gap_translation_m = 0.0
        self.recovery_gap_rotation_deg = 0.0
        self.recovery_finalized_buffer = []

    def _seed_local_recovery(
        self,
        current_counter: int,
        current_raw_pose: np.ndarray,
        request: FrameRequest,
        gap_metric_delta: np.ndarray,
        output_anchor_pose: np.ndarray,
    ) -> str:
        gap_translation_m = float(np.linalg.norm(gap_metric_delta[:3, 3]))
        gap_rotation_deg = rotation_angle_deg(gap_metric_delta[:3, :3])
        if (
            not np.all(np.isfinite(gap_metric_delta))
            or gap_translation_m
            > self.options.local_recovery_max_gap_translation_m
            or gap_rotation_deg > self.options.local_recovery_max_gap_rotation_deg
        ):
            self._clear_local_recovery()
            return (
                "local_recovery=not_seeded "
                f"gap={gap_translation_m:.3f}m/{gap_rotation_deg:.2f}deg"
            )
        candidate_output_pose = gap_metric_delta @ output_anchor_pose
        if not np.all(np.isfinite(candidate_output_pose)):
            self._clear_local_recovery()
            return "local_recovery=not_seeded non_finite_candidate"
        self.recovery_counter = current_counter
        self.recovery_raw_pose = current_raw_pose.copy()
        self.recovery_depth = request.depth_mm.copy()
        self.recovery_timestamp_us = request.timestamp_us
        self.recovery_imu_rotation_cw = self.imu_rotation_cw.copy()
        self.recovery_output_pose = candidate_output_pose
        self.recovery_streak = 0
        self.recovery_gap_translation_m = gap_translation_m
        self.recovery_gap_rotation_deg = gap_rotation_deg
        self.recovery_finalized_buffer = [
            (request.frame_index, request.timestamp_us, candidate_output_pose.copy())
        ]
        return (
            "local_recovery=seeded "
            f"gap={gap_translation_m:.3f}m/{gap_rotation_deg:.2f}deg"
        )

    def _try_local_recovery(
        self,
        current_counter: int,
        current_raw_pose: np.ndarray,
        request: FrameRequest,
        metric_scale: float,
        gap_metric_delta: np.ndarray,
        output_anchor_pose: np.ndarray,
        anchor_correction_applied: bool,
        scale_samples: int,
        n_before: int,
    ) -> tuple[Optional[Estimate], str]:
        required = self.options.local_recovery_consecutive_frames
        if required <= 0:
            return None, "local_recovery=disabled"
        if (
            self.recovery_raw_pose is None
            or self.recovery_depth is None
            or self.recovery_timestamp_us is None
            or self.recovery_imu_rotation_cw is None
            or self.recovery_output_pose is None
        ):
            return None, self._seed_local_recovery(
                current_counter,
                current_raw_pose,
                request,
                gap_metric_delta,
                output_anchor_pose,
            )

        local_raw_delta = current_raw_pose @ np.linalg.inv(self.recovery_raw_pose)
        if not np.all(np.isfinite(local_raw_delta)):
            return None, self._seed_local_recovery(
                current_counter,
                current_raw_pose,
                request,
                gap_metric_delta,
                output_anchor_pose,
            )
        local_metric_delta = local_raw_delta.copy()
        local_metric_delta[:3, 3] *= metric_scale
        intrinsics = (request.fx, request.fy, request.cx, request.cy)
        quality = depth_consistency(
            self.recovery_depth,
            request.depth_mm,
            local_metric_delta,
            intrinsics,
            self.options,
        )
        translation_m = float(np.linalg.norm(local_metric_delta[:3, 3]))
        rotation_deg = rotation_angle_deg(local_metric_delta[:3, :3])
        dt = max(
            1e-3,
            (request.timestamp_us - self.recovery_timestamp_us) * 1e-6,
        )
        translation_gate = min(
            self.options.local_recovery_max_step_translation_m,
            self.options.base_translation_gate_m
            + self.options.trusted_motion_max_linear_speed_mps * dt,
        )
        normal_rotation_gate = min(
            self.options.local_recovery_max_step_rotation_deg,
            self.options.base_rotation_gate_deg
            + self.options.max_angular_speed_dps * dt,
        )
        imu_relative = self.imu_rotation_cw @ self.recovery_imu_rotation_cw.T
        imu_rotation_deg = rotation_angle_deg(imu_relative)
        rotation_gate, imu_consistent = imu_supported_rotation_gate(
            rotation_deg,
            imu_rotation_deg,
            self.gyro_samples_seen,
            normal_rotation_gate,
            self.options,
        )
        rotation_gate = min(
            rotation_gate, self.options.local_recovery_max_step_rotation_deg
        )
        quality_ok = (
            quality.correspondences
            >= self.options.local_recovery_min_correspondences
            and quality.inlier_ratio
            >= self.options.local_recovery_min_inlier_ratio
            and quality.inlier_rmse_m
            <= self.options.local_recovery_max_depth_rmse_m
        )
        continuity_ok = (
            translation_m <= translation_gate and rotation_deg <= rotation_gate
        )
        if not quality_ok or not continuity_ok:
            reset_note = self._seed_local_recovery(
                current_counter,
                current_raw_pose,
                request,
                gap_metric_delta,
                output_anchor_pose,
            )
            return (
                None,
                f"local_recovery=reset corr={quality.correspondences} "
                f"inlier={quality.inlier_ratio:.3f} "
                f"rmse={quality.inlier_rmse_m:.3f}m "
                f"step={translation_m:.3f}m/{rotation_deg:.2f}deg "
                f"gate={translation_gate:.3f}m/{rotation_gate:.2f}deg "
                f"imu={imu_rotation_deg:.2f}deg; {reset_note}",
            )

        candidate_output_pose = local_metric_delta @ self.recovery_output_pose
        if not np.all(np.isfinite(candidate_output_pose)):
            return None, self._seed_local_recovery(
                current_counter,
                current_raw_pose,
                request,
                gap_metric_delta,
                output_anchor_pose,
            )
        (
            candidate_output_pose,
            gravity_corrected,
            gravity_before_deg,
            gravity_after_deg,
        ) = self._apply_online_gravity_constraint(candidate_output_pose)
        self.recovery_counter = current_counter
        self.recovery_raw_pose = current_raw_pose.copy()
        self.recovery_depth = request.depth_mm.copy()
        self.recovery_timestamp_us = request.timestamp_us
        self.recovery_imu_rotation_cw = self.imu_rotation_cw.copy()
        self.recovery_output_pose = candidate_output_pose
        self.recovery_streak += 1
        self.recovery_finalized_buffer.append(
            (request.frame_index, request.timestamp_us, candidate_output_pose.copy())
        )
        if self.recovery_streak < required:
            return (
                None,
                f"local_recovery=verifying {self.recovery_streak}/{required} "
                f"corr={quality.correspondences} "
                f"inlier={quality.inlier_ratio:.3f} "
                f"rmse={quality.inlier_rmse_m:.3f}m",
            )

        gap_translation_m = self.recovery_gap_translation_m
        gap_rotation_deg = self.recovery_gap_rotation_deg
        self.last_output_pose = candidate_output_pose
        self.last_accepted_counter = current_counter
        self.last_accepted_depth = request.depth_mm.copy()
        self.last_accepted_timestamp_us = request.timestamp_us
        self.last_accepted_imu_rotation_cw = self.imu_rotation_cw.copy()
        self.accepted += 1
        if anchor_correction_applied:
            self.pending_anchor_correction_raw = None
        estimate = Estimate(initialized=True)
        estimate.valid = True
        estimate.pose_cw_m = candidate_output_pose.copy()
        estimate.correspondences = quality.correspondences
        estimate.inliers = quality.inliers
        estimate.inlier_ratio = quality.inlier_ratio
        estimate.depth_inlier_ratio = quality.inlier_ratio
        estimate.translation_m = translation_m
        estimate.rotation_deg = rotation_deg
        estimate.keyframe = int(self.slam.n) > n_before
        estimate.imu_used = imu_consistent or gravity_corrected
        estimate.reason = (
            f"local recovery committed after {required} verified steps; "
            f"gap={gap_translation_m:.3f}m/{gap_rotation_deg:.2f}deg "
            f"depth_inlier={quality.inlier_ratio:.3f} "
            f"depth_rmse={quality.inlier_rmse_m:.3f}m "
            f"gravity_corrected={int(gravity_corrected)} "
            f"gravity_residual={gravity_before_deg}/"
            f"{gravity_after_deg}deg"
        )
        logging.info(
            "LOCAL_RECOVERY_COMMITTED frame=%d verified=%d gap=%.3fm/%.2fdeg "
            "step=%.3fm/%.2fdeg depth_inlier=%.3f depth_rmse=%.4fm",
            request.frame_index,
            required,
            gap_translation_m,
            gap_rotation_deg,
            translation_m,
            rotation_deg,
            quality.inlier_ratio,
            quality.inlier_rmse_m,
        )
        for frame_id, timestamp_us, pose in self.recovery_finalized_buffer:
            self._emit_finalized_pose(
                frame_id,
                timestamp_us,
                pose,
                source="DPV-SLAM:local_recovery_backfill",
                finalized_at_frame=request.frame_index,
            )
        self._clear_local_recovery()
        return estimate, "local_recovery=committed"

    def _corrected_output_anchor(
        self, metric_scale: float, *, preserve_rejected: bool = False
    ) -> tuple[np.ndarray, str, bool, bool]:
        if (
            not self.options.apply_global_ba_anchor_correction
            or self.pending_anchor_correction_raw is None
        ):
            return self.last_output_pose, "anchor_correction=none", False, False
        correction = self.pending_anchor_correction_raw.copy()
        correction[:3, 3] *= metric_scale
        translation_m = float(np.linalg.norm(correction[:3, 3]))
        rotation_deg = rotation_angle_deg(correction[:3, :3])
        if (
            not np.all(np.isfinite(correction))
            or translation_m
            > self.options.global_ba_max_anchor_translation_m
            or rotation_deg > self.options.global_ba_max_anchor_rotation_deg
        ):
            logging.warning(
                "GLOBAL_BA_ANCHOR_REJECTED correction=%.3fm/%.2fdeg",
                translation_m,
                rotation_deg,
            )
            if not preserve_rejected:
                self.pending_anchor_correction_raw = None
            return (
                self.last_output_pose,
                f"anchor_correction=rejected({translation_m:.3f}m/"
                f"{rotation_deg:.2f}deg)",
                False,
                True,
            )
        corrected = correction @ self.last_output_pose
        return (
            corrected,
            f"anchor_correction=pending({translation_m:.3f}m/"
            f"{rotation_deg:.2f}deg)",
            True,
            False,
        )

    @property
    def initialized(self) -> bool:
        return bool(self.slam is not None and self.slam.is_initialized)

    def _create_slam(self, height: int, width: int) -> None:
        if height % 16 or width % 16:
            raise ValueError("DPV-SLAM input dimensions must be divisible by 16")
        self.slam = self.DPVO(
            self.cfg,
            str(self.network_path),
            ht=height,
            wd=width,
            viz=False,
        )
        self.input_shape = (height, width)
        logging.info(
            "MODEL_READY gpu=%s allocated=%.2fGiB",
            torch.cuda.get_device_name(),
            torch.cuda.memory_allocated() / (1024**3),
        )

    def _active_stamps(self) -> list[int]:
        if self.slam is None or self.slam.n <= 0:
            return []
        return [int(x) for x in self.slam.pg.tstamps_[: self.slam.n]]

    def _prune_depth_history(self, current_counter: int) -> None:
        stamps = self._active_stamps()[-self.options.depth_history_keyframes :]
        keep = set(stamps)
        keep.add(current_counter)
        if self.last_accepted_counter is not None:
            keep.add(self.last_accepted_counter)
        for counter in list(self.depth_by_counter):
            if counter not in keep:
                del self.depth_by_counter[counter]

    def _resolve_pose_matrix(self, counter: int) -> np.ndarray:
        assert self.slam is not None
        self.slam.traj = {
            int(self.slam.pg.tstamps_[index]): self.slam.pg.poses_[index]
            for index in range(self.slam.n)
        }
        pose = self.slam.get_pose(counter)
        matrix = pose.matrix().detach().float().cpu().numpy()
        if matrix.ndim == 3:
            matrix = matrix[0]
        return np.asarray(matrix, dtype=np.float64)

    def _patch_scale_for_index(self, active_index: int) -> tuple[Optional[float], int]:
        if self.slam is None or self.slam.n <= 0:
            return None, 0
        counter = int(self.slam.pg.tstamps_[active_index])
        depth = self.depth_by_counter.get(counter)
        if depth is None:
            return None, 0
        patch = (
            self.slam.pg.patches_[active_index, :, :, 1, 1]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        u = np.rint(patch[:, 0] * self.slam.RES).astype(np.int32)
        v = np.rint(patch[:, 1] * self.slam.RES).astype(np.int32)
        disparity = patch[:, 2].astype(np.float64)
        inside = (
            (u >= 0)
            & (u < depth.shape[1])
            & (v >= 0)
            & (v < depth.shape[0])
            & np.isfinite(disparity)
            & (disparity > 1e-5)
        )
        if not np.any(inside):
            return None, 0
        z_m = depth[v[inside], u[inside]].astype(np.float64) * 0.001
        samples = z_m * disparity[inside]
        valid = (
            (z_m >= self.options.min_depth_m)
            & (z_m <= self.options.max_depth_m)
            & np.isfinite(samples)
            & (samples > 0.01)
            & (samples < 100.0)
        )
        samples = samples[valid]
        if samples.size < self.options.min_scale_samples:
            return None, int(samples.size)
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        if mad > 1e-9:
            samples = samples[np.abs(samples - median) <= 3.5 * 1.4826 * mad]
        if samples.size < self.options.min_scale_samples:
            return None, int(samples.size)
        return float(np.median(samples)), int(samples.size)

    def _patch_metric_scale(self) -> tuple[Optional[float], int, int]:
        """Estimate one DPVO gauge from several RGB-D-backed keyframes."""
        if self.slam is None or self.slam.n <= 0:
            return None, 0, 0
        begin = max(0, self.slam.n - self.options.max_scale_keyframes)
        frame_scales: list[float] = []
        total_samples = 0
        for active_index in range(begin, self.slam.n):
            frame_scale, samples = self._patch_scale_for_index(active_index)
            if frame_scale is None:
                continue
            frame_scales.append(frame_scale)
            total_samples += samples
        if len(frame_scales) < self.options.min_scale_keyframes:
            return None, total_samples, len(frame_scales)

        values = np.asarray(frame_scales, dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-9:
            values = values[np.abs(values - median) <= 3.5 * 1.4826 * mad]
        if values.size < self.options.min_scale_keyframes:
            return None, total_samples, int(values.size)
        return float(np.median(values)), total_samples, int(values.size)

    def track(self, request: FrameRequest) -> Estimate:
        started = time.perf_counter()
        self._update_imu(request.imu)
        if self.slam is None:
            self._create_slam(request.height, request.width)
        if self.input_shape != (request.height, request.width):
            raise ValueError("camera dimensions changed after DPV-SLAM initialization")
        assert self.slam is not None

        counter = int(self.slam.counter)
        self.frame_metadata_by_counter[counter] = (
            request.frame_index,
            request.timestamp_us,
        )
        self.depth_by_counter[counter] = request.depth_mm.copy()
        image = torch.from_numpy(request.color_bgr).permute(2, 0, 1).cuda()
        intrinsics_tensor = torch.tensor(
            [request.fx, request.fy, request.cx, request.cy],
            dtype=torch.float32,
            device="cuda",
        )
        n_before = int(self.slam.n)
        anchor_raw_before_update: Optional[np.ndarray] = None
        reference_raw_before_update: Optional[np.ndarray] = None
        if (
            self.options.apply_global_ba_anchor_correction
            and self.last_accepted_counter is not None
        ):
            try:
                anchor_raw_before_update = self._resolve_pose_matrix(
                    self.last_accepted_counter
                )
                if self.global_reference_counter is not None:
                    reference_raw_before_update = self._resolve_pose_matrix(
                        self.global_reference_counter
                    )
            except (KeyError, RecursionError):
                anchor_raw_before_update = None
                reference_raw_before_update = None
        with torch.inference_mode():
            self.slam(request.timestamp_us * 1e-6, image, intrinsics_tensor)
        del image, intrinsics_tensor
        proximity_epoch = int(getattr(self.slam, "last_global_ba", -1000))
        new_proximity_event = False
        if self.cfg.LOOP_CLOSURE and proximity_epoch > self.last_proximity_epoch:
            new_proximity_event = True
            self.last_proximity_epoch = proximity_epoch
            self.pending_gauge_rebase_epoch = proximity_epoch
            self.pending_gauge_rebase_until_counter = (
                int(self.slam.counter)
                + self.options.proximity_gauge_rebase_window_frames
            )
            logging.info(
                "PROXIMITY_LOOP factors_added_at_keyframe=%d active_keyframes=%d "
                "gauge_rebase_window_until=%d",
                proximity_epoch,
                self.slam.n,
                self.pending_gauge_rebase_until_counter,
            )
        if (
            new_proximity_event
            and anchor_raw_before_update is not None
            and reference_raw_before_update is not None
        ):
            try:
                anchor_raw_after_update = self._resolve_pose_matrix(
                    int(self.last_accepted_counter)
                )
                reference_raw_after_update = self._resolve_pose_matrix(
                    int(self.global_reference_counter)
                )
                relative_before = anchor_raw_before_update @ np.linalg.inv(
                    reference_raw_before_update
                )
                relative_after = anchor_raw_after_update @ np.linalg.inv(
                    reference_raw_after_update
                )
                correction = relative_after @ np.linalg.inv(relative_before)
                if np.all(np.isfinite(correction)):
                    if self.pending_anchor_correction_raw is None:
                        self.pending_anchor_correction_raw = correction
                    else:
                        self.pending_anchor_correction_raw = (
                            correction @ self.pending_anchor_correction_raw
                        )
                    logging.info(
                        "GLOBAL_BA_RELATIVE_PENDING epoch=%d raw_translation=%.6f "
                        "rotation=%.3fdeg",
                        proximity_epoch,
                        float(np.linalg.norm(correction[:3, 3])),
                        rotation_angle_deg(correction[:3, :3]),
                    )
            except (KeyError, RecursionError, np.linalg.LinAlgError):
                logging.warning(
                    "GLOBAL_BA_RELATIVE_UNAVAILABLE epoch=%d", proximity_epoch
                )
        self._prune_depth_history(counter)

        estimate = Estimate(initialized=self.initialized)
        if not self.initialized:
            self.rejected += 1
            estimate.reason = (
                f"DPV-SLAM initializing: keyframes={self.slam.n}/8; "
                "move the camera slowly with parallax"
            )
            return estimate

        current_counter = int(self.slam.counter - 1)
        if (
            self.pending_gauge_rebase_epoch is not None
            and current_counter > self.pending_gauge_rebase_until_counter
        ):
            logging.warning(
                "GAUGE_REBASE_WINDOW_EXPIRED epoch=%d current_counter=%d",
                self.pending_gauge_rebase_epoch,
                current_counter,
            )
            self.pending_gauge_rebase_epoch = None
            self.pending_gauge_rebase_until_counter = -1
        patch_scale, scale_samples, scale_keyframes = self._patch_metric_scale()
        if patch_scale is None:
            self.rejected += 1
            estimate.reason = (
                "metric scale unavailable: "
                f"RGB-D-backed keyframes={scale_keyframes} "
                f"< {self.options.min_scale_keyframes}, samples={scale_samples}"
            )
            return estimate

        current_raw_pose = self._resolve_pose_matrix(current_counter)
        if self.last_accepted_counter is None:
            self.scale_warmup.append(patch_scale)
            stable_scale, warmup_mad, warmup_ratio = assess_scale_stability(
                list(self.scale_warmup), self.options
            )
            if (
                stable_scale is None
                or int(self.slam.n) < self.options.scale_warmup_min_keyframes
            ):
                self.rejected += 1
                estimate.reason = (
                    "metric scale warming up: "
                    f"measurements={len(self.scale_warmup)}/"
                    f"{self.options.scale_warmup_measurements}, "
                    f"DPVO keyframes={self.slam.n}/"
                    f"{self.options.scale_warmup_min_keyframes}, "
                    f"MAD={warmup_mad:.3f}, p90/p10={warmup_ratio:.3f}; "
                    "map remains paused"
                )
                return estimate
            metric_scale, _ = self.scale_filter.update(
                stable_scale, informative=True
            )
            self.last_accepted_counter = current_counter
            self.global_reference_counter = current_counter
            self.last_accepted_depth = request.depth_mm.copy()
            self.last_accepted_timestamp_us = request.timestamp_us
            self.last_output_pose, gravity_note = (
                self._initialize_gravity_aligned_origin()
            )
            self.last_accepted_imu_rotation_cw = self.imu_rotation_cw.copy()
            self.accepted += 1
            estimate.valid = True
            estimate.pose_cw_m = self.last_output_pose.copy()
            estimate.keyframe = int(self.slam.n) > n_before
            estimate.correspondences = scale_samples
            estimate.inliers = scale_samples
            estimate.inlier_ratio = 1.0
            estimate.depth_inlier_ratio = 1.0
            estimate.imu_used = self.gravity_alignment_applied
            estimate.reason = (
                f"metric origin initialized after stable warmup; "
                f"scale={metric_scale:.4f} m/unit, MAD={warmup_mad:.3f}, "
                f"p90/p10={warmup_ratio:.3f}, DPVO keyframes={self.slam.n}; "
                f"{gravity_note}"
            )
            self._emit_warmup_backfill(
                current_counter,
                current_raw_pose,
                metric_scale,
                self.last_output_pose,
                request.frame_index,
            )
            return estimate

        try:
            previous_raw_pose = self._resolve_pose_matrix(self.last_accepted_counter)
        except (KeyError, RecursionError):
            self.rejected += 1
            estimate.reason = "DPV-SLAM could not resolve the last accepted pose"
            return estimate
        raw_delta = current_raw_pose @ np.linalg.inv(previous_raw_pose)
        if not np.all(np.isfinite(raw_delta)):
            self.rejected += 1
            estimate.reason = "DPV-SLAM returned a non-finite relative pose"
            return estimate

        if self.scale_filter.value is None:
            raise RuntimeError("metric scale filter was not initialized")
        committed_metric_scale = self.scale_filter.value
        rotation_deg = rotation_angle_deg(raw_delta[:3, :3])
        scale_probe_translation_m = float(
            np.linalg.norm(raw_delta[:3, 3]) * committed_metric_scale
        )
        scale_informative = scale_update_is_informative(
            scale_probe_translation_m, rotation_deg, self.options
        )
        proximity_rebase_active = (
            self.pending_gauge_rebase_epoch is not None
            and current_counter <= self.pending_gauge_rebase_until_counter
        )
        gauge_rebase_candidate = proximity_gauge_rebase_is_supported(
            committed_metric_scale,
            patch_scale,
            proximity_rebase_active,
            scale_samples,
            scale_keyframes,
            self.options,
        )
        if gauge_rebase_candidate:
            # A proximity global BA may rescale every pose and inverse depth
            # at once.  Using the stale gauge makes a small real motion appear
            # discontinuous.  Test the robust RGB-D scale directly, then
            # atomically replace the gauge only if all pose checks pass.
            metric_scale = patch_scale
            scale_observed = True
        else:
            # Ordinary gauge drift remains rate limited.  Observations can
            # accumulate during rejected frames, but are committed only after
            # the metric pose passes the normal RGB-D and continuity checks.
            metric_scale, scale_observed = self.scale_filter.observe(
                patch_scale, informative=scale_informative
            )
        metric_delta = raw_delta.copy()
        metric_delta[:3, 3] *= metric_scale
        intrinsics = (request.fx, request.fy, request.cx, request.cy)
        assert self.last_accepted_depth is not None
        quality = depth_consistency(
            self.last_accepted_depth,
            request.depth_mm,
            metric_delta,
            intrinsics,
            self.options,
        )
        translation_m = float(np.linalg.norm(metric_delta[:3, 3]))
        dt = max(
            1e-3,
            (request.timestamp_us - int(self.last_accepted_timestamp_us or request.timestamp_us))
            * 1e-6,
        )
        translation_gate = min(
            self.options.max_translation_gate_m,
            self.options.base_translation_gate_m + self.options.max_linear_speed_mps * dt,
        )
        rotation_gate = min(
            self.options.max_rotation_gate_deg,
            self.options.base_rotation_gate_deg + self.options.max_angular_speed_dps * dt,
        )
        if dt > 0.5:
            translation_gate = self.options.max_recovery_translation_m
            rotation_gate = self.options.max_recovery_rotation_deg

        imu_relative = (
            self.imu_rotation_cw @ self.last_accepted_imu_rotation_cw.T
        )
        imu_rotation_deg = rotation_angle_deg(imu_relative)
        normal_rotation_gate = rotation_gate
        imu_quality_ok = (
            quality.inlier_ratio >= self.options.fast_rotation_min_depth_inlier_ratio
            and quality.inlier_rmse_m
            <= self.options.fast_rotation_max_depth_rmse_m
        )
        rotation_gate, imu_consistent = imu_supported_rotation_gate(
            rotation_deg,
            imu_rotation_deg,
            self.gyro_samples_seen,
            rotation_gate,
            self.options,
        )
        if rotation_deg > normal_rotation_gate and not imu_quality_ok:
            rotation_gate = normal_rotation_gate
            imu_consistent = False
        translation_gate, trusted_motion = trusted_translation_gate(
            quality,
            imu_consistent,
            dt,
            translation_gate,
            self.options,
        )
        (
            output_anchor_pose,
            anchor_note,
            anchor_correction_applied,
            anchor_correction_rejected,
        ) = self._corrected_output_anchor(
            metric_scale,
            preserve_rejected=gauge_rebase_candidate,
        )

        if gauge_rebase_candidate and anchor_correction_rejected:
            self.rejected += 1
            estimate.correspondences = quality.correspondences
            estimate.inliers = quality.inliers
            estimate.inlier_ratio = quality.inlier_ratio
            estimate.depth_inlier_ratio = quality.inlier_ratio
            estimate.translation_m = translation_m
            estimate.rotation_deg = rotation_deg
            estimate.reason = (
                "atomic gauge/anchor transaction rolled back: anchor correction "
                "failed safety bounds; "
                f"scale_committed={committed_metric_scale:.4f} "
                f"scale_candidate={metric_scale:.4f} "
                f"proximity_epoch={self.pending_gauge_rebase_epoch}; "
                f"{anchor_note}"
            )
            logging.warning(
                "GAUGE_ANCHOR_TRANSACTION_ROLLED_BACK frame=%d epoch=%s "
                "old_scale=%.6f candidate_scale=%.6f reason=%s",
                request.frame_index,
                self.pending_gauge_rebase_epoch,
                committed_metric_scale,
                metric_scale,
                anchor_note,
            )
            return estimate

        reasons = []
        if quality.correspondences < self.options.min_depth_correspondences:
            reasons.append(
                f"depth correspondences {quality.correspondences} "
                f"< {self.options.min_depth_correspondences}"
            )
        if quality.inlier_ratio < self.options.min_depth_inlier_ratio:
            reasons.append(
                f"depth inlier ratio {quality.inlier_ratio:.3f} "
                f"< {self.options.min_depth_inlier_ratio:.3f}"
            )
        if quality.inlier_rmse_m > self.options.max_depth_inlier_rmse_m:
            reasons.append(
                f"depth RMSE {quality.inlier_rmse_m:.3f}m "
                f"> {self.options.max_depth_inlier_rmse_m:.3f}m"
            )
        if translation_m > translation_gate or rotation_deg > rotation_gate:
            reasons.append(
                f"continuity {translation_m:.3f}m/{rotation_deg:.2f}deg "
                f"> {translation_gate:.3f}m/{rotation_gate:.2f}deg"
            )
        if reasons:
            recovery_estimate, recovery_note = self._try_local_recovery(
                current_counter,
                current_raw_pose,
                request,
                metric_scale,
                metric_delta,
                output_anchor_pose,
                anchor_correction_applied,
                scale_samples,
                n_before,
            )
            if recovery_estimate is not None:
                return recovery_estimate
            self.rejected += 1
            estimate.correspondences = quality.correspondences
            estimate.inliers = quality.inliers
            estimate.inlier_ratio = quality.inlier_ratio
            estimate.depth_inlier_ratio = quality.inlier_ratio
            estimate.translation_m = translation_m
            estimate.rotation_deg = rotation_deg
            estimate.reason = "; ".join(reasons) + (
                f"; scale_committed={committed_metric_scale:.4f} "
                f"scale_candidate={metric_scale:.4f} "
                f"measured={patch_scale:.4f} updated=0 "
                f"observed={int(scale_observed)} "
                f"informative={int(scale_informative)} "
                f"gauge_rebase_candidate={int(gauge_rebase_candidate)} "
                f"proximity_epoch={self.pending_gauge_rebase_epoch} "
                f"keyframes={scale_keyframes} "
                f"samples={scale_samples} imu_rot={imu_rotation_deg:.2f}deg "
                f"imu_consistent={int(imu_consistent)} "
                f"trusted_motion={int(trusted_motion)} "
                f"depth_rmse={quality.inlier_rmse_m:.3f}m; "
                f"{recovery_note}; {anchor_note}"
            )
            return estimate

        self._clear_local_recovery()
        output_pose = metric_delta @ output_anchor_pose
        if not np.all(np.isfinite(output_pose)):
            self.rejected += 1
            estimate.reason = "metric pose composition produced non-finite values"
            return estimate
        (
            output_pose,
            gravity_corrected,
            gravity_before_deg,
            gravity_after_deg,
        ) = self._apply_online_gravity_constraint(output_pose)

        if anchor_correction_applied:
            correction = self.pending_anchor_correction_raw.copy()
            correction[:3, 3] *= metric_scale
            logging.info(
                "GLOBAL_BA_ANCHOR_COMMITTED frame=%d correction=%.3fm/%.2fdeg",
                request.frame_index,
                float(np.linalg.norm(correction[:3, 3])),
                rotation_angle_deg(correction[:3, :3]),
            )
            self.pending_anchor_correction_raw = None

        self.last_output_pose = output_pose
        self.last_accepted_counter = current_counter
        self.last_accepted_depth = request.depth_mm.copy()
        self.last_accepted_timestamp_us = request.timestamp_us
        self.last_accepted_imu_rotation_cw = self.imu_rotation_cw.copy()
        gravity_residual_deg = self._gravity_residual_deg(output_pose)
        if gauge_rebase_candidate:
            next_metric_scale = self.scale_filter.rebase(metric_scale)
            logging.info(
                "GAUGE_REBASE_COMMITTED epoch=%d old_scale=%.6f new_scale=%.6f "
                "ratio=%.3f frame=%d depth_inlier=%.3f depth_rmse=%.4fm "
                "motion=%.4fm/%.2fdeg",
                int(self.pending_gauge_rebase_epoch or -1),
                committed_metric_scale,
                next_metric_scale,
                max(
                    committed_metric_scale / next_metric_scale,
                    next_metric_scale / committed_metric_scale,
                ),
                request.frame_index,
                quality.inlier_ratio,
                quality.inlier_rmse_m,
                translation_m,
                rotation_deg,
            )
        elif scale_observed:
            next_metric_scale = self.scale_filter.commit(metric_scale)
        else:
            next_metric_scale = committed_metric_scale
        scale_updated = bool(
            scale_observed
            and not math.isclose(
                next_metric_scale, committed_metric_scale, rel_tol=1e-6, abs_tol=1e-9
            )
        )
        completed_proximity_epoch = self.pending_gauge_rebase_epoch
        if completed_proximity_epoch is not None:
            self.pending_gauge_rebase_epoch = None
            self.pending_gauge_rebase_until_counter = -1
        self.accepted += 1
        estimate.valid = True
        estimate.pose_cw_m = output_pose.copy()
        estimate.correspondences = quality.correspondences
        estimate.inliers = quality.inliers
        estimate.inlier_ratio = quality.inlier_ratio
        estimate.depth_inlier_ratio = quality.inlier_ratio
        estimate.translation_m = translation_m
        estimate.rotation_deg = rotation_deg
        estimate.keyframe = int(self.slam.n) > n_before
        estimate.imu_used = imu_consistent or gravity_corrected
        estimate.reason = (
            f"scale={metric_scale:.4f} committed={next_metric_scale:.4f} "
            f"measured={patch_scale:.4f} m/unit "
            f"updated={int(scale_updated)} "
            f"gauge_rebase={int(gauge_rebase_candidate)} "
            f"proximity_epoch={completed_proximity_epoch} "
            f"keyframes={scale_keyframes} "
            f"samples={scale_samples} "
            f"depth_rmse={quality.inlier_rmse_m:.3f}m "
            f"dpv_kf={self.slam.n} imu={int(imu_consistent)} "
            f"imu_rot={imu_rotation_deg:.2f}deg "
            f"gravity_residual="
            f"{gravity_residual_deg if gravity_residual_deg is not None else math.nan:.2f}deg "
            f"gravity_corrected={int(gravity_corrected)} "
            f"gravity_step={gravity_before_deg}/{gravity_after_deg}deg "
            f"trusted_motion={int(trusted_motion)} "
            f"{anchor_note} "
            f"latency={(time.perf_counter()-started)*1000:.1f}ms"
        )
        self._emit_finalized_pose(
            request.frame_index,
            request.timestamp_us,
            output_pose,
            source="DPV-SLAM:online",
            finalized_at_frame=request.frame_index,
        )
        return estimate


def serve(socket_path: Path, tracker: DPVSLAMMetricTracker) -> None:
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(1)
    listener.settimeout(1.0)
    logging.info("READY socket=%s device=cuda", socket_path)
    running = True

    def stop_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        while running:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            logging.info("SceneGraphFusion client connected")
            with connection:
                connection.settimeout(10.0)
                while running:
                    try:
                        request = read_request(connection)
                        if request is None:
                            break
                        started = time.perf_counter()
                        try:
                            estimate = tracker.track(request)
                        except torch.cuda.OutOfMemoryError as error:
                            torch.cuda.empty_cache()
                            logging.exception("CUDA OOM on frame %d", request.frame_index)
                            estimate = Estimate(
                                initialized=tracker.initialized,
                                reason=f"DPV-SLAM CUDA out of memory: {error}",
                            )
                        except Exception as error:
                            logging.exception("pose worker failed on frame %d", request.frame_index)
                            estimate = Estimate(
                                initialized=tracker.initialized,
                                reason=f"pose worker exception: {type(error).__name__}: {error}",
                            )
                        send_estimate(connection, estimate)
                        logging.info(
                            "frame=%d valid=%d initialized=%d corr=%d inliers=%d "
                            "depth=%.3f motion=%.3fm/%.2fdeg kf=%d time=%.1fms reason=%s",
                            request.frame_index,
                            estimate.valid,
                            estimate.initialized,
                            estimate.correspondences,
                            estimate.inliers,
                            estimate.depth_inlier_ratio,
                            estimate.translation_m,
                            estimate.rotation_deg,
                            estimate.keyframe,
                            (time.perf_counter() - started) * 1000.0,
                            estimate.reason,
                        )
                    except socket.timeout:
                        logging.info("SceneGraphFusion client idle; keeping connection alive")
                        continue
                    except (EOFError, ConnectionError) as error:
                        logging.warning("client disconnected: %s", error)
                        break
                    except Exception:
                        logging.exception("protocol error")
                        break
            logging.info("SceneGraphFusion client disconnected")
    finally:
        listener.close()
        tracker.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def load_metric_options(path: Optional[Path]) -> Options:
    if path is None:
        return Options()
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError("metric config must contain one JSON object")
    known = {field.name for field in dataclasses.fields(Options)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown metric config options: {', '.join(unknown)}")
    options = Options(**values)
    if (
        options.scale_warmup_measurements <= 0
        or options.scale_warmup_min_keyframes < 8
        or options.local_recovery_consecutive_frames < 0
        or not 0.0 <= options.local_recovery_min_inlier_ratio <= 1.0
        or options.local_recovery_min_correspondences < 0
    ):
        raise ValueError("metric config contains an out-of-range option")
    return options


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=Path("/tmp/sgf_dpvslam_pose.sock"))
    parser.add_argument("--dpvo-root", type=Path, default=project_root / "third_party" / "DPVO")
    parser.add_argument("--network", type=Path, default=project_root / "third_party" / "DPVO" / "dpvo.pth")
    parser.add_argument("--config", type=Path, default=project_root / "config" / "dpvslam_live.yaml")
    parser.add_argument(
        "--metric-config",
        type=Path,
        help="optional JSON overrides for RGB-D metric validation/recovery",
    )
    parser.add_argument("--loop-closure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gravity-align", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--finalized-trajectory",
        type=Path,
        help="create-only JSONL sidecar for online and retroactively finalized poses",
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[dpvslam] %(asctime)s %(levelname)s %(message)s",
    )
    required_paths = [args.dpvo_root, args.network, args.config]
    if args.metric_config is not None:
        required_paths.append(args.metric_config)
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    logging.info("RANDOM_SEED value=%d", args.seed)
    metric_options = load_metric_options(args.metric_config)
    logging.info(
        "METRIC_CONFIG path=%s warmup=%d/%d local_recovery=%d",
        args.metric_config or "defaults",
        metric_options.scale_warmup_measurements,
        metric_options.scale_warmup_min_keyframes,
        metric_options.local_recovery_consecutive_frames,
    )
    tracker = DPVSLAMMetricTracker(
        args.dpvo_root.resolve(),
        args.network.resolve(),
        args.config.resolve(),
        metric_options,
        args.loop_closure,
        args.gravity_align,
        args.finalized_trajectory.resolve() if args.finalized_trajectory else None,
    )
    serve(args.socket, tracker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
