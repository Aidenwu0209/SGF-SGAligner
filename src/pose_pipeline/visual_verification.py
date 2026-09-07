"""Optional, GT-free RGB-D witnesses for a proposed 3-D loop transform.

The historical ``recall`` thresholds are a frozen pilot configuration, not a
claim of production accuracy.  Visual estimates constitute one solver family;
they cannot independently authorize a pose or an edge entering PGO.

FrameRecord intrinsics describe the native depth image.  Like the submap and
refusion readers, this reader assumes color is already registered to depth
(possibly at another resolution).  Resizing cannot calibrate separate cameras;
unregistered RGB-D must be aligned upstream before constructing the manifest.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import threading

import numpy as np

from .contracts import FORBIDDEN_PARTS, FrameRecord, sha256_file, validate_se3
from .robust_backend import transform_distance


# OpenCV's RNG state must not interleave between this module's PnP calls.
_PNP_LOCK = threading.Lock()


@dataclass(frozen=True)
class VisualVerificationConfig:
    enabled: bool = False
    maximum_features: int = 4000
    contrast_threshold: float = 0.02
    ratio_threshold: float = 0.75
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 10.0
    ransac_iterations: int = 1000
    reprojection_error_px: float = 3.0
    ransac_confidence: float = 0.999
    minimum_depth_correspondences: int = 12
    minimum_inliers: int = 7
    minimum_inlier_ratio: float = 0.20
    maximum_registration_rotation_deg: float = 10.0
    maximum_registration_translation_m: float = 0.35
    maximum_cycle_rotation_deg: float = 10.0
    maximum_cycle_translation_m: float = 0.35
    seed: int = 20260904

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("visual verification enabled must be boolean")
        integers = {
            "maximum_features": self.maximum_features,
            "ransac_iterations": self.ransac_iterations,
            "minimum_depth_correspondences": self.minimum_depth_correspondences,
            "minimum_inliers": self.minimum_inliers,
            "seed": self.seed,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if value < (0 if name == "seed" else 1) or value > 2**31 - 2:
                raise ValueError(f"{name} is outside the supported integer range")
        if self.minimum_depth_correspondences < 4 or self.minimum_inliers < 4:
            raise ValueError("PnP requires at least four depth points and inliers")
        scalars = (
            self.contrast_threshold, self.ratio_threshold,
            self.minimum_depth_m, self.maximum_depth_m,
            self.reprojection_error_px, self.ransac_confidence,
            self.minimum_inlier_ratio,
            self.maximum_registration_rotation_deg,
            self.maximum_registration_translation_m,
            self.maximum_cycle_rotation_deg, self.maximum_cycle_translation_m,
        )
        try:
            valid = all(
                not isinstance(value, (bool, str))
                and np.ndim(value) == 0 and np.isfinite(value) and value > 0
                for value in scalars
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("visual verification thresholds must be finite and positive")
        if not 0 < self.ratio_threshold < 1 or not 0 < self.ransac_confidence < 1:
            raise ValueError("ratio threshold and RANSAC confidence must be in (0, 1)")
        if self.minimum_inlier_ratio > 1:
            raise ValueError("minimum inlier ratio must be at most one")
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum depth must exceed minimum depth")
        if max(self.maximum_registration_rotation_deg, self.maximum_cycle_rotation_deg) > 180:
            raise ValueError("rotation thresholds must be at most 180 degrees")
        # Evidence is written with allow_nan=False; normalize accepted NumPy
        # numeric scalars as well as ordinary YAML/Python values to JSON types.
        for name, value in asdict(self).items():
            if name != "enabled":
                object.__setattr__(self, name, int(value) if name in integers else float(value))


def _read_frame(cv2, frame: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for path in (frame.color_path, frame.depth_path):
        if {part.lower() for part in Path(path).resolve().parts} & FORBIDDEN_PARTS:
            raise ValueError("visual input contains a forbidden GT path component")
    color = cv2.imread(str(frame.color_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise OSError(f"RGB-D input missing or unreadable for frame {frame.frame_id}")
    if depth.ndim != 2 or depth.dtype != np.uint16 or 0 in depth.shape:
        raise ValueError("visual depth input must be a nonempty uint16 HxW image")
    if color.ndim != 2 or color.dtype != np.uint8 or 0 in color.shape:
        raise ValueError("visual color input must decode to nonempty uint8 grayscale")
    intrinsic_values = np.asarray(frame.intrinsics, dtype=np.float64)
    if intrinsic_values.shape != (4,) or not np.isfinite(intrinsic_values).all():
        raise ValueError("visual frame intrinsics must contain four finite values")
    fx, fy, cx, cy = intrinsic_values
    if fx <= 0 or fy <= 0:
        raise ValueError("visual frame focal lengths must be positive")
    if frame.rotate_ccw:
        old_width = depth.shape[1]
        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fx, fy, cx, cy = fy, fx, cy, old_width - 1.0 - cx
    height, width = depth.shape
    if color.shape != depth.shape:
        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    intrinsic = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    return color, depth, intrinsic


def _ratio_matches(cv2, source_descriptors, target_descriptors, ratio: float):
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        source_descriptors, target_descriptors, k=2,
    )
    return [
        pair[0] for pair in matches
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]


def _depth_correspondences(
    source_keypoints, target_keypoints, matches, depth: np.ndarray,
    intrinsic: np.ndarray, depth_scale: float, config: VisualVerificationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    object_points, image_points = [], []
    inverse_intrinsic = np.linalg.inv(intrinsic)
    height, width = depth.shape
    for match in matches:
        u, v = source_keypoints[int(match.queryIdx)].pt
        target_pixel = target_keypoints[int(match.trainIdx)].pt
        if not np.isfinite([u, v, *target_pixel]).all():
            continue
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height):
            continue
        z = float(depth[y, x]) / depth_scale
        if not config.minimum_depth_m < z < config.maximum_depth_m:
            continue
        object_points.append(inverse_intrinsic @ np.array([u, v, 1.]) * z)
        image_points.append(target_pixel)
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
    )


def _solve_pnp(
    cv2, source_points: np.ndarray, target_pixels: np.ndarray,
    target_intrinsic: np.ndarray, config: VisualVerificationConfig, seed: int,
) -> dict | None:
    if len(source_points) < max(4, config.minimum_depth_correspondences):
        return None
    with _PNP_LOCK:
        cv2.setRNGSeed(int(seed))
        ok, rotation_vector, translation, indices = cv2.solvePnPRansac(
            np.ascontiguousarray(source_points, dtype=np.float64),
            np.ascontiguousarray(target_pixels, dtype=np.float64),
            np.asarray(target_intrinsic, dtype=np.float64), None,
            iterationsCount=int(config.ransac_iterations),
            reprojectionError=float(config.reprojection_error_px),
            confidence=float(config.ransac_confidence), flags=cv2.SOLVEPNP_EPNP,
        )
    if not ok or indices is None or rotation_vector is None or translation is None:
        return None
    indices = np.asarray(indices).reshape(-1)
    if (
        not np.issubdtype(indices.dtype, np.integer) or len(indices) < 4
        or np.any(indices < 0) or np.any(indices >= len(source_points))
        or len(np.unique(indices)) != len(indices)
    ):
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    transform[:3, 3] = np.asarray(translation).reshape(3)
    transform = validate_se3(transform, "visual PnP T_target_source_m")
    target_points = source_points[indices] @ transform[:3, :3].T + transform[:3, 3]
    if not np.isfinite(target_points).all() or np.any(target_points[:, 2] <= 0):
        return None
    return {
        "transform": transform,
        "depth_correspondences": int(len(source_points)),
        "inliers": int(len(indices)),
        "inlier_ratio": float(len(indices) / len(source_points)),
    }


def _gate(
    evidence: dict, config: VisualVerificationConfig, *, require_registration: bool = True,
) -> tuple[bool, str]:
    for direction in ("forward", "reverse"):
        item = evidence.get(direction)
        if item is None:
            return False, f"{direction}_pnp_unavailable"
        checks = [
            ("depth_correspondences", config.minimum_depth_correspondences, True),
            ("inliers", config.minimum_inliers, True),
            ("inlier_ratio", config.minimum_inlier_ratio, True),
        ]
        if require_registration:
            checks.extend((
                ("registration_rotation_deg", config.maximum_registration_rotation_deg, False),
                ("registration_translation_m", config.maximum_registration_translation_m, False),
            ))
        for name, threshold, lower_bound in checks:
            value = item.get(name)
            if value is None or not np.isscalar(value) or not np.isfinite(value):
                return False, f"{direction}_{name}_invalid"
            fails = value < threshold if lower_bound else value > threshold
            if fails:
                return False, f"{direction}_{name}_gate_failed"
    for name, threshold in (
        ("cycle_rotation_deg", config.maximum_cycle_rotation_deg),
        ("cycle_translation_m", config.maximum_cycle_translation_m),
    ):
        value = evidence.get(name)
        if value is None or not np.isscalar(value) or not np.isfinite(value):
            return False, f"{name}_invalid"
        if value > threshold:
            return False, f"{name}_gate_failed"
    return True, (
        "bidirectional_rgbd_pnp_corroborates_registration" if require_registration
        else "bidirectional_rgbd_pnp_quality_and_cycle_pass"
    )


def estimate_rgbd_loop(
    source_frame: FrameRecord, target_frame: FrameRecord,
    depth_scale: float,
    config: VisualVerificationConfig = VisualVerificationConfig(),
) -> dict:
    """Estimate a single visual solver family without a 3-D registration input.

    Both output witnesses use T_target_source_m; the reverse solve is inverted.
    Accepted means only that depth/inlier/cycle quality passed.  It never
    authorizes a PGO edge: independent cross-family consensus, 3-D geometry
    gates and final visual-to-refined-registration comparison remain required.
    Disabled estimation performs no file or OpenCV access.
    """
    result = {
        "schema": "rgbd_visual_loop_estimate.v1",
        "solver_family": "rgbd_pnp",
        "usable_for_reconstruction": False,
        "evidence_role": "independent_solver_hypothesis_only",
        "accepted": False,
        "reason": "visual_verification_disabled",
        "forward": None, "reverse": None,
        "cycle_rotation_deg": None, "cycle_translation_m": None,
        "config": asdict(config),
        "calibration_status": "historical_recall_pilot_requires_revalidation",
        "matrix_convention": "T_target_source_m",
        "camera_basis": (
            "image_rotated_ccw_from_native"
            if source_frame.rotate_ccw else "native_sensor_camera"
        ),
        "rgbd_alignment_assumption": "registered_color_resized_to_depth_grid",
        "source_frame_id": int(source_frame.frame_id),
        "target_frame_id": int(target_frame.frame_id),
        "gt_consumed": False,
    }
    if not config.enabled:
        return result
    cv2 = None
    try:
        if (
            isinstance(depth_scale, bool)
            or not isinstance(depth_scale, (int, float, np.integer, np.floating))
            or not np.isfinite(depth_scale) or depth_scale <= 0
        ):
            raise ValueError("depth_scale must be finite and positive")
        if source_frame.frame_id == target_frame.frame_id:
            raise ValueError("visual loop endpoints must be distinct frames")
        if bool(source_frame.rotate_ccw) != bool(target_frame.rotate_ccw):
            raise ValueError("visual loop endpoints have inconsistent camera bases")
        cv2 = importlib.import_module("cv2")
        source_color, source_depth, source_k = _read_frame(cv2, source_frame)
        target_color, target_depth, target_k = _read_frame(cv2, target_frame)
        result["rgbd_inputs_sha256"] = {
            f"{role}_{kind}": sha256_file(path)
            for role, frame in (("source", source_frame), ("target", target_frame))
            for kind, path in (("color", frame.color_path), ("depth", frame.depth_path))
        }
        result["intrinsics_depth_basis"] = {
            "source": source_k.tolist(), "target": target_k.tolist(),
            "depth_scale": float(depth_scale),
        }
        result["opencv_version"] = cv2.__version__
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV runtime does not provide SIFT")
        sift = cv2.SIFT_create(
            nfeatures=int(config.maximum_features),
            contrastThreshold=float(config.contrast_threshold),
        )
        source_keypoints, source_desc = sift.detectAndCompute(source_color, None)
        target_keypoints, target_desc = sift.detectAndCompute(target_color, None)
        if source_desc is None or target_desc is None or min(len(source_desc), len(target_desc)) < 2:
            result["reason"] = "sift_descriptors_missing"
            return result
        forward_matches = _ratio_matches(cv2, source_desc, target_desc, config.ratio_threshold)
        reverse_matches = _ratio_matches(cv2, target_desc, source_desc, config.ratio_threshold)
        result["ratio_test_matches"] = {
            "forward": len(forward_matches), "reverse": len(reverse_matches),
        }
        source_3d, target_2d = _depth_correspondences(
            source_keypoints, target_keypoints, forward_matches,
            source_depth, source_k, float(depth_scale), config,
        )
        target_3d, source_2d = _depth_correspondences(
            target_keypoints, source_keypoints, reverse_matches,
            target_depth, target_k, float(depth_scale), config,
        )
        forward = _solve_pnp(cv2, source_3d, target_2d, target_k, config, config.seed)
        reverse = _solve_pnp(cv2, target_3d, source_2d, source_k, config, config.seed + 1)
        if reverse is not None:
            reverse["transform"] = validate_se3(np.linalg.inv(reverse["transform"]))
        for direction, item in (("forward", forward), ("reverse", reverse)):
            if item is not None:
                item["transform"] = item["transform"].tolist()
            result[direction] = item
        if forward is not None and reverse is not None:
            result["cycle_rotation_deg"], result["cycle_translation_m"] = transform_distance(
                forward["transform"], reverse["transform"],
            )
        result["accepted"], result["reason"] = _gate(result, config, require_registration=False)
    except (ImportError, OSError, RuntimeError, ValueError, MemoryError) as error:
        result["reason"] = "visual_verification_exception_fail_closed"
        result["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        # CV2 raises its own extension exception type; do not swallow unrelated
        # programming errors while still rejecting unavailable/corrupt inputs.
        if cv2 is None or not isinstance(error, cv2.error):
            raise
        result["reason"] = "visual_verification_exception_fail_closed"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def compare_visual_registration(
    evidence: dict, source_to_target: np.ndarray,
    config: VisualVerificationConfig = VisualVerificationConfig(),
) -> dict:
    """Pure comparison of cached visual evidence to a refined 3-D transform.

    The input is never modified and no file, SIFT, or PnP operation is repeated.
    Both visual witnesses must still pass quality/cycle and agree with 3-D.
    """
    result = deepcopy(evidence)
    result.update({
        "schema": "rgbd_visual_loop_verification.v1",
        "accepted": False,
        "config": asdict(config),
        "evidence_role": "registration_corroboration_only",
        "usable_for_reconstruction": False,
        "gt_consumed": False,
    })
    if not config.enabled:
        result["reason"] = "visual_verification_disabled"
        return result
    try:
        if evidence.get("schema") != "rgbd_visual_loop_estimate.v1":
            raise ValueError("visual estimate schema mismatch")
        if evidence.get("gt_consumed") is not False:
            raise ValueError("visual estimate must declare gt_consumed=false")
        if evidence.get("matrix_convention") != "T_target_source_m":
            raise ValueError("visual estimate transform convention mismatch")
        if evidence.get("accepted") is not True:
            # An estimator failure cannot be turned into success by supplying
            # a convenient registration or looser comparison configuration.
            result["reason"] = str(evidence.get("reason", "visual_estimate_rejected"))
            return result
        registration = validate_se3(source_to_target, "3-D T_target_source_m")
        result["registration_transform"] = registration.tolist()
        for direction in ("forward", "reverse"):
            item = result.get(direction)
            if item is not None:
                rotation, translation = transform_distance(item["transform"], registration)
                item["registration_rotation_deg"] = rotation
                item["registration_translation_m"] = translation
        # Recompute cycle from the actual matrices; cached scalar evidence
        # cannot substitute for a consistent pair of visual transformations.
        if result.get("forward") is not None and result.get("reverse") is not None:
            result["cycle_rotation_deg"], result["cycle_translation_m"] = transform_distance(
                result["forward"]["transform"], result["reverse"]["transform"],
            )
        result["accepted"], result["reason"] = _gate(result, config)
    except (KeyError, TypeError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        result["reason"] = "visual_verification_exception_fail_closed"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def verify_rgbd_loop(
    source_frame: FrameRecord, target_frame: FrameRecord,
    depth_scale: float, source_to_target: np.ndarray,
    config: VisualVerificationConfig = VisualVerificationConfig(),
) -> dict:
    """Compatibility wrapper: estimate once, then corroborate the 3-D pose."""
    evidence = estimate_rgbd_loop(source_frame, target_frame, depth_scale, config)
    return compare_visual_registration(evidence, source_to_target, config)
