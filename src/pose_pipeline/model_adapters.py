"""Adapters for official ABot-Recon, SLAM-Former and MapAnything outputs.

The adapters never run a model, read GT, fill missing poses with identities, or
accept a partial trajectory as a continuous frontend. Heavy model environments
remain isolated; only small NumPy/JSON/TUM artifacts cross this boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    PoseRecord,
    SequenceManifest,
    load_manifest,
    load_trajectory,
    sha256_file,
    validate_se3,
    write_trajectory,
)
from .model_contracts import write_sparse_proposals
from .pose_graph import propagate_anchor_corrections


def _exact_version(model_commit: str, checkpoint_sha256: str) -> tuple[str, str]:
    for value, length, name in (
        (model_commit, 40, "model commit"),
        (checkpoint_sha256, 64, "checkpoint SHA256"),
    ):
        if len(value) != length:
            raise ValueError(f"{name} must contain {length} hexadecimal characters")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{name} must be hexadecimal") from exc
    return model_commit.lower(), checkpoint_sha256.lower()


def estimate_metric_scale(
    predicted_depth: object,
    sensor_depth_m: object,
    *,
    confidence: object | None = None,
    minimum_samples: int = 64,
) -> tuple[float, dict]:
    """Robustly estimate metres/model-unit from paired depth observations."""
    predicted = np.asarray(predicted_depth, dtype=np.float64).reshape(-1)
    sensor = np.asarray(sensor_depth_m, dtype=np.float64).reshape(-1)
    if predicted.shape != sensor.shape:
        raise ValueError("predicted and sensor depth shapes differ")
    valid = np.isfinite(predicted) & np.isfinite(sensor) & (predicted > 1e-6) & (sensor > 1e-6)
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
        if conf.shape != predicted.shape:
            raise ValueError("confidence shape differs from depth")
        valid &= np.isfinite(conf) & (conf > 0)
    ratios = sensor[valid] / predicted[valid]
    if len(ratios) < minimum_samples:
        raise ValueError(f"metric scale needs at least {minimum_samples} valid samples")
    median = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - median)))
    if mad > 0:
        ratios = ratios[np.abs(ratios - median) <= 3.5 * 1.4826 * mad]
    lower, upper = np.percentile(ratios, [5, 95])
    trimmed = ratios[(ratios >= lower) & (ratios <= upper)]
    if len(trimmed) < minimum_samples:
        raise ValueError("metric scale became under-supported after outlier rejection")
    scale = float(np.median(trimmed))
    relative_spread = float((np.percentile(trimmed, 95) - np.percentile(trimmed, 5)) / scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric scale is invalid")
    return scale, {
        "method": "paired_sensor_depth_robust_median",
        "valid_sample_count": int(len(trimmed)),
        "scale_m_per_model_unit": scale,
        "p05_p95_relative_spread": relative_spread,
    }


def load_scale_evidence(path: Path) -> tuple[float, dict]:
    """Read a create-only NPZ with predicted_depth and sensor_depth_m arrays."""
    with np.load(path) as payload:
        if "predicted_depth" not in payload or "sensor_depth_m" not in payload:
            raise ValueError("scale evidence needs predicted_depth and sensor_depth_m")
        confidence = payload["confidence"] if "confidence" in payload else None
        scale, report = estimate_metric_scale(
            payload["predicted_depth"], payload["sensor_depth_m"], confidence=confidence,
        )
        if "manifest_sha256" in payload:
            report["input_manifest_sha256"] = str(payload["manifest_sha256"].item())
    report["evidence_sha256"] = sha256_file(path)
    return scale, report


def build_abot_scale_evidence(
    *,
    manifest_path: Path,
    local_points_path: Path,
    output_path: Path,
    confidence_path: Path | None = None,
    maximum_frames: int = 32,
    sample_stride: int = 8,
) -> dict:
    """Pair official ABot local point-map Z with the original metric depth."""
    if maximum_frames < 1 or sample_stride < 1:
        raise ValueError("maximum frames and sample stride must be positive")
    import cv2
    import torch

    manifest = load_manifest(manifest_path)
    points_tensor = torch.load(local_points_path, map_location="cpu", weights_only=True)
    points = np.asarray(points_tensor.detach().cpu(), dtype=np.float64)
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("ABot local_points.pt must have shape [N,H,W,3]")
    if len(points) != len(manifest.frames):
        raise ValueError("ABot local point maps must cover every manifest frame")
    confidence = None
    if confidence_path is not None:
        tensor = torch.load(confidence_path, map_location="cpu", weights_only=True)
        confidence = np.asarray(tensor.detach().cpu(), dtype=np.float64)
        confidence = np.squeeze(confidence)
        if confidence.shape != points.shape[:3]:
            raise ValueError("ABot confidence does not align with local point maps")
    selected = np.unique(np.linspace(
        0, len(manifest.frames) - 1,
        min(maximum_frames, len(manifest.frames)), dtype=np.int64,
    ))
    predicted_rows, sensor_rows, confidence_rows = [], [], []
    used_frame_ids = []
    for ordinal in selected:
        frame = manifest.frames[int(ordinal)]
        raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2:
            raise ValueError(f"cannot read metric depth for frame {frame.frame_id}")
        if frame.rotate_ccw:
            raw = np.rot90(raw)
        height, width = points[int(ordinal)].shape[:2]
        sensor_m = cv2.resize(
            raw.astype(np.float64) / manifest.depth_scale,
            (width, height), interpolation=cv2.INTER_NEAREST,
        )
        predicted = points[int(ordinal), :, :, 2]
        predicted_rows.append(predicted[::sample_stride, ::sample_stride].reshape(-1))
        sensor_rows.append(sensor_m[::sample_stride, ::sample_stride].reshape(-1))
        if confidence is not None:
            confidence_rows.append(
                confidence[int(ordinal), ::sample_stride, ::sample_stride].reshape(-1)
            )
        used_frame_ids.append(frame.frame_id)
    predicted_depth = np.concatenate(predicted_rows)
    sensor_depth_m = np.concatenate(sensor_rows)
    confidence_values = np.concatenate(confidence_rows) if confidence_rows else None
    scale, report = estimate_metric_scale(
        predicted_depth, sensor_depth_m, confidence=confidence_values,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        values = {
            "predicted_depth": predicted_depth,
            "sensor_depth_m": sensor_depth_m,
            "manifest_sha256": np.asarray(manifest.as_dict()["payload_sha256"]),
            "local_points_sha256": np.asarray(sha256_file(local_points_path)),
            "frame_ids": np.asarray(used_frame_ids, dtype=np.int64),
        }
        if confidence_values is not None:
            values["confidence"] = confidence_values
            values["confidence_sha256"] = np.asarray(sha256_file(confidence_path))
        np.savez_compressed(stream, **values)
    return {
        **report,
        "path": str(target.resolve()),
        "evidence_sha256": sha256_file(target),
        "frame_ids": used_frame_ids,
        "gt_consumed": False,
    }


def _scaled_poses(poses: object, scale: float) -> list[np.ndarray]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not np.isfinite(values).all():
        raise ValueError("camera poses must have finite shape [N,4,4]")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric scale must be finite and positive")
    output = []
    for index, value in enumerate(values):
        pose = value.copy()
        pose[:3, 3] *= scale
        output.append(validate_se3(pose, f"model pose {index}"))
    return output


def _records_for_manifest(
    manifest: SequenceManifest,
    poses: Sequence[np.ndarray],
    *,
    source: str,
) -> list[PoseRecord]:
    if len(poses) != len(manifest.frames):
        raise ValueError(
            f"continuous frontend must cover every manifest frame: {len(poses)} != {len(manifest.frames)}"
        )
    return [PoseRecord(
        frame_id=frame.frame_id,
        timestamp_us=frame.timestamp_us,
        t_world_camera=pose,
        source=source,
    ) for frame, pose in zip(manifest.frames, poses)]


def adapt_abot_trajectory(
    *,
    manifest_path: Path,
    poses_path: Path,
    output_path: Path,
    mode: str,
    metric_scale: float,
    scale_evidence: Mapping[str, object],
    model_commit: str,
    checkpoint_sha256: str,
) -> dict:
    """Import ABot full-frame c2w output as a metric trajectory."""
    if mode not in {"noloop", "official_loop"}:
        raise ValueError("ABot mode must be noloop or official_loop")
    expected_name = "camera_poses_noloop.npy" if mode == "noloop" else "camera_poses_loop.npy"
    if Path(poses_path).name != expected_name:
        raise ValueError(f"ABot {mode} arm requires {expected_name}")
    model_commit, checkpoint_sha256 = _exact_version(model_commit, checkpoint_sha256)
    manifest = load_manifest(manifest_path)
    evidence_manifest = scale_evidence.get("input_manifest_sha256")
    if evidence_manifest is not None and evidence_manifest != manifest.as_dict()["payload_sha256"]:
        raise ValueError("ABot scale evidence belongs to a different manifest")
    poses = _scaled_poses(np.load(poses_path), metric_scale)
    records = _records_for_manifest(manifest, poses, source=f"ABot-Recon:{mode}")
    write_trajectory(
        output_path, records, sequence_id=manifest.sequence_id,
        arm=f"abot_{mode}", metadata={
            "provider": "ABot-Recon",
            "model_commit": model_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "official_output_sha256": sha256_file(poses_path),
            "official_loop_mode": mode == "official_loop",
            "scale_evidence": dict(scale_evidence),
            "coordinate_conversion": "official c2w -> T_world_camera; translation scaled only",
        },
    )
    return {"trajectory": str(Path(output_path).resolve()), "poses": len(records), "scale": metric_scale}


def import_abot_loop_proposals(
    *,
    manifest_path: Path,
    loop_edges_path: Path,
    output_path: Path,
    metric_scale: float,
) -> dict:
    """Import official ABot revisit edges as undecided sparse proposals."""
    manifest = load_manifest(manifest_path)
    rows = json.loads(Path(loop_edges_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("ABot loop_edges.json must contain a list")
    frame_ids = [frame.frame_id for frame in manifest.frames]
    proposals = []
    for row in rows:
        source_ordinal = int(row["src_frame"])
        target_ordinal = int(row["dst_frame"])
        if not (0 <= source_ordinal < len(frame_ids) and 0 <= target_ordinal < len(frame_ids)):
            raise ValueError("ABot loop ordinal is outside the manifest")
        transform = np.asarray(row["transform_ji"], dtype=np.float64)
        transform[:3, 3] *= metric_scale
        proposals.append({
            "source_frame_id": frame_ids[source_ordinal],
            "target_frame_id": frame_ids[target_ordinal],
            "T_target_source_m": validate_se3(transform),
            "score": float(row.get("score", 0.0)),
            "method": str(row.get("method", "ABot-Recon")),
        })
    return write_sparse_proposals(
        output_path, sequence_id=manifest.sequence_id, provider="ABot-Recon",
        proposals=proposals, source_sha256=sha256_file(loop_edges_path),
    )


def _quaternion_xyzw_matrix(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain finite x,y,z,w")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_slamformer_anchors(
    path: Path,
    manifest: SequenceManifest,
    *,
    metric_scale: float,
    identifier_mode: str = "auto",
) -> tuple[list[int], list[np.ndarray]]:
    """Parse official ``final_traj.txt`` (id tx ty tz qx qy qz qw)."""
    values = np.loadtxt(path, dtype=np.float64)
    values = values.reshape(1, -1) if values.ndim == 1 else values
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError("SLAM-Former trajectory must be TUM rows with 8 fields")
    by_frame = {frame.frame_id: index for index, frame in enumerate(manifest.frames)}
    by_timestamp = {frame.timestamp_us: index for index, frame in enumerate(manifest.frames)}

    def resolve(identifier: float) -> int:
        choices = []
        rounded = int(round(identifier))
        if identifier_mode in {"auto", "frame_id"} and abs(identifier - rounded) < 1e-5 and rounded in by_frame:
            choices.append(by_frame[rounded])
        if identifier_mode in {"auto", "timestamp_us"} and abs(identifier - rounded) < 1e-5 and rounded in by_timestamp:
            choices.append(by_timestamp[rounded])
        timestamp_us = int(round(identifier * 1_000_000.0))
        if identifier_mode in {"auto", "timestamp_s"} and timestamp_us in by_timestamp:
            choices.append(by_timestamp[timestamp_us])
        if len(set(choices)) != 1:
            raise ValueError(f"SLAM-Former keyframe identifier is missing or ambiguous: {identifier}")
        return choices[0]

    ordinals, poses = [], []
    for row in values:
        ordinal = resolve(float(row[0]))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _quaternion_xyzw_matrix(row[4:8])
        pose[:3, 3] = row[1:4] * metric_scale
        ordinals.append(ordinal)
        poses.append(validate_se3(pose, f"SLAM-Former anchor {row[0]}"))
    if ordinals != sorted(set(ordinals)):
        raise ValueError("SLAM-Former anchors must be unique and manifest-ordered")
    return ordinals, poses


def _propagate_external_anchors(
    baseline: Sequence[PoseRecord],
    ordinals: Sequence[int],
    external_poses: Sequence[np.ndarray],
    *,
    source: str,
) -> list[PoseRecord]:
    if len(ordinals) < 2:
        raise ValueError("at least two external anchors are required")
    gauge = baseline[ordinals[0]].t_world_camera @ np.linalg.inv(external_poses[0])
    aligned = [validate_se3(gauge @ pose) for pose in external_poses]
    corrected = propagate_anchor_corrections(baseline, ordinals, aligned)
    return [PoseRecord(
        frame_id=row.frame_id, timestamp_us=row.timestamp_us,
        t_world_camera=row.t_world_camera, source=source,
    ) for row in corrected]


def adapt_slamformer_revision(
    *,
    manifest_path: Path,
    baseline_trajectory_path: Path,
    final_traj_path: Path,
    output_path: Path,
    metric_scale: float,
    scale_evidence: Mapping[str, object],
    identifier_mode: str,
    model_variant: str,
    model_commit: str,
    checkpoint_sha256: str,
) -> dict:
    model_commit, checkpoint_sha256 = _exact_version(model_commit, checkpoint_sha256)
    manifest = load_manifest(manifest_path)
    evidence_manifest = scale_evidence.get("input_manifest_sha256")
    if evidence_manifest is not None and evidence_manifest != manifest.as_dict()["payload_sha256"]:
        raise ValueError("SLAM-Former scale evidence belongs to a different manifest")
    baseline, baseline_payload = load_trajectory(baseline_trajectory_path)
    if baseline_payload["sequence_id"] != manifest.sequence_id:
        raise ValueError("baseline and manifest sequence mismatch")
    if [(row.frame_id, row.timestamp_us) for row in baseline] != [
        (frame.frame_id, frame.timestamp_us) for frame in manifest.frames
    ]:
        raise ValueError("baseline must cover the complete manifest")
    ordinals, anchors = load_slamformer_anchors(
        final_traj_path, manifest, metric_scale=metric_scale,
        identifier_mode=identifier_mode,
    )
    records = _propagate_external_anchors(
        baseline, ordinals, anchors, source=f"DPV+SLAM-Former:{model_variant}",
    )
    write_trajectory(
        output_path, records, sequence_id=manifest.sequence_id,
        arm="slamformer_anchor_revision", metadata={
            "provider": "SLAM-Former",
            "variant": model_variant,
            "model_commit": model_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "official_output_sha256": sha256_file(final_traj_path),
            "anchor_frame_ids": [manifest.frames[index].frame_id for index in ordinals],
            "non_keyframe_policy": "propagate_anchor_correction_over_complete_DPV_trajectory",
            "scale_evidence": dict(scale_evidence),
        },
    )
    return {"trajectory": str(Path(output_path).resolve()), "anchors": len(ordinals), "poses": len(records)}


def load_mapanything_window(path: Path, *, metric_scale: float) -> tuple[list[int], list[np.ndarray]]:
    with np.load(path) as payload:
        if "frame_ids" not in payload or "camera_poses" not in payload:
            raise ValueError("MapAnything window needs frame_ids and camera_poses")
        frame_ids = [int(value) for value in payload["frame_ids"].reshape(-1)]
        poses = _scaled_poses(payload["camera_poses"], metric_scale)
    if len(frame_ids) != len(poses) or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("MapAnything window frame ids are duplicated or misaligned")
    return frame_ids, poses


def stitch_mapanything_windows(
    windows: Sequence[tuple[Sequence[int], Sequence[np.ndarray]]],
    *,
    maximum_overlap_translation_m: float = 0.08,
    maximum_overlap_rotation_deg: float = 5.0,
) -> tuple[dict[int, np.ndarray], dict]:
    """Gauge-align overlapping windows and reject inconsistent windows."""
    if not windows:
        raise ValueError("no MapAnything windows")
    merged: dict[int, np.ndarray] = {}
    accepted, rejected = [], []
    for window_index, (frame_ids, poses) in enumerate(windows):
        current = {int(frame_id): validate_se3(pose) for frame_id, pose in zip(frame_ids, poses)}
        if not current or len(current) != len(frame_ids):
            raise ValueError("invalid MapAnything window")
        overlap = sorted(set(merged) & set(current))
        if not merged:
            gauge = np.eye(4)
        elif not overlap:
            rejected.append({"window": window_index, "reason": "no_overlap"})
            continue
        else:
            first = overlap[0]
            gauge = merged[first] @ np.linalg.inv(current[first])
            translation_errors, rotation_errors = [], []
            for frame_id in overlap:
                residual = np.linalg.inv(merged[frame_id]) @ validate_se3(gauge @ current[frame_id])
                translation_errors.append(float(np.linalg.norm(residual[:3, 3])))
                cosine = np.clip((np.trace(residual[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
                rotation_errors.append(float(np.degrees(np.arccos(cosine))))
            if max(translation_errors) > maximum_overlap_translation_m or max(rotation_errors) > maximum_overlap_rotation_deg:
                rejected.append({
                    "window": window_index,
                    "reason": "overlap_se3_inconsistent",
                    "maximum_translation_m": max(translation_errors),
                    "maximum_rotation_deg": max(rotation_errors),
                })
                continue
        aligned = {frame_id: validate_se3(gauge @ pose) for frame_id, pose in current.items()}
        for frame_id, pose in aligned.items():
            merged.setdefault(frame_id, pose)
        accepted.append(window_index)
    return merged, {
        "accepted_windows": accepted,
        "rejected_windows": rejected,
        "fail_closed": True,
    }


def adapt_mapanything_revision(
    *,
    manifest_path: Path,
    baseline_trajectory_path: Path,
    window_paths: Sequence[Path],
    output_path: Path,
    metric_scale: float,
    input_mode: str,
    window_size: int,
    model_commit: str,
    checkpoint_sha256: str,
    maximum_overlap_translation_m: float = 0.08,
    maximum_overlap_rotation_deg: float = 5.0,
) -> dict:
    if input_mode not in {"independent_rgb_intrinsics_depth", "conditioned_on_dpv_pose"}:
        raise ValueError("unsupported MapAnything input mode")
    if window_size not in {8, 16}:
        raise ValueError("MapAnything window size must be 8 or 16")
    model_commit, checkpoint_sha256 = _exact_version(model_commit, checkpoint_sha256)
    manifest = load_manifest(manifest_path)
    baseline, baseline_payload = load_trajectory(baseline_trajectory_path)
    if baseline_payload["sequence_id"] != manifest.sequence_id or len(baseline) != len(manifest.frames):
        raise ValueError("MapAnything requires a complete matching DPV baseline")
    windows = [load_mapanything_window(path, metric_scale=metric_scale) for path in window_paths]
    if any(len(frame_ids) > window_size for frame_ids, _ in windows):
        raise ValueError("MapAnything output exceeds declared window size")
    merged, stitch_report = stitch_mapanything_windows(
        windows,
        maximum_overlap_translation_m=maximum_overlap_translation_m,
        maximum_overlap_rotation_deg=maximum_overlap_rotation_deg,
    )
    ordinal_by_id = {frame.frame_id: index for index, frame in enumerate(manifest.frames)}
    if not set(merged) <= set(ordinal_by_id):
        raise ValueError("MapAnything output contains unknown frame ids")
    ordinals = sorted(ordinal_by_id[frame_id] for frame_id in merged)
    if len(ordinals) < 2:
        raise ValueError("MapAnything produced fewer than two accepted anchors")
    anchors = [merged[manifest.frames[ordinal].frame_id] for ordinal in ordinals]
    records = _propagate_external_anchors(
        baseline, ordinals, anchors, source=f"DPV+MapAnything:{input_mode}",
    )
    write_trajectory(
        output_path, records, sequence_id=manifest.sequence_id,
        arm="mapanything_background_revision", metadata={
            "provider": "MapAnything",
            "input_mode": input_mode,
            "window_size": window_size,
            "model_commit": model_commit,
            "checkpoint_sha256": checkpoint_sha256,
            "window_sha256": [sha256_file(path) for path in window_paths],
            "anchor_frame_ids": [manifest.frames[index].frame_id for index in ordinals],
            "stitch_report": stitch_report,
            "execution_policy": "background_nonblocking",
            "non_anchor_policy": "propagate_anchor_correction_over_complete_DPV_trajectory",
        },
    )
    return {
        "trajectory": str(Path(output_path).resolve()),
        "anchors": len(ordinals),
        "accepted_windows": len(stitch_report["accepted_windows"]),
        "rejected_windows": len(stitch_report["rejected_windows"]),
    }
