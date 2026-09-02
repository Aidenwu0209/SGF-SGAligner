"""Fail-closed contracts for external pose models and delayed map updates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    load_manifest,
    load_trajectory,
    sha256_file,
    stable_json_sha256,
)


MODEL_RUNTIME_SCHEMA = "model_runtime_report.v1"
TRAJECTORY_REVISION_SCHEMA = "trajectory_revision.v1"
SPARSE_PROPOSAL_SCHEMA = "sparse_constraint_proposals.v1"
EXTERNAL_ARTIFACT_SCHEMA = "external_artifact_manifest.v1"


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value.lower()


def _require_git_commit(value: str) -> str:
    if len(value) != 40:
        raise ValueError("model commit must be a full 40-character Git SHA")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("model commit must be hexadecimal") from exc
    return value.lower()


def _create_only_json(path: Path, payload: Mapping[str, object]) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    unsigned = dict(payload)
    value = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return value


def _load_signed_json(path: Path, schema: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"expected {schema}, got {payload.get('schema')}")
    expected = payload.pop("payload_sha256", None)
    if expected != stable_json_sha256(payload):
        raise ValueError(f"{schema} payload SHA mismatch")
    return {**payload, "payload_sha256": expected}


def write_model_runtime_report(
    path: Path,
    *,
    manifest_path: Path,
    model: str,
    model_commit: str,
    checkpoint_path: Path | None,
    checkpoint_sha256: str | None = None,
    resolution: tuple[int, int],
    latency_ms: Sequence[float],
    peak_gpu_memory_mb: float | None,
    output_pose_count: int,
    dropped_frame_ids: Sequence[int] = (),
    queue_depth_peak: int = 0,
    wall_time_s: float | None = None,
    mode: str = "official_weights",
    status: str = "completed",
    failure: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    """Write a reproducible runtime profile without accepting GT metadata."""
    manifest = load_manifest(manifest_path)
    samples = np.asarray(latency_ms, dtype=np.float64)
    if status not in {"completed", "failed"}:
        raise ValueError("runtime status must be completed or failed")
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("latency_ms must be a finite one-dimensional list")
    if status == "completed" and not len(samples):
        raise ValueError("completed runtime report needs latency samples")
    if status == "failed" and not failure:
        raise ValueError("failed runtime report needs structured failure metadata")
    if status == "completed" and failure:
        raise ValueError("completed runtime report may not contain failure metadata")
    if (samples < 0).any():
        raise ValueError("latency samples must be non-negative")
    if not model:
        raise ValueError("model is required")
    model_commit = _require_git_commit(model_commit)
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("resolution must be positive width,height")
    if peak_gpu_memory_mb is None:
        if status == "completed":
            raise ValueError("completed runtime report needs peak GPU memory")
    elif peak_gpu_memory_mb < 0 or not np.isfinite(peak_gpu_memory_mb):
        raise ValueError("peak GPU memory must be finite and non-negative")
    if output_pose_count < 0 or output_pose_count > len(manifest.frames):
        raise ValueError("output pose count is outside manifest range")
    dropped = sorted(set(int(value) for value in dropped_frame_ids))
    frame_ids = {frame.frame_id for frame in manifest.frames}
    if not set(dropped) <= frame_ids:
        raise ValueError("dropped frame ids are not in the manifest")
    if checkpoint_path is not None:
        actual = sha256_file(Path(checkpoint_path))
        if checkpoint_sha256 is not None and checkpoint_sha256 != actual:
            raise ValueError("checkpoint SHA mismatch")
        checkpoint_sha256 = actual
    if not checkpoint_sha256:
        raise ValueError("an exact checkpoint SHA256 is required")
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint SHA256")
    if wall_time_s is None and not len(samples):
        raise ValueError("failed run without latency samples requires wall time")
    elapsed = float(wall_time_s if wall_time_s is not None else samples.sum() / 1000.0)
    if elapsed <= 0 or not np.isfinite(elapsed):
        raise ValueError("wall time must be finite and positive")
    coverage = output_pose_count / len(manifest.frames)
    payload = {
        "schema": MODEL_RUNTIME_SCHEMA,
        "sequence_id": manifest.sequence_id,
        "model": model,
        "mode": mode,
        "status": status,
        "failure": dict(failure) if failure else None,
        "model_commit": model_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "input_manifest_sha256": manifest.as_dict()["payload_sha256"],
        "resolution": {"width": int(resolution[0]), "height": int(resolution[1])},
        "input_frame_count": len(manifest.frames),
        "output_pose_count": int(output_pose_count),
        "coverage": float(coverage),
        "dropped_frame_ids": dropped,
        "dropped_frame_count": len(dropped),
        "latency_ms": {
            "sample_count": int(len(samples)),
            "p50": float(np.percentile(samples, 50)) if len(samples) else None,
            "p95": float(np.percentile(samples, 95)) if len(samples) else None,
        },
        "wall_time_s": elapsed,
        "throughput_fps": float(output_pose_count / elapsed),
        "attempted_input_fps": float(len(manifest.frames) / elapsed),
        "peak_gpu_memory_mb": (
            float(peak_gpu_memory_mb) if peak_gpu_memory_mb is not None else None
        ),
        "queue_depth_peak": int(queue_depth_peak),
        "gt_consumed": False,
        "identity_fallback_used": False,
        "metadata": dict(metadata or {}),
    }
    return _create_only_json(path, payload)


def load_model_runtime_report(path: Path) -> dict:
    payload = _load_signed_json(path, MODEL_RUNTIME_SCHEMA)
    if payload.get("gt_consumed") is not False:
        raise ValueError("runtime report consumed GT")
    if payload.get("identity_fallback_used") is not False:
        raise ValueError("runtime report used identity fallback")
    if payload.get("status") == "failed" and not payload.get("failure"):
        raise ValueError("failed runtime report misses failure metadata")
    return payload


def write_trajectory_revision(
    path: Path,
    *,
    parent_trajectory_path: Path,
    revised_trajectory_path: Path,
    source: str,
    affected_frame_ids: Sequence[int],
    runtime_report_path: Path | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    """Bind a full revised trajectory to its parent and delay map replacement."""
    parent, parent_payload = load_trajectory(parent_trajectory_path)
    revised, revised_payload = load_trajectory(revised_trajectory_path)
    if parent_payload["sequence_id"] != revised_payload["sequence_id"]:
        raise ValueError("trajectory revision sequence mismatch")
    parent_keys = [(row.frame_id, row.timestamp_us) for row in parent]
    revised_keys = [(row.frame_id, row.timestamp_us) for row in revised]
    if parent_keys != revised_keys:
        raise ValueError("trajectory revision must preserve complete frame coverage")
    actual_changed = [
        old.frame_id for old, new in zip(parent, revised)
        if not np.allclose(old.t_world_camera, new.t_world_camera, atol=1e-9)
    ]
    claimed = sorted(set(int(value) for value in affected_frame_ids))
    if not actual_changed:
        raise ValueError("trajectory revision contains no pose change")
    if not set(actual_changed) <= set(claimed):
        raise ValueError("affected frame ids do not cover every changed pose")
    if not set(claimed) <= {row.frame_id for row in parent}:
        raise ValueError("affected frame ids are outside the trajectory")
    corrections = [
        new.t_world_camera @ np.linalg.inv(old.t_world_camera)
        for old, new in zip(parent, revised)
    ]

    def motion(transform: np.ndarray) -> tuple[float, float]:
        translation = float(np.linalg.norm(transform[:3, 3]))
        cosine = float(np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        return translation, float(np.degrees(np.arccos(cosine)))

    correction_motion = [motion(value) for value in corrections]
    derivative_motion = [
        motion(np.linalg.inv(left) @ right)
        for left, right in zip(corrections, corrections[1:])
    ]
    runtime_sha = None
    if runtime_report_path is not None:
        report = load_model_runtime_report(runtime_report_path)
        if report["sequence_id"] != parent_payload["sequence_id"]:
            raise ValueError("runtime report sequence mismatch")
        runtime_sha = report["payload_sha256"]
    payload = {
        "schema": TRAJECTORY_REVISION_SCHEMA,
        "sequence_id": parent_payload["sequence_id"],
        "source": source,
        "parent_trajectory_sha256": parent_payload["payload_sha256"],
        "revised_trajectory_sha256": revised_payload["payload_sha256"],
        "runtime_report_sha256": runtime_sha,
        "affected_frame_ids": claimed,
        "affected_frame_range": [min(claimed), max(claimed)],
        "actual_changed_frame_count": len(actual_changed),
        "correction_audit": {
            "maximum_translation_m": max(value[0] for value in correction_motion),
            "maximum_rotation_deg": max(value[1] for value in correction_motion),
            "maximum_adjacent_translation_derivative_m": (
                max(value[0] for value in derivative_motion) if derivative_motion else 0.0
            ),
            "maximum_adjacent_rotation_derivative_deg": (
                max(value[1] for value in derivative_motion) if derivative_motion else 0.0
            ),
            "finite": bool(all(np.isfinite(value).all() for value in corrections)),
        },
        "pose_display_update": "incremental_allowed",
        "map_update_policy": "delayed_full_refusion",
        "map_may_switch_before_full_refusion": False,
        "gt_consumed": False,
        "identity_fallback_used": False,
        "metadata": dict(metadata or {}),
    }
    return _create_only_json(path, payload)


def load_trajectory_revision(path: Path) -> dict:
    payload = _load_signed_json(path, TRAJECTORY_REVISION_SCHEMA)
    if payload.get("map_update_policy") != "delayed_full_refusion":
        raise ValueError("trajectory revision may not update the map incrementally")
    return payload


def write_sparse_proposals(
    path: Path,
    *,
    sequence_id: str,
    provider: str,
    proposals: Sequence[Mapping[str, object]],
    source_sha256: str,
) -> dict:
    if not proposals:
        raise ValueError("sparse proposal set is empty")
    normalized = []
    for proposal in proposals:
        transform = np.asarray(proposal["T_target_source_m"], dtype=np.float64)
        from .contracts import validate_se3
        normalized.append({
            "source_frame_id": int(proposal["source_frame_id"]),
            "target_frame_id": int(proposal["target_frame_id"]),
            "T_target_source_m": validate_se3(transform).reshape(-1).tolist(),
            "score": float(proposal.get("score", 0.0)),
            "method": str(proposal.get("method", provider)),
            "decision_status": "pending_registration_decision",
        })
    return _create_only_json(path, {
        "schema": SPARSE_PROPOSAL_SCHEMA,
        "sequence_id": sequence_id,
        "provider": provider,
        "source_sha256": source_sha256,
        "proposals": normalized,
        "may_bypass_registration_decision": False,
        "gt_consumed": False,
    })


def write_external_artifact_manifest(
    path: Path,
    *,
    manifest_path: Path,
    system: str,
    role: str,
    artifacts: Sequence[Path],
    source_frame_ids: Sequence[int],
    runtime_s: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict:
    allowed_roles = {"offline_geometry_control", "presentation_only"}
    if role not in allowed_roles:
        raise ValueError(f"external artifact role must be one of {sorted(allowed_roles)}")
    manifest = load_manifest(manifest_path)
    selected = [int(value) for value in source_frame_ids]
    known = {frame.frame_id for frame in manifest.frames}
    if not selected or not set(selected) <= known or len(selected) != len(set(selected)):
        raise ValueError("source frame list is empty, duplicated, or outside manifest")
    if system.lower() == "fixanything" and (role != "presentation_only" or len(selected) != 61):
        raise ValueError("FixAnything requires exactly 61 frames and presentation_only role")
    rows = []
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        rows.append({"path": str(resolved), "sha256": sha256_file(resolved)})
    if not rows:
        raise ValueError("at least one external artifact is required")
    suffixes = {Path(row["path"]).suffix.lower() for row in rows}
    if system.lower() == "mipmap" and ".ply" not in suffixes:
        raise ValueError("MipMap control requires an exported PLY")
    if system.lower() == "fixanything" and not suffixes & {".mp4", ".mov", ".webm"}:
        raise ValueError("FixAnything presentation artifact requires a video")
    return _create_only_json(path, {
        "schema": EXTERNAL_ARTIFACT_SCHEMA,
        "sequence_id": manifest.sequence_id,
        "system": system,
        "role": role,
        "input_manifest_sha256": manifest.as_dict()["payload_sha256"],
        "source_frame_ids": selected,
        "artifacts": rows,
        "runtime_s": runtime_s,
        "eligible_for_online_pose_pipeline": False,
        "may_feed_rgbd_semantic_or_geometry_pipeline": False,
        "gt_consumed": False,
        "metadata": dict(metadata or {}),
    })
