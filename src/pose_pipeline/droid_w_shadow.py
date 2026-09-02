"""Fail-closed importer for DROID-W shadow trajectories.

The official full-trajectory text is written before its evaluation-only Sim(3)
alignment.  This importer accepts only that raw trajectory together with an
explicit no-GT provenance record and turns it into the SGF metric pose contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import (
    PoseRecord, load_manifest, sha256_file, stable_json_sha256, validate_se3,
    write_trajectory,
)


REQUIRED_PROVENANCE = {
    "provider_commit", "tracking_checkpoint_sha256", "metric_depth_provider",
    "metric_depth_checkpoint_sha256", "config_sha256", "gt_consumed",
    "sim3_alignment_used", "trajectory_stage",
}


def _rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = q / norm
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _load_provenance(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    missing = REQUIRED_PROVENANCE - set(value)
    if missing:
        raise ValueError(f"DROID-W provenance missing: {sorted(missing)}")
    if value["gt_consumed"] is not False:
        raise ValueError("DROID-W inference provenance consumed GT")
    if value["sim3_alignment_used"] is not False:
        raise ValueError("evaluation-aligned DROID-W trajectory is forbidden")
    if value["trajectory_stage"] != "raw_full_trajectory_before_evaluation":
        raise ValueError("DROID-W trajectory must be captured before evaluation")
    for key in ("provider_commit", "tracking_checkpoint_sha256",
                "metric_depth_checkpoint_sha256", "config_sha256"):
        expected = 40 if key == "provider_commit" else 64
        if len(str(value[key])) != expected:
            raise ValueError(f"bad provenance digest: {key}")
    return value


def import_droid_w_shadow(
    *, manifest_path: Path, trajectory_path: Path, provenance_path: Path,
    output_path: Path, audit_path: Path,
    maximum_step_translation_m: float = 1.5,
) -> dict:
    manifest = load_manifest(manifest_path)
    provenance = _load_provenance(provenance_path)
    raw = np.loadtxt(trajectory_path, dtype=np.float64, ndmin=2)
    if raw.shape != (len(manifest.frames), 8) or not np.isfinite(raw).all():
        raise ValueError("DROID-W output must contain one finite TUM row per manifest frame")

    # DROID-W save_traj writes the ordinal in column zero.  Do not silently
    # associate or interpolate missing frames.
    ordinals = raw[:, 0].astype(np.int64)
    if not np.array_equal(raw[:, 0], ordinals) or not np.array_equal(
        ordinals, np.arange(len(manifest.frames), dtype=np.int64),
    ):
        raise ValueError("DROID-W output ordinals are incomplete or reordered")

    matrices = []
    for row in raw:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _rotation_from_xyzw(row[4:8])
        matrix[:3, 3] = row[1:4]
        matrices.append(validate_se3(matrix, "DROID-W raw pose"))
    origin_inverse = np.linalg.inv(matrices[0])
    matrices = [validate_se3(origin_inverse @ matrix) for matrix in matrices]
    steps = [float(np.linalg.norm((np.linalg.inv(a) @ b)[:3, 3]))
             for a, b in zip(matrices, matrices[1:])]
    if steps and max(steps) > maximum_step_translation_m:
        raise ValueError("DROID-W trajectory exceeds the metric motion gate")

    records = [PoseRecord(
        frame_id=frame.frame_id,
        timestamp_us=frame.timestamp_us,
        t_world_camera=matrix,
        source="DROID-W-shadow-raw",
    ) for frame, matrix in zip(manifest.frames, matrices)]
    metadata = {
        **provenance,
        "trajectory_sha256": sha256_file(trajectory_path),
        "provenance_sha256": sha256_file(provenance_path),
        "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
    }
    write_trajectory(
        output_path, records, sequence_id=manifest.sequence_id,
        arm="droid_w_shadow", metadata=metadata,
    )
    unsigned = {
        "schema": "droid_w_shadow_import_audit.v1",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(records),
        "complete_manifest_coverage": True,
        "maximum_step_translation_m": max(steps, default=0.0),
        "raw_trajectory_sha256": sha256_file(trajectory_path),
        "output_trajectory_sha256": sha256_file(output_path),
        "gt_consumed": False,
        "sim3_alignment_used": False,
        "identity_fallback_used": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")
    return unsigned
