"""Fail-closed import of Generalized-CVO consecutive RGB-D refinements.

GCVO reports ``T_source_target`` with the convention
``p_source = T_source_target @ p_target``.  For consecutive camera frames this
is exactly ``inv(T_world_source) @ T_world_target``.  This adapter never reads
GT and never fills a missing transform with identity.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import (
    PoseRecord,
    load_manifest,
    load_trajectory,
    sha256_file,
    stable_json_sha256,
    validate_se3,
    write_trajectory,
)
from .robust_backend import transform_distance


GCVO_RESULT_SCHEMA = "gcvo_relative_refinement.v1"


@dataclass(frozen=True)
class GCVORefinementConfig:
    maximum_baseline_disagreement_translation_m: float = 0.08
    maximum_baseline_disagreement_rotation_deg: float = 8.0
    maximum_step_translation_m: float = 0.20
    maximum_step_rotation_deg: float = 20.0
    minimum_accepted_fraction: float = 0.80

    def validate(self) -> "GCVORefinementConfig":
        values = asdict(self)
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("GCVO refinement thresholds must be finite")
        if any(value <= 0 for key, value in values.items() if key != "minimum_accepted_fraction"):
            raise ValueError("GCVO refinement motion thresholds must be positive")
        if not 0.0 < self.minimum_accepted_fraction <= 1.0:
            raise ValueError("minimum accepted fraction must be in (0,1]")
        return self


def _load_gcvo_rows(path: Path) -> tuple[list[dict], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != GCVO_RESULT_SCHEMA:
        raise ValueError("GCVO result schema mismatch")
    if payload.get("gt_consumed") is not False:
        raise ValueError("GCVO result must declare gt_consumed=false")
    if payload.get("matrix_convention") != "p_source=T_source_target@p_target":
        raise ValueError("GCVO matrix convention mismatch")
    forbidden = {key.lower() for key in payload if "gt" in key.lower() or "ground_truth" in key.lower()}
    forbidden.discard("gt_consumed")
    if forbidden:
        raise ValueError(f"GCVO result exposes forbidden evaluation fields: {sorted(forbidden)}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("GCVO result rows must be a non-empty list")
    return rows, payload


def apply_gcvo_refinement(
    *,
    manifest_path: Path,
    baseline_trajectory_path: Path,
    gcvo_result_path: Path,
    output_trajectory_path: Path,
    output_audit_path: Path,
    config: GCVORefinementConfig = GCVORefinementConfig(),
) -> dict:
    """Gate consecutive GCVO deltas and write a complete metric trajectory."""
    config.validate()
    manifest = load_manifest(manifest_path)
    baseline, baseline_payload = load_trajectory(baseline_trajectory_path)
    baseline_by_id = {row.frame_id: row for row in baseline}
    ordered = [baseline_by_id[frame.frame_id] for frame in manifest.frames if frame.frame_id in baseline_by_id]
    if len(ordered) != len(manifest.frames):
        raise ValueError("baseline trajectory must cover every manifest frame")
    rows, source_payload = _load_gcvo_rows(gcvo_result_path)
    expected_pairs = [
        (left.frame_id, right.frame_id) for left, right in zip(ordered, ordered[1:])
    ]
    by_pair: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in rows:
        pair = (int(row["source_frame_id"]), int(row["target_frame_id"]))
        if pair in by_pair:
            raise ValueError(f"duplicate GCVO pair: {pair}")
        by_pair[pair] = row
    if set(by_pair) != set(expected_pairs):
        missing = sorted(set(expected_pairs) - set(by_pair))
        extra = sorted(set(by_pair) - set(expected_pairs))
        raise ValueError(f"GCVO consecutive coverage mismatch missing={missing[:5]} extra={extra[:5]}")

    candidate = [ordered[0].t_world_camera.copy()]
    accepted, audit_rows = 0, []
    for ordinal, pair in enumerate(expected_pairs):
        source, target = ordered[ordinal], ordered[ordinal + 1]
        baseline_delta = np.linalg.inv(source.t_world_camera) @ target.t_world_camera
        row = by_pair[pair]
        gcvo_delta = validate_se3(row["T_source_target"], f"GCVO pair {pair}")
        disagreement_rotation, disagreement_translation = transform_distance(
            baseline_delta, gcvo_delta,
        )
        step_rotation, step_translation = transform_distance(np.eye(4), gcvo_delta)
        reasons = []
        if disagreement_translation > config.maximum_baseline_disagreement_translation_m:
            reasons.append("baseline_translation_disagreement")
        if disagreement_rotation > config.maximum_baseline_disagreement_rotation_deg:
            reasons.append("baseline_rotation_disagreement")
        if step_translation > config.maximum_step_translation_m:
            reasons.append("step_translation_gate")
        if step_rotation > config.maximum_step_rotation_deg:
            reasons.append("step_rotation_gate")
        use_gcvo = not reasons
        selected_delta = gcvo_delta if use_gcvo else baseline_delta
        candidate.append(validate_se3(candidate[-1] @ selected_delta, f"candidate frame {target.frame_id}"))
        accepted += int(use_gcvo)
        audit_rows.append({
            "source_frame_id": pair[0],
            "target_frame_id": pair[1],
            "accepted": use_gcvo,
            "reasons": reasons,
            "baseline_disagreement_translation_m": disagreement_translation,
            "baseline_disagreement_rotation_deg": disagreement_rotation,
            "selected": "gcvo" if use_gcvo else "baseline_fail_closed",
        })
    accepted_fraction = accepted / len(expected_pairs)
    promoted = accepted_fraction >= config.minimum_accepted_fraction
    if not promoted:
        candidate = [row.t_world_camera.copy() for row in ordered]
    records = [PoseRecord(
        frame_id=row.frame_id,
        timestamp_us=row.timestamp_us,
        t_world_camera=pose,
        source="GCVO-refined-DPV" if promoted else "DPV-retained-GCVO-gate-failed",
    ) for row, pose in zip(ordered, candidate)]
    write_trajectory(
        output_trajectory_path,
        records,
        sequence_id=manifest.sequence_id,
        arm="gcvo_rgbd_refinement",
        metadata={
            "parent_trajectory_sha256": baseline_payload["payload_sha256"],
            "gcvo_result_sha256": sha256_file(gcvo_result_path),
            "backend_pose_graph_unchanged": True,
            "complete_frame_coverage": True,
            "promotion_gate_passed": promoted,
        },
    )
    unsigned = {
        "schema": "gcvo_refinement_audit.v1",
        "sequence_id": manifest.sequence_id,
        "config": asdict(config),
        "pair_count": len(expected_pairs),
        "accepted_pair_count": accepted,
        "accepted_fraction": accepted_fraction,
        "promotion_gate_passed": promoted,
        "fail_closed_action": None if promoted else "retain_complete_baseline_trajectory",
        "source_metadata": {
            "gcvo_commit": source_payload.get("gcvo_commit"),
            "gcvo_config_sha256": source_payload.get("gcvo_config_sha256"),
        },
        "rows": audit_rows,
        "gt_consumed": False,
        "identity_fallback_used": False,
    }
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output_audit_path.open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return unsigned

