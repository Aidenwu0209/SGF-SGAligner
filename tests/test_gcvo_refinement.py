from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, load_trajectory,
    write_manifest, write_trajectory,
)
from pose_pipeline.gcvo_refinement import GCVORefinementConfig, apply_gcvo_refinement


def _pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames, poses = [], []
    for index in range(4):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index, color, depth, (500.0, 500.0, 1.0, 1.0)))
        poses.append(PoseRecord(index, index, _pose(index * 0.10), source="DPV"))
    manifest = tmp_path / "manifest.json"
    trajectory = tmp_path / "baseline.json"
    write_manifest(manifest, SequenceManifest("orbbec", "demo", tmp_path, 1000.0, tuple(frames), "test"))
    write_trajectory(trajectory, poses, sequence_id="demo", arm="baseline")
    return manifest, trajectory


def _gcvo(path: Path, steps: list[float]) -> None:
    rows = []
    for index, step in enumerate(steps):
        rows.append({
            "source_frame_id": index,
            "target_frame_id": index + 1,
            "T_source_target": _pose(step).tolist(),
        })
    path.write_text(json.dumps({
        "schema": "gcvo_relative_refinement.v1",
        "matrix_convention": "p_source=T_source_target@p_target",
        "gcvo_commit": "a" * 40,
        "gcvo_config_sha256": "b" * 64,
        "gt_consumed": False,
        "rows": rows,
    }))


def test_gcvo_refinement_keeps_complete_metric_trajectory(tmp_path: Path):
    manifest, baseline = _fixture(tmp_path)
    source = tmp_path / "gcvo.json"
    _gcvo(source, [0.09, 0.09, 0.09])
    result = apply_gcvo_refinement(
        manifest_path=manifest,
        baseline_trajectory_path=baseline,
        gcvo_result_path=source,
        output_trajectory_path=tmp_path / "candidate.json",
        output_audit_path=tmp_path / "audit.json",
    )
    rows, payload = load_trajectory(tmp_path / "candidate.json")
    assert result["promotion_gate_passed"] is True
    assert len(rows) == 4
    assert rows[-1].t_world_camera[0, 3] == 0.27
    assert payload["identity_fallback_used"] is False


def test_gcvo_under_supported_result_retains_baseline(tmp_path: Path):
    manifest, baseline = _fixture(tmp_path)
    source = tmp_path / "gcvo.json"
    _gcvo(source, [0.09, 0.50, 0.50])
    result = apply_gcvo_refinement(
        manifest_path=manifest,
        baseline_trajectory_path=baseline,
        gcvo_result_path=source,
        output_trajectory_path=tmp_path / "candidate.json",
        output_audit_path=tmp_path / "audit.json",
        config=GCVORefinementConfig(minimum_accepted_fraction=0.80),
    )
    rows, _ = load_trajectory(tmp_path / "candidate.json")
    assert result["promotion_gate_passed"] is False
    assert np.isclose(rows[-1].t_world_camera[0, 3], 0.30)
