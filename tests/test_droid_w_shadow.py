from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_pipeline.contracts import FrameRecord, SequenceManifest, load_trajectory, write_manifest
from pose_pipeline.droid_w_shadow import import_droid_w_shadow


def _fixture(tmp_path: Path, *, gt_consumed: bool = False):
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    for index in range(3):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index * 1000, color, depth, (500., 500., 2., 2.)))
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, SequenceManifest("orbbec", "shadow", tmp_path, 1000., tuple(frames), "test"))
    trajectory = tmp_path / "est_poses_full.txt"
    np.savetxt(trajectory, np.asarray([
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, .1, 0, 0, 0, 0, 0, 1],
        [2, .2, 0, 0, 0, 0, 0, 1],
    ], dtype=float))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "provider_commit": "a" * 40,
        "tracking_checkpoint_sha256": "b" * 64,
        "metric_depth_provider": "Metric3D-v2",
        "metric_depth_checkpoint_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "gt_consumed": gt_consumed,
        "sim3_alignment_used": False,
        "trajectory_stage": "raw_full_trajectory_before_evaluation",
    }))
    return manifest, trajectory, provenance


def test_imports_complete_raw_metric_trajectory(tmp_path: Path):
    manifest, trajectory, provenance = _fixture(tmp_path)
    report = import_droid_w_shadow(
        manifest_path=manifest, trajectory_path=trajectory,
        provenance_path=provenance, output_path=tmp_path / "out.json",
        audit_path=tmp_path / "audit.json",
    )
    rows, payload = load_trajectory(tmp_path / "out.json")
    assert report["frame_count"] == len(rows) == 3
    assert payload["metadata"]["metric_depth_provider"] == "Metric3D-v2"


def test_rejects_gt_provenance(tmp_path: Path):
    manifest, trajectory, provenance = _fixture(tmp_path, gt_consumed=True)
    with pytest.raises(ValueError, match="consumed GT"):
        import_droid_w_shadow(
            manifest_path=manifest, trajectory_path=trajectory,
            provenance_path=provenance, output_path=tmp_path / "out.json",
            audit_path=tmp_path / "audit.json",
        )


def test_rejects_missing_frame(tmp_path: Path):
    manifest, trajectory, provenance = _fixture(tmp_path)
    values = np.loadtxt(trajectory)[:2]
    np.savetxt(trajectory, values)
    with pytest.raises(ValueError, match="one finite TUM row"):
        import_droid_w_shadow(
            manifest_path=manifest, trajectory_path=trajectory,
            provenance_path=provenance, output_path=tmp_path / "out.json",
            audit_path=tmp_path / "audit.json",
        )
