from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, load_trajectory,
    write_manifest, write_trajectory,
)
from pose_pipeline.model_adapters import (
    adapt_abot_trajectory, adapt_mapanything_revision,
    adapt_slamformer_revision, estimate_metric_scale,
    import_abot_loop_proposals, stitch_mapanything_windows,
)


def _pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


def _fixture(tmp_path: Path, count: int = 4) -> tuple[Path, Path]:
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    for index in range(count):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(
            index, 1_000_000 + index * 1_000, color, depth,
            (500.0, 500.0, 320.0, 240.0),
        ))
    manifest = SequenceManifest(
        "orbbec", "journal", tmp_path, 1000.0, tuple(frames), "test",
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    trajectory_path = tmp_path / "baseline.json"
    write_trajectory(
        trajectory_path,
        [PoseRecord(row.frame_id, row.timestamp_us, _pose(float(index)), source="DPV") for index, row in enumerate(frames)],
        sequence_id="journal", arm="baseline",
    )
    return manifest_path, trajectory_path


def test_robust_metric_scale_rejects_outlier():
    predicted = np.ones(100)
    sensor = np.full(100, 2.0)
    sensor[-1] = 200.0
    scale, report = estimate_metric_scale(predicted, sensor)
    assert scale == pytest.approx(2.0)
    assert report["valid_sample_count"] >= 64


def test_abot_noloop_requires_full_frame_and_separate_mode(tmp_path: Path):
    manifest, _ = _fixture(tmp_path)
    poses = np.stack([_pose(index * 0.5) for index in range(4)])
    poses_path = tmp_path / "camera_poses_noloop.npy"
    np.save(poses_path, poses)
    output = tmp_path / "abot.json"
    adapt_abot_trajectory(
        manifest_path=manifest, poses_path=poses_path, output_path=output,
        mode="noloop", metric_scale=2.0,
        scale_evidence={"method": "test", "scale_m_per_model_unit": 2.0},
        model_commit="a" * 40, checkpoint_sha256="b" * 64,
    )
    rows, payload = load_trajectory(output)
    assert len(rows) == 4
    assert rows[-1].t_world_camera[0, 3] == pytest.approx(3.0)
    assert payload["metadata"]["official_loop_mode"] is False
    with pytest.raises(ValueError, match="camera_poses_loop"):
        adapt_abot_trajectory(
            manifest_path=manifest, poses_path=poses_path,
            output_path=tmp_path / "bad.json", mode="official_loop",
            metric_scale=1.0, scale_evidence={"method": "test"},
            model_commit="a", checkpoint_sha256="b" * 64,
        )


def test_abot_loop_edges_stay_pending_registration_decision(tmp_path: Path):
    manifest, _ = _fixture(tmp_path)
    edges = [{
        "src_frame": 0, "dst_frame": 3, "score": 0.9,
        "method": "salad_online_abot_reinfer",
        "transform_ji": _pose(-1.5).tolist(),
    }]
    source = tmp_path / "loop_edges.json"
    source.write_text(json.dumps(edges))
    output = tmp_path / "proposals.json"
    payload = import_abot_loop_proposals(
        manifest_path=manifest, loop_edges_path=source,
        output_path=output, metric_scale=2.0,
    )
    assert payload["may_bypass_registration_decision"] is False
    assert payload["proposals"][0]["decision_status"] == "pending_registration_decision"
    assert payload["proposals"][0]["T_target_source_m"][3] == pytest.approx(-3.0)


def test_slamformer_anchors_correct_complete_dpv_without_filling(tmp_path: Path):
    manifest, baseline = _fixture(tmp_path)
    tum = tmp_path / "final_traj.txt"
    tum.write_text(
        "0 0 0 0 0 0 0 1\n"
        "3 2.7 0 0 0 0 0 1\n"
    )
    output = tmp_path / "slamformer.json"
    result = adapt_slamformer_revision(
        manifest_path=manifest, baseline_trajectory_path=baseline,
        final_traj_path=tum, output_path=output, metric_scale=1.0,
        scale_evidence={"method": "test"}, identifier_mode="frame_id",
        model_variant="V1.1-long@224", model_commit="c" * 40,
        checkpoint_sha256="d" * 64,
    )
    rows, payload = load_trajectory(output)
    assert result["anchors"] == 2 and len(rows) == 4
    assert rows[-1].t_world_camera[0, 3] == pytest.approx(2.7)
    assert payload["metadata"]["anchor_frame_ids"] == [0, 3]


def test_mapanything_overlap_is_fail_closed_and_propagates(tmp_path: Path):
    manifest, baseline = _fixture(tmp_path)
    first = tmp_path / "window0.npz"
    second = tmp_path / "window1.npz"
    np.savez(first, frame_ids=np.array([0, 1, 2]), camera_poses=np.stack([_pose(0), _pose(1), _pose(2)]))
    np.savez(second, frame_ids=np.array([1, 2, 3]), camera_poses=np.stack([_pose(1), _pose(2), _pose(2.8)]))
    output = tmp_path / "mapanything.json"
    result = adapt_mapanything_revision(
        manifest_path=manifest, baseline_trajectory_path=baseline,
        window_paths=[first, second], output_path=output,
        metric_scale=1.0, input_mode="conditioned_on_dpv_pose",
        window_size=8, model_commit="e" * 40, checkpoint_sha256="f" * 64,
    )
    rows, _ = load_trajectory(output)
    assert result["accepted_windows"] == 2
    assert len(rows) == 4
    assert rows[-1].t_world_camera[0, 3] == pytest.approx(2.8)

    bad = ([1, 2, 3], [_pose(1), _pose(3), _pose(4)])
    _, report = stitch_mapanything_windows([
        ([0, 1, 2], [_pose(0), _pose(1), _pose(2)]), bad,
    ])
    assert report["rejected_windows"][0]["reason"] == "overlap_se3_inconsistent"
