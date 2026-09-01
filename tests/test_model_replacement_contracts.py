from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, write_manifest, write_trajectory,
)
from pose_pipeline.model_contracts import (
    load_model_runtime_report, load_trajectory_revision,
    write_external_artifact_manifest, write_model_runtime_report,
    write_trajectory_revision,
)
from pose_pipeline.model_validation import (
    audit_split_isolation, fine_tune_eligibility, promotion_eligibility,
    write_model_comparison,
)


def _setup(tmp_path: Path):
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    for index in range(3):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index, color, depth, (1.0, 1.0, 0.0, 0.0)))
    manifest = SequenceManifest("orbbec", "seq", tmp_path, 1000.0, tuple(frames), "test")
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    parent = tmp_path / "parent.json"
    revised = tmp_path / "revised.json"
    parent_rows, revised_rows = [], []
    for index in range(3):
        old, new = np.eye(4), np.eye(4)
        old[0, 3] = index
        new[0, 3] = index * 0.9
        parent_rows.append(PoseRecord(index, index, old))
        revised_rows.append(PoseRecord(index, index, new))
    write_trajectory(parent, parent_rows, sequence_id="seq", arm="baseline")
    write_trajectory(revised, revised_rows, sequence_id="seq", arm="candidate")
    return manifest_path, parent, revised


def test_runtime_and_delayed_revision_contracts(tmp_path: Path):
    manifest, parent, revised = _setup(tmp_path)
    runtime_path = tmp_path / "runtime.json"
    write_model_runtime_report(
        runtime_path, manifest_path=manifest, model="ABot-Recon",
        model_commit="a" * 40, checkpoint_path=None,
        checkpoint_sha256="b" * 64, resolution=(640, 480),
        latency_ms=[10, 20, 30], peak_gpu_memory_mb=1024,
        output_pose_count=3, wall_time_s=0.1,
    )
    runtime = load_model_runtime_report(runtime_path)
    assert runtime["latency_ms"]["p95"] == pytest.approx(29.0)
    revision_path = tmp_path / "revision.json"
    write_trajectory_revision(
        revision_path, parent_trajectory_path=parent,
        revised_trajectory_path=revised, source="test",
        affected_frame_ids=[1, 2], runtime_report_path=runtime_path,
    )
    revision = load_trajectory_revision(revision_path)
    assert revision["map_update_policy"] == "delayed_full_refusion"
    assert revision["map_may_switch_before_full_refusion"] is False
    assert revision["correction_audit"]["maximum_translation_m"] == pytest.approx(0.2)


def test_fixanything_is_61_frame_presentation_only(tmp_path: Path):
    manifest, _, _ = _setup(tmp_path)
    video = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    with pytest.raises(ValueError, match="61 frames"):
        write_external_artifact_manifest(
            tmp_path / "bad.json", manifest_path=manifest,
            system="FixAnything", role="presentation_only",
            artifacts=[video], source_frame_ids=[0, 1, 2],
        )


def test_training_and_promotion_gates(tmp_path: Path):
    audit = audit_split_isolation(
        training_ids=["train", "held"], held_out_scannet_ids=["held"],
        validation_3rscan_ids=[], orbbec_ids=[],
    )
    assert not audit["passed"] and audit["leaked_ids"] == ["held"]
    rows = [{
        "sequence_id": f"dev{i}", "coverage": 0.9,
        "scale_jump": False, "catastrophic_trajectory": False,
        "primary_metric_improved": i < 3,
    } for i in range(5)]
    assert fine_tune_eligibility(rows)["eligible"]
    promotion = {
        "full_refusion_complete": True, "pose_coverage_loss": 0.0,
        "catastrophic_edges": 0, "unevaluable_accepted_edges": 0,
        "scannet_metric_pose_improvement": 0.11,
        "scannet_geometry_improvement": 0.12,
        "scannet_joint_ci_lower": 0.01,
        "orbbec_improved_sequences": 4, "orbbec_total_sequences": 5,
        "orbbec_max_deterioration": 0.05,
    }
    assert promotion_eligibility(promotion)["eligible_for_opt_in_online_integration"]
    output = tmp_path / "comparison.json"
    result = write_model_comparison(output, candidates=[{
        "model": "candidate", "role": "continuous_pose_frontend",
        "quality_score": 1.0, "promotion_summary": promotion,
    }], frozen_baseline_commit="f3d1adb")
    assert result["winner"] is None
    assert result["role_rankings"]["continuous_pose_frontend"] == ["candidate"]
