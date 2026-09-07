from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import patch

from pose_pipeline.bounded_backend import BoundedBackendConfig
from pose_pipeline.contracts import FrameRecord, PoseRecord, SequenceManifest, sha256_file, write_manifest, write_trajectory
from pose_pipeline.pose_graph import CorrectionAuditConfig
from pose_pipeline.runner import PrecommitGeometryConfig, run_sequence
from pose_pipeline.unified import load_unified_config
from pose_pipeline.visual_verification import VisualVerificationConfig


def fixture(root):
    frames, poses = [], []
    for index in range(2):
        color, depth = root / f"{index}.jpg", root / f"{index}.png"
        color.write_bytes(b"color")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index, color, depth, (500, 500, 10, 10)))
        matrix = np.eye(4)
        matrix[0, 3] = index
        poses.append(PoseRecord(index, index, matrix, source="DPV-SLAM"))
    manifest, trajectory = root / "manifest.json", root / "trajectory.json"
    write_manifest(manifest, SequenceManifest("scannet", "fixture", root, 1000, tuple(frames), "test"))
    write_trajectory(trajectory, poses, sequence_id="fixture", arm="baseline")
    return manifest, trajectory, poses


def config():
    return dict(
        visual_config=VisualVerificationConfig(enabled=True),
        bounded_config=BoundedBackendConfig(enabled=True),
        correction_config=CorrectionAuditConfig(
            maximum_absolute_correction_translation_m=.25,
            maximum_absolute_correction_rotation_deg=5,
        ),
        precommit_geometry_config=PrecommitGeometryConfig(
            enabled=True, require_scene_improvement=True, frame_stride=1,
        ),
    )


def fake_frontend(stack, *, visual_accepted=True):
    submap = SimpleNamespace(anchor_frame_id=0, source_frame_ids=(0,),
                              points=np.zeros((500, 3)), points_sha256="0" * 64)
    stack.enter_context(patch("pose_pipeline.runner.build_submap", return_value=submap))
    stack.enter_context(patch("pose_pipeline.runner.save_submap"))
    stack.enter_context(patch("pose_pipeline.runner.propose_loop_pairs", return_value=[{
        "source_anchor_index": 0, "target_anchor_index": 1,
    }]))
    stack.enter_context(patch("pose_pipeline.runner.register_submaps_bidirectional", return_value={
        "accepted": True, "transform": np.eye(4).tolist(),
        "forward": {"verification": {"minimum_overlap": .5}},
    }))
    stack.enter_context(patch("pose_pipeline.runner.estimate_rgbd_loop", return_value={
        "accepted": True, "schema": "rgbd_visual_loop_estimate.v1", "gt_consumed": False,
    }))
    return stack.enter_context(patch("pose_pipeline.runner.compare_visual_registration", return_value={
        "accepted": visual_accepted, "reason": "fixture", "gt_consumed": False,
    }))


def test_pnp_rejection_never_reaches_optimizer_and_rolls_back_exactly(tmp_path):
    manifest, trajectory, _ = fixture(tmp_path)
    with ExitStack() as stack:
        visual = fake_frontend(stack, visual_accepted=False)
        optimizer = stack.enter_context(patch("pose_pipeline.runner.optimize_bounded_trajectory"))
        result = run_sequence(arm="candidate", manifest_path=manifest,
                              trajectory_path=trajectory, output_dir=tmp_path / "out", **config())
    visual.assert_called_once()
    optimizer.assert_not_called()
    assert result["reason"] == "no_verified_loop"
    assert sha256_file(trajectory) == sha256_file(tmp_path / "out/trajectory.json")
    evidence = json.loads((tmp_path / "out/loop_evidence.json").read_text())
    assert evidence["evidence"][0]["edge_verified"] is False


def test_bounded_solver_failure_is_byte_rollback(tmp_path):
    manifest, trajectory, _ = fixture(tmp_path)
    with ExitStack() as stack:
        fake_frontend(stack)
        stack.enter_context(patch("pose_pipeline.runner.optimize_bounded_trajectory",
                                  side_effect=RuntimeError("solver unavailable")))
        result = run_sequence(arm="candidate", manifest_path=manifest,
                              trajectory_path=trajectory, output_dir=tmp_path / "out", **config())
    assert result["reason"] == "pose_graph_exception_fail_closed"
    assert result["fail_closed_byte_identical_to_source"]
    assert sha256_file(trajectory) == sha256_file(tmp_path / "out/trajectory.json")


def test_all_loops_removed_does_not_claim_applied_correction(tmp_path):
    manifest, trajectory, poses = fixture(tmp_path)
    with ExitStack() as stack:
        fake_frontend(stack)
        stack.enter_context(patch("pose_pipeline.runner.optimize_bounded_trajectory", return_value=(
            poses, {"success": True, "pose_graph": {}, "no_op": True},
        )))
        result = run_sequence(arm="candidate", manifest_path=manifest,
                              trajectory_path=trajectory, output_dir=tmp_path / "out", **config())
    assert result["accepted"] is False
    assert result["reason"] == "all_loops_rejected_by_influence"
    assert result["backend_correction_applied"] is False
    assert sha256_file(trajectory) == sha256_file(tmp_path / "out/trajectory.json")


@pytest.mark.parametrize("improves", [False, True])
def test_full_frame_geometry_decides_committed_trajectory(tmp_path, improves):
    manifest, trajectory, poses = fixture(tmp_path)
    correction = np.eye(4)
    correction[1, 3] = .01
    corrected = [replace(pose, t_world_camera=correction @ pose.t_world_camera) for pose in poses]
    calls = []

    def refuse(request):
        calls.append(request)
        return {"cloud": str(request.output_dir / "refused.ply")}

    with ExitStack() as stack:
        fake_frontend(stack)
        stack.enter_context(patch("pose_pipeline.runner.optimize_bounded_trajectory", return_value=(
            corrected, {"success": True, "pose_graph": {"success": True, "accepted_loop_edge_count": 1}},
        )))
        stack.enter_context(patch("reconstruction.rgbd_refusion.run_full_rgbd_refusion", side_effect=refuse))
        stack.enter_context(patch("pose_pipeline.geometry_metrics.ply_geometry_metrics", return_value={}))
        stack.enter_context(patch("pose_pipeline.geometry_metrics.compare_no_gt_geometry_v2", return_value={
            "passes_scene_safety": True, "passes_scene_improvement": improves,
        }))
        result = run_sequence(arm="candidate", manifest_path=manifest,
                              trajectory_path=trajectory, output_dir=tmp_path / "out", **config())
    assert len(calls) == 2
    assert calls[0].fused_frame_ids == calls[1].fused_frame_ids == (0, 1)
    assert result["accepted"] is improves
    committed = tmp_path / "out/trajectory.json"
    source = tmp_path / "out/trajectory.proposed.json" if improves else trajectory
    assert sha256_file(committed) == sha256_file(source)


@pytest.mark.parametrize("weakening", ["sparse", "no_improvement", "no_absolute", "no_pnp"])
def test_unified_contract_cannot_accidentally_skip_guard(tmp_path, weakening):
    options = config()
    if weakening == "sparse":
        options["precommit_geometry_config"] = replace(options["precommit_geometry_config"], frame_stride=8)
    if weakening == "no_improvement":
        options["precommit_geometry_config"] = replace(options["precommit_geometry_config"], require_scene_improvement=False)
    if weakening == "no_absolute":
        options["correction_config"] = CorrectionAuditConfig()
    if weakening == "no_pnp":
        options["visual_config"] = VisualVerificationConfig()
    with pytest.raises(ValueError, match="unified backend requires"):
        run_sequence(arm="candidate", manifest_path=tmp_path / "absent",
                     trajectory_path=tmp_path / "absent", output_dir=tmp_path / "out", **options)
    assert not (tmp_path / "out").exists()


def test_shipped_config_is_consumed_with_guard_and_frozen_scales(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/pose/unified_backend.yaml"
    options = load_unified_config(path, clip_download_root=tmp_path / "clip")
    assert options["visual_config"].enabled
    assert options["bounded_config"].correction_backtracking_scales == (1, .5, .25, .125, .0625)
    assert options["precommit_geometry_config"].frame_stride == 1
    assert options["correction_config"].maximum_absolute_correction_translation_m == .25


def test_disabling_both_modules_cannot_bypass_unified_profile(tmp_path):
    import yaml
    path = Path(__file__).resolve().parents[1] / "configs/pose/unified_backend.yaml"
    value = yaml.safe_load(path.read_text())
    value["visual"]["enabled"] = value["bounded"]["enabled"] = False
    changed = tmp_path / "config.yaml"
    changed.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="cannot disable"):
        load_unified_config(changed, clip_download_root=tmp_path / "clip")
