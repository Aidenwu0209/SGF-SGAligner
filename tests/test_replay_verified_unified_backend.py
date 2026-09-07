from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pose_pipeline.bounded_backend import BoundedBackendConfig
from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, load_trajectory, sha256_file,
    stable_json_sha256, write_manifest, write_trajectory,
)
from pose_pipeline.pose_graph import CorrectionAuditConfig, PoseGraphOptimizationConfig
from pose_pipeline.runner import PrecommitGeometryConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_verified_unified_backend.py"
SPEC = importlib.util.spec_from_file_location("verified_backend_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


@pytest.fixture
def case(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))
    root = tmp_path
    baseline, frames = [], []
    for i in range(3):
        frame = 10 * i
        transform = np.eye(4)
        transform[0, 3] = .2 * i
        baseline.append(PoseRecord(frame, frame * 1000, transform, source="DPV"))
        color, depth = root / f"color_{frame}.png", root / f"depth_{frame}.png"
        color.write_bytes(b"synthetic sealed color" + bytes([i]))
        depth.write_bytes(b"synthetic sealed depth" + bytes([i]))
        frames.append(FrameRecord(frame, frame * 1000, color, depth, (1., 1., 0., 0.)))
    manifest = SequenceManifest("scannet", "fixture", root, 1000., tuple(frames), "recorded_rgbd")
    paths = {name: root / (name + ".json") for name in replay.legacy.INPUT_NAMES}
    write_manifest(paths["manifest"], manifest)
    write_trajectory(paths["trajectory"], baseline, sequence_id="fixture", arm="dpv")
    paths["baseline_cloud"].write_bytes(b"sealed baseline point cloud")
    for name in ("loop_evidence", "registration_inference", "visual_inference"):
        write(paths[name], {"gt_consumed": False})
    receipt = {"status": "completed", "gt_consumed": False, "identity_fallback_used": False,
               "manifest_sha256": sha256_file(paths["manifest"]),
               "trajectory_sha256": sha256_file(paths["trajectory"]),
               "cloud_sha256": sha256_file(paths["baseline_cloud"]),
               "integrated_frame_count": 3, "requested_frame_count": 3, "trajectory_pose_count": 3,
               "voxel_length_m": .02, "sdf_trunc_m": .08, "depth_trunc_m": 4.5}
    write(paths["baseline_refusion_receipt"], receipt)
    scene = {"scene_id": "fixture", **{name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()}}
    input_spec = root / "inputs.json"
    write(input_spec, {"schema": "unified_pose_replay_inputs.v1", "scenes": [scene]})
    bounded = asdict(BoundedBackendConfig(enabled=True))
    correction = asdict(CorrectionAuditConfig(maximum_absolute_correction_translation_m=.25,
                                               maximum_absolute_correction_rotation_deg=5.))
    geometry = asdict(PrecommitGeometryConfig(enabled=True, frame_stride=1,
                                             require_scene_improvement=True))
    config = {"bounded_config": bounded, "correction_config": correction,
              "pose_graph_config": asdict(PoseGraphOptimizationConfig()),
              "precommit_geometry_config": geometry}
    binding = {"schema": "pose_pipeline_input_binding.v1", "gt_consumed": False,
               "bounded_config": bounded, "correction_config": correction,
               "precommit_geometry_config": geometry,
               "manifest_sha256": sha256_file(paths["manifest"]),
               "trajectory_sha256": sha256_file(paths["trajectory"]),
               "admitted_frame_sha256": stable_json_sha256([0, 10, 20])}
    summary = {"schema": "unified_pose_result.v1", "gt_consumed": False, "identity_fallback_used": False,
               "sequence_id": "fixture", "source_trajectory_sha256": sha256_file(paths["trajectory"]),
               "config": config, "config_sha256": stable_json_sha256(config), "frame_count": 3,
               "integrated_frame_count": 3, "accepted_loop_count": 1}
    transform = np.eye(4)
    transform[0, 3] = -.3
    registration = {"accepted": True, "transform": transform.tolist(),
                    "decision": {"usable_for_reconstruction": True, "fallback_used": False},
                    "edge_confidence": .8, "information_matrix": np.eye(6).tolist()}
    pixels = {direction + "_" + kind: sha256_file(path)
              for direction, frame in (("source", frames[0]), ("target", frames[-1]))
              for kind, path in (("color", frame.color_path), ("depth", frame.depth_path))}
    witness = {"accepted": True, "source_frame_id": 0, "target_frame_id": 20,
               "registration_transform": transform.tolist(), "rgbd_inputs_sha256": pixels,
               "gt_consumed": False}
    evidence = {"schema": "loop_evidence.v1", "sequence_id": "fixture", "gt_consumed": False,
                "bounded_config": bounded,
                "anchors": [{"anchor_ordinal": i, "anchor_frame_id": 10 * i} for i in range(3)],
                "evidence": [{"source_anchor_index": 0, "target_anchor_index": 2,
                              "source_frame_id": 0, "target_frame_id": 20, "edge_verified": True,
                              "registration": registration, "visual_verification": witness,
                              "visual_estimate": deepcopy(witness)}]}
    loop = {"source": 0, "target": 2, "kind": "robust_submap_loop", "weight": 1.,
            "confidence": .8, "provenance": "synthetic_independent_verification",
            "information_matrix": np.eye(6).tolist(), "T_target_source_m": transform.tolist()}
    graph = {"success": True, "optimizer_success": True, "gt_consumed": False,
             "fallback_used": False, "accepted_loop_edge_count": 1, "edges": [loop]}
    bounded_report = {"gt_consumed": False, "fallback_used": False, "config": bounded,
                      "pose_graph": graph, "loop_selection": {
                          "retained_edges": [{"source": 0, "target": 2, "weight": 1.}],
                          "weights": [{"source": 0, "target": 2, "effective_weight": 1.}]}}
    data = dict(zip(replay.REQUIRED_SOURCE, [summary, binding, evidence]))
    data.update({"pose_graph_result.json": graph, "bounded_backend.json": bounded_report})
    source_root = root / "fresh_fixture"
    def flush():
        source_root.mkdir(exist_ok=True)
        for name in replay.REQUIRED_SOURCE + replay.OPTIONAL_SOURCE:
            if name in data:
                write(source_root / name, data[name])
            elif (source_root / name).exists():
                (source_root / name).unlink()
    flush()
    return SimpleNamespace(root=root, data=data, manifest=manifest, baseline=baseline, scene=scene,
                           paths=paths, source=source_root, flush=flush, inputs=input_spec)


def test_extract_uses_only_verified_actual_pgo_edges_and_effective_weight(case):
    ordinals, edges = replay.extract_verified_edges(case.data, case.manifest, case.baseline)
    assert ordinals == [0, 1, 2]
    assert len(edges) == 1 and edges[0].weight == 1. and edges[0].confidence == .8
    assert edges[0].source_to_target[0, 3] == -.3
    assert edges[0].provenance == "synthetic_independent_verification"
    row = deepcopy(case.data["loop_evidence.json"]["evidence"][0])
    row.update(source_anchor_index=0, target_anchor_index=1, target_frame_id=10, edge_verified=False)
    case.data["loop_evidence.json"]["evidence"].append(row)
    assert len(replay.extract_verified_edges(case.data, case.manifest, case.baseline)[1]) == 1


@pytest.mark.parametrize("field", ["edge_verified", "visual_verification", "registration"])
def test_unverified_loop_cannot_reenter_graph(case, field):
    row = case.data["loop_evidence.json"]["evidence"][0]
    if field == "edge_verified":
        row[field] = False
    else:
        row[field]["accepted"] = False
    with pytest.raises(ValueError, match="not independently verified"):
        replay.extract_verified_edges(case.data, case.manifest, case.baseline)


@pytest.mark.parametrize("change", ["transform", "weight", "confidence", "pixels", "anchor"])
def test_source_link_tampering_fails_closed(case, change):
    row = case.data["loop_evidence.json"]["evidence"][0]
    if change == "transform":
        row["registration"]["transform"][0][3] += .01
    elif change == "weight":
        case.data["bounded_backend.json"]["loop_selection"]["weights"][0]["effective_weight"] = 1.5
    elif change == "confidence":
        row["registration"]["edge_confidence"] = .7
    elif change == "pixels":
        case.manifest.frames[0].depth_path.write_bytes(b"changed source pixels")
    else:
        case.data["loop_evidence.json"]["anchors"][1]["anchor_frame_id"] = 11
    with pytest.raises(ValueError):
        replay.extract_verified_edges(case.data, case.manifest, case.baseline)


def test_only_scaling_policy_may_change(case):
    original, target, *_ = replay.source_settings(
        case.data["unified_result.json"], case.data["input_binding.json"],
        case.data["loop_evidence.json"], {"bounded": {"correction_scaling_policy": "smooth_local"}})
    assert original.correction_scaling_policy == "global"
    assert target.correction_scaling_policy == "smooth_local"
    for settings in ({"bounded": {"maximum_loop_weight": 3.}},
                     {"correction": {"maximum_absolute_correction_translation_m": 1.}},
                     {"optimization": {"robustifier": "adaptive_gnc"}}):
        with pytest.raises(ValueError, match="only correction scaling may change"):
            replay.source_settings(case.data["unified_result.json"], case.data["input_binding.json"],
                                   case.data["loop_evidence.json"], settings)


def test_source_hash_gt_fallback_and_frame_binding_are_checked(case):
    hashes = {name: sha256_file(case.source / name) for name in case.data}
    replay.load_source(case.source, case.scene, case.paths, case.manifest, case.baseline,
                       {"source_sha256": hashes})
    bad = dict(hashes, **{"loop_evidence.json": "0" * 64})
    with pytest.raises(ValueError, match="frozen source file hashes"):
        replay.load_source(case.source, case.scene, case.paths, case.manifest, case.baseline,
                           {"source_sha256": bad})
    for key, value in (("gt_consumed", True), ("identity_fallback_used", True)):
        original = case.data["unified_result.json"][key]
        case.data["unified_result.json"][key] = value
        case.flush()
        with pytest.raises(ValueError):
            replay.load_source(case.source, case.scene, case.paths, case.manifest, case.baseline, {})
        case.data["unified_result.json"][key] = original
    case.data["input_binding.json"]["admitted_frame_sha256"] = stable_json_sha256([20, 10, 0])
    case.flush()
    with pytest.raises(ValueError, match="frame identity/order"):
        replay.load_source(case.source, case.scene, case.paths, case.manifest, case.baseline, {})


def test_no_loop_source_copies_dpv_bytes_and_creates_no_candidate(case):
    case.data["loop_evidence.json"]["evidence"][0]["edge_verified"] = False
    case.data["unified_result.json"]["accepted_loop_count"] = 0
    for name in replay.OPTIONAL_SOURCE:
        del case.data[name]
    case.flush()
    output = case.root / "replay"
    args = SimpleNamespace(inputs=case.inputs, scene="fixture", config=None,
                           run_root=case.source, output=output)
    with patch.object(replay, "optimize_bounded_trajectory", side_effect=AssertionError("must not optimize")):
        replay.infer(args)
    summary = json.loads((output / "summary.json").read_text())
    row = summary["scenes"][0]
    assert row["arms"] == {}
    assert row["no_candidate"]["candidate_generated"] is False
    committed = Path(row["no_candidate"]["committed_trajectory"])
    assert committed.read_bytes() == case.paths["trajectory"].read_bytes()
    assert not list(output.rglob("candidate_trajectory.json"))
    with pytest.raises(FileExistsError):
        replay.infer(args)


def test_pipeline_preserves_same_edges_and_rolls_back_when_geometry_does_not_improve(case):
    output = case.root / "replay"
    args = SimpleNamespace(inputs=case.inputs, scene="fixture", config=None,
                           run_root=case.source, output=output)
    def fuse(request):
        request.output_dir.mkdir(parents=True)
        cloud = request.output_dir / "refused.ply"
        cloud.write_bytes(b"synthetic candidate cloud")
        return {"status": "completed", "gt_consumed": False, "identity_fallback_used": False,
                "manifest_sha256": sha256_file(request.manifest), "trajectory_sha256": sha256_file(request.trajectory),
                "cloud": str(cloud), "cloud_sha256": sha256_file(cloud), "integrated_frame_count": 3,
                "requested_frame_count": 3, "trajectory_pose_count": 3,
                "voxel_length_m": .02, "sdf_trunc_m": .08, "depth_trunc_m": 4.5}
    with patch("reconstruction.rgbd_refusion.run_full_rgbd_refusion", side_effect=fuse), \
            patch("pose_pipeline.geometry_metrics.ply_geometry_metrics", return_value={}), \
            patch("pose_pipeline.geometry_metrics.compare_no_gt_geometry_v2", return_value={
                "passes_scene_safety": True, "passes_scene_improvement": False}):
        replay.infer(args)
    summary = json.loads((output / "summary.json").read_text())
    result = summary["scenes"][0]["arms"][replay.ARM]
    assert result["correction_applied"] and not result["committed_candidate"]
    assert result["dpv_rollback_byte_identical"]
    arm = output / "fixture" / replay.ARM
    assert (arm / "committed_trajectory.json").read_bytes() == case.paths["trajectory"].read_bytes()
    assert [p.frame_id for p in load_trajectory(arm / "candidate_trajectory.json")[0]] == [0, 10, 20]
    assert result["frozen_verified_edge_count"] == result["applied_loop_count"] == 1
