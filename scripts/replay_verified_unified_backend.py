#!/usr/bin/env python3
"""Replay exactly the admitted PGO loops of a sealed ``run-unified`` result.

PYTHONPATH=src python scripts/replay_verified_unified_backend.py infer \
    --run-root FRESH_SCENE_DIR --inputs frozen-inputs.json --scene SCENE \
    --config settings.json --output NEW_DIRECTORY

Settings may contain ``bounded: {correction_scaling_policy: smooth_local}``.
Every other supplied solver/Guard setting must equal the source run. Optional
``source_sha256`` maps source filenames to their already frozen SHA256 values;
when supplied, it must cover every consumed source file. Source files are also
copied and hashed before inference and checked again before the final summary.
No CLIP, PnP, registration, proposal generation or GT is run in this phase.

The existing validate_unified_pose_replay.py ``evaluate`` command consumes the
output. A source with no admitted loops has an empty scene ``arms`` mapping and
an explicit byte-identical DPV result; no candidate or improvement is invented.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import re
import time

import numpy as np

from pose_pipeline.bounded_backend import BoundedBackendConfig, optimize_bounded_trajectory
from pose_pipeline.contracts import (
    load_manifest, load_trajectory, sha256_file, stable_json_sha256,
    validate_se3, write_trajectory,
)
from pose_pipeline.pose_graph import (
    CorrectionAuditConfig, PoseGraphEdge, PoseGraphOptimizationConfig,
    audit_corrected_trajectory,
)

# Reuse the existing strict input, receipt, copy and posthoc-evaluation helpers.
_SPEC = importlib.util.spec_from_file_location(
    "_verified_replay_contracts", Path(__file__).with_name("validate_unified_pose_replay.py"),
)
legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(legacy)
ARM = "same_verified_edges_smooth_local"
REQUIRED_SOURCE = ("unified_result.json", "input_binding.json", "loop_evidence.json")
OPTIONAL_SOURCE = ("pose_graph_result.json", "bounded_backend.json")


def assert_no_fallback(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"fallback_used", "identity_fallback_used", "used_fallback"} and child is not False:
                raise ValueError("source declares fallback consumption")
            assert_no_fallback(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_fallback(child)


def bounded_config(raw: dict) -> BoundedBackendConfig:
    raw = dict(raw)
    if "correction_backtracking_scales" in raw:
        raw["correction_backtracking_scales"] = tuple(raw["correction_backtracking_scales"])
    return BoundedBackendConfig(**raw)


def same_value(left: object, right: object, label: str) -> None:
    if stable_json_sha256(left) != stable_json_sha256(right):
        raise ValueError(f"source binding mismatch: {label}")


def same_matrix(left: object, right: object, label: str) -> None:
    first, second = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError(f"invalid matrix binding: {label}")
    if not np.allclose(first, second, atol=1e-10, rtol=0.0):
        raise ValueError(f"matrix binding mismatch: {label}")


def source_settings(summary: dict, binding: dict, evidence: dict, settings: dict):
    unknown = set(settings) - {"bounded", "correction", "optimization", "source_sha256"}
    if unknown:
        raise ValueError(f"unsupported replay settings: {sorted(unknown)}")
    config = summary["config"]
    # run-unified records the original YAML byte hash here, not a hash of the
    # hydrated dataclasses below. The sealed summary binds those dataclasses;
    # do not incorrectly compare the two different serializations.
    stable_json_sha256(config)
    if not isinstance(summary.get("config_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", summary["config_sha256"]):
        raise ValueError("source unified configuration byte digest is missing")
    original = bounded_config(config["bounded_config"])
    if not original.enabled or original.enforce_leave_one_out:
        raise ValueError("same-edge replay requires enabled bounded backend without edge re-filtering")
    if original.correction_scaling_policy != "global":
        raise ValueError("source must use global correction scaling for this single-factor replay")
    same_value(asdict(original), asdict(bounded_config(binding["bounded_config"])), "bounded input")
    same_value(asdict(original), asdict(bounded_config(evidence["bounded_config"])), "bounded evidence")
    correction = CorrectionAuditConfig(**config["correction_config"])
    optimizer = PoseGraphOptimizationConfig(**config["pose_graph_config"])
    same_value(asdict(correction), binding["correction_config"], "correction input")
    if (correction.maximum_absolute_correction_translation_m is None
            or correction.maximum_absolute_correction_rotation_deg is None
            or correction.propagation != "legacy_slerp_linear"
            or optimizer.robustifier != "huber"):
        raise ValueError("source must preserve absolute Guard caps, legacy propagation and Huber")
    target = replace(original, correction_scaling_policy="smooth_local")
    for name, expected in (("bounded", asdict(target)), ("correction", asdict(correction)),
                           ("optimization", asdict(optimizer))):
        for key, value in settings.get(name, {}).items():
            if key not in expected:
                raise ValueError(f"unknown {name} setting: {key}")
            same_value(value, expected[key], f"only correction scaling may change: {name}.{key}")
    geometry = config["precommit_geometry_config"]
    same_value(geometry, binding["precommit_geometry_config"], "geometry input")
    if (geometry.get("enabled") is not True or geometry.get("frame_stride") != 1
            or geometry.get("require_scene_improvement") is not True):
        raise ValueError("source must require full-frame safety and improvement gates")
    return original, target, correction, optimizer, geometry


def load_source(root: Path, scene: dict, paths: dict, manifest, baseline, settings: dict):
    files = {name: root / name for name in REQUIRED_SOURCE}
    files.update({name: root / name for name in OPTIONAL_SOURCE if (root / name).is_file()})
    hashes = {name: sha256_file(path) for name, path in files.items()}
    expected = settings.get("source_sha256")
    if expected is not None:
        same_value(expected, hashes, "frozen source file hashes")
    data = {name: json.loads(path.read_text()) for name, path in files.items()}
    for name, value in data.items():
        legacy.assert_gt_free(value, name)
        assert_no_fallback(value)
        if value.get("gt_consumed") is not False:
            raise ValueError(f"source must explicitly declare no GT: {name}")
    summary, binding, evidence = (data[name] for name in REQUIRED_SOURCE)
    if (summary.get("schema") != "unified_pose_result.v1"
            or binding.get("schema") != "pose_pipeline_input_binding.v1"):
        raise ValueError("unsupported source schema")
    for value in (summary, evidence):
        if value.get("sequence_id") != scene["scene_id"]:
            raise ValueError("source sequence mismatch")
    if summary.get("identity_fallback_used") is not False:
        raise ValueError("source identity-fallback contract is missing")
    for name in ("manifest", "trajectory"):
        if binding.get(name + "_sha256") != sha256_file(paths[name]):
            raise ValueError(f"source frozen {name} mismatch")
    if summary.get("source_trajectory_sha256") != sha256_file(paths["trajectory"]):
        raise ValueError("source DPV trajectory mismatch")
    admitted = stable_json_sha256([row.frame_id for row in baseline])
    if binding.get("admitted_frame_sha256") != admitted:
        raise ValueError("source admitted frame identity/order mismatch")
    if summary.get("frame_count") != len(baseline) or summary.get("integrated_frame_count") != len(baseline):
        raise ValueError("source final refusion is incomplete")
    configs = source_settings(summary, binding, evidence, settings)
    ordinals, edges = extract_verified_edges(data, manifest, baseline)
    return data, files, hashes, configs, ordinals, edges


def extract_verified_edges(data: dict, manifest, baseline):
    evidence = data["loop_evidence.json"]
    anchors = evidence["anchors"]
    ordinals = [row["anchor_ordinal"] for row in anchors]
    if (len(ordinals) < 2 or any(type(index) is not int for index in ordinals)
            or ordinals != sorted(set(ordinals)) or ordinals[0] != 0
            or ordinals[-1] != len(baseline) - 1):
        raise ValueError("source anchor ordinals must bind complete baseline endpoints")
    frames = [baseline[index].frame_id for index in ordinals]
    if frames != [row["anchor_frame_id"] for row in anchors]:
        raise ValueError("source anchor frame binding mismatch")
    proposals = {}
    for row in evidence["evidence"]:
        pair = (row["source_anchor_index"], row["target_anchor_index"])
        if pair in proposals or any(type(i) is not int or not 0 <= i < len(anchors) for i in pair):
            raise ValueError("duplicate or invalid source proposal anchors")
        if (frames[pair[0]], frames[pair[1]]) != (row["source_frame_id"], row["target_frame_id"]):
            raise ValueError("source proposal frame binding mismatch")
        proposals[pair] = row
    graph, bounded = data.get("pose_graph_result.json"), data.get("bounded_backend.json")
    if graph is None or bounded is None:
        if graph is not None or bounded is not None or any(row.get("edge_verified") for row in proposals.values()):
            raise ValueError("verified source loops require sealed PGO and bounded reports")
        if data["unified_result.json"].get("accepted_loop_count") != 0:
            raise ValueError("missing source PGO despite reported loops")
        return ordinals, []
    same_value(graph, bounded["pose_graph"], "PGO report versus bounded report")
    same_value(asdict(bounded_config(bounded["config"])),
               asdict(bounded_config(evidence["bounded_config"])), "bounded report configuration")
    if graph.get("success") is not True or graph.get("optimizer_success") is not True:
        raise ValueError("source PGO did not complete successfully")
    retained = {(row["source"], row["target"]): row for row in bounded["loop_selection"]["retained_edges"]}
    weights = {(row["source"], row["target"]): row for row in bounded["loop_selection"]["weights"]}
    rgbd = {row.frame_id: row for row in manifest.frames}
    edges, seen = [], set()
    for row in graph["edges"]:
        if row["kind"] == "odometry":
            continue
        pair = (row["source"], row["target"])
        if pair in seen or pair not in proposals or pair not in retained or pair not in weights:
            raise ValueError("PGO loop missing unique verified retained evidence")
        seen.add(pair)
        proposal = proposals[pair]
        registration = proposal["registration"]
        visual = proposal.get("visual_verification") or {}
        estimate = proposal.get("visual_estimate") or {}
        if (proposal.get("edge_verified") is not True or registration.get("accepted") is not True
                or registration.get("decision", {}).get("usable_for_reconstruction") is not True
                or visual.get("accepted") is not True or estimate.get("accepted") is not True):
            raise ValueError("PGO loop was not independently verified")
        transform = validate_se3(row["T_target_source_m"])
        same_matrix(transform, registration["transform"], "PGO versus registration transform")
        same_matrix(transform, visual["registration_transform"], "visual registration transform")
        same_value(row["weight"], retained[pair]["weight"], "retained effective weight")
        same_value(row["weight"], weights[pair]["effective_weight"], "original effective weight")
        same_value(row["confidence"], registration["edge_confidence"], "loop confidence")
        same_matrix(row["information_matrix"], registration["information_matrix"], "loop information")
        for witness in (visual, estimate):
            if (witness.get("source_frame_id"), witness.get("target_frame_id")) != (frames[pair[0]], frames[pair[1]]):
                raise ValueError("visual witness frame binding mismatch")
            for direction, frame_id in (("source", frames[pair[0]]), ("target", frames[pair[1]])):
                for kind, path in (("color", rgbd[frame_id].color_path), ("depth", rgbd[frame_id].depth_path)):
                    if witness["rgbd_inputs_sha256"][direction + "_" + kind] != sha256_file(path):
                        raise ValueError("visual witness RGB-D pixels changed")
        edges.append(PoseGraphEdge(
            source=row["source"], target=row["target"], source_to_target=transform,
            kind=row["kind"], weight=float(row["weight"]), provenance=row["provenance"],
            information=np.asarray(row["information_matrix"], dtype=float),
            confidence=float(row["confidence"]),
        ))
    if len(edges) != graph["accepted_loop_edge_count"] or seen != set(retained):
        raise ValueError("source admitted loop count/set mismatch")
    return ordinals, edges


def edge_fingerprint(edge: PoseGraphEdge) -> dict:
    return {"source": edge.source, "target": edge.target, "kind": edge.kind,
            "provenance": edge.provenance, "weight": edge.weight,
            "transform": edge.source_to_target.tolist(),
            "information": None if edge.information is None else edge.information.tolist(),
            "confidence": edge.confidence}


def assert_replay_kept_edges(report: dict, edges: list[PoseGraphEdge]) -> None:
    selection = report["loop_selection"]
    if (selection["selected_count"] != len(edges) or selection["retained_count"] != len(edges)
            or selection["rejected"] or report["influence"]["rejected_edges"]):
        raise ValueError("same-edge replay unexpectedly changed the admitted edge set")
    expected = {(e.source, e.target): e for e in edges}
    for row in selection["weights"]:
        edge = expected[(row["source"], row["target"])]
        if row["input_weight"] != edge.weight or row["effective_weight"] != edge.weight:
            raise ValueError("same-edge replay unexpectedly changed effective weights")


def infer(args) -> None:
    started = time.perf_counter()
    spec = json.loads(args.inputs.read_text())
    if spec.get("schema") != "unified_pose_replay_inputs.v1":
        raise ValueError("unsupported frozen input schema")
    scenes = [row for row in spec["scenes"] if row["scene_id"] == args.scene]
    if len(scenes) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.scene):
        raise ValueError("need exactly one matching safe scene id")
    scene = scenes[0]
    settings = {} if args.config is None else json.loads(args.config.read_text())
    legacy.assert_gt_free(settings, "config")
    paths = legacy.resolve_bound_inputs(scene, args.inputs.resolve().parent)
    manifest = load_manifest(paths["manifest"])
    baseline, payload = load_trajectory(paths["trajectory"])
    legacy.assert_gt_free(payload, "baseline")
    assert_no_fallback(payload)
    if manifest.sequence_id != args.scene or payload["sequence_id"] != args.scene:
        raise ValueError("manifest or baseline sequence mismatch")
    legacy.validate_exact_coverage(manifest, baseline)
    receipt = json.loads(paths["baseline_refusion_receipt"].read_text())
    parameters = legacy.validate_baseline_receipt(receipt, paths, len(baseline))
    data, source_files, hashes, configs, ordinals, edges = load_source(
        args.run_root, scene, paths, manifest, baseline, settings)
    original, bounded, correction, optimizer, geometry_config = configs
    for name in parameters:
        if parameters[name] != geometry_config[name]:
            raise ValueError("source TSDF parameters differ from frozen baseline")
    args.output.mkdir(parents=True, exist_ok=False)
    legacy.copy_exact(args.inputs, args.output / "inputs.json")
    legacy.write_json(args.output / "configuration.json", settings)
    destination = args.output / args.scene
    destination.mkdir()
    for name, path in source_files.items():
        legacy.copy_exact(path, destination / "source_evidence" / name)
    seal = {"schema": "sealed_verified_unified_edges.v1", "gt_consumed": False,
            "source_root": str(args.run_root.resolve()), "source_sha256": hashes,
            "edges_already_sparsified": True, "edge_set_expansion_allowed": False,
            "effective_edge_weights_preserved": True,
            "source_yaml_byte_sha256": data["unified_result.json"]["config_sha256"],
            "source_hydrated_config_sha256": stable_json_sha256(data["unified_result.json"]["config"]),
            "original_bounded_config": asdict(original), "target_bounded_config": asdict(bounded),
            "anchor_ordinals": ordinals, "edges": [edge_fingerprint(e) for e in edges],
            "frozen_inputs": scene}
    legacy.write_json(destination / "source_binding.json", seal)
    manifest_path = destination / "tracked_manifest.json"
    baseline_path = destination / "baseline" / "trajectory.json"
    baseline_cloud = destination / "baseline" / "refused.ply"
    for source, target in ((paths["manifest"], manifest_path), (paths["trajectory"], baseline_path),
                            (paths["baseline_cloud"], baseline_cloud)):
        legacy.copy_exact(source, target)
    legacy.write_json(destination / "baseline" / "reuse_receipt.json", {
        "schema": "sealed_refusion_reuse_receipt.v1", "verified_receipt": receipt,
        "parameters": parameters, "exact_admitted_frame_count": len(baseline), "gt_consumed": False})
    results = {}
    noop = None
    if not edges:
        committed = destination / "committed_trajectory.json"
        legacy.copy_exact(baseline_path, committed)
        noop = {"reason": "source_has_no_verified_pgo_loops", "candidate_generated": False,
                "committed_trajectory": str(committed), "committed_trajectory_sha256": sha256_file(committed),
                "dpv_rollback_byte_identical": True, "final_cloud": str(baseline_cloud),
                "final_cloud_sha256": sha256_file(baseline_cloud), "complete_frame_coverage": True,
                "pose_count": len(baseline), "gt_consumed": False}
    else:
        results[ARM] = run_arm(destination, args.scene, baseline, ordinals, edges, bounded,
                               correction, optimizer, geometry_config, parameters)
    # Neither source evidence nor original baseline inputs may change during inference.
    same_value(hashes, {name: sha256_file(path) for name, path in source_files.items()}, "source changed during replay")
    legacy.resolve_bound_inputs(scene, args.inputs.resolve().parent)
    row = {"scene_id": args.scene, "inputs": scene, "arms": results, "no_candidate": noop,
           "source_binding_sha256": sha256_file(destination / "source_binding.json"),
           "gt_consumed": False, "runtime_s": time.perf_counter() - started}
    legacy.write_json(destination / "result.json", row)
    legacy.write_json(args.output / "summary.json", {
        "schema": "unified_pose_replay.v1", "diagnostic_only": True, "promotion_eligible": False,
        "gt_consumed": False, "input_spec_sha256": sha256_file(args.inputs), "arms": [ARM],
        "scenes": [row], "registration_gate_recomputed": False,
        "edge_set_expansion_allowed": False, "source_evidence_unchanged": True,
        "decision": "EXPERIMENT_ONLY_SAME_VERIFIED_EDGES_LOCAL_SCALING"})


def run_arm(destination, scene_id, baseline, ordinals, edges, bounded,
            correction, optimizer, geometry_config, parameters):
    from pose_pipeline.geometry_metrics import compare_no_gt_geometry_v2, ply_geometry_metrics
    from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion

    started = time.perf_counter()
    manifest_path = destination / "tracked_manifest.json"
    baseline_path, baseline_cloud = destination / "baseline" / "trajectory.json", destination / "baseline" / "refused.ply"
    arm_dir = destination / ARM
    arm_dir.mkdir()
    print(json.dumps({"scene": scene_id, "stage": "same_edges_pgo", "frozen_loop_count": len(edges)}), flush=True)
    corrected, report = optimize_bounded_trajectory(
        baseline, ordinals, edges, config=bounded, optimization_config=optimizer,
        correction_config=correction)
    assert_replay_kept_edges(report, edges)
    legacy.validate_exact_coverage(load_manifest(manifest_path), corrected)
    candidate = arm_dir / "candidate_trajectory.json"
    write_trajectory(candidate, corrected, sequence_id=scene_id, arm=ARM, metadata={
        "diagnostic_only": True, "gt_consumed": False, "baseline_sha256": sha256_file(baseline_path),
        "source_binding_sha256": sha256_file(destination / "source_binding.json"),
        "edges_already_sparsified": True, "edge_set_expansion_allowed": False})
    legacy.write_json(arm_dir / "bounded_backend.json", report)
    print(json.dumps({"scene": scene_id, "stage": "full_refusion", "pose_count": len(baseline)}), flush=True)
    refusion = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest_path, trajectory=candidate, output_dir=arm_dir / "candidate_refusion", **parameters))
    candidate_paths = {"manifest": manifest_path, "trajectory": candidate, "baseline_cloud": Path(refusion["cloud"])}
    legacy.validate_baseline_receipt(refusion, candidate_paths, len(baseline))
    baseline_metrics, candidate_metrics = ply_geometry_metrics(baseline_cloud), ply_geometry_metrics(Path(refusion["cloud"]))
    geometry = compare_no_gt_geometry_v2(
        baseline_metrics, candidate_metrics,
        admitted_frame_sha256=stable_json_sha256([row.frame_id for row in baseline]),
        **{key: value for key, value in geometry_config.items() if key in {
            "minimum_occupied_voxel_ratio", "minimum_each_robust_extent_ratio",
            "maximum_thickness_ratio", "maximum_layer_conflict_ratio",
            "maximum_matched_plane_tilt_regression_deg"}})
    guard = audit_corrected_trajectory(baseline, corrected, correction)
    changed = any(not np.allclose(a.t_world_camera, b.t_world_camera, rtol=0.0, atol=1e-12)
                  for a, b in zip(baseline, corrected))
    applied = changed and report.get("no_op") is not True and report.get("applied_loop_count", 0) > 0
    accepted = bool(applied and report["success"] and guard["passes"]
                    and geometry["passes_scene_safety"] and geometry["passes_scene_improvement"])
    committed = arm_dir / "committed_trajectory.json"
    legacy.copy_exact(candidate if accepted else baseline_path, committed)
    for name, value in (("candidate_geometry.json", candidate_metrics), ("baseline_geometry.json", baseline_metrics),
                        ("geometry_comparison.json", geometry), ("strict_correction_guard.json", guard)):
        legacy.write_json(arm_dir / name, value)
    result = {"arm": ARM, "diagnostic_only": True, "promotion_eligible": False, "gt_consumed": False,
              "complete_frame_coverage": True, "identity_fallback_used": False, "pose_count": len(baseline),
              "bounded_config": asdict(bounded), "correction_config": asdict(correction),
              "optimization_config": asdict(optimizer), "backend_success": report["success"],
              "correction_applied": applied, "applied_loop_count": report.get("applied_loop_count", 0),
              "strict_correction_guard": guard, "selected_correction_scale": report.get("selected_correction_scale"),
              "correction_scaling_policy": bounded.correction_scaling_policy,
              "selected_anchor_correction_scales": report.get("selected_anchor_correction_scales"),
              "candidate_trajectory_sha256": sha256_file(candidate), "committed_trajectory_sha256": sha256_file(committed),
              "committed_candidate": accepted,
              "dpv_rollback_byte_identical": not accepted and sha256_file(committed) == sha256_file(baseline_path),
              "no_gt_geometry": geometry, "candidate_refusion": refusion,
              "frozen_verified_edge_count": len(edges), "edges_already_sparsified": True,
              "effective_edge_weights_preserved": True, "runtime_s": time.perf_counter() - started}
    legacy.write_json(arm_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    run = phases.add_parser("infer")
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--scene", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.set_defaults(handler=infer)
    evaluation = phases.add_parser("evaluate")
    evaluation.add_argument("--run-root", type=Path, required=True)
    evaluation.add_argument("--references", type=Path, required=True)
    evaluation.set_defaults(handler=legacy.evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
