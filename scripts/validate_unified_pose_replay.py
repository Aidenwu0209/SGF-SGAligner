#!/usr/bin/env python3
"""Create-only replay of sealed RGB-D PnP edges, with separate GT evaluation.

Run with PYTHONPATH=src. ``infer --inputs inputs.json --output NEW_DIR``
defaults to ``combined_bounded``. Optional arms are ``original_pnp``,
``weight_only``, ``correction_only``, and ``combined_loo``. The last arm really
runs strict leave-one-out checks; combined_bounded does not claim that check.

The input spec has schema unified_pose_replay_inputs.v1 and a ``scenes`` list.
Each scene has scene_id and seven {path, sha256} entries: manifest, trajectory,
loop_evidence, registration_inference, visual_inference,
baseline_refusion_receipt, baseline_cloud. Paths may be absolute or relative
to the input spec. Freeze their actual digests before starting the experiment.
The manifest MUST contain exactly the admitted baseline frames in order.

``evaluate --run-root DIR --references references.json`` is a separate process.
The reference spec has a scenes list with scene_id, scene_root, optional mesh.
Evaluation never rewrites inference decisions or committed trajectories.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from pose_pipeline.contracts import (
    bind_manifest_trajectory, load_manifest, load_trajectory, sha256_file,
    stable_json_sha256, validate_se3, write_trajectory,
)
from pose_pipeline.pose_graph import (
    CorrectionAuditConfig, LoopWeightConfig, PoseGraphEdge,
    PoseGraphOptimizationConfig, audit_corrected_trajectory, loop_edge_weight,
)

ARMS = ("original_pnp", "weight_only", "correction_only",
        "combined_bounded", "combined_loo")
REGISTRATION_ARM = "registration_v3_symmetric_fallback"
INPUT_NAMES = ("manifest", "trajectory", "loop_evidence",
               "registration_inference", "visual_inference",
               "baseline_refusion_receipt", "baseline_cloud")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_file, target.open("xb") as target_file:
        shutil.copyfileobj(source_file, target_file)
    if sha256_file(source) != sha256_file(target):
        raise ValueError("exact-copy digest mismatch")


def assert_gt_free(value: object, location: str = "input") -> None:
    """Reject declared GT consumption, including nested registration arms."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"gt_consumed", "gt_at_inference", "used_gt"} and child is not False:
                raise ValueError(f"GT consumption in {location}.{key}")
            if key in {"gt_transform", "ground_truth_transform", "gt_pose",
                       "ground_truth_pose", "reference_trajectory"}:
                raise ValueError(f"GT payload in {location}.{key}")
            assert_gt_free(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_gt_free(child, f"{location}[{index}]")


def resolve_bound_inputs(scene: dict, base: Path) -> dict[str, Path]:
    paths = {}
    for name in INPUT_NAMES:
        entry = scene[name]
        expected = entry["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"missing frozen SHA256 for {name}")
        path = Path(entry["path"])
        path = (base / path).resolve() if not path.is_absolute() else path.resolve()
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input digest mismatch: {name}")
        paths[name] = path
    return paths


def validate_exact_coverage(manifest, trajectory) -> None:
    if [row.frame_id for row in manifest.frames] != [row.frame_id for row in trajectory]:
        raise ValueError("manifest must contain exactly the admitted trajectory frames in order")
    bind_manifest_trajectory(manifest, trajectory, allow_manifest_superset=False)


def validate_baseline_receipt(receipt: dict, paths: dict, pose_count: int) -> dict:
    if receipt.get("status") != "completed" or receipt.get("gt_consumed") is not False:
        raise ValueError("baseline refusion is not completed and GT-free")
    if receipt.get("identity_fallback_used") is not False:
        raise ValueError("baseline refusion identity-fallback contract missing")
    for field, name in (("trajectory_sha256", "trajectory"),
                        ("manifest_sha256", "manifest"),
                        ("cloud_sha256", "baseline_cloud")):
        if receipt.get(field) != sha256_file(paths[name]):
            raise ValueError(f"baseline receipt binding mismatch: {field}")
    for field in ("integrated_frame_count", "requested_frame_count", "trajectory_pose_count"):
        if receipt.get(field) != pose_count:
            raise ValueError(f"baseline refusion incomplete: {field}")
    parameters = {key: float(receipt[key]) for key in
                  ("voxel_length_m", "sdf_trunc_m", "depth_trunc_m")}
    if not all(np.isfinite(v) and v > 0 for v in parameters.values()):
        raise ValueError("invalid baseline TSDF parameters")
    return parameters


def frozen_pnp_edges(registration: dict, visual: dict, evidence: dict,
                     baseline, registration_sha: str, scene_id: str, manifest=None):
    for name, value in (("registration", registration), ("visual", visual),
                        ("loop_evidence", evidence)):
        assert_gt_free(value, name)
        if value.get("gt_consumed") is not False:
            raise ValueError(f"{name} must declare gt_consumed=false")
        if value.get("sequence_id") != scene_id:
            raise ValueError(f"{name} sequence mismatch")
    if visual.get("registration_inference_sha256") != registration_sha:
        raise ValueError("visual-to-registration digest mismatch")
    if visual.get("registration_arm") != REGISTRATION_ARM:
        raise ValueError("visual registration arm mismatch")
    anchors = evidence["anchors"]
    ordinals = [int(row["anchor_ordinal"]) for row in anchors]
    frames = [int(row["anchor_frame_id"]) for row in anchors]
    if len(ordinals) < 2 or ordinals != sorted(set(ordinals)):
        raise ValueError("anchor ordinals must be unique and increasing")
    if any(index < 0 or index >= len(baseline) for index in ordinals):
        raise ValueError("anchor ordinal outside baseline")
    if [baseline[index].frame_id for index in ordinals] != frames:
        raise ValueError("anchor-to-baseline frame mismatch")
    indices = [int(row["registration_row_index"]) for row in visual["rows"]
               if row["accepted"].get("recall") is True]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate visual selection")
    visual_rows = {int(row["registration_row_index"]): row for row in visual["rows"]
                   if row["accepted"].get("recall") is True}
    rgbd_frames = {} if manifest is None else {row.frame_id: row for row in manifest.frames}
    edges = []
    for index in sorted(indices):
        if not 0 <= index < len(registration["rows"]):
            raise ValueError("visual selection outside registration rows")
        row = registration["rows"][index]
        result = row["arms"][REGISTRATION_ARM]
        if result.get("accepted") is not True:
            raise ValueError("visual selection outside fallback accepts")
        source, target = int(row["source_anchor_index"]), int(row["target_anchor_index"])
        if not (0 <= source < len(anchors) and 0 <= target < len(anchors)):
            raise ValueError("registration anchor outside evidence")
        if (frames[source], frames[target]) != (int(row["source_frame_id"]), int(row["target_frame_id"])):
            raise ValueError("registration-to-anchor frame mismatch")
        selected = visual_rows[index]
        if (int(selected["source_frame_id"]), int(selected["target_frame_id"])) != (frames[source], frames[target]):
            raise ValueError("visual-to-registration frame mismatch")
        if not np.allclose(validate_se3(selected["registration_transform"]),
                           validate_se3(result["transform"]), rtol=0.0, atol=1e-8):
            raise ValueError("visual-to-registration transform mismatch")
        if manifest is not None:
            for name, frame_id in (("source", frames[source]), ("target", frames[target])):
                frame = rgbd_frames[frame_id]
                for kind, path in (("color", frame.color_path), ("depth", frame.depth_path)):
                    if selected["rgbd_inputs_sha256"][f"{name}_{kind}"] != sha256_file(path):
                        raise ValueError("visual RGB-D input digest mismatch")
        edges.append(PoseGraphEdge(
            source=source, target=target,
            source_to_target=validate_se3(result["transform"]),
            kind="diagnostic_filtered_loop",
            weight=loop_edge_weight(
                float(result["forward"]["verification"]["minimum_overlap"]),
                source, target, len(anchors), LoopWeightConfig()),
            provenance=f"sealed_pnp_registration_row_{index}",
            information=np.asarray(result.get("information_matrix", np.eye(6)), dtype=float),
            confidence=float(result.get("edge_confidence", 1.0)),
        ))
    return ordinals, sorted(indices), edges


def arm_configuration(arm: str, settings: dict):
    from pose_pipeline.bounded_backend import BoundedBackendConfig
    if arm not in ARMS:
        raise ValueError(f"unknown replay arm: {arm}")
    bounded_settings = {"enabled": True, **settings.get("bounded", {})}
    if "correction_backtracking_scales" in bounded_settings:
        bounded_settings["correction_backtracking_scales"] = tuple(
            bounded_settings["correction_backtracking_scales"])
    bounded = BoundedBackendConfig(**bounded_settings)
    correction = CorrectionAuditConfig(**{
        "maximum_absolute_correction_translation_m": 0.25,
        "maximum_absolute_correction_rotation_deg": 5.0,
        **settings.get("correction", {}),
    })
    optimizer = PoseGraphOptimizationConfig(**settings.get("optimization", {}))
    if arm in {"original_pnp", "weight_only"}:
        correction = replace(correction,
            maximum_absolute_correction_translation_m=None,
            maximum_absolute_correction_rotation_deg=None)
        bounded = replace(bounded, correction_backtracking_scales=(1.0,))
    if arm in {"original_pnp", "correction_only"}:
        bounded = replace(bounded, maximum_loop_weight=None,
            high_leverage_min_span_fraction=None)
    bounded = replace(bounded, enabled=arm != "original_pnp",
                      enforce_leave_one_out=arm == "combined_loo")
    return bounded, correction, optimizer


def infer_scene(scene: dict, spec_base: Path, output: Path, arms: list[str],
                settings: dict) -> dict:
    from pose_pipeline.bounded_backend import optimize_bounded_trajectory
    from pose_pipeline.geometry_metrics import compare_no_gt_geometry_v2, ply_geometry_metrics
    from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion
    started = time.perf_counter()
    scene_id = scene["scene_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id):
        raise ValueError("invalid scene id")
    paths = resolve_bound_inputs(scene, spec_base)
    manifest = load_manifest(paths["manifest"])
    baseline, payload = load_trajectory(paths["trajectory"])
    assert_gt_free(payload, "baseline trajectory")
    if manifest.sequence_id != scene_id or payload["sequence_id"] != scene_id:
        raise ValueError("manifest/trajectory sequence mismatch")
    validate_exact_coverage(manifest, baseline)
    receipt = json.loads(paths["baseline_refusion_receipt"].read_text())
    parameters = validate_baseline_receipt(receipt, paths, len(baseline))
    ordinals, indices, edges = frozen_pnp_edges(
        json.loads(paths["registration_inference"].read_text()),
        json.loads(paths["visual_inference"].read_text()),
        json.loads(paths["loop_evidence"].read_text()), baseline,
        sha256_file(paths["registration_inference"]), scene_id, manifest)
    destination = output / scene_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / "tracked_manifest.json"
    baseline_path = destination / "baseline" / "trajectory.json"
    baseline_cloud = destination / "baseline" / "refused.ply"
    copy_exact(paths["manifest"], manifest_path)
    copy_exact(paths["trajectory"], baseline_path)
    copy_exact(paths["baseline_cloud"], baseline_cloud)
    write_json(destination / "baseline" / "reuse_receipt.json", {
        "schema": "sealed_refusion_reuse_receipt.v1", "source": scene,
        "verified_receipt": receipt, "parameters": parameters,
        "exact_admitted_frame_count": len(baseline), "gt_consumed": False,
    })
    baseline_metrics = ply_geometry_metrics(baseline_cloud)
    write_json(destination / "baseline" / "geometry.json", baseline_metrics)
    results = {}
    for arm in arms:
        arm_started = time.perf_counter()
        print(json.dumps({"scene": scene_id, "arm": arm, "stage": "pgo",
                          "selected_pnp_edges": len(edges)}), flush=True)
        bounded, correction, optimizer = arm_configuration(arm, settings)
        corrected, report = optimize_bounded_trajectory(
            baseline, ordinals, edges, config=bounded,
            optimization_config=optimizer, correction_config=correction)
        validate_exact_coverage(manifest, corrected)
        arm_dir = destination / arm
        arm_dir.mkdir()
        candidate = arm_dir / "candidate_trajectory.json"
        write_trajectory(candidate, corrected, sequence_id=scene_id, arm=arm, metadata={
            "diagnostic_only": True, "gt_consumed": False,
            "baseline_sha256": sha256_file(baseline_path),
            "registration_inference_sha256": sha256_file(paths["registration_inference"]),
            "visual_inference_sha256": sha256_file(paths["visual_inference"]),
            "selected_registration_rows": indices,
        })
        write_json(arm_dir / "bounded_backend.json", report)
        print(json.dumps({"scene": scene_id, "arm": arm, "stage": "full_refusion",
                          "optimizer_accepted": report["success"]}), flush=True)
        refusion = run_full_rgbd_refusion(FullRefusionRequest(
            manifest=manifest_path, trajectory=candidate,
            output_dir=arm_dir / "candidate_refusion", **parameters))
        if refusion["integrated_frame_count"] != len(baseline):
            raise ValueError("candidate refusion did not integrate every admitted frame")
        candidate_metrics = ply_geometry_metrics(Path(refusion["cloud"]))
        geometry = compare_no_gt_geometry_v2(
            baseline_metrics, candidate_metrics,
            admitted_frame_sha256=stable_json_sha256([row.frame_id for row in baseline]))
        write_json(arm_dir / "candidate_geometry.json", candidate_metrics)
        write_json(arm_dir / "geometry_comparison.json", geometry)
        # The historical arm may produce a diagnostic even when the current
        # strict Guard would reject it. Apply the same commitment gate to all arms.
        strict_guard_config = arm_configuration("combined_bounded", settings)[1]
        strict_guard = audit_corrected_trajectory(baseline, corrected, strict_guard_config)
        write_json(arm_dir / "strict_correction_guard.json", strict_guard)
        changed = any(not np.allclose(before.t_world_camera, after.t_world_camera,
                                     rtol=0.0, atol=1e-12)
                      for before, after in zip(baseline, corrected))
        applied = (report.get("no_op") is not True
                   and report.get("applied_loop_count", 0) > 0 and changed)
        accept = (applied and report["success"] is True and strict_guard["passes"] is True
                  and geometry["passes_scene_safety"] is True
                  and geometry["passes_scene_improvement"] is True)
        committed = arm_dir / "committed_trajectory.json"
        copy_exact(candidate if accept else baseline_path, committed)
        rollback_exact = not accept and sha256_file(committed) == sha256_file(baseline_path)
        result = {
            "arm": arm, "diagnostic_only": True, "promotion_eligible": False,
            "gt_consumed": False, "complete_frame_coverage": True,
            "identity_fallback_used": False, "pose_count": len(baseline),
            "bounded_config": asdict(bounded), "correction_config": asdict(correction),
            "optimization_config": asdict(optimizer), "backend_success": report["success"],
            "correction_applied": applied, "applied_loop_count": report.get("applied_loop_count", 0),
            "strict_correction_guard": strict_guard,
            "selected_correction_scale": report.get("selected_correction_scale"),
            "correction_scaling_policy": bounded.correction_scaling_policy,
            "selected_anchor_correction_scales": report.get("selected_anchor_correction_scales"),
            "strict_leave_one_out_enabled": bounded.enforce_leave_one_out,
            "candidate_trajectory_sha256": sha256_file(candidate),
            "committed_trajectory_sha256": sha256_file(committed),
            "committed_candidate": accept, "dpv_rollback_byte_identical": rollback_exact,
            "no_gt_geometry": geometry, "candidate_refusion": refusion,
            "selected_registration_rows": indices,
            "runtime_s": time.perf_counter() - arm_started,
        }
        write_json(arm_dir / "result.json", result)
        results[arm] = result
        print(json.dumps({"scene": scene_id, "arm": arm, "stage": "completed",
                          "committed_candidate": accept, "runtime_s": result["runtime_s"]}), flush=True)
    result = {"scene_id": scene_id, "inputs": scene, "arms": results,
              "runtime_s": time.perf_counter() - started, "gt_consumed": False}
    write_json(destination / "result.json", result)
    return result


def infer(args) -> None:
    spec = json.loads(args.inputs.read_text())
    if spec.get("schema") != "unified_pose_replay_inputs.v1":
        raise ValueError("unsupported input spec schema")
    settings = {} if args.config is None else json.loads(args.config.read_text())
    assert_gt_free(settings, "config")
    scenes = spec["scenes"]
    if len({row["scene_id"] for row in scenes}) != len(scenes):
        raise ValueError("duplicate scene ids")
    if args.scenes:
        selected = set(args.scenes)
        if not selected <= {row["scene_id"] for row in scenes}:
            raise ValueError("requested scene absent from input spec")
        scenes = [row for row in scenes if row["scene_id"] in selected]
    if not scenes or len(args.arms) != len(set(args.arms)):
        raise ValueError("need scenes and unique arms")
    args.output.mkdir(parents=True, exist_ok=False)
    copy_exact(args.inputs, args.output / "inputs.json")
    write_json(args.output / "configuration.json", settings)
    results = [infer_scene(row, args.inputs.resolve().parent, args.output,
                           args.arms, settings) for row in scenes]
    write_json(args.output / "summary.json", {
        "schema": "unified_pose_replay.v1", "diagnostic_only": True,
        "promotion_eligible": False, "gt_consumed": False,
        "input_spec_sha256": sha256_file(args.inputs), "arms": args.arms,
        "scenes": results, "registration_gate_recomputed": False,
        "decision": "EXPERIMENT_ONLY_REGISTRATION_PROMOTION_UNPROVEN",
    })


def evaluate(args) -> None:
    # Evaluation imports and GT files are reachable only in this phase.
    from pose_pipeline.evaluation import (
        build_scannet_common_observed_surface, reconstruction_surface_metrics,
        scannet_reference_trajectory, trajectory_metrics,
    )
    summary_path = args.run_root / "summary.json"
    summary = json.loads(summary_path.read_text())
    refs = json.loads(args.references.read_text())
    by_scene = {row["scene_id"]: row for row in refs["scenes"]}
    output = args.run_root / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    all_results = {}
    for row in summary["scenes"]:
        scene_id = row["scene_id"]
        reference = by_scene[scene_id]
        root = Path(reference["scene_root"])
        if not root.is_absolute():
            root = args.references.resolve().parent / root
        directory = args.run_root / scene_id
        baseline_path = directory / "baseline" / "trajectory.json"
        for path, name in ((baseline_path, "trajectory"),
                           (directory / "baseline" / "refused.ply", "baseline_cloud"),
                           (directory / "tracked_manifest.json", "manifest")):
            if sha256_file(path) != row["inputs"][name]["sha256"]:
                raise ValueError("sealed baseline input changed before evaluation")
        baseline, _ = load_trajectory(baseline_path)
        timestamps = {pose.frame_id: pose.timestamp_us for pose in baseline}
        truth = scannet_reference_trajectory(root, list(timestamps), timestamps)
        mesh = Path(reference["mesh"]) if reference.get("mesh") else None
        if mesh is not None and not mesh.is_absolute():
            mesh = args.references.resolve().parent / mesh
        scene_results = {}
        for arm, result in row["arms"].items():
            arm_dir = directory / arm
            candidate = arm_dir / "candidate_trajectory.json"
            committed = arm_dir / "committed_trajectory.json"
            cloud = arm_dir / "candidate_refusion" / "refused.ply"
            for path, digest in ((candidate, result["candidate_trajectory_sha256"]),
                                 (committed, result["committed_trajectory_sha256"]),
                                 (cloud, result["candidate_refusion"]["cloud_sha256"])):
                if sha256_file(path) != digest:
                    raise ValueError("sealed inference output changed before evaluation")
            estimates = {"baseline": baseline, "candidate": load_trajectory(candidate)[0],
                         "committed": load_trajectory(committed)[0]}
            metrics = {name: trajectory_metrics(value, truth) for name, value in estimates.items()}
            evaluated = {"trajectory": metrics, "mesh_status": "missing_reference_mesh",
                         "committed_candidate": result["committed_candidate"],
                         "gt_role": "evaluation_only"}
            destination = output / scene_id / arm
            destination.mkdir(parents=True)
            if mesh is not None:
                if not mesh.is_file():
                    raise FileNotFoundError(mesh)
                common = build_scannet_common_observed_surface(
                    root, directory / "tracked_manifest.json", baseline_path,
                    candidate, mesh, destination / "common_observed_surface.ply")
                write_json(destination / "common_observed_surface.json", common)
                geometry = {}
                for name, trajectory_path, point_cloud in (
                    ("baseline", baseline_path, directory / "baseline" / "refused.ply"),
                    ("candidate", candidate, cloud),
                ):
                    poses = load_trajectory(trajectory_path)[0]
                    alignment = None
                    for pose in poses:
                        try:
                            world = validate_se3(np.loadtxt(root / "pose" / f"{pose.frame_id}.txt"))
                        except ValueError:
                            continue
                        alignment = validate_se3(world @ np.linalg.inv(pose.t_world_camera))
                        break
                    if alignment is None:
                        raise ValueError("no finite GT alignment pose")
                    geometry[name] = {
                        "primary_common_observed_region": reconstruction_surface_metrics(
                            point_cloud, Path(common["surface"]), alignment),
                        "secondary_full_mesh": reconstruction_surface_metrics(point_cloud, mesh, alignment),
                    }
                geometry["committed"] = geometry["candidate" if result["committed_candidate"] else "baseline"]
                evaluated.update(mesh_status="completed", mesh_sha256=sha256_file(mesh), geometry=geometry)
            write_json(destination / "result.json", evaluated)
            scene_results[arm] = evaluated
        all_results[scene_id] = scene_results
    write_json(output / "summary.json", {
        "schema": "unified_pose_replay_evaluation.v1", "gt_role": "evaluation_only",
        "inference_summary_sha256": sha256_file(summary_path),
        "reference_spec_sha256": sha256_file(args.references), "scenes": all_results,
        "inference_decisions_unchanged": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    run = subparsers.add_parser("infer")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--arms", nargs="+", choices=ARMS, default=["combined_bounded"])
    run.add_argument("--scenes", nargs="+")
    run.set_defaults(handler=infer)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--run-root", type=Path, required=True)
    evaluation.add_argument("--references", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
