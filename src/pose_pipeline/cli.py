"""Unified CLI for manifest, replay, baseline/candidate, and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .adapters import orbbec_manifest, scan3r_manifest, scannet_manifest
from .contracts import (
    bind_manifest_trajectory, load_dpv_response_jsonl, load_legacy_tcw_mm,
    load_manifest, write_manifest, write_trajectory,
)
from .evaluation import evaluate_trajectory_files
from .model_adapters import (
    adapt_abot_trajectory,
    adapt_mapanything_revision,
    adapt_slamformer_revision,
    build_abot_scale_evidence,
    import_abot_loop_proposals,
    load_scale_evidence,
)
from .model_contracts import (
    write_external_artifact_manifest,
    write_model_runtime_report,
    write_trajectory_revision,
)
from .model_validation import write_model_comparison
from .pose_graph import LoopWeightConfig
from .replay import replay_manifest
from .runner import run_sequence
from .submaps import LoopProposalConfig


def _manifest(args: argparse.Namespace) -> None:
    if args.dataset == "scannet":
        value = scannet_manifest(args.input)
    elif args.dataset == "3rscan":
        value = scan3r_manifest(args.input, rotate_ccw=not args.no_rotate_ccw)
    else:
        value = orbbec_manifest(args.input)
    if args.frame_id:
        requested = set(args.frame_id)
        available = {frame.frame_id for frame in value.frames}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"requested frame IDs are unavailable: {missing}")
        value = replace(
            value,
            frames=tuple(frame for frame in value.frames if frame.frame_id in requested),
        ).validate()
    write_manifest(args.output, value)
    print(json.dumps({"manifest": str(args.output.resolve()), "frames": len(value.frames)}))


def _replay(args: argparse.Namespace) -> None:
    print(json.dumps(replay_manifest(
        manifest_path=args.manifest,
        socket_path=args.socket,
        output_dir=args.output,
        timeout_s=args.timeout,
    ), indent=2))


def _run(args: argparse.Namespace) -> None:
    print(json.dumps(run_sequence(
        arm=args.arm,
        manifest_path=args.manifest,
        trajectory_path=args.trajectory,
        output_dir=args.output,
        proposal_config=LoopProposalConfig(maximum_pairs=args.maximum_loop_pairs),
        loop_weight_config=LoopWeightConfig(
            high_leverage_min_span_fraction=(
                args.high_leverage_loop_min_span_fraction
            ),
            high_leverage_weight_cap=args.high_leverage_loop_weight_cap,
        ),
    ), indent=2))


def _import_trajectory(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    allowed = {frame.frame_id for frame in manifest.frames}
    if args.format == "legacy_tcw_mm":
        records = load_legacy_tcw_mm(
            args.input, allowed_frame_ids=allowed, source=args.source,
        )
    else:
        records = load_dpv_response_jsonl(
            args.input, allowed_frame_ids=allowed, source=args.source,
        )
    bind_manifest_trajectory(manifest, records)
    write_trajectory(
        args.output, records, sequence_id=manifest.sequence_id, arm="baseline",
        metadata={
            "import_format": args.format,
            "filtered_to_manifest": True,
        },
    )
    print(json.dumps({
        "trajectory": str(args.output.resolve()), "poses": len(records),
    }))


def _evaluate(args: argparse.Namespace) -> None:
    print(json.dumps(evaluate_trajectory_files(
        args.estimate, args.reference, args.output,
    ), indent=2))


def _refuse(args: argparse.Namespace) -> None:
    from reconstruction.rgbd_refusion import (
        FullRefusionRequest, run_full_rgbd_refusion,
    )

    frame_ids = None
    if args.fused_frame_ids is not None:
        payload = json.loads(args.fused_frame_ids.read_text())
        if isinstance(payload, dict):
            payload = payload.get("frame_ids")
        if not isinstance(payload, list):
            raise ValueError("fused frame ids must be a JSON list")
        frame_ids = tuple(int(value) for value in payload)
    print(json.dumps(run_full_rgbd_refusion(FullRefusionRequest(
        manifest=args.manifest,
        trajectory=args.trajectory,
        output_dir=args.output,
        fused_frame_ids=frame_ids,
        voxel_length_m=args.voxel_length,
        sdf_trunc_m=args.sdf_trunc,
        depth_trunc_m=args.depth_trunc,
    )), indent=2))


def _metric_scale(args: argparse.Namespace) -> tuple[float, dict]:
    if args.scale_evidence is not None:
        return load_scale_evidence(args.scale_evidence)
    if args.metric_scale is None or args.scale_method is None:
        raise ValueError("provide --scale-evidence or both --metric-scale and --scale-method")
    return float(args.metric_scale), {
        "method": args.scale_method,
        "scale_m_per_model_unit": float(args.metric_scale),
        "evidence_sha256": None,
    }


def _adapt_abot(args: argparse.Namespace) -> None:
    scale, evidence = _metric_scale(args)
    print(json.dumps(adapt_abot_trajectory(
        manifest_path=args.manifest, poses_path=args.poses, output_path=args.output,
        mode=args.mode, metric_scale=scale, scale_evidence=evidence,
        model_commit=args.model_commit, checkpoint_sha256=args.checkpoint_sha256,
    ), indent=2))


def _abot_scale_evidence(args: argparse.Namespace) -> None:
    print(json.dumps(build_abot_scale_evidence(
        manifest_path=args.manifest, local_points_path=args.local_points,
        confidence_path=args.confidence, output_path=args.output,
        maximum_frames=args.maximum_frames, sample_stride=args.sample_stride,
    ), indent=2))


def _import_abot_loops(args: argparse.Namespace) -> None:
    scale, _ = _metric_scale(args)
    print(json.dumps(import_abot_loop_proposals(
        manifest_path=args.manifest, loop_edges_path=args.loop_edges,
        output_path=args.output, metric_scale=scale,
    ), indent=2))


def _adapt_slamformer(args: argparse.Namespace) -> None:
    scale, evidence = _metric_scale(args)
    print(json.dumps(adapt_slamformer_revision(
        manifest_path=args.manifest,
        baseline_trajectory_path=args.baseline_trajectory,
        final_traj_path=args.final_traj, output_path=args.output,
        metric_scale=scale, scale_evidence=evidence,
        identifier_mode=args.identifier_mode, model_variant=args.model_variant,
        model_commit=args.model_commit, checkpoint_sha256=args.checkpoint_sha256,
    ), indent=2))


def _adapt_mapanything(args: argparse.Namespace) -> None:
    scale, _ = _metric_scale(args)
    print(json.dumps(adapt_mapanything_revision(
        manifest_path=args.manifest,
        baseline_trajectory_path=args.baseline_trajectory,
        window_paths=args.windows, output_path=args.output,
        metric_scale=scale, input_mode=args.input_mode,
        window_size=args.window_size, model_commit=args.model_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        maximum_overlap_translation_m=args.maximum_overlap_translation,
        maximum_overlap_rotation_deg=args.maximum_overlap_rotation,
    ), indent=2))


def _runtime_report(args: argparse.Namespace) -> None:
    samples = json.loads(args.latency_ms.read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("latency_ms")
    failure = None
    if args.failure_json is not None:
        failure = json.loads(args.failure_json.read_text(encoding="utf-8"))
    print(json.dumps(write_model_runtime_report(
        args.output, manifest_path=args.manifest, model=args.model,
        model_commit=args.model_commit, checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        resolution=(args.width, args.height), latency_ms=samples,
        peak_gpu_memory_mb=args.peak_gpu_memory_mb,
        output_pose_count=args.output_pose_count,
        dropped_frame_ids=args.dropped_frame_id,
        queue_depth_peak=args.queue_depth_peak, wall_time_s=args.wall_time,
        mode=args.mode,
        status=args.status, failure=failure,
    ), indent=2))


def _trajectory_revision(args: argparse.Namespace) -> None:
    print(json.dumps(write_trajectory_revision(
        args.output, parent_trajectory_path=args.parent,
        revised_trajectory_path=args.revised, source=args.source,
        affected_frame_ids=args.affected_frame_id,
        runtime_report_path=args.runtime_report,
    ), indent=2))


def _external_artifact(args: argparse.Namespace) -> None:
    frame_ids = json.loads(args.source_frame_ids.read_text(encoding="utf-8"))
    if isinstance(frame_ids, dict):
        frame_ids = frame_ids.get("frame_ids")
    print(json.dumps(write_external_artifact_manifest(
        args.output, manifest_path=args.manifest, system=args.system,
        role=args.role, artifacts=args.artifact, source_frame_ids=frame_ids,
        runtime_s=args.runtime,
    ), indent=2))


def _model_comparison(args: argparse.Namespace) -> None:
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    if isinstance(candidates, dict):
        candidates = candidates.get("candidates")
    print(json.dumps(write_model_comparison(
        args.output, candidates=candidates,
        frozen_baseline_commit=args.frozen_baseline_commit,
    ), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pose_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--dataset", choices=("scannet", "3rscan", "orbbec"), required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--no-rotate-ccw", action="store_true")
    manifest.add_argument("--frame-id", type=int, action="append", default=[])
    manifest.set_defaults(handler=_manifest)
    replay = commands.add_parser("replay")
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--socket", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--timeout", type=float, default=30.0)
    replay.set_defaults(handler=_replay)
    run = commands.add_parser("run")
    run.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--trajectory", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--maximum-loop-pairs", type=int, default=36)
    run.add_argument(
        "--high-leverage-loop-min-span-fraction", type=float,
        help="opt-in span fraction at which loop weights are capped",
    )
    run.add_argument(
        "--high-leverage-loop-weight-cap", type=float, default=1.5,
    )
    run.set_defaults(handler=_run)
    import_trajectory = commands.add_parser("import-trajectory")
    import_trajectory.add_argument("--input", type=Path, required=True)
    import_trajectory.add_argument("--manifest", type=Path, required=True)
    import_trajectory.add_argument("--output", type=Path, required=True)
    import_trajectory.add_argument("--source", default="DPV-SLAM")
    import_trajectory.add_argument(
        "--format", choices=("legacy_tcw_mm", "dpv_response_jsonl"),
        default="legacy_tcw_mm",
    )
    import_trajectory.set_defaults(handler=_import_trajectory)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--estimate", type=Path, required=True)
    evaluate.add_argument("--reference", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(handler=_evaluate)
    refuse = commands.add_parser("refuse")
    refuse.add_argument("--manifest", type=Path, required=True)
    refuse.add_argument("--trajectory", type=Path, required=True)
    refuse.add_argument("--output", type=Path, required=True)
    refuse.add_argument("--fused-frame-ids", type=Path)
    refuse.add_argument("--voxel-length", type=float, default=0.02)
    refuse.add_argument("--sdf-trunc", type=float, default=0.08)
    refuse.add_argument("--depth-trunc", type=float, default=4.50)
    refuse.set_defaults(handler=_refuse)

    def add_scale_options(command: argparse.ArgumentParser) -> None:
        group = command.add_mutually_exclusive_group(required=True)
        group.add_argument("--scale-evidence", type=Path)
        group.add_argument("--metric-scale", type=float)
        command.add_argument("--scale-method")

    abot = commands.add_parser("adapt-abot")
    abot.add_argument("--manifest", type=Path, required=True)
    abot.add_argument("--poses", type=Path, required=True)
    abot.add_argument("--output", type=Path, required=True)
    abot.add_argument("--mode", choices=("noloop", "official_loop"), required=True)
    abot.add_argument("--model-commit", required=True)
    abot.add_argument("--checkpoint-sha256", required=True)
    add_scale_options(abot)
    abot.set_defaults(handler=_adapt_abot)

    abot_scale = commands.add_parser("abot-scale-evidence")
    abot_scale.add_argument("--manifest", type=Path, required=True)
    abot_scale.add_argument("--local-points", type=Path, required=True)
    abot_scale.add_argument("--confidence", type=Path)
    abot_scale.add_argument("--maximum-frames", type=int, default=32)
    abot_scale.add_argument("--sample-stride", type=int, default=8)
    abot_scale.add_argument("--output", type=Path, required=True)
    abot_scale.set_defaults(handler=_abot_scale_evidence)

    abot_loops = commands.add_parser("import-abot-loops")
    abot_loops.add_argument("--manifest", type=Path, required=True)
    abot_loops.add_argument("--loop-edges", type=Path, required=True)
    abot_loops.add_argument("--output", type=Path, required=True)
    add_scale_options(abot_loops)
    abot_loops.set_defaults(handler=_import_abot_loops)

    slamformer = commands.add_parser("adapt-slamformer")
    slamformer.add_argument("--manifest", type=Path, required=True)
    slamformer.add_argument("--baseline-trajectory", type=Path, required=True)
    slamformer.add_argument("--final-traj", type=Path, required=True)
    slamformer.add_argument("--output", type=Path, required=True)
    slamformer.add_argument("--identifier-mode", choices=("auto", "frame_id", "timestamp_us", "timestamp_s"), default="auto")
    slamformer.add_argument("--model-variant", choices=("V1.1-long@224", "V1.1@518"), required=True)
    slamformer.add_argument("--model-commit", required=True)
    slamformer.add_argument("--checkpoint-sha256", required=True)
    add_scale_options(slamformer)
    slamformer.set_defaults(handler=_adapt_slamformer)

    mapanything = commands.add_parser("adapt-mapanything")
    mapanything.add_argument("--manifest", type=Path, required=True)
    mapanything.add_argument("--baseline-trajectory", type=Path, required=True)
    mapanything.add_argument("--windows", type=Path, nargs="+", required=True)
    mapanything.add_argument("--output", type=Path, required=True)
    mapanything.add_argument("--input-mode", choices=("independent_rgb_intrinsics_depth", "conditioned_on_dpv_pose"), required=True)
    mapanything.add_argument("--window-size", type=int, choices=(8, 16), required=True)
    mapanything.add_argument("--model-commit", required=True)
    mapanything.add_argument("--checkpoint-sha256", required=True)
    mapanything.add_argument("--maximum-overlap-translation", type=float, default=0.08)
    mapanything.add_argument("--maximum-overlap-rotation", type=float, default=5.0)
    add_scale_options(mapanything)
    mapanything.set_defaults(handler=_adapt_mapanything)

    runtime = commands.add_parser("model-runtime-report")
    runtime.add_argument("--manifest", type=Path, required=True)
    runtime.add_argument("--model", required=True)
    runtime.add_argument("--model-commit", required=True)
    checkpoint = runtime.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--checkpoint", type=Path)
    checkpoint.add_argument("--checkpoint-sha256")
    runtime.add_argument("--width", type=int, required=True)
    runtime.add_argument("--height", type=int, required=True)
    runtime.add_argument("--latency-ms", type=Path, required=True)
    runtime.add_argument("--peak-gpu-memory-mb", type=float)
    runtime.add_argument("--output-pose-count", type=int, required=True)
    runtime.add_argument("--dropped-frame-id", type=int, action="append", default=[])
    runtime.add_argument("--queue-depth-peak", type=int, default=0)
    runtime.add_argument("--wall-time", type=float)
    runtime.add_argument("--mode", default="official_weights")
    runtime.add_argument("--status", choices=("completed", "failed"), default="completed")
    runtime.add_argument("--failure-json", type=Path)
    runtime.add_argument("--output", type=Path, required=True)
    runtime.set_defaults(handler=_runtime_report)

    revision = commands.add_parser("trajectory-revision")
    revision.add_argument("--parent", type=Path, required=True)
    revision.add_argument("--revised", type=Path, required=True)
    revision.add_argument("--source", required=True)
    revision.add_argument("--affected-frame-id", type=int, action="append", required=True)
    revision.add_argument("--runtime-report", type=Path)
    revision.add_argument("--output", type=Path, required=True)
    revision.set_defaults(handler=_trajectory_revision)

    external = commands.add_parser("external-artifact")
    external.add_argument("--manifest", type=Path, required=True)
    external.add_argument("--system", choices=("MipMap", "FixAnything"), required=True)
    external.add_argument("--role", choices=("offline_geometry_control", "presentation_only"), required=True)
    external.add_argument("--artifact", type=Path, action="append", required=True)
    external.add_argument("--source-frame-ids", type=Path, required=True)
    external.add_argument("--runtime", type=float)
    external.add_argument("--output", type=Path, required=True)
    external.set_defaults(handler=_external_artifact)

    comparison = commands.add_parser("model-comparison")
    comparison.add_argument("--candidates", type=Path, required=True)
    comparison.add_argument("--frozen-baseline-commit", required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.set_defaults(handler=_model_comparison)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
