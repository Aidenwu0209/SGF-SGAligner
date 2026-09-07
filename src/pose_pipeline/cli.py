"""Unified CLI for manifest, replay, baseline/candidate, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .appearance import AppearanceConfig
from .adapters import orbbec_manifest, scan3r_manifest, scannet_manifest
from .contracts import (
    load_legacy_tcw_mm, load_manifest, write_manifest, write_trajectory,
)
from .evaluation import evaluate_trajectory_files
from .geometry_backend import GeometryBootstrapConfig
from .pose_graph import (
    CorrectionAuditConfig, LoopWeightConfig, PoseGraphOptimizationConfig,
)
from .replay import replay_manifest
from .runner import PrecommitGeometryConfig, run_sequence
from .submaps import LoopProposalConfig


def _manifest(args: argparse.Namespace) -> None:
    if args.dataset == "scannet":
        value = scannet_manifest(args.input)
    elif args.dataset == "3rscan":
        preprocessing = (
            "native" if args.no_rotate_ccw else args.scan3r_preprocessing
        )
        value = scan3r_manifest(args.input, preprocessing=preprocessing)
    else:
        value = orbbec_manifest(args.input)
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
        proposal_config=LoopProposalConfig(
            policy=args.proposal_policy,
            maximum_pairs=args.maximum_loop_pairs,
            appearance_mutual_top_k=args.appearance_mutual_top_k,
            maximum_pairs_per_anchor=args.maximum_pairs_per_anchor,
            temporal_bin_count=args.temporal_bin_count,
        ),
        appearance_config=AppearanceConfig(
            provider=args.appearance_provider,
            model_name=args.clip_model,
            device=args.clip_device,
            download_root=(
                None if args.clip_download_root is None
                else str(args.clip_download_root)
            ),
        ),
        geometry_config=GeometryBootstrapConfig(
            correspondence_policy=args.correspondence_policy,
            decision_version=args.registration_decision_version,
        ),
        pose_graph_config=PoseGraphOptimizationConfig(
            robustifier=args.pose_graph_robustifier,
            calculate_leave_one_out=args.leave_one_edge_out,
        ),
        correction_config=CorrectionAuditConfig(
            propagation=args.correction_propagation,
            maximum_adjacent_correction_translation_m=(
                args.maximum_adjacent_correction_translation
            ),
            maximum_adjacent_correction_rotation_deg=(
                args.maximum_adjacent_correction_rotation
            ),
            maximum_absolute_correction_translation_m=(
                args.maximum_absolute_correction_translation
            ),
            maximum_absolute_correction_rotation_deg=(
                args.maximum_absolute_correction_rotation
            ),
        ),
        precommit_geometry_config=PrecommitGeometryConfig(
            enabled=args.precommit_geometry_gate,
            require_scene_improvement=args.precommit_require_improvement,
            frame_stride=args.precommit_frame_stride,
        ),
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
    records = load_legacy_tcw_mm(
        args.input, allowed_frame_ids=allowed, source=args.source,
    )
    write_trajectory(
        args.output, records, sequence_id=manifest.sequence_id, arm="baseline",
        metadata={
            "import_format": "T_cw_row_major_translation_mm",
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


def _run_unified(args: argparse.Namespace) -> None:
    from .unified import run_unified_sequence

    print(json.dumps(run_unified_sequence(
        manifest_path=args.manifest, trajectory_path=args.trajectory,
        output_dir=args.output, config_path=args.config,
        clip_download_root=args.clip_download_root,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pose_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    unified = commands.add_parser(
        "run-unified", help="development Hybrid36 + PnP + bounded Huber + full-frame Guard",
    )
    unified.add_argument("--manifest", type=Path, required=True)
    unified.add_argument("--trajectory", type=Path, required=True)
    unified.add_argument("--output", type=Path, required=True)
    unified.add_argument("--config", type=Path, required=True)
    unified.add_argument("--clip-download-root", type=Path, required=True)
    unified.set_defaults(handler=_run_unified)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--dataset", choices=("scannet", "3rscan", "orbbec"), required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--no-rotate-ccw", action="store_true")
    manifest.add_argument(
        "--scan3r-preprocessing",
        choices=("rotated_ccw", "native"),
        default="rotated_ccw",
        help="explicit 3RScan image/camera basis; legacy default is rotated_ccw",
    )
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
        "--proposal-policy", choices=("distance_topk", "hybrid36"),
        default="distance_topk",
    )
    run.add_argument(
        "--appearance-provider", choices=("none", "openai_clip"),
        default="none",
    )
    run.add_argument("--appearance-mutual-top-k", type=int, default=2)
    run.add_argument("--maximum-pairs-per-anchor", type=int, default=4)
    run.add_argument("--temporal-bin-count", type=int, default=3)
    run.add_argument("--clip-model", default="ViT-B/32")
    run.add_argument("--clip-device", default="auto")
    run.add_argument("--clip-download-root", type=Path)
    run.add_argument(
        "--correspondence-policy",
        choices=("baseline_mutual_fpfh", "spatial_balanced_v2"),
        default="baseline_mutual_fpfh",
    )
    run.add_argument(
        "--registration-decision-version", type=int, choices=(2, 3), default=2,
    )
    run.add_argument(
        "--pose-graph-robustifier", choices=("huber", "adaptive_gnc"),
        default="huber",
    )
    run.add_argument("--leave-one-edge-out", action="store_true")
    run.add_argument(
        "--correction-propagation",
        choices=("legacy_slerp_linear", "se3_correction_field"),
        default="legacy_slerp_linear",
    )
    run.add_argument(
        "--maximum-adjacent-correction-translation", type=float, default=0.05,
    )
    run.add_argument(
        "--maximum-adjacent-correction-rotation", type=float, default=2.0,
    )
    run.add_argument(
        "--maximum-absolute-correction-translation", type=float,
        help="opt-in maximum per-frame correction magnitude in metres",
    )
    run.add_argument(
        "--maximum-absolute-correction-rotation", type=float,
        help="opt-in maximum per-frame correction magnitude in degrees",
    )
    run.add_argument("--precommit-geometry-gate", action="store_true")
    run.add_argument(
        "--precommit-require-improvement", action="store_true",
        help="rollback unless the same-frame refusion passes safety and improvement gates",
    )
    run.add_argument("--precommit-frame-stride", type=int, default=8)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
