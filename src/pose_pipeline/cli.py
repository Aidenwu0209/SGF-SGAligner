"""Unified CLI for manifest, replay, baseline/candidate, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import orbbec_manifest, scan3r_manifest, scannet_manifest
from .contracts import (
    load_legacy_tcw_mm, load_manifest, write_manifest, write_trajectory,
)
from .evaluation import evaluate_trajectory_files
from .replay import replay_manifest
from .runner import run_sequence


def _manifest(args: argparse.Namespace) -> None:
    if args.dataset == "scannet":
        value = scannet_manifest(args.input)
    elif args.dataset == "3rscan":
        value = scan3r_manifest(args.input, rotate_ccw=not args.no_rotate_ccw)
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
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--dataset", choices=("scannet", "3rscan", "orbbec"), required=True)
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--no-rotate-ccw", action="store_true")
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
