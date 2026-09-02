#!/usr/bin/env python3
"""Create-only Orbbec DPV validation with explicit loop-closure causality arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_pipeline.contracts import (  # noqa: E402
    load_manifest,
    load_trajectory,
    sha256_file,
    write_input_sha256_audit,
)
from pose_pipeline.replay import replay_manifest  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _wait_for_socket(path: Path, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"DPV worker exited with code {process.returncode}")
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"DPV socket not ready: {path}")


def _worker_command(
    args: argparse.Namespace,
    socket_path: Path,
    finalized_path: Path,
) -> list[str]:
    command = [
        str(args.dpv_python),
        "-u",
        str(args.dpv_worker),
        "--socket",
        str(socket_path),
        "--dpvo-root",
        str(args.dpvo_root),
        "--network",
        str(args.network),
        "--config",
        str(args.dpvo_config),
        "--metric-config",
        str(args.metric_config),
        "--finalized-trajectory",
        str(finalized_path),
        "--seed",
        str(args.seed),
        "--no-gravity-align",
    ]
    command.append("--loop-closure" if args.loop_closure else "--no-loop-closure")
    return command


def _run_one(
    args: argparse.Namespace,
    manifest_path: Path,
    sequence_output: Path,
) -> dict:
    manifest = load_manifest(manifest_path)
    sequence_output.mkdir(parents=True, exist_ok=False)
    audit = write_input_sha256_audit(sequence_output / "inputs.sha256.jsonl", manifest)
    socket_id = hashlib.sha256(str(sequence_output).encode()).hexdigest()[:12]
    socket_path = Path(f"/tmp/sgf_orbbec_{socket_id}.sock")
    if socket_path.exists():
        socket_path.unlink()
    finalized_path = sequence_output / "finalized_poses.jsonl"
    command = _worker_command(args, socket_path, finalized_path)
    log_path = sequence_output / "worker.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_socket(socket_path, process, args.startup_timeout)
            replay = replay_manifest(
                manifest_path=manifest_path,
                socket_path=socket_path,
                output_dir=sequence_output / "frontend",
                timeout_s=args.frame_timeout,
                finalized_trajectory_path=finalized_path,
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15.0)
            if socket_path.exists():
                socket_path.unlink()

    poses, trajectory_payload = load_trajectory(
        sequence_output / "frontend" / "trajectory.json"
    )
    expected_ids = [frame.frame_id for frame in manifest.frames]
    actual_ids = [pose.frame_id for pose in poses]
    complete = actual_ids == expected_ids
    result = {
        **replay,
        "schema": "orbbec_five_fix_sequence_result.v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "input_audit": audit,
        "worker_command": command,
        "worker_log": str(log_path.resolve()),
        "loop_closure": args.loop_closure,
        "trajectory_complete": complete,
        "missing_frame_ids": sorted(set(expected_ids) - set(actual_ids)),
        "stable_pose_sha256_q1e7": trajectory_payload["stable_pose_sha256_q1e7"],
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    _write_json(sequence_output / "result.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpv-python", type=Path, required=True)
    parser.add_argument("--dpv-worker", type=Path, required=True)
    parser.add_argument("--dpvo-root", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--dpvo-config", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument(
        "--loop-closure", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--frame-timeout", type=float, default=60.0)
    parser.add_argument(
        "--fail-fast-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    required = [
        args.dpv_python,
        args.dpv_worker,
        args.dpvo_root,
        args.network,
        args.dpvo_config,
        args.metric_config,
        *args.manifest,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")
    results = []
    stopped_early = False
    for manifest_path in args.manifest:
        manifest = load_manifest(manifest_path)
        result = _run_one(args, manifest_path, output_root / manifest.sequence_id)
        results.append(result)
        if args.fail_fast_incomplete and not result["trajectory_complete"]:
            stopped_early = True
            break
    summary = {
        "schema": "orbbec_five_fix_validation.v1",
        "loop_closure": args.loop_closure,
        "sequence_count_requested": len(args.manifest),
        "sequence_count_completed": len(results),
        "all_trajectories_complete": bool(results) and all(
            result["trajectory_complete"] for result in results
        ),
        "stopped_early": stopped_early,
        "identity_fallback_used": False,
        "gt_consumed": False,
        "results": results,
    }
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["all_trajectories_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
