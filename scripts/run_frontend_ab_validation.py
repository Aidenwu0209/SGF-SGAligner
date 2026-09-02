#!/usr/bin/env python3
"""Run a create-only, same-input DPV frontend A/B validation.

The baseline and candidate workers see the same GT-free RGB-D manifest. Dataset
poses are opened only after both workers have exited.  The candidate may expose
the create-only finalized-pose sidecar used to recover real warm-up/recovery
estimates; the baseline worker is not required to support that option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pose_pipeline.adapters import scan3r_manifest, scannet_manifest  # noqa: E402
from pose_pipeline.contracts import (  # noqa: E402
    PoseRecord,
    load_manifest,
    load_trajectory,
    sha256_file,
    write_input_sha256_audit,
    write_manifest,
    write_trajectory,
)
from pose_pipeline.evaluation import (  # noqa: E402
    evaluate_trajectory_files,
    scan3r_reference_trajectory,
    scannet_reference_trajectory,
)
from pose_pipeline.geometry_metrics import ply_geometry_metrics  # noqa: E402
from pose_pipeline.replay import replay_manifest  # noqa: E402
from reconstruction.rgbd_refusion import (  # noqa: E402
    FullRefusionRequest,
    run_full_rgbd_refusion,
)


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
    *,
    worker: Path,
    socket_path: Path,
    finalized_path: Path | None,
) -> list[str]:
    command = [
        str(args.dpv_python), "-u", str(worker),
        "--socket", str(socket_path),
        "--dpvo-root", str(args.dpvo_root),
        "--network", str(args.network),
        "--config", str(args.dpvo_config),
        "--metric-config", str(args.metric_config),
        "--seed", str(args.seed),
        "--no-gravity-align",
        "--loop-closure" if args.loop_closure else "--no-loop-closure",
    ]
    if finalized_path is not None:
        command.extend(("--finalized-trajectory", str(finalized_path)))
    return command


def _run_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    worker: Path,
    manifest_path: Path,
    output: Path,
    use_finalized_sidecar: bool,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    socket_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    socket_path = Path(f"/tmp/sgf_frontend_ab_{socket_id}.sock")
    if socket_path.exists():
        socket_path.unlink()
    finalized_path = output / "finalized_poses.jsonl" if use_finalized_sidecar else None
    command = _worker_command(
        args,
        worker=worker,
        socket_path=socket_path,
        finalized_path=finalized_path,
    )
    log_path = output / "worker.log"
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
                output_dir=output / "frontend",
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
    result = {
        **replay,
        "arm": arm,
        "worker": str(worker.resolve()),
        "worker_sha256": sha256_file(worker),
        "worker_command": command,
        "worker_log": str(log_path.resolve()),
        "uses_finalized_sidecar": use_finalized_sidecar,
    }
    _write_json(output / "result.json", result)
    return result


def _reference(
    dataset: str,
    sequence: Path,
    frame_ids: list[int],
    timestamp_by_id: dict[int, int],
) -> list[PoseRecord]:
    if dataset == "scannet":
        return scannet_reference_trajectory(sequence, frame_ids, timestamp_by_id)
    return scan3r_reference_trajectory(sequence, frame_ids, timestamp_by_id)


def _write_reference_and_estimate(
    *,
    name: str,
    dataset: str,
    sequence: Path,
    estimates_by_id: dict[int, PoseRecord],
    requested_ids: list[int],
    timestamp_by_id: dict[int, int],
    evaluation_dir: Path,
) -> dict:
    reference = _reference(dataset, sequence, requested_ids, timestamp_by_id)
    evaluable_ids = [row.frame_id for row in reference]
    estimates = [estimates_by_id[frame_id] for frame_id in evaluable_ids]
    estimate_path = evaluation_dir / f"{name}_estimate.json"
    reference_path = evaluation_dir / f"{name}_reference.json"
    write_trajectory(
        estimate_path,
        estimates,
        sequence_id=sequence.parent.name if dataset == "3rscan" else sequence.name,
        arm=name,
        metadata={"gt_consumed": False},
    )
    write_trajectory(
        reference_path,
        reference,
        sequence_id=sequence.parent.name if dataset == "3rscan" else sequence.name,
        arm=f"{name}_evaluation_reference",
        metadata={"gt_role": "evaluation_only", "opened_after_both_inference_arms": True},
    )
    return evaluate_trajectory_files(
        estimate_path,
        reference_path,
        evaluation_dir / f"{name}_metrics.json",
    )


def _metric_not_regressed(
    baseline: float,
    candidate: float,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    return candidate <= baseline * (1.0 + relative_tolerance) + absolute_tolerance


def _pose_gate(
    baseline_replay: dict,
    candidate_replay: dict,
    baseline_common: dict,
    candidate_common: dict,
    *,
    baseline_ids: set[int],
    candidate_ids: set[int],
    input_frame_count: int,
) -> dict:
    checks = {
        "candidate_complete": len(candidate_ids) == input_frame_count,
        "candidate_contains_all_baseline_frames": baseline_ids <= candidate_ids,
        "coverage_not_lower": candidate_replay["coverage"] >= baseline_replay["coverage"],
        "common_ate_translation_not_regressed": _metric_not_regressed(
            baseline_common["absolute_translation_m"]["rmse"],
            candidate_common["absolute_translation_m"]["rmse"],
            relative_tolerance=0.05,
            absolute_tolerance=0.01,
        ),
        "common_ate_rotation_not_regressed": _metric_not_regressed(
            baseline_common["absolute_rotation_deg"]["rmse"],
            candidate_common["absolute_rotation_deg"]["rmse"],
            relative_tolerance=0.05,
            absolute_tolerance=0.5,
        ),
        "common_rpe_translation_not_regressed": _metric_not_regressed(
            baseline_common["relative_translation_m"]["rmse"],
            candidate_common["relative_translation_m"]["rmse"],
            relative_tolerance=0.05,
            absolute_tolerance=0.005,
        ),
        "common_rpe_rotation_not_regressed": _metric_not_regressed(
            baseline_common["relative_rotation_deg"]["rmse"],
            candidate_common["relative_rotation_deg"]["rmse"],
            relative_tolerance=0.05,
            absolute_tolerance=0.25,
        ),
        "identity_fallback_rejected": not candidate_replay["identity_fallback_used"],
        "gt_not_consumed_during_inference": not candidate_replay["gt_consumed"],
    }
    return {
        "schema": "frontend_ab_pose_gate.v1",
        "checks": checks,
        "passed": all(checks.values()),
        "coverage_gain_frames": len(candidate_ids) - len(baseline_ids),
        "coverage_gain_fraction": candidate_replay["coverage"] - baseline_replay["coverage"],
        "tolerances": {
            "relative": 0.05,
            "ate_translation_absolute_m": 0.01,
            "ate_rotation_absolute_deg": 0.5,
            "rpe_translation_absolute_m": 0.005,
            "rpe_rotation_absolute_deg": 0.25,
        },
    }


def _run_scene(args: argparse.Namespace, sequence: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest = (
        scannet_manifest(sequence)
        if args.dataset == "scannet"
        else scan3r_manifest(sequence)
    )
    manifest_path = output / "manifest.json"
    write_manifest(manifest_path, manifest)
    input_audit = write_input_sha256_audit(output / "inputs.sha256.jsonl", manifest)

    # No pose/GT file is opened until both calls below have completed.
    baseline_replay = _run_arm(
        args,
        arm="baseline",
        worker=args.baseline_worker,
        manifest_path=manifest_path,
        output=output / "baseline",
        use_finalized_sidecar=False,
    )
    candidate_replay = _run_arm(
        args,
        arm="candidate",
        worker=args.candidate_worker,
        manifest_path=manifest_path,
        output=output / "candidate",
        use_finalized_sidecar=True,
    )

    baseline, _ = load_trajectory(output / "baseline" / "frontend" / "trajectory.json")
    candidate, _ = load_trajectory(output / "candidate" / "frontend" / "trajectory.json")
    baseline_by_id = {row.frame_id: row for row in baseline}
    candidate_by_id = {row.frame_id: row for row in candidate}
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    common_ids = sorted(baseline_ids & candidate_ids)
    if not common_ids:
        raise RuntimeError("baseline and candidate have no common evaluable frames")
    timestamp_by_id = {frame.frame_id: frame.timestamp_us for frame in manifest.frames}
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    baseline_all = _write_reference_and_estimate(
        name="baseline_all",
        dataset=args.dataset,
        sequence=sequence,
        estimates_by_id=baseline_by_id,
        requested_ids=sorted(baseline_ids),
        timestamp_by_id=timestamp_by_id,
        evaluation_dir=evaluation_dir,
    )
    candidate_all = _write_reference_and_estimate(
        name="candidate_all",
        dataset=args.dataset,
        sequence=sequence,
        estimates_by_id=candidate_by_id,
        requested_ids=sorted(candidate_ids),
        timestamp_by_id=timestamp_by_id,
        evaluation_dir=evaluation_dir,
    )
    baseline_common = _write_reference_and_estimate(
        name="baseline_common",
        dataset=args.dataset,
        sequence=sequence,
        estimates_by_id=baseline_by_id,
        requested_ids=common_ids,
        timestamp_by_id=timestamp_by_id,
        evaluation_dir=evaluation_dir,
    )
    candidate_common = _write_reference_and_estimate(
        name="candidate_common",
        dataset=args.dataset,
        sequence=sequence,
        estimates_by_id=candidate_by_id,
        requested_ids=common_ids,
        timestamp_by_id=timestamp_by_id,
        evaluation_dir=evaluation_dir,
    )
    gate = _pose_gate(
        baseline_replay,
        candidate_replay,
        baseline_common,
        candidate_common,
        baseline_ids=baseline_ids,
        candidate_ids=candidate_ids,
        input_frame_count=len(manifest.frames),
    )
    _write_json(evaluation_dir / "pose_gate.json", gate)

    refusion = None
    if args.refuse_candidate and gate["passed"]:
        refusion = run_full_rgbd_refusion(FullRefusionRequest(
            manifest=manifest_path,
            trajectory=output / "candidate" / "frontend" / "trajectory.json",
            output_dir=output / "candidate_refusion",
        ))
        geometry = ply_geometry_metrics(Path(refusion["cloud"]))
        _write_json(output / "candidate_refusion" / "geometry.json", geometry)
        refusion = {**refusion, "geometry": geometry}

    result = {
        "schema": "frontend_ab_sequence.v1",
        "dataset": args.dataset,
        "sequence_id": manifest.sequence_id,
        "input_frames": len(manifest.frames),
        "baseline_valid_poses": len(baseline),
        "candidate_valid_poses": len(candidate),
        "common_valid_poses": len(common_ids),
        "baseline_coverage": baseline_replay["coverage"],
        "candidate_coverage": candidate_replay["coverage"],
        "candidate_online_valid_poses": candidate_replay["online_valid_pose_count"],
        "candidate_backfilled_poses": candidate_replay["backfilled_pose_count"],
        "baseline_all_metrics": baseline_all,
        "candidate_all_metrics": candidate_all,
        "baseline_common_metrics": baseline_common,
        "candidate_common_metrics": candidate_common,
        "pose_gate": gate,
        "candidate_refusion": refusion,
        "input_records_sha256": input_audit["records_sha256"],
        "identity_fallback_used": False,
        "gt_consumed_during_inference": False,
    }
    _write_json(output / "result.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scannet", "3rscan"), required=True)
    parser.add_argument("--sequence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpv-python", type=Path, required=True)
    parser.add_argument("--baseline-worker", type=Path, required=True)
    parser.add_argument("--candidate-worker", type=Path, required=True)
    parser.add_argument("--dpvo-root", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--dpvo-config", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--loop-closure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refuse-candidate", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--frame-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    required = [
        args.dpv_python,
        args.baseline_worker,
        args.candidate_worker,
        args.dpvo_root,
        args.network,
        args.dpvo_config,
        args.metric_config,
        *args.sequence,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing validation inputs: {missing}")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = {
        "schema": "frontend_ab_environment.v1",
        "platform": platform.platform(),
        "python": sys.version,
        "seed": args.seed,
        "loop_closure": args.loop_closure,
        "baseline_worker_sha256": sha256_file(args.baseline_worker),
        "candidate_worker_sha256": sha256_file(args.candidate_worker),
        "network_sha256": sha256_file(args.network),
        "dpvo_config_sha256": sha256_file(args.dpvo_config),
        "metric_config_sha256": sha256_file(args.metric_config),
        "gt_consumed_during_inference": False,
    }
    _write_json(args.output / "environment.json", environment)
    rows = []
    for sequence in args.sequence:
        sequence_id = sequence.parent.name if args.dataset == "3rscan" else sequence.name
        try:
            row = _run_scene(args, sequence.resolve(), args.output / sequence_id)
        except Exception as error:
            row = {
                "schema": "frontend_ab_sequence.v1",
                "dataset": args.dataset,
                "sequence_id": sequence_id,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            row["status"] = "passed" if row["pose_gate"]["passed"] else "gate_failed"
        rows.append(row)
        _write_json(args.output / f"progress_{len(rows):04d}.json", row)
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        if args.stop_on_failure and row["status"] != "passed":
            break
    completed = [row for row in rows if row["status"] in {"passed", "gate_failed"}]
    coverage_gain = sum(
        row["candidate_valid_poses"] - row["baseline_valid_poses"] for row in completed
    )
    all_refusions_completed = (
        not args.refuse_candidate
        or bool(completed) and all(
            row["candidate_refusion"] is not None
            and row["candidate_refusion"]["status"] == "completed"
            for row in completed
        )
    )
    summary = {
        "schema": "frontend_ab_validation.v1",
        "dataset": args.dataset,
        "requested_sequences": len(args.sequence),
        "attempted_sequences": len(rows),
        "completed_sequences": len(completed),
        "passed_sequences": sum(row["status"] == "passed" for row in rows),
        "coverage_gain_frames": coverage_gain,
        "has_measured_improvement": coverage_gain > 0,
        "all_pose_gates_passed": len(rows) == len(args.sequence) and all(
            row["status"] == "passed" for row in rows
        ),
        "all_requested_refusions_completed": all_refusions_completed,
        "identity_fallback_used": False,
        "gt_consumed_during_inference": False,
        "rows": rows,
    }
    summary["passes_promotion_gate"] = bool(
        summary["all_pose_gates_passed"]
        and summary["has_measured_improvement"]
        and summary["all_requested_refusions_completed"]
    )
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if summary["passes_promotion_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
