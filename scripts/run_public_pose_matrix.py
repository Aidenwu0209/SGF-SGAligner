#!/usr/bin/env python3
"""Run create-only ScanNet or 3RScan DPV/backend A/B experiments.

Ground-truth pose files are opened only after both inference arms have
finished.  The DPV worker is restarted for every sequence so tracker state can
never cross a dataset boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

from pose_pipeline.adapters import scan3r_manifest, scannet_manifest
from pose_pipeline.contracts import (
    PoseRecord,
    load_manifest,
    load_trajectory,
    sha256_file,
    stable_json_sha256,
    write_input_sha256_audit,
    write_manifest,
    write_trajectory,
)
from pose_pipeline.evaluation import (
    evaluate_trajectory_files,
    paired_bootstrap_improvement,
    scan3r_reference_trajectory,
    scannet_reference_trajectory,
)
from pose_pipeline.geometry_metrics import (
    compare_no_gt_geometry,
    ply_geometry_metrics,
    render_fixed_comparison_views,
)
from pose_pipeline.replay import replay_manifest
from pose_pipeline.robust_backend import transform_distance
from pose_pipeline.runner import run_sequence
from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _wait_for_socket(path: Path, process: subprocess.Popen, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"DPV worker exited with code {process.returncode}")
        if path.exists():
            return
        time.sleep(0.2)
    raise TimeoutError(f"DPV socket not ready: {path}")


def _run_frontend(args: argparse.Namespace, manifest_path: Path, output: Path) -> dict:
    socket_id = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    socket_path = Path(f"/tmp/sgf_pose_{socket_id}.sock")
    if socket_path.exists():
        socket_path.unlink()
    command = [
        str(args.dpv_python), "-u", str(args.dpv_worker),
        "--socket", str(socket_path),
        "--dpvo-root", str(args.dpvo_root),
        "--network", str(args.dpvo_network),
        "--config", str(args.dpvo_config),
        "--seed", str(args.seed),
        "--loop-closure", "--no-gravity-align",
    ]
    log_path = output / "worker.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_for_socket(socket_path, process)
            result = replay_manifest(
                manifest_path=manifest_path,
                socket_path=socket_path,
                output_dir=output / "frontend",
                timeout_s=args.frame_timeout,
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
    return {**result, "worker_command": command, "worker_log": str(log_path.resolve())}


def _evaluation_reference(
    dataset: str, sequence: Path, trajectory: list[PoseRecord], manifest_path: Path,
) -> list[PoseRecord]:
    manifest = load_manifest(manifest_path)
    timestamp_by_id = {frame.frame_id: frame.timestamp_us for frame in manifest.frames}
    frame_ids = [row.frame_id for row in trajectory]
    if dataset == "scannet":
        return scannet_reference_trajectory(sequence, frame_ids, timestamp_by_id)
    return scan3r_reference_trajectory(sequence, frame_ids, timestamp_by_id)


def _write_evaluation_input_audit(
    dataset: str, sequence: Path, frame_ids: Iterable[int], output: Path,
) -> dict:
    rows = []
    for frame_id in frame_ids:
        path = (
            sequence / "pose" / f"{frame_id}.txt"
            if dataset == "scannet"
            else sequence / f"frame-{frame_id:06d}.pose.txt"
        )
        rows.append({
            "frame_id": frame_id,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "role": "evaluation_only",
        })
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "schema": "pose_evaluation_input_audit.v1",
        "frame_count": len(rows),
        "records_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest(),
        "gt_role": "evaluation_only",
    }


def _catastrophic_edges(candidate: Path, reference: list[PoseRecord]) -> dict:
    evidence_path = candidate / "loop_evidence.json"
    if not evidence_path.is_file():
        return {
            "evaluated_edges": 0, "evaluable_edges": 0,
            "unevaluable_edges": 0, "catastrophic_edges": 0, "rows": [],
        }
    reference_by_id = {row.frame_id: row.t_world_camera for row in reference}
    payload = json.loads(evidence_path.read_text())
    rows = []
    for item in payload.get("evidence", []):
        registration = item.get("registration", {})
        if registration.get("accepted") is not True:
            continue
        source = int(item["source_frame_id"])
        target = int(item["target_frame_id"])
        if source not in reference_by_id or target not in reference_by_id:
            rows.append({
                "source_frame_id": source,
                "target_frame_id": target,
                "evaluable": False,
                "reason": "non_finite_or_missing_evaluation_pose",
                "catastrophic": False,
            })
            continue
        estimate = np.asarray(registration["transform"], dtype=np.float64)
        truth = np.linalg.inv(reference_by_id[target]) @ reference_by_id[source]
        rotation_deg, translation_m = transform_distance(estimate, truth)
        rows.append({
            "source_frame_id": source,
            "target_frame_id": target,
            "rotation_error_deg": rotation_deg,
            "translation_error_m": translation_m,
            "evaluable": True,
            "catastrophic": rotation_deg > 20.0 or translation_m > 0.5,
        })
    return {
        "evaluated_edges": len(rows),
        "evaluable_edges": sum(bool(row.get("evaluable")) for row in rows),
        "unevaluable_edges": sum(not bool(row.get("evaluable")) for row in rows),
        "catastrophic_edges": sum(bool(row["catastrophic"]) for row in rows),
        "rows": rows,
    }


def _refuse_pair(
    manifest_path: Path, baseline: Path, candidate: Path, output: Path,
) -> dict:
    baseline_result = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest_path,
        trajectory=baseline / "trajectory.json",
        output_dir=output / "baseline_refusion",
    ))
    candidate_result = run_full_rgbd_refusion(FullRefusionRequest(
        manifest=manifest_path,
        trajectory=candidate / "trajectory.json",
        output_dir=output / "candidate_refusion",
    ))
    baseline_geometry = ply_geometry_metrics(Path(baseline_result["cloud"]))
    candidate_geometry = ply_geometry_metrics(Path(candidate_result["cloud"]))
    comparison = compare_no_gt_geometry(baseline_geometry, candidate_geometry)
    write_json(output / "baseline_geometry.json", baseline_geometry)
    write_json(output / "candidate_geometry.json", candidate_geometry)
    write_json(output / "geometry_comparison.json", comparison)
    view = render_fixed_comparison_views(
        Path(baseline_result["cloud"]), Path(candidate_result["cloud"]),
        output / "baseline_candidate_fixed_views.png",
    )
    write_json(output / "baseline_candidate_fixed_views.json", view)
    return comparison


def run_scene(
    args: argparse.Namespace, sequence_id: str, sequence: Path, output: Path,
    *, split: str,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    manifest = (
        scannet_manifest(sequence)
        if args.dataset == "scannet"
        else scan3r_manifest(sequence)
    )
    manifest_path = output / "manifest.json"
    write_manifest(manifest_path, manifest)
    input_audit = write_input_sha256_audit(output / "inference_inputs.sha256.jsonl", manifest)
    frontend = _run_frontend(args, manifest_path, output)
    environment = {
        "schema": "pose_pipeline_environment.v1",
        "hostname": platform.node(),
        "platform": platform.platform(),
        "orchestrator_python": sys.version,
        "dpv_python": str(args.dpv_python.resolve()),
        "dpv_worker": str(args.dpv_worker.resolve()),
        "dpv_worker_sha256": sha256_file(args.dpv_worker),
        "dpvo_network_sha256": sha256_file(args.dpvo_network),
        "dpvo_config_sha256": sha256_file(args.dpvo_config),
        "worker_command": frontend["worker_command"],
        "seed": args.seed,
        "gt_consumed": False,
    }
    try:
        environment["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        environment["git_head"] = None
    write_json(output / "environment.json", environment)
    tracked_manifest = output / "frontend" / "tracked_manifest.json"
    frontend_trajectory = output / "frontend" / "trajectory.json"
    baseline = run_sequence(
        arm="baseline", manifest_path=tracked_manifest,
        trajectory_path=frontend_trajectory, output_dir=output / "baseline",
    )
    candidate = run_sequence(
        arm="candidate", manifest_path=tracked_manifest,
        trajectory_path=frontend_trajectory, output_dir=output / "candidate",
    )

    # Evaluation phase starts here; no code above this line opens pose files.
    estimate, _ = load_trajectory(frontend_trajectory)
    reference = _evaluation_reference(args.dataset, sequence, estimate, manifest_path)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    eval_audit = _write_evaluation_input_audit(
        args.dataset, sequence, [row.frame_id for row in estimate],
        evaluation_dir / "pose_inputs.sha256.jsonl",
    )
    write_json(evaluation_dir / "pose_inputs.audit.json", eval_audit)
    reference_path = evaluation_dir / "reference_trajectory.json"
    write_trajectory(
        reference_path, reference, sequence_id=sequence_id, arm="evaluation_reference",
        metadata={"gt_role": "evaluation_only", "opened_after_both_inference_arms": True},
    )
    baseline_metrics = evaluate_trajectory_files(
        output / "baseline" / "trajectory.json", reference_path,
        evaluation_dir / "baseline_pose.json",
    )
    candidate_metrics = evaluate_trajectory_files(
        output / "candidate" / "trajectory.json", reference_path,
        evaluation_dir / "candidate_pose.json",
    )
    edge_safety = _catastrophic_edges(output / "candidate", reference)
    write_json(evaluation_dir / "accepted_edge_safety.json", edge_safety)
    geometry = None
    if args.refuse:
        geometry = _refuse_pair(
            tracked_manifest, output / "baseline", output / "candidate", output,
        )
    return {
        "sequence_id": sequence_id,
        "split": split,
        "status": "completed",
        "input_frames": frontend["input_frame_count"],
        "valid_poses": frontend["valid_pose_count"],
        "coverage": frontend["coverage"],
        "evaluation_coverage": baseline_metrics["evaluation_coverage"],
        "candidate_backend_correction": bool(candidate.get("backend_correction_applied", False)),
        "candidate_reason": candidate.get("reason"),
        "accepted_loops": int(candidate.get("accepted_loop_count", 0)),
        "baseline_ate_translation_rmse_m": baseline_metrics["absolute_translation_m"]["rmse"],
        "candidate_ate_translation_rmse_m": candidate_metrics["absolute_translation_m"]["rmse"],
        "baseline_ate_rotation_rmse_deg": baseline_metrics["absolute_rotation_deg"]["rmse"],
        "candidate_ate_rotation_rmse_deg": candidate_metrics["absolute_rotation_deg"]["rmse"],
        "catastrophic_edges": edge_safety["catastrophic_edges"],
        "unevaluable_edges": edge_safety["unevaluable_edges"],
        "input_records_sha256": input_audit["records_sha256"],
        "geometry_safety": None if geometry is None else geometry["passes_scene_safety"],
        "geometry_improvement": None if geometry is None else geometry["passes_scene_improvement"],
    }


def _scannet_sequences(root: Path) -> list[tuple[str, Path, str]]:
    rows = []
    for scene in sorted(root.glob("scene*")):
        if all((scene / name).is_dir() for name in ("color", "depth", "pose", "intrinsic")):
            rows.append((scene.name, scene, "held_out"))
    development = {
        sequence_id for sequence_id, _, _ in sorted(
            rows, key=lambda row: hashlib.sha256(row[0].encode()).hexdigest(),
        )[:4]
    }
    return [(name, path, "development" if name in development else split) for name, path, split in rows]


def _scan3r_sequences(
    root: Path, selection_path: Path,
) -> tuple[list[tuple[str, Path, str]], dict]:
    selection = json.loads(selection_path.read_text())
    if selection.get("schema") != "scan3r_pose_selection.v1":
        raise ValueError("3RScan selection schema mismatch")
    unsigned = dict(selection)
    expected = unsigned.pop("payload_sha256", None)
    if expected != stable_json_sha256(unsigned):
        raise ValueError("3RScan selection payload SHA mismatch")
    if selection.get("contains_transforms") is not False:
        raise ValueError("3RScan inference selection must not contain transforms")
    def forbidden_key(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                "transform" in str(key).lower() or forbidden_key(child)
                for key, child in value.items()
                if key != "contains_transforms"
            )
        if isinstance(value, list):
            return any(forbidden_key(child) for child in value)
        return False
    if forbidden_key(selection):
        raise ValueError("3RScan inference selection contains a transform-like key")
    rows, missing = [], []
    for item in selection.get("sequences", []):
        sequence_id, split = str(item["sequence_id"]), str(item["split"])
        sequence = root / sequence_id / "sequence"
        if bool(item.get("present")) and (sequence / "_info.txt").is_file():
            rows.append((sequence_id, sequence, split))
        else:
            missing.append({"sequence_id": sequence_id, "split": split})
    audit = {
        "schema": "scan3r_validation_selection.v1",
        "selection_sha256": sha256_file(selection_path),
        "selection_payload_sha256": selection.get("payload_sha256"),
        "contains_transforms": False,
        "listed_sequences": len(selection.get("sequences", [])),
        "present_sequences": len(rows),
        "missing_sequences": missing,
        "validation_group_count": sum(
            group.get("split") == "validation"
            for group in selection.get("groups", [])
        ),
        "development_group_count": sum(
            group.get("split") == "development"
            for group in selection.get("groups", [])
        ),
    }
    return rows, audit


def _aggregate(dataset: str, rows: list[dict], selection: dict | None) -> dict:
    complete = [row for row in rows if row.get("status") == "completed"]
    paired = [row for row in complete if row["split"] not in {"development", "train_failure_sentinel"}]
    pose_bootstrap = None
    if paired:
        pose_bootstrap = {
            "ate_translation_rmse": paired_bootstrap_improvement(
                [row["baseline_ate_translation_rmse_m"] for row in paired],
                [row["candidate_ate_translation_rmse_m"] for row in paired],
            ),
            "ate_rotation_rmse": paired_bootstrap_improvement(
                [row["baseline_ate_rotation_rmse_deg"] for row in paired],
                [row["candidate_ate_rotation_rmse_deg"] for row in paired],
            ),
        }
    catastrophic = sum(int(row.get("catastrophic_edges", 0)) for row in complete)
    unevaluable = sum(int(row.get("unevaluable_edges", 0)) for row in complete)
    corrected = sum(bool(row.get("candidate_backend_correction")) for row in complete)
    baseline_coverage = float(np.mean([
        row["coverage"] for row in complete
    ])) if complete else 0.0
    candidate_coverage = baseline_coverage
    return {
        "schema": "public_pose_matrix.v1",
        "dataset": dataset,
        "requested_sequences": len(rows),
        "completed_sequences": len(complete),
        "corrected_sequences": corrected,
        "fail_closed_noop_sequences": len(complete) - corrected,
        "baseline_valid_pose_coverage_mean": baseline_coverage,
        "candidate_valid_pose_coverage_mean": candidate_coverage,
        "candidate_coverage_loss_fraction": baseline_coverage - candidate_coverage,
        "catastrophic_edges": catastrophic,
        "unevaluable_edges": unevaluable,
        "pose_bootstrap": pose_bootstrap,
        "passes_pose_gate": bool(
            paired and pose_bootstrap
            and pose_bootstrap["ate_translation_rmse"]["passes_10pct_and_positive_ci"]
            and pose_bootstrap["ate_rotation_rmse"]["passes_10pct_and_positive_ci"]
            and catastrophic == 0 and unevaluable == 0
            and baseline_coverage - candidate_coverage <= 0.01
        ),
        "selection_audit": selection,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scannet", "3rscan"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scan3r-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpv-python", type=Path, required=True)
    parser.add_argument("--dpv-worker", type=Path, required=True)
    parser.add_argument("--dpvo-root", type=Path, required=True)
    parser.add_argument("--dpvo-network", type=Path, required=True)
    parser.add_argument("--dpvo-config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--frame-timeout", type=float, default=30.0)
    parser.add_argument("--refuse", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", action="append", help="exact sequence id to run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = None
    if args.dataset == "scannet":
        sequences = _scannet_sequences(args.data_root)
    else:
        if args.scan3r_selection is None:
            parser.error("--scan3r-selection is required for 3rscan")
        sequences, selection = _scan3r_sequences(
            args.data_root, args.scan3r_selection,
        )
        write_json(args.output / "selection_audit.json", selection)
    if args.only:
        wanted = set(args.only)
        sequences = [row for row in sequences if row[0] in wanted]
        missing = sorted(wanted - {row[0] for row in sequences})
        if missing:
            raise ValueError(f"requested sequences not selected/present: {missing}")
    if args.limit is not None:
        sequences = sequences[:args.limit]
    rows = []
    for sequence_id, sequence, split in sequences:
        try:
            row = run_scene(
                args, sequence_id, sequence,
                args.output / sequence_id, split=split,
            )
        except Exception as error:
            row = {
                "sequence_id": sequence_id,
                "split": split,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        write_json(args.output / f"progress_{len(rows):04d}.json", row)
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = _aggregate(args.dataset, rows, selection)
    write_json(args.output / "summary.json", summary)
    fields = [
        "sequence_id", "split", "status", "input_frames", "valid_poses", "coverage", "evaluation_coverage",
        "candidate_backend_correction", "candidate_reason", "accepted_loops",
        "baseline_ate_translation_rmse_m", "candidate_ate_translation_rmse_m",
        "baseline_ate_rotation_rmse_deg", "candidate_ate_rotation_rmse_deg",
        "catastrophic_edges", "unevaluable_edges", "geometry_safety", "geometry_improvement", "error",
    ]
    with (args.output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
