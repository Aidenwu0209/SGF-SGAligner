#!/usr/bin/env python3
"""Create-only fixed-trajectory A/B for canonical SGF-SGA depth profiles."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pose_pipeline.contracts import (
    load_manifest,
    sha256_file,
    write_input_sha256_audit,
)
from pose_pipeline.depth_filter import (
    DEPTH_FILTER_PROFILES,
    DepthFilterConfig,
)
from pose_pipeline.geometry_metrics import (
    ply_geometry_metrics,
    render_fixed_comparison_views,
)
from reconstruction.rgbd_refusion import (
    FullRefusionRequest,
    run_full_rgbd_refusion,
)


TIE_BREAK_ORDER = (
    "range_v1",
    "bilateral_light_v1",
    "bilateral_medium_v1",
    "bilateral_light_v1+SOR",
)

SOURCE_FILES = (
    "scripts/run_depth_denoise_ab.py",
    "src/pose_pipeline/depth_filter.py",
    "src/pose_pipeline/replay.py",
    "src/pose_pipeline/submaps.py",
    "src/pose_pipeline/runner.py",
    "src/pose_pipeline/cli.py",
    "src/pose_pipeline/geometry_metrics.py",
    "src/reconstruction/rgbd_refusion.py",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _append_event(path: Path, event: str, **fields) -> None:
    row = {"event": event, "monotonic_s": time.monotonic(), **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            row, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ) + "\n")


def write_source_sha256(path: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    rows = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
        }
        for relative in SOURCE_FILES
    ]
    report = {
        "schema": "depth_denoise_source_sha256.v1",
        "repository_root": str(root),
        "files": rows,
    }
    _write_json(path, report)
    return report


def write_artifact_sha256_manifest(output_dir: Path) -> Path:
    output_dir = Path(output_dir).resolve()
    path = output_dir / "MANIFEST.sha256"
    files = sorted(
        item for item in output_dir.rglob("*")
        if item.is_file() and item != path
    )
    with path.open("x", encoding="utf-8") as stream:
        for item in files:
            stream.write(
                f"{sha256_file(item)}  {item.relative_to(output_dir)}\n"
            )
    return path


def _dominant_plane(metrics: dict) -> dict | None:
    return max(
        metrics.get("horizontal_planes", []),
        key=lambda row: int(row.get("points", 0)), default=None,
    )


def _fraction_improvement(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-12)


def _acceptance(
    baseline: dict, candidate: dict, refusion: dict,
) -> dict:
    baseline_extent = np.asarray(
        baseline["robust_extent_p99_p01_m"], dtype=float,
    )
    candidate_extent = np.asarray(
        candidate["robust_extent_p99_p01_m"], dtype=float,
    )
    extent_ratios = candidate_extent / np.maximum(baseline_extent, 1e-12)
    occupied_ratio = (
        candidate["occupied_voxels_2cm"]
        / max(baseline["occupied_voxels_2cm"], 1)
    )
    conflict_improvement = _fraction_improvement(
        float(baseline["near_parallel_layer_conflict_ratio"]),
        float(candidate["near_parallel_layer_conflict_ratio"]),
    )
    baseline_plane = _dominant_plane(baseline)
    candidate_plane = _dominant_plane(candidate)
    thickness_improvement = None
    if baseline_plane is not None and candidate_plane is not None:
        thickness_improvement = _fraction_improvement(
            float(baseline_plane["thickness_p90_p10_m"]),
            float(candidate_plane["thickness_p90_p10_m"]),
        )
    filter_audit = refusion["depth_filter"]
    latency_p95 = filter_audit["filter_elapsed_ms_p95"]
    edge_retention = filter_audit["strong_edge_half_gradient_retention"]
    coverage_complete = (
        refusion["requested_frame_count"]
        == refusion["integrated_frame_count"]
        == refusion["trajectory_pose_count"]
    )
    improvement_values = [conflict_improvement]
    if thickness_improvement is not None:
        improvement_values.append(thickness_improvement)
    safety_gates = {
        "filter_latency_p95_at_most_10ms": (
            latency_p95 is not None and latency_p95 <= 10.0
        ),
        "strong_edge_half_gradient_retention_at_least_99_5pct": (
            edge_retention is None or edge_retention >= 0.995
        ),
        "all_vertices_finite": bool(candidate["all_vertices_finite"]),
        "complete_fixed_trajectory_coverage": coverage_complete,
        "identity_fallback_absent": not refusion["identity_fallback_used"],
        "gt_input_absent": not refusion["gt_consumed"],
        "occupied_voxels_at_least_95pct": occupied_ratio >= 0.95,
        "every_robust_extent_axis_at_least_98pct": bool(
            np.all(extent_ratios >= 0.98)
        ),
        "layer_conflict_not_worse_over_5pct": conflict_improvement >= -0.05,
        "plane_thickness_not_worse_over_5pct": (
            thickness_improvement is None or thickness_improvement >= -0.05
        ),
    }
    improvement_gates = {
        "plane_thickness_or_layer_conflict_improves_10pct": (
            max(improvement_values) >= 0.10
        ),
    }
    passes_safety = all(safety_gates.values())
    passes_improvement = all(improvement_gates.values())
    return {
        "schema": "depth_denoise_scene_acceptance.v1",
        "passes_scene_safety": passes_safety,
        "passes_scene_improvement": passes_improvement,
        "passes_pilot": passes_safety and passes_improvement,
        "safety_gates": safety_gates,
        "improvement_gates": improvement_gates,
        "occupied_voxel_ratio": occupied_ratio,
        "robust_extent_axis_ratios": extent_ratios.tolist(),
        "layer_conflict_improvement_fraction": conflict_improvement,
        "dominant_plane_thickness_improvement_fraction": thickness_improvement,
        "gt_consumed": False,
    }


def _sor_diagnostic(source_cloud: Path, output_dir: Path) -> dict:
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=False)
    cloud = o3d.io.read_point_cloud(str(source_cloud))
    before = len(cloud.points)
    if before == 0:
        raise RuntimeError("SOR source cloud is empty")
    filtered, _indices = cloud.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0,
    )
    path = output_dir / "sor_diagnostic.ply"
    if not o3d.io.write_point_cloud(str(path), filtered, write_ascii=False):
        raise RuntimeError("Open3D failed to write SOR diagnostic PLY")
    report = {
        "schema": "final_ply_sor_diagnostic.v1",
        "diagnostic_only": True,
        "feeds_tsdf_dpv_or_submaps": False,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "input_point_count": before,
        "output_point_count": len(filtered.points),
        "removed_fraction": (before - len(filtered.points)) / before,
        "cloud": str(path.resolve()),
        "cloud_sha256": sha256_file(path),
        "gt_consumed": False,
    }
    _write_json(output_dir / "sor_result.json", report)
    return report


def _environment() -> dict:
    import cv2
    import open3d

    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "schema": "depth_denoise_environment.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "open3d": open3d.__version__,
        "git_head": revision,
        "runner_sha256": sha256_file(Path(__file__)),
    }


def run_ab(
    *, manifest_path: Path, trajectory_path: Path,
    profiles: tuple[str, ...], output_dir: Path,
    include_sor_diagnostic: bool = True,
) -> dict:
    if "off" not in profiles:
        raise ValueError("profiles must include off as the fixed baseline")
    if len(profiles) != len(set(profiles)):
        raise ValueError("profiles must be unique")
    configs = [DepthFilterConfig.from_profile(value) for value in profiles]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run_log.jsonl"
    log_path.touch(exist_ok=False)
    _append_event(log_path, "started", profiles=list(profiles))

    manifest = load_manifest(manifest_path)
    input_audit = write_input_sha256_audit(
        output_dir / "inference_inputs.sha256.jsonl", manifest,
    )
    environment = _environment()
    environment.update({
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "trajectory": str(Path(trajectory_path).resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "input_audit": input_audit,
        "profile_parameter_hashes": {
            config.profile: config.parameters_sha256 for config in configs
        },
        "gt_consumed": False,
    })
    _write_json(output_dir / "environment.json", environment)
    write_source_sha256(output_dir / "source_sha256.json")

    arms: dict[str, dict] = {}
    for config in configs:
        _append_event(log_path, "profile_started", profile=config.profile)
        arm_dir = output_dir / config.profile
        refusion = run_full_rgbd_refusion(FullRefusionRequest(
            manifest=manifest_path,
            trajectory=trajectory_path,
            output_dir=arm_dir / "refusion",
            voxel_length_m=0.02,
            sdf_trunc_m=0.08,
            depth_trunc_m=4.50,
            depth_filter_config=config,
        ))
        geometry = ply_geometry_metrics(Path(refusion["cloud"]))
        _write_json(arm_dir / "geometry_metrics.json", geometry)
        arms[config.profile] = {
            "refusion": refusion,
            "geometry": geometry,
        }
        _append_event(
            log_path, "profile_completed", profile=config.profile,
            point_count=refusion["point_count"],
        )

    baseline = arms["off"]
    for profile, row in arms.items():
        if profile == "off":
            continue
        acceptance = _acceptance(
            baseline["geometry"], row["geometry"], row["refusion"],
        )
        row["acceptance"] = acceptance
        _write_json(output_dir / profile / "acceptance.json", acceptance)
        view = render_fixed_comparison_views(
            Path(baseline["refusion"]["cloud"]),
            Path(row["refusion"]["cloud"]),
            output_dir / profile / "off_candidate_fixed_views.png",
        )
        row["fixed_views"] = view

    if include_sor_diagnostic and "bilateral_light_v1" in arms:
        profile = "bilateral_light_v1+SOR"
        source = arms["bilateral_light_v1"]
        sor = _sor_diagnostic(
            Path(source["refusion"]["cloud"]), output_dir / "sor_diagnostic",
        )
        geometry = ply_geometry_metrics(Path(sor["cloud"]))
        _write_json(
            output_dir / "sor_diagnostic" / "geometry_metrics.json", geometry,
        )
        surrogate_refusion = dict(source["refusion"])
        surrogate_refusion["point_count"] = sor["output_point_count"]
        acceptance = _acceptance(
            baseline["geometry"], geometry, surrogate_refusion,
        )
        _write_json(
            output_dir / "sor_diagnostic" / "acceptance.json", acceptance,
        )
        view = render_fixed_comparison_views(
            Path(baseline["refusion"]["cloud"]), Path(sor["cloud"]),
            output_dir / "sor_diagnostic" / "off_candidate_fixed_views.png",
        )
        arms[profile] = {
            "refusion": surrogate_refusion,
            "sor": sor,
            "geometry": geometry,
            "acceptance": acceptance,
            "fixed_views": view,
            "diagnostic_only": True,
        }

    passing = [
        profile for profile in TIE_BREAK_ORDER
        if profile in arms
        and arms[profile].get("acceptance", {}).get("passes_pilot")
    ]
    summary = {
        "schema": "depth_denoise_fixed_trajectory_ab.v1",
        "status": "completed",
        "sequence_id": manifest.sequence_id,
        "profiles": list(profiles),
        "fixed_trajectory": True,
        "tsdf": {
            "voxel_length_m": 0.02,
            "sdf_trunc_m": 0.08,
            "depth_trunc_m": 4.50,
        },
        "arms": arms,
        "passing_profiles_in_tie_break_order": passing,
        "pilot_preferred_profile": passing[0] if passing else None,
        "production_default_changed": False,
        "production_default": "off",
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    _write_json(output_dir / "ab_summary.json", summary)
    _append_event(
        log_path, "completed",
        pilot_preferred_profile=summary["pilot_preferred_profile"],
    )
    write_artifact_sha256_manifest(output_dir)
    return summary


def _parse_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(profiles) - set(DEPTH_FILTER_PROFILES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown profiles: {unknown}")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--profiles", type=_parse_profiles,
        default=_parse_profiles(
            "off,range_v1,bilateral_light_v1,bilateral_medium_v1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-sor-diagnostic", action="store_true",
        help="skip the final-PLY-only SOR diagnostic arm",
    )
    args = parser.parse_args()
    result = run_ab(
        manifest_path=args.manifest,
        trajectory_path=args.trajectory,
        profiles=args.profiles,
        output_dir=args.output,
        include_sor_diagnostic=not args.no_sor_diagnostic,
    )
    print(json.dumps({
        "status": result["status"],
        "sequence_id": result["sequence_id"],
        "pilot_preferred_profile": result["pilot_preferred_profile"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
