#!/usr/bin/env python3
"""Create-only, same-input A/B audit for fixed and adaptive RGB-D anchors."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_pipeline.contracts import (
    bind_manifest_trajectory,
    load_manifest,
    load_trajectory,
    sha256_file,
)
from pose_pipeline.submaps import (
    AdaptiveAnchorConfig,
    SubmapConfig,
    audit_anchor_schedule,
    select_adaptive_anchor_ordinals,
    select_anchor_ordinals,
)


def _summary(rows: list[dict[str, Any]], anchors: list[int]) -> dict[str, Any]:
    flows = np.asarray([
        row["stats"]["median_flow_px"]
        for row in rows
        if row["stats"]["median_flow_px"] is not None
    ], dtype=np.float64)
    overlaps = np.asarray([
        row["stats"]["in_bounds_fraction"] for row in rows
    ], dtype=np.float64)
    translations = np.asarray([
        row["stats"]["relative_translation_m"] for row in rows
    ], dtype=np.float64)
    rotations = np.asarray([
        row["stats"]["relative_rotation_deg"] for row in rows
    ], dtype=np.float64)
    gaps = np.diff(np.asarray(anchors, dtype=np.int64))
    return {
        "anchor_count": len(anchors),
        "anchors": anchors,
        "interval_gap_min": int(np.min(gaps)),
        "interval_gap_median": float(np.median(gaps)),
        "interval_gap_max": int(np.max(gaps)),
        "median_flow_px_p50": float(np.quantile(flows, 0.50)),
        "median_flow_px_p95": float(np.quantile(flows, 0.95)),
        "median_flow_px_max": float(np.max(flows)),
        "in_bounds_fraction_min": float(np.min(overlaps)),
        "in_bounds_fraction_p05": float(np.quantile(overlaps, 0.05)),
        "relative_translation_m_max": float(np.max(translations)),
        "relative_rotation_deg_max": float(np.max(rotations)),
        "audited_frame_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-stride", type=int, default=80)
    parser.add_argument("--minimum-gap", type=int, default=20)
    parser.add_argument("--maximum-gap", type=int, default=80)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--flow-threshold-px", type=float, default=24.0)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.35)
    parser.add_argument("--translation-threshold-m", type=float, default=0.25)
    parser.add_argument("--rotation-threshold-deg", type=float, default=12.0)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    trajectory, trajectory_payload = load_trajectory(args.trajectory)
    bound = bind_manifest_trajectory(
        manifest, trajectory, allow_manifest_superset=True,
    )
    adaptive_config = AdaptiveAnchorConfig(
        minimum_gap=args.minimum_gap,
        maximum_gap=args.maximum_gap,
        pixel_stride=args.pixel_stride,
        flow_threshold_px=args.flow_threshold_px,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
        translation_threshold_m=args.translation_threshold_m,
        rotation_threshold_deg=args.rotation_threshold_deg,
    )
    submap_config = SubmapConfig(anchor_stride=args.fixed_stride)
    fixed_anchors = select_anchor_ordinals(len(bound), args.fixed_stride)
    adaptive_anchors, selection_evidence = select_adaptive_anchor_ordinals(
        bound, manifest.depth_scale, adaptive_config, submap_config,
    )
    fixed_rows = audit_anchor_schedule(
        bound, fixed_anchors, manifest.depth_scale,
        adaptive_config, submap_config,
    )
    adaptive_rows = audit_anchor_schedule(
        bound, adaptive_anchors, manifest.depth_scale,
        adaptive_config, submap_config,
    )
    fixed = _summary(fixed_rows, fixed_anchors)
    adaptive = _summary(adaptive_rows, adaptive_anchors)
    fixed_p95 = fixed["median_flow_px_p95"]
    adaptive_p95 = adaptive["median_flow_px_p95"]
    fixed_flow_already_zero = fixed_p95 <= 1e-12
    flow_reduction = (
        0.0 if fixed_flow_already_zero
        else 1.0 - adaptive_p95 / fixed_p95
    )
    anchor_multiplier = adaptive["anchor_count"] / fixed["anchor_count"]
    gates = {
        "complete_sorted_endpoints": (
            adaptive_anchors[0] == 0
            and adaptive_anchors[-1] == len(bound) - 1
            and adaptive_anchors == sorted(set(adaptive_anchors))
        ),
        "maximum_gap_bounded": adaptive["interval_gap_max"] <= args.maximum_gap,
        "p95_flow_reduced_20pct_or_already_zero": (
            adaptive_p95 <= 1e-12 if fixed_flow_already_zero
            else flow_reduction >= 0.20
        ),
        "p05_overlap_not_worse_by_more_than_5pct": (
            adaptive["in_bounds_fraction_p05"]
            >= fixed["in_bounds_fraction_p05"] - 0.05
        ),
        "anchor_count_at_most_4x_fixed": anchor_multiplier <= 4.0,
    }
    passes = all(gates.values())
    payload = {
        "schema": "adaptive_anchor_ab.v1",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(bound),
        "matrix_convention": "T_world_camera_m",
        "input": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "trajectory": str(args.trajectory.resolve()),
            "trajectory_sha256": sha256_file(args.trajectory),
            "trajectory_payload_sha256": trajectory_payload["payload_sha256"],
        },
        "fixed_config": asdict(submap_config),
        "adaptive_config": asdict(adaptive_config),
        "fixed": fixed,
        "adaptive": adaptive,
        "comparison": {
            "p95_median_flow_reduction_fraction": flow_reduction,
            "p05_overlap_delta": (
                adaptive["in_bounds_fraction_p05"]
                - fixed["in_bounds_fraction_p05"]
            ),
            "anchor_count_multiplier": anchor_multiplier,
            "gates": gates,
            "passes_anchor_schedule_gate": passes,
        },
        "adaptive_selection_evidence": selection_evidence,
        "verdict": (
            "candidate_for_sparse_backend_ab" if passes
            else "reject_or_retune_before_sparse_backend"
        ),
        "final_geometry_claim": "not_evaluated_by_anchor_schedule_audit",
        "identity_fallback_used": False,
        "gt_consumed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "fixed_anchor_count": fixed["anchor_count"],
        "adaptive_anchor_count": adaptive["anchor_count"],
        "p95_median_flow_reduction_fraction": flow_reduction,
        "passes_anchor_schedule_gate": passes,
        "verdict": payload["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
