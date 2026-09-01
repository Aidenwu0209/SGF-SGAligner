"""Pre-registered fine-tuning and promotion gates for replacement models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import stable_json_sha256


MODEL_COMPARISON_SCHEMA = "model_comparison_summary.v1"


def audit_split_isolation(
    *,
    training_ids: Sequence[str],
    held_out_scannet_ids: Sequence[str],
    validation_3rscan_ids: Sequence[str],
    orbbec_ids: Sequence[str],
) -> dict:
    training = set(training_ids)
    forbidden = (
        set(held_out_scannet_ids) | set(validation_3rscan_ids) | set(orbbec_ids)
    )
    leaked = sorted(training & forbidden)
    return {
        "passed": not leaked,
        "leaked_ids": leaked,
        "training_id_count": len(training),
        "forbidden_id_count": len(forbidden),
    }


def fine_tune_eligibility(development_rows: Sequence[Mapping[str, object]]) -> dict:
    """Apply the frozen 5-scene zero-shot gate before any fine-tuning."""
    if len(development_rows) != 5:
        raise ValueError("fine-tune gate requires exactly five development scenes")
    low_coverage = [str(row["sequence_id"]) for row in development_rows if float(row["coverage"]) < 0.80]
    scale_failures = [str(row["sequence_id"]) for row in development_rows if bool(row.get("scale_jump", False))]
    catastrophic = [str(row["sequence_id"]) for row in development_rows if bool(row.get("catastrophic_trajectory", False))]
    improved = [str(row["sequence_id"]) for row in development_rows if bool(row.get("primary_metric_improved", False))]
    eligible = not low_coverage and not scale_failures and not catastrophic and len(improved) >= 3
    return {
        "eligible": eligible,
        "minimum_coverage": 0.80,
        "low_coverage_scenes": low_coverage,
        "scale_jump_scenes": scale_failures,
        "catastrophic_scenes": catastrophic,
        "improved_scene_count": len(improved),
        "required_improved_scene_count": 3,
        "decision": "fine_tune_allowed" if eligible else "official_weights_only",
    }


def promotion_eligibility(summary: Mapping[str, object]) -> dict:
    """Apply the full held-out online promotion gate without choosing a winner."""
    required = {
        "full_refusion_complete", "pose_coverage_loss", "catastrophic_edges",
        "unevaluable_accepted_edges", "scannet_metric_pose_improvement",
        "scannet_geometry_improvement", "scannet_joint_ci_lower",
        "orbbec_improved_sequences", "orbbec_total_sequences",
        "orbbec_max_deterioration",
    }
    missing = sorted(required - set(summary))
    if missing:
        raise ValueError(f"promotion summary misses fields: {missing}")
    checks = {
        "full_refusion_complete": bool(summary["full_refusion_complete"]),
        "pose_coverage_loss_lte_1pct": float(summary["pose_coverage_loss"]) <= 0.01,
        "zero_catastrophic_edges": int(summary["catastrophic_edges"]) == 0,
        "zero_unevaluable_accepted_edges": int(summary["unevaluable_accepted_edges"]) == 0,
        "scannet_metric_pose_improvement_gte_10pct": float(summary["scannet_metric_pose_improvement"]) >= 0.10,
        "scannet_geometry_improvement_gte_10pct": float(summary["scannet_geometry_improvement"]) >= 0.10,
        "scannet_joint_ci_positive": float(summary["scannet_joint_ci_lower"]) > 0.0,
        "orbbec_at_least_4_of_5_improved": (
            int(summary["orbbec_total_sequences"]) == 5
            and int(summary["orbbec_improved_sequences"]) >= 4
        ),
        "orbbec_no_sequence_worse_than_10pct": float(summary["orbbec_max_deterioration"]) <= 0.10,
    }
    eligible = all(checks.values())
    return {
        "eligible_for_opt_in_online_integration": eligible,
        "checks": checks,
        "decision": "user_selection_required" if eligible else "research_control_only",
        "auto_selected": False,
    }


def _quality_score(row: Mapping[str, object]) -> float:
    value = float(row.get("quality_score", float("nan")))
    return value if np.isfinite(value) else float("-inf")


def write_model_comparison(
    path: Path,
    *,
    candidates: Sequence[Mapping[str, object]],
    frozen_baseline_commit: str,
) -> dict:
    roles = {
        "continuous_pose_frontend": [],
        "global_or_local_revision": [],
        "offline_or_presentation": [],
    }
    normalized = []
    for raw in candidates:
        row = dict(raw)
        role = str(row.get("role"))
        if role not in roles:
            raise ValueError(f"unsupported model role: {role}")
        if not row.get("model") or "quality_score" not in row:
            raise ValueError("candidate needs model and quality_score")
        score = _quality_score(row)
        if not np.isfinite(score):
            raise ValueError("quality score must be finite")
        row["promotion"] = (
            promotion_eligibility(row["promotion_summary"])
            if row.get("promotion_summary") is not None
            else {
                "eligible_for_opt_in_online_integration": False,
                "decision": "not_applicable_to_online_pose",
                "auto_selected": False,
            }
        )
        roles[role].append(row)
        normalized.append(row)
    rankings = {
        role: [row["model"] for row in sorted(rows, key=lambda value: (-_quality_score(value), str(value["model"])))]
        for role, rows in roles.items()
    }
    unsigned = {
        "schema": MODEL_COMPARISON_SCHEMA,
        "frozen_baseline_commit": frozen_baseline_commit,
        "candidates": normalized,
        "role_rankings": rankings,
        "speed_is_reported_not_gated": True,
        "winner": None,
        "winner_selection_policy": "user_reviews_metrics_and_fixed_view_PLY_then_selects",
    }
    payload = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload
