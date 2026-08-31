"""RegistrationDecision — migrated safety gate (project adaptation).

Migrated from the legacy project's ``inseg_sgaligner.registration_decision``
(Phase 7, deterministic AND-rule, GT-free whitelisted features).  The
official migration reuses the same frozen rule so accepted transforms
remain reconstruction-safe; this copy is byte-equivalent in behaviour
with the legacy module at git 7ed954b and keeps the legacy project
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_KEYS = (
    "ransac_inliers", "ransac_inlier_ratio", "spatial_extent_m",
    "spatial_second_axis_m", "icp_update_translation_m",
    "icp_update_rotation_deg", "bidirectional_rotation_deg",
    "bidirectional_translation_m", "overlap_ratio", "icp_converged",
)
REQUIRED_FEATURE_KEYS = (
    "ransac_inliers", "spatial_extent_m", "spatial_second_axis_m",
    "icp_update_translation_m", "icp_update_rotation_deg",
    "bidirectional_rotation_deg", "bidirectional_translation_m",
    "overlap_ratio",
)
RULE_FIELDS = (
    "min_inliers", "min_extent", "min_second_axis",
    "max_icp_update_translation", "max_icp_update_rotation",
    "max_bidirectional_rotation", "max_bidirectional_translation",
    "min_overlap",
)
RULE_VERSION = "v1-legacy-7ed954b"

FROZEN_RULE = {
    "min_inliers": 6,
    "min_extent": 2.0,
    "min_second_axis": 0.1,
    "max_icp_update_translation": 0.2,
    "max_icp_update_rotation": 10,
    "max_bidirectional_rotation": 5,
    "max_bidirectional_translation": 0.2,
    "min_overlap": 0.1,
}


def _rule_violations(features: dict, rule: dict) -> list[str]:
    violations = []
    if features["ransac_inliers"] < rule["min_inliers"]:
        violations.append(
            f"ransac_inliers {features['ransac_inliers']} < {rule['min_inliers']}"
        )
    if features["spatial_extent_m"] < rule["min_extent"]:
        violations.append(
            f"spatial extent {features['spatial_extent_m']:.2f}m < "
            f"{rule['min_extent']}m"
        )
    if features["spatial_second_axis_m"] < rule["min_second_axis"]:
        violations.append(
            f"second axis {features['spatial_second_axis_m']:.2f}m < "
            f"{rule['min_second_axis']}m"
        )
    if features["icp_update_translation_m"] > rule["max_icp_update_translation"]:
        violations.append("icp translation update too large")
    if features["icp_update_rotation_deg"] > rule["max_icp_update_rotation"]:
        violations.append("icp rotation update too large")
    if features["bidirectional_rotation_deg"] > rule["max_bidirectional_rotation"]:
        violations.append("bidirectional rotation inconsistent")
    if features["bidirectional_translation_m"] > rule["max_bidirectional_translation"]:
        violations.append("bidirectional translation inconsistent")
    if features["overlap_ratio"] < rule["min_overlap"]:
        violations.append(
            f"overlap {features['overlap_ratio']:.2f} < {rule['min_overlap']}"
        )
    return violations


def evaluate_registration_decision(
    features: dict, rule: dict = FROZEN_RULE
) -> dict:
    """Evaluate the frozen rule; reject non-whitelisted (GT) keys."""
    unknown = sorted(set(features) - set(FEATURE_KEYS))
    if unknown:
        raise ValueError(
            f"registration decision received non-whitelisted features: "
            f"{unknown}; ground-truth fields must never enter the decision"
        )
    gt_keys = sorted(key for key in features if key.startswith("gt_"))
    if gt_keys:
        raise ValueError(f"ground-truth fields leaked into decision: {gt_keys}")
    missing = [k for k in REQUIRED_FEATURE_KEYS if features.get(k) is None]
    if missing:
        raise ValueError(f"decision features missing values: {missing}")
    missing_rule = [f for f in RULE_FIELDS if f not in rule]
    if missing_rule:
        raise ValueError(f"rule missing fields: {missing_rule}")
    violations = _rule_violations(features, rule)
    accepted = not violations
    return {
        "status": "accepted" if accepted else "rejected",
        "usable_for_reconstruction": accepted,
        "rejection_reasons": [] if accepted else violations,
        "rule_version": RULE_VERSION,
        "features": {key: features.get(key) for key in FEATURE_KEYS},
    }


def rotation_angle_deg(rotation) -> float:
    trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace)))


def spatial_support(centroids: np.ndarray):
    if len(centroids) < 3:
        return 0.0, 0.0
    centered = centroids - centroids.mean(axis=0)
    singulars = np.linalg.svd(centered, compute_uv=False)
    extent = float(singulars.max()) if singulars.max() > 0 else 0.0
    second = float(np.sort(singulars)[-2]) if len(singulars) >= 2 else 0.0
    return extent, second


def write_decision_files(
    output_dir: str | Path,
    decision: dict,
    transform: np.ndarray | None,
) -> Path:
    """Only accepted pairs receive transform.txt (fail-closed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registration_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    if decision["usable_for_reconstruction"]:
        if transform is None:
            raise ValueError(
                "accepted decision requires a transform (fail-closed)"
            )
        np.savetxt(output_dir / "transform.txt", transform, fmt="%.10f")
    return output_dir
