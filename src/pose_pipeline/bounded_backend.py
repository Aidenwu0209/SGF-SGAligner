"""GT-free, bounded Huber correction of a complete frontend trajectory.

Weights and the loop-degree selection are frozen before influence testing.
Correction backtracking scales every anchor correction by the same factor;
it never clips individual frames.  An unsuccessful result returns the input
records, and callers must retain their original serialized trajectory bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from .contracts import PoseRecord, validate_se3
from .pose_graph import (
    CorrectionAuditConfig,
    PoseGraphEdge,
    PoseGraphOptimizationConfig,
    audit_corrected_trajectory,
    interpolate_transform,
    optimize_pose_graph,
    propagate_anchor_corrections,
    sparsify_loop_edges,
)


def _positive_finite(value: object, name: str) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BoundedBackendConfig:
    enabled: bool = False
    maximum_loop_degree: int = 2
    maximum_loop_weight: float | None = 1.5
    high_leverage_min_span_fraction: float | None = 0.75
    high_leverage_weight_cap: float = 1.0
    correction_backtracking_scales: tuple[float, ...] = (
        1.0, 0.5, 0.25, 0.125, 0.0625,
    )
    enforce_leave_one_out: bool = False
    maximum_leave_one_out_translation_m: float = 0.10
    maximum_leave_one_out_rotation_deg: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.enforce_leave_one_out, bool):
            raise ValueError("backend switches must be booleans")
        _positive_integer(self.maximum_loop_degree, "maximum_loop_degree")
        if self.maximum_loop_weight is not None:
            _positive_finite(self.maximum_loop_weight, "maximum_loop_weight")
        threshold = self.high_leverage_min_span_fraction
        if threshold is not None:
            _positive_finite(threshold, "high_leverage_min_span_fraction")
            if threshold > 1:
                raise ValueError("high_leverage_min_span_fraction must be in (0, 1]")
        _positive_finite(self.high_leverage_weight_cap, "high_leverage_weight_cap")
        _positive_finite(self.maximum_leave_one_out_translation_m, "maximum_leave_one_out_translation_m")
        _positive_finite(self.maximum_leave_one_out_rotation_deg, "maximum_leave_one_out_rotation_deg")
        scales = self.correction_backtracking_scales
        if not isinstance(scales, tuple) or not scales:
            raise ValueError("correction_backtracking_scales must be a nonempty tuple")
        for value in scales:
            _positive_finite(value, "correction backtracking scale")
            if value > 1:
                raise ValueError("correction backtracking scales must be in (0, 1]")
        if scales[0] != 1.0 or any(left <= right for left, right in zip(scales, scales[1:])):
            raise ValueError("correction backtracking scales must start at 1 and strictly decrease")


def _validate_configs(
    config: BoundedBackendConfig,
    optimization: PoseGraphOptimizationConfig,
    correction: CorrectionAuditConfig,
) -> None:
    # Validate all fields, including dormant ones: NaN must never silently
    # disable a comparison in an inference gate or enter a serialized report.
    config.__post_init__()
    optimization.__post_init__()
    correction.__post_init__()
    if optimization.robustifier != "huber":
        raise ValueError("bounded trajectory backend requires the Huber optimizer")
    if correction.propagation != "legacy_slerp_linear":
        raise ValueError("bounded trajectory backend requires legacy_slerp_linear propagation")
    _positive_integer(optimization.max_nfev, "max_nfev")
    _positive_integer(optimization.gnc_iterations, "gnc_iterations")
    _positive_finite(optimization.gnc_initial_mu, "gnc_initial_mu")
    _positive_finite(optimization.gnc_decay, "gnc_decay")
    minimum_weight = optimization.gnc_minimum_weight
    if not isinstance(minimum_weight, Real) or not np.isfinite(minimum_weight) or not 0 <= minimum_weight <= 1:
        raise ValueError("gnc_minimum_weight must be finite and in [0, 1]")
    if not isinstance(optimization.calculate_leave_one_out, bool):
        raise ValueError("calculate_leave_one_out must be a boolean")
    for name, value in asdict(correction).items():
        if name != "propagation" and value is not None:
            _positive_finite(value, name)


def _edge_row(edge: PoseGraphEdge) -> dict:
    return {
        "source": int(edge.source), "target": int(edge.target),
        "kind": edge.kind, "weight": float(edge.weight),
        "provenance": edge.provenance,
    }


def _validate_inputs(
    trajectory: Sequence[PoseRecord], anchors: Sequence[int],
    loop_edges: Sequence[PoseGraphEdge],
) -> None:
    if len(anchors) < 2:
        raise ValueError("bounded trajectory backend requires at least two anchors")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
        or not 0 <= index < len(trajectory) for index in anchors
    ) or list(anchors) != sorted(set(anchors)):
        raise ValueError("anchor ordinals must be sorted unique indices into the trajectory")
    seen = set()
    previous_timestamp = None
    for index, record in enumerate(trajectory):
        validate_se3(record.t_world_camera, f"trajectory frame {index}")
        if not record.valid:
            raise ValueError("bounded trajectory backend requires every frontend pose to be valid")
        if record.frame_id in seen:
            raise ValueError("trajectory frame ids must be unique")
        if previous_timestamp is not None and record.timestamp_us < previous_timestamp:
            raise ValueError("trajectory timestamps must be monotonic")
        seen.add(record.frame_id)
        previous_timestamp = record.timestamp_us
    for edge in loop_edges:
        if any(
            isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
            or not 0 <= index < len(anchors) for index in (edge.source, edge.target)
        ):
            raise ValueError("loop endpoint is outside the anchor range")
        if edge.kind == "odometry":
            raise ValueError("loop inputs must not replace frontend odometry edges")
        validate_se3(edge.source_to_target, "loop transform")
        _positive_finite(edge.weight, "loop weight")
        if not isinstance(edge.confidence, Real) or not np.isfinite(edge.confidence) or not 0 <= edge.confidence <= 1:
            raise ValueError("loop confidence must be finite and in [0, 1]")
        if edge.information is not None:
            information = np.asarray(edge.information, dtype=np.float64)
            if information.shape != (6, 6) or not np.isfinite(information).all():
                raise ValueError("loop information matrix must be finite 6x6")
            if np.linalg.eigvalsh(0.5 * (information + information.T))[0] <= 0:
                raise ValueError("loop information matrix must be positive definite")


def _select_loops(
    loops: Sequence[PoseGraphEdge], anchor_count: int,
    config: BoundedBackendConfig,
) -> tuple[list[PoseGraphEdge], dict]:
    weighted = []
    weights = []
    for edge in loops:
        span = abs(edge.target - edge.source) / float(anchor_count - 1)
        weight = float(edge.weight)
        reasons = []
        if config.enabled:
            if config.maximum_loop_weight is not None and weight > config.maximum_loop_weight:
                weight = float(config.maximum_loop_weight)
                reasons.append("maximum_loop_weight")
            if (
                config.high_leverage_min_span_fraction is not None
                and span >= config.high_leverage_min_span_fraction
                and weight > config.high_leverage_weight_cap
            ):
                weight = float(config.high_leverage_weight_cap)
                reasons.append("high_leverage_weight_cap")
        weighted.append(replace(edge, weight=weight))
        weights.append({
            **_edge_row(edge), "span_fraction": span,
            "input_weight": float(edge.weight), "effective_weight": weight,
            "weight_cap_reasons": reasons,
        })
    selected, rejected = sparsify_loop_edges(
        weighted, maximum_loop_degree=config.maximum_loop_degree,
    )
    return selected, {
        "input_count": len(loops), "selected_count": len(selected),
        "rejected": rejected, "weights": weights,
        "selected_edges": [_edge_row(edge) for edge in selected],
        "maximum_loop_degree": config.maximum_loop_degree,
        "selection_frozen_before_influence": True,
        "weight_semantics": "Huber residual multiplier; confidence/information diagnostic only",
    }


def _solve_and_audit(
    trajectory: Sequence[PoseRecord], anchors: Sequence[int],
    loops: Sequence[PoseGraphEdge], config: BoundedBackendConfig,
    optimization: PoseGraphOptimizationConfig, correction: CorrectionAuditConfig,
) -> tuple[list[PoseRecord], list[PoseRecord] | None, dict]:
    initial = [trajectory[index].t_world_camera for index in anchors]
    report = {
        "success": False, "pose_graph": None, "correction_audit": None,
        "scale_trials": [], "selected_correction_scale": None,
    }
    try:
        # Internal adaptive-GNC leave-one-out has different semantics.  Every
        # influence solve below invokes this exact Huber path from its zero init.
        optimized, pose_graph = optimize_pose_graph(
            initial, loops, maximum_loop_degree=config.maximum_loop_degree,
            optimization_config=replace(optimization, calculate_leave_one_out=False),
        )
        report["pose_graph"] = pose_graph
        if not pose_graph["success"]:
            report["failure_reason"] = "pose_graph_optimization_failed"
            return list(trajectory), None, report
        raw = propagate_anchor_corrections(
            trajectory, anchors, optimized, propagation=correction.propagation,
        )
        corrections = [
            validate_se3(after @ np.linalg.inv(before))
            for before, after in zip(initial, optimized)
        ]
        scales = config.correction_backtracking_scales if config.enabled else (1.0,)
        for scale in scales:
            if scale == 1.0:
                candidate = raw
            else:
                scaled_anchors = [
                    interpolate_transform(np.eye(4), delta, scale) @ before
                    for delta, before in zip(corrections, initial)
                ]
                candidate = propagate_anchor_corrections(
                    trajectory, anchors, scaled_anchors,
                    propagation=correction.propagation,
                )
            audit = audit_corrected_trajectory(trajectory, candidate, correction)
            report["scale_trials"].append({"scale": float(scale), "audit": audit})
            report["correction_audit"] = audit
            if audit["passes"]:
                report["success"] = True
                report["selected_correction_scale"] = float(scale)
                return candidate, raw, report
        report["failure_reason"] = "no_declared_correction_scale_passed"
        return list(trajectory), raw, report
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        report["failure_reason"] = "pose_graph_or_correction_exception"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return list(trajectory), None, report


def _trajectory_difference(left: Sequence[PoseRecord], right: Sequence[PoseRecord]) -> dict:
    from scipy.spatial.transform import Rotation

    if len(left) != len(right):
        raise ValueError("influence trajectories must cover the same frames")
    translations, rotations = [], []
    for before, after in zip(left, right):
        if (before.frame_id, before.timestamp_us) != (after.frame_id, after.timestamp_us):
            raise ValueError("influence trajectory identity mismatch")
        delta = np.linalg.inv(before.t_world_camera) @ after.t_world_camera
        translations.append(float(np.linalg.norm(delta[:3, 3])))
        rotations.append(float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(delta[:3, :3]).as_rotvec(),
        ))))
    return {
        "maximum_translation_m": max(translations, default=0.0),
        "maximum_rotation_deg": max(rotations, default=0.0),
        "pose_count": len(left),
    }


def optimize_bounded_trajectory(
    trajectory: Sequence[PoseRecord],
    anchor_ordinals: Sequence[int],
    loop_edges: Sequence[PoseGraphEdge],
    *,
    config: BoundedBackendConfig = BoundedBackendConfig(),
    optimization_config: PoseGraphOptimizationConfig = PoseGraphOptimizationConfig(),
    correction_config: CorrectionAuditConfig = CorrectionAuditConfig(),
) -> tuple[list[PoseRecord], dict]:
    """Return a completely audited correction, or the original input records.

    Leave-one-out is optional and deliberately conservative: every omitted
    solve must converge and pass the same complete-trajectory correction
    audit.  Influence is measured on *unscaled* complete trajectories so that
    backtracking cannot hide an unsafe edge.  A rejected round removes all
    failing edges, re-solves and re-checks every retained edge without refilling
    the frozen degree-limited selection.  No GT enters any decision.
    """
    _validate_configs(config, optimization_config, correction_config)
    _validate_inputs(trajectory, anchor_ordinals, loop_edges)
    active, selection = _select_loops(loop_edges, len(anchor_ordinals), config)
    influence = {
        "enabled": bool(config.enabled and config.enforce_leave_one_out),
        "comparison": "unscaled_complete_trajectory",
        "maximum_translation_m": config.maximum_leave_one_out_translation_m,
        "maximum_rotation_deg": config.maximum_leave_one_out_rotation_deg,
        "rounds": [], "rejected_edges": [],
    }
    report = {
        "schema": "bounded_trajectory_backend.v1", "config": asdict(config),
        "loop_selection": selection, "influence": influence,
        "gt_consumed": False, "fallback_used": False,
        "correction_scaling": "single_global_factor_on_all_anchor_corrections",
    }
    while True:
        corrected, raw, solution = _solve_and_audit(
            trajectory, anchor_ordinals, active, config,
            optimization_config, correction_config,
        )
        report.update(solution)
        if not solution["success"] or not influence["enabled"] or not active:
            break
        assert raw is not None
        rejected_indices = set()
        checks = []
        for index, edge in enumerate(active):
            omitted_corrected, omitted_raw, omitted_report = _solve_and_audit(
                trajectory, anchor_ordinals, active[:index] + active[index + 1:],
                config, optimization_config, correction_config,
            )
            check = {
                "omitted_edge": _edge_row(edge),
                "solve_success": bool(omitted_report["success"]),
                "pose_graph": omitted_report["pose_graph"],
                "correction_audit": omitted_report["correction_audit"],
                "selected_correction_scale": omitted_report["selected_correction_scale"],
                "scale_trials": omitted_report["scale_trials"],
            }
            reasons = []
            if not omitted_report["success"] or omitted_raw is None:
                reasons.append("leave_one_out_solve_or_audit_failed")
                check["failure_reason"] = omitted_report.get("failure_reason")
            else:
                difference = _trajectory_difference(raw, omitted_raw)
                check["raw_influence"] = difference
                check["applied_influence"] = _trajectory_difference(corrected, omitted_corrected)
                if difference["maximum_translation_m"] > config.maximum_leave_one_out_translation_m:
                    reasons.append("leave_one_out_translation_limit")
                if difference["maximum_rotation_deg"] > config.maximum_leave_one_out_rotation_deg:
                    reasons.append("leave_one_out_rotation_limit")
            check["passes"] = not reasons
            check["reasons"] = reasons
            checks.append(check)
            if reasons:
                rejected_indices.add(index)
                influence["rejected_edges"].append({**_edge_row(edge), "reasons": reasons})
        influence["rounds"].append({
            "round": len(influence["rounds"]), "loop_count": len(active),
            "base_selected_correction_scale": solution["selected_correction_scale"],
            "checks": checks, "rejected_count": len(rejected_indices),
        })
        if not rejected_indices:
            break
        active = [edge for index, edge in enumerate(active) if index not in rejected_indices]
    selection["retained_count"] = len(active)
    selection["retained_edges"] = [_edge_row(edge) for edge in active]
    report["applied_loop_count"] = len(active) if report["success"] else 0
    report["no_op"] = bool(report["success"] and not active)
    report["requires_byte_rollback"] = not report["success"] or report["no_op"]
    if report["no_op"]:
        corrected = list(trajectory)
        report["correction_audit"] = audit_corrected_trajectory(
            trajectory, corrected, correction_config,
        )
    return (corrected if report["success"] else list(trajectory)), report
