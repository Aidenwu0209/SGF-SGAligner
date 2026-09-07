"""Robust SE(3) pose graph and complete all-frame correction propagation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import PoseRecord, validate_se3


POSE_GRAPH_SCHEMA = "pose_graph_result.v1"
POSE_GRAPH_SCHEMA_V2 = "pose_graph_result.v2"


@dataclass(frozen=True)
class PoseGraphEdge:
    source: int
    target: int
    source_to_target: np.ndarray
    kind: str
    weight: float = 1.0
    provenance: str = ""
    information: np.ndarray | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class PoseGraphOptimizationConfig:
    robustifier: str = "huber"
    gnc_iterations: int = 6
    gnc_initial_mu: float = 32.0
    gnc_decay: float = 0.5
    gnc_minimum_weight: float = 1e-4
    max_nfev: int = 250
    calculate_leave_one_out: bool = False

    def __post_init__(self) -> None:
        if self.robustifier not in {"huber", "adaptive_gnc"}:
            raise ValueError("robustifier must be huber or adaptive_gnc")
        if self.gnc_iterations < 1:
            raise ValueError("GNC iterations must be positive")
        if self.gnc_initial_mu <= 0.0 or not 0.0 < self.gnc_decay < 1.0:
            raise ValueError("invalid GNC schedule")
        if not 0.0 <= self.gnc_minimum_weight <= 1.0:
            raise ValueError("invalid minimum robust weight")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be positive")


@dataclass(frozen=True)
class CorrectionAuditConfig:
    propagation: str = "legacy_slerp_linear"
    maximum_adjacent_correction_translation_m: float = 0.05
    maximum_adjacent_correction_rotation_deg: float = 2.0
    maximum_absolute_correction_translation_m: float | None = None
    maximum_absolute_correction_rotation_deg: float | None = None

    def __post_init__(self) -> None:
        if self.propagation not in {
            "legacy_slerp_linear", "se3_correction_field",
        }:
            raise ValueError("unsupported correction propagation")
        if not np.isfinite(self.maximum_adjacent_correction_translation_m) or self.maximum_adjacent_correction_translation_m <= 0.0:
            raise ValueError("correction translation limit must be positive")
        if not np.isfinite(self.maximum_adjacent_correction_rotation_deg) or self.maximum_adjacent_correction_rotation_deg <= 0.0:
            raise ValueError("correction rotation limit must be positive")
        optional_limits = (
            self.maximum_absolute_correction_translation_m,
            self.maximum_absolute_correction_rotation_deg,
        )
        if any(
            value is not None
            and (not np.isfinite(value) or value <= 0.0)
            for value in optional_limits
        ):
            raise ValueError("absolute correction limits must be finite and positive")


@dataclass(frozen=True)
class LoopWeightConfig:
    """Map verified overlap to a pose-graph weight without changing acceptance."""

    overlap_reference: float = 0.35
    minimum_weight: float = 0.7
    maximum_weight: float = 1.5
    high_leverage_min_span_fraction: float | None = None
    high_leverage_weight_cap: float = 1.5

    def __post_init__(self) -> None:
        values = (
            self.overlap_reference,
            self.minimum_weight,
            self.maximum_weight,
            self.high_leverage_weight_cap,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("loop weight config must be finite")
        if self.overlap_reference <= 0.0:
            raise ValueError("overlap reference must be positive")
        if self.minimum_weight <= 0.0 or self.maximum_weight < self.minimum_weight:
            raise ValueError("loop weight bounds are invalid")
        if self.high_leverage_weight_cap <= 0.0:
            raise ValueError("high-leverage loop weight cap must be positive")
        threshold = self.high_leverage_min_span_fraction
        if threshold is not None and (
            not np.isfinite(threshold) or not 0.0 < threshold <= 1.0
        ):
            raise ValueError("high-leverage span fraction must be in (0, 1]")


def loop_edge_weight(
    overlap: float,
    source: int,
    target: int,
    anchor_count: int,
    config: LoopWeightConfig = LoopWeightConfig(),
) -> float:
    """Return a deterministic weight; an opt-in cap protects full-span loops."""

    if not np.isfinite(overlap) or overlap < 0.0:
        raise ValueError("loop overlap must be finite and non-negative")
    if anchor_count < 2:
        raise ValueError("at least two anchors are required to weight a loop")
    if not (0 <= source < anchor_count and 0 <= target < anchor_count):
        raise ValueError("loop endpoint is outside the anchor range")
    threshold = config.high_leverage_min_span_fraction
    weight = float(np.clip(
        overlap / config.overlap_reference,
        config.minimum_weight,
        config.maximum_weight,
    ))
    span_fraction = abs(target - source) / float(anchor_count - 1)
    if threshold is not None and span_fraction >= threshold:
        weight = min(weight, config.high_leverage_weight_cap)
    return float(weight)


def _rotation_tools():
    from scipy.spatial.transform import Rotation, Slerp
    return Rotation, Slerp


def _exp_se3(value: np.ndarray) -> np.ndarray:
    Rotation, _ = _rotation_tools()
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(value[:3]).as_matrix()
    transform[:3, 3] = value[3:]
    return transform


def _log_se3(transform: np.ndarray) -> np.ndarray:
    Rotation, _ = _rotation_tools()
    transform = validate_se3(transform)
    return np.r_[
        Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
        transform[:3, 3],
    ]


def build_odometry_edges(initial: Sequence[np.ndarray]) -> list[PoseGraphEdge]:
    return [PoseGraphEdge(
        source=index,
        target=index + 1,
        source_to_target=np.linalg.inv(initial[index + 1]) @ initial[index],
        kind="odometry",
        weight=1.0,
        provenance="continuous_frontend",
    ) for index in range(len(initial) - 1)]


def sparsify_loop_edges(
    edges: Sequence[PoseGraphEdge], *, maximum_loop_degree: int = 2,
) -> tuple[list[PoseGraphEdge], list[dict]]:
    if maximum_loop_degree < 1:
        raise ValueError("maximum loop degree must be positive")
    best_by_pair: dict[tuple[int, int], PoseGraphEdge] = {}
    rejected = []
    for edge in edges:
        if edge.source == edge.target:
            rejected.append({"reason": "self_edge", "source": edge.source, "target": edge.target})
            continue
        key = tuple(sorted((edge.source, edge.target)))
        old = best_by_pair.get(key)
        if old is None or (edge.weight, edge.provenance) > (old.weight, old.provenance):
            if old is not None:
                rejected.append({"reason": "duplicate_pair", "source": old.source, "target": old.target})
            best_by_pair[key] = edge
        else:
            rejected.append({"reason": "duplicate_pair", "source": edge.source, "target": edge.target})
    degrees: dict[int, int] = {}
    accepted = []
    for edge in sorted(
        best_by_pair.values(),
        key=lambda row: (-row.weight, min(row.source, row.target), max(row.source, row.target), row.provenance),
    ):
        if degrees.get(edge.source, 0) >= maximum_loop_degree or degrees.get(edge.target, 0) >= maximum_loop_degree:
            rejected.append({"reason": "loop_degree_cap", "source": edge.source, "target": edge.target})
            continue
        accepted.append(edge)
        degrees[edge.source] = degrees.get(edge.source, 0) + 1
        degrees[edge.target] = degrees.get(edge.target, 0) + 1
    return sorted(accepted, key=lambda row: (row.source, row.target)), rejected


def _information_sqrt(edge: PoseGraphEdge) -> tuple[np.ndarray, float, float]:
    if edge.information is None:
        return np.eye(6, dtype=np.float64), 1.0, 1.0
    information = np.asarray(edge.information, dtype=np.float64)
    if information.shape != (6, 6) or not np.isfinite(information).all():
        raise ValueError("pose graph information matrix must be finite 6x6")
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("pose graph information matrix must be positive definite")
    normalized = eigenvalues / max(float(eigenvalues[-1]), 1e-12)
    root = eigenvectors @ np.diag(np.sqrt(normalized)) @ eigenvectors.T
    spectral_confidence = float(np.clip(
        np.exp(np.mean(np.log(np.maximum(normalized, 1e-12)))), 1e-4, 1.0,
    ))
    condition = float(eigenvalues[-1] / eigenvalues[0])
    return root, spectral_confidence, condition


def _pose_values(
    parameters: np.ndarray, initial: Sequence[np.ndarray],
) -> list[np.ndarray]:
    values = [initial[0]]
    for index in range(1, len(initial)):
        delta = parameters[(index - 1) * 6:index * 6]
        values.append(_exp_se3(delta) @ initial[index])
    return values


def _edge_error(
    values: Sequence[np.ndarray], edge: PoseGraphEdge,
    rotation_scale: float, translation_scale: float,
) -> np.ndarray:
    predicted = np.linalg.inv(values[edge.target]) @ values[edge.source]
    error = np.linalg.inv(validate_se3(edge.source_to_target)) @ predicted
    tangent = _log_se3(error)
    return np.r_[
        tangent[:3] / rotation_scale,
        tangent[3:] / translation_scale,
    ]


def _edge_diagnostics(
    parameters: np.ndarray, initial: Sequence[np.ndarray],
    edges: Sequence[PoseGraphEdge], rotation_scale: float,
    translation_scale: float,
) -> list[dict]:
    values = _pose_values(parameters, initial)
    rows = []
    for edge in edges:
        tangent = _edge_error(
            values, edge, rotation_scale, translation_scale,
        )
        root, spectral, condition = _information_sqrt(edge)
        rows.append({
            "normalized_residual": tangent,
            "information_residual": root @ tangent,
            "spectral_confidence": spectral,
            "information_condition_number": condition,
        })
    return rows


def _solve_adaptive_gnc(
    initial: Sequence[np.ndarray], edges: Sequence[PoseGraphEdge],
    config: PoseGraphOptimizationConfig, rotation_scale: float,
    translation_scale: float,
) -> tuple[np.ndarray, object, np.ndarray, list[dict]]:
    from scipy.optimize import least_squares

    dimension = 6 * (len(initial) - 1)
    parameters = np.zeros(dimension, dtype=np.float64)
    robust_weights = np.ones(len(edges), dtype=np.float64)
    history = []

    roots, base_weights = [], []
    for edge in edges:
        root, spectral, _condition = _information_sqrt(edge)
        roots.append(root)
        if edge.kind == "odometry":
            base_weights.append(1.0)
        else:
            base_weights.append(
                max(float(edge.weight), 1e-9)
                * float(np.clip(edge.confidence, 0.0, 1.0))
                * spectral
            )
    base_weights = np.asarray(base_weights, dtype=np.float64)

    def residual(values: np.ndarray, active: np.ndarray) -> np.ndarray:
        poses = _pose_values(values, initial)
        rows = []
        for index, edge in enumerate(edges):
            error = _edge_error(
                poses, edge, rotation_scale, translation_scale,
            )
            scale = math.sqrt(max(base_weights[index] * active[index], 1e-12))
            rows.extend((roots[index] @ error * scale).tolist())
        return np.asarray(rows, dtype=np.float64)

    result = None
    mu = config.gnc_initial_mu
    for iteration in range(config.gnc_iterations):
        result = least_squares(
            lambda value: residual(value, robust_weights), parameters,
            loss="linear", max_nfev=config.max_nfev,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        parameters = result.x
        diagnostics = _edge_diagnostics(
            parameters, initial, edges, rotation_scale, translation_scale,
        )
        loop_norms = np.asarray([
            np.linalg.norm(row["information_residual"])
            for edge, row in zip(edges, diagnostics)
            if edge.kind != "odometry"
        ], dtype=np.float64)
        median = float(np.median(loop_norms)) if len(loop_norms) else 1.0
        mad = (
            float(np.median(np.abs(loop_norms - median)))
            if len(loop_norms) else 0.0
        )
        cutoff = max(2.0, median + 2.5 * max(1.4826 * mad, 1e-6))
        next_weights = np.ones(len(edges), dtype=np.float64)
        for index, (edge, row) in enumerate(zip(edges, diagnostics)):
            if edge.kind == "odometry":
                continue
            squared = float(np.dot(
                row["information_residual"], row["information_residual"],
            ))
            numerator = mu * cutoff * cutoff
            next_weights[index] = max(
                config.gnc_minimum_weight,
                (numerator / (squared + numerator)) ** 2,
            )
        history.append({
            "iteration": iteration,
            "mu": mu,
            "adaptive_cutoff": cutoff,
            "minimum_loop_weight": float(min(
                (next_weights[index] for index, edge in enumerate(edges)
                 if edge.kind != "odometry"), default=1.0,
            )),
        })
        robust_weights = next_weights
        mu *= config.gnc_decay
    assert result is not None
    return parameters, result, robust_weights, history


def optimize_pose_graph(
    initial_world_camera: Sequence[np.ndarray],
    loop_edges: Sequence[PoseGraphEdge],
    *,
    translation_sigma_m: float = 0.04,
    rotation_sigma_deg: float = 2.0,
    maximum_loop_degree: int = 2,
    optimization_config: PoseGraphOptimizationConfig = PoseGraphOptimizationConfig(),
) -> tuple[list[np.ndarray], dict]:
    from scipy.optimize import least_squares

    initial = [validate_se3(value, f"initial node {index}") for index, value in enumerate(initial_world_camera)]
    if len(initial) < 2:
        raise ValueError("pose graph requires at least two nodes")
    sparse_loops, rejected = sparsify_loop_edges(
        loop_edges, maximum_loop_degree=maximum_loop_degree,
    )
    edges = build_odometry_edges(initial) + sparse_loops
    dimension = 6 * (len(initial) - 1)

    def poses(parameters: np.ndarray) -> list[np.ndarray]:
        return _pose_values(parameters, initial)

    rotation_scale = math.radians(rotation_sigma_deg)

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = poses(parameters)
        rows = []
        for edge in edges:
            predicted = np.linalg.inv(values[edge.target]) @ values[edge.source]
            error = np.linalg.inv(validate_se3(edge.source_to_target)) @ predicted
            tangent = _log_se3(error)
            rows.extend((tangent[:3] / rotation_scale * edge.weight).tolist())
            rows.extend((tangent[3:] / translation_sigma_m * edge.weight).tolist())
        return np.asarray(rows, dtype=np.float64)

    zero = np.zeros(dimension, dtype=np.float64)
    before = residual(zero)
    robust_weights = np.ones(len(edges), dtype=np.float64)
    robust_history: list[dict] = []
    if optimization_config.robustifier == "huber":
        result = least_squares(
            residual, zero, loss="huber", f_scale=2.0,
            max_nfev=optimization_config.max_nfev,
            xtol=1e-10, ftol=1e-10, gtol=1e-10,
        )
        parameters = result.x
    else:
        parameters, result, robust_weights, robust_history = _solve_adaptive_gnc(
            initial, edges, optimization_config, rotation_scale,
            translation_sigma_m,
        )
    optimized = [validate_se3(value) for value in poses(parameters)]
    after = residual(parameters)
    corrections = [
        _log_se3(optimized[index] @ np.linalg.inv(initial[index]))
        for index in range(len(initial))
    ]
    success = bool(
        result.success
        and np.isfinite(after).all()
        and np.sqrt(np.mean(after ** 2)) <= np.sqrt(np.mean(before ** 2)) + 1e-9
    )
    before_edges = _edge_diagnostics(
        zero, initial, edges, rotation_scale, translation_sigma_m,
    )
    after_edges = _edge_diagnostics(
        parameters, initial, edges, rotation_scale, translation_sigma_m,
    )
    leave_one_out = [0.0] * len(edges)
    if (
        optimization_config.robustifier == "adaptive_gnc"
        and optimization_config.calculate_leave_one_out
    ):
        for omitted, edge in enumerate(edges):
            if edge.kind == "odometry":
                continue
            retained = [value for index, value in enumerate(edges) if index != omitted]
            retained_weights = np.asarray([
                robust_weights[index] for index in range(len(edges))
                if index != omitted
            ], dtype=np.float64)
            retained_roots = [_information_sqrt(value)[0] for value in retained]
            retained_base = np.asarray([
                1.0 if value.kind == "odometry" else (
                    max(float(value.weight), 1e-9)
                    * float(np.clip(value.confidence, 0.0, 1.0))
                    * _information_sqrt(value)[1]
                ) for value in retained
            ], dtype=np.float64)

            def loo_residual(values: np.ndarray) -> np.ndarray:
                current = poses(values)
                rows = []
                for index, value in enumerate(retained):
                    tangent = _edge_error(
                        current, value, rotation_scale, translation_sigma_m,
                    )
                    factor = math.sqrt(max(
                        retained_base[index] * retained_weights[index], 1e-12,
                    ))
                    rows.extend((retained_roots[index] @ tangent * factor).tolist())
                return np.asarray(rows, dtype=np.float64)

            omitted_result = least_squares(
                loo_residual, parameters, loss="linear",
                max_nfev=min(80, optimization_config.max_nfev),
            )
            omitted_poses = poses(omitted_result.x)
            leave_one_out[omitted] = float(max(
                np.linalg.norm(
                    omitted_poses[index][:3, 3] - optimized[index][:3, 3],
                ) for index in range(len(optimized))
            ))
    report_edges = []
    for index, edge in enumerate(edges):
        report_edges.append({
            "source": edge.source,
            "target": edge.target,
            "kind": edge.kind,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "spectral_confidence": before_edges[index]["spectral_confidence"],
            "information_condition_number": before_edges[index]["information_condition_number"],
            "initial_normalized_residual": float(np.linalg.norm(
                before_edges[index]["normalized_residual"],
            )),
            "final_normalized_residual": float(np.linalg.norm(
                after_edges[index]["normalized_residual"],
            )),
            "final_robust_weight": float(robust_weights[index]),
            "leave_one_edge_out_max_translation_m": leave_one_out[index],
            "provenance": edge.provenance,
            "T_target_source_m": validate_se3(edge.source_to_target).tolist(),
            "information_matrix": (
                None if edge.information is None
                else np.asarray(edge.information, dtype=np.float64).tolist()
            ),
        })
    report = {
        "schema": (
            POSE_GRAPH_SCHEMA_V2
            if optimization_config.robustifier == "adaptive_gnc"
            else POSE_GRAPH_SCHEMA
        ),
        "success": success,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "function_evaluations": int(result.nfev),
        "node_count": len(initial),
        "odometry_edge_count": len(initial) - 1,
        "input_loop_edge_count": len(loop_edges),
        "accepted_loop_edge_count": len(sparse_loops),
        "rejected_loop_edges": rejected,
        "initial_residual_rms": float(np.sqrt(np.mean(before ** 2))),
        "final_residual_rms": float(np.sqrt(np.mean(after ** 2))),
        "maximum_anchor_correction_translation_m": float(max(np.linalg.norm(value[3:]) for value in corrections)),
        "maximum_anchor_correction_rotation_deg": float(max(np.degrees(np.linalg.norm(value[:3])) for value in corrections)),
        "robustifier": optimization_config.robustifier,
        "robust_history": robust_history,
        "edges": report_edges,
        "gt_consumed": False,
        "fallback_used": False,
    }
    return optimized, report


def interpolate_transform(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    Rotation, Slerp = _rotation_tools()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rotations = Rotation.from_matrix(np.stack([left[:3, :3], right[:3, :3]]))
    rotation = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return validate_se3(transform)


def _se3_exp_coupled(value: np.ndarray) -> np.ndarray:
    Rotation, _ = _rotation_tools()
    value = np.asarray(value, dtype=np.float64)
    omega, rho = value[:3], value[3:]
    theta = float(np.linalg.norm(omega))
    skew = np.asarray([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-8:
        v_matrix = np.eye(3) + 0.5 * skew + (skew @ skew) / 6.0
    else:
        v_matrix = (
            np.eye(3)
            + (1.0 - math.cos(theta)) / (theta * theta) * skew
            + (theta - math.sin(theta)) / (theta ** 3) * (skew @ skew)
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    transform[:3, 3] = v_matrix @ rho
    return validate_se3(transform)


def _se3_log_coupled(transform: np.ndarray) -> np.ndarray:
    Rotation, _ = _rotation_tools()
    transform = validate_se3(transform)
    omega = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(omega))
    skew = np.asarray([
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0],
    ])
    if theta < 1e-8:
        inverse_v = np.eye(3) - 0.5 * skew + (skew @ skew) / 12.0
    else:
        coefficient = (
            1.0 / (theta * theta)
            - (1.0 + math.cos(theta))
            / (2.0 * theta * max(math.sin(theta), 1e-12))
        )
        inverse_v = np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)
    return np.r_[omega, inverse_v @ transform[:3, 3]]


def _smooth_correction_field(
    length: int, anchor_ordinals: Sequence[int],
    corrections: Sequence[np.ndarray],
) -> list[np.ndarray]:
    from scipy.interpolate import PchipInterpolator

    anchor_x = np.asarray(anchor_ordinals, dtype=np.float64)
    tangents = np.stack([_se3_log_coupled(value) for value in corrections])
    query = np.arange(length, dtype=np.float64)
    if len(anchor_ordinals) == 1:
        values = np.repeat(tangents, length, axis=0)
    else:
        values = PchipInterpolator(
            anchor_x, tangents, axis=0, extrapolate=False,
        )(np.clip(query, anchor_x[0], anchor_x[-1]))
    output = [_se3_exp_coupled(value) for value in values]
    for ordinal, correction in zip(anchor_ordinals, corrections):
        output[int(ordinal)] = validate_se3(correction)
    return output


def propagate_anchor_corrections(
    trajectory: Sequence[PoseRecord],
    anchor_ordinals: Sequence[int],
    optimized_anchor_world_camera: Sequence[np.ndarray],
    *,
    propagation: str = "legacy_slerp_linear",
) -> list[PoseRecord]:
    if len(anchor_ordinals) != len(optimized_anchor_world_camera):
        raise ValueError("anchor correction count mismatch")
    if not anchor_ordinals or list(anchor_ordinals) != sorted(set(anchor_ordinals)):
        raise ValueError("anchor ordinals must be sorted and unique")
    initial = [trajectory[index].t_world_camera for index in anchor_ordinals]
    corrections = [
        validate_se3(optimized_anchor_world_camera[index]) @ np.linalg.inv(initial[index])
        for index in range(len(anchor_ordinals))
    ]
    if propagation not in {"legacy_slerp_linear", "se3_correction_field"}:
        raise ValueError("unsupported correction propagation")
    smooth = (
        _smooth_correction_field(len(trajectory), anchor_ordinals, corrections)
        if propagation == "se3_correction_field" else None
    )
    output = []
    interval = 0
    for ordinal, row in enumerate(trajectory):
        while interval + 1 < len(anchor_ordinals) - 1 and ordinal > anchor_ordinals[interval + 1]:
            interval += 1
        if smooth is not None:
            correction = smooth[ordinal]
        elif ordinal <= anchor_ordinals[0]:
            correction = corrections[0]
        elif ordinal >= anchor_ordinals[-1]:
            correction = corrections[-1]
        else:
            left, right = anchor_ordinals[interval], anchor_ordinals[interval + 1]
            alpha = (ordinal - left) / max(1, right - left)
            correction = interpolate_transform(corrections[interval], corrections[interval + 1], alpha)
        corrected = validate_se3(correction @ row.t_world_camera)
        output.append(PoseRecord(
            frame_id=row.frame_id,
            timestamp_us=row.timestamp_us,
            t_world_camera=corrected,
            valid=True,
            source="dpv_plus_sparse_pose_graph",
        ))
    if len(output) != len(trajectory):
        raise RuntimeError("all-frame correction propagation lost poses")
    return output


def audit_corrected_trajectory(
    original: Sequence[PoseRecord], corrected: Sequence[PoseRecord],
    config: CorrectionAuditConfig = CorrectionAuditConfig(),
) -> dict:
    same_count = len(original) == len(corrected)
    same_ids = same_count and all(
        left.frame_id == right.frame_id
        and left.timestamp_us == right.timestamp_us
        for left, right in zip(original, corrected)
    )
    finite = same_count and all(
        np.isfinite(row.t_world_camera).all() for row in corrected
    )
    translations, rotations = [], []
    absolute_translations, absolute_rotations = [], []
    if same_count and finite:
        corrections = [
            validate_se3(right.t_world_camera) @ np.linalg.inv(
                validate_se3(left.t_world_camera),
            ) for left, right in zip(original, corrected)
        ]
        for correction in corrections:
            delta = _log_se3(correction)
            absolute_rotations.append(float(np.degrees(np.linalg.norm(delta[:3]))))
            absolute_translations.append(float(np.linalg.norm(delta[3:])))
        for left, right in zip(corrections, corrections[1:]):
            delta = _log_se3(np.linalg.inv(left) @ right)
            rotations.append(float(np.degrees(np.linalg.norm(delta[:3]))))
            translations.append(float(np.linalg.norm(delta[3:])))
    maximum_translation = max(translations, default=0.0)
    maximum_rotation = max(rotations, default=0.0)
    maximum_absolute_translation = max(absolute_translations, default=0.0)
    maximum_absolute_rotation = max(absolute_rotations, default=0.0)
    absolute_translation_limit = (
        config.maximum_absolute_correction_translation_m
    )
    absolute_rotation_limit = config.maximum_absolute_correction_rotation_deg
    gates = {
        "pose_count_preserved": same_count,
        "frame_identity_preserved": same_ids,
        "finite_se3": finite,
        "adjacent_correction_translation_within_limit": (
            maximum_translation
            <= config.maximum_adjacent_correction_translation_m
        ),
        "adjacent_correction_rotation_within_limit": (
            maximum_rotation
            <= config.maximum_adjacent_correction_rotation_deg
        ),
        "absolute_correction_translation_within_limit": (
            absolute_translation_limit is None
            or maximum_absolute_translation <= absolute_translation_limit
        ),
        "absolute_correction_rotation_within_limit": (
            absolute_rotation_limit is None
            or maximum_absolute_rotation <= absolute_rotation_limit
        ),
    }
    return {
        "schema": "correction_audit.v1",
        "propagation": config.propagation,
        "passes": all(gates.values()),
        "gates": gates,
        "pose_count": len(corrected),
        "maximum_adjacent_correction_translation_m": maximum_translation,
        "maximum_adjacent_correction_rotation_deg": maximum_rotation,
        "maximum_absolute_correction_translation_m": maximum_absolute_translation,
        "maximum_absolute_correction_rotation_deg": maximum_absolute_rotation,
        "limits": {
            "maximum_adjacent_correction_translation_m": (
                config.maximum_adjacent_correction_translation_m
            ),
            "maximum_adjacent_correction_rotation_deg": (
                config.maximum_adjacent_correction_rotation_deg
            ),
            "maximum_absolute_correction_translation_m": (
                absolute_translation_limit
            ),
            "maximum_absolute_correction_rotation_deg": absolute_rotation_limit,
        },
        "gt_consumed": False,
    }
