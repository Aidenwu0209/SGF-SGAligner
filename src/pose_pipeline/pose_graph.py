"""Robust SE(3) pose graph and complete all-frame correction propagation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import PoseRecord, validate_se3


POSE_GRAPH_SCHEMA = "pose_graph_result.v1"


@dataclass(frozen=True)
class PoseGraphEdge:
    source: int
    target: int
    source_to_target: np.ndarray
    kind: str
    weight: float = 1.0
    provenance: str = ""


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


def optimize_pose_graph(
    initial_world_camera: Sequence[np.ndarray],
    loop_edges: Sequence[PoseGraphEdge],
    *,
    translation_sigma_m: float = 0.04,
    rotation_sigma_deg: float = 2.0,
    maximum_loop_degree: int = 2,
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
        values = [initial[0]]
        for index in range(1, len(initial)):
            delta = parameters[(index - 1) * 6:index * 6]
            values.append(_exp_se3(delta) @ initial[index])
        return values

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
    result = least_squares(
        residual, zero, loss="huber", f_scale=2.0, max_nfev=250,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )
    optimized = [validate_se3(value) for value in poses(result.x)]
    after = residual(result.x)
    corrections = [
        _log_se3(optimized[index] @ np.linalg.inv(initial[index]))
        for index in range(len(initial))
    ]
    success = bool(
        result.success
        and np.isfinite(after).all()
        and np.sqrt(np.mean(after ** 2)) <= np.sqrt(np.mean(before ** 2)) + 1e-9
    )
    report = {
        "schema": POSE_GRAPH_SCHEMA,
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
        "edges": [{
            "source": edge.source,
            "target": edge.target,
            "kind": edge.kind,
            "weight": edge.weight,
            "provenance": edge.provenance,
            "T_target_source_m": validate_se3(edge.source_to_target).tolist(),
        } for edge in edges],
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


def propagate_anchor_corrections(
    trajectory: Sequence[PoseRecord],
    anchor_ordinals: Sequence[int],
    optimized_anchor_world_camera: Sequence[np.ndarray],
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
    output = []
    interval = 0
    for ordinal, row in enumerate(trajectory):
        while interval + 1 < len(anchor_ordinals) - 1 and ordinal > anchor_ordinals[interval + 1]:
            interval += 1
        if ordinal <= anchor_ordinals[0]:
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
