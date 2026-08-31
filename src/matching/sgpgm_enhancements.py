"""Opt-in SG-PGM-inspired matching enhancements.

The production default remains the historical official top-3 matcher.  This
module contains the three independently testable additions promoted by the
selection89/FIXED4 shadow ablation:

* balanced Sinkhorn scores followed by a learned partial one-to-one budget;
* rigid-invariant object geometry fused with the scene-graph embedding score;
* scene-graph priors used to re-rank cached/dense point correspondences.

The geometry path is deliberately named ``p2sg_lite``.  It does not claim to
be SG-PGM's KPConv pooling network; doing that faithfully requires training a
new checkpoint.  All frozen calibration constants below were fit on the 85
selection89 development pairs with FIXED4 excluded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


CALIBRATION_ID = "selection89-dev85-fixed4-excluded-20260831"

# Ridge model for matched-object fraction.  Feature order is documented in
# ``assignment_features``.  GT anchors were development-only labels; inference
# reads embeddings and point clouds only.
ADAPTIVE_K_WEIGHTS = np.asarray([
    0.33918627171339694,
    -0.09910162374335424,
    0.16083780825288274,
    -0.0026449210407624916,
    -0.39853245508448365,
    0.1974853689109799,
    -0.5459786045377532,
    0.21067244927213977,
    -0.03099165408283802,
    0.5617020589547764,
], dtype=np.float64)

# Development-only statistics for the 13-D rigid-invariant geometry signature.
GEOMETRY_MEAN = np.asarray([
    0.2112330542894722, 0.11010276021724245, 0.009425096036059897,
    0.08908034047150303, 0.14167062879312126, 0.212431411252425,
    0.28858108510633196, 0.35417863065384536, 0.09913058467242153,
    0.1682364824577604, 0.275303771008473, 0.40801737688473433,
    0.5324028935686054,
], dtype=np.float64)
GEOMETRY_STD = np.asarray([
    0.16073428791820596, 0.08738225413605005, 0.014608818150332794,
    0.06853186095732525, 0.10770327432322058, 0.16014519259511867,
    0.21700451192914236, 0.2640851861128105, 0.07721946195151462,
    0.12641807161008556, 0.20533257541570693, 0.307958480960051,
    0.4012149352776762,
], dtype=np.float64)


@dataclass(frozen=True)
class SGPGMEnhancementConfig:
    """Inference-only enhancement configuration.

    ``official_top3`` with zero fusion/rescore values is byte-compatible with
    the existing path.  ``validated_preset`` is provided by the CLI rather
    than silently enabled here.
    """

    matching_policy: str = "official_top3"
    sinkhorn_temperature: float = 0.08
    sinkhorn_iterations: int = 30
    geometry_fusion_alpha: float = 0.0
    graph_rescore_beta: float = 0.0
    min_matches: int = 1

    def validate(self) -> None:
        if self.matching_policy not in {"official_top3", "sinkhorn_partial"}:
            raise ValueError("matching_policy must be official_top3 or sinkhorn_partial")
        if not (0.0 <= self.geometry_fusion_alpha <= 1.0):
            raise ValueError("geometry_fusion_alpha must be in [0, 1]")
        if self.geometry_fusion_alpha and self.matching_policy != "sinkhorn_partial":
            raise ValueError("geometry fusion requires sinkhorn_partial matching")
        if self.sinkhorn_temperature <= 0 or self.sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn temperature/iterations must be positive")
        if self.graph_rescore_beta < 0:
            raise ValueError("graph_rescore_beta must be non-negative")
        if self.min_matches < 1:
            raise ValueError("min_matches must be positive")

    def provenance(self) -> dict:
        return {
            **asdict(self),
            "calibration_id": CALIBRATION_ID,
            "gt_at_inference": False,
            "geometry_scope": "p2sg_lite_rigid_invariant_descriptor",
            "production_default_enabled": False,
        }


@dataclass(frozen=True)
class EnhancedMatchResult:
    node_corrs: list[tuple[int, int]]
    node_scores: dict[tuple[int, int], float]
    diagnostics: dict


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_similarity_matrix(embedding: np.ndarray, src_count: int) -> np.ndarray:
    normalized = _normalize_rows(embedding)
    return normalized[:src_count] @ normalized[src_count:].T


def log_sinkhorn(
    similarity: np.ndarray,
    temperature: float = 0.08,
    iterations: int = 30,
) -> np.ndarray:
    """Return deterministic balanced assignment scores."""
    logits = np.asarray(similarity, dtype=np.float64) / temperature
    logits -= float(np.max(logits))
    matrix = np.exp(np.clip(logits, -60.0, 0.0)) + 1e-12
    for _ in range(iterations):
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
        matrix /= np.maximum(matrix.sum(axis=0, keepdims=True), 1e-12)
    return matrix


def assignment_features(similarity: np.ndarray, sinkhorn: np.ndarray) -> np.ndarray:
    """Features used by the frozen adaptive correspondence-budget model."""
    rows, cols = similarity.shape
    ordered = np.sort(similarity, axis=1)
    best = ordered[:, -1]
    margin = best - (ordered[:, -2] if cols > 1 else 0.0)
    entropy = -(sinkhorn * np.log(np.maximum(sinkhorn, 1e-12))).sum(axis=1)
    row_best = np.argmax(similarity, axis=1)
    col_best = np.argmax(similarity, axis=0)
    mutual_fraction = np.mean([
        col_best[reference] == source
        for source, reference in enumerate(row_best)
    ])
    return np.asarray([
        1.0,
        math.log1p(rows),
        math.log1p(cols),
        min(rows, cols) / max(rows, cols),
        float(np.mean(best)),
        float(np.std(best)),
        float(np.mean(margin)),
        float(np.median(margin)),
        float(np.mean(entropy)),
        float(mutual_fraction),
    ], dtype=np.float64)


def object_geometry_descriptors(object_points: np.ndarray) -> np.ndarray:
    """Build 13-D translation/rotation-invariant descriptors per object."""
    object_points = np.asarray(object_points, dtype=np.float64)
    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError("object_points must have shape [objects, points, 3]")
    descriptors = []
    for cloud in object_points:
        centered = cloud - cloud.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigen_scale = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        radius = np.linalg.norm(centered, axis=1)
        radial_quantiles = np.quantile(radius, [0.1, 0.25, 0.5, 0.75, 0.9])
        paired_distance = np.linalg.norm(centered[::2] - centered[1::2], axis=1)
        pair_quantiles = np.quantile(
            paired_distance, [0.1, 0.25, 0.5, 0.75, 0.9]
        )
        descriptors.append(np.concatenate([
            eigen_scale, radial_quantiles, pair_quantiles,
        ]))
    return np.asarray(descriptors, dtype=np.float64)


def geometry_similarity_matrix(object_points: np.ndarray, src_count: int) -> np.ndarray:
    descriptors = object_geometry_descriptors(object_points)
    normalized = (descriptors - GEOMETRY_MEAN) / np.maximum(GEOMETRY_STD, 1e-12)
    src = normalized[:src_count]
    ref = normalized[src_count:]
    mean_squared_distance = ((src[:, None, :] - ref[None, :, :]) ** 2).mean(axis=2)
    return np.exp(-0.5 * mean_squared_distance)


def enhance_node_matching(
    embedding: np.ndarray,
    object_points: np.ndarray,
    src_count: int,
    config: SGPGMEnhancementConfig,
) -> EnhancedMatchResult:
    """Run partial assignment and optional P2SG-lite score fusion."""
    config.validate()
    if config.matching_policy != "sinkhorn_partial":
        raise ValueError("enhance_node_matching requires sinkhorn_partial")
    if not (0 < src_count < len(embedding)):
        raise ValueError("src_count must split two nonempty graphs")
    graph_similarity = cosine_similarity_matrix(embedding, src_count)
    fused_similarity = graph_similarity
    if config.geometry_fusion_alpha:
        geometry_similarity = geometry_similarity_matrix(object_points, src_count)
        alpha = config.geometry_fusion_alpha
        fused_similarity = (
            (1.0 - alpha) * graph_similarity + alpha * geometry_similarity
        )
    sinkhorn = log_sinkhorn(
        fused_similarity,
        config.sinkhorn_temperature,
        config.sinkhorn_iterations,
    )
    features = assignment_features(graph_similarity, log_sinkhorn(
        graph_similarity,
        config.sinkhorn_temperature,
        config.sinkhorn_iterations,
    ))
    max_matches = min(fused_similarity.shape)
    predicted_fraction = float(np.clip(
        features @ ADAPTIVE_K_WEIGHTS,
        config.min_matches / max_matches,
        1.0,
    ))
    predicted_matches = int(np.clip(
        round(predicted_fraction * max_matches),
        config.min_matches,
        max_matches,
    ))
    rows, cols = linear_sum_assignment(-sinkhorn)
    selected = np.argsort(sinkhorn[rows, cols])[::-1][:predicted_matches]
    node_corrs = [
        (int(rows[index]), int(cols[index] + src_count))
        for index in selected
    ]
    node_scores = {
        pair: float(fused_similarity[pair[0], pair[1] - src_count])
        for pair in node_corrs
    }
    return EnhancedMatchResult(
        node_corrs=node_corrs,
        node_scores=node_scores,
        diagnostics={
            "policy": config.matching_policy,
            "predicted_fraction": predicted_fraction,
            "predicted_matches": predicted_matches,
            "max_one_to_one_matches": max_matches,
            "one_to_one": True,
            "geometry_fusion_enabled": bool(config.geometry_fusion_alpha),
            "calibration_id": CALIBRATION_ID,
        },
    )


def rescore_point_correspondences(
    raw_scores: Sequence[np.ndarray],
    node_pairs: Sequence[tuple[int, int]],
    node_scores: Mapping[tuple[int, int], float],
    beta: float,
) -> list[np.ndarray]:
    """Apply a standardized scene-graph prior to per-node point scores."""
    if len(raw_scores) != len(node_pairs):
        raise ValueError("raw_scores and node_pairs must have equal length")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if not raw_scores:
        return []
    priors = np.asarray([
        float(node_scores.get(tuple(pair), 0.0)) for pair in node_pairs
    ], dtype=np.float64)
    prior_mean = float(priors.mean())
    prior_std = float(priors.std())
    if prior_std < 1e-8:
        prior_std = 1.0
    output = []
    for scores, prior in zip(raw_scores, priors):
        values = np.asarray(scores, dtype=np.float64)
        normalized = values / max(float(np.max(values)), 1e-12)
        output.append(normalized * math.exp(
            beta * (float(prior) - prior_mean) / prior_std
        ))
    return output
