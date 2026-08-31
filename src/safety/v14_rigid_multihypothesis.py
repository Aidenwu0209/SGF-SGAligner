"""Deterministic GT-free rigid-mode separation for V14 research shadow.

The module consumes only the exact three-key ColorPCR correspondence caches.
It emits exact three-key candidate caches for the unchanged V13 dual-solver
stack.  Its Kabsch poses are construction diagnostics and are never solver
initializations or reconstruction decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v13_dual_solver_runtime import (
    array_sha256, load_frozen_correspondences, sha256_file,
    stable_json_sha256, transform_distance,
)


SCHEMA = "v14-direction-rigid-hypotheses-v1"
PAIR_SCHEMA = "v14-bidirectional-candidate-set-v1"
CANDIDATE_SCHEMA = "v14-bidirectional-candidate-v1"
STRICT_SCHEMA = "v13-strict-pair-gate-v1"
STRICT_AUTHORITY = "fixed_trace_icp_plus_unchanged_rule_b_plus_dual_solver_q4"


@dataclass(frozen=True)
class RigidMultiHypothesisConfig:
    min_input_correspondences: int = 40
    max_input_correspondences: int = 1000
    seed_pool_size: int = 64
    max_seed_triangles: int = 256
    compatibility_min_separation_m: float = 0.02
    compatibility_absolute_tolerance_m: float = 0.05
    compatibility_relative_tolerance: float = 0.05
    residual_threshold_m: float = 0.10
    min_hypothesis_correspondences: int = 40
    max_direction_hypotheses: int = 8
    duplicate_jaccard_min: float = 0.80
    transform_cluster_rotation_deg: float = 5.0
    transform_cluster_translation_m: float = 0.10
    max_bidirectional_candidates: int = 8


FROZEN_CONFIG = RigidMultiHypothesisConfig()


class RigidMultiHypothesisError(RuntimeError):
    """Malformed input or a deviation from the frozen V14 protocol."""


def _require_frozen(config: RigidMultiHypothesisConfig) -> None:
    if config != FROZEN_CONFIG:
        raise RigidMultiHypothesisError("V14 candidate config is not frozen")


def _validate_sha(value: str, name: str) -> str:
    value = str(value)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RigidMultiHypothesisError(f"{name} is not lowercase SHA-256")
    return value


def _points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1:] != (3,)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise RigidMultiHypothesisError(f"{name} must be finite floating Nx3")
    return np.ascontiguousarray(array, dtype=np.float64)


def _scores(value: Any, count: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != (count,) or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise RigidMultiHypothesisError("scores must be finite floating N")
    return np.ascontiguousarray(array, dtype=np.float64)


def _proper_kabsch(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(source) < 3:
        raise RigidMultiHypothesisError("Kabsch needs at least three rows")
    src_mean, ref_mean = source.mean(0), reference.mean(0)
    covariance = (source - src_mean).T @ (reference - ref_mean)
    u, singular, vt = np.linalg.svd(covariance)
    if singular[1] <= max(1e-12, singular[0] * 1e-7):
        raise RigidMultiHypothesisError("degenerate Kabsch support")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise RigidMultiHypothesisError("Kabsch did not produce proper SE3")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = ref_mean - rotation @ src_mean
    return transform


def _residual(source: np.ndarray, reference: np.ndarray,
              transform: np.ndarray) -> np.ndarray:
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    return np.linalg.norm(moved - reference, axis=1)


def _compatibility_graph(source: np.ndarray, reference: np.ndarray,
                         config: RigidMultiHypothesisConfig,
                         ) -> tuple[np.ndarray, np.ndarray, str]:
    src_distance = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    ref_distance = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=2)
    tolerance = np.maximum(
        config.compatibility_absolute_tolerance_m,
        config.compatibility_relative_tolerance
        * np.maximum(src_distance, ref_distance),
    )
    graph = (src_distance >= config.compatibility_min_separation_m) \
        & (ref_distance >= config.compatibility_min_separation_m) \
        & (np.abs(src_distance - ref_distance) <= tolerance)
    np.fill_diagonal(graph, False)
    left, right = np.where(np.triu(graph, 1))
    edges = np.ascontiguousarray(np.stack([left, right], axis=1), dtype=np.int32)
    return graph, edges, array_sha256(edges)


def _triangle_hash(*, cache_sha: str, pair_id: str, arm: str,
                   direction: str, config_sha: str,
                   original: Sequence[int]) -> str:
    payload = {
        "arm": arm, "config_sha256": config_sha, "direction": direction,
        "original_indices": [int(value) for value in original],
        "pair_id": pair_id, "source_cache_sha256": cache_sha,
    }
    return stable_json_sha256(payload)


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-int(row["correspondence_count"]),
            -float(row["weighted_support"]),
            float(row["median_residual_m"]),
            tuple(int(value) for value in row["support_original_indices"]),
            tuple(int(value) for value in row["seed_original_indices"]))


def generate_direction_hypotheses(
    source: Any, reference: Any, scores: Any, *, source_cache_sha256: str,
    pair_id: str, arm: str, direction: str,
    original_indices: Any | None = None,
    config: RigidMultiHypothesisConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    """Return a bounded deterministic set of rigid correspondence supports."""
    _require_frozen(config)
    source_cache_sha256 = _validate_sha(source_cache_sha256,
                                        "source_cache_sha256")
    if arm not in ("sgf_selected_union", "fullscan"):
        raise RigidMultiHypothesisError("unknown arm")
    if direction not in ("forward", "reverse"):
        raise RigidMultiHypothesisError("unknown direction")
    source = _points(source, "source")
    reference = _points(reference, "reference")
    if source.shape != reference.shape:
        raise RigidMultiHypothesisError("source/reference are not aligned")
    scores = _scores(scores, len(source))
    if original_indices is None:
        original = np.arange(len(source), dtype=np.int64)
    else:
        original = np.asarray(original_indices)
        if (original.shape != (len(source),)
                or not np.issubdtype(original.dtype, np.integer)
                or len(np.unique(original)) != len(original)
                or (original < 0).any()):
            raise RigidMultiHypothesisError("original indices are malformed")
        original = np.ascontiguousarray(original, dtype=np.int64)
    order = np.lexsort((original, -scores))[:config.max_input_correspondences]
    source, reference, scores, original = (
        np.ascontiguousarray(source[order]),
        np.ascontiguousarray(reference[order]),
        np.ascontiguousarray(scores[order]),
        np.ascontiguousarray(original[order]),
    )
    if len(source) < config.min_input_correspondences:
        raise RigidMultiHypothesisError("at least 40 correspondences are required")
    config_dict = asdict(config)
    config_sha = stable_json_sha256(config_dict)
    graph, edges, edge_sha = _compatibility_graph(source, reference, config)
    seed_count = min(config.seed_pool_size, len(source))
    triangles = []
    for local in combinations(range(seed_count), 3):
        if not (graph[local[0], local[1]] and graph[local[0], local[2]]
                and graph[local[1], local[2]]):
            continue
        raw = tuple(int(original[index]) for index in local)
        triangles.append((_triangle_hash(
            cache_sha=source_cache_sha256, pair_id=pair_id, arm=arm,
            direction=direction, config_sha=config_sha, original=raw), local))
    triangles.sort(key=lambda item: (item[0], item[1]))
    triangles = triangles[:config.max_seed_triangles]
    raw_candidates: list[dict[str, Any]] = []
    rejected = {"degenerate": 0, "insufficient_support": 0}
    for triangle_hash, local in triangles:
        try:
            transform = _proper_kabsch(source[list(local)], reference[list(local)])
        except RigidMultiHypothesisError:
            rejected["degenerate"] += 1
            continue
        distance = _residual(source, reference, transform)
        compatible_with_seed = graph[:, list(local)].sum(axis=1) >= 2
        support = (distance <= config.residual_threshold_m) & compatible_with_seed
        if int(support.sum()) < config.min_hypothesis_correspondences:
            rejected["insufficient_support"] += 1
            continue
        try:
            transform = _proper_kabsch(source[support], reference[support])
        except RigidMultiHypothesisError:
            rejected["degenerate"] += 1
            continue
        distance = _residual(source, reference, transform)
        support = (distance <= config.residual_threshold_m) & compatible_with_seed
        if int(support.sum()) < config.min_hypothesis_correspondences:
            rejected["insufficient_support"] += 1
            continue
        local_indices = np.flatnonzero(support).astype(np.int64)
        original_support = np.sort(original[local_indices]).astype(np.int64)
        inlier_distance = distance[local_indices]
        payload = {
            "correspondence_count": int(len(local_indices)),
            "diagnostic_transform": transform.tolist(),
            "median_residual_m": float(np.median(inlier_distance)),
            "seed_original_indices": [int(original[index]) for index in local],
            "seed_triangle_sha256": triangle_hash,
            "support_local_indices": [int(value) for value in local_indices],
            "support_original_indices": [int(value) for value in original_support],
            "support_original_indices_sha256": array_sha256(original_support),
            "weighted_support": float(scores[local_indices].sum()),
        }
        payload["hypothesis_sha256"] = stable_json_sha256(payload)
        raw_candidates.append(payload)
    raw_candidates.sort(key=_candidate_sort_key)
    kept: list[dict[str, Any]] = []
    for candidate in raw_candidates:
        current = set(candidate["support_original_indices"])
        duplicate = False
        for old in kept:
            previous = set(old["support_original_indices"])
            jaccard = len(current & previous) / len(current | previous)
            rotation, translation = transform_distance(
                candidate["diagnostic_transform"], old["diagnostic_transform"])
            if (jaccard >= config.duplicate_jaccard_min
                    and rotation <= config.transform_cluster_rotation_deg
                    and translation <= config.transform_cluster_translation_m):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= config.max_direction_hypotheses:
            break
    for index, row in enumerate(kept):
        row["hypothesis_index"] = index
        unsigned_hypothesis = dict(row)
        unsigned_hypothesis.pop("hypothesis_sha256", None)
        row["hypothesis_sha256"] = stable_json_sha256(unsigned_hypothesis)
    unsigned = {
        "schema": SCHEMA, "pair_id": str(pair_id), "arm": arm,
        "direction": direction, "source_cache_sha256": source_cache_sha256,
        "config": config_dict, "config_sha256": config_sha,
        "selected_original_indices": [int(value) for value in original],
        "selected_original_indices_sha256": array_sha256(original),
        "compatibility_edge_count": int(len(edges)),
        "compatibility_edges_sha256": edge_sha,
        "seed_triangles_considered": int(len(triangles)),
        "rejected_seed_counts": rejected,
        "hypotheses": kept, "hypothesis_count": len(kept),
        "gt_consumed": False, "gt_inputs": [], "fallback_used": False,
        "diagnostic_pose_passed_downstream": False,
    }
    return {**unsigned, "manifest_sha256": stable_json_sha256(unsigned)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def build_direction_candidates(
    cache_path: Path, output_dir: Path, *, pair_id: str, arm: str,
    direction: str, config: RigidMultiHypothesisConfig = FROZEN_CONFIG,
    preregister_path: Path | None = None,
) -> dict[str, Any]:
    """Seal exact-three-key candidate caches plus rehashable index receipts."""
    _require_frozen(config)
    cache_path, output_dir = Path(cache_path).resolve(), Path(output_dir).resolve()
    cache = load_frozen_correspondences(cache_path)
    generated = generate_direction_hypotheses(
        cache.src, cache.ref, cache.scores,
        source_cache_sha256=cache.cache_sha256, pair_id=pair_id, arm=arm,
        direction=direction, original_indices=cache.selected_original_indices,
        config=config)
    rows = []
    original_to_local = {int(original): local for local, original in enumerate(
        cache.selected_original_indices)}
    for row in generated["hypotheses"]:
        index = int(row["hypothesis_index"])
        stem = f"candidate_{index:02d}"
        original = np.asarray(row["support_original_indices"], dtype=np.int64)
        local = np.asarray([original_to_local[int(value)] for value in original],
                           dtype=np.int64)
        candidate_path = output_dir / f"{stem}.npz"
        indices_path = output_dir / f"{stem}.original_indices.npy"
        _atomic_npz(candidate_path, src_corr=cache.src[local],
                    ref_corr=cache.ref[local], scores=cache.scores[local])
        _atomic_npy(indices_path, original)
        receipt = {
            "schema": "v14-direction-candidate-receipt-v1",
            "pair_id": str(pair_id), "arm": arm, "direction": direction,
            "hypothesis_sha256": row["hypothesis_sha256"],
            "source_cache_path": str(cache_path),
            "source_cache_sha256": cache.cache_sha256,
            "candidate_cache_path": str(candidate_path),
            "candidate_cache_sha256": sha256_file(candidate_path),
            "support_indices_path": str(indices_path),
            "support_indices_sha256": sha256_file(indices_path),
            "support_original_indices_sha256": array_sha256(original),
            "correspondence_count": int(len(original)),
            "config_sha256": generated["config_sha256"],
            "diagnostic_pose_passed_downstream": False,
            "gt_consumed": False, "fallback_used": False,
        }
        receipt_path = output_dir / f"{stem}.receipt.json"
        receipt["payload_sha256"] = stable_json_sha256(receipt)
        _atomic_json(receipt_path, receipt)
        rows.append({**row,
                     "candidate_cache_path": str(candidate_path),
                     "candidate_cache_sha256": sha256_file(candidate_path),
                     "support_indices_path": str(indices_path),
                     "support_indices_sha256": sha256_file(indices_path),
                     "candidate_receipt_path": str(receipt_path),
                     "candidate_receipt_sha256": sha256_file(receipt_path)})
    final = {**generated, "source_cache_path": str(cache_path),
             "hypotheses": rows, "candidate_count": len(rows)}
    final.pop("manifest_sha256", None)
    if preregister_path is not None:
        preregister_path = Path(preregister_path).resolve()
        preregister = json.loads(preregister_path.read_text())
        if (preregister.get("schema")
                != "v14-rigid-multihypothesis-preregister-v1"
                or preregister.get("gt_allowed") is not False):
            raise RigidMultiHypothesisError(
                "V14 preregistration contract mismatch")
        final["preregister_path"] = str(preregister_path)
        final["preregister_sha256"] = sha256_file(preregister_path)
    final["payload_sha256"] = stable_json_sha256(final)
    _atomic_json(output_dir / "manifest.json", final)
    return final


def bind_direction_manifest(manifest_path: Path,
                            preregister_path: Path) -> dict[str, Any]:
    """Reject the historical two-write manifest binding sequence.

    Production callers must pass ``preregister_path`` to
    :func:`build_direction_candidates`, which creates the final bound manifest
    once.  Retaining this function as an explicit fail-closed shim makes stale
    callers visible without ever truncating their first manifest.
    """
    raise RigidMultiHypothesisError(
        "two-write direction binding is forbidden; pass preregister_path "
        "to build_direction_candidates")


def pair_bidirectional_hypotheses(
    forward: Mapping[str, Any], reverse: Mapping[str, Any],
    config: RigidMultiHypothesisConfig = FROZEN_CONFIG,
) -> dict[str, Any]:
    """Pair independently generated forward/reverse rigid hypotheses."""
    _require_frozen(config)
    for manifest, direction in ((forward, "forward"), (reverse, "reverse")):
        if "payload_sha256" in manifest:
            unsigned_manifest = dict(manifest)
            observed_payload = unsigned_manifest.pop("payload_sha256")
            if observed_payload != stable_json_sha256(unsigned_manifest):
                raise RigidMultiHypothesisError(
                    f"{direction} manifest payload SHA mismatch")
        if (manifest.get("schema") != SCHEMA
                or manifest.get("direction") != direction
                or manifest.get("config_sha256")
                != stable_json_sha256(asdict(config))):
            raise RigidMultiHypothesisError("direction manifest contract mismatch")
    if any(forward.get(key) != reverse.get(key) for key in ("pair_id", "arm")):
        raise RigidMultiHypothesisError("forward/reverse identity mismatch")
    candidates = []
    for forward_row in forward.get("hypotheses", []):
        for reverse_row in reverse.get("hypotheses", []):
            try:
                inverse = np.linalg.inv(np.asarray(
                    reverse_row["diagnostic_transform"], dtype=np.float64))
                rotation, translation = transform_distance(
                    forward_row["diagnostic_transform"], inverse)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if (rotation > config.transform_cluster_rotation_deg
                    or translation > config.transform_cluster_translation_m):
                continue
            payload = {
                "schema": CANDIDATE_SCHEMA,
                "pair_id": forward["pair_id"], "arm": forward["arm"],
                "forward_hypothesis_sha256": forward_row["hypothesis_sha256"],
                "reverse_hypothesis_sha256": reverse_row["hypothesis_sha256"],
                "forward_cache_sha256": forward["source_cache_sha256"],
                "reverse_cache_sha256": reverse["source_cache_sha256"],
                "forward_candidate_cache_sha256": forward_row.get(
                    "candidate_cache_sha256"),
                "reverse_candidate_cache_sha256": reverse_row.get(
                    "candidate_cache_sha256"),
                "forward_candidate_cache_path": forward_row.get(
                    "candidate_cache_path"),
                "reverse_candidate_cache_path": reverse_row.get(
                    "candidate_cache_path"),
                "forward_candidate_receipt_sha256": forward_row.get(
                    "candidate_receipt_sha256"),
                "reverse_candidate_receipt_sha256": reverse_row.get(
                    "candidate_receipt_sha256"),
                "forward_candidate_receipt_path": forward_row.get(
                    "candidate_receipt_path"),
                "reverse_candidate_receipt_path": reverse_row.get(
                    "candidate_receipt_path"),
                "forward_correspondence_count": forward_row["correspondence_count"],
                "reverse_correspondence_count": reverse_row["correspondence_count"],
                "rotation_deg": rotation, "translation_m": translation,
                "gt_consumed": False, "fallback_used": False,
            }
            payload["candidate_sha256"] = stable_json_sha256(payload)
            candidates.append(payload)
    candidates.sort(key=lambda row: (
        row["rotation_deg"], row["translation_m"],
        -min(row["forward_correspondence_count"],
             row["reverse_correspondence_count"]),
        row["forward_hypothesis_sha256"], row["reverse_hypothesis_sha256"]))
    candidates = candidates[:config.max_bidirectional_candidates]
    unsigned = {
        "schema": PAIR_SCHEMA, "pair_id": forward["pair_id"],
        "arm": forward["arm"], "config_sha256": stable_json_sha256(asdict(config)),
        "forward_manifest_payload_sha256": forward.get(
            "payload_sha256", forward.get("manifest_sha256")),
        "reverse_manifest_payload_sha256": reverse.get(
            "payload_sha256", reverse.get("manifest_sha256")),
        "candidates": candidates, "candidate_count": len(candidates),
        "gt_consumed": False, "fallback_used": False,
    }
    return {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}


def seal_bidirectional_candidate_set(
    forward_manifest_path: Path, reverse_manifest_path: Path,
    output_path: Path, preregister_path: Path,
) -> dict[str, Any]:
    """Rehash both direction manifests and atomically seal their pairing."""
    forward_manifest_path = Path(forward_manifest_path).resolve()
    reverse_manifest_path = Path(reverse_manifest_path).resolve()
    preregister_path = Path(preregister_path).resolve()
    preregister = json.loads(preregister_path.read_text())
    if (preregister.get("schema")
            != "v14-rigid-multihypothesis-preregister-v1"
            or preregister.get("gt_allowed") is not False):
        raise RigidMultiHypothesisError("V14 preregistration contract mismatch")
    forward = json.loads(forward_manifest_path.read_text())
    reverse = json.loads(reverse_manifest_path.read_text())
    expected_preregister = sha256_file(preregister_path)
    if (forward.get("preregister_sha256") != expected_preregister
            or reverse.get("preregister_sha256") != expected_preregister):
        raise RigidMultiHypothesisError(
            "direction manifest preregistration mismatch")
    value = pair_bidirectional_hypotheses(forward, reverse)
    value.pop("payload_sha256", None)
    value.update({
        "forward_manifest_path": str(forward_manifest_path),
        "forward_manifest_file_sha256": sha256_file(forward_manifest_path),
        "reverse_manifest_path": str(reverse_manifest_path),
        "reverse_manifest_file_sha256": sha256_file(reverse_manifest_path),
        "preregister_path": str(preregister_path),
        "preregister_sha256": expected_preregister,
    })
    value["payload_sha256"] = stable_json_sha256(value)
    _atomic_json(Path(output_path).resolve(), value)
    return value


_DERIVED_HYPOTHESIS_FIELDS = {
    "candidate_cache_path", "candidate_cache_sha256",
    "support_indices_path", "support_indices_sha256",
    "candidate_receipt_path", "candidate_receipt_sha256",
}


def _hypothesis_core(row: Mapping[str, Any]) -> dict[str, Any]:
    core = {key: item for key, item in row.items()
            if key not in _DERIVED_HYPOTHESIS_FIELDS
            and key != "hypothesis_sha256"}
    observed = row.get("hypothesis_sha256")
    if observed != stable_json_sha256(core):
        raise RigidMultiHypothesisError("direction hypothesis payload mismatch")
    return core


def _raw_three_key(path: Path) -> dict[str, np.ndarray]:
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"src_corr", "ref_corr", "scores"}:
            raise RigidMultiHypothesisError("candidate cache is not exact-three")
        arrays = {key: np.ascontiguousarray(np.asarray(data[key]))
                  for key in ("src_corr", "ref_corr", "scores")}
    if sha256_file(path) != before:
        raise RigidMultiHypothesisError("candidate cache changed while reading")
    return arrays


def _verify_direction_manifest(
    manifest: Mapping[str, Any], *, direction: str,
) -> None:
    """Regenerate one direction from its rehashed source and verify artifacts."""
    source_path = Path(str(manifest.get("source_cache_path", ""))).resolve()
    if not source_path.is_file():
        raise RigidMultiHypothesisError(
            f"{direction} source cache path is not rehashable")
    source = load_frozen_correspondences(source_path)
    if (manifest.get("source_cache_sha256") != source.cache_sha256
            or manifest.get("selected_original_indices")
            != [int(value) for value in source.selected_original_indices]
            or manifest.get("selected_original_indices_sha256")
            != array_sha256(source.selected_original_indices)):
        raise RigidMultiHypothesisError(
            f"{direction} source cache selection mismatch")
    regenerated = generate_direction_hypotheses(
        source.src, source.ref, source.scores,
        source_cache_sha256=source.cache_sha256,
        pair_id=str(manifest.get("pair_id")), arm=str(manifest.get("arm")),
        direction=direction,
        original_indices=source.selected_original_indices)
    for key, expected in regenerated.items():
        if key in ("manifest_sha256", "hypotheses"):
            continue
        if manifest.get(key) != expected:
            raise RigidMultiHypothesisError(
                f"{direction} deterministic manifest mismatch: {key}")
    rows = manifest.get("hypotheses")
    if (not isinstance(rows, list)
            or len(rows) != len(regenerated["hypotheses"])
            or manifest.get("candidate_count") != len(rows)):
        raise RigidMultiHypothesisError(
            f"{direction} hypothesis count mismatch")
    original_to_local = {int(original): local for local, original in enumerate(
        source.selected_original_indices)}
    for observed, expected in zip(rows, regenerated["hypotheses"]):
        if (_hypothesis_core(observed) != _hypothesis_core(expected)
                or observed.get("hypothesis_sha256")
                != expected.get("hypothesis_sha256")):
            raise RigidMultiHypothesisError(
                f"{direction} deterministic hypothesis mismatch")
        support = np.asarray(observed.get("support_original_indices"))
        if (support.ndim != 1 or not np.issubdtype(support.dtype, np.integer)
                or len(np.unique(support)) != len(support)
                or any(int(value) not in original_to_local for value in support)):
            raise RigidMultiHypothesisError(
                f"{direction} support is not a source-cache subset")
        support = np.ascontiguousarray(support, dtype=np.int64)
        local = np.asarray([original_to_local[int(value)] for value in support],
                           dtype=np.int64)
        cache_path = Path(str(observed.get("candidate_cache_path", ""))).resolve()
        receipt_path = Path(str(observed.get("candidate_receipt_path", ""))).resolve()
        indices_path = Path(str(observed.get("support_indices_path", ""))).resolve()
        if (not cache_path.is_file() or not receipt_path.is_file()
                or not indices_path.is_file()
                or sha256_file(cache_path)
                != observed.get("candidate_cache_sha256")
                or sha256_file(receipt_path)
                != observed.get("candidate_receipt_sha256")
                or sha256_file(indices_path)
                != observed.get("support_indices_sha256")):
            raise RigidMultiHypothesisError(
                f"{direction} hypothesis artifact closure mismatch")
        stored_support = np.load(indices_path, allow_pickle=False)
        if (stored_support.dtype != support.dtype
                or not np.array_equal(stored_support, support)):
            raise RigidMultiHypothesisError(
                f"{direction} support indices differ from manifest")
        raw = _raw_three_key(cache_path)
        expected_arrays = {
            "src_corr": np.ascontiguousarray(source.src[local]),
            "ref_corr": np.ascontiguousarray(source.ref[local]),
            "scores": np.ascontiguousarray(source.scores[local]),
        }
        if any(raw[key].dtype != expected_arrays[key].dtype
               or raw[key].shape != expected_arrays[key].shape
               or array_sha256(raw[key]) != array_sha256(expected_arrays[key])
               or not np.array_equal(raw[key], expected_arrays[key])
               for key in expected_arrays):
            raise RigidMultiHypothesisError(
                f"{direction} candidate is not the exact source-cache subset")
        receipt = json.loads(receipt_path.read_text())
        receipt_unsigned = dict(receipt)
        receipt_payload_sha = receipt_unsigned.pop("payload_sha256", None)
        exact = {
            "schema": "v14-direction-candidate-receipt-v1",
            "pair_id": manifest["pair_id"], "arm": manifest["arm"],
            "direction": direction,
            "hypothesis_sha256": observed["hypothesis_sha256"],
            "source_cache_path": str(source_path),
            "source_cache_sha256": source.cache_sha256,
            "candidate_cache_path": str(cache_path),
            "candidate_cache_sha256": sha256_file(cache_path),
            "support_indices_path": str(indices_path),
            "support_indices_sha256": sha256_file(indices_path),
            "support_original_indices_sha256": array_sha256(support),
            "correspondence_count": len(support),
            "config_sha256": manifest["config_sha256"],
            "diagnostic_pose_passed_downstream": False,
            "gt_consumed": False, "fallback_used": False,
        }
        if (receipt_payload_sha != stable_json_sha256(receipt_unsigned)
                or receipt_unsigned != exact):
            raise RigidMultiHypothesisError(
                f"{direction} candidate receipt mismatch")


def verify_candidate_set_contract(candidate_set_path: Path) -> dict[str, Any]:
    """Verify the whole candidate set, including an empty set, recursively."""
    candidate_set_path = Path(candidate_set_path).resolve()
    candidate_set_sha = sha256_file(candidate_set_path)
    value = json.loads(candidate_set_path.read_text())
    if not isinstance(value, dict) or value.get("schema") != PAIR_SCHEMA:
        raise RigidMultiHypothesisError("candidate set schema mismatch")
    payload_sha = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if payload_sha != stable_json_sha256(unsigned):
        raise RigidMultiHypothesisError("candidate set payload SHA mismatch")
    preregister_path = Path(str(value.get("preregister_path", ""))).resolve()
    if (not preregister_path.is_file()
            or sha256_file(preregister_path) != value.get("preregister_sha256")):
        raise RigidMultiHypothesisError(
            "candidate set preregistration file closure mismatch")
    direction_manifests = {}
    for direction in ("forward", "reverse"):
        manifest_path = Path(str(value.get(
            f"{direction}_manifest_path", ""))).resolve()
        if (not manifest_path.is_file()
                or sha256_file(manifest_path) != value.get(
                    f"{direction}_manifest_file_sha256")):
            raise RigidMultiHypothesisError(
                f"{direction} direction manifest file closure mismatch")
        manifest = json.loads(manifest_path.read_text())
        manifest_unsigned = dict(manifest)
        observed = manifest_unsigned.pop("payload_sha256", None)
        if (observed != stable_json_sha256(manifest_unsigned)
                or observed != value.get(
                    f"{direction}_manifest_payload_sha256")):
            raise RigidMultiHypothesisError(
                f"{direction} direction manifest payload mismatch")
        if manifest.get("preregister_sha256") != value.get("preregister_sha256"):
            raise RigidMultiHypothesisError(
                f"{direction} direction preregistration mismatch")
        _verify_direction_manifest(manifest, direction=direction)
        direction_manifests[direction] = manifest
    forward_source = Path(direction_manifests["forward"][
        "source_cache_path"]).resolve()
    reverse_source = Path(direction_manifests["reverse"][
        "source_cache_path"]).resolve()
    if (forward_source == reverse_source
            or direction_manifests["forward"].get("source_cache_sha256")
            == direction_manifests["reverse"].get("source_cache_sha256")):
        raise RigidMultiHypothesisError(
            "forward/reverse source caches must be distinct")
    recomputed = pair_bidirectional_hypotheses(
        direction_manifests["forward"], direction_manifests["reverse"])
    if (recomputed.get("candidates") != value.get("candidates")
            or recomputed.get("config_sha256") != value.get("config_sha256")
            or recomputed.get("candidate_count") != value.get("candidate_count")):
        raise RigidMultiHypothesisError(
            "candidate set is not the deterministic direction pairing")
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise RigidMultiHypothesisError("candidate set candidates are malformed")
    return {
        "candidate_set_path": str(candidate_set_path),
        "candidate_set_sha256": candidate_set_sha,
        "candidate_set_payload_sha256": payload_sha,
        "preregister_path": value.get("preregister_path"),
        "preregister_sha256": value.get("preregister_sha256"),
        "value": value, "direction_manifests": direction_manifests,
        "gt_consumed": False, "fallback_used": False,
    }


def load_candidate_contract(candidate_set_path: Path, candidate_index: int,
                            ) -> dict[str, Any]:
    """Load one candidate only after recursively verifying the whole set."""
    verified = verify_candidate_set_contract(candidate_set_path)
    value = verified["value"]
    candidates = value["candidates"]
    if not 0 <= candidate_index < len(candidates):
        raise RigidMultiHypothesisError("candidate index is outside frozen set")
    candidate = dict(candidates[candidate_index])
    expected_candidate_sha = candidate.pop("candidate_sha256", None)
    if expected_candidate_sha != stable_json_sha256(candidate):
        raise RigidMultiHypothesisError("candidate payload SHA mismatch")
    candidate["candidate_sha256"] = expected_candidate_sha
    return {
        "candidate_set_path": verified["candidate_set_path"],
        "candidate_set_sha256": verified["candidate_set_sha256"],
        "candidate_set_payload_sha256": verified[
            "candidate_set_payload_sha256"],
        "candidate_index": int(candidate_index), "candidate": candidate,
        "preregister_path": value.get("preregister_path"),
        "preregister_sha256": value.get("preregister_sha256"),
        "cache_sha256": {
            direction: candidate[f"{direction}_candidate_cache_sha256"]
            for direction in ("forward", "reverse")},
        "candidate_receipt_sha256": {
            direction: candidate[f"{direction}_candidate_receipt_sha256"]
            for direction in ("forward", "reverse")},
        "gt_consumed": False, "fallback_used": False,
    }


def select_unique_safe_candidate(
    evidence: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]], *,
    known_bad: bool,
) -> dict[str, Any]:
    """Accept exactly one fully bound V13-strict candidate; otherwise reject."""
    safe = []
    for contract, strict in evidence:
        candidate = contract.get("candidate", {})
        if candidate.get("schema") != CANDIDATE_SCHEMA:
            continue
        expected_cache = {
            "forward": candidate.get("forward_candidate_cache_sha256"),
            "reverse": candidate.get("reverse_candidate_cache_sha256"),
        }
        expected_receipt = {
            "forward": candidate.get("forward_candidate_receipt_sha256"),
            "reverse": candidate.get("reverse_candidate_receipt_sha256"),
        }
        expected_cache_path = {
            direction: candidate.get(f"{direction}_candidate_cache_path")
            for direction in ("forward", "reverse")}
        expected_receipt_path = {
            direction: candidate.get(f"{direction}_candidate_receipt_path")
            for direction in ("forward", "reverse")}
        expected_identity = {
            "candidate_sha256": candidate.get("candidate_sha256"),
            "candidate_index": contract.get("candidate_index"),
            "candidate_set_path": contract.get("candidate_set_path"),
            "candidate_set_sha256": contract.get("candidate_set_sha256"),
            "pair_id": candidate.get("pair_id"),
            "arm": candidate.get("arm"),
            "cache_sha256": expected_cache,
            "candidate_cache_path": expected_cache_path,
            "candidate_receipt_sha256": expected_receipt,
            "candidate_receipt_path": expected_receipt_path,
        }
        if (not all(isinstance(value, str) and len(value) == 64
                    for value in (*expected_cache.values(),
                                  *expected_receipt.values()))
                or not isinstance(expected_identity["candidate_sha256"], str)
                or len(expected_identity["candidate_sha256"]) != 64
                or not isinstance(expected_identity["candidate_index"], int)
                or not all(isinstance(value, str) and value
                           for value in (*expected_cache_path.values(),
                                         *expected_receipt_path.values(),
                                         expected_identity["candidate_set_path"],
                                         expected_identity["pair_id"],
                                         expected_identity["arm"]))
                or not isinstance(expected_identity["candidate_set_sha256"], str)
                or len(expected_identity["candidate_set_sha256"]) != 64
                or strict.get("schema") != STRICT_SCHEMA
                or strict.get("gate_authority") != STRICT_AUTHORITY
                or strict.get("safe") is not True
                or strict.get("gt_consumed") is not False
                or strict.get("fallback_used") is not False
                or any(strict.get(key) != value
                       for key, value in expected_identity.items())):
            continue
        safe.append(str(candidate.get("candidate_sha256")))
    if known_bad:
        return {"accepted": False, "reason": "known_bad_veto",
                "safe_candidate_count_before_veto": len(safe),
                "safe_candidate_sha256": safe}
    if not safe:
        return {"accepted": False, "reason": "no_safe_candidate",
                "safe_candidate_count": 0, "safe_candidate_sha256": []}
    if len(safe) > 1:
        return {"accepted": False,
                "reason": "ambiguous_multiple_safe_candidates",
                "safe_candidate_count": len(safe),
                "safe_candidate_sha256": safe}
    return {"accepted": True, "reason": "unique_safe_candidate",
            "safe_candidate_count": 1, "candidate_sha256": safe[0]}


def aggregate_fixed4_research(rows: Sequence[Mapping[str, Any]],
                              preregister: Mapping[str, Any]) -> dict[str, Any]:
    """Sole V14 fixed4 authority; the fullscan control cannot rescue primary."""
    if (preregister.get("schema")
            != "v14-rigid-multihypothesis-preregister-v1"
            or preregister.get("selection_rule")
            != "exactly_one_strict_safe_candidate_else_reject"):
        raise RigidMultiHypothesisError("V14 aggregate preregistration mismatch")
    pair_order = [str(value) for value in preregister.get("fixed_pair_order", ())]
    primary, control = (str(preregister.get("primary_arm", "")),
                        str(preregister.get("control_arm", "")))
    known_bad = str(preregister.get("known_bad_pair_id", ""))
    expected = [(pair_id, arm) for pair_id in pair_order
                for arm in (primary, control)]
    actual = [(str(row.get("pair_id")), str(row.get("arm"))) for row in rows]
    if len(pair_order) != 4 or len(set(pair_order)) != 4 \
            or known_bad != pair_order[-1] or actual != expected:
        raise RigidMultiHypothesisError("V14 aggregate requires exact ordered 4x2 rows")
    by_key = {(row["pair_id"], row["arm"]): row["decision"] for row in rows}
    normal = pair_order[:-1]
    primary_safe = {pair_id: bool(by_key[(pair_id, primary)].get("accepted"))
                    for pair_id in normal}
    control_safe = {pair_id: bool(by_key[(pair_id, control)].get("accepted"))
                    for pair_id in normal}
    known_bad_veto = {
        arm: by_key[(known_bad, arm)].get("reason") == "known_bad_veto"
        for arm in (primary, control)
    }
    ambiguous = [
        {"pair_id": pair_id, "arm": arm}
        for pair_id, arm in expected
        if by_key[(pair_id, arm)].get("reason")
        == "ambiguous_multiple_safe_candidates"
    ]
    safe = all(primary_safe.values()) and all(known_bad_veto.values()) \
        and not ambiguous
    return {
        "schema": "v14-fixed4-rigid-multihypothesis-aggregate-v1",
        "safe": safe,
        "reason": ("fixed4_research_gate_pass" if safe else
                   "ambiguous_safe_candidates" if ambiguous else
                   "known_bad_veto_failed" if not all(known_bad_veto.values())
                   else "normal_primary_failed"),
        "primary_arm": primary, "control_arm": control,
        "control_can_rescue": False,
        "normal_primary_safe": primary_safe,
        "normal_primary_failures": [pair_id for pair_id in normal
                                    if not primary_safe[pair_id]],
        "control_safe_diagnostic": control_safe,
        "known_bad_pair_id": known_bad,
        "known_bad_veto_by_arm": known_bad_veto,
        "ambiguous_rows": ambiguous,
        "gt_consumed": False, "official92_run": False,
    }
