"""Frozen-correspondence, GT-free dual rigid-solver runtime for V13.

The public convention is a column-vector source-to-reference SE(3):
``reference = R @ source + t``.  A failed solver has no transform; identity is
never substituted.  The runtime is deliberately separate from ColorPCR and
from RegistrationDecision so its output is only correspondence-solver
evidence, not a reconstruction verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA = "v13-dual-solver-worker-v1"
SUMMARY_SCHEMA = "v13-dual-solver-summary-v1"
SOLVERS = ("pointdsc", "pygcransac")
DIRECTIONS = ("forward", "reverse")
REPEATS = 5
QUORUM = 4
MAX_CORRESPONDENCES = 1000
INLIER_THRESHOLD_M = 0.10
MAX_ROTATION_DEG = 5.0
MAX_TRANSLATION_M = 0.10
POINTDSC_SOURCE_COMMIT = "b009d536ac10b570853833f2178397c154745da9"
POINTDSC_IMPLEMENTATION_SHA256 = (
    "dd91eb6cc0b92d5023ea4804a4c45c7f8aaa9d3d0750ac7096f3181c8bc319de"
)
POINTDSC_CHECKPOINT_SHA256 = (
    "20662778fca1a7d2c4e2f79f381d4be6cb891834d7bb4bd91ade9d89b0d13bd4"
)
PYGCRANSAC_VERSION = "0.1.1"
PYGCRANSAC_EXTENSION_SHA256 = (
    "5c1c5aee8988f6de2a710997b3491fc3142e62a88bf1a7871a729c029c8d7216"
)


class RuntimeContractError(RuntimeError):
    """Typed contract failure; callers must not fabricate a transform."""


class DependencyMismatch(RuntimeContractError):
    pass


class InsufficientCorrespondences(RuntimeContractError):
    pass


class SolverFailure(RuntimeContractError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class FrozenCorrespondences:
    src: np.ndarray
    ref: np.ndarray
    scores: np.ndarray
    selected_original_indices: np.ndarray
    cache_path: str
    cache_sha256: str
    input_sha256: str


def _validate_points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        raise RuntimeContractError(f"{name} must be non-empty N x 3")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise RuntimeContractError(f"{name} must be finite floating-point metres")
    return np.ascontiguousarray(array, dtype=np.float64)


def load_frozen_correspondences(path: Path) -> FrozenCorrespondences:
    """Load exactly src_corr/ref_corr/scores and perform a stable score top-k."""
    path = Path(path).resolve()
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"src_corr", "ref_corr", "scores"}:
            raise RuntimeContractError(
                "frozen cache must contain exactly src_corr/ref_corr/scores")
        src = _validate_points(data["src_corr"], "src_corr")
        ref = _validate_points(data["ref_corr"], "ref_corr")
        scores = np.asarray(data["scores"])
    if sha256_file(path) != before:
        raise RuntimeContractError("correspondence cache changed while reading")
    if ref.shape != src.shape or scores.shape != (len(src),):
        raise RuntimeContractError("src/ref/scores rows must be aligned")
    if not np.issubdtype(scores.dtype, np.floating) or not np.isfinite(scores).all():
        raise RuntimeContractError("scores must be finite floating-point")
    # Primary key score descending, secondary key original row ascending.
    order = np.lexsort((np.arange(len(scores), dtype=np.int64), -scores.astype(np.float64)))
    order = np.ascontiguousarray(order[:min(len(order), MAX_CORRESPONDENCES)], np.int64)
    if len(order) < 40:  # official PointDSC 3DMatch k=40 and ratio=0.1
        raise InsufficientCorrespondences("at least 40 correspondences are required")
    src, ref = src[order], ref[order]
    scores = np.ascontiguousarray(scores[order], dtype=np.float64)
    binding = {
        "cache_sha256": before,
        "src_corr_sha256": array_sha256(src),
        "ref_corr_sha256": array_sha256(ref),
        "scores_sha256": array_sha256(scores),
        "selected_original_indices_sha256": array_sha256(order),
        "unit": "metre",
        "direction": "source_to_reference",
        "top_k": len(order),
    }
    return FrozenCorrespondences(src, ref, scores, order, str(path), before,
                                 stable_json_sha256(binding))


def fixed_permutation(input_sha256: str, direction: str, repeat: int,
                      length: int) -> tuple[np.ndarray, int]:
    if direction not in DIRECTIONS or repeat not in range(REPEATS):
        raise RuntimeContractError("invalid direction/repeat")
    if repeat == 0:
        return np.arange(length, dtype=np.int64), 0
    material = f"{input_sha256}|{direction}|{repeat}|v13".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed).permutation(length), seed


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = validate_se3(transform)
    return np.asarray(points, np.float64) @ transform[:3, :3].T + transform[:3, 3]


def validate_se3(transform: Any) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise SolverFailure("solver did not return finite 4x4 transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise SolverFailure("invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise SolverFailure("rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise SolverFailure("rotation is not proper")
    return value


def transform_distance(a: Any, b: Any) -> tuple[float, float]:
    a, b = validate_se3(a), validate_se3(b)
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine))), float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _dependency_sha(module: Any) -> str:
    path = Path(module.__file__).resolve()
    return sha256_file(path)


def _pygcransac_extension_path(module: Any) -> Path:
    root = Path(module.__file__).resolve().parent
    candidates = sorted(root.glob("pygcransac*.so"))
    if len(candidates) != 1:
        raise DependencyMismatch(
            f"expected exactly one pygcransac compiled extension, found {len(candidates)}")
    extension = candidates[0]
    if sha256_file(extension) != PYGCRANSAC_EXTENSION_SHA256:
        raise DependencyMismatch("pygcransac compiled extension SHA-256 mismatch")
    return extension


def pygcransac_row_to_column(raw_pose: Any) -> np.ndarray:
    """Convert pygcransac 0.1.1 row-vector pose to public column SE(3)."""
    raw = np.asarray(raw_pose, dtype=np.float64)
    if raw.shape != (4, 4) or not np.isfinite(raw).all():
        raise SolverFailure("pygcransac returned invalid raw pose")
    return validate_se3(raw.T)


def solve_pygcransac(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, dict]:
    try:
        import pygcransac
    except Exception as exc:  # pragma: no cover - dependency environment
        raise DependencyMismatch(f"pygcransac import failed: {exc}") from exc
    if getattr(pygcransac, "__version__", None) != PYGCRANSAC_VERSION:
        raise DependencyMismatch("pygcransac version is not sealed 0.1.1")
    extension = _pygcransac_extension_path(pygcransac)
    correspondences = np.ascontiguousarray(
        np.concatenate([src, ref], axis=1), dtype=np.float64)
    # pyGCRANSAC 0.1.1 exposes the Nx6+probability API.  Score ranking has
    # already frozen the top-k; uniform probabilities avoid silently applying
    # a second, solver-specific score interpretation.
    probabilities = np.ones(len(correspondences), dtype=np.float64)
    raw_pose, mask = pygcransac.findRigidTransform(
        correspondences, probabilities,
        threshold=INLIER_THRESHOLD_M, conf=0.99999999,
        spatial_coherence_weight=0.1, max_iters=5000, use_sprt=True,
        min_inlier_ratio_for_sprt=0.1, sampler=0, neighborhood=0,
        use_space_partitioning=False,
    )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(src),) or int(mask.sum()) < 3:
        raise SolverFailure("pygcransac_insufficient_inliers")
    transform = pygcransac_row_to_column(raw_pose)
    return transform, {
        "inlier_count": int(mask.sum()),
        "inlier_ratio": float(mask.mean()),
        "dependency_path": str(extension),
        "dependency_sha256": sha256_file(extension),
        "dependency_version": PYGCRANSAC_VERSION,
        "checkpoint": "not_applicable_stateless_solver",
    }


class PointDSCSolver:
    """Strict loader for official PointDSC 3DMatch source and weights."""

    def __init__(self, source_root: Path, checkpoint: Path, device: str = "cpu"):
        import torch
        self.torch = torch
        self.source_root = Path(source_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if sha256_file(self.checkpoint) != POINTDSC_CHECKPOINT_SHA256:
            raise DependencyMismatch("PointDSC checkpoint SHA-256 mismatch")
        git_head = subprocess.run(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if git_head != POINTDSC_SOURCE_COMMIT:
            raise DependencyMismatch("PointDSC source commit mismatch")
        tracked = subprocess.run(
            ["git", "-C", str(self.source_root), "status", "--porcelain=v1",
             "--untracked-files=no"], check=True, capture_output=True, text=True,
        ).stdout
        if tracked.strip():
            raise DependencyMismatch("PointDSC tracked source differs from sealed HEAD")
        config = json.loads((self.source_root / "snapshot/PointDSC_3DMatch_release/config.json").read_text())
        sys.path.insert(0, str(self.source_root))
        try:
            from models.PointDSC import PointDSC
        finally:
            if sys.path[0] == str(self.source_root):
                sys.path.pop(0)
        self.device = torch.device(device)
        self.model = PointDSC(
            in_dim=config["in_dim"], num_layers=config["num_layers"],
            num_channels=config["num_channels"], num_iterations=config["num_iterations"],
            ratio=config["ratio"], sigma_d=config["sigma_d"], k=config["k"],
            nms_radius=config["inlier_threshold"],
        ).to(self.device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys != ["gamma"]:
            raise DependencyMismatch(
                f"PointDSC state mismatch missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}")
        self.model.eval()
        self.source_sha256 = sha256_file(self.source_root / "models/PointDSC.py")
        if self.source_sha256 != POINTDSC_IMPLEMENTATION_SHA256:
            raise DependencyMismatch("PointDSC implementation SHA-256 mismatch")

    def solve(self, src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, dict]:
        torch = self.torch
        src_tensor = torch.from_numpy(np.asarray(src, np.float32))[None].to(self.device)
        ref_tensor = torch.from_numpy(np.asarray(ref, np.float32))[None].to(self.device)
        corr_pos = torch.cat([src_tensor, ref_tensor], dim=-1)
        corr_pos = corr_pos - corr_pos.mean(dim=1, keepdim=True)
        with torch.no_grad():
            result = self.model({"corr_pos": corr_pos, "src_keypts": src_tensor,
                                 "tgt_keypts": ref_tensor, "testing": True})
        transform = validate_se3(result["final_trans"][0].detach().cpu().numpy())
        labels = result["final_labels"][0].detach().cpu().numpy() > 0.5
        if int(labels.sum()) < 3:
            raise SolverFailure("pointdsc_insufficient_inliers")
        return transform, {
            "inlier_count": int(labels.sum()),
            "inlier_ratio": float(labels.mean()),
            "dependency_path": str(self.source_root),
            "dependency_sha256": self.source_sha256,
            "dependency_version": POINTDSC_SOURCE_COMMIT,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": POINTDSC_CHECKPOINT_SHA256,
        }


def residual_statistics(src: np.ndarray, ref: np.ndarray, transform: np.ndarray) -> dict:
    residuals = np.linalg.norm(apply_transform(src, transform) - ref, axis=1)
    return {
        "median_residual_m": float(np.median(residuals)),
        "mean_residual_m": float(np.mean(residuals)),
        "p90_residual_m": float(np.quantile(residuals, 0.9)),
        "threshold_inlier_count": int((residuals <= INLIER_THRESHOLD_M).sum()),
        "threshold_inlier_ratio": float((residuals <= INLIER_THRESHOLD_M).mean()),
    }


def worker_payload(cache: FrozenCorrespondences, solver: str, direction: str,
                   repeat: int, solver_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict]],
                   runtime_sha256: str,
                   dependency_record: Mapping[str, Any] | None = None,
                   known_bad: bool = False) -> dict:
    permutation, seed = fixed_permutation(cache.input_sha256, direction, repeat, len(cache.src))
    # Each direction has its own independently generated ColorPCR cache.  In
    # particular, reverse must consume the reverse cache's own src/ref arrays;
    # swapping the forward cache would erase the upstream directional test.
    src, ref = cache.src[permutation], cache.ref[permutation]
    base = {
        "schema": SCHEMA, "solver": solver, "direction": direction,
        "repeat": repeat, "permutation_seed": seed,
        "selected_original_indices_sha256": array_sha256(cache.selected_original_indices[permutation]),
        "cache_path": cache.cache_path, "cache_sha256": cache.cache_sha256,
        "correspondence_sha256": cache.input_sha256,
        "runtime_sha256": runtime_sha256, "unit": "metre",
        "transform_direction": f"{'source_to_reference' if direction == 'forward' else 'reference_to_source'}",
        "gt_free": True, "gt_inputs": [], "fallback_used": False,
        "correspondence_count": len(src),
        "dependency": dict(dependency_record or {}),
        "known_bad_pair": bool(known_bad),
    }
    try:
        transform, diagnostics = solver_fn(src, ref)
        transform = validate_se3(transform)
        base.update({"status": "ok", "failure_type": None,
                     "transform": transform.tolist(),
                     "diagnostics": {**diagnostics, **residual_statistics(src, ref, transform)}})
    except Exception as exc:
        # Failure is typed and has no transform.  Never manufacture identity.
        kind = type(exc).__name__ if isinstance(exc, RuntimeContractError) else "unexpected_solver_exception"
        base.update({"status": "failed", "failure_type": kind,
                     "failure_message": str(exc), "transform": None,
                     "diagnostics": {}})
    evidence_view = dict(base)
    base["evidence_sha256"] = stable_json_sha256(evidence_view)
    return base


def _compatible(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    dr, dt = transform_distance(a["transform"], b["transform"])
    return dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M


def complete_linkage_q4(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Unique complete-linkage q4 cluster, with no rival accepted cluster."""
    from itertools import combinations
    valid = [dict(row) for row in rows if row.get("status") == "ok"]
    cliques: list[tuple[int, ...]] = []
    for size in range(len(valid), 0, -1):
        for indices in combinations(range(len(valid)), size):
            if all(_compatible(valid[i], valid[j]) for i, j in combinations(indices, 2)):
                if not any(set(indices) < set(old) for old in cliques):
                    cliques.append(indices)
    maximal = [c for c in cliques if not any(set(c) < set(o) for o in cliques)]
    maximal.sort(key=lambda c: (-len(c), c))
    largest = len(maximal[0]) if maximal else 0
    winners = [c for c in maximal if len(c) == largest]
    rival = any(len(c) >= QUORUM for c in maximal[1:])
    usable = len(rows) == REPEATS and largest >= QUORUM and len(winners) == 1 and not rival
    winning = winners[0] if usable else ()
    medoid = None
    if winning:
        def medoid_cost(index: int) -> tuple[float, int]:
            total = sum(transform_distance(valid[index]["transform"], valid[j]["transform"])[0] / MAX_ROTATION_DEG
                        + transform_distance(valid[index]["transform"], valid[j]["transform"])[1] / MAX_TRANSLATION_M
                        for j in winning)
            return total, int(valid[index]["repeat"])
        medoid = min(winning, key=medoid_cost)
    return {
        "usable": usable, "requested": len(rows), "valid": len(valid),
        "clique_sizes": [len(c) for c in maximal],
        "winning_repeats": [int(valid[i]["repeat"]) for i in winning],
        "medoid_repeat": int(valid[medoid]["repeat"]) if medoid is not None else None,
        "medoid_transform": valid[medoid]["transform"] if medoid is not None else None,
        "reasons": [] if usable else ["unique_complete_linkage_q4_not_met"],
    }


def summarize_workers(rows: Sequence[Mapping[str, Any]], known_bad: bool = False) -> dict:
    expected = {(solver, direction, repeat) for solver in SOLVERS
                for direction in DIRECTIONS for repeat in range(REPEATS)}
    actual = {(str(row["solver"]), str(row["direction"]), int(row["repeat"])) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeContractError("worker matrix must be exactly 2x2x5")
    gates: dict[str, Any] = {}
    medoids: dict[tuple[str, str], np.ndarray] = {}
    for solver in SOLVERS:
        for direction in DIRECTIONS:
            group = sorted([row for row in rows if row["solver"] == solver and row["direction"] == direction],
                           key=lambda row: int(row["repeat"]))
            gate = complete_linkage_q4(group)
            gates[f"{solver}/{direction}"] = gate
            if gate["usable"]:
                medoids[(solver, direction)] = validate_se3(gate["medoid_transform"])
    if len(medoids) != 4:
        return {"schema": SUMMARY_SCHEMA, "safe": False,
                "reason": "solver_direction_q4_failed", "gates": gates}
    direction_checks = {}
    for solver in SOLVERS:
        dr, dt = transform_distance(medoids[(solver, "forward")],
                                    np.linalg.inv(medoids[(solver, "reverse")]))
        direction_checks[solver] = {"rotation_deg": dr, "translation_m": dt,
                                    "usable": dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M}
    dr, dt = transform_distance(medoids[("pointdsc", "forward")],
                                medoids[("pygcransac", "forward")])
    cross = {"rotation_deg": dr, "translation_m": dt,
             "usable": dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M}
    consensus_safe = all(value["usable"] for value in direction_checks.values()) and cross["usable"]
    safe = consensus_safe and not known_bad
    return {"schema": SUMMARY_SCHEMA, "safe": safe,
            "reason": ("known_bad_veto" if known_bad else
                       "dual_solver_consensus_only" if safe else
                       "direction_or_solver_disagreement"),
            "registration_decision_authorized": False,
            "colorpcr_or_fixed4_claimed": False,
            "gates": gates, "direction_checks": direction_checks,
            "cross_solver_check": cross,
            "gt_free": True, "gt_inputs": [], "fallback_used": False,
            "known_bad_pair": bool(known_bad)}


def run_matrix(forward_cache_path: Path, reverse_cache_path: Path,
               output_dir: Path, pointdsc_root: Path,
               pointdsc_checkpoint: Path, device: str = "cpu",
               known_bad: bool = False) -> dict:
    caches = {
        "forward": load_frozen_correspondences(forward_cache_path),
        "reverse": load_frozen_correspondences(reverse_cache_path),
    }
    if caches["forward"].cache_sha256 == caches["reverse"].cache_sha256:
        raise RuntimeContractError("forward and reverse caches must be independently generated")
    output_dir = Path(output_dir)
    runtime_sha = sha256_file(Path(__file__))
    pointdsc = PointDSCSolver(pointdsc_root, pointdsc_checkpoint, device=device)
    functions = {"pointdsc": pointdsc.solve, "pygcransac": solve_pygcransac}
    import pygcransac
    if getattr(pygcransac, "__version__", None) != PYGCRANSAC_VERSION:
        raise DependencyMismatch("pygcransac version is not sealed 0.1.1")
    pyg_extension = _pygcransac_extension_path(pygcransac)
    dependencies = {
        "pointdsc": {
            "source_commit": POINTDSC_SOURCE_COMMIT,
            "implementation_sha256": pointdsc.source_sha256,
            "tracked_diff_head_empty": True,
            "checkpoint_sha256": POINTDSC_CHECKPOINT_SHA256,
        },
        "pygcransac": {
            "version": PYGCRANSAC_VERSION,
            "implementation_path": str(pyg_extension),
            "implementation_sha256": sha256_file(pyg_extension),
            "checkpoint": "not_applicable_stateless_solver",
        },
    }
    rows = []
    for solver in SOLVERS:
        for direction in DIRECTIONS:
            for repeat in range(REPEATS):
                row = worker_payload(caches[direction], solver, direction, repeat,
                                     functions[solver], runtime_sha,
                                     dependencies[solver], known_bad=known_bad)
                path = output_dir / "workers" / f"{solver}_{direction}_{repeat}.json"
                atomic_json(path, row)
                rows.append(row)
    summary = summarize_workers(rows, known_bad=known_bad)
    summary.update({"cache_sha256": {direction: cache.cache_sha256
                                     for direction, cache in caches.items()},
                    "correspondence_sha256": {direction: cache.input_sha256
                                              for direction, cache in caches.items()},
                    "runtime_sha256": runtime_sha,
                    "worker_count": len(rows),
                    "worker_evidence_sha256": {f"{r['solver']}/{r['direction']}/{r['repeat']}": r["evidence_sha256"] for r in rows}})
    atomic_json(output_dir / "summary.json", summary)
    return summary
