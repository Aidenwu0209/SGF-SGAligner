"""Research-only V7 repeated-registration pilot.

The controller and workers consume the immutable V6-Fix B/selection cache.
They never run SGAligner or GeoTransformer and never load posthoc labels.  A
separate process is used for every direction/replicate solve so native RANSAC
state cannot leak between observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from canonical_inputs import build_canonical_pair  # noqa: E402
from safety import decision_features as dfx  # noqa: E402
from safety.registration_consensus import (  # noqa: E402
    ConsensusConfig,
    cross_direction_agreement,
    evaluate_direction,
    transform_distance,
)
from safety.registration_decision import spatial_support  # noqa: E402
from v3b_cache_runner import ransac_from_pooled  # noqa: E402


SCHEMA = "v7-registration-veto-pilot-v1"
WORKER_SCHEMA = "v7-registration-veto-worker-v1"
CACHE_SCHEMA = "v6fix-inference-cache-v2"
CHECKPOINT_ID = "B"
CHECKPOINT_SHA256 = (
    "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200")
NEAR_MISS_PAIR = (
    "6a36052f-fa53-2915-9400-831b60c63077_to_"
    "6a36052d-fa53-2915-9764-30d81b2cc2b5")
FORMAL_ROOT = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official-v6fix-audit/outputs/"
    "official_sgaligner_v6_fix_consistency_audit_20260829/formal_v2")
DEFAULT_CACHE_ROOT = FORMAL_ROOT / "cache_v2/B/selection"
DEFAULT_OUT = CODE_ROOT / "outputs/v7_registration_veto_pilot_20260830"
PROTOCOL = CODE_ROOT / "docs/V7_REGISTRATION_CONSENSUS_PROTOCOL.md"
POLICIES = tuple(
    ConsensusConfig(quorum=quorum, max_rotation_deg=rotation,
                    max_translation_m=translation)
    for rotation in (2.5, 5.0)
    for translation in (0.05, 0.10)
    for quorum in (4, 5)
)


class PilotEvidenceError(RuntimeError):
    """Immutable evidence or worker output failed validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def atomic_create_json(path: Path, value: Any) -> None:
    """Atomically create a JSON artifact; an existing path is never replaced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PilotEvidenceError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def tracked_git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=CODE_ROOT).returncode != 0
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=CODE_ROOT).returncode != 0
    if dirty or staged:
        raise PilotEvidenceError("pilot requires a clean tracked worktree")
    return {"head": head, "tracked_dirty": False}


def protocol_sha256() -> str:
    if not PROTOCOL.is_file():
        raise PilotEvidenceError(f"missing protocol {PROTOCOL}")
    return sha256_file(PROTOCOL)


def cache_path(cache_root: Path, pair_id: str) -> Path:
    root = cache_root.resolve()
    expected = DEFAULT_CACHE_ROOT.resolve()
    if root != expected:
        raise PilotEvidenceError(
            f"cache root is not immutable formal B/selection root: {root}")
    path = root / f"{pair_id}.pt"
    if not path.is_file() or path.parent.resolve() != expected:
        raise PilotEvidenceError(f"missing immutable cache {path}")
    return path


def load_validated_cache(path: Path, pair_id: str,
                         expected_file_sha: str | None = None) -> dict:
    before = sha256_file(path)
    if expected_file_sha is not None and before != expected_file_sha:
        raise PilotEvidenceError("cache file SHA differs from controller")
    cached = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "cache_schema", "pair_id", "checkpoint_id", "checkpoint_sha256",
        "input_sha256", "node_corrs", "provenance", "geot",
    }
    if not isinstance(cached, dict) or not required.issubset(cached):
        raise PilotEvidenceError("cache schema fields missing")
    if cached["cache_schema"] != CACHE_SCHEMA:
        raise PilotEvidenceError("cache schema mismatch")
    if cached["pair_id"] != pair_id or cached["checkpoint_id"] != CHECKPOINT_ID:
        raise PilotEvidenceError("cache pair/checkpoint mismatch")
    if cached["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise PilotEvidenceError("checkpoint SHA mismatch")
    provenance = cached["provenance"]
    if (not isinstance(provenance, dict)
            or provenance.get("cache_key") != cached["input_sha256"]
            or provenance.get("pair_id") != pair_id
            or provenance.get("checkpoint_id") != CHECKPOINT_ID
            or provenance.get("checkpoint_sha256") != CHECKPOINT_SHA256):
        raise PilotEvidenceError("cache provenance mismatch")
    members = [(int(a), int(b)) for a, b in cached["node_corrs"]]
    if not members or len(members) != len(set(members)):
        raise PilotEvidenceError("flat node correspondence set is empty/duplicated")
    if sha256_file(path) != before:
        raise PilotEvidenceError("cache changed while being read")
    cached["_file_sha256"] = before
    cached["_members"] = members
    return cached


def validate_canonical_surfaces(data: Mapping[str, Any],
                                cached: Mapping[str, Any]) -> None:
    provenance = cached["provenance"]
    if int(data["src_count"]) != int(provenance["src_count"]):
        raise PilotEvidenceError("canonical src_count differs from cache")
    observed_ids = np.asarray(data["obj_ids"]).tolist()
    if observed_ids != provenance["object_ids_order"]:
        raise PilotEvidenceError("canonical object order differs from cache")
    expected = {
        int(row["index"]): (int(row["points"]), row["sha256"])
        for row in provenance["registration_surfaces"]
    }
    objects = data["registration_pts"]
    if set(int(key) for key in objects) != set(expected):
        raise PilotEvidenceError("registration surface index set mismatch")
    for index, (count, digest) in expected.items():
        points = np.asarray(objects[index], dtype=np.float32)
        actual = hashlib.sha256(
            np.ascontiguousarray(points).tobytes()).hexdigest()
        if len(points) != count or actual != digest:
            raise PilotEvidenceError(f"registration surface mismatch {index}")


def stable_row_permutation(count: int, *, pair_id: str, direction: str,
                           replicate: int, protocol_sha: str,
                           hypothesis: str = "F") -> tuple[np.ndarray, str]:
    if count <= 0 or direction not in ("forward", "reverse"):
        raise ValueError("invalid row permutation request")
    context = {
        "pair_id": pair_id,
        "hypothesis": hypothesis,
        "direction": direction,
        "replicate": int(replicate),
        "protocol_sha256": protocol_sha,
    }
    context_sha = stable_json_hash(context)
    seed = int.from_bytes(bytes.fromhex(context_sha)[:8], "big")
    permutation = np.random.default_rng(seed).permutation(count)
    permutation_sha = hashlib.sha256(
        np.ascontiguousarray(permutation.astype(np.int64)).tobytes()
    ).hexdigest()
    return permutation, stable_json_hash({
        "context_sha256": context_sha,
        "permutation_sha256": permutation_sha,
        "count": count,
    })


def pool_correspondences(cached: Mapping[str, Any], direction: str,
                         permutation: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray,
                                                               list, list]:
    members = cached["_members"]
    cap_total = int(cached["provenance"]["matcher_contract"][
        "point_correspondence_cap"])
    cap = max(cap_total // len(members), 1)
    source_rows, reference_rows, used, failures = [], [], [], []
    for src_idx, ref_idx in members:
        entry = cached["geot"].get((src_idx, ref_idx))
        if entry is None or entry.get("status") != "ok":
            failures.append({
                "src_index": src_idx, "ref_index": ref_idx,
                "stage": entry.get("status", "missing") if entry else "missing",
            })
            continue
        src = np.asarray(entry.get("src_corr"), dtype=np.float64)
        ref = np.asarray(entry.get("ref_corr"), dtype=np.float64)
        scores = np.asarray(entry.get("scores"), dtype=np.float64)
        if (src.ndim != 2 or ref.shape != src.shape or src.shape[1:] != (3,)
                or scores.shape != (len(src),) or not np.isfinite(src).all()
                or not np.isfinite(ref).all() or not np.isfinite(scores).all()):
            raise PilotEvidenceError("malformed cached GeoTransformer entry")
        if "sha256" in entry:
            actual = hashlib.sha256(b"".join(
                np.ascontiguousarray(np.asarray(entry[key])).tobytes()
                for key in ("src_corr", "ref_corr", "scores")
            )).hexdigest()
            if actual != entry["sha256"]:
                raise PilotEvidenceError("GeoTransformer entry SHA mismatch")
        if len(src) > cap:
            keep = np.argsort(-scores, kind="stable")[:cap]
            src, ref = src[keep], ref[keep]
        source_rows.append(src if direction == "forward" else ref)
        reference_rows.append(ref if direction == "forward" else src)
        used.append((src_idx, ref_idx))
    if not source_rows:
        raise PilotEvidenceError("no usable cached correspondences")
    src_all = np.concatenate(source_rows)
    ref_all = np.concatenate(reference_rows)
    if len(permutation) != len(src_all):
        raise PilotEvidenceError("permutation length mismatch")
    return src_all[permutation], ref_all[permutation], used, failures


def surface_union(data: Mapping[str, Any], used: list[tuple[int, int]],
                  direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    objects = data["registration_pts"]
    if direction == "forward":
        source_indices = [a for a, _ in used]
        reference_indices = [b for _, b in used]
    else:
        source_indices = [b for _, b in used]
        reference_indices = [a for a, _ in used]
    source = np.concatenate([np.asarray(objects[i]) for i in source_indices])
    reference = np.concatenate([
        np.asarray(objects[i]) for i in reference_indices])
    barycentres = np.asarray([
        np.asarray(objects[i]).mean(axis=0) for i in source_indices])
    return source, reference, barycentres


def _surface_rmse(source: np.ndarray, tree: cKDTree,
                  transform: np.ndarray, threshold: float) -> tuple[float, int]:
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    distances, _ = tree.query(moved, k=1)
    keep = distances <= threshold
    count = int(keep.sum())
    if count == 0:
        return float("inf"), 0
    return float(np.sqrt(np.mean(distances[keep] ** 2))), count


def _fixed_correspondence_rmse(source: np.ndarray, reference: np.ndarray,
                               transform: np.ndarray) -> float:
    """RMSE for one immutable correspondence set under ``transform``."""
    if source.shape != reference.shape or source.ndim != 2 \
            or source.shape[1:] != (3,) or len(source) == 0:
        return float("inf")
    moved = source @ transform[:3, :3].T + transform[:3, 3]
    residual = moved - reference
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def segment_icp_with_trace(source: np.ndarray, reference: np.ndarray,
                           initial: np.ndarray, *, seed: int) -> dict:
    """Mirror the frozen segment ICP while retaining every update."""
    threshold, max_iterations, max_points = 0.20, 30, 30000
    rng = np.random.default_rng(seed)
    src = np.asarray(source, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if len(src) > max_points:
        src = src[rng.choice(len(src), max_points, replace=False)]
    if len(ref) > max_points:
        ref = ref[rng.choice(len(ref), max_points, replace=False)]
    tree = cKDTree(ref)
    transform = np.asarray(initial, dtype=np.float64).copy()
    trace = []
    for iteration in range(max_iterations):
        moved = src @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved, k=1)
        keep = distances <= threshold
        count = int(keep.sum())
        surface_before = (float(np.sqrt(np.mean(distances[keep] ** 2)))
                          if count else float("inf"))
        if count < 10:
            trace.append({
                "iteration": iteration + 1,
                "correspondences": count,
                "rmse_before_m": surface_before,
                "rmse_after_m": surface_before,
                "surface_rmse_before_m": surface_before,
                "surface_rmse_after_m": surface_before,
                "surface_correspondences_before": count,
                "surface_correspondences_after": count,
                "fixed_correspondence_rmse_before_m": surface_before,
                "fixed_correspondence_rmse_after_m": surface_before,
                "update_rotation_deg": 0.0,
                "update_translation_m": 0.0,
                "updated": False,
            })
            break
        a, b = src[keep], ref[indices[keep]]
        fixed_before = _fixed_correspondence_rmse(a, b, transform)
        centre_a, centre_b = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - centre_a).T @ (b - centre_b))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        updated = np.eye(4)
        updated[:3, :3] = rotation
        updated[:3, 3] = centre_b - rotation @ centre_a
        update_rotation, update_translation = transform_distance(
            transform, updated)
        fixed_after = _fixed_correspondence_rmse(a, b, updated)
        surface_after, surface_count_after = _surface_rmse(
            src, tree, updated, threshold)
        trace.append({
            "iteration": iteration + 1,
            "correspondences": count,
            # Preserve the historical thresholded/full-surface diagnostic.
            # Its NN/threshold set changes after the update and is not the
            # Kabsch objective used by the monotonicity safety gate.
            "rmse_before_m": surface_before,
            "rmse_after_m": surface_after,
            "surface_rmse_before_m": surface_before,
            "surface_rmse_after_m": surface_after,
            "surface_correspondences_before": count,
            "surface_correspondences_after": surface_count_after,
            # Kabsch minimizes this exact immutable correspondence objective.
            "fixed_correspondence_rmse_before_m": fixed_before,
            "fixed_correspondence_rmse_after_m": fixed_after,
            "update_rotation_deg": update_rotation,
            "update_translation_m": update_translation,
            "updated": True,
        })
        transform = updated
    final_rmse, final_count = _surface_rmse(src, tree, transform, threshold)
    fitness = float(final_count / len(src)) if len(src) else 0.0
    total_rotation, total_translation = transform_distance(initial, transform)
    return {
        "transform": transform,
        "converged": bool(fitness > 0.0 and np.isfinite(final_rmse)),
        "fitness": fitness,
        "rmse_m": final_rmse,
        "update_rotation_deg": total_rotation,
        "update_translation_m": total_translation,
        "iterations_run": len(trace),
        "trace": trace,
        "sampled_source_points": int(len(src)),
        "sampled_reference_points": int(len(ref)),
        "trace_gate_metric": "fixed_correspondence_rmse_m",
        "seed": int(seed),
    }


def rule_b_features(source: np.ndarray, reference: np.ndarray,
                    barycentres: np.ndarray, raw_transform: np.ndarray,
                    inliers: int, correspondence_count: int,
                    successful_pairs: int, failed_pairs: int,
                    icp: Mapping[str, Any], *, direction: str) -> tuple[dict, dict]:
    evidence = dfx.surface_evidence(
        source, reference, raw_transform, seed=42)
    try:
        backward = dfx.segment_icp(
            reference, source, np.linalg.inv(raw_transform), seed=43)
        bidirectional_rotation, bidirectional_translation = (
            dfx.transform_discrepancy(raw_transform, backward.transform))
        bidirectional_available = True
    except Exception:  # recorded and rejected by unchanged Rule B
        bidirectional_rotation = bidirectional_translation = None
        bidirectional_available = False
    extent, second = spatial_support(barycentres)
    denominator = successful_pairs + failed_pairs
    features = {
        "ransac_inliers": int(inliers),
        "ransac_inlier_ratio": float(inliers / max(correspondence_count, 1)),
        "spatial_extent_m": float(extent),
        "spatial_second_axis_m": float(second),
        "icp_update_translation_m": float(icp["update_translation_m"]),
        "icp_update_rotation_deg": float(icp["update_rotation_deg"]),
        "bidirectional_rotation_deg": bidirectional_rotation,
        "bidirectional_translation_m": bidirectional_translation,
        "overlap_ratio": evidence.overlap_10cm,
        "icp_converged": bool(icp["converged"]),
        "overlap_10cm": evidence.overlap_10cm,
        "overlap_5cm": evidence.overlap_5cm,
        "symmetric_trimmed_chamfer_m": evidence.symmetric_trimmed_chamfer_m,
        "median_residual_m": evidence.median_residual_m,
        "p90_residual_m": evidence.p90_residual_m,
        "icp_fitness": float(icp["fitness"]),
        "icp_rmse_m": float(icp["rmse_m"]),
        "node_pair_success_ratio": float(
            successful_pairs / denominator if denominator else 0.0),
        "successful_node_pairs": int(successful_pairs),
        "failed_node_pairs": int(failed_pairs),
        "bidirectional_available": bidirectional_available,
        "direction": direction,
    }
    evaluation = dict(features)
    if not bidirectional_available:
        evaluation["bidirectional_rotation_deg"] = 1e9
        evaluation["bidirectional_translation_m"] = 1e9
    violations = dfx.evaluate_rule_b(evaluation)
    decision = {
        "status": "accepted" if not violations else "rejected",
        "usable_for_reconstruction": not violations,
        "rejection_reasons": violations,
        "rule": "fix2-B-unchanged",
        "thresholds": dict(dfx.RULE_THRESHOLDS),
    }
    return features, decision


def run_worker(args: argparse.Namespace) -> int:
    path = cache_path(args.cache_root, args.pair)
    cached = load_validated_cache(path, args.pair, args.cache_sha256)
    data, _ = build_canonical_pair(args.pair, with_labels=False)
    validate_canonical_surfaces(data, cached)
    protocol_sha = protocol_sha256()
    if protocol_sha != args.protocol_sha256:
        raise PilotEvidenceError("protocol SHA differs from controller")

    # Determine pooled length once, then derive and apply the frozen row order.
    identity = np.arange(sum(
        min(len(cached["geot"][(a, b)]["src_corr"]),
            max(int(cached["provenance"]["matcher_contract"][
                "point_correspondence_cap"]) // len(cached["_members"]), 1))
        for a, b in cached["_members"]
        if cached["geot"].get((a, b), {}).get("status") == "ok"
    ), dtype=np.int64)
    permutation, permutation_provenance = stable_row_permutation(
        len(identity), pair_id=args.pair, direction=args.direction,
        replicate=args.replicate, protocol_sha=protocol_sha)
    src, ref, used, failures = pool_correspondences(
        cached, args.direction, permutation)
    raw_transform, inliers = ransac_from_pooled(src, ref)
    source_surface, reference_surface, barycentres = surface_union(
        data, used, args.direction)
    icp = segment_icp_with_trace(
        source_surface, reference_surface, raw_transform,
        seed=42 if args.direction == "forward" else 43)
    features, decision = rule_b_features(
        source_surface, reference_surface, barycentres, raw_transform,
        inliers, len(src), len(used), len(failures), icp,
        direction=args.direction)
    worker = {
        "schema": WORKER_SCHEMA,
        "pair_id": args.pair,
        "hypothesis": "F",
        "direction": args.direction,
        "replicate": int(args.replicate),
        "status": "ok",
        "cache": {
            "path": str(path),
            "sha256": cached["_file_sha256"],
            "schema": cached["cache_schema"],
            "input_sha256": cached["input_sha256"],
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_sha256": cached["checkpoint_sha256"],
        },
        "protocol_sha256": protocol_sha,
        "permutation_provenance_sha256": permutation_provenance,
        "permutation_sha256": hashlib.sha256(
            np.ascontiguousarray(permutation.astype(np.int64)).tobytes()
        ).hexdigest(),
        "correspondence_count": int(len(src)),
        "node_pairs_used": (
            used if args.direction == "forward"
            else [(ref_idx, src_idx) for src_idx, ref_idx in used]),
        "node_pairs_used_original_index_frame": used,
        "node_pair_failures": failures,
        "raw_transform": raw_transform,
        "raw_transform_sha256": array_sha256(raw_transform),
        "final_transform": icp["transform"],
        "final_transform_sha256": array_sha256(icp["transform"]),
        "ransac": {
            "inliers_10cm": int(inliers),
            "inlier_ratio_10cm": float(inliers / max(len(src), 1)),
        },
        "icp": icp,
        "rule_b_features": features,
        "decision": decision,
        "rule_b_accepted": bool(decision["usable_for_reconstruction"]),
        "source_hashes": {
            "runner": sha256_file(Path(__file__)),
            "consensus": sha256_file(
                CODE_ROOT / "src/safety/registration_consensus.py"),
        },
    }
    worker["evidence_sha256"] = stable_json_hash(worker)
    atomic_create_json(args.worker_out, worker)
    return 0


def worker_consensus_record(worker: Mapping[str, Any], field: str,
                            *, invert: bool = False) -> dict[str, Any]:
    transform = np.asarray(worker[field], dtype=np.float64)
    if invert:
        transform = np.linalg.inv(transform)
    return {
        "status": worker["status"],
        "transform": transform,
        "rule_b_accepted": worker["rule_b_accepted"],
        "stable_signature": worker["permutation_provenance_sha256"],
    }


def trace_gate(worker: Mapping[str, Any]) -> dict[str, Any]:
    trace = worker["icp"]["trace"]
    fixed_metric_complete = bool(trace) and all(
        "fixed_correspondence_rmse_before_m" in step
        and "fixed_correspondence_rmse_after_m" in step
        for step in trace)
    monotonic = fixed_metric_complete and all(
        np.isfinite(step["fixed_correspondence_rmse_before_m"])
        and np.isfinite(step["fixed_correspondence_rmse_after_m"])
        and step["fixed_correspondence_rmse_after_m"]
        <= step["fixed_correspondence_rmse_before_m"] + 1e-12
        for step in trace)
    last = trace[-1] if trace else {}
    stable_last_update = bool(
        last and last.get("update_rotation_deg", float("inf")) <= 0.25
        and last.get("update_translation_m", float("inf")) <= 0.005)
    reasons = []
    if not monotonic:
        reasons.append("icp_rmse_not_monotonic")
    if not stable_last_update:
        reasons.append("icp_last_update_too_large")
    return {
        "usable": not reasons,
        "rmse_non_increasing": monotonic,
        "rmse_metric": "fixed_correspondence_rmse_m",
        "fixed_metric_complete": fixed_metric_complete,
        "last_update_stable": stable_last_update,
        "rejection_reasons": reasons,
    }


def policy_name(config: ConsensusConfig) -> str:
    return (f"r{config.max_rotation_deg:g}_t{config.max_translation_m:g}_"
            f"q{config.quorum}")


def aggregate_policy(workers: list[dict], config: ConsensusConfig) -> dict:
    forward_workers = sorted(
        (row for row in workers if row["direction"] == "forward"),
        key=lambda row: row["replicate"])
    reverse_workers = sorted(
        (row for row in workers if row["direction"] == "reverse"),
        key=lambda row: row["replicate"])
    if len(forward_workers) != 5 or len(reverse_workers) != 5:
        raise PilotEvidenceError("controller requires 5+5 worker records")
    forward_raw = [worker_consensus_record(row, "raw_transform")
                   for row in forward_workers]
    forward_final = [worker_consensus_record(row, "final_transform")
                     for row in forward_workers]
    reverse_raw = [worker_consensus_record(
        row, "raw_transform", invert=True) for row in reverse_workers]
    reverse_final = [worker_consensus_record(
        row, "final_transform", invert=True) for row in reverse_workers]
    summaries = {
        "forward_raw": evaluate_direction(forward_raw, config),
        "forward_final": evaluate_direction(forward_final, config),
        "reverse_raw_inverted": evaluate_direction(reverse_raw, config),
        "reverse_final_inverted": evaluate_direction(reverse_final, config),
        "cross_raw": cross_direction_agreement(
            forward_raw, reverse_raw, config),
        "cross_final": cross_direction_agreement(
            forward_final, reverse_final, config),
    }
    fwd_medoid = summaries["forward_final"].get("medoid_original_index")
    rev_medoid = summaries["reverse_final_inverted"].get(
        "medoid_original_index")
    trace_checks = {
        "forward_medoid": (trace_gate(forward_workers[fwd_medoid])
                           if fwd_medoid is not None else {
                               "usable": False,
                               "rejection_reasons": ["missing_forward_medoid"]}),
        "reverse_medoid": (trace_gate(reverse_workers[rev_medoid])
                           if rev_medoid is not None else {
                               "usable": False,
                               "rejection_reasons": ["missing_reverse_medoid"]}),
    }
    usable = (
        all(summary["usable"] for summary in summaries.values())
        and all(check["usable"] for check in trace_checks.values())
        and fwd_medoid is not None
        and forward_workers[fwd_medoid]["rule_b_accepted"] is True
    )
    selected = None
    if fwd_medoid is not None:
        selected = {
            "forward_replicate": int(forward_workers[fwd_medoid]["replicate"]),
            "raw_transform": forward_workers[fwd_medoid]["raw_transform"],
            "final_transform": forward_workers[fwd_medoid]["final_transform"],
            "worker_evidence_sha256": forward_workers[fwd_medoid][
                "evidence_sha256"],
            "rule_b_accepted": forward_workers[fwd_medoid][
                "rule_b_accepted"],
        }
    reasons = [name for name, summary in summaries.items()
               if not summary["usable"]]
    reasons.extend(name for name, check in trace_checks.items()
                   if not check["usable"])
    return {
        "config": {
            "repeats": config.repeats,
            "quorum": config.quorum,
            "max_rotation_deg": config.max_rotation_deg,
            "max_translation_m": config.max_translation_m,
        },
        "usable_for_reconstruction": bool(usable),
        "veto_reasons": reasons,
        "consensus": summaries,
        "trace_checks": trace_checks,
        "selected_observed_forward_medoid": selected,
    }


def load_worker(path: Path, *, pair_id: str, direction: str,
                replicate: int, cache_sha: str, protocol_sha: str) -> dict:
    data = json.loads(path.read_text())
    if (data.get("schema") != WORKER_SCHEMA or data.get("pair_id") != pair_id
            or data.get("direction") != direction
            or data.get("replicate") != replicate
            or data.get("cache", {}).get("sha256") != cache_sha
            or data.get("protocol_sha256") != protocol_sha):
        raise PilotEvidenceError(f"worker provenance mismatch {path}")
    expected = data.pop("evidence_sha256", None)
    actual = stable_json_hash(data)
    data["evidence_sha256"] = expected
    if expected != actual:
        raise PilotEvidenceError(f"worker evidence SHA mismatch {path}")
    return data


def run_outer(args: argparse.Namespace, outer_repeat: int,
              repository: Mapping[str, Any]) -> Path:
    cache = cache_path(args.cache_root, args.pair)
    cache_sha = sha256_file(cache)
    protocol_sha = protocol_sha256()
    run_dir = args.out / f"outer_{outer_repeat:02d}" / args.pair
    if run_dir.exists():
        raise PilotEvidenceError(f"refusing to reuse output directory {run_dir}")
    run_dir.mkdir(parents=True)
    workers = []
    python = Path(sys.executable).resolve()
    for direction in ("forward", "reverse"):
        for replicate in range(5):
            worker_out = run_dir / f"{direction}_{replicate:02d}.json"
            command = [
                str(python), str(Path(__file__).resolve()), "--worker",
                "--pair", args.pair,
                "--direction", direction,
                "--replicate", str(replicate),
                "--cache-root", str(args.cache_root),
                "--cache-sha256", cache_sha,
                "--protocol-sha256", protocol_sha,
                "--worker-out", str(worker_out),
            ]
            environment = dict(os.environ)
            environment["OMP_NUM_THREADS"] = "1"
            completed = subprocess.run(
                command, cwd=CODE_ROOT, env=environment,
                capture_output=True, text=True)
            if completed.returncode != 0:
                raise PilotEvidenceError(
                    f"worker failed {direction}/{replicate}: "
                    f"{completed.stderr[-4000:]}")
            workers.append(load_worker(
                worker_out, pair_id=args.pair, direction=direction,
                replicate=replicate, cache_sha=cache_sha,
                protocol_sha=protocol_sha))
    if sha256_file(cache) != cache_sha:
        raise PilotEvidenceError("immutable cache changed during pilot")
    policies = {
        policy_name(config): aggregate_policy(workers, config)
        for config in POLICIES
    }
    aggregate = {
        "schema": SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "research_only": True,
        "pair_id": args.pair,
        "known_near_miss": True,
        "outer_repeat": outer_repeat,
        "repository": repository,
        "cache": {
            "path": str(cache), "sha256": cache_sha,
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_sha},
        "worker_evidence_sha256": sorted(
            row["evidence_sha256"] for row in workers),
        "workers": {
            "requested": 10,
            "completed": len(workers),
            "exceptions": 0,
            "nonfinite_transforms": int(sum(
                not np.isfinite(np.asarray(row["raw_transform"])).all()
                or not np.isfinite(np.asarray(row["final_transform"])).all()
                for row in workers)),
        },
        "policies": policies,
    }
    aggregate["evidence_sha256"] = stable_json_hash(aggregate)
    destination = run_dir / "gt_free_aggregate.json"
    atomic_create_json(destination, aggregate)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pair", default=NEAR_MISS_PAIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--direction", choices=("forward", "reverse"))
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--worker-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        required = (args.direction, args.replicate, args.cache_sha256,
                    args.protocol_sha256, args.worker_out)
        if any(value is None for value in required):
            raise PilotEvidenceError("worker arguments incomplete")
        return run_worker(args)
    if args.outer_repeats != 2:
        raise PilotEvidenceError("near-miss pilot is exactly two outer runs")
    if args.pair != NEAR_MISS_PAIR:
        raise PilotEvidenceError("this phase authorises only the known pair")
    args.cache_root = args.cache_root.resolve()
    args.out = args.out.resolve()
    if FORMAL_ROOT.resolve() == args.out or FORMAL_ROOT.resolve() in args.out.parents:
        raise PilotEvidenceError("pilot output must not enter immutable V6 outputs")
    repository = tracked_git_state()
    destinations = [run_outer(args, repeat, repository)
                    for repeat in range(args.outer_repeats)]
    receipt = {
        "schema": "v7-registration-veto-pilot-receipt-v1",
        "status": "GT_FREE_COMPLETE",
        "pair_id": args.pair,
        "outer_repeats": args.outer_repeats,
        "aggregates": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in destinations
        ],
        "posthoc_not_run": True,
    }
    atomic_create_json(args.out / "gt_free_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
