#!/usr/bin/env python3
"""GT-free node-pair rigid-subset / multi-hypothesis research runner.

The structural pass never calls RegistrationDecision and is suitable for
pre-label selection89 diagnostics.  ``--full-pair`` additionally replays each
cross-direction rigid mode through the frozen pooled RANSAC, ICP trace, Rule-B,
and V8 directional consensus.  Multiple mutually inconsistent safe modes are
rejected rather than ranked after the fact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts",
             ROOT / "src/inference/sgf_official"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from safety import decision_features as dfx  # noqa: E402
from safety.node_pair_rigid_hypotheses import (  # noqa: E402
    NodePairHypothesisConfig,
    cross_direction_mode_matches,
    disjoint_complete_linkage_modes,
    estimate_cache_node_pairs,
)
from safety.v8_stage_order_consensus import V8Config, evaluate_stage_order  # noqa: E402
SCHEMA = "v9-nodepair-multihypothesis-gtfree-v1"
DEFAULT_CACHE_ROOT = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official-v6fix-audit/outputs/"
    "official_sgaligner_v6_fix_consistency_audit_20260829/formal_v2/"
    "cache_v2/B/selection")
CACHE_SCHEMA = "v6fix-inference-cache-v2"
CHECKPOINT_ID = "B"
CHECKPOINT_SHA256 = (
    "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200")
KNOWN_BAD = (
    "6a36052f-fa53-2915-9400-831b60c63077_to_"
    "6a36052d-fa53-2915-9764-30d81b2cc2b5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(value, dtype=np.float64)).tobytes()).hexdigest()


def atomic_create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(
        jsonable(value), indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_validated_cache(path: Path, pair_id: str,
                         expected_file_sha: str) -> dict[str, Any]:
    before = sha256_file(path)
    if before != expected_file_sha:
        raise RuntimeError("cache file SHA differs from sealed manifest")
    cached = torch.load(path, map_location="cpu", weights_only=False)
    required = {"cache_schema", "pair_id", "checkpoint_id",
                "checkpoint_sha256", "input_sha256", "node_corrs",
                "provenance", "geot"}
    if not isinstance(cached, dict) or not required.issubset(cached):
        raise RuntimeError("cache schema fields missing")
    if cached["cache_schema"] != CACHE_SCHEMA:
        raise RuntimeError("cache schema mismatch")
    if cached["pair_id"] != pair_id or cached["checkpoint_id"] != CHECKPOINT_ID:
        raise RuntimeError("cache pair/checkpoint mismatch")
    if cached["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA mismatch")
    members = [(int(a), int(b)) for a, b in cached["node_corrs"]]
    if not members or len(members) != len(set(members)):
        raise RuntimeError("empty/duplicated node correspondence set")
    if sha256_file(path) != before:
        raise RuntimeError("cache changed while being read")
    cached["_file_sha256"] = before
    cached["_members"] = members
    return cached


def config_dict(config: NodePairHypothesisConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def mode_public(mode: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mode.items()
            if key != "member_indices"}


def structural_pair(cached: Mapping[str, Any],
                    config: NodePairHypothesisConfig) -> dict[str, Any]:
    forward = estimate_cache_node_pairs(
        cached, direction="forward", config=config)
    reverse = estimate_cache_node_pairs(
        cached, direction="reverse", config=config)
    forward_modes = disjoint_complete_linkage_modes(
        forward["estimates"], config)
    reverse_modes = disjoint_complete_linkage_modes(
        reverse["estimates"], config)
    matches = cross_direction_mode_matches(
        forward_modes["eligible_modes"],
        reverse_modes["eligible_modes"], config)
    left_degree, right_degree = {}, {}
    for match in matches:
        left_degree[match["forward_mode"]] = (
            left_degree.get(match["forward_mode"], 0) + 1)
        right_degree[match["reverse_mode"]] = (
            right_degree.get(match["reverse_mode"], 0) + 1)
    reasons = []
    if not matches:
        reasons.append("no_cross_direction_rigid_mode")
    if any(value > 1 for value in left_degree.values()) or any(
            value > 1 for value in right_degree.values()):
        reasons.append("cross_direction_mode_assignment_ambiguous")
    if len(matches) > 1:
        reasons.append("multiple_rigid_modes_require_rule_b_veto")
    return {
        "pair_id": cached["pair_id"],
        "cache_sha256": cached["_file_sha256"],
        "node_pairs_requested": len(cached["_members"]),
        "forward": {
            "estimated": len(forward["estimates"]),
            "failed": len(forward["failures"]),
            "modes": [mode_public(row) for row in forward_modes["modes"]],
        },
        "reverse": {
            "estimated": len(reverse["estimates"]),
            "failed": len(reverse["failures"]),
            "modes": [mode_public(row) for row in reverse_modes["modes"]],
        },
        "cross_direction_matches": matches,
        "unique_structural_mode": len(matches) == 1 and not reasons,
        "structural_rejection_reasons": reasons,
        # Kept in memory for --full-pair; stripped by jsonable artifacts.
        "_forward_estimates": forward["estimates"],
        "_reverse_estimates": reverse["estimates"],
        "_forward_eligible": forward_modes["eligible_modes"],
        "_reverse_eligible": reverse_modes["eligible_modes"],
    }


def _subset_pool(cached: Mapping[str, Any], members: Sequence[Sequence[int]],
                 direction: str, replicate: int) -> tuple[np.ndarray,
                                                              np.ndarray,
                                                              str]:
    cap_total = int(cached["provenance"]["matcher_contract"][
        "point_correspondence_cap"])
    cap = max(cap_total // len(members), 1)
    source_rows, reference_rows = [], []
    for source_index, reference_index in members:
        entry = cached["geot"][(int(source_index), int(reference_index))]
        source = np.asarray(entry["src_corr"], dtype=np.float64)
        reference = np.asarray(entry["ref_corr"], dtype=np.float64)
        scores = np.asarray(entry["scores"], dtype=np.float64)
        keep = np.argsort(-scores, kind="stable")[:cap]
        source, reference = source[keep], reference[keep]
        if direction == "reverse":
            source, reference = reference, source
        source_rows.append(source)
        reference_rows.append(reference)
    source = np.concatenate(source_rows)
    reference = np.concatenate(reference_rows)
    context = stable_json_hash({
        "schema": SCHEMA, "pair_id": cached["pair_id"],
        "members": [list(row) for row in members],
        "direction": direction, "replicate": replicate,
    })
    seed = int(context[:16], 16)
    permutation = np.random.default_rng(seed).permutation(len(source))
    return source[permutation], reference[permutation], context


def _failed_worker(pair_id: str, direction: str, replicate: int,
                   signature: str, reason: str) -> dict[str, Any]:
    identity = np.eye(4)
    return {
        "pair_id": pair_id, "direction": direction,
        "replicate": replicate, "status": "failed",
        "permutation_provenance_sha256": signature,
        "raw_transform": identity, "final_transform": identity,
        "rule_b_features": {}, "decision": {
            "rejection_reasons": [reason]}, "rule_b_accepted": False,
        "icp": {"trace": []},
        "evidence_sha256": stable_json_hash({
            "pair_id": pair_id, "direction": direction,
            "replicate": replicate, "reason": reason}),
    }


def cluster_worker(cached: Mapping[str, Any], data: Mapping[str, Any],
                   members: Sequence[Sequence[int]], direction: str,
                   replicate: int) -> dict[str, Any]:
    signature = stable_json_hash({
        "schema": SCHEMA, "members": [list(row) for row in members],
        "direction": direction, "replicate": replicate})
    try:
        from v3b_cache_runner import ransac_from_pooled
        from v7_registration_pilot import (
            rule_b_features, segment_icp_with_trace, surface_union)
        source, reference, signature = _subset_pool(
            cached, members, direction, replicate)
        raw, inliers = ransac_from_pooled(source, reference)
        original_members = [(int(a), int(b)) for a, b in members]
        source_surface, reference_surface, barycentres = surface_union(
            data, original_members, direction)
        icp = segment_icp_with_trace(
            source_surface, reference_surface, raw,
            seed=42 if direction == "forward" else 43)
        features, decision = rule_b_features(
            source_surface, reference_surface, barycentres, raw,
            inliers, len(source), len(original_members), 0, icp,
            direction=direction)
        worker = {
            "pair_id": cached["pair_id"], "direction": direction,
            "replicate": replicate, "status": "ok",
            "permutation_provenance_sha256": signature,
            "correspondence_count": len(source),
            "node_pairs_used_original_index_frame": original_members,
            "raw_transform": raw, "raw_transform_sha256": array_sha256(raw),
            "final_transform": icp["transform"],
            "final_transform_sha256": array_sha256(icp["transform"]),
            "ransac": {"inliers_10cm": inliers},
            "icp": icp, "rule_b_features": features,
            "decision": decision,
            "rule_b_accepted": bool(decision["usable_for_reconstruction"]),
        }
        worker["evidence_sha256"] = stable_json_hash(worker)
        return worker
    except Exception as exc:
        return _failed_worker(
            cached["pair_id"], direction, replicate, signature,
            f"{type(exc).__name__}:{exc}")


def full_pair(cached: Mapping[str, Any], structural: Mapping[str, Any],
              config: NodePairHypothesisConfig) -> dict[str, Any]:
    from canonical_inputs import build_canonical_pair
    from v7_registration_pilot import validate_canonical_surfaces
    data, _ = build_canonical_pair(cached["pair_id"], with_labels=False)
    validate_canonical_surfaces(data, cached)
    candidates = []
    for match_index, match in enumerate(structural[
            "cross_direction_matches"]):
        forward_mode = structural["_forward_eligible"][
            match["forward_mode"]]
        reverse_mode = structural["_reverse_eligible"][
            match["reverse_mode"]]
        workers = []
        for direction, mode in (("forward", forward_mode),
                                ("reverse", reverse_mode)):
            for replicate in range(5):
                workers.append(cluster_worker(
                    cached, data, mode["members"], direction, replicate))
        gate = evaluate_stage_order(
            workers, V8Config(repeats=5, quorum=4,
                              max_rotation_deg=config.max_rotation_deg,
                              max_translation_m=config.max_translation_m),
            dfx.evaluate_rule_b, require_fixed_trace=True)
        candidates.append({
            "candidate_index": match_index,
            "forward_members": forward_mode["members"],
            "reverse_members": reverse_mode["members"],
            "cross_nodepair_medoid": match,
            "workers": workers, "gate": gate,
        })
    safe = [row for row in candidates
            if row["gate"]["usable_for_reconstruction"]]
    reasons = []
    if not safe:
        reasons.append("no_rule_b_safe_rigid_mode")
    if len(safe) > 1:
        reasons.append("multiple_inconsistent_safe_rigid_modes")
    return {
        "candidate_count": len(candidates),
        "safe_candidate_count": len(safe),
        "usable_for_reconstruction": len(safe) == 1,
        "rejection_reasons": reasons,
        "selected_candidate_index": (
            safe[0]["candidate_index"] if len(safe) == 1 else None),
        "candidates": candidates,
    }


def clean_structural(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items()
            if not key.startswith("_")}


def manifest_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("manifest has no pairs")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path,
                        default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-pair")
    args = parser.parse_args()
    config = NodePairHypothesisConfig()
    entries = manifest_pairs(args.manifest)
    if args.full_pair:
        entries = [row for row in entries
                   if row["pair_id"] == args.full_pair]
        if len(entries) != 1:
            raise ValueError("full pair is absent/duplicated in manifest")
    rows = []
    for index, entry in enumerate(entries, 1):
        pair_id = entry["pair_id"]
        cache = args.cache_root / entry["cache_basename"]
        cached = load_validated_cache(
            cache, pair_id, entry["cache_sha256"])
        structural = structural_pair(cached, config)
        clean = clean_structural(structural)
        if args.full_pair:
            clean["full_pipeline"] = full_pair(cached, structural, config)
        rows.append(clean)
        print(f"[{index}/{len(entries)}] {pair_id} "
              f"matches={len(clean['cross_direction_matches'])}",
              flush=True)
    summary = {
        "schema": SCHEMA,
        "evidence_class": "GT-free development diagnostic only",
        "forbidden_inputs": [
            "selection labels", "GT transforms", "posthoc", "official92"],
        "config": config_dict(config),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "pair_count": len(rows),
        "pairs_with_any_cross_mode": sum(
            bool(row["cross_direction_matches"]) for row in rows),
        "pairs_with_unique_structural_mode": sum(
            row["unique_structural_mode"] for row in rows),
        "pairs_with_multiple_cross_modes": sum(
            len(row["cross_direction_matches"]) > 1 for row in rows),
        "known_bad": next(({
            "pair_id": row["pair_id"],
            "unique_structural_mode": row["unique_structural_mode"],
            "rejection_reasons": row["structural_rejection_reasons"],
        } for row in rows if row["pair_id"] == KNOWN_BAD), None),
        "pairs": rows,
    }
    summary["payload_sha256"] = stable_json_hash(summary)
    atomic_create_json(args.out, summary)
    print(json.dumps(jsonable({key: value for key, value in summary.items()
                               if key != "pairs"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
