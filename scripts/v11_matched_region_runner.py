#!/usr/bin/env python3
"""Sealed GT-free V11 matched-region structural/pilot runner.

The structural stage only generates deterministic multi-object hypotheses.
The pilot stage caches one independent official GeoTransformer result per
hypothesis direction, replays five pyGCRANSAC/ICP/Rule-B workers per direction,
and applies the frozen q4 V8 gate.  The full-scene union is diagnostic-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts",
             ROOT / "src/inference/sgf_official"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from inference import GEOT_SNAPSHOT, geotransformer_forward  # noqa: E402
from safety import decision_features as dfx  # noqa: E402
from safety.matched_region_hypotheses import (  # noqa: E402
    MatchedRegionConfig,
    full_scene_union,
    generate_matched_region_hypotheses,
    union_hypothesis_surfaces,
    unique_safe_hypothesis,
)
from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config, evaluate_stage_order,
)
from v10_crossgraph_candidate_cache_helpers import atomic_torch_save  # noqa: E402
from v9_nodepair_multihypothesis import (  # noqa: E402
    KNOWN_BAD, array_sha256, atomic_create_json, jsonable, sha256_file,
    stable_json_hash,
)
from v7_registration_pilot import (  # noqa: E402
    rule_b_features, segment_icp_with_trace,
)
from v3b_cache_runner import ransac_from_pooled  # noqa: E402


SCHEMA = "v11-matched-region-gtfree-v1"
CACHE_SCHEMA = "v11-region-geot-cache-v1"
WORKER_SCHEMA = "v11-region-worker-v1"
PILOT_POSITIONS = (0, 44, 88)
REPEATS = 5
Q4 = V8Config(repeats=5, quorum=4,
              max_rotation_deg=5.0, max_translation_m=0.10)
EXPECTED_RULE_B = {
    "min_overlap_10cm": 0.10,
    "max_median_residual_m": 0.10,
    "max_symmetric_trimmed_chamfer_m": 0.10,
    "max_icp_update_translation_m": 0.20,
    "max_icp_update_rotation_deg": 10.0,
    "min_icp_fitness": 0.30,
    "max_bidirectional_rotation_deg": 5.0,
    "max_bidirectional_translation_m": 0.20,
    "min_node_pair_success_ratio": 0.50,
    "min_ransac_inliers": 6,
    "min_spatial_extent_m": 1.0,
}
FORBIDDEN_INPUTS = (
    "selection labels", "GT transforms", "posthoc", "official92")


class V11EvidenceError(RuntimeError):
    """An immutable input or resumable artifact failed validation."""


def protocol_path() -> Path:
    return ROOT / "docs/V11_MATCHED_REGION_MULTI_OBJECT_PROTOCOL.md"


def protocol_sha256() -> str:
    return sha256_file(protocol_path())


def assert_frozen_rule_b() -> str:
    observed = dict(dfx.RULE_THRESHOLDS)
    if observed != EXPECTED_RULE_B:
        raise V11EvidenceError("Rule-B constants differ from pre-registration")
    return stable_json_hash(observed)


def config_dict(config: MatchedRegionConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in config.__dataclass_fields__}


def _payload_without_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items()
            if key not in ("payload_sha256", "_file_sha256")}


def _validate_payload(value: Mapping[str, Any], expected: Mapping[str, Any],
                      schema: str) -> dict[str, Any]:
    if value.get("schema") != schema:
        raise V11EvidenceError("resume schema mismatch")
    for key, item in expected.items():
        if value.get(key) != item:
            raise V11EvidenceError(f"resume contract mismatch: {key}")
    if value.get("payload_sha256") != stable_json_hash(
            _payload_without_hash(value)):
        raise V11EvidenceError("resume payload hash mismatch")
    return dict(value)


def load_or_create_json(path: Path, payload: Mapping[str, Any],
                        expected: Mapping[str, Any], schema: str) -> dict[str, Any]:
    if path.exists():
        value = json.loads(path.read_text())
        value = _validate_payload(value, expected, schema)
        if payload and value["payload_sha256"] != stable_json_hash(payload):
            raise V11EvidenceError("resume recomputation differs from artifact")
        value["_file_sha256"] = sha256_file(path)
        return value
    value = dict(payload)
    value["payload_sha256"] = stable_json_hash(value)
    atomic_create_json(path, value)
    value["_file_sha256"] = sha256_file(path)
    return value


def load_or_create_torch(path: Path, payload: Mapping[str, Any],
                         expected: Mapping[str, Any], schema: str) -> dict[str, Any]:
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=False)
        value = _validate_payload(value, expected, schema)
        value["_file_sha256"] = sha256_file(path)
        return value
    value = copy.deepcopy(dict(payload))
    value["payload_sha256"] = stable_json_hash(value)
    atomic_torch_save(path, value)
    value["_file_sha256"] = sha256_file(path)
    return value


def load_v10_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    rows = value.get("pair_caches")
    if (not isinstance(rows, list) or len(rows) != 89
            or value.get("pair_count") != 89):
        raise V11EvidenceError("V10 manifest is not the frozen 89-pair set")
    return value


def load_v10_cache(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(entry["cache_path"])
    before = sha256_file(path)
    if before != entry["cache_sha256"]:
        raise V11EvidenceError("V10 cache SHA mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {"pair_id", "candidate_rank_records", "candidate_fingerprint",
                "checkpoint_sha256", "forbidden_inputs"}
    if (not required.issubset(value)
            or value["pair_id"] != entry["pair_id"]):
        raise V11EvidenceError("V10 candidate cache contract mismatch")
    if sha256_file(path) != before:
        raise V11EvidenceError("V10 cache changed while read")
    value["_file_sha256"] = before
    return value


def _surface_hashes(surfaces: Mapping[int, Any]) -> dict[str, str]:
    return {str(index): hashlib.sha256(np.ascontiguousarray(
        np.asarray(points, dtype=np.float32)).tobytes()).hexdigest()
        for index, points in sorted(surfaces.items())}


def structural_plan(entry: Mapping[str, Any], config: MatchedRegionConfig,
                    protocol_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cached = load_v10_cache(entry)
    data, _contracts = build_canonical_pair(entry["pair_id"], with_labels=False)
    surfaces = data["registration_pts"]
    hypotheses = generate_matched_region_hypotheses(
        cached["candidate_rank_records"], surfaces, int(data["src_count"]),
        explicit_edges=data["edges_explicit"], config=config)
    plan = {
        "schema": SCHEMA,
        "evidence_class": "GT-free structural precheck only",
        "pair_id": entry["pair_id"],
        "protocol_sha256": protocol_sha,
        "config": config_dict(config),
        "v10_cache_sha256": cached["_file_sha256"],
        "candidate_fingerprint": cached["candidate_fingerprint"],
        "checkpoint_sha256": cached["checkpoint_sha256"],
        "canonical_input_sha256": stable_json_hash({
            "src_count": int(data["src_count"]),
            "surface_hashes": _surface_hashes(surfaces),
            "explicit_edges": np.asarray(data["edges_explicit"]).tolist(),
        }),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    }
    return plan, data


def write_structural(v10_manifest: Path, out_root: Path) -> dict[str, Any]:
    protocol_sha = protocol_sha256()
    config = MatchedRegionConfig()
    source = load_v10_manifest(v10_manifest)
    rows = []
    for index, entry in enumerate(source["pair_caches"], 1):
        plan, _data = structural_plan(entry, config, protocol_sha)
        expected = {"pair_id": entry["pair_id"],
                    "protocol_sha256": protocol_sha,
                    "v10_cache_sha256": entry["cache_sha256"]}
        path = out_root / "structural" / f"{entry['pair_id']}.json"
        saved = load_or_create_json(path, plan, expected, SCHEMA)
        rows.append({"pair_id": entry["pair_id"],
                     "hypothesis_count": saved["hypothesis_count"],
                     "plan_path": str(path),
                     "plan_sha256": saved["_file_sha256"]})
        print(f"[{index}/89] {entry['pair_id']} "
              f"hypotheses={saved['hypothesis_count']}", flush=True)
    summary = {
        "schema": SCHEMA,
        "stage": "structural",
        "protocol_sha256": protocol_sha,
        "v10_manifest": str(v10_manifest),
        "v10_manifest_sha256": sha256_file(v10_manifest),
        "pair_count": len(rows),
        "pairs_zero_hypotheses": sum(row["hypothesis_count"] == 0
                                     for row in rows),
        "pairs_with_hypotheses": sum(row["hypothesis_count"] > 0
                                     for row in rows),
        "total_hypotheses": sum(row["hypothesis_count"] for row in rows),
        "pairs": rows,
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    }
    return load_or_create_json(
        out_root / "structural_manifest.json", summary,
        {"stage": "structural", "protocol_sha256": protocol_sha}, SCHEMA)


def pilot_pair_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if len(rows) != 89:
        raise V11EvidenceError("pilot selection requires frozen 89-pair order")
    selected = [str(rows[index]["pair_id"]) for index in PILOT_POSITIONS]
    selected.append(KNOWN_BAD)
    output = []
    for pair_id in selected:
        if pair_id not in output:
            output.append(pair_id)
    if any(sum(row["pair_id"] == pair_id for row in rows) != 1
           for pair_id in output):
        raise V11EvidenceError("fixed pilot pair is absent or duplicated")
    return output


def _normalise_geot(status: str, output: Mapping[str, Any]) -> dict[str, Any]:
    if status != "ok":
        return {"status": str(status), "detail": jsonable(output or {})}
    keys = ("src_corr_points", "ref_corr_points", "corr_scores")
    if any(key not in output for key in keys):
        return {"status": "malformed_geot_output",
                "detail": {"missing": [key for key in keys if key not in output]}}
    source = np.asarray(output[keys[0]], dtype=np.float32)
    reference = np.asarray(output[keys[1]], dtype=np.float32)
    scores = np.asarray(output[keys[2]], dtype=np.float32)
    if (source.ndim != 2 or source.shape != reference.shape
            or source.shape[1:] != (3,) or scores.shape != (len(source),)
            or not np.isfinite(source).all()
            or not np.isfinite(reference).all()
            or not np.isfinite(scores).all() or len(source) == 0):
        return {"status": "malformed_geot_output", "detail": {}}
    return {"status": "ok", "src_corr": source,
            "ref_corr": reference, "scores": scores,
            "correspondence_sha256": hashlib.sha256(
                source.tobytes() + reference.tobytes()
                + scores.tobytes()).hexdigest()}


def build_region_cache(pair_id: str, hypothesis: Mapping[str, Any],
                       surfaces: Mapping[int, Any], path: Path,
                       protocol_sha: str, checkpoint_sha: str,
                       device: str) -> dict[str, Any]:
    expected = {"pair_id": pair_id,
                "hypothesis_sha256": hypothesis["hypothesis_sha256"],
                "protocol_sha256": protocol_sha,
                "checkpoint_sha256": checkpoint_sha,
                "arm": "matched_region"}
    if path.exists():
        return load_or_create_torch(path, {}, expected, CACHE_SCHEMA)
    source, reference = union_hypothesis_surfaces(hypothesis, surfaces)
    forward = _normalise_geot(*geotransformer_forward(
        source, reference, device=device))
    reverse = _normalise_geot(*geotransformer_forward(
        reference, source, device=device))
    payload = {
        "schema": CACHE_SCHEMA, **expected,
        "members": hypothesis["members"],
        "surface_sha256": {
            "source": hashlib.sha256(source.tobytes()).hexdigest(),
            "reference": hashlib.sha256(reference.tobytes()).hexdigest()},
        "surface_points": {"source": len(source), "reference": len(reference)},
        "forward": forward, "reverse": reverse,
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    }
    return load_or_create_torch(path, payload, expected, CACHE_SCHEMA)


def build_diagnostic_cache(pair_id: str, surfaces: Mapping[int, Any],
                           src_count: int, path: Path, protocol_sha: str,
                           checkpoint_sha: str, device: str) -> dict[str, Any]:
    expected = {"pair_id": pair_id, "protocol_sha256": protocol_sha,
                "checkpoint_sha256": checkpoint_sha,
                "arm": "whole_scene_diagnostic_only"}
    if path.exists():
        return load_or_create_torch(path, {}, expected, CACHE_SCHEMA)
    source, reference = full_scene_union(surfaces, src_count)
    payload = {
        "schema": CACHE_SCHEMA, **expected,
        "selector_eligible": False,
        "surface_sha256": {
            "source": hashlib.sha256(source.tobytes()).hexdigest(),
            "reference": hashlib.sha256(reference.tobytes()).hexdigest()},
        "surface_points": {"source": len(source), "reference": len(reference)},
        "forward": _normalise_geot(*geotransformer_forward(
            source, reference, device=device)),
        "reverse": _normalise_geot(*geotransformer_forward(
            reference, source, device=device)),
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
    }
    return load_or_create_torch(path, payload, expected, CACHE_SCHEMA)


def _worker_failure(pair_id: str, hypothesis_sha: str, direction: str,
                    replicate: int, reason: str, signature: str) -> dict[str, Any]:
    identity = np.eye(4)
    row = {
        "schema": WORKER_SCHEMA, "pair_id": pair_id,
        "hypothesis_sha256": hypothesis_sha, "direction": direction,
        "replicate": replicate, "status": "failed",
        "permutation_provenance_sha256": signature,
        "raw_transform": identity, "final_transform": identity,
        "rule_b_features": {}, "rule_b_accepted": False,
        "decision": {"rejection_reasons": [reason]}, "icp": {"trace": []},
    }
    row["evidence_sha256"] = stable_json_hash(row)
    return row


def run_worker(pair_id: str, hypothesis: Mapping[str, Any],
               cache: Mapping[str, Any], surfaces: Mapping[int, Any],
               direction: str, replicate: int, protocol_sha: str) -> dict[str, Any]:
    signature = stable_json_hash({
        "schema": WORKER_SCHEMA, "pair_id": pair_id,
        "hypothesis_sha256": hypothesis["hypothesis_sha256"],
        "direction": direction, "replicate": replicate,
        "protocol_sha256": protocol_sha})
    geot = cache[direction]
    if geot.get("status") != "ok":
        return _worker_failure(
            pair_id, hypothesis["hypothesis_sha256"], direction, replicate,
            f"geot:{geot.get('status', 'missing')}", signature)
    source = np.asarray(geot["src_corr"], dtype=np.float64)
    reference = np.asarray(geot["ref_corr"], dtype=np.float64)
    scores = np.asarray(geot["scores"], dtype=np.float64)
    keep = np.argsort(-scores, kind="stable")[:1000]
    source, reference = source[keep], reference[keep]
    permutation = np.random.default_rng(
        int(signature[:16], 16)).permutation(len(source))
    source, reference = source[permutation], reference[permutation]
    try:
        raw, inliers = ransac_from_pooled(source, reference)
        f_surface, r_surface = union_hypothesis_surfaces(
            hypothesis, surfaces)
        members = [tuple(row) for row in hypothesis["members"]]
        if direction == "reverse":
            f_surface, r_surface = r_surface, f_surface
            barycentres = np.asarray(
                [np.asarray(surfaces[b]).mean(axis=0) for _a, b in members])
        else:
            barycentres = np.asarray(
                [np.asarray(surfaces[a]).mean(axis=0) for a, _b in members])
        icp = segment_icp_with_trace(
            f_surface, r_surface, raw,
            seed=42 if direction == "forward" else 43)
        features, decision = rule_b_features(
            f_surface, r_surface, barycentres, raw, inliers, len(source),
            len(members), 0, icp, direction=direction)
        row = {
            "schema": WORKER_SCHEMA, "pair_id": pair_id,
            "hypothesis_sha256": hypothesis["hypothesis_sha256"],
            "direction": direction, "replicate": replicate, "status": "ok",
            "permutation_provenance_sha256": signature,
            "permutation_sha256": hashlib.sha256(np.ascontiguousarray(
                permutation.astype(np.int64)).tobytes()).hexdigest(),
            "correspondence_count": len(source),
            "raw_transform": raw, "raw_transform_sha256": array_sha256(raw),
            "final_transform": icp["transform"],
            "final_transform_sha256": array_sha256(icp["transform"]),
            "ransac": {"inliers_10cm": int(inliers)}, "icp": icp,
            "rule_b_features": features, "decision": decision,
            "rule_b_accepted": bool(decision["usable_for_reconstruction"]),
        }
        row["evidence_sha256"] = stable_json_hash(row)
        return row
    except Exception as exc:
        return _worker_failure(
            pair_id, hypothesis["hypothesis_sha256"], direction, replicate,
            f"{type(exc).__name__}:{exc}", signature)


def selector_from_hypothesis_gates(
    gates: Sequence[Mapping[str, Any]], *, known_bad: bool,
) -> dict[str, Any]:
    """Whole-scene diagnostics are intentionally absent from this API."""
    rows = [{
        "hypothesis_sha256": row["hypothesis_sha256"],
        "forward_status": "ok" if row["worker_failures"] == 0 else "failed",
        "reverse_status": "ok" if row["worker_failures"] == 0 else "failed",
        "cross_direction_consistent": bool(
            row["gate"]["cross_final"]["usable"]),
        "rule_b_safe": bool(
            row["gate"]["medoid_rule_b"]["forward"]["usable"]
            and row["gate"]["medoid_rule_b"]["reverse"]["usable"]),
        "q4_stable": bool(row["gate"]["fresh_v8_qualified"]),
    } for row in gates]
    return unique_safe_hypothesis(rows, known_bad=known_bad)


def pilot_pair(entry: Mapping[str, Any], plan: Mapping[str, Any],
               out_root: Path, device: str) -> dict[str, Any]:
    pair_id = entry["pair_id"]
    protocol_sha = protocol_sha256()
    assert_frozen_rule_b()
    data, _contracts = build_canonical_pair(pair_id, with_labels=False)
    surfaces = data["registration_pts"]
    gates, artifacts, unknown = [], [], []
    for hypothesis in plan["hypotheses"]:
        hypothesis_sha = hypothesis["hypothesis_sha256"]
        cache_path = out_root / "region_cache" / pair_id / f"{hypothesis_sha}.pt"
        cache = build_region_cache(
            pair_id, hypothesis, surfaces, cache_path, protocol_sha,
            plan["checkpoint_sha256"], device)
        artifacts.append({"path": str(cache_path),
                          "sha256": cache["_file_sha256"],
                          "kind": "matched_region_geot"})
        workers = []
        for direction in ("forward", "reverse"):
            for replicate in range(REPEATS):
                worker_path = (out_root / "workers" / pair_id / hypothesis_sha
                               / f"{direction}_{replicate}.json")
                expected = {"pair_id": pair_id,
                            "hypothesis_sha256": hypothesis_sha,
                            "direction": direction, "replicate": replicate}
                payload = ({} if worker_path.exists() else run_worker(
                    pair_id, hypothesis, cache, surfaces, direction,
                    replicate, protocol_sha))
                worker = load_or_create_json(
                    worker_path, payload, expected, WORKER_SCHEMA)
                workers.append(worker)
                artifacts.append({"path": str(worker_path),
                                  "sha256": worker["_file_sha256"],
                                  "kind": "worker"})
                reason = worker.get("decision", {}).get("rejection_reasons", [])
                if worker["status"] == "failed" and any(
                        not str(item).startswith(("geot:", "RuntimeError:",
                                                  "ValueError:"))
                        for item in reason):
                    unknown.extend(reason)
        gate = evaluate_stage_order(
            workers, Q4, dfx.evaluate_rule_b, require_fixed_trace=True)
        gate_payload = {
            "schema": SCHEMA, "stage": "hypothesis_gate",
            "pair_id": pair_id, "hypothesis_sha256": hypothesis_sha,
            "protocol_sha256": protocol_sha,
            "worker_failures": sum(row["status"] != "ok" for row in workers),
            "worker_evidence_sha256": [row["evidence_sha256"] for row in workers],
            "gate": gate,
        }
        gate_path = out_root / "gates" / pair_id / f"{hypothesis_sha}.json"
        saved_gate = load_or_create_json(
            gate_path, gate_payload,
            {"pair_id": pair_id, "hypothesis_sha256": hypothesis_sha,
             "protocol_sha256": protocol_sha}, SCHEMA)
        artifacts.append({"path": str(gate_path),
                          "sha256": saved_gate["_file_sha256"],
                          "kind": "hypothesis_gate"})
        gates.append(saved_gate)
    diagnostic_path = out_root / "diagnostic" / f"{pair_id}.pt"
    diagnostic = build_diagnostic_cache(
        pair_id, surfaces, int(data["src_count"]), diagnostic_path,
        protocol_sha, plan["checkpoint_sha256"], device)
    artifacts.append({"path": str(diagnostic_path),
                      "sha256": diagnostic["_file_sha256"],
                      "kind": "whole_scene_diagnostic_only"})
    selector = selector_from_hypothesis_gates(
        gates, known_bad=pair_id == KNOWN_BAD)
    return {
        "pair_id": pair_id, "hypothesis_count": len(plan["hypotheses"]),
        "gates": gates, "selector": selector,
        "whole_scene_diagnostic": {
            "selector_eligible": False,
            "forward_status": diagnostic["forward"]["status"],
            "reverse_status": diagnostic["reverse"]["status"],
            "artifact_sha256": diagnostic["_file_sha256"]},
        "unknown_errors": unknown, "artifacts": artifacts,
    }


def run_pilot(v10_manifest: Path, out_root: Path, device: str) -> dict[str, Any]:
    protocol_sha = protocol_sha256()
    source = load_v10_manifest(v10_manifest)
    ids = pilot_pair_ids(source["pair_caches"])
    by_id = {row["pair_id"]: row for row in source["pair_caches"]}
    structural_manifest = json.loads(
        (out_root / "structural_manifest.json").read_text())
    plan_paths = {row["pair_id"]: Path(row["plan_path"])
                  for row in structural_manifest["pairs"]}
    rows = []
    for index, pair_id in enumerate(ids, 1):
        plan = json.loads(plan_paths[pair_id].read_text())
        rows.append(pilot_pair(by_id[pair_id], plan, out_root, device))
        print(f"pilot [{index}/{len(ids)}] {pair_id} "
              f"accepted={rows[-1]['selector']['accepted']}", flush=True)
    known = next(row for row in rows if row["pair_id"] == KNOWN_BAD)
    summary = {
        "schema": SCHEMA, "stage": "pilot", "protocol_sha256": protocol_sha,
        "rule_b_sha256": assert_frozen_rule_b(),
        "pilot_positions": list(PILOT_POSITIONS), "pilot_pair_ids": ids,
        "pair_count": len(rows),
        "zero_unknown_errors": not any(row["unknown_errors"] for row in rows),
        "known_bad_veto": not known["selector"]["accepted"],
        "unique_safe_coverage": sum(row["selector"]["accepted"] for row in rows),
        "pairs": rows, "forbidden_inputs": list(FORBIDDEN_INPUTS),
    }
    return load_or_create_json(
        out_root / "pilot_manifest.json", summary,
        {"stage": "pilot", "protocol_sha256": protocol_sha}, SCHEMA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("structural", "pilot"), required=True)
    parser.add_argument("--v10-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage == "structural":
        result = write_structural(args.v10_manifest, args.out_root)
    else:
        result = run_pilot(args.v10_manifest, args.out_root, args.device)
    print(json.dumps(jsonable({key: value for key, value in result.items()
                               if key != "pairs"}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
