#!/usr/bin/env python3
"""Build sealed GT-free cross-graph candidate caches and V9 diagnostics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts",
             ROOT / "src/inference/sgf_official"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from inference import geotransformer_forward  # noqa: E402
from safety.cross_graph_candidates import (  # noqa: E402
    CrossGraphCandidateConfig,
    candidate_fingerprint,
    cross_graph_candidates,
)
from v10_crossgraph_candidate_cache_helpers import atomic_torch_save  # noqa: E402
from v7_registration_pilot import validate_canonical_surfaces  # noqa: E402
from v9_nodepair_multihypothesis import (  # noqa: E402
    KNOWN_BAD,
    NodePairHypothesisConfig,
    atomic_create_json,
    clean_structural,
    jsonable,
    load_validated_cache,
    sha256_file,
    stable_json_hash,
    structural_pair,
)


SCHEMA = "v10-crossgraph-candidate-cache-v1"
DEFAULT_MANIFEST = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-v8-selection89-dev/outputs/"
    "v8_selection89_manifest_seal_v2_20260830/"
    "v8_selection89_manifest.json")


def entry_sha(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(np.asarray(entry[key])).tobytes()
        for key in ("src_corr", "ref_corr", "scores"))).hexdigest()


def validate_reused_entry(entry: Mapping[str, Any]) -> None:
    if entry.get("status") != "ok":
        return
    if entry_sha(entry) != entry.get("sha256"):
        raise RuntimeError("immutable GeoTransformer entry SHA mismatch")


def _load_existing(path: Path, *, pair_id: str, fingerprint: str,
                   source_cache_sha: str) -> dict[str, Any]:
    cached = torch.load(path, map_location="cpu", weights_only=False)
    if (cached.get("cache_schema") != SCHEMA
            or cached.get("pair_id") != pair_id
            or cached.get("candidate_fingerprint") != fingerprint
            or cached.get("source_cache_sha256") != source_cache_sha):
        raise RuntimeError(f"stale V10 cache {path}")
    cached["_file_sha256"] = sha256_file(path)
    cached["_members"] = [tuple(row) for row in cached["node_corrs"]]
    return cached


def build_pair(entry: Mapping[str, Any], cache_root: Path,
               out_cache: Path, device: str,
               config: CrossGraphCandidateConfig) -> tuple[dict, dict]:
    source_path = cache_root / entry["cache_basename"]
    source = load_validated_cache(
        source_path, entry["pair_id"], entry["cache_sha256"])
    candidates = cross_graph_candidates(
        source["rank_list"], int(source["provenance"]["src_count"]), config)
    fingerprint = candidate_fingerprint(candidates, config)
    if out_cache.exists():
        return _load_existing(
            out_cache, pair_id=entry["pair_id"], fingerprint=fingerprint,
            source_cache_sha=entry["cache_sha256"]), {
                "resumed": True, "new_geot_executed": 0,
                "old_geot_reused": 0}

    data, _ = build_canonical_pair(entry["pair_id"], with_labels=False)
    validate_canonical_surfaces(data, source)
    surfaces = data["registration_pts"]
    geot, reused, executed = {}, 0, 0
    for offset, row in enumerate(candidates, 1):
        node_pair = (row["source_index"], row["reference_index"])
        old = source["geot"].get(node_pair)
        if old is not None:
            validate_reused_entry(old)
            geot[node_pair] = copy.deepcopy(old)
            reused += 1
            continue
        source_points = surfaces.get(node_pair[0])
        reference_points = surfaces.get(node_pair[1])
        if (source_points is None or reference_points is None
                or len(source_points) < 50 or len(reference_points) < 50):
            geot[node_pair] = {"status": "insufficient_raw_points"}
            continue
        executed += 1
        try:
            status, output = geotransformer_forward(
                source_points, reference_points, device=device)
            if status != "ok" or len(output["src_corr_points"]) == 0:
                geot[node_pair] = {"status": str(status)}
            else:
                geot[node_pair] = {
                    "status": "ok",
                    "src_corr": output["src_corr_points"].astype(np.float32),
                    "ref_corr": output["ref_corr_points"].astype(np.float32),
                    "scores": output["corr_scores"].astype(np.float32),
                }
                geot[node_pair]["sha256"] = entry_sha(geot[node_pair])
        except Exception as exc:
            geot[node_pair] = {
                "status": "geotransformer_runtime_error",
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            }
        print(f"  geot {offset}/{len(candidates)}", flush=True)
    payload = {
        "cache_schema": SCHEMA,
        "pair_id": entry["pair_id"],
        "checkpoint_id": source["checkpoint_id"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "input_sha256": source["input_sha256"],
        "embedding_sha256": source.get("embedding_sha256"),
        "rank_list": source["rank_list"],
        "node_corrs": [(row["source_index"], row["reference_index"])
                       for row in candidates],
        "candidate_rank_records": candidates,
        "candidate_fingerprint": fingerprint,
        "candidate_config": {
            name: getattr(config, name) for name in config.__dataclass_fields__},
        "source_cache_path": str(source_path),
        "source_cache_sha256": entry["cache_sha256"],
        "provenance": copy.deepcopy(source["provenance"]),
        "geot": geot,
        "forbidden_inputs": [
            "selection labels", "GT transforms", "posthoc", "official92"],
    }
    payload["provenance"]["v10_candidate_contract"] = {
        "adapter_only": True,
        "default_path_unchanged": True,
        "candidate_fingerprint": fingerprint,
        "candidate_config": payload["candidate_config"],
    }
    atomic_torch_save(out_cache, payload)
    payload["_file_sha256"] = sha256_file(out_cache)
    payload["_members"] = list(payload["node_corrs"])
    return payload, {"resumed": False, "new_geot_executed": executed,
                     "old_geot_reused": reused}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-name", default="v10_manifest.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    source_manifest = json.loads(args.manifest.read_text())
    entries = source_manifest["pairs"]
    config = CrossGraphCandidateConfig()
    rigid_config = NodePairHypothesisConfig()
    cache_dir = args.out_root / "candidate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_rows, structural_rows = [], []
    total_executed = total_reused = 0
    for index, entry in enumerate(entries, 1):
        print(f"[{index}/{len(entries)}] {entry['pair_id']}", flush=True)
        path = cache_dir / f"{entry['pair_id']}.pt"
        cache, usage = build_pair(
            entry, args.cache_root, path, args.device, config)
        structural = clean_structural(structural_pair(cache, rigid_config))
        structural_rows.append(structural)
        total_executed += usage["new_geot_executed"]
        total_reused += usage["old_geot_reused"]
        pair_rows.append({
            "pair_id": entry["pair_id"], "cache_path": str(path),
            "cache_sha256": cache["_file_sha256"],
            "candidate_count": len(cache["node_corrs"]),
            "candidate_fingerprint": cache["candidate_fingerprint"],
            **usage,
            "structural_sha256": stable_json_hash(structural),
        })
    structural_payload = {
        "config": {name: getattr(config, name)
                   for name in config.__dataclass_fields__},
        "rigid_config": {name: getattr(rigid_config, name)
                         for name in rigid_config.__dataclass_fields__},
        "pairs": structural_rows,
    }
    result = {
        "schema": SCHEMA,
        "evidence_class": "GT-free development diagnostic only",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "pair_count": len(pair_rows),
        "candidate_config": structural_payload["config"],
        "resource_usage": {"new_geot_executed": total_executed,
                           "old_geot_reused": total_reused},
        "pairs_with_any_cross_mode": sum(
            bool(row["cross_direction_matches"])
            for row in structural_rows),
        "pairs_with_unique_structural_mode": sum(
            row["unique_structural_mode"] for row in structural_rows),
        "pairs_with_multiple_cross_modes": sum(
            len(row["cross_direction_matches"]) > 1
            for row in structural_rows),
        "known_bad": next(({
            "pair_id": row["pair_id"],
            "unique_structural_mode": row["unique_structural_mode"],
            "rejection_reasons": row["structural_rejection_reasons"],
        } for row in structural_rows if row["pair_id"] == KNOWN_BAD), None),
        "structural_payload_sha256": stable_json_hash(structural_payload),
        "pair_caches": pair_rows,
        "structural_pairs": structural_rows,
        "forbidden_inputs": [
            "selection labels", "GT transforms", "posthoc", "official92"],
    }
    result["payload_sha256"] = stable_json_hash(result)
    atomic_create_json(args.out_root / args.manifest_name, result)
    print(json.dumps(jsonable({key: value for key, value in result.items()
                               if key not in ("pair_caches",
                                              "structural_pairs")}),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
