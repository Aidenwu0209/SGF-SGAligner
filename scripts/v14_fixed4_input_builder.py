#!/usr/bin/env python3
"""Seal exact V13 fixed4 solver caches into the V14 4x2 input manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from safety.v13_dual_solver_runtime import (
    array_sha256, atomic_json, sha256_file, stable_json_sha256,
)
from safety.v14_rigid_multihypothesis import verify_candidate_set_contract
from scripts.v13_corr_cache_converter import FROZEN_NEIGHBOR_LIMITS
from scripts.v13_formal_source_manifest import formal_source_sha256 as v13_sources
from scripts.v14_formal_source_manifest import (
    formal_source_sha256, verify_reviewed_source_authorization,
)


class Fixed4InputError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise Fixed4InputError(f"JSON object required: {path}")
    return value


def _payload(value: dict, *, schema: str, name: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("payload_sha256", None)
    if value.get("schema") != schema or observed != stable_json_sha256(unsigned):
        raise Fixed4InputError(f"{name} payload closure mismatch")


def _npz_arrays(path: Path, keys: set[str]) -> dict[str, np.ndarray]:
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != keys:
            raise Fixed4InputError(f"NPZ schema mismatch: {path}")
        result = {key: np.ascontiguousarray(np.asarray(data[key])) for key in keys}
    if sha256_file(path) != before:
        raise Fixed4InputError(f"NPZ changed while reading: {path}")
    return result


def verify_conversion_lineage(
    *, repo: Path, cache_path: Path, receipt_path: Path,
    prepared_path: Path, prepared_record: dict, pair_id: str,
    arm: str, direction: str,
) -> dict[str, Any]:
    """Verify converter receipt -> two-sentinel cache -> prepared input."""
    cache_path = Path(cache_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    prepared_path = Path(prepared_path).resolve()
    if not all(path.is_file() for path in (cache_path, receipt_path, prepared_path)):
        raise Fixed4InputError("V13 lineage artifact is missing")
    prepared_sha = sha256_file(prepared_path)
    if (prepared_sha != prepared_record.get("prepared_npz_sha256")
            or str(prepared_path) != str(Path(
                prepared_record.get("prepared_npz_path", "")).resolve())):
        raise Fixed4InputError("prepared input differs from preflight record")
    prepared = _npz_arrays(prepared_path, set(
        np.load(prepared_path, allow_pickle=False).files))
    if "manifest_json" not in prepared:
        raise Fixed4InputError("prepared input manifest_json missing")
    prepared_manifest = json.loads(str(prepared["manifest_json"].item()))
    _payload(prepared_manifest, schema="v13-color-preserving-pair-v2",
             name="prepared pair")
    if (prepared_manifest.get("pair_id") != pair_id
            or prepared_manifest.get("payload_sha256")
            != prepared_record.get("payload_sha256")):
        raise Fixed4InputError("prepared pair identity/payload mismatch")
    receipt = load_json(receipt_path)
    exact = {
        "schema": "v13-colorpcr-corr-conversion-receipt-v1",
        "pair_id": pair_id, "arm": arm, "direction": direction,
        "prepared_input": str(prepared_path),
        "prepared_input_sha256": prepared_sha,
        "prepared_manifest_payload_sha256": prepared_manifest["payload_sha256"],
        "output_cache": str(cache_path),
        "output_cache_sha256": sha256_file(cache_path),
        "estimated_transform_discarded": True,
        "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
        "sampling": "voxel10", "coarsest_cap": 512,
        "gt_consumed": False, "fallback_used": False,
        "converter_sha256": v13_sources(repo)["converter"],
    }
    if any(receipt.get(key) != item for key, item in exact.items()):
        raise Fixed4InputError("V13 conversion receipt mismatch")
    cache = _npz_arrays(cache_path, {"src_corr", "ref_corr", "scores"})
    if receipt.get("output_keys") != ["src_corr", "ref_corr", "scores"] \
            or receipt.get("output_array_sha256") != {
                key: array_sha256(cache[key]) for key in cache}:
        raise Fixed4InputError("three-key cache arrays differ from receipt")
    sentinel_path = Path(str(receipt.get("source_sentinel_cache", ""))).resolve()
    if (not sentinel_path.is_file()
            or sha256_file(sentinel_path)
            != receipt.get("source_sentinel_cache_sha256")):
        raise Fixed4InputError("source sentinel cache closure mismatch")
    sentinel = _npz_arrays(sentinel_path, {
        "src_corr_points", "ref_corr_points", "corr_scores",
        "estimated_transform", "meta_json"})
    meta = json.loads(str(sentinel["meta_json"].item()))
    worker = meta.get("worker_contract", {})
    if (meta.get("schema") != "v13-colorpcr-corr-cache-v2"
            or meta.get("sentinel_invariant") is not True
            or meta.get("gt_consumed") is not False
            or meta.get("identity_fallback") is not False
            or meta.get("input_sha256") != prepared_sha
            or worker.get("arm") != arm or worker.get("direction") != direction
            or worker.get("neighbor_limits") != FROZEN_NEIGHBOR_LIMITS
            or worker.get("sampling") != "voxel10"
            or worker.get("coarsest_cap") != 512):
        raise Fixed4InputError("source sentinel metadata mismatch")
    source_arrays = {
        "src_corr_points": sentinel["src_corr_points"],
        "ref_corr_points": sentinel["ref_corr_points"],
        "corr_scores": sentinel["corr_scores"],
        "estimated_transform": sentinel["estimated_transform"],
    }
    if receipt.get("source_array_sha256") != {
            key: array_sha256(item) for key, item in source_arrays.items()}:
        raise Fixed4InputError("source sentinel arrays differ from receipt")
    mapping = {"src_corr": "src_corr_points", "ref_corr": "ref_corr_points",
               "scores": "corr_scores"}
    if any(cache[out].dtype != source_arrays[src].dtype
           or cache[out].shape != source_arrays[src].shape
           or not np.array_equal(cache[out], source_arrays[src])
           for out, src in mapping.items()):
        raise Fixed4InputError("three-key cache is not the sentinel projection")
    artifact_paths = receipt.get("sentinel_artifact_path", {})
    artifact_hashes = receipt.get("sentinel_artifact_sha256", {})
    if set(artifact_paths) != {"identity", "proper_nonzero"} \
            or set(artifact_hashes) != set(artifact_paths):
        raise Fixed4InputError("two-sentinel artifacts are missing")
    for name, raw_path in artifact_paths.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file() or sha256_file(path) != artifact_hashes[name]:
            raise Fixed4InputError(f"sentinel artifact closure mismatch: {name}")
    return {
        "cache_path": str(cache_path), "cache_sha256": sha256_file(cache_path),
        "conversion_receipt_path": str(receipt_path),
        "conversion_receipt_sha256": sha256_file(receipt_path),
        "source_sentinel_cache_path": str(sentinel_path),
        "source_sentinel_cache_sha256": sha256_file(sentinel_path),
    }


def verify_v13_fixed4_root(
    root: Path, *, repo: Path, v13_preregister: Path, preflight: Path,
    pairs: list[str], arms: tuple[str, str],
) -> dict[str, Any]:
    """Bind the immutable V13 closure and all eight formal pair receipts."""
    root = Path(root).resolve()
    artifact_path, closure_path = (root / "artifact_manifest.json",
                                   root / "closure.json")
    artifact, closure = load_json(artifact_path), load_json(closure_path)
    if (closure.get("schema") != "v13-fixed4-closure-v1"
            or closure.get("artifact_manifest_sha256")
            != sha256_file(artifact_path)
            or closure.get("summary_sha256") != sha256_file(root / "summary.json")
            or closure.get("preregister_sha256") != sha256_file(v13_preregister)
            or closure.get("preflight_sha256") != sha256_file(preflight)):
        raise Fixed4InputError("V13 fixed4 closure mismatch")
    if artifact.get("schema") != "v13-fixed4-artifact-manifest-v1":
        raise Fixed4InputError("V13 artifact manifest schema mismatch")
    for relative, record in artifact.get("files", {}).items():
        path = (root / relative).resolve()
        if (root not in path.parents or not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")):
            raise Fixed4InputError("V13 artifact manifest mismatch")
    receipts = {}
    expected_sources = v13_sources(repo)
    for pair_id in pairs:
        for arm in arms:
            path = root / "pair_receipts" / f"{pair_id}.{arm}.json"
            receipt = load_json(path)
            if (receipt.get("schema") != "v13-fixed4-pair-receipt-v1"
                    or receipt.get("pair_id") != pair_id
                    or receipt.get("arm") != arm
                    or receipt.get("formal_source_sha256") != expected_sources
                    or receipt.get("preregister_sha256")
                    != sha256_file(v13_preregister)
                    or receipt.get("preflight_sha256") != sha256_file(preflight)
                    or sha256_file(Path(receipt.get("summary_path", "")))
                    != receipt.get("summary_sha256")):
                raise Fixed4InputError("V13 pair receipt closure mismatch")
            receipts[(pair_id, arm)] = {
                "path": str(path), "sha256": sha256_file(path)}
    return {
        "root": str(root), "artifact_manifest_path": str(artifact_path),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "closure_path": str(closure_path), "closure_sha256": sha256_file(closure_path),
        "pair_receipts": receipts,
    }


def build_fixed4_inputs(
    *, repo: Path, v13_root: Path, candidate_root: Path,
    v13_preregister: Path, v14_preregister: Path, preflight: Path,
    output: Path,
) -> dict[str, Any]:
    prereg = load_json(v14_preregister)
    if prereg.get("allow_real_pilot") is not True:
        raise Fixed4InputError("V14 real CPU pilot is not explicitly authorized")
    verify_reviewed_source_authorization(repo, prereg)
    preflight_value = load_json(preflight)
    _payload(preflight_value, schema="v13-colorpcr-pointdsc-shadow-v2",
             name="V13 preflight")
    pairs = list(prereg.get("fixed_pair_order", ()))
    arms = (str(prereg.get("primary_arm")), str(prereg.get("control_arm")))
    if len(pairs) != 4 or preflight_value.get("pair_ids") != pairs:
        raise Fixed4InputError("fixed4 pair order mismatch")
    preflight_by_pair = {row["pair_id"]: row
                         for row in preflight_value.get("pairs", ())}
    v13_binding = verify_v13_fixed4_root(
        v13_root, repo=repo, v13_preregister=v13_preregister,
        preflight=preflight,
        pairs=pairs, arms=arms)
    rows = []
    for pair_id in pairs:
        prepared_record = preflight_by_pair[pair_id]
        prepared = Path(prepared_record["prepared_npz_path"]).resolve()
        for arm in arms:
            direction_lineage = {}
            for direction in ("forward", "reverse"):
                cache = (Path(v13_root).resolve() / "pairs" / pair_id / arm
                         / "solver_cache" / f"{direction}.three_key.npz")
                direction_lineage[direction] = verify_conversion_lineage(
                    repo=repo, cache_path=cache,
                    receipt_path=cache.with_suffix(".receipt.json"),
                    prepared_path=prepared, prepared_record=prepared_record,
                    pair_id=pair_id, arm=arm, direction=direction)
            if (direction_lineage["forward"]["cache_path"]
                    == direction_lineage["reverse"]["cache_path"]
                    or direction_lineage["forward"]["cache_sha256"]
                    == direction_lineage["reverse"]["cache_sha256"]
                    or direction_lineage["forward"]["source_sentinel_cache_sha256"]
                    == direction_lineage["reverse"]["source_sentinel_cache_sha256"]):
                raise Fixed4InputError("forward/reverse lineage is not independent")
            candidate_set = (Path(candidate_root).resolve() / "pairs" / pair_id
                             / arm / "candidate_set.json")
            verified = verify_candidate_set_contract(candidate_set)
            value = verified["value"]
            if value.get("pair_id") != pair_id or value.get("arm") != arm:
                raise Fixed4InputError("candidate set fixed4 identity mismatch")
            for direction in ("forward", "reverse"):
                direction_manifest = verified["direction_manifests"][direction]
                if (direction_manifest.get("source_cache_path")
                        != direction_lineage[direction]["cache_path"]
                        or direction_manifest.get("source_cache_sha256")
                        != direction_lineage[direction]["cache_sha256"]):
                    raise Fixed4InputError(
                        "candidate source differs from V13 conversion lineage")
            rows.append({
                "pair_id": pair_id, "arm": arm,
                "candidate_set_path": str(candidate_set),
                "candidate_set_sha256": sha256_file(candidate_set),
                "prepared_input_path": str(prepared),
                "prepared_input_sha256": sha256_file(prepared),
                "direction_lineage": direction_lineage,
                "v13_pair_receipt": v13_binding["pair_receipts"][(pair_id, arm)],
            })
    unsigned = {
        "schema": "v14-fixed4-candidate-inputs-v1",
        "v14_preregister_path": str(Path(v14_preregister).resolve()),
        "v14_preregister_sha256": sha256_file(v14_preregister),
        "v13_preregister_path": str(Path(v13_preregister).resolve()),
        "v13_preregister_sha256": sha256_file(v13_preregister),
        "preflight_manifest_path": str(Path(preflight).resolve()),
        "preflight_manifest_sha256": sha256_file(preflight),
        "formal_source_sha256": formal_source_sha256(repo),
        "v13_fixed4_binding": {key: item for key, item in v13_binding.items()
                               if key != "pair_receipts"},
        "rows": rows,
    }
    value = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
    atomic_json(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--v13-fixed4-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--v13-preregister", type=Path, required=True)
    parser.add_argument("--v14-preregister", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_fixed4_inputs(
        repo=args.repo, v13_root=args.v13_fixed4_root,
        candidate_root=args.candidate_root,
        v13_preregister=args.v13_preregister,
        v14_preregister=args.v14_preregister,
        preflight=args.preflight_manifest, output=args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
