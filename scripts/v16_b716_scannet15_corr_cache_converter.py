#!/usr/bin/env python3
"""ScanNet15 sibling for the frozen V13 correspondence-cache converter.

The historical V13 converter remains byte-identical because its source SHA is
part of the fixed4 formal closure.  This sibling preserves the exact numerical
conversion and frozen resource checks, but replaces only the prepared-input
identity check with the preregistered ScanNet15 schema contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.v13_corr_cache_converter import (
    FROZEN_NEIGHBOR_LIMITS, OUTPUT_KEYS, SOURCE_KEYS,
    ConversionContractError, _atomic_json, _atomic_npz, array_sha256,
    sha256_file,
)
from safety.v16_b716_scannet15_identity import (
    ScanNet15IdentityError, validate_prepared_npz, validate_preregister,
)


def convert(
    source: Path, prepared_input: Path, output: Path, receipt_path: Path, *,
    pair_id: str, arm: str, direction: str, identity_preregister_path: Path,
) -> dict[str, Any]:
    source = Path(source).resolve()
    prepared_input = Path(prepared_input).resolve()
    output = Path(output).resolve()
    receipt_path = Path(receipt_path).resolve()
    identity_preregister_path = Path(identity_preregister_path).resolve()
    if arm not in ("sgf_selected_union", "fullscan"):
        raise ConversionContractError("unknown arm")
    if direction not in ("forward", "reverse"):
        raise ConversionContractError("unknown direction")
    if pair_id.count("_to_") != 1:
        raise ConversionContractError("pair id must be exact src_to_ref")
    if output.exists() or receipt_path.exists():
        raise ConversionContractError(
            "ScanNet15 conversion outputs are create-only")
    if (not identity_preregister_path.is_file()
            or identity_preregister_path.is_symlink()):
        raise ConversionContractError(
            "ScanNet15 identity preregister is not a regular file")
    preregister_sha = sha256_file(identity_preregister_path)
    source_sha = sha256_file(source)
    prepared_sha = sha256_file(prepared_input)
    with np.load(prepared_input, allow_pickle=False) as prepared:
        if "manifest_json" not in prepared.files:
            raise ConversionContractError("prepared input manifest_json missing")
        prepared_manifest = json.loads(str(prepared["manifest_json"].item()))
    try:
        preregister = json.loads(identity_preregister_path.read_text())
        validate_preregister(preregister)
        pair_identity, prepared_validation = validate_prepared_npz(
            prepared_input, pair_id=pair_id, preregister=preregister)
    except (json.JSONDecodeError, ScanNet15IdentityError) as exc:
        raise ConversionContractError(
            f"ScanNet15 prepared identity mismatch: {exc}") from exc
    prepared_payload_sha = prepared_manifest["payload_sha256"]
    if sha256_file(identity_preregister_path) != preregister_sha:
        raise ConversionContractError(
            "ScanNet15 identity preregister changed while reading")
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != SOURCE_KEYS:
            raise ConversionContractError("sentinel cache schema is not exact")
        meta = json.loads(str(data["meta_json"].item()))
        raw = {key: np.asarray(data[key])
               for key in SOURCE_KEYS - {"meta_json"}}
    if sha256_file(source) != source_sha:
        raise ConversionContractError("sentinel cache changed while reading")
    if (meta.get("schema") != "v13-colorpcr-corr-cache-v2"
            or meta.get("sentinel_invariant") is not True
            or meta.get("gt_consumed") is not False
            or meta.get("identity_fallback") is not False):
        raise ConversionContractError("ColorPCR sentinel gate not sealed")
    worker = meta.get("worker_contract", {})
    if worker.get("arm") != arm or worker.get("direction") != direction:
        raise ConversionContractError("arm/direction metadata mismatch")
    if meta.get("input_sha256") != prepared_sha:
        raise ConversionContractError("prepared input SHA mismatch")
    if (worker.get("neighbor_limits") != FROZEN_NEIGHBOR_LIMITS
            or worker.get("sampling") != "voxel10"
            or worker.get("coarsest_cap") != 512):
        raise ConversionContractError(
            "frozen ColorPCR resource contract mismatch")
    sentinel_paths = meta.get("sentinel_artifact_path", {})
    sentinel_hashes = meta.get("sentinel_artifact_sha256", {})
    if (set(sentinel_paths) != {"identity", "proper_nonzero"}
            or set(sentinel_hashes) != set(sentinel_paths)):
        raise ConversionContractError(
            "persistent two-sentinel evidence is missing")
    for name, raw_path in sentinel_paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or sha256_file(path) != sentinel_hashes[name]:
            raise ConversionContractError(
                f"sentinel evidence closure mismatch: {name}")
    src = np.asarray(raw["src_corr_points"])
    ref = np.asarray(raw["ref_corr_points"])
    scores = np.asarray(raw["corr_scores"])
    if (src.ndim != 2 or src.shape[1:] != (3,) or ref.shape != src.shape
            or scores.shape != (len(src),) or len(src) < 40):
        raise ConversionContractError(
            "correspondence arrays are not aligned")
    if not all(np.issubdtype(value.dtype, np.floating)
               and np.isfinite(value).all()
               for value in (src, ref, scores)):
        raise ConversionContractError(
            "correspondences must be finite floating point")
    arrays = {
        "src_corr": np.ascontiguousarray(src),
        "ref_corr": np.ascontiguousarray(ref),
        "scores": np.ascontiguousarray(scores),
    }
    _atomic_npz(output, arrays)
    with np.load(output, allow_pickle=False) as check:
        if set(check.files) != set(OUTPUT_KEYS):
            raise ConversionContractError("converted cache is not exact-three")
    receipt = {
        "schema": "v16-b716-scannet15-colorpcr-corr-conversion-receipt-v1",
        "pair_id": pair_id, "arm": arm, "direction": direction,
        "source_sentinel_cache": str(source),
        "source_sentinel_cache_sha256": source_sha,
        "prepared_input": str(prepared_input),
        "prepared_input_sha256": prepared_sha,
        "prepared_manifest_payload_sha256": prepared_payload_sha,
        "scannet15_identity_preregister": str(identity_preregister_path),
        "scannet15_identity_preregister_sha256":
            preregister_sha,
        "scannet15_pair_identity_payload_sha256":
            pair_identity["identity_payload_sha256"],
        "prepared_validation": prepared_validation,
        "source_array_sha256": {
            key: array_sha256(value) for key, value in raw.items()},
        "estimated_transform_discarded": True,
        "sentinel_artifact_path": sentinel_paths,
        "sentinel_artifact_sha256": sentinel_hashes,
        "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
        "sampling": "voxel10", "coarsest_cap": 512,
        "output_cache": str(output),
        "output_cache_sha256": sha256_file(output),
        "output_array_sha256": {
            key: array_sha256(value) for key, value in arrays.items()},
        "output_keys": list(OUTPUT_KEYS),
        "gt_consumed": False, "fallback_used": False,
        "converter_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepared-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--arm", choices=("sgf_selected_union", "fullscan"),
                        required=True)
    parser.add_argument("--direction", choices=("forward", "reverse"),
                        required=True)
    parser.add_argument("--identity-preregister", type=Path, required=True)
    args = parser.parse_args()
    receipt = convert(
        args.source, args.prepared_input, args.output, args.receipt,
        pair_id=args.pair_id, arm=args.arm, direction=args.direction,
        identity_preregister_path=args.identity_preregister)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
