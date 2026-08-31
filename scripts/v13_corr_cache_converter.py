#!/usr/bin/env python3
"""Convert one sealed ColorPCR sentinel cache to an exact-three solver cache.

The converter is intentionally one-way: ColorPCR's estimated transform is
hashed into the receipt and then discarded.  Downstream rigid solvers receive
only independently generated point correspondences and scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_KEYS = {
    "src_corr_points", "ref_corr_points", "corr_scores",
    "estimated_transform", "meta_json",
}
OUTPUT_KEYS = ("src_corr", "ref_corr", "scores")
FROZEN_NEIGHBOR_LIMITS = [38, 36, 36, 38]


class ConversionContractError(RuntimeError):
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


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def convert(source: Path, prepared_input: Path, output: Path, receipt_path: Path,
            *, pair_id: str, arm: str, direction: str) -> dict[str, Any]:
    source = Path(source).resolve()
    prepared_input = Path(prepared_input).resolve()
    output = Path(output).resolve()
    receipt_path = Path(receipt_path).resolve()
    if arm not in ("sgf_selected_union", "fullscan"):
        raise ConversionContractError("unknown arm")
    if direction not in ("forward", "reverse"):
        raise ConversionContractError("unknown direction")
    if pair_id.count("_to_") != 1:
        raise ConversionContractError("pair id must be exact src_to_ref")
    source_sha = sha256_file(source)
    prepared_sha = sha256_file(prepared_input)
    with np.load(prepared_input, allow_pickle=False) as prepared:
        if "manifest_json" not in prepared.files:
            raise ConversionContractError("prepared input manifest_json missing")
        prepared_manifest = json.loads(str(prepared["manifest_json"].item()))
    if prepared_manifest.get("schema") != "v13-color-preserving-pair-v2" \
            or prepared_manifest.get("pair_id") != pair_id:
        raise ConversionContractError("prepared pair identity mismatch")
    prepared_payload_sha = prepared_manifest.get("payload_sha256")
    if not isinstance(prepared_payload_sha, str) or len(prepared_payload_sha) != 64:
        raise ConversionContractError("prepared manifest payload SHA missing")
    with np.load(source, allow_pickle=False) as data:
        if set(data.files) != SOURCE_KEYS:
            raise ConversionContractError("sentinel cache schema is not exact")
        meta = json.loads(str(data["meta_json"].item()))
        raw = {key: np.asarray(data[key]) for key in SOURCE_KEYS - {"meta_json"}}
    if sha256_file(source) != source_sha:
        raise ConversionContractError("sentinel cache changed while reading")
    if meta.get("schema") != "v13-colorpcr-corr-cache-v2" \
            or meta.get("sentinel_invariant") is not True \
            or meta.get("gt_consumed") is not False \
            or meta.get("identity_fallback") is not False:
        raise ConversionContractError("ColorPCR sentinel gate not sealed")
    worker = meta.get("worker_contract", {})
    if worker.get("arm") != arm or worker.get("direction") != direction:
        raise ConversionContractError("arm/direction metadata mismatch")
    if meta.get("input_sha256") != prepared_sha:
        raise ConversionContractError("prepared input SHA mismatch")
    if worker.get("neighbor_limits") != FROZEN_NEIGHBOR_LIMITS \
            or worker.get("sampling") != "voxel10" \
            or worker.get("coarsest_cap") != 512:
        raise ConversionContractError("frozen ColorPCR resource contract mismatch")
    sentinel_paths = meta.get("sentinel_artifact_path", {})
    sentinel_hashes = meta.get("sentinel_artifact_sha256", {})
    if set(sentinel_paths) != {"identity", "proper_nonzero"} \
            or set(sentinel_hashes) != set(sentinel_paths):
        raise ConversionContractError("persistent two-sentinel evidence is missing")
    for name, raw_path in sentinel_paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file() or sha256_file(path) != sentinel_hashes[name]:
            raise ConversionContractError(f"sentinel evidence closure mismatch: {name}")
    src = np.asarray(raw["src_corr_points"])
    ref = np.asarray(raw["ref_corr_points"])
    scores = np.asarray(raw["corr_scores"])
    if src.ndim != 2 or src.shape[1:] != (3,) or ref.shape != src.shape \
            or scores.shape != (len(src),) or len(src) < 40:
        raise ConversionContractError("correspondence arrays are not aligned")
    if not all(np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all()
               for value in (src, ref, scores)):
        raise ConversionContractError("correspondences must be finite floating point")
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
        "schema": "v13-colorpcr-corr-conversion-receipt-v1",
        "pair_id": pair_id, "arm": arm, "direction": direction,
        "source_sentinel_cache": str(source),
        "source_sentinel_cache_sha256": source_sha,
        "prepared_input": str(prepared_input),
        "prepared_input_sha256": prepared_sha,
        "prepared_manifest_payload_sha256": prepared_payload_sha,
        "source_array_sha256": {key: array_sha256(value) for key, value in raw.items()},
        "estimated_transform_discarded": True,
        "sentinel_artifact_path": sentinel_paths,
        "sentinel_artifact_sha256": sentinel_hashes,
        "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
        "sampling": "voxel10", "coarsest_cap": 512,
        "output_cache": str(output), "output_cache_sha256": sha256_file(output),
        "output_array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
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
    parser.add_argument("--arm", choices=("sgf_selected_union", "fullscan"), required=True)
    parser.add_argument("--direction", choices=("forward", "reverse"), required=True)
    args = parser.parse_args()
    receipt = convert(args.source, args.prepared_input, args.output, args.receipt,
                      pair_id=args.pair_id, arm=args.arm, direction=args.direction)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
