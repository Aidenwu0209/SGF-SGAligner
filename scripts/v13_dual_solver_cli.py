#!/usr/bin/env python3
"""Sole V13 strict one-pair CLI after two independent ColorPCR caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from safety.v13_dual_solver_runtime import atomic_json, run_matrix, sha256_file
from safety.v13_strict_pair_gate import strict_pair_gate
from v7_registration_pilot import rule_b_features, segment_icp_with_trace
from v13_formal_source_manifest import formal_source_sha256


CONVERSION_SCHEMA = "v13-colorpcr-corr-conversion-receipt-v1"
SAFETY_AUTHORITY_SHA256 = {
    "src/safety/decision_features.py":
        "3795af00ba7c494f10bc949e35c104a2f67e37876e309fa923945ad248a393eb",
    "scripts/v7_registration_pilot.py":
        "aeecad81374a65cb4a128e646c1d5c85ec0c8df591d454df76ade1a2b8ad3a5f",
    "src/safety/v8_stage_order_consensus.py":
        "38617c61b0ce2faebceb31673b01db0158a28251f034f91b570d9d9909784d1c",
}


class StrictCliContractError(RuntimeError):
    pass


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise StrictCliContractError(f"JSON object required: {path}")
    return value


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode()).hexdigest()


def _verify_preflight_closure(*, preregister_path: Path,
                              preflight_path: Path,
                              prepared_path: Path,
                              pair_id: str,
                              preregister: dict,
                              preflight: dict) -> dict:
    """Bind the formal input to one exact frozen preflight pair entry.

    A matching manifest embedded in an arbitrary NPZ is deliberately
    insufficient: both the resolved path and file SHA must be the values
    frozen in the preflight.  The surrounding preflight/artifact/preregister
    hashes are re-derived so a copied or edited evidence set fails closed.
    """
    preregister_path = Path(preregister_path).resolve()
    preflight_path = Path(preflight_path).resolve()
    prepared_path = Path(prepared_path).resolve()
    if preregister != _load_json(preregister_path):
        raise StrictCliContractError("preregister file/content mismatch")
    if preflight != _load_json(preflight_path):
        raise StrictCliContractError("preflight file/content mismatch")

    payload_sha = preflight.get("payload_sha256")
    unsigned_preflight = dict(preflight)
    unsigned_preflight.pop("payload_sha256", None)
    if not isinstance(payload_sha, str) \
            or payload_sha != _stable_json_sha256(unsigned_preflight):
        raise StrictCliContractError("preflight payload SHA mismatch")

    preregister_sha = sha256_file(preregister_path)
    if preflight.get("preregister_sha256") != preregister_sha:
        raise StrictCliContractError("preflight preregister SHA mismatch")

    pairs = preflight.get("pairs")
    if not isinstance(pairs, list):
        raise StrictCliContractError("preflight pairs[] missing")
    pair_rows = [row.get("pair_id") for row in pairs
                 if isinstance(row, dict)]
    if len(pair_rows) != len(pairs) or pair_rows != preflight.get("pair_ids"):
        raise StrictCliContractError(
            "preflight pairs[] identity/order differs from pair_ids")
    matches = [row for row in pairs if isinstance(row, dict)
               and row.get("pair_id") == pair_id]
    if len(matches) != 1:
        raise StrictCliContractError("preflight pair entry is not unique")
    row = matches[0]
    expected_path = Path(str(row.get("prepared_npz_path", ""))).resolve()
    expected_sha = row.get("prepared_npz_sha256")
    if expected_path != prepared_path:
        raise StrictCliContractError("prepared input path differs from frozen preflight")
    actual_prepared_sha = sha256_file(prepared_path)
    if not isinstance(expected_sha, str) or expected_sha != actual_prepared_sha:
        raise StrictCliContractError("prepared input SHA differs from frozen preflight")

    # build_color_pair hashes the semantic pair manifest before appending the
    # filesystem path and NPZ SHA to the enclosing preflight row.
    pair_payload_sha = row.get("payload_sha256")
    unsigned_pair = dict(row)
    for key in ("payload_sha256", "prepared_npz_path", "prepared_npz_sha256"):
        unsigned_pair.pop(key, None)
    if not isinstance(pair_payload_sha, str) \
            or pair_payload_sha != _stable_json_sha256(unsigned_pair):
        raise StrictCliContractError("preflight pair payload SHA mismatch")
    with np.load(prepared_path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise StrictCliContractError("prepared input manifest_json missing")
        embedded = json.loads(str(data["manifest_json"].item()))
    expected_embedded = dict(unsigned_pair)
    expected_embedded["payload_sha256"] = pair_payload_sha
    if embedded != expected_embedded:
        raise StrictCliContractError("prepared embedded manifest differs from preflight")

    artifact_path = preflight_path.parent / "artifact_manifest.json"
    artifact = _load_json(artifact_path)
    if artifact.get("payload_sha256") != payload_sha \
            or artifact.get("preregister_sha256") != preregister_sha:
        raise StrictCliContractError("preflight artifact closure mismatch")
    files = artifact.get("files")
    if not isinstance(files, list):
        raise StrictCliContractError("preflight artifact files[] missing")
    by_path = {str(item.get("path")): item for item in files
               if isinstance(item, dict)}
    try:
        preflight_relative = str(preflight_path.relative_to(preflight_path.parent))
        prepared_relative = str(prepared_path.relative_to(preflight_path.parent))
    except ValueError as exc:
        raise StrictCliContractError(
            "prepared input is outside frozen preflight root") from exc
    for relative, path in ((preflight_relative, preflight_path),
                           (prepared_relative, prepared_path)):
        item = by_path.get(relative)
        if item is None or item.get("sha256") != sha256_file(path) \
                or item.get("bytes") != path.stat().st_size:
            raise StrictCliContractError(
                f"preflight artifact file closure mismatch: {relative}")

    return {
        "preflight_manifest": str(preflight_path),
        "preflight_manifest_sha256": sha256_file(preflight_path),
        "preflight_payload_sha256": payload_sha,
        "artifact_manifest": str(artifact_path.resolve()),
        "artifact_manifest_sha256": sha256_file(artifact_path),
        "preregister": str(preregister_path),
        "preregister_sha256": preregister_sha,
        "prepared_input": str(prepared_path),
        "prepared_input_sha256": actual_prepared_sha,
        "prepared_pair_payload_sha256": pair_payload_sha,
    }


def _verify_safety_authority(repo: Path) -> dict[str, str]:
    observed = {relative: sha256_file(repo / relative)
                for relative in SAFETY_AUTHORITY_SHA256}
    if observed != SAFETY_AUTHORITY_SHA256:
        raise StrictCliContractError("imported safety authority SHA mismatch")
    return observed


def _verify_fixed_identity(preregister: dict, preflight: dict,
                           pair_id: str) -> None:
    normals = [str(value) for value in preregister.get("normal_pair_ids", ())]
    known_bad = str(preregister.get("known_bad_pair_id", ""))
    frozen = normals + [known_bad]
    if len(normals) != 3 or len(set(frozen)) != 4 or not known_bad:
        raise StrictCliContractError("pre-registration is not exact fixed4")
    if preflight.get("pair_ids") != frozen or pair_id not in frozen:
        raise StrictCliContractError("pair identity/order differs from frozen preflight")


def _verify_conversion_receipt(*, receipt_path: Path, cache_path: Path,
                               prepared_path: Path, pair_id: str,
                               arm: str, direction: str,
                               converter_path: Path) -> dict:
    receipt = _load_json(receipt_path)
    expected = {
        "schema": CONVERSION_SCHEMA,
        "pair_id": pair_id,
        "arm": arm,
        "direction": direction,
        "prepared_input_sha256": sha256_file(prepared_path),
        "output_cache_sha256": sha256_file(cache_path),
        "converter_sha256": sha256_file(converter_path),
        "estimated_transform_discarded": True,
        "gt_consumed": False,
        "fallback_used": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise StrictCliContractError(
                f"conversion receipt mismatch for {direction}: {key}")
    if Path(receipt.get("output_cache", "")).resolve() != cache_path.resolve():
        raise StrictCliContractError("conversion receipt output path mismatch")
    source = Path(receipt.get("source_sentinel_cache", ""))
    if not source.is_file() or sha256_file(source) != receipt.get(
            "source_sentinel_cache_sha256"):
        raise StrictCliContractError("source sentinel cache is not rehashable")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-cache", required=True, type=Path)
    parser.add_argument("--reverse-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pointdsc-root", required=True, type=Path)
    parser.add_argument("--pointdsc-checkpoint", required=True, type=Path)
    parser.add_argument("--prepared-input", required=True, type=Path)
    parser.add_argument("--arm", required=True,
                        choices=("sgf_selected_union", "fullscan"))
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--preregister", required=True, type=Path)
    parser.add_argument("--preflight-manifest", required=True, type=Path)
    parser.add_argument("--forward-receipt", required=True, type=Path)
    parser.add_argument("--reverse-receipt", required=True, type=Path)
    parser.add_argument("--driver-source", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preregister = _load_json(args.preregister)
    preflight = _load_json(args.preflight_manifest)
    _verify_fixed_identity(preregister, preflight, args.pair_id)
    preflight_closure = _verify_preflight_closure(
        preregister_path=args.preregister,
        preflight_path=args.preflight_manifest,
        prepared_path=args.prepared_input,
        pair_id=args.pair_id,
        preregister=preregister,
        preflight=preflight,
    )
    safety_sha = _verify_safety_authority(repo)
    converter = repo / "scripts/v13_corr_cache_converter.py"
    driver_source = args.driver_source.resolve()
    if driver_source != (repo / "scripts/v13_fixed4_driver.py").resolve():
        raise StrictCliContractError("formal driver source path mismatch")
    receipts = {
        "forward": _verify_conversion_receipt(
            receipt_path=args.forward_receipt,
            cache_path=args.forward_cache.resolve(),
            prepared_path=args.prepared_input.resolve(), pair_id=args.pair_id,
            arm=args.arm, direction="forward", converter_path=converter),
        "reverse": _verify_conversion_receipt(
            receipt_path=args.reverse_receipt,
            cache_path=args.reverse_cache.resolve(),
            prepared_path=args.prepared_input.resolve(), pair_id=args.pair_id,
            arm=args.arm, direction="reverse", converter_path=converter),
    }
    raw = run_matrix(args.forward_cache, args.reverse_cache,
                     output / "raw", args.pointdsc_root,
                     args.pointdsc_checkpoint, device="cpu", known_bad=False)
    atomic_json(output / "raw_summary.json", raw)
    strict = strict_pair_gate(
        pair_id=args.pair_id, arm=args.arm,
        prepared_path=args.prepared_input,
        forward_cache_path=args.forward_cache,
        reverse_cache_path=args.reverse_cache,
        dual_summary=raw, preregistration=preregister,
        icp_fn=segment_icp_with_trace,
        rule_features_fn=rule_b_features,
    )
    strict.update({
        "safety_authority_sha256": safety_sha,
        "rule_c_claimed": False,
        "node_pair_evidence": "unavailable_not_applicable_to_rule_b",
        "successful_node_pairs": 0,
        "failed_node_pairs": 0,
        "conversion_receipt_sha256": {
            "forward": sha256_file(args.forward_receipt),
            "reverse": sha256_file(args.reverse_receipt),
        },
        "source_sentinel_cache_sha256": {
            direction: receipt["source_sentinel_cache_sha256"]
            for direction, receipt in receipts.items()
        },
        "preflight_closure": preflight_closure,
        "formal_source_sha256": formal_source_sha256(repo),
    })
    atomic_json(output / "summary.json", strict)
    print(json.dumps(strict, sort_keys=True, indent=2))
    return 0 if strict["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
