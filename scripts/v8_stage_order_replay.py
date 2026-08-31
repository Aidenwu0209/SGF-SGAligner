"""GT-free offline V8 replay over a frozen V7 worker batch.

This command performs no registration, imports no GT loader, and never edits
its inputs.  It revalidates every manifest, receipt, aggregate and worker hash
before applying the frozen V8 stage order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_batch as v7_batch  # noqa: E402
import v7_registration_pilot as v7_pilot  # noqa: E402
from safety import decision_features  # noqa: E402
from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config,
    evaluate_stage_order,
)


SCHEMA = "v8-stage-order-consensus-replay-v1"
CONFIG = V8Config()
LEGACY_V7_BATCH_FILE_SHA256 = (
    "bee5784d5e857e929aed3c21ee28dddaf54c9e10ea0484065cb0c90551a39664")
LEGACY_V7_BATCH_EVIDENCE_SHA256 = (
    "9336845426c2d6bb026420d92b0ad5678fdaa868687e584a471c043a6020c5f9")
LEGACY_V7_SOURCE_SNAPSHOT_SHA256 = (
    "148672da4aca51b7e110c9ec6743f894e615f1301663efa1fb52c63c3e28734c")
LEGACY_V7_SOURCE_HEAD = "2afdb3101cbc86f7a17e5aec9cfbb32d54f82510"
LEGACY_V7_SOURCE_FILES = {
    name: path for name, path in v7_batch.SOURCE_FILES.items()
    if name != "pilot_gate"
}


class V8ReplayError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise V8ReplayError(f"invalid JSON {path}") from exc
    if not isinstance(data, dict):
        raise V8ReplayError(f"JSON object required {path}")
    return data


def _source_binding() -> dict[str, Any]:
    files = {
        "protocol": CODE_ROOT / "docs/V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md",
        "core": CODE_ROOT / "src/safety/v8_stage_order_consensus.py",
        "runner": Path(__file__).resolve(),
    }
    return {
        name: {"path": str(path.relative_to(CODE_ROOT)),
               "sha256": v7_pilot.sha256_file(path)}
        for name, path in files.items()
    }


def _verify_embedded_evidence(data: Mapping[str, Any], *, label: str) -> None:
    expected = data.get("evidence_sha256")
    unsigned = {key: value for key, value in data.items()
                if key != "evidence_sha256"}
    if not isinstance(expected, str) or v7_pilot.stable_json_hash(
            unsigned) != expected:
        raise V8ReplayError(f"{label} embedded evidence SHA mismatch")


def _validate_legacy_worker(path: Path, *, pair: Mapping[str, Any],
                            direction: str, replicate: int,
                            protocol_sha: str,
                            snapshot: Mapping[str, Any]) -> dict[str, Any]:
    row = v7_pilot.load_worker(
        path, pair_id=pair["pair_id"], direction=direction,
        replicate=replicate, cache_sha=pair["cache_sha256"],
        protocol_sha=protocol_sha)
    if (row.get("status") != "ok"
            or row.get("cache", {}).get("checkpoint_id")
            != v7_pilot.CHECKPOINT_ID
            or row.get("cache", {}).get("checkpoint_sha256")
            != v7_pilot.CHECKPOINT_SHA256):
        raise V8ReplayError(f"legacy worker checkpoint/status mismatch {path}")
    for field in ("raw_transform", "final_transform"):
        value = np.asarray(row.get(field), dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise V8ReplayError(f"legacy worker nonfinite transform {path}")
        if v7_pilot.array_sha256(value) != row.get(f"{field}_sha256"):
            raise V8ReplayError(f"legacy worker transform SHA mismatch {path}")
    count = row.get("correspondence_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise V8ReplayError(f"legacy worker correspondence count invalid {path}")
    permutation, provenance = v7_pilot.stable_row_permutation(
        count, pair_id=pair["pair_id"], direction=direction,
        replicate=replicate, protocol_sha=protocol_sha)
    permutation_sha = hashlib.sha256(
        np.ascontiguousarray(permutation.astype(np.int64)).tobytes()
    ).hexdigest()
    if (row.get("permutation_provenance_sha256") != provenance
            or row.get("permutation_sha256") != permutation_sha):
        raise V8ReplayError(f"legacy worker permutation SHA mismatch {path}")
    sources = snapshot.get("source_files", {})
    if (row.get("source_hashes", {}).get("runner")
            != sources.get("pilot_runner", {}).get("sha256")
            or row.get("source_hashes", {}).get("consensus")
            != sources.get("consensus", {}).get("sha256")):
        raise V8ReplayError(f"legacy worker source SHA mismatch {path}")
    return row


def _validate_legacy_source_batch(
    receipt_path: Path,
    manifest: Mapping[str, Any],
    receipt: dict[str, Any],
    *,
    enforce_known_identity: bool,
) -> dict[str, Any]:
    """Read-only validator for the one pre-hardening V7 development batch.

    The compatibility path never fills missing fields or writes a replacement
    receipt.  The CLI enables it only for the exact known file/evidence/source
    identities; tests may disable that outer identity pin while still exercising
    the full cryptographic chain with synthetic fixtures.
    """
    _verify_embedded_evidence(receipt, label="legacy batch receipt")
    snapshot = receipt.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise V8ReplayError("legacy batch source snapshot missing")
    snapshot_expected = snapshot.get("snapshot_sha256")
    snapshot_actual = v7_pilot.stable_json_hash({
        key: value for key, value in snapshot.items()
        if key != "snapshot_sha256"})
    if snapshot_expected != snapshot_actual:
        raise V8ReplayError("legacy source snapshot SHA mismatch")
    if enforce_known_identity and (
            v7_pilot.sha256_file(receipt_path) != LEGACY_V7_BATCH_FILE_SHA256
            or receipt.get("evidence_sha256")
            != LEGACY_V7_BATCH_EVIDENCE_SHA256
            or snapshot_expected != LEGACY_V7_SOURCE_SNAPSHOT_SHA256
            or snapshot.get("repository", {}).get("head")
            != LEGACY_V7_SOURCE_HEAD):
        raise V8ReplayError("not the exact known superseded V7 batch")
    if (receipt.get("schema") != v7_batch.BATCH_SCHEMA
            or receipt.get("status") != "GT_FREE_COMPLETE"
            or receipt.get("research_only") is not True
            or receipt.get("posthoc_not_run") is not True
            or receipt.get("pair_count") != 12
            or receipt.get("outer_repeats_per_pair") != 2
            or receipt.get("replicates_per_outer")
            != {"forward": 5, "reverse": 5}
            or receipt.get("manifest", {}).get("sha256")
            != manifest["_file_sha256"]
            or receipt.get("global_fail_closed_counts")
            != {"exceptions": 0, "nonfinite_transforms": 0,
                "cache_mismatches": 0}
            or "evidence_mode" in receipt
            or "formal_preregistered" in receipt):
        raise V8ReplayError("legacy batch shape/provenance mismatch")
    source_files = snapshot.get("source_files")
    if (not isinstance(source_files, dict)
            or set(source_files) != set(LEGACY_V7_SOURCE_FILES)
            or any(record.get("path") != LEGACY_V7_SOURCE_FILES[name]
                   or not v7_batch._is_sha256(record.get("sha256"))
                   for name, record in source_files.items())):
        raise V8ReplayError("legacy source-file snapshot malformed")
    pair_rows = receipt.get("pair_receipts")
    expected_pair_ids = [row["pair_id"] for row in manifest["pairs"]]
    if (not isinstance(pair_rows, list) or len(pair_rows) != 12
            or [row.get("pair_id") for row in pair_rows]
            != expected_pair_ids):
        raise V8ReplayError("legacy pair receipt coverage/order mismatch")
    pairs_by_id = {row["pair_id"]: row for row in manifest["pairs"]}
    for binding in pair_rows:
        pair = pairs_by_id[binding["pair_id"]]
        pair_path = Path(binding["path"]).resolve()
        if v7_pilot.sha256_file(pair_path) != binding.get("sha256"):
            raise V8ReplayError("legacy pair receipt file SHA mismatch")
        pair_receipt = _load_json(pair_path)
        _verify_embedded_evidence(pair_receipt, label="legacy pair receipt")
        if (pair_receipt.get("schema") != v7_batch.PAIR_RECEIPT_SCHEMA
                or pair_receipt.get("status") != "GT_FREE_COMPLETE"
                or pair_receipt.get("pair_id") != pair["pair_id"]
                or pair_receipt.get("outer_repeats") != 2
                or pair_receipt.get("posthoc_not_run") is not True
                or pair_receipt.get("batch") != {
                    "manifest_sha256": manifest["_file_sha256"],
                    "source_snapshot_sha256": snapshot_expected,
                    "role": pair["role"],
                }
                or len(pair_receipt.get("aggregates", [])) != 2):
            raise V8ReplayError("legacy pair receipt provenance mismatch")
        cache_checked = False
        for outer, aggregate_binding in enumerate(
                pair_receipt["aggregates"]):
            aggregate_path = Path(aggregate_binding["path"]).resolve()
            if v7_pilot.sha256_file(aggregate_path) != aggregate_binding.get(
                    "sha256"):
                raise V8ReplayError("legacy aggregate file SHA mismatch")
            aggregate = _load_json(aggregate_path)
            _verify_embedded_evidence(aggregate, label="legacy aggregate")
            protocol_sha = aggregate.get("protocol", {}).get("sha256")
            workers = aggregate.get("workers", {})
            if (aggregate.get("schema") != v7_pilot.SCHEMA
                    or aggregate.get("status") != "GT_FREE_COMPLETE"
                    or aggregate.get("research_only") is not True
                    or aggregate.get("pair_id") != pair["pair_id"]
                    or aggregate.get("outer_repeat") != outer
                    or aggregate.get("cache", {}).get("sha256")
                    != pair["cache_sha256"]
                    or aggregate.get("cache", {}).get("checkpoint_id")
                    != v7_pilot.CHECKPOINT_ID
                    or aggregate.get("cache", {}).get("checkpoint_sha256")
                    != v7_pilot.CHECKPOINT_SHA256
                    or protocol_sha != manifest["protocol_sha256"]
                    or aggregate.get("batch") != {
                        "manifest_sha256": manifest["_file_sha256"],
                        "source_snapshot_sha256": snapshot_expected,
                        "pair_role": pair["role"],
                    }
                    or {key: workers.get(key) for key in (
                        "requested", "completed", "exceptions",
                        "nonfinite_transforms", "cache_mismatches")}
                    != {"requested": 10, "completed": 10, "exceptions": 0,
                        "nonfinite_transforms": 0, "cache_mismatches": 0}):
                raise V8ReplayError("legacy aggregate provenance/gate mismatch")
            expected_names = {
                *(f"{direction}_{replicate:02d}.json"
                  for direction in ("forward", "reverse")
                  for replicate in range(5)),
                "gt_free_aggregate.json",
            }
            if {path.name for path in aggregate_path.parent.iterdir()} \
                    != expected_names:
                raise V8ReplayError("legacy aggregate directory shape mismatch")
            worker_hashes = []
            for direction in ("forward", "reverse"):
                for replicate in range(5):
                    worker_path = aggregate_path.parent / (
                        f"{direction}_{replicate:02d}.json")
                    row = _validate_legacy_worker(
                        worker_path, pair=pair, direction=direction,
                        replicate=replicate, protocol_sha=protocol_sha,
                        snapshot=snapshot)
                    worker_hashes.append(row["evidence_sha256"])
            if sorted(worker_hashes) != aggregate.get(
                    "worker_evidence_sha256"):
                raise V8ReplayError("legacy aggregate worker hash-set mismatch")
            if not cache_checked:
                cache_path = Path(aggregate["cache"]["path"]).resolve()
                if v7_pilot.sha256_file(cache_path) != pair["cache_sha256"]:
                    raise V8ReplayError("legacy immutable cache SHA mismatch")
                cache_checked = True
    return receipt


def validate_source_batch(receipt_path: Path, manifest_path: Path,
                          manifest_sha256: str) -> tuple[
                              dict[str, Any], dict[str, Any], str]:
    """Revalidate the full V7 chain using its frozen source snapshot."""
    manifest = v7_batch.validate_manifest(
        manifest_path, manifest_sha256, allow_non_preregistered=True)
    raw = _load_json(receipt_path)
    snapshot = raw.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise V8ReplayError("source receipt has no source snapshot")
    expected = snapshot.get("snapshot_sha256")
    actual = v7_pilot.stable_json_hash({
        key: value for key, value in snapshot.items()
        if key != "snapshot_sha256"})
    if expected != actual:
        raise V8ReplayError("V7 source snapshot SHA mismatch")
    try:
        receipt = v7_batch.validate_batch_receipt(
            receipt_path, manifest=manifest, snapshot=snapshot)
        mode = "CURRENT_V7_VERIFIER"
    except v7_batch.BatchEvidenceError:
        receipt = _validate_legacy_source_batch(
            receipt_path, manifest, raw, enforce_known_identity=True)
        mode = "KNOWN_SUPERSEDED_V7_READ_ONLY"
    return manifest, receipt, mode


def _workers_for_aggregate(aggregate_path: Path,
                           pair: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate = _load_json(aggregate_path)
    protocol_sha = aggregate.get("protocol", {}).get("sha256")
    workers = []
    bindings = []
    for direction in ("forward", "reverse"):
        for replicate in range(5):
            path = aggregate_path.parent / f"{direction}_{replicate:02d}.json"
            row = v7_batch._load_batch_worker(
                path, pair_id=pair["pair_id"], direction=direction,
                replicate=replicate, cache_sha=pair["cache_sha256"],
                protocol_sha=protocol_sha)
            workers.append(row)
            bindings.append({
                "direction": direction,
                "replicate": replicate,
                "path": str(path),
                "file_sha256": v7_pilot.sha256_file(path),
                "evidence_sha256": row["evidence_sha256"],
                "raw_transform_sha256": row["raw_transform_sha256"],
                "final_transform_sha256": row["final_transform_sha256"],
                "permutation_provenance_sha256": row[
                    "permutation_provenance_sha256"],
            })
    if sorted(row["evidence_sha256"] for row in workers) != aggregate.get(
            "worker_evidence_sha256"):
        raise V8ReplayError("aggregate worker evidence binding mismatch")
    return workers, bindings


def replay(receipt_path: Path, manifest_path: Path,
           manifest_sha256: str) -> dict[str, Any]:
    manifest, batch_receipt, source_validation_mode = validate_source_batch(
        receipt_path, manifest_path, manifest_sha256)
    pairs_by_id = {row["pair_id"]: row for row in manifest["pairs"]}
    pair_rows = []
    worker_bindings = []
    for pair_receipt_row in batch_receipt["pair_receipts"]:
        pair_id = pair_receipt_row["pair_id"]
        pair = pairs_by_id[pair_id]
        pair_receipt = _load_json(Path(pair_receipt_row["path"]).resolve())
        outers = []
        for outer, aggregate_row in enumerate(pair_receipt["aggregates"]):
            aggregate_path = Path(aggregate_row["path"]).resolve()
            workers, bindings = _workers_for_aggregate(aggregate_path, pair)
            result = evaluate_stage_order(
                workers, CONFIG, decision_features.evaluate_rule_b,
                # Frozen V7 traces predate the fixed-correspondence contract.
                # The replay is development evidence, never a fresh V8 gate.
                require_fixed_trace=False)
            outers.append({
                "outer_repeat": outer,
                "source_aggregate": {
                    "path": str(aggregate_path),
                    "file_sha256": v7_pilot.sha256_file(aggregate_path),
                    "evidence_sha256": _load_json(aggregate_path)[
                        "evidence_sha256"],
                },
                "result": result,
            })
            worker_bindings.extend({"pair_id": pair_id,
                                    "outer_repeat": outer, **row}
                                   for row in bindings)
        outcomes = [row["result"]["usable_for_reconstruction"]
                    for row in outers]
        pair_rows.append({
            "pair_id": pair_id,
            "role": pair["role"],
            "cache_sha256": pair["cache_sha256"],
            "outer_outcomes": outcomes,
            "repeatable": len(set(outcomes)) == 1,
            "outers": outers,
        })
    output = {
        "schema": SCHEMA,
        "status": "GT_FREE_DEVELOPMENT_REPLAY_COMPLETE",
        "research_only": True,
        "development_split_exposed": True,
        "qualifies_as_blind_gate": False,
        "source_batch_validation": {
            "mode": source_validation_mode,
            "development_only": source_validation_mode
            == "KNOWN_SUPERSEDED_V7_READ_ONLY",
            "upgraded_or_rewritten": False,
        },
        "fixed_trace_contract": (
            "each pair/outer result sets fresh_v8_qualified only when both "
            "observed medoids carry and pass fixed-correspondence ICP trace"),
        "source_batch_receipt": {
            "path": str(receipt_path.resolve()),
            "file_sha256": v7_pilot.sha256_file(receipt_path),
            "evidence_sha256": batch_receipt["evidence_sha256"],
            "source_snapshot_sha256": batch_receipt["source_snapshot"][
                "snapshot_sha256"],
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest["_file_sha256"],
            "pair_ids_sha256": manifest["pair_ids_sha256"],
        },
        "v8_source_bindings": _source_binding(),
        "config": {
            "repeats": CONFIG.repeats, "quorum": CONFIG.quorum,
            "max_rotation_deg": CONFIG.max_rotation_deg,
            "max_translation_m": CONFIG.max_translation_m,
        },
        "pair_count": len(pair_rows),
        "usable_pairs_per_outer": [
            sum(row["outer_outcomes"][outer] for row in pair_rows)
            for outer in range(2)],
        "all_pair_outcomes_repeatable": all(
            row["repeatable"] for row in pair_rows),
        "fresh_v8_qualified_pairs_per_outer": [
            sum(row["outers"][outer]["result"]["fresh_v8_qualified"]
                for row in pair_rows) for outer in range(2)],
        "pairs": pair_rows,
        "worker_bindings": worker_bindings,
        "worker_binding_count": len(worker_bindings),
        "posthoc_not_run": True,
    }
    output["worker_binding_sha256"] = v7_pilot.stable_json_hash(
        worker_bindings)
    output["evidence_sha256"] = v7_pilot.stable_json_hash(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = replay(
        args.batch_receipt.resolve(), args.manifest.resolve(),
        args.manifest_sha256)
    v7_pilot.atomic_create_json(args.out.resolve(), output)
    print(json.dumps({
        "status": output["status"],
        "usable_pairs_per_outer": output["usable_pairs_per_outer"],
        "all_pair_outcomes_repeatable": output[
            "all_pair_outcomes_repeatable"],
        "evidence_sha256": output["evidence_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
