"""GT-free V8 final-consensus replay for the frozen selection89 workers.

The expensive workers are generated once by ``v8_selection89_development``.
This command revalidates every file/evidence/transform hash and applies the
single frozen V8 stage-order policy offline.  It has no GT imports; the existing
``v8_stage_order_posthoc.py`` is the only label process and may run only after
this receipt has been atomically frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from safety import decision_features  # noqa: E402
from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config,
    evaluate_stage_order,
)
import v7_registration_pilot as pilot  # noqa: E402
import v8_selection89_development as dev  # noqa: E402
import v8_stage_order_replay as v8_replay  # noqa: E402


SCHEMA = v8_replay.SCHEMA
CONFIG = V8Config()


class Selection89ReplayError(RuntimeError):
    """The frozen selection89 worker chain failed revalidation."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Selection89ReplayError(f"invalid JSON {path}") from exc
    if not isinstance(value, dict):
        raise Selection89ReplayError(f"JSON object required {path}")
    return value


def _validate_evidence(path: Path, schema: str) -> dict[str, Any]:
    value = _load_json(path)
    expected = value.pop("evidence_sha256", None)
    actual = dev.stable_json_hash(value)
    value["evidence_sha256"] = expected
    if expected != actual or value.get("schema") != schema:
        raise Selection89ReplayError(f"evidence/schema mismatch {path}")
    return value


def _validate_batch(
        batch_path: Path, manifest_path: Path,
        manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = dev.validate_manifest(
        manifest_path, manifest_sha256, verify_caches=True)
    batch = _validate_evidence(batch_path, dev.BATCH_SCHEMA)
    snapshot = batch.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise Selection89ReplayError("batch source snapshot missing")
    expected_snapshot = snapshot.get("snapshot_sha256")
    actual_snapshot = dev.stable_json_hash({
        key: value for key, value in snapshot.items()
        if key != "snapshot_sha256"})
    if expected_snapshot != actual_snapshot:
        raise Selection89ReplayError("batch source snapshot SHA mismatch")
    if (batch.get("status") != "GT_FREE_WORKERS_COMPLETE"
            or batch.get("evidence_class") != dev.EVIDENCE_CLASS
            or batch.get("posthoc_not_run") is not True
            or batch.get("policy_not_applied") is not True
            or batch.get("pair_count") != dev.PAIR_COUNT
            or batch.get("outer_repeats_per_pair") != dev.OUTER_REPEATS
            or batch.get("workers_per_outer") != dev.WORKERS_PER_OUTER
            or batch.get("total_workers") != dev.TOTAL_WORKERS
            or batch.get("manifest", {}).get("sha256")
            != manifest["_file_sha256"]):
        raise Selection89ReplayError("batch identity/shape contract mismatch")
    expected_ids = [row["pair_id"] for row in manifest["pairs"]]
    rows = batch.get("pair_receipts")
    if (not isinstance(rows, list) or len(rows) != dev.PAIR_COUNT
            or [row.get("pair_id") for row in rows] != expected_ids):
        raise Selection89ReplayError("batch pair order/coverage mismatch")
    pair_by_id = {row["pair_id"]: row for row in manifest["pairs"]}
    loaded: list[dict[str, Any]] = []
    for batch_row in rows:
        pair = pair_by_id[batch_row["pair_id"]]
        pair_path = Path(batch_row["path"]).resolve()
        if (not dev.SHA256_RE.fullmatch(str(batch_row.get("sha256", "")))
                or dev.sha256_file(pair_path) != batch_row["sha256"]):
            raise Selection89ReplayError(f"pair receipt SHA mismatch {pair_path}")
        pair_receipt = _validate_evidence(pair_path, dev.PAIR_SCHEMA)
        if (pair_receipt.get("status") != "GT_FREE_WORKERS_COMPLETE"
                or pair_receipt.get("pair_id") != pair["pair_id"]
                or pair_receipt.get("outer_repeats") != dev.OUTER_REPEATS
                or pair_receipt.get("posthoc_not_run") is not True):
            raise Selection89ReplayError(
                f"pair receipt identity mismatch {pair_path}")
        outers = pair_receipt.get("outers")
        if (not isinstance(outers, list)
                or [row.get("outer_repeat") for row in outers] != [0, 1]):
            raise Selection89ReplayError(f"pair outer shape mismatch {pair_path}")
        loaded_outers = []
        for outer, outer_row in enumerate(outers):
            outer_path = Path(outer_row["path"]).resolve()
            if (not dev.SHA256_RE.fullmatch(str(outer_row.get("sha256", "")))
                    or dev.sha256_file(outer_path) != outer_row["sha256"]):
                raise Selection89ReplayError(
                    f"outer receipt SHA mismatch {outer_path}")
            outer_receipt = dev._validate_outer(
                outer_path, pair=pair, outer=outer,
                manifest_sha=manifest["_file_sha256"],
                snapshot_sha=expected_snapshot)
            workers = []
            bindings = []
            for worker_row in outer_receipt["workers"]:
                worker_path = Path(worker_row["path"]).resolve()
                worker = dev._validate_worker(
                    worker_path, pair=pair,
                    direction=worker_row["direction"],
                    replicate=worker_row["replicate"])
                workers.append(worker)
                bindings.append({
                    "direction": worker_row["direction"],
                    "replicate": worker_row["replicate"],
                    "path": str(worker_path),
                    "file_sha256": worker_row["sha256"],
                    "evidence_sha256": worker["evidence_sha256"],
                    "raw_transform_sha256": worker["raw_transform_sha256"],
                    "final_transform_sha256": worker[
                        "final_transform_sha256"],
                    "permutation_provenance_sha256": worker[
                        "permutation_provenance_sha256"],
                })
            loaded_outers.append({
                "outer_repeat": outer,
                "receipt_path": str(outer_path),
                "receipt_sha256": outer_row["sha256"],
                "receipt_evidence_sha256": outer_receipt["evidence_sha256"],
                "workers": workers,
                "bindings": bindings,
            })
        loaded.append({"pair": pair, "outers": loaded_outers})
    return manifest, batch, loaded


def _source_binding() -> dict[str, Any]:
    files = {
        "protocol": CODE_ROOT / "docs/V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md",
        "core": CODE_ROOT / "src/safety/v8_stage_order_consensus.py",
        "runner": Path(__file__).resolve(),
        "worker_controller": CODE_ROOT / "scripts/v8_selection89_development.py",
    }
    return {
        name: {"path": str(path.relative_to(CODE_ROOT)),
               "sha256": dev.sha256_file(path)}
        for name, path in files.items()
    }


def replay(batch_path: Path, manifest_path: Path,
           manifest_sha256: str) -> dict[str, Any]:
    manifest, batch, loaded = _validate_batch(
        batch_path, manifest_path, manifest_sha256)
    pair_rows = []
    worker_bindings = []
    for loaded_pair in loaded:
        pair = loaded_pair["pair"]
        outers = []
        for outer_row in loaded_pair["outers"]:
            result = evaluate_stage_order(
                outer_row["workers"], CONFIG,
                decision_features.evaluate_rule_b,
                require_fixed_trace=True)
            outers.append({
                "outer_repeat": outer_row["outer_repeat"],
                "source_aggregate": {
                    "path": outer_row["receipt_path"],
                    "file_sha256": outer_row["receipt_sha256"],
                    "evidence_sha256": outer_row[
                        "receipt_evidence_sha256"],
                },
                "result": result,
            })
            worker_bindings.extend({
                "pair_id": pair["pair_id"],
                "outer_repeat": outer_row["outer_repeat"], **binding,
            } for binding in outer_row["bindings"])
        outcomes = [row["result"]["usable_for_reconstruction"]
                    for row in outers]
        pair_rows.append({
            "pair_id": pair["pair_id"],
            "role": "selection89_development",
            "cache_sha256": pair["cache_sha256"],
            "outer_outcomes": outcomes,
            "repeatable": outcomes[0] == outcomes[1],
            "outers": outers,
        })
    output = {
        "schema": SCHEMA,
        "status": "GT_FREE_DEVELOPMENT_REPLAY_COMPLETE",
        "research_only": True,
        "development_split_exposed": True,
        "qualifies_as_blind_gate": False,
        "fresh_v8_worker_cache": True,
        "fixed_trace_contract": (
            "both observed medoids must carry and pass the corrected "
            "fixed-correspondence ICP trace"),
        "source_batch_receipt": {
            "path": str(batch_path.resolve()),
            "file_sha256": dev.sha256_file(batch_path),
            "evidence_sha256": batch["evidence_sha256"],
            "source_snapshot_sha256": batch["source_snapshot"][
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
            for outer in range(dev.OUTER_REPEATS)],
        "all_pair_outcomes_repeatable": all(
            row["repeatable"] for row in pair_rows),
        "fresh_v8_qualified_pairs_per_outer": [
            sum(row["outers"][outer]["result"]["fresh_v8_qualified"]
                for row in pair_rows) for outer in range(dev.OUTER_REPEATS)],
        "pairs": pair_rows,
        "worker_bindings": worker_bindings,
        "worker_binding_count": len(worker_bindings),
        "posthoc_not_run": True,
    }
    # Match the shared V8 posthoc receipt contract exactly.  The worker-cache
    # controller has its own newline-terminated manifest hash convention, but
    # V8 replay/posthoc receipts use the pre-existing pilot stable hash.
    output["worker_binding_sha256"] = pilot.stable_json_hash(worker_bindings)
    output["evidence_sha256"] = pilot.stable_json_hash(output)
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
    pilot.atomic_create_json(args.out.resolve(), output)
    print(json.dumps({
        "status": output["status"],
        "usable_pairs_per_outer": output["usable_pairs_per_outer"],
        "fresh_v8_qualified_pairs_per_outer": output[
            "fresh_v8_qualified_pairs_per_outer"],
        "all_pair_outcomes_repeatable": output[
            "all_pair_outcomes_repeatable"],
        "evidence_sha256": output["evidence_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
