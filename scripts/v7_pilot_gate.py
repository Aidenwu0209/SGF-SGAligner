"""Formal V7 12-pair pilot gate over frozen GT-free and posthoc evidence.

The evaluator never selects a policy.  It reports all eight frozen policies
and authorises the later selection89 stage only when every policy satisfies
the pre-registered pilot gate.  A non-default manifest is always classified
as non-preregistered research and can never produce PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_batch as batch  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402


SCHEMA = "v7-registration-veto-formal-pilot-gate-v1"
V6_CONTROL_ROOT = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official-v6fix-audit/outputs/"
    "official_sgaligner_v6_fix_consistency_audit_20260829/formal_v2/"
    "selection/B")
V6_CONTROL_FILES = (
    ("repeat_00.json",
     "97dab7f5158113d92a2810a2b0cfd9fa4599662ac7d7fb9126bfa3a1b004d589"),
    ("repeat_01.json",
     "d445e810f6c0785abc105e920fba52277631a646077cfc2b6b153792bb3129e3"),
    ("repeat_02.json",
     "1dc077a81c45eb8130cfbde90a92fb64a8e6e8c512e38bd5ebf8df6d24fe88ae"),
)
KNOWN_PAIR = pilot.NEAR_MISS_PAIR
MAX_LOSS = 1
EXPECTED_WORKERS = 12 * 2 * 10


class PilotGateError(RuntimeError):
    """Input evidence is malformed or not cryptographically bound."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotGateError(f"invalid JSON {path}") from exc
    if not isinstance(data, dict):
        raise PilotGateError(f"JSON object required {path}")
    return data


def load_v6_controls(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load exactly the three pre-bound B/F controls and derive baseline.

    ``majority_existing`` deliberately means BOTH raw-strict in at least two
    of three repeats and accepted-correct in at least two of three repeats.
    This is more conservative than taking the union of the two properties.
    """
    pair_ids = [row["pair_id"] for row in manifest["pairs"]]
    repeats: list[dict[str, Mapping[str, Any]]] = []
    files = []
    for repeat, (name, expected_sha) in enumerate(V6_CONTROL_FILES):
        path = V6_CONTROL_ROOT / name
        actual_sha = pilot.sha256_file(path)
        if actual_sha != expected_sha:
            raise PilotGateError(f"V6 control hash mismatch {path}")
        data = _load_json(path)
        if (data.get("schema") != "v6fix-consistency-audit-v2"
                or data.get("split") != "selection"
                or data.get("checkpoint") != "B"
                or data.get("checkpoint_sha256")
                != pilot.CHECKPOINT_SHA256
                or data.get("repeat") != repeat):
            raise PilotGateError(f"V6 control provenance mismatch {path}")
        rows = {}
        for row in data.get("rows", []):
            pair_id = row.get("pair_id")
            flat = row.get("paths", {}).get("F")
            if isinstance(pair_id, str) and isinstance(flat, dict):
                rows[pair_id] = flat
        if any(pair_id not in rows for pair_id in pair_ids):
            raise PilotGateError(f"V6 control lacks frozen pilot pair {path}")
        repeats.append(rows)
        files.append({"repeat": repeat, "path": str(path),
                      "sha256": expected_sha})

    histories = {}
    majority_existing = []
    repeat_counts = []
    for repeat, rows in enumerate(repeats):
        subset = [rows[pair_id] for pair_id in pair_ids]
        repeat_counts.append({
            "repeat": repeat,
            "raw_strict": sum(bool(row.get("strict")) for row in subset),
            "accepted_correct": sum(
                bool(row.get("accepted_correct")) for row in subset),
            "accepted_error": sum(
                bool(row.get("accepted_error")) for row in subset),
        })
    for pair_id in pair_ids:
        strict = [bool(rows[pair_id].get("strict")) for rows in repeats]
        correct = [bool(rows[pair_id].get("accepted_correct"))
                   for rows in repeats]
        error = [bool(rows[pair_id].get("accepted_error"))
                 for rows in repeats]
        is_existing = sum(strict) >= 2 and sum(correct) >= 2
        histories[pair_id] = {
            "raw_strict": strict,
            "accepted_correct": correct,
            "accepted_error": error,
            "majority_existing": is_existing,
        }
        if is_existing:
            majority_existing.append(pair_id)
    medians = {
        field: float(statistics.median(
            row[field] for row in repeat_counts))
        for field in ("raw_strict", "accepted_correct", "accepted_error")
    }
    if len(majority_existing) != 7:
        raise PilotGateError(
            "pre-bound conservative majority-existing set is not seven")
    return {
        "semantics": (
            "majority_2_of_3_intersection: pair is existing only when both "
            "raw_strict and accepted_correct are true in at least 2 of 3 "
            "pre-bound V6 B/F repeats; pilot count thresholds use the median "
            "over those three repeats; no union or best-repeat semantics"),
        "files": files,
        "repeat_counts": repeat_counts,
        "median_counts": medians,
        "pair_histories": histories,
        "majority_existing_pair_ids": majority_existing,
        "majority_existing_count": len(majority_existing),
    }


def _index_posthoc(posthoc_batch: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {}
    for row in posthoc_batch.get("posthoc", []):
        pair_id = row.get("pair_id")
        evidence = _load_json(Path(row["path"]).resolve())
        if pair_id in indexed or evidence.get("pair_id") != pair_id:
            raise PilotGateError("posthoc pair identity is duplicated/mismatched")
        indexed[pair_id] = evidence
    return indexed


def _run_label(run: Mapping[str, Any], policy_name: str) -> dict[str, bool]:
    label = run.get("policies", {}).get(policy_name)
    if not isinstance(label, dict):
        raise PilotGateError(f"missing policy label {policy_name}")
    raw = label.get("official_raw")
    raw_strict = bool(isinstance(raw, dict) and raw.get("strict") is True)
    usable = label.get("usable_for_reconstruction") is True
    accepted_error = label.get("accepted_strict_error") is True
    return {
        "raw_strict": raw_strict,
        "accepted_correct": bool(usable and raw_strict),
        "accepted_error": accepted_error,
        "usable": usable,
    }


def evaluate_policy(policy_name: str, *, pair_ids: list[str],
                    posthoc: Mapping[str, Any],
                    batch_receipt: Mapping[str, Any],
                    baseline: Mapping[str, Any]) -> dict[str, Any]:
    majority = set(baseline["majority_existing_pair_ids"])
    outer_rows = []
    for outer in range(2):
        labels = {}
        for pair_id in pair_ids:
            runs = posthoc[pair_id].get("runs", [])
            if len(runs) != 2 or runs[outer].get("outer_repeat") != outer:
                raise PilotGateError("posthoc outer structure mismatch")
            labels[pair_id] = _run_label(runs[outer], policy_name)
        retained_raw = sorted(
            pair_id for pair_id in majority
            if labels[pair_id]["raw_strict"])
        retained_correct = sorted(
            pair_id for pair_id in majority
            if labels[pair_id]["accepted_correct"])
        outer_rows.append({
            "outer_repeat": outer,
            "raw_strict": sum(row["raw_strict"] for row in labels.values()),
            "accepted_correct": sum(
                row["accepted_correct"] for row in labels.values()),
            "accepted_error": sum(
                row["accepted_error"] for row in labels.values()),
            "majority_existing_raw_strict_retained": len(retained_raw),
            "majority_existing_accepted_correct_retained": len(
                retained_correct),
            "majority_existing_raw_strict_pair_ids": retained_raw,
            "majority_existing_accepted_correct_pair_ids": retained_correct,
        })

    summary = batch_receipt.get("policy_pair_summary", {}).get(policy_name)
    if not isinstance(summary, dict):
        raise PilotGateError(f"missing GT-free policy summary {policy_name}")
    known_rows = [row for row in summary.get("pairs", [])
                  if row.get("pair_id") == KNOWN_PAIR]
    known_veto = (
        len(known_rows) == 1
        and known_rows[0].get("outer_usable") == [False, False]
        and known_rows[0].get("outcome") == "veto")
    repeatable = (
        summary.get("all_pair_outcomes_repeatable") is True
        and summary.get("mixed_pairs") == 0)
    current_medians = {
        field: float(statistics.median(row[field] for row in outer_rows))
        for field in ("raw_strict", "accepted_correct", "accepted_error")
    }
    baseline_medians = baseline["median_counts"]
    checks = {
        "known_pair_veto_twice": known_veto,
        "all_pair_decisions_repeatable": repeatable,
        "zero_accepted_error": all(
            row["accepted_error"] == 0 for row in outer_rows),
        "majority_existing_raw_strict_loss_at_most_one_each_outer": all(
            row["majority_existing_raw_strict_retained"]
            >= baseline["majority_existing_count"] - MAX_LOSS
            for row in outer_rows),
        "majority_existing_accepted_correct_loss_at_most_one_each_outer": all(
            row["majority_existing_accepted_correct_retained"]
            >= baseline["majority_existing_count"] - MAX_LOSS
            for row in outer_rows),
        "median_raw_strict_loss_at_most_one": (
            current_medians["raw_strict"]
            >= baseline_medians["raw_strict"] - MAX_LOSS),
        "median_accepted_correct_loss_at_most_one": (
            current_medians["accepted_correct"]
            >= baseline_medians["accepted_correct"] - MAX_LOSS),
    }
    return {
        "outer_metrics": outer_rows,
        "current_median_counts": current_medians,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate(manifest: Mapping[str, Any], *,
             batch_receipt: Mapping[str, Any],
             posthoc_batch: Mapping[str, Any]) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in manifest["pairs"]]
    posthoc = _index_posthoc(posthoc_batch)
    if set(posthoc) != set(pair_ids):
        raise PilotGateError("posthoc coverage differs from frozen manifest")
    baseline = load_v6_controls(manifest)
    policy_names = sorted(batch_receipt.get("policy_pair_summary", {}))
    if len(policy_names) != 8:
        raise PilotGateError("formal pilot requires exactly eight policies")
    workers = sum(
        10 for pair in batch_receipt.get("pair_receipts", [])
        for _outer in range(2))
    worker_complete = (
        workers == EXPECTED_WORKERS
        and batch_receipt.get("pair_count") == 12
        and batch_receipt.get("outer_repeats_per_pair") == 2
        and batch_receipt.get("replicates_per_outer")
        == {"forward": 5, "reverse": 5}
        and batch_receipt.get("global_fail_closed_counts")
        == {"exceptions": 0, "nonfinite_transforms": 0,
            "cache_mismatches": 0})
    policies = {
        name: evaluate_policy(
            name, pair_ids=pair_ids, posthoc=posthoc,
            batch_receipt=batch_receipt, baseline=baseline)
        for name in policy_names
    }
    formal = (
        manifest.get("_evidence_mode") == batch.FORMAL_EVIDENCE_MODE
        and manifest.get("_formal_preregistered") is True
        and batch_receipt.get("evidence_mode") == batch.FORMAL_EVIDENCE_MODE
        and posthoc_batch.get("evidence_mode") == batch.FORMAL_EVIDENCE_MODE)
    if not formal:
        status = "INDETERMINATE"
        reasons = ["NON_PREREGISTERED_RESEARCH evidence cannot pass formal gate"]
    elif not worker_complete:
        status = "FAIL"
        reasons = ["240-worker completeness/fail-closed gate failed"]
    elif not all(row["pass"] for row in policies.values()):
        status = "FAIL"
        reasons = ["one or more of all eight frozen policies failed pilot"]
    else:
        status = "PASS"
        reasons = []
    return {
        "schema": SCHEMA,
        "status": status,
        "selection89_authorized": status == "PASS",
        "formal_preregistered": formal,
        "evidence_mode": manifest.get("_evidence_mode"),
        "baseline": baseline,
        "worker_completeness": {
            "expected": EXPECTED_WORKERS,
            "observed": workers,
            "complete": worker_complete,
        },
        "all_8_policies_pass": all(row["pass"] for row in policies.values()),
        "policies": policies,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=batch.DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256",
                        default=batch.DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--research-non-preregistered", action="store_true")
    parser.add_argument("--batch-receipt", type=Path, required=True)
    parser.add_argument("--posthoc-batch-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = batch.validate_manifest(
        args.manifest, args.manifest_sha256,
        allow_non_preregistered=args.research_non_preregistered)
    batch_path = args.batch_receipt.resolve()
    posthoc_path = args.posthoc_batch_receipt.resolve()
    repository = batch.repository_state(batch_path.parent)
    snapshot = batch.source_snapshot(repository)
    snapshot["manifest_evidence_mode"] = manifest["_evidence_mode"]
    snapshot["snapshot_sha256"] = pilot.stable_json_hash(
        {key: value for key, value in snapshot.items()
         if key != "snapshot_sha256"})
    batch_receipt = batch.validate_batch_receipt(
        batch_path, manifest=manifest, snapshot=snapshot)
    posthoc_batch = batch.validate_posthoc_batch_receipt(
        posthoc_path, batch_receipt_path=batch_path,
        batch=batch_receipt, snapshot=snapshot)
    output = evaluate(
        manifest, batch_receipt=batch_receipt,
        posthoc_batch=posthoc_batch)
    output["inputs"] = {
        "manifest": {"path": manifest["_path"],
                     "sha256": manifest["_file_sha256"]},
        "batch_receipt": {"path": str(batch_path),
                          "sha256": pilot.sha256_file(batch_path)},
        "posthoc_batch_receipt": {
            "path": str(posthoc_path),
            "sha256": pilot.sha256_file(posthoc_path)},
    }
    output["source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    output["evidence_sha256"] = pilot.stable_json_hash(output)
    pilot.atomic_create_json(args.out.resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
