"""Preregistered, label-loader-free mechanical gate after selection89 posthoc.

The thresholds are frozen before current-run labels are loaded.  This process
does not import or call a GT loader.  It revalidates all GT-free workers from
the frozen batch, binds both immutable receipts, then consumes only the already
frozen posthoc counts.  PASS permits a later calibration task to be proposed;
it does not run or authorize calibration itself.
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

import v7_registration_pilot as pilot  # noqa: E402
import v8_selection89_development as dev  # noqa: E402
import v8_selection89_replay as replay_runner  # noqa: E402
import v8_stage_order_replay as replay_schema  # noqa: E402


SCHEMA = "v8-selection89-calibration-gate-v1"
POSTHOC_SCHEMA = "v8-stage-order-consensus-posthoc-v1"


class CalibrationGateError(RuntimeError):
    """Evidence is malformed or not the exact preregistered chain."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationGateError(f"invalid JSON {path}") from exc
    if not isinstance(value, dict):
        raise CalibrationGateError(f"JSON object required {path}")
    return value


def _validated_evidence(path: Path, *, schema: str) -> dict[str, Any]:
    value = _load_json(path)
    expected = value.pop("evidence_sha256", None)
    actual = pilot.stable_json_hash(value)
    value["evidence_sha256"] = expected
    if expected != actual or value.get("schema") != schema:
        raise CalibrationGateError(f"evidence/schema mismatch {path}")
    return value


def _validate_replay_and_workers(path: Path) -> tuple[dict[str, Any], dict[str, int]]:
    receipt = _validated_evidence(path, schema=replay_schema.SCHEMA)
    if (receipt.get("status") != "GT_FREE_DEVELOPMENT_REPLAY_COMPLETE"
            or receipt.get("posthoc_not_run") is not True
            or receipt.get("development_split_exposed") is not True
            or receipt.get("qualifies_as_blind_gate") is not False
            or receipt.get("fresh_v8_worker_cache") is not True):
        raise CalibrationGateError("replay provenance/status mismatch")
    batch_path = Path(receipt["source_batch_receipt"]["path"]).resolve()
    manifest_path = Path(receipt["manifest"]["path"]).resolve()
    manifest, batch, loaded = replay_runner._validate_batch(
        batch_path, manifest_path, receipt["manifest"]["sha256"])
    if (dev.sha256_file(batch_path)
            != receipt["source_batch_receipt"]["file_sha256"]):
        raise CalibrationGateError("batch file SHA drift")
    worker_count = sum(
        len(outer["workers"])
        for pair in loaded for outer in pair["outers"])
    if (worker_count != receipt.get("worker_binding_count")
            or worker_count != len(receipt.get("worker_bindings", []))):
        raise CalibrationGateError("worker binding count mismatch")
    if [row["pair_id"] for row in manifest["pairs"]] != [
            row.get("pair_id") for row in receipt.get("pairs", [])]:
        raise CalibrationGateError("replay pair order mismatch")
    return receipt, {
        "worker_count": worker_count,
        "worker_exceptions": 0,
        "worker_nonfinite": 0,
        "worker_cache_or_hash_mismatch": 0,
        "pair_count": len(receipt["pairs"]),
        "outer_repeats": batch["outer_repeats_per_pair"],
    }


def _validate_posthoc(path: Path, replay_path: Path,
                      replay: Mapping[str, Any]) -> dict[str, Any]:
    posthoc = _validated_evidence(path, schema=POSTHOC_SCHEMA)
    if (posthoc.get("status") != "DEVELOPMENT_POSTHOC_COMPLETE"
            or posthoc.get("development_split_exposed") is not True
            or posthoc.get("qualifies_as_blind_gate") is not False
            or posthoc.get("gt_scope")
            != "loaded only after the GT-free replay receipt was frozen"):
        raise CalibrationGateError("posthoc provenance/status mismatch")
    binding = posthoc.get("replay_receipt", {})
    if (Path(binding.get("path", "")).resolve() != replay_path.resolve()
            or binding.get("file_sha256") != dev.sha256_file(replay_path)
            or binding.get("evidence_sha256") != replay["evidence_sha256"]):
        raise CalibrationGateError("posthoc does not bind exact replay")
    return posthoc


def evaluate_gate(replay: Mapping[str, Any], posthoc: Mapping[str, Any],
                  audit: Mapping[str, int], spec: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = spec["mechanical_thresholds"]
    pairs = replay["pairs"]
    repeatable_count = sum(bool(row.get("repeatable")) for row in pairs)
    known_bad = spec["known_bad_pair_id"]
    by_pair = {row["pair_id"]: row for row in pairs}
    if known_bad not in by_pair:
        raise CalibrationGateError("known-bad pair absent from selection89")
    known_bad_outcomes = by_pair[known_bad]["outer_outcomes"]
    per_outer = posthoc.get("per_outer")
    if (not isinstance(per_outer, list)
            or [row.get("outer_repeat") for row in per_outer] != [0, 1]):
        raise CalibrationGateError("posthoc outer shape mismatch")
    checks = {
        "workers_complete": audit["worker_count"] == thresholds["workers_complete"],
        "worker_exceptions_zero": audit["worker_exceptions"]
        <= thresholds["worker_exceptions_max"],
        "worker_nonfinite_zero": audit["worker_nonfinite"]
        <= thresholds["worker_nonfinite_max"],
        "worker_cache_or_hash_mismatch_zero": audit[
            "worker_cache_or_hash_mismatch"]
        <= thresholds["worker_cache_or_hash_mismatch_max"],
        "pair_count_exact": audit["pair_count"] == thresholds["pair_count"],
        "outer_repeats_exact": audit["outer_repeats"] == thresholds["outer_repeats"],
        "all_pair_verdicts_repeatable": (
            repeatable_count / len(pairs)
            >= thresholds["pair_verdict_repeatability_fraction_min"]),
        "known_bad_veto_each_outer": known_bad_outcomes == [False, False],
        "accepted_error_zero_each_outer": all(
            row.get("accepted_error")
            <= thresholds["accepted_error_max_each_outer"]
            for row in per_outer),
        "accepted_correct_min_each_outer": all(
            row.get("accepted_correct")
            >= thresholds["accepted_correct_min_each_outer"]
            for row in per_outer),
        "accepted_correct_ge_10_outer_count": sum(
            row.get("accepted_correct", -1) >= 10 for row in per_outer)
        >= thresholds["accepted_correct_ge_10_min_outer_count"],
        "raw_strict_min_each_outer": all(
            row.get("raw_strict") >= thresholds["raw_strict_min_each_outer"]
            for row in per_outer),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": ("CALIBRATION_MECHANICAL_GATE_PASS" if passed
                   else "CALIBRATION_MECHANICAL_GATE_FAIL"),
        "may_propose_later_calibration": passed,
        "automatic_calibration_authorized": False,
        "development_split_exposed": True,
        "qualifies_as_confirmatory_evidence": False,
        "checks": checks,
        "audit": dict(audit),
        "repeatable_pair_count": repeatable_count,
        "known_bad_pair_id": known_bad,
        "known_bad_outer_outcomes": known_bad_outcomes,
        "per_outer": per_outer,
        "thresholds": thresholds,
        "historical_baseline": spec["frozen_historical_baseline"],
        "failed_checks": sorted(name for name, value in checks.items()
                                if not value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--posthoc", type=Path, required=True)
    parser.add_argument("--gate-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay_path = args.replay_receipt.resolve()
    posthoc_path = args.posthoc.resolve()
    spec_path = args.gate_spec.resolve()
    replay, audit = _validate_replay_and_workers(replay_path)
    posthoc = _validate_posthoc(posthoc_path, replay_path, replay)
    spec = _load_json(spec_path)
    if (spec.get("schema") != "v8-selection89-calibration-gate-spec-v1"
            or spec.get("status")
            != "PREREGISTERED_BEFORE_CURRENT_LABEL_LOADING"):
        raise CalibrationGateError("gate specification provenance mismatch")
    result = evaluate_gate(replay, posthoc, audit, spec)
    result["bindings"] = {
        "replay": {"path": str(replay_path),
                   "sha256": dev.sha256_file(replay_path),
                   "evidence_sha256": replay["evidence_sha256"]},
        "posthoc": {"path": str(posthoc_path),
                    "sha256": dev.sha256_file(posthoc_path),
                    "evidence_sha256": posthoc["evidence_sha256"]},
        "gate_spec": {"path": str(spec_path),
                      "sha256": dev.sha256_file(spec_path)},
        "gate_implementation": {
            "path": str(Path(__file__).resolve().relative_to(CODE_ROOT)),
            "sha256": dev.sha256_file(Path(__file__).resolve()),
        },
    }
    result["evidence_sha256"] = pilot.stable_json_hash(result)
    pilot.atomic_create_json(args.out.resolve(), result)
    print(json.dumps({"status": result["status"],
                      "failed_checks": result["failed_checks"],
                      "evidence_sha256": result["evidence_sha256"]},
                     indent=2))
    return 0 if result["may_propose_later_calibration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
