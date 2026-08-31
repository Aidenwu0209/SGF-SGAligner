"""GT-free stop gate that must pass before any selection89 labels are loaded."""
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
import v8_selection89_calibration_gate as final_gate  # noqa: E402
import v8_selection89_development as dev  # noqa: E402


SCHEMA = "v8-selection89-prelabel-gate-v1"


def evaluate_prelabel(replay: Mapping[str, Any], audit: Mapping[str, int],
                      spec: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = spec["mechanical_thresholds"]
    pairs = replay["pairs"]
    repeatable_count = sum(bool(row.get("repeatable")) for row in pairs)
    known_bad = spec["known_bad_pair_id"]
    by_pair = {row["pair_id"]: row for row in pairs}
    if known_bad not in by_pair:
        raise final_gate.CalibrationGateError(
            "known-bad pair absent from selection89")
    known_bad_outcomes = by_pair[known_bad]["outer_outcomes"]
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
            len(pairs) > 0 and repeatable_count / len(pairs)
            >= thresholds["pair_verdict_repeatability_fraction_min"]),
        "known_bad_veto_each_outer": known_bad_outcomes == [False, False],
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": ("GTFREE_PRELABEL_GATE_PASS" if passed
                   else "GTFREE_PRELABEL_GATE_FAIL"),
        "label_loading_authorized": passed,
        "calibration_authorized": False,
        "development_split_exposed": True,
        "checks": checks,
        "audit": dict(audit),
        "repeatable_pair_count": repeatable_count,
        "repeatability_fraction": repeatable_count / len(pairs),
        "known_bad_pair_id": known_bad,
        "known_bad_outer_outcomes": known_bad_outcomes,
        "failed_checks": sorted(name for name, value in checks.items()
                                if not value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--gate-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay_path = args.replay_receipt.resolve()
    spec_path = args.gate_spec.resolve()
    replay, audit = final_gate._validate_replay_and_workers(replay_path)
    spec = final_gate._load_json(spec_path)
    if (spec.get("schema") != "v8-selection89-calibration-gate-spec-v1"
            or spec.get("status")
            != "PREREGISTERED_BEFORE_CURRENT_LABEL_LOADING"):
        raise final_gate.CalibrationGateError(
            "gate specification provenance mismatch")
    result = evaluate_prelabel(replay, audit, spec)
    result["bindings"] = {
        "replay": {"path": str(replay_path),
                   "sha256": dev.sha256_file(replay_path),
                   "evidence_sha256": replay["evidence_sha256"]},
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
                      "label_loading_authorized": result[
                          "label_loading_authorized"],
                      "repeatable_pair_count": result[
                          "repeatable_pair_count"],
                      "known_bad_outer_outcomes": result[
                          "known_bad_outer_outcomes"],
                      "failed_checks": result["failed_checks"],
                      "evidence_sha256": result["evidence_sha256"]},
                     indent=2))
    return 0 if result["label_loading_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
