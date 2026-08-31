"""Independent, single-use posthoc gate for a frozen calibration90 batch.

This is the only V8 calibration module that imports ground truth.  It must not
be executed until the GT-free receipt and its source/cache hashes are frozen.
The gate cannot tune thresholds or rerun the worker batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from adapters.sgf.data_sources import load_gt_transform  # noqa: E402
from inference import RELAXED, STRICT  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402
import v8_calibration90_locked as locked  # noqa: E402

SCHEMA = "v8-calibration90-posthoc-gate-v1"


class Calibration90PosthocError(RuntimeError):
    pass


def claim_posthoc(manifest: Mapping[str, Any], batch_path: Path,
                  batch: Mapping[str, Any]) -> Path:
    """Consume the independent label-opening right before GT is loaded."""
    root = Path(manifest["single_use"]["claim_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (
        f"{manifest['_file_sha256']}.posthoc-started.json")
    value = {
        "schema": "v8-calibration90-posthoc-single-use-claim-v1",
        "status": "CONSUMED",
        "manifest_path": manifest["_path"],
        "manifest_sha256": manifest["_file_sha256"],
        "batch_receipt_path": str(batch_path.resolve()),
        "batch_receipt_sha256": locked.sha256_file(batch_path.resolve()),
        "batch_evidence_sha256": batch["evidence_sha256"],
        "rerun_allowed": False,
        "claim_removed_on_failure": False,
    }
    locked._atomic_create(destination, value)
    return destination


def pose_error(transform: Any, truth: np.ndarray) -> dict[str, Any]:
    estimate = np.asarray(transform, dtype=np.float64)
    if estimate.shape != (4, 4) or not np.isfinite(estimate).all():
        raise Calibration90PosthocError("selected transform is not finite 4x4")
    cosine = (np.trace(estimate[:3, :3].T @ truth[:3, :3]) - 1.0) / 2.0
    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    return {
        "rotation_error_deg": rotation,
        "translation_error_m": translation,
        "strict": bool(rotation <= STRICT[0] and translation <= STRICT[1]),
        "relaxed": bool(rotation <= RELAXED[0] and translation <= RELAXED[1]),
    }


def _load_outer(path: Path, expected_sha: str, *, pair_id: str,
                cache_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    if locked.sha256_file(path) != expected_sha:
        raise Calibration90PosthocError("outer receipt file SHA mismatch")
    value = json.loads(path.read_text())
    evidence = value.pop("evidence_sha256", None)
    if evidence != locked.stable_hash(value):
        raise Calibration90PosthocError("outer receipt embedded SHA mismatch")
    value["evidence_sha256"] = evidence
    if (value.get("schema") != locked.OUTER_SCHEMA
            or value.get("status") != "GT_FREE_COMPLETE"
            or value.get("posthoc_not_run") is not True
            or value.get("worker_count") != 10
            or value.get("pair_id") != pair_id
            or value.get("cache_sha256") != cache_sha256
            or value.get("manifest_sha256") != manifest_sha256):
        raise Calibration90PosthocError("outer receipt status mismatch")
    bindings = value.get("workers", [])
    expected = {(direction, replicate)
                for direction in locked.DIRECTIONS
                for replicate in range(locked.REPLICATES)}
    actual = {(row.get("direction"), row.get("replicate"))
              for row in bindings}
    if len(bindings) != 10 or actual != expected:
        raise Calibration90PosthocError("outer worker coverage mismatch")
    for row in bindings:
        worker_path = Path(row["path"]).resolve()
        if locked.sha256_file(worker_path) != row.get("file_sha256"):
            raise Calibration90PosthocError("worker file SHA mismatch")
        worker = locked.v7_batch._load_batch_worker(
            worker_path, pair_id=pair_id, direction=row["direction"],
            replicate=row["replicate"], cache_sha=cache_sha256,
            protocol_sha=pilot.protocol_sha256())
        if worker.get("evidence_sha256") != row.get("evidence_sha256"):
            raise Calibration90PosthocError("worker evidence binding mismatch")
    return value


def evaluate(manifest: Mapping[str, Any], batch: Mapping[str, Any], *,
             gt_loader: Callable[[str], Any] = load_gt_transform) \
        -> dict[str, Any]:
    thresholds = manifest["thresholds"]
    labelled_pairs = []
    for pair_row in batch["pair_receipts"]:
        pair_receipt_path = Path(pair_row["path"]).resolve()
        pair_receipt = json.loads(pair_receipt_path.read_text())
        pair_evidence = pair_receipt.pop("evidence_sha256", None)
        if (locked.sha256_file(pair_receipt_path)
                != pair_row["file_sha256"]
                or pair_evidence != locked.stable_hash(pair_receipt)):
            raise Calibration90PosthocError("pair receipt binding mismatch")
        pair_id = pair_row["pair_id"]
        manifest_pair = next(
            (row for row in manifest["pairs"] if row["pair_id"] == pair_id),
            None)
        if (manifest_pair is None
                or pair_receipt.get("schema") != locked.PAIR_SCHEMA
                or pair_receipt.get("status") != "GT_FREE_COMPLETE"
                or pair_receipt.get("pair_id") != pair_id
                or pair_receipt.get("posthoc_not_run") is not True
                or len(pair_receipt.get("outers", []))
                != locked.OUTER_REPEATS):
            raise Calibration90PosthocError("pair receipt contract mismatch")
        truth = np.asarray(gt_loader(pair_id), dtype=np.float64).reshape(4, 4)
        outers = []
        for outer_row in pair_receipt["outers"]:
            outer = _load_outer(
                Path(outer_row["path"]).resolve(), outer_row["file_sha256"],
                pair_id=pair_id,
                cache_sha256=manifest_pair["cache_sha256"],
                manifest_sha256=manifest["_file_sha256"])
            result = outer["v8_result"]
            selected = result.get("selected_observed_forward_medoid")
            usable = result.get("usable_for_reconstruction") is True
            if selected is None:
                outers.append({
                    "outer_repeat": outer["outer_repeat"],
                    "selected": False, "usable": False,
                    "raw": None, "final": None,
                    "accepted_correct": False, "accepted_error": False})
                continue
            raw = pose_error(selected["raw_transform"], truth)
            final = pose_error(selected["final_transform"], truth)
            outers.append({
                "outer_repeat": outer["outer_repeat"],
                "selected": True, "usable": usable,
                "raw": raw, "final": final,
                "accepted_correct": bool(usable and raw["strict"]),
                "accepted_error": bool(usable and not raw["strict"]),
                "worker_evidence_sha256": selected[
                    "worker_evidence_sha256"],
            })
        labelled_pairs.append({"pair_id": pair_id, "outers": outers})

    summaries = []
    for outer in range(locked.OUTER_REPEATS):
        rows = [pair["outers"][outer] for pair in labelled_pairs]
        summaries.append({
            "outer_repeat": outer,
            "completed": len(rows),
            "strict": sum(bool(row["raw"] and row["raw"]["strict"])
                          for row in rows),
            "relaxed": sum(bool(row["raw"] and row["raw"]["relaxed"])
                           for row in rows),
            "accepted_correct": sum(row["accepted_correct"] for row in rows),
            "accepted_error": sum(row["accepted_error"] for row in rows),
        })
    checks = {
        "completed": all(row["completed"] == thresholds["completed"]
                         for row in summaries),
        "strict_floor": all(row["strict"] >= thresholds["strict_min"]
                            for row in summaries),
        "relaxed_floor": all(row["relaxed"] >= thresholds["relaxed_min"]
                             for row in summaries),
        "accepted_correct_floor": all(
            row["accepted_correct"] >= thresholds["accepted_correct_min"]
            for row in summaries),
        "zero_accepted_error": all(
            row["accepted_error"] <= thresholds["accepted_error_max"]
            for row in summaries),
        "all_pair_outcomes_repeatable": batch["repeatable_pairs"]
            == thresholds["repeatable_pairs"],
        "worker_fail_closed_counts": batch["global_fail_closed_counts"]
            == {"exceptions": 0, "nonfinite": 0, "cache_mismatches": 0},
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "calibration90_first_open": True,
        "thresholds_precommitted": dict(thresholds),
        "checks": checks,
        "per_outer": summaries,
        "pairs": labelled_pairs,
        "fixed12_authorized": passed,
        "official92_authorized": False,
        "rerun_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--batch-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().exists():
        raise Calibration90PosthocError("posthoc requires a fresh output path")
    manifest = locked.validate_manifest(
        args.manifest, args.manifest_sha256, verify_caches=True)
    batch = locked.validate_batch_receipt(args.batch_receipt, manifest)
    claim = claim_posthoc(manifest, args.batch_receipt, batch)
    result = evaluate(manifest, batch)
    result["manifest"] = {
        "path": str(args.manifest.resolve()),
        "file_sha256": args.manifest_sha256,
    }
    result["batch_receipt"] = {
        "path": str(args.batch_receipt.resolve()),
        "file_sha256": locked.sha256_file(args.batch_receipt.resolve()),
        "evidence_sha256": batch["evidence_sha256"],
    }
    result["single_use_claim"] = {
        "path": str(claim),
        "file_sha256": locked.sha256_file(claim),
    }
    result["evidence_sha256"] = locked.stable_hash(result)
    locked._atomic_create(args.out.resolve(), result)
    print(json.dumps({"status": result["status"],
                      "fixed12_authorized": result["fixed12_authorized"],
                      "official92_authorized": False,
                      "evidence_sha256": result["evidence_sha256"]},
                     indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
