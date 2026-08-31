"""Independent posthoc labels for frozen V7 GT-free pilot evidence."""
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
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from adapters.sgf.data_sources import load_gt_transform  # noqa: E402
from inference import RELAXED, STRICT  # noqa: E402
from v7_registration_pilot import (  # noqa: E402
    SCHEMA,
    atomic_create_json,
    sha256_file,
    stable_json_hash,
)


POSTHOC_SCHEMA = "v7-registration-veto-posthoc-v1"


class PosthocEvidenceError(RuntimeError):
    pass


def pose_error(transform: Any, truth: np.ndarray) -> dict[str, Any]:
    estimate = np.asarray(transform, dtype=np.float64)
    if estimate.shape != (4, 4) or not np.isfinite(estimate).all():
        raise PosthocEvidenceError("selected transform is not finite 4x4")
    cosine = (np.trace(estimate[:3, :3].T @ truth[:3, :3]) - 1.0) / 2.0
    rotation_error = float(np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation_error = float(np.linalg.norm(
        estimate[:3, 3] - truth[:3, 3]))
    return {
        "rotation_error_deg": rotation_error,
        "translation_error_m": translation_error,
        "strict": bool(rotation_error <= STRICT[0]
                       and translation_error <= STRICT[1]),
        "relaxed": bool(rotation_error <= RELAXED[0]
                        and translation_error <= RELAXED[1]),
    }


def validate_aggregate(path: Path, expected_sha: str, *, pair_id: str,
                       outer_repeat: int, manifest_sha256: str,
                       source_snapshot_sha256: str,
                       evidence_mode: str) -> dict:
    if sha256_file(path) != expected_sha:
        raise PosthocEvidenceError(f"aggregate file SHA mismatch {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or data.get("status") != "GT_FREE_COMPLETE":
        raise PosthocEvidenceError(f"not a frozen GT-free aggregate {path}")
    expected = data.pop("evidence_sha256", None)
    actual = stable_json_hash(data)
    data["evidence_sha256"] = expected
    if expected != actual:
        raise PosthocEvidenceError(f"aggregate evidence SHA mismatch {path}")
    if (data.get("pair_id") != pair_id
            or data.get("outer_repeat") != outer_repeat
            or data.get("batch", {}).get("manifest_sha256")
            != manifest_sha256
            or data.get("batch", {}).get("source_snapshot_sha256")
            != source_snapshot_sha256
            or data.get("batch", {}).get("evidence_mode") != evidence_mode):
        raise PosthocEvidenceError(f"aggregate provenance mismatch {path}")
    return data


def label_aggregate(data: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = data["pair_id"]
    truth = np.asarray(load_gt_transform(pair_id), dtype=np.float64).reshape(4, 4)
    policies = {}
    for name, policy in sorted(data["policies"].items()):
        selected = policy.get("selected_observed_forward_medoid")
        if selected is None:
            policies[name] = {
                "selected": False,
                "usable_for_reconstruction": False,
                "raw": None,
                "final": None,
                "accepted_strict_error": False,
            }
            continue
        raw = pose_error(selected["raw_transform"], truth)
        final = pose_error(selected["final_transform"], truth)
        usable = bool(policy["usable_for_reconstruction"])
        policies[name] = {
            "selected": True,
            "usable_for_reconstruction": usable,
            # Official strict is always the raw registration transform.
            "official_raw": raw,
            # Reconstruction consumes the final ICP transform.
            "reconstruction_final": final,
            "accepted_strict_error": bool(usable and not raw["strict"]),
            "worker_evidence_sha256": selected["worker_evidence_sha256"],
        }
    return {
        "pair_id": pair_id,
        "outer_repeat": data["outer_repeat"],
        "gt_free_evidence_sha256": data["evidence_sha256"],
        "policies": policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text())
    receipt_expected = receipt.pop("evidence_sha256", None)
    receipt_actual = stable_json_hash(receipt)
    receipt["evidence_sha256"] = receipt_expected
    if (receipt.get("schema") != "v7-registration-veto-pilot-receipt-v1"
            or receipt.get("status") != "GT_FREE_COMPLETE"
            or receipt.get("posthoc_not_run") is not True
            or receipt.get("outer_repeats") != 2
            or len(receipt.get("aggregates", [])) != 2
            or receipt_expected != receipt_actual):
        raise PosthocEvidenceError("pilot receipt is incomplete")
    pair_id = receipt.get("pair_id")
    manifest_sha = receipt.get("batch", {}).get("manifest_sha256")
    source_snapshot_sha = receipt.get("batch", {}).get(
        "source_snapshot_sha256")
    evidence_mode = receipt.get("batch", {}).get("evidence_mode")
    if (not isinstance(pair_id, str)
            or not isinstance(manifest_sha, str)
            or not isinstance(source_snapshot_sha, str)
            or not isinstance(evidence_mode, str)):
        raise PosthocEvidenceError("pilot receipt bindings are incomplete")
    labelled = []
    aggregate_bindings = []
    for outer, row in enumerate(receipt["aggregates"]):
        path = Path(row["path"]).resolve()
        labelled.append(label_aggregate(validate_aggregate(
            path, row["sha256"], pair_id=pair_id, outer_repeat=outer,
            manifest_sha256=manifest_sha,
            source_snapshot_sha256=source_snapshot_sha,
            evidence_mode=evidence_mode)))
        aggregate_bindings.append({
            "outer_repeat": outer,
            "path": row["path"],
            "sha256": row["sha256"],
        })
    output = {
        "schema": POSTHOC_SCHEMA,
        "status": "POSTHOC_COMPLETE",
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        },
        "pair_id": pair_id,
        "outer_repeats": 2,
        "aggregate_bindings": aggregate_bindings,
        "manifest_sha256": manifest_sha,
        "source_snapshot_sha256": source_snapshot_sha,
        "evidence_mode": evidence_mode,
        "gt_scope": "loaded only after both GT-free aggregates were frozen",
        "runs": labelled,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output["evidence_sha256"] = stable_json_hash(output)
    atomic_create_json(args.out.resolve(), output)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
