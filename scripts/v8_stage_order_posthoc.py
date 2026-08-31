"""Independent label evaluator for a frozen V8 GT-free replay receipt."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from adapters.sgf.data_sources import load_gt_transform  # noqa: E402
from inference import RELAXED, STRICT  # noqa: E402
import v7_registration_pilot as v7_pilot  # noqa: E402
import v8_stage_order_replay as replay  # noqa: E402


SCHEMA = "v8-stage-order-consensus-posthoc-v1"


class V8PosthocError(RuntimeError):
    pass


def _pose_error(transform: Any, truth: np.ndarray) -> dict[str, Any]:
    estimate = np.asarray(transform, dtype=np.float64)
    if estimate.shape != (4, 4) or not np.isfinite(estimate).all():
        raise V8PosthocError("selected transform is not finite 4x4")
    cosine = (np.trace(estimate[:3, :3].T @ truth[:3, :3]) - 1.0) / 2.0
    rotation = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    return {
        "rotation_error_deg": rotation,
        "translation_error_m": translation,
        "strict": bool(rotation <= STRICT[0] and translation <= STRICT[1]),
        "relaxed": bool(
            rotation <= RELAXED[0] and translation <= RELAXED[1]),
    }


def validate_replay(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise V8PosthocError("invalid replay receipt") from exc
    expected = data.pop("evidence_sha256", None)
    actual = v7_pilot.stable_json_hash(data)
    data["evidence_sha256"] = expected
    if (data.get("schema") != replay.SCHEMA
            or data.get("status") != "GT_FREE_DEVELOPMENT_REPLAY_COMPLETE"
            or data.get("posthoc_not_run") is not True
            or data.get("development_split_exposed") is not True
            or data.get("qualifies_as_blind_gate") is not False
            or expected != actual):
        raise V8PosthocError("replay receipt provenance/status mismatch")
    source = Path(data["source_batch_receipt"]["path"]).resolve()
    if v7_pilot.sha256_file(source) != data["source_batch_receipt"][
            "file_sha256"]:
        raise V8PosthocError("source batch receipt changed")
    manifest = Path(data["manifest"]["path"]).resolve()
    if v7_pilot.sha256_file(manifest) != data["manifest"]["sha256"]:
        raise V8PosthocError("manifest changed")
    if (v7_pilot.stable_json_hash(data["worker_bindings"])
            != data["worker_binding_sha256"]):
        raise V8PosthocError("worker binding list changed")
    for row in data["worker_bindings"]:
        worker = Path(row["path"]).resolve()
        if v7_pilot.sha256_file(worker) != row["file_sha256"]:
            raise V8PosthocError(f"worker file changed {worker}")
        body = json.loads(worker.read_text())
        if (body.get("evidence_sha256") != row["evidence_sha256"]
                or body.get("raw_transform_sha256")
                != row["raw_transform_sha256"]
                or body.get("final_transform_sha256")
                != row["final_transform_sha256"]):
            raise V8PosthocError(f"worker internal binding changed {worker}")
    return data


def label(receipt: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for pair in receipt["pairs"]:
        truth = np.asarray(
            load_gt_transform(pair["pair_id"]), dtype=np.float64).reshape(4, 4)
        outers = []
        for row in pair["outers"]:
            result = row["result"]
            selected = result.get("selected_observed_forward_medoid")
            if selected is None:
                outers.append({
                    "outer_repeat": row["outer_repeat"], "selected": False,
                    "usable": False, "official_raw": None,
                    "reconstruction_final": None,
                    "accepted_strict_error": False})
                continue
            raw = _pose_error(selected["raw_transform"], truth)
            final = _pose_error(selected["final_transform"], truth)
            usable = result["usable_for_reconstruction"] is True
            outers.append({
                "outer_repeat": row["outer_repeat"], "selected": True,
                "usable": usable, "official_raw": raw,
                "reconstruction_final": final,
                "accepted_strict_error": bool(usable and not raw["strict"]),
                "worker_evidence_sha256": selected[
                    "worker_evidence_sha256"],
            })
        pairs.append({"pair_id": pair["pair_id"], "outers": outers})
    per_outer = []
    for outer in range(2):
        rows = [pair["outers"][outer] for pair in pairs]
        per_outer.append({
            "outer_repeat": outer,
            "usable": sum(row["usable"] for row in rows),
            "raw_strict": sum(bool(row["official_raw"]
                                   and row["official_raw"]["strict"])
                              for row in rows),
            "reconstruction_final_strict": sum(bool(
                row["reconstruction_final"]
                and row["reconstruction_final"]["strict"]) for row in rows),
            "accepted_correct": sum(bool(
                row["usable"] and row["official_raw"]
                and row["official_raw"]["strict"]) for row in rows),
            "accepted_error": sum(
                row["accepted_strict_error"] for row in rows),
        })
    return {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_POSTHOC_COMPLETE",
        "development_split_exposed": True,
        "qualifies_as_blind_gate": False,
        "gt_scope": "loaded only after the GT-free replay receipt was frozen",
        "pairs": pairs,
        "per_outer": per_outer,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frozen = validate_replay(args.receipt.resolve())
    output = label(frozen)
    output["replay_receipt"] = {
        "path": str(args.receipt.resolve()),
        "file_sha256": v7_pilot.sha256_file(args.receipt),
        "evidence_sha256": frozen["evidence_sha256"],
    }
    output["source_sha256"] = v7_pilot.sha256_file(Path(__file__).resolve())
    output["evidence_sha256"] = v7_pilot.stable_json_hash(output)
    v7_pilot.atomic_create_json(args.out.resolve(), output)
    print(json.dumps({"status": output["status"],
                      "per_outer": output["per_outer"],
                      "evidence_sha256": output["evidence_sha256"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
