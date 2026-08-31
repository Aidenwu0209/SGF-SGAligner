"""Label-free failure diagnostics for a frozen selection89 V8 replay.

This report is diagnostic only.  Pooled 10+10 and leave-one-worker-out results
must never be used to alter the preregistered gate or to claim calibration
eligibility.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config,
    cluster_direction,
    cross_final_agreement,
)
import v7_registration_pilot as pilot  # noqa: E402
import v8_selection89_calibration_gate as gate  # noqa: E402
import v8_selection89_development as dev  # noqa: E402
import v8_selection89_replay as replay_runner  # noqa: E402


SCHEMA = "v8-selection89-gtfree-diagnostics-v1"


def _records(workers: Sequence[Mapping[str, Any]], direction: str) -> list[dict[str, Any]]:
    rows = sorted((row for row in workers if row["direction"] == direction),
                  key=lambda row: row["replicate"])
    output = []
    for row in rows:
        transform = np.asarray(row["final_transform"], dtype=np.float64)
        if direction == "reverse":
            transform = np.linalg.inv(transform)
        output.append({"status": row["status"], "transform": transform,
                       "stable_signature": row[
                           "permutation_provenance_sha256"]})
    return output


def _loo(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    config = V8Config(repeats=9, quorum=8)
    for omitted in range(len(records)):
        cluster = cluster_direction(
            [row for index, row in enumerate(records) if index != omitted],
            config)
        rows.append({"omitted_index": omitted, "usable": cluster["usable"],
                     "largest_clique": max(cluster["clique_sizes"], default=0),
                     "rejection_reasons": cluster["rejection_reasons"]})
    return {
        "runs": rows,
        "all_usable": all(row["usable"] for row in rows),
        "largest_clique_min": min(row["largest_clique"] for row in rows),
        "largest_clique_max": max(row["largest_clique"] for row in rows),
    }


def diagnose(replay: Mapping[str, Any], loaded: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    loaded_by_id = {row["pair"]["pair_id"]: row for row in loaded}
    failures: collections.Counter[str] = collections.Counter()
    pair_rows = []
    for pair in replay["pairs"]:
        pair_id = pair["pair_id"]
        stage_rows = []
        for outer in pair["outers"]:
            result = outer["result"]
            f_cluster = result["directional_final_consensus"]["forward"]
            r_cluster = result["directional_final_consensus"]["reverse_inverted"]
            f_rule = result["medoid_rule_b"]["forward"]
            r_rule = result["medoid_rule_b"]["reverse"]
            cross = result["cross_final"]
            f_trace = result["medoid_fixed_trace"]["forward"]
            r_trace = result["medoid_fixed_trace"]["reverse"]
            for prefix, row in (("forward_cluster", f_cluster),
                                ("reverse_cluster", r_cluster),
                                ("forward_rule_b", f_rule),
                                ("reverse_rule_b", r_rule),
                                ("cross_final", cross),
                                ("forward_trace", f_trace),
                                ("reverse_trace", r_trace)):
                if not row.get("usable"):
                    for reason in row.get("rejection_reasons", []):
                        failures[f"{prefix}:{reason}"] += 1
            stage_rows.append({
                "outer_repeat": outer["outer_repeat"],
                "usable": result["usable_for_reconstruction"],
                "forward_clique_sizes": f_cluster["clique_sizes"],
                "reverse_clique_sizes": r_cluster["clique_sizes"],
                "cross_agreement_count": cross["agreement_count"],
                "forward_rule_b": f_rule,
                "reverse_rule_b": r_rule,
                "forward_trace": f_trace,
                "reverse_trace": r_trace,
            })
        loaded_pair = loaded_by_id[pair_id]
        pooled_workers = [worker for outer in loaded_pair["outers"]
                          for worker in outer["workers"]]
        forward = _records(pooled_workers, "forward")
        reverse = _records(pooled_workers, "reverse")
        pooled_config = V8Config(repeats=10, quorum=8)
        f_pooled = cluster_direction(forward, pooled_config)
        r_pooled = cluster_direction(reverse, pooled_config)
        cross = cross_final_agreement(
            forward, reverse, f_pooled["winning_original_indices"],
            r_pooled["winning_original_indices"], pooled_config)
        pair_rows.append({
            "pair_id": pair_id,
            "outer_outcomes": pair["outer_outcomes"],
            "repeatable": pair["repeatable"],
            "outer_stage_diagnostics": stage_rows,
            "pooled_10_forward_plus_10_reverse": {
                "diagnostic_only": True,
                "quorum": 8,
                "forward": f_pooled,
                "reverse_inverted": r_pooled,
                "cross": cross,
                "forward_leave_one_worker_out": _loo(forward),
                "reverse_leave_one_worker_out": _loo(reverse),
            },
        })
    mismatches = [row for row in pair_rows if not row["repeatable"]]
    return {
        "schema": SCHEMA,
        "status": "GT_FREE_DIAGNOSTICS_COMPLETE",
        "diagnostic_only": True,
        "must_not_change_preregistered_gate": True,
        "label_data_loaded": False,
        "pair_count": len(pair_rows),
        "outer_mismatch_count": len(mismatches),
        "outer_mismatch_pair_ids": [row["pair_id"] for row in mismatches],
        "failure_distribution": dict(sorted(failures.items())),
        "pairs": pair_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay_path = args.replay_receipt.resolve()
    replay, _audit = gate._validate_replay_and_workers(replay_path)
    batch_path = Path(replay["source_batch_receipt"]["path"]).resolve()
    manifest_path = Path(replay["manifest"]["path"]).resolve()
    _manifest, _batch, loaded = replay_runner._validate_batch(
        batch_path, manifest_path, replay["manifest"]["sha256"])
    output = diagnose(replay, loaded)
    output["bindings"] = {
        "replay": {"path": str(replay_path),
                   "sha256": dev.sha256_file(replay_path),
                   "evidence_sha256": replay["evidence_sha256"]},
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(CODE_ROOT)),
            "sha256": dev.sha256_file(Path(__file__).resolve()),
        },
    }
    output["evidence_sha256"] = pilot.stable_json_hash(output)
    pilot.atomic_create_json(args.out.resolve(), output)
    print(json.dumps({
        "status": output["status"],
        "outer_mismatch_count": output["outer_mismatch_count"],
        "outer_mismatch_pair_ids": output["outer_mismatch_pair_ids"],
        "failure_distribution": output["failure_distribution"],
        "evidence_sha256": output["evidence_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
