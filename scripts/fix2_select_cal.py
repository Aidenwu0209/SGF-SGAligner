"""Fix-2 rule selection (selection split) + freeze (calibration split).

Evaluates candidates A/B/C on selection, picks by: 0 accepted-strict-
errors first, then max accepted.  Freezes on calibration (thresholds
already round constants; calibration validates only).  GT only produces
strict/relaxed labels for offline precision — never inference features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from inference import run_pair  # noqa: E402

from adapters.sgf.data_sources import (  # noqa: E402
    OracleGraphSource, PredictedGraphSource,
)
from adapters.sgf.object_adapter import adapt_objects  # noqa: E402


def load_split(split: str) -> list[str]:
    man = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    return [
        line.strip()
        for line in (man / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]


def process(job):
    pair_id, mode, rule, out_root = job
    out_dir = Path(out_root) / f"{mode}_{rule}_{pair_id[:8]}_{pair_id[-4:]}"
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    return run_pair(pair_id, mode, out_dir, device="cuda",
                    decision_rule=rule)


def evaluate_arm(rows, anchors_label="strict"):
    ok = [r for r in rows if r["status"] == "ok"]
    strict = sum(1 for r in ok if r.get("strict"))
    relaxed = sum(1 for r in ok if r.get("relaxed"))
    accepted = sum(1 for r in ok if r.get("accepted"))
    acc_err = sum(
        1 for r in ok if r.get("accepted") and not r.get("strict")
    )
    return {
        "n": len(rows), "ok": len(ok), "strict": strict,
        "relaxed": relaxed, "accepted": accepted,
        "accepted_strict_errors": acc_err,
        "accepted_relaxed_errors": sum(
            1 for r in ok if r.get("accepted") and not r.get("relaxed")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("selection", "calibration"),
                        required=True)
    parser.add_argument("--mode", default="official_sgf_predicted")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = load_split(args.split)
    if args.limit:
        pairs = pairs[: args.limit]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "mode": args.mode, "rules": {}}
    for rule in ("A", "B", "C"):
        jobs = [
            (p, args.mode, rule, out_root) for p in pairs
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(process, jobs):
                rows.append(row)
        summary["rules"][rule] = evaluate_arm(rows)
        print(rule, json.dumps(summary["rules"][rule]), flush=True)
    (out_root / f"rule_selection_{args.split}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
