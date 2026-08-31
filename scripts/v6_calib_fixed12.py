"""V6 calibration + fixed12 for the frozen winner (B_ep20).

Calibration: single run, no reselection, no threshold changes;
accepted-strict-error must be 0; report paired improved/regressed/
flat vs the V5 B_ep10 baseline.

fixed12: single run (3 repeats per pair for RANSAC variance),
full 12 denominator; gates: strict>=4, correct accepted>=4,
accepted error=0.
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
for p in (str(ROOT), str(ROOT / "src"),
          str(ROOT / "src/inference/sgf_official"),
          str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from v6_registration import (  # noqa: E402
    registration_with_consistency, node_eval,
)
from canonical_inputs import build_canonical_pair  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402

OUT = ROOT / (
    "outputs/official_sgaligner_v6_sgf_domain_matcher_20260829")
BASELINE = (
    ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828/"
    "training/B/epoch_00010.pt")


def build_split(pairs):
    samples = []
    for pair_id in pairs:
        dd, _ = build_canonical_pair(pair_id, with_labels=False)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        objects = dd["registration_pts"]
        n = dd["tot_obj_pts"].shape[0]
        src_count = dd["src_count"]
        samples.append((
            pair_id, dd, anchor_idx,
            ({i: objects[i].mean(axis=0)
              for i in range(src_count)},
             {j - src_count: objects[j].mean(axis=0)
              for j in range(src_count, n)})))
    return samples


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranking = json.loads(
        (OUT / "checkpoint_ranking.json").read_text())
    winner = next(
        c for c in ranking["candidates"]
        if c["selection_rank"] == 1)
    winner_ck = ROOT / next(
        r["checkpoint"] for arm_rows in json.loads(
            (OUT / "selection_node_metrics.json").read_text()
        )["arms"].values() for r in arm_rows
        if r["arm"] == winner["arm"]
        and r["epoch"] == winner["epoch"])

    # ---------------- calibration (single run) ----------------------
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/calibration.txt")
    cal_pairs = [l.strip() for l in pl.read_text().splitlines()
                 if l.strip()]
    cal_samples = build_split(cal_pairs)

    # node metrics for winner + baseline on calibration
    w_agg, w_pp = node_eval(winner_ck, cal_samples, device)
    b_agg, b_pp = node_eval(BASELINE, cal_samples, device)
    # registration
    w_counts, w_rows = registration_with_consistency(
        winner_ck, cal_samples, device)
    b_counts, b_rows = registration_with_consistency(
        BASELINE, cal_samples, device)
    w_counts["checkpoint_sha256"] = winner["counts"][
        "checkpoint_sha256"]

    # paired comparison
    b_by = {r["pair_id"]: r for r in b_rows}
    improved = regressed = flat = 0
    for r in w_rows:
        b = b_by[r["pair_id"]]
        if r.get("strict") and not b.get("strict"):
            improved += 1
        elif b.get("strict") and not r.get("strict"):
            regressed += 1
        else:
            flat += 1
    calibration = {
        "winner": {
            "label": winner["label"],
            "checkpoint": str(winner_ck.relative_to(ROOT)),
            "checkpoint_sha256": winner["counts"][
                "checkpoint_sha256"],
            "epoch": winner["epoch"]},
        "winner_node": w_agg,
        "baseline_node": b_agg,
        "winner_registration": w_counts,
        "baseline_registration": b_counts,
        "paired_vs_baseline": {
            "improved": improved, "regressed": regressed,
            "flat": flat},
        "gate": {
            "accepted_strict_error_zero":
                w_counts["accepted_strict_error"] == 0,
            "no_obvious_reverse_regression": (
                w_counts["raw_strict"]
                >= b_counts["raw_strict"] - 1),
        },
    }
    calibration["gate"]["all_pass"] = all(
        calibration["gate"].values())
    (OUT / "calibration" ).mkdir(exist_ok=True)
    (OUT / "calibration" / "calibration90.json").write_text(
        json.dumps(calibration, indent=2) + "\n")
    print("calibration:", json.dumps(
        {"winner_reg": {k: w_counts[k] for k in (
            "raw_strict", "raw_relaxed", "accepted",
            "accepted_strict_correct",
            "accepted_strict_error")},
         "baseline_reg": {k: b_counts[k] for k in (
             "raw_strict", "raw_relaxed", "accepted",
             "accepted_strict_correct",
             "accepted_strict_error")},
         "paired": calibration["paired_vs_baseline"],
         "gate": calibration["gate"]}, indent=1))

    if not calibration["gate"]["all_pass"]:
        print("CALIBRATION GATE FAILED — stopping before fixed12")
        return

    # ---------------- fixed12 (3 repeats per pair) -------------------
    smoke = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/phase6_registration_aware_closure/"
        "smoke12/native")
    fixed_pairs = sorted(
        d.name for d in smoke.iterdir()
        if d.is_dir() and "_to_" in d.name)
    fixed_samples = build_split(fixed_pairs)
    REPEATS = 3
    summary = {
        "runs_per_pair": REPEATS,
        "distinct_pairs_strict_correct": 0,
        "distinct_pairs_accepted_correct": 0,
        "outcomes": [],
    }
    counts_acc = {"accepted_strict_correct": 0,
                  "accepted_strict_error": 0, "rejected": 0,
                  "failed": 0, "zero_candidate": 0,
                  "raw_strict": 0, "raw_relaxed": 0}
    strict_pairs = set()
    acc_pairs = set()
    for r in range(REPEATS):
        counts, rows = registration_with_consistency(
            winner_ck, fixed_samples, device)
        for row in rows:
            if row.get("outcome") == \
                    "accepted_strict_correct":
                counts_acc["accepted_strict_correct"] += 1
                acc_pairs.add(row["pair_id"])
            elif row.get("outcome") == \
                    "accepted_strict_error":
                counts_acc["accepted_strict_error"] += 1
            elif row.get("outcome") == "rejected":
                counts_acc["rejected"] += 1
            elif row.get("outcome") in (
                    "failed", "zero_candidate"):
                counts_acc[row["outcome"]] += 1
            if row.get("strict"):
                counts_acc["raw_strict"] += 1
                strict_pairs.add(row["pair_id"])
            if row.get("relaxed"):
                counts_acc["raw_relaxed"] += 1
        summary["outcomes"].append({"repeat": r, "counts": counts})
    summary["distinct_pairs_strict_correct"] = len(strict_pairs)
    summary["distinct_pairs_accepted_correct"] = len(acc_pairs)
    summary["aggregate"] = counts_acc
    summary["gates"] = {
        "strict_ge_4_distinct": len(strict_pairs) >= 4,
        "accepted_correct_ge_4_distinct": len(acc_pairs) >= 4,
        "accepted_strict_error_zero":
            counts_acc["accepted_strict_error"] == 0,
    }
    summary["gates"]["all_pass"] = all(
        summary["gates"].values())
    (OUT / "fixed12").mkdir(exist_ok=True)
    (OUT / "fixed12" / "fixed12.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("fixed12:", json.dumps(summary["aggregate"], indent=1))
    print("gates:", json.dumps(summary["gates"]))


if __name__ == "__main__":
    main()
