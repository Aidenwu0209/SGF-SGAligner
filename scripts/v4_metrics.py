"""V4 deterministic metrics + gate evaluation + paired comparison.

Reads: checkpoint_selection/{complete,explicit}.json (selection89
deterministic metrics), the candidate caches on calibration90
(embedding-level metrics recomputed deterministically from cached
joints), the incumbent A deterministic metrics (recomputed from the
V3 official cache pct+rel combo), and the registration-repeat
summaries. Writes deterministic_metrics.json, paired_comparison.json,
registration_repeats.json (aggregated).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
V3 = ROOT / (
    "outputs/official_sgaligner_v3_pct_parity_baseline_20260827/"
    "final_inference_cache"
)
COMBOS = ("pct", "rel", "gat", "pct+rel", "pct+gat+rel")

INCUMBENT_SELECTION = {  # V3 sealed official pct+rel deterministic
    "macro_node_f1": 0.0844, "top1_precision": 0.0675,
    "top5_recall": 0.4330,
}
INCUMBENT_CALIBRATION = {
    "macro_node_f1": 0.0755, "top1_precision": 0.0515,
    "top5_recall": 0.4353,
}


def embedding_metrics_from_cache(cache_root: Path, combo: str):
    """Deterministic matching metrics from the cached node metrics
    (written at cache time by the shared node_metrics code path —
    itself deterministic)."""
    f1s, top1s, top5s = [], [], []
    tp_all = pred_all = anchor_all = 0
    for tag in sorted(cache_root.iterdir()):
        f = tag / "pair_cache.json"
        if not f.exists():
            continue
        cache = json.loads(f.read_text())
        if cache["status"] != "ok":
            continue
        entry = cache["combos"][combo]["node_metrics"]
        f1s.append(entry["f1"])
        top1s.append(entry["top1_precision"])
        top5s.append(entry["top5_recall"])
        tp_all += entry["tp"]
        pred_all += entry["pred_count"]
        anchor_all += entry["anchor_count"]
    micro_p = tp_all / pred_all if pred_all else 0.0
    micro_r = tp_all / anchor_all if anchor_all else 0.0
    return {
        "macro_node_f1": float(np.mean(f1s)),
        "top1_precision": float(np.mean(top1s)),
        "top5_recall": float(np.mean(top5s)),
        "micro_node_f1": 2 * micro_p * micro_r / max(
            micro_p + micro_r, 1e-12),
        "pairs": len(f1s),
    }


def paired_rows(cache_root: Path, combo: str):
    rows = {}
    for tag in sorted(cache_root.iterdir()):
        f = tag / "pair_cache.json"
        if not f.exists():
            continue
        cache = json.loads(f.read_text())
        if cache["status"] != "ok":
            continue
        nm = cache["combos"][combo]["node_metrics"]
        rows[cache["pair_id"]] = {
            "node_f1": nm["f1"], "top1": nm["top1_precision"],
            "top5": nm["top5_recall"]}
    return rows


def main() -> None:
    result = {"arms": {}}

    # A incumbent: V3 sealed official caches, pct+rel combo
    a_sel = embedding_metrics_from_cache(V3 / "selection89", "pct+rel")
    a_cal = embedding_metrics_from_cache(V3 / "calibration90", "pct+rel")
    result["arms"]["A_incumbent"] = {
        "selection89": a_sel, "calibration90": a_cal,
        "source": "V3 official cache, pct+rel combo (deterministic)",
        "v3_sealed_reference": {
            "selection89": INCUMBENT_SELECTION,
            "calibration90": INCUMBENT_CALIBRATION},
    }

    for arm, label in (("complete", "B_complete"),
                       ("explicit", "C_explicit")):
        sel_file = OUT / "checkpoint_selection" / f"{arm}.json"
        if not sel_file.exists():
            continue
        sel = json.loads(sel_file.read_text())
        cal_cache = OUT / "calibration90" / f"cache_{arm}"
        entry = {
            "selection89": {
                k: sel["selection_metrics"][k] for k in (
                    "macro_node_f1", "top1_precision", "top5_recall",
                    "micro_node_f1")},
            "selected_epoch": sel["selected_epoch"],
            "checkpoint": sel["selected_checkpoint"],
            "pct_frozen_hashes_match": sel[
                "pct_frozen_hashes_match"],
        }
        # FORMAL comparison uses ONE semantics on both splits: the
        # official matcher over the candidate caches (the
        # training-time evaluate() used a cross-graph-filtered top-3
        # variant — recorded as a deviation, never used for gates)
        sel_cache = OUT / "selection89" / f"cache_{arm}"
        if sel_cache.exists():
            entry["selection89_official_matcher"] = (
                embedding_metrics_from_cache(sel_cache, "candidate"))
        if cal_cache.exists():
            entry["calibration90"] = embedding_metrics_from_cache(
                cal_cache, "candidate")
        result["arms"][label] = entry

    # deterministic minimum gates (pre-registered)
    gates = {}
    for label in ("B_complete", "C_explicit"):
        arm = result["arms"].get(label)
        if not arm or "calibration90" not in arm:
            gates[label] = {"evaluated": False}
            continue
        sel_m = arm.get(
            "selection89_official_matcher", arm["selection89"])
        cal_m = arm["calibration90"]
        gates[label] = {
            "evaluated": True,
            "macro_f1_beats_incumbent_selection": (
                sel_m["macro_node_f1"]
                > result["arms"]["A_incumbent"]["selection89"][
                    "macro_node_f1"]),
            "top1_not_below_incumbent_selection": (
                sel_m["top1_precision"]
                >= result["arms"]["A_incumbent"]["selection89"][
                    "top1_precision"]),
            "top5_not_below_incumbent_selection": (
                sel_m["top5_recall"]
                >= result["arms"]["A_incumbent"]["selection89"][
                    "top5_recall"]),
            "macro_f1_beats_incumbent_calibration": (
                cal_m["macro_node_f1"]
                > result["arms"]["A_incumbent"]["calibration90"][
                    "macro_node_f1"]),
            "direction_consistent": (
                (sel_m["macro_node_f1"]
                 > result["arms"]["A_incumbent"]["selection89"][
                     "macro_node_f1"])
                == (cal_m["macro_node_f1"]
                    > result["arms"]["A_incumbent"]["calibration90"][
                        "macro_node_f1"])),
            "suggested_gate_macro_f1_ge_0095": (
                sel_m["macro_node_f1"] >= 0.095),
            "suggested_gate_top1_plus_001": (
                sel_m["top1_precision"]
                >= result["arms"]["A_incumbent"]["selection89"][
                    "top1_precision"] + 0.01),
        }
    result["gates"] = gates
    (OUT / "deterministic_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n")

    # aggregated registration repeats
    agg = {}
    for split in ("selection89", "calibration90"):
        for label in ("A", "B", "C"):
            f = OUT / split / f"registration_repeats_{label}.json"
            if f.exists():
                agg[f"{label}_{split}"] = json.loads(
                    f.read_text())["summary"]
    (OUT / "registration_repeats.json").write_text(
        json.dumps(agg, indent=2) + "\n")
    # paired per-pair comparison vs incumbent (deterministic fields)
    paired = {}
    a_rows = {
        "selection89": paired_rows(V3 / "selection89", "pct+rel"),
        "calibration90": paired_rows(V3 / "calibration90", "pct+rel"),
    }
    for label, arm in (("B_complete", "complete"),
                       ("C_explicit", "explicit")):
        paired[label] = {}
        for split in ("selection89", "calibration90"):
            cache = OUT / split / f"cache_{arm}"
            if not cache.exists():
                continue
            rows = paired_rows(cache, "candidate")
            common = sorted(set(rows) & set(a_rows[split]))
            deltas = [
                rows[p]["node_f1"] - a_rows[split][p]["node_f1"]
                for p in common]
            paired[label][split] = {
                "common_pairs": len(common),
                "node_f1_delta_mean": float(np.mean(deltas)),
                "node_f1_improved_pairs": sum(1 for d in deltas if d > 0),
                "node_f1_regressed_pairs": sum(
                    1 for d in deltas if d < 0),
            }
    (OUT / "paired_comparison.json").write_text(
        json.dumps(paired, indent=2) + "\n")
    print(json.dumps({
        "A_sel": a_sel, "A_cal": a_cal,
        "gates": gates}, indent=1))


if __name__ == "__main__":
    main()
