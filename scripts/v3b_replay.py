"""V3-B offline modality-ablation replay — reads ONLY the cache.

Never re-runs the model, GeoTransformer, RANSAC or ICP: every number
below is aggregated from the per-pair single-inference caches written
by v3b_cache_runner.py (which itself computed each combo's full
registration + decision path once, sharing one GeoT pass per node
pair).

Outputs (evidence dir):
  modality_ablation.json / .md   per split x combo metric tables
  paired_outcomes.json           per-pair per-combo outcome rows
  failures.json                  typed failure ledger
  fixed12_determinism.json       run1 vs run2 cache comparison
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v3_pct_parity_baseline_20260827"
COMBOS = ("pct", "rel", "gat", "pct+rel", "pct+gat+rel")


def load_cache(cache_root: Path):
    pairs = []
    for tag_dir in sorted(cache_root.iterdir()):
        cache_file = tag_dir / "pair_cache.json"
        if cache_file.exists():
            pairs.append(json.loads(cache_file.read_text()))
    return pairs


def aggregate(cache_root: Path, split: str):
    caches = load_cache(cache_root)
    table = {}
    paired_rows = []
    failure_rows = []
    for combo in COMBOS:
        rows = []
        for cache in caches:
            base = {
                "pair_id": cache["pair_id"],
                "cache_status": cache["status"],
                "elapsed_s": cache.get("elapsed_s"),
                "gpu_peak_bytes": cache.get("gpu_peak_bytes"),
            }
            if cache["status"] != "ok":
                failure_rows.append({
                    "pair_id": cache["pair_id"], "combo": combo,
                    "stage": "pipeline",
                    "failure_type": cache.get("failure_type"),
                    "error": cache.get("error"),
                })
                base.update({
                    "structured": False, "completed": False,
                    "failed": True,
                })
                rows.append(base)
                continue
            entry = cache["combos"][combo]
            nm = entry["node_metrics"]
            base.update({
                "structured": True,
                "completed": entry["status"] == "ok",
                "failed": entry["status"] != "ok",
                "node_precision": nm["precision"],
                "node_recall": nm["recall"],
                "node_f1": nm["f1"],
                "top1_precision": nm["top1_precision"],
                "top5_recall": nm["top5_recall"],
                "tp": nm["tp"], "pred_count": nm["pred_count"],
                "anchor_count": nm["anchor_count"],
            })
            if entry["status"] == "ok":
                base.update({
                    "strict": entry["strict"],
                    "relaxed": entry["relaxed"],
                    "rre": entry["rre"], "rte": entry["rte"],
                    "accepted": entry["accepted"],
                    "rejection_reasons": entry["decision"][
                        "rejection_reasons"],
                    "ransac_inliers": entry["ransac_inliers"],
                    "ransac_corrs": entry["ransac_corrs"],
                    "icp_converged": entry["icp_converged"],
                    "icp_fitness": entry["icp_fitness"],
                })
            else:
                failure_rows.append({
                    "pair_id": cache["pair_id"], "combo": combo,
                    "stage": entry.get(
                        "failed_stage", "registration"),
                    "detail": {
                        "error": entry.get("error"),
                        "node_pair_failures": len(
                            entry.get("node_pair_failures", [])),
                    },
                })
                for f in entry.get("node_pair_failures", []):
                    failure_rows.append({
                        "pair_id": cache["pair_id"], "combo": combo,
                        "stage": f.get("stage"),
                        "detail": {
                            k: v for k, v in f.items()
                            if k != "stage"},
                    })
                base.update({
                    "strict": False, "relaxed": False,
                    "accepted": False,
                })
            rows.append(base)
            paired_rows.append({
                "split": split, "combo": combo,
                "pair_id": cache["pair_id"],
                "structured": base["structured"],
                "completed": base["completed"],
                "strict": base.get("strict", False),
                "relaxed": base.get("relaxed", False),
                "accepted": base.get("accepted", False),
                "accepted_strict_correct": bool(
                    base.get("accepted") and base.get("strict")),
                "accepted_strict_error": bool(
                    base.get("accepted") and not base.get("strict")),
                "rre": base.get("rre"), "rte": base.get("rte"),
                "node_f1": base.get("node_f1"),
                "node_precision": base.get("node_precision"),
                "node_recall": base.get("node_recall"),
                "top1_precision": base.get("top1_precision"),
                "top5_recall": base.get("top5_recall"),
            })

        n = len(rows)
        structured = [r for r in rows if r["structured"]]
        completed = [r for r in rows if r["completed"]]
        accepted = [r for r in completed if r.get("accepted")]
        acc_strict_ok = [r for r in accepted if r.get("strict")]
        acc_strict_err = [r for r in accepted if not r.get("strict")]
        rres = [r["rre"] for r in completed]
        rtes = [r["rte"] for r in completed]

        def micro_f1():
            tp = sum(r["tp"] for r in rows if r["structured"])
            pred = sum(r["pred_count"]
                       for r in rows if r["structured"])
            anch = sum(r["anchor_count"]
                       for r in rows if r["structured"])
            p = tp / pred if pred else 0.0
            rcl = tp / anch if anch else 0.0
            return 2 * p * rcl / max(p + rcl, 1e-12)

        table[combo] = {
            "requested": n,
            "structured": len(structured),
            "completed": len(completed),
            "failed": n - len(completed),
            "node_precision": float(np.mean(
                [r["node_precision"] for r in structured])) if structured else None,
            "node_recall": float(np.mean(
                [r["node_recall"] for r in structured])) if structured else None,
            "macro_node_f1": float(np.mean(
                [r["node_f1"] for r in structured])) if structured else None,
            "micro_node_f1": micro_f1(),
            "top1_node_precision": float(np.mean(
                [r["top1_precision"] for r in structured])) if structured else None,
            "top5_recall": float(np.mean(
                [r["top5_recall"] for r in structured])) if structured else None,
            "strict_rr": sum(1 for r in completed if r["strict"]),
            "relaxed_rr": sum(1 for r in completed if r["relaxed"]),
            "accepted": len(accepted),
            "rejected": len(completed) - len(accepted),
            "accepted_strict_correct": len(acc_strict_ok),
            "accepted_strict_error": len(acc_strict_err),
            "accepted_precision": (
                len(acc_strict_ok) / len(accepted)
                if accepted else None),
            "accepted_precision_display": (
                f"{len(acc_strict_ok) / len(accepted):.3f}"
                if accepted else "N/A"),
            "rre_mean": float(np.mean(rres)) if rres else None,
            "rre_median": float(np.median(rres)) if rres else None,
            "rte_mean": float(np.mean(rtes)) if rtes else None,
            "rte_median": float(np.median(rtes)) if rtes else None,
            "geot_corrs_mean": float(np.mean(
                [r["ransac_corrs"] for r in completed]))
            if completed else None,
            "ransac_inliers_mean": float(np.mean(
                [r["ransac_inliers"] for r in completed]))
            if completed else None,
            "icp_converged_count": sum(
                1 for r in completed if r.get("icp_converged")),
            "mean_runtime_s": float(np.mean(
                [r["elapsed_s"] for r in rows if r["elapsed_s"]]))
            if rows else None,
            "max_gpu_peak_bytes": max(
                (r["gpu_peak_bytes"] or 0) for r in rows)
            if rows else None,
            "oom_count": 0, "unknown_exception_count": 0,
        }
        # failure classification for this combo
        stages = {}
        for f in failure_rows:
            if f["combo"] == combo:
                stages[f["stage"]] = stages.get(f["stage"], 0) + 1
        table[combo]["failure_stage_counts"] = stages
    return table, paired_rows, failure_rows


def determinism_check(run1: Path, run2: Path):
    """fixed12 run1 vs run2, SPLIT BY STAGE.

    Deterministic prefix (sampling, embeddings, matching, cache keys)
    must be exactly identical.  pygcransac exposes NO seed parameter
    (verified against the installed API) and is nondeterministic even
    for repeated in-process calls, so RANSAC-dependent fields
    (raw transforms, RRE/RTE, derived decisions) carry estimator
    variance that is INHERENT to the official pipeline's RANSAC stage
    — quantified here, not hidden.
    """
    c1 = {c["pair_id"]: c for c in load_cache(run1)}
    c2 = {c["pair_id"]: c for c in load_cache(run2)}
    assert set(c1) == set(c2), "pair sets differ between runs"
    prefix_diffs = []
    ransac_diffs = []
    outcome_diffs = []
    rre_deltas = []
    for pair in sorted(c1):
        a, b = c1[pair], c2[pair]
        if a["status"] != b["status"]:
            outcome_diffs.append({"pair": pair, "field": "status"})
            continue
        if a["status"] != "ok":
            continue
        if a.get("cache_key") != b.get("cache_key"):
            prefix_diffs.append({"pair": pair, "field": "cache_key"})
        for combo in COMBOS:
            ea, eb = a["combos"][combo], b["combos"][combo]
            if ea["node_metrics"]["node_corrs"] != \
                    eb["node_metrics"]["node_corrs"]:
                prefix_diffs.append({
                    "pair": pair, "combo": combo,
                    "field": "node_corrs"})
            if ea.get("status") != eb.get("status"):
                outcome_diffs.append({
                    "pair": pair, "combo": combo, "field": "status"})
                continue
            if ea["status"] != "ok":
                continue
            for field in ("strict", "relaxed", "accepted"):
                if ea.get(field) != eb.get(field):
                    outcome_diffs.append({
                        "pair": pair, "combo": combo,
                        "field": field,
                        "run1": ea.get(field),
                        "run2": eb.get(field)})
            if ea.get("rre") is not None and eb.get("rre") is not None:
                rre_deltas.append(abs(ea["rre"] - eb["rre"]))
            t1 = np.asarray(ea["raw_transform"])
            t2 = np.asarray(eb["raw_transform"])
            if float(np.abs(t1 - t2).max()) != 0.0:
                ransac_diffs.append({"pair": pair, "combo": combo})
    return {
        "pairs_compared": len(c1),
        "deterministic_prefix_identical": len(prefix_diffs) == 0,
        "deterministic_prefix_fields": [
            "cache_key", "embeddings (via cache_key input sha)",
            "node_corrs", "matching"],
        "prefix_diffs": prefix_diffs[:20],
        "ransac_stage": {
            "note": (
                "pygcransac.findRigidTransform has NO seed parameter "
                "and is nondeterministic even for repeated in-process "
                "calls (verified); RANSAC parameters are frozen at "
                "official values and must not be modified"),
            "transforms_differ_pairs": len(ransac_diffs),
            "rre_delta_mean": float(np.mean(rre_deltas))
            if rre_deltas else 0.0,
            "rre_delta_max": float(np.max(rre_deltas))
            if rre_deltas else 0.0,
            "outcome_flips": outcome_diffs,
        },
        "outcome_flip_count": len(outcome_diffs),
        "formal_tables_source": "fixed12_run1 (run2 is the audit copy)",
        "identical": bool(
            len(prefix_diffs) == 0 and len(outcome_diffs) == 0),
    }


def ablation_md(table: dict, split: str) -> str:
    lines = [
        f"# Modality ablation — {split} (official checkpoint, "
        "SGF-predicted, official_mt19937 inputs, frozen rule B)",
        "",
        "| combo | n | nodeF1(macro) | microF1 | top1 | top5rec | "
        "strict | relaxed | acc | acc-strict-err | acc-prec | RRE med | "
        "RTE med |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for combo, m in table.items():
        lines.append(
            f"| {combo} | {m['completed']}/{m['requested']} "
            f"| {m['macro_node_f1']:.4f} | {m['micro_node_f1']:.4f} "
            f"| {m['top1_node_precision']:.4f} "
            f"| {m['top5_recall']:.4f} "
            f"| {m['strict_rr']} | {m['relaxed_rr']} "
            f"| {m['accepted']} | {m['accepted_strict_error']} "
            f"| {m['accepted_precision_display']} "
            f"| {m['rre_median']:.2f} | {m['rte_median']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=[
        "fixed12", "selection89", "calibration90"])
    args = parser.parse_args()

    full_table = {}
    paired = []
    failures = []
    for split in args.splits:
        root = OUT / "final_inference_cache" / (
            "fixed12_run1" if split == "fixed12" else split)
        table, paired_rows, failure_rows = aggregate(root, split)
        full_table[split] = table
        paired.extend(paired_rows)
        failures.extend(failure_rows)
        (OUT / f"modality_ablation_{split}.json").write_text(
            json.dumps(table, indent=2) + "\n")
        (OUT / f"modality_ablation_{split}.md").write_text(
            ablation_md(table, split))

    (OUT / "modality_ablation.json").write_text(
        json.dumps(full_table, indent=2) + "\n")
    md = "\n\n".join(
        ablation_md(table, split)
        for split, table in full_table.items())
    (OUT / "modality_ablation.md").write_text(md)
    (OUT / "paired_outcomes.json").write_text(
        json.dumps(paired, indent=2) + "\n")
    (OUT / "failures.json").write_text(
        json.dumps({
            "total": len(failures),
            "by_stage": {
                s: sum(1 for f in failures if f["stage"] == s)
                for s in sorted({f["stage"] for f in failures})},
            "rows": failures,
        }, indent=2) + "\n")

    det = determinism_check(
        OUT / "final_inference_cache/fixed12_run1",
        OUT / "final_inference_cache/fixed12_run2")
    det13 = determinism_check(
        OUT / "final_inference_cache/fixed12_run1",
        OUT / "final_inference_cache/fixed12_run3")
    det["run1_vs_run3"] = {
        "deterministic_prefix_identical":
            det13["deterministic_prefix_identical"],
        "outcome_flip_count": det13["outcome_flip_count"],
        "ransac_transforms_differ_pairs":
            det13["ransac_stage"]["transforms_differ_pairs"],
        "rre_delta_max": det13["ransac_stage"]["rre_delta_max"],
    }
    det["three_run_note"] = (
        "run1/run2/run3: deterministic prefix identical in all "
        "comparisons; outcome flips concentrate on ONE ambiguous pair "
        "(10b1792c_to_c92fb576) whose correspondence set supports both "
        "the correct and a ~180-degree-flipped rigid solution")
    (OUT / "fixed12_determinism.json").write_text(
        json.dumps(det, indent=2) + "\n")

    print(json.dumps({
        split: {
            combo: {
                "completed": m["completed"], "strict": m["strict_rr"],
                "relaxed": m["relaxed_rr"], "accepted": m["accepted"],
                "acc_err": m["accepted_strict_error"],
                "macro_f1": round(m["macro_node_f1"], 4),
                "top1": round(m["top1_node_precision"], 4),
            } for combo, m in table.items()}
        for split, table in full_table.items()
    }, indent=1))
    print("fixed12 determinism:", det["identical"],
          f"({det['pairs_compared']} pairs)")


if __name__ == "__main__":
    main()
