"""Frozen metric semantics (V4-Fix-Seal Part 6).

macro vs micro are BOTH computed and STRICTLY separated:

  per-pair quantities (official matcher node_corrs = P, anchors = A):
    tp = |P ∩ A|; p = tp/|P| (0 if |P|=0); r = tp/|A| (0 if |A|=0)
    f1 = 2pr/(p+r) (0 if p+r=0)
    top1_pair = hits_1/queries_1 ; top5_pair = hits_5/|A|
  macro_X = mean over pairs of the per-pair X
  micro:
    micro_f1 from pooled tp/|P|/|A|
    micro_top1 = total hits_1 / total queries_1
    micro_top5 = total hits_5 / total anchors
  margin = macro mean of (pos_sim_mean − neg_sim_mean) over each
  pair's top-5 cross-graph candidates
  zero_candidate_pairs = #{pairs : |P| = 0}

Ranking & historical gates use ONLY: macro_node_f1 → macro_top1 →
macro_top5 → margin → earlier epoch.  micro values are reported
alongside and never substituted into the macro fields.
"""
from __future__ import annotations

import numpy as np


def per_pair_node_metrics(node_corrs, rank_list, src_count,
                          anchor_idx, sim=None):
    """Per-pair quantities under the official matcher output."""
    pred = set(node_corrs)
    tp = len(pred & anchor_idx)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(anchor_idx) if anchor_idx else 0.0
    f1 = 2 * p * r / max(p + r, 1e-12)
    top1_hit = top1_total = top5_hits = 0
    pos_sims, neg_sims = [], []
    for i in range(src_count):
        refs = [x for x in rank_list[i] if x >= src_count][:5]
        if not refs:
            continue
        top1_total += 1
        if (i, int(refs[0])) in anchor_idx:
            top1_hit += 1
        for x in refs[:5]:
            if (i, int(x)) in anchor_idx:
                top5_hits += 1
                if sim is not None:
                    pos_sims.append(float(sim[i, int(x)]))
            elif sim is not None:
                neg_sims.append(float(sim[i, int(x)]))
    return {
        "tp": tp, "pred_count": len(pred),
        "anchor_count": len(anchor_idx),
        "precision": p, "recall": r, "f1": f1,
        "top1_hit": top1_hit, "top1_total": top1_total,
        "top5_hits": top5_hits,
        "margin": (
            float(np.mean(pos_sims) - np.mean(neg_sims))
            if pos_sims and neg_sims else None),
    }


def aggregate(per_pair):
    """macro + micro from per-pair rows (single definition)."""
    n = len(per_pair)
    tp = sum(x["tp"] for x in per_pair)
    pred = sum(x["pred_count"] for x in per_pair)
    anch = sum(x["anchor_count"] for x in per_pair)
    micro_p = tp / pred if pred else 0.0
    micro_r = tp / anch if anch else 0.0
    top1_h = sum(x["top1_hit"] for x in per_pair)
    top1_t = sum(x["top1_total"] for x in per_pair)
    top5_h = sum(x["top5_hits"] for x in per_pair)
    margins = [x["margin"] for x in per_pair
               if x["margin"] is not None]
    return {
        "macro_node_f1": float(np.mean([x["f1"] for x in per_pair])),
        "micro_node_f1": 2 * micro_p * micro_r / max(
            micro_p + micro_r, 1e-12),
        "macro_top1": float(np.mean([
            x["top1_hit"] / x["top1_total"]
            if x["top1_total"] else 0.0 for x in per_pair])),
        "micro_top1": top1_h / top1_t if top1_t else 0.0,
        "macro_top5": float(np.mean([
            x["top5_hits"] / x["anchor_count"]
            if x["anchor_count"] else 0.0 for x in per_pair])),
        "micro_top5": top5_h / anch if anch else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "zero_candidate_pairs": sum(
            1 for x in per_pair if x["pred_count"] == 0),
        "pairs": n,
        "pooled": {"tp": tp, "pred": pred, "anchors": anch,
                   "top1_hits": top1_h, "top1_queries": top1_t,
                   "top5_hits": top5_h},
    }


RANKING_KEY = ("macro_node_f1", "macro_top1", "macro_top5",
               "margin", "epoch_asc")
