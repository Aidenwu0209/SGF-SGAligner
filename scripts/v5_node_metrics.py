"""V5 Part 8 (node metrics): ALL checkpoints of arms B/C on
selection89 with canonical inputs + the REAL official matcher +
frozen macro/micro semantics; adds candidate precision/recall and
split/merge recall; deterministic double-run; arm fingerprints.

Outputs: selection_node_metrics.json (full), per-arm top-3 by the
pre-registered node key (macro_node_f1 -> macro_top1 -> macro_top5
-> margin -> epoch) for the registration stage.
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import subprocess  # noqa: E402
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

from canonical_inputs import build_canonical_pair  # noqa: E402
from v4_train import batch_for  # noqa: E402
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from v4seal_metrics import (  # noqa: E402
    per_pair_node_metrics, aggregate,
)

OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_merge_recall(pred, anchors):
    """Recall restricted to split (src with >=2 anchors) and merge
    (ref with >=2 anchors) anchor subsets."""
    from collections import Counter

    src_count = Counter(a for a, _ in anchors)
    ref_count = Counter(b for _, b in anchors)
    split_set = {
        (a, b) for a, b in anchors if src_count[a] >= 2}
    merge_set = {
        (a, b) for a, b in anchors if ref_count[b] >= 2}
    out = {}
    for name, s in (("split", split_set), ("merge", merge_set)):
        out[f"{name}_anchors"] = len(s)
        out[f"{name}_recall"] = (
            len(pred & s) / len(s) if s else None)
    return out


def evaluate(ckpt_path, samples, device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    per_pair = []
    tp = pred_n = anch_n = 0
    for pair_id, dd, anchor_idx in samples:
        with torch.no_grad():
            emb = model(
                batch_for(dd, "explicit", device))["joint"
            ].cpu().numpy().astype(np.float32)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count, anchor_idx, sim=sim)
        pp["pair_id"] = pair_id
        pred = set(node_corrs)
        pp.update(split_merge_recall(pred, anchor_idx))
        per_pair.append(pp)
        tp += pp["tp"]
        pred_n += pp["pred_count"]
        anch_n += pp["anchor_count"]
    agg = aggregate([
        {"tp": p["tp"], "pred_count": p["pred_count"],
         "anchor_count": p["anchor_count"], "f1": p["f1"],
         "top1_hit": p["top1_hit"], "top1_total": p["top1_total"],
         "top5_hits": p["top5_hits"], "margin": p["margin"]}
        for p in per_pair])
    agg["candidate_precision"] = (
        tp / pred_n if pred_n else 0.0)
    agg["candidate_recall"] = tp / anch_n if anch_n else 0.0
    agg["split_recall_macro"] = float(np.mean([
        p["split_recall"] for p in per_pair
        if p["split_recall"] is not None])) if any(
        p["split_recall"] is not None for p in per_pair) else None
    agg["merge_recall_macro"] = float(np.mean([
        p["merge_recall"] for p in per_pair
        if p["merge_recall"] is not None])) if any(
        p["merge_recall"] is not None for p in per_pair) else None
    return agg, per_pair


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/selection.txt")
    pairs = [l.strip() for l in pl.read_text().splitlines()
             if l.strip()]
    print("building canonical selection89 inputs...", flush=True)
    samples = []
    for pair_id in pairs:
        dd, _ = build_canonical_pair(pair_id, with_labels=False)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        samples.append((pair_id, dd, anchor_idx))

    results = {"code_head": code_head,
               "matcher": "inference.official_matching (verbatim)",
               "inputs": "canonical production builder",
               "arms": {}}
    determinism = []
    for arm, epochs in (("B", range(5, 61, 5)),
                        ("C", range(5, 51, 5))):
        rows = []
        for epoch in epochs:
            ckpt = (OUT / "training" / arm /
                    f"epoch_{epoch:05d}.pt")
            if not ckpt.exists():
                continue
            agg1, pp1 = evaluate(ckpt, samples, device)
            agg2, pp2 = evaluate(ckpt, samples, device)
            identical = (
                agg1 == agg2
                and [p["f1"] for p in pp1] == [p["f1"] for p in pp2]
                and [p["pred_count"] for p in pp1]
                == [p["pred_count"] for p in pp2])
            determinism.append({
                "arm": arm, "epoch": epoch,
                "identical": identical})
            if not identical:
                raise RuntimeError(
                    f"FAIL-CLOSED nondeterministic {arm} ep{epoch}")
            rows.append({
                "arm": arm, "epoch": epoch,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(ckpt),
                "metrics": agg1,
            })
            print(f"{arm} ep{epoch}: macroF1 "
                  f"{agg1['macro_node_f1']:.4f} top1 "
                  f"{agg1['macro_top1']:.4f} top5 "
                  f"{agg1['macro_top5']:.4f} zero-cand "
                  f"{agg1['zero_candidate_pairs']}", flush=True)
        ranked = sorted(
            rows,
            key=lambda r: (-r["metrics"]["macro_node_f1"],
                           -r["metrics"]["macro_top1"],
                           -r["metrics"]["macro_top5"],
                           -r["metrics"]["margin"], r["epoch"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        results["arms"][arm] = ranked
    (OUT / "selection_node_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n")
    (OUT / "determinism_node_metrics.json").write_text(
        json.dumps({
            "all_identical": all(d["identical"]
                                 for d in determinism),
            "checks": determinism}, indent=2) + "\n")
    for arm in ("B", "C"):
        top3 = [r for r in results["arms"][arm] if r["rank"] <= 3]
        print(f"arm {arm} top3:",
              [(r["epoch"], round(r["metrics"]["macro_node_f1"], 4))
               for r in top3])


if __name__ == "__main__":
    main()
