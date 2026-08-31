"""V4-Fix Part 3: ONE deterministic calibration90 evaluation per
frozen winner (after fair selection + determinism replay passed).

Same official-matcher metrics as the fair selection; paired comparison
against the incumbent, the OLD C winner (epoch 20 — same checkpoint,
old numbers from the V4 cache) and the OLD B winner (epoch 40, V4
cache numbers).  Calibration NEVER feeds back into checkpoint choice.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from v4_train import build_split_samples, batch_for  # noqa: E402
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import (  # noqa: E402
    PredictedGraphSource, load_anchor_ids,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

OLD = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
NEW = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"


def evaluate_on(ckpt_path, arm, samples, anchors_by_pair, device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    f1s, top1s, top5s = [], [], []
    tp_all = pred_all = anchor_all = 0
    total_cand = zero_cand = 0
    per_pair = []
    for pair_id, data_dict, _labels in samples:
        with torch.no_grad():
            batch = batch_for(data_dict, arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(np.float32)
        src_count = data_dict["src_count"]
        node_corrs, rank_list, _sim = official_matching(emb, src_count)
        anchors = anchors_by_pair[pair_id]
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + src_count)
            for s, r in anchors if s in src_map and r in ref_map}
        pred = set(node_corrs)
        tp = len(pred & anchor_idx)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(anchor_idx) if anchor_idx else 0.0
        f1s.append(2 * p * r / max(p + r, 1e-12))
        tp_all += tp
        pred_all += len(pred)
        anchor_all += len(anchor_idx)
        total_cand += len(pred)
        if not pred:
            zero_cand += 1
        top1_hit = top1_total = top5_hits = 0
        for i in range(src_count):
            refs = [x for x in rank_list[i] if x >= src_count][:5]
            if not refs:
                continue
            top1_total += 1
            if (i, int(refs[0])) in anchor_idx:
                top1_hit += 1
            top5_hits += sum(
                1 for x in refs[:5] if (i, int(x)) in anchor_idx)
        top1s.append((top1_hit, top1_total))
        top5s.append((top5_hits, len(anchor_idx)))
        per_pair.append({
            "pair_id": pair_id, "f1": f1s[-1], "candidates": len(pred)})
    micro_p = tp_all / pred_all if pred_all else 0.0
    micro_r = tp_all / anchor_all if anchor_all else 0.0
    return {
        "macro_node_f1": float(np.mean(f1s)),
        "micro_node_f1": 2 * micro_p * micro_r / max(
            micro_p + micro_r, 1e-12),
        "top1_precision": sum(a for a, _ in top1s) / max(
            sum(b for _, b in top1s), 1),
        "top5_recall": sum(a for a, _ in top5s) / max(anchor_all, 1),
        "total_valid_candidates": total_cand,
        "zero_candidate_pairs": zero_cand,
        "pairs": len(f1s),
    }, {pp["pair_id"]: pp["f1"] for pp in per_pair}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("building calibration90 tensors...", flush=True)
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    cal_pairs = [l.strip() for l in
                 (pairlists / "calibration.txt").read_text().splitlines()
                 if l.strip()]
    samples, _ = build_split_samples(
        cal_pairs, PredictedGraphSource())
    anchors_by_pair = {
        pid: set(load_anchor_ids(pid)) for pid, _d, _l in samples}

    winners = {}
    for label in ("B", "C"):
        sel = json.loads(
            (NEW / f"checkpoint_selection_corrected_{label}.json"
             ).read_text())
        arm = "complete" if label == "B" else "explicit"
        winners[label] = (arm, sel["winner_epoch"],
                          sel["winner_checkpoint"])

    out = {"note": "one deterministic calibration evaluation per "
                   "frozen winner; calibration does not influence "
                   "checkpoint selection"}
    for label, (arm, epoch, ckpt_rel) in winners.items():
        metrics, per_pair_f1 = evaluate_on(
            ROOT / ckpt_rel, arm, samples, anchors_by_pair, device)
        out[f"winner_{label}"] = {
            "arm": arm, "epoch": epoch, "checkpoint": ckpt_rel,
            "calibration90": metrics}
        (NEW / f"calibration_winner_{label}.json").write_text(
            json.dumps({
                "winner": label, "arm": arm, "epoch": epoch,
                "checkpoint": ckpt_rel,
                "calibration90": metrics,
                "per_pair_f1": per_pair_f1,
            }, indent=2) + "\n")
        print(label, arm, "ep", epoch, json.dumps(metrics))

    # incumbent + OLD winners from V4 caches (official semantics)
    def cache_metrics(cache_root, combo):
        per_pair = {}
        tp = pred = anch = zero = 0
        for tag in sorted(cache_root.iterdir()):
            f = tag / "pair_cache.json"
            if not f.exists():
                continue
            c = json.loads(f.read_text())
            if c["status"] != "ok":
                continue
            nm = c["combos"][combo]["node_metrics"]
            per_pair[c["pair_id"]] = nm["f1"]
            tp += nm["tp"]
            pred += nm["pred_count"]
            anch += nm["anchor_count"]
            if not nm["node_corrs"]:
                zero += 1
        f1s = list(per_pair.values())
        mp = tp / pred if pred else 0.0
        mr = tp / anch if anch else 0.0
        return {
            "macro_node_f1": float(np.mean(f1s)),
            "micro_node_f1": 2 * mp * mr / max(mp + mr, 1e-12),
            "total_valid_candidates": pred,
            "zero_candidate_pairs": zero,
        }, per_pair

    v4 = OLD / "calibration90"
    incumbent, inc_pairs = cache_metrics(
        ROOT / ("outputs/official_sgaligner_v3_pct_parity_baseline_"
                "20260827/final_inference_cache/calibration90"),
        "pct+rel")
    old_c, old_c_pairs = cache_metrics(
        v4 / "cache_explicit", "candidate")
    old_b, old_b_pairs = cache_metrics(
        v4 / "cache_complete", "candidate")

    def paired(candidate_pairs, base_pairs):
        common = sorted(set(candidate_pairs) & set(base_pairs))
        deltas = [candidate_pairs[p_] - base_pairs[p_]
                  for p_ in common]
        return {
            "common_pairs": len(common),
            "node_f1_delta_mean": float(np.mean(deltas)),
            "improved": sum(1 for d in deltas if d > 0),
            "regressed": sum(1 for d in deltas if d < 0),
        }

    winner_c_pairs = json.loads(
        (NEW / "calibration_winner_C.json").read_text())["per_pair_f1"]
    winner_b_pairs = json.loads(
        (NEW / "calibration_winner_B.json").read_text())["per_pair_f1"]
    comparison = {
        "incumbent_pct_rel": incumbent,
        "old_C_epoch20": old_c,
        "old_B_epoch40": old_b,
        "paired_vs_incumbent": {
            "winner_B_epoch15": paired(winner_b_pairs, inc_pairs),
            "winner_C_epoch20": paired(winner_c_pairs, inc_pairs),
            "old_C_epoch20": paired(old_c_pairs, inc_pairs),
            "old_B_epoch40": paired(old_b_pairs, inc_pairs),
        },
    }
    out["comparison"] = comparison
    (NEW / "calibration_paired_comparison.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main()
