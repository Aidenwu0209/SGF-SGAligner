"""Fix2 stage 5: modality contribution A/B on selection-89.

Same frozen official checkpoint, same pairs, same backend.  Six
combinations built from the per-modality embeddings of one forward:
single modalities + equal-normalized concatenation for combos (the
official fusion is a learned softmax over full-module sets; for the A/B
we use the SAME aggregation rule across all combos so differences
isolate modality content, not fusion weights).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from inference import build_pair_inputs  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402

OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"

COMBOS = {
    "pct": ["pct"],
    "gat": ["gat"],
    "rel": ["rel"],
    "pct+gat": ["pct", "gat"],
    "pct+rel": ["pct", "rel"],
    "pct+gat+rel": ["pct", "gat", "rel"],
}


def combo_embedding(out, combo, device):
    parts = []
    for m in combo:
        e = out[m].cpu().numpy()
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        parts.append(e / np.sqrt(len(combo)))
    return np.concatenate(parts, axis=1)


def main() -> None:
    out_dir = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix2"
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to("cuda")
    state = torch.load(OFFICIAL, map_location="cuda", weights_only=False)
    official = dict(state["model"])
    official.pop("fusion.weight")
    model.load_state_dict(official, strict=False)
    model.eval()

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = [l.strip() for l in
             (pairlists / "selection.txt").read_text().splitlines() if l.strip()]

    per_combo = {name: {"top1_hits": 0, "top5_hits": 0, "total": 0,
                        "tp": 0, "cand": 0, "anchor_total": 0,
                        "f1s": []}
                 for name in COMBOS}
    for pair in pairs:
        try:
            data_dict, _ = build_pair_inputs(pair, "official_sgf_predicted")
            batch = {
                "tot_obj_pts": torch.from_numpy(
                    data_dict["tot_obj_pts"]).to("cuda"),
                "tot_bow_vec_object_edge_feats": torch.from_numpy(
                    data_dict["tot_bow_vec_object_edge_feats"]).to("cuda"),
                "tot_rel_pose": torch.from_numpy(
                    data_dict["tot_rel_pose"]).to("cuda"),
                "edges": torch.from_numpy(
                    data_dict["edges"].astype(np.int64)).to("cuda"),
                "graph_per_obj_count": [np.asarray(
                    data_dict["graph_per_obj_count"], dtype=np.int64)],
                "graph_per_edge_count": [np.asarray(
                    data_dict["graph_per_edge_count"], dtype=np.int64)],
                "batch_size": 1,
                "tot_bow_vec_object_attr_feats": torch.zeros(
                    (data_dict["tot_obj_pts"].shape[0], 164)).to("cuda"),
            }
            with torch.no_grad():
                out = model(batch)
        except Exception:  # noqa: BLE001
            continue
        src_count = data_dict["src_count"]
        anchors = set(load_anchor_ids(pair))
        for name, combo in COMBOS.items():
            emb = combo_embedding(out, combo, "cuda")
            sim = emb @ emb.T
            tp = 0
            cand = 0
            top1 = 0
            top5 = 0
            for i in range(src_count):
                order = np.argsort(-sim[i])
                js = [x for x in order if x >= src_count][:5]
                if not js:
                    continue
                cand += 1
                s_label = int(data_dict["obj_ids"][i])
                r1 = int(data_dict["obj_ids"][js[0]])
                if (s_label, r1) in anchors:
                    tp += 1
                    top1 += 1
                if any((s_label, int(data_dict["obj_ids"][j])) in anchors
                       for j in js):
                    top5 += 1
            c = per_combo[name]
            c["tp"] += tp
            c["cand"] += cand
            c["anchor_total"] += len(anchors)
            c["top1_hits"] += top1
            c["top5_hits"] += top5
            c["total"] += 1
            p = tp / max(cand, 1)
            r = tp / max(len(anchors), 1)
            c["f1s"].append(2 * p * r / max(p + r, 1e-12))

    results = {}
    for name, c in per_combo.items():
        results[name] = {
            "pairs": c["total"],
            "micro_node_precision": c["tp"] / max(c["cand"], 1),
            "micro_node_recall": c["tp"] / max(c["anchor_total"], 1),
            "micro_node_f1": (
                2 * (c["tp"] / max(c["cand"], 1))
                * (c["tp"] / max(c["anchor_total"], 1))
            ) / max(
                (c["tp"] / max(c["cand"], 1))
                + (c["tp"] / max(c["anchor_total"], 1)), 1e-12
            ),
            "macro_node_f1": float(np.mean(c["f1s"])),
            "top1_precision": c["top1_hits"] / max(c["cand"], 1),
            "top5_precision": c["top5_hits"] / max(c["cand"], 1),
        }
    (out_dir / "modality_ab_matching.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    for name, r in results.items():
        print(f"{name:14s} top1 {r['top1_precision']:.4f} "
              f"top5 {r['top5_precision']:.4f} "
              f"macroF1 {r['macro_node_f1']:.4f}")


if __name__ == "__main__":
    main()
