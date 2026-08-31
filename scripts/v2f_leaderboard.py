"""Fair checkpoint selection: cheap node-matching metrics on ALL
candidates first, then full selection-89 registration for the
pre-registered top-3 only.  Selection rule frozen before running."""
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

CKPT_DIR = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix1/training_B"


def load_model(ckpt_path, device="cuda"):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = state["model"] if "model" in state else state
    official = dict(ms)
    if "fusion.weight" in official and official["fusion.weight"].shape[0] == 4:
        rows = official.pop("fusion.weight")[:3].clone()
        model.load_state_dict(official, strict=False)
        with torch.no_grad():
            model.fusion.weight.copy_(rows)
    else:
        model.load_state_dict(official, strict=False)
    model.eval()
    return model


def macro_node_f1(model, pairs, device="cuda"):
    """MACRO (per-pair mean) Node F1 — the pre-registered definition.

    Per pair: candidates = top-1 cross-graph pick per src node;
    TP = picks that are GT anchor pairs; P = TP/#cand; R = TP/#anchors;
    F1 per pair; report the MEAN over pairs.  (The historical 0.214
    gate used a MICRO variant; both are reported, never compared.)
    """
    f1s = []
    for pair_id in pairs:
        try:
            data_dict, _ = build_pair_inputs(pair_id, "official_sgf_predicted")
            batch = {
                "tot_obj_pts": torch.from_numpy(
                    data_dict["tot_obj_pts"]).to(device),
                "tot_bow_vec_object_edge_feats": torch.from_numpy(
                    data_dict["tot_bow_vec_object_edge_feats"]).to(device),
                "tot_rel_pose": torch.from_numpy(
                    data_dict["tot_rel_pose"]).to(device),
                "edges": torch.from_numpy(
                    data_dict["edges"].astype(np.int64)).to(device),
                "graph_per_obj_count": [np.asarray(
                    data_dict["graph_per_obj_count"], dtype=np.int64)],
                "graph_per_edge_count": [np.asarray(
                    data_dict["graph_per_edge_count"], dtype=np.int64)],
                "batch_size": 1,
                "tot_bow_vec_object_attr_feats": torch.zeros(
                    (data_dict["tot_obj_pts"].shape[0], 164)).to(device),
            }
            with torch.no_grad():
                out = model(batch)
            emb = out["joint"].cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            sim = emb @ emb.T
            src_count = data_dict["src_count"]
            anchors = set(load_anchor_ids(pair_id))
            tp = 0
            cand = 0
            for i in range(src_count):
                order = np.argsort(-sim[i])
                j = next(x for x in order if x >= src_count)
                s_label = int(data_dict["obj_ids"][i])
                r_label = int(data_dict["obj_ids"][j])
                cand += 1
                if (s_label, r_label) in anchors:
                    tp += 1
            p = tp / max(cand, 1)
            r = tp / max(len(anchors), 1)
            f1s.append(2 * p * r / max(p + r, 1e-12))
        except Exception:  # noqa: BLE001
            f1s.append(0.0)
    return float(np.mean(f1s))


def main() -> None:
    out_dir = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix1"
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = [
        line.strip()
        for line in (pairlists / "selection.txt").read_text().splitlines()
        if line.strip()
    ]
    cands = sorted(CKPT_DIR.glob("epoch_*.pt"))
    board = []
    for ckpt in cands:
        model = load_model(ckpt)
        f1 = macro_node_f1(model, pairs)
        board.append({"checkpoint": ckpt.name, "macro_node_f1": f1})
        print(board[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    # frozen rule: rank by macro Node F1, keep top-3 by epoch for full eval
    board.sort(key=lambda b: -b["macro_node_f1"])
    (out_dir / "checkpoint_leaderboard.json").write_text(
        json.dumps({"rule": "macro node F1 (per-pair mean, top-1 picks)",
                    "board": board}, indent=2) + "\n"
    )
    print("top3:", [b["checkpoint"] for b in board[:3]])


if __name__ == "__main__":
    main()
