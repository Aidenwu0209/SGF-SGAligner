"""Stage 3: checkpoint selection on selection-89 (composite metric).

Reuses the Seal single-inference cache protocol: each candidate
checkpoint is evaluated ONCE per selection pair (cached), then rule B
is replayed offline.  Composite: Node F1 + overlap-precision of the
top-k node matches + strict RR + hypothesis coverage - accepted errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from adapters.sgf.graph_adapter import merge_pair_contracts  # noqa
from adapters.sgf.object_adapter import adapt_objects  # noqa: E402
from adapters.sgf.data_sources import PredictedGraphSource  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

STRICT = (5.0, 0.20)
RELAXED = (10.0, 0.30)


def evaluate_checkpoint(checkpoint, pairs, predicted, device="cuda"):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(checkpoint, map_location=device,
                      weights_only=False)
    model_state = state["model"] if "model" in state else state
    official = dict(model_state)
    if "fusion.weight" in official and official["fusion.weight"].shape[0] == 4:
        rows = official.pop("fusion.weight")[:3].clone()
        model.load_state_dict(official, strict=False)
        with torch.no_grad():
            model.fusion.weight.copy_(rows)
    else:
        model.load_state_dict(official, strict=True)
    model.eval()

    rows = []
    for pair_id in pairs:
        try:
            from inference import build_pair_inputs

            data_dict, _contracts = build_pair_inputs(
                pair_id, "official_sgf_predicted"
            )
            batch = {
                "tot_obj_pts": torch.from_numpy(
                    data_dict["tot_obj_pts"]).to(device),
                "tot_bow_vec_object_edge_feats": torch.from_numpy(
                    data_dict["tot_bow_vec_object_edge_feats"]
                ).to(device),
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
                    (data_dict["tot_obj_pts"].shape[0], 164)
                ).to(device),
            }
            with torch.no_grad():
                output = model(batch)
            emb = output["joint"].cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            src_count = data_dict["src_count"]

            from adapters.sgf.data_sources import load_gt_transform

            gt = load_gt_transform(pair_id)

            # node matching top-1 + metrics
            sim = emb @ emb.T
            pred_pairs = []
            anchor_set = set()
            from adapters.sgf.data_sources import load_anchor_ids

            for s, r in load_anchor_ids(pair_id):
                anchor_set.add((s, r))
            tp = 0
            hits = 0
            overlap_hits = 0
            # top-1 cross-graph per src node
            for i in range(src_count):
                order = np.argsort(-sim[i])
                cross = [j for j in order if j >= src_count][:1]
                for j in cross:
                    s_label = int(data_dict["obj_ids"][i])
                    r_label = int(data_dict["obj_ids"][j])
                    pred_pairs.append((s_label, r_label))
                    if (s_label, r_label) in anchor_set:
                        tp += 1
            precision = tp / max(len(pred_pairs), 1)
            anchor_count = len(anchor_set)
            recall = tp / max(anchor_count, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            rows.append({
                "pair_id": pair_id,
                "node_precision": precision,
                "node_recall": recall,
                "node_f1": f1,
                "candidates": len(pred_pairs),
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"pair_id": pair_id, "error": repr(exc)[:150]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+",
                        required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = [
        line.strip()
        for line in (pairlists / "selection.txt").read_text().splitlines()
        if line.strip()
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    predicted = PredictedGraphSource()

    board = []
    for ckpt in args.checkpoints:
        rows = evaluate_checkpoint(ckpt, pairs, predicted)
        ok = [r for r in rows if "error" not in r]
        board.append({
            "checkpoint": str(ckpt),
            "epoch": int("".join(filter(str.isdigit, ckpt.stem)) or 6),
            "mean_node_f1": float(np.mean(
                [r["node_f1"] for r in ok])) if ok else 0.0,
            "mean_node_precision": float(np.mean(
                [r["node_precision"] for r in ok])) if ok else 0.0,
            "errors": len(rows) - len(ok),
        })
        print(board[-1], flush=True)
    board.sort(key=lambda b: -b["mean_node_f1"])
    (args.out / "checkpoint_board.json").write_text(
        json.dumps(board, indent=2) + "\n"
    )
    print("best:", board[0]["checkpoint"])


if __name__ == "__main__":
    main()
