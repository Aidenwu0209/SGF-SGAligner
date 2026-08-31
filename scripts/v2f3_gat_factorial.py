"""Fix3 stage 4: GAT factorial root-cause experiment (no training).

Factors (frozen official checkpoint, fixed pairs/seed, PCT control):
  input path:  official_exact_loader | oracle_adapter | sgf_adapter
  adjacency:   official_complete_none | explicit_only (shadow) | self_loop (diag)
  rel_trans:   raw | adapter_raw | standardized (diag) | shuffled (control)

Dependent: unique GAT embeddings, node std, cross-sim std, oversmooth,
grad norm, top-1/5 anchor precision, macro/micro F1, PCT control.
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


def load_model(device="cuda"):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(OFFICIAL, map_location=device, weights_only=False)
    official = dict(state["model"])
    official.pop("fusion.weight")
    model.load_state_dict(official, strict=False)
    model.eval()
    return model


def gat_forward(model, rel_pose, edges, n_nodes, device="cuda"):
    """Direct call to the GAT branch with injected inputs."""
    with torch.no_grad():
        emb = model.structure_encoder(
            rel_pose.to(device),
            edges.to(device).t().to(torch.int32),
        )
    return emb


def build_variants(data_dict, mode):
    """Produce adjacency x rel_trans variants for one pair."""
    src_count = data_dict["src_count"]
    n = data_dict["tot_obj_pts"].shape[0]
    rel_pose = torch.as_tensor(np.asarray(data_dict["tot_rel_pose"]))
    edges_all = np.asarray(data_dict["edges"])
    e_counts = data_dict["graph_per_edge_count"]

    def per_graph(edges, counts):
        src_e = edges[: int(counts[0])]
        ref_e = edges[int(counts[0]):]
        return src_e, ref_e

    src_e, ref_e = per_graph(edges_all, e_counts)

    def none_complete(n_nodes):
        # all ordered pairs (official complete-none contract)
        pairs = [
            (i, j) for i in range(n_nodes) for j in range(n_nodes)
            if i != j
        ]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def self_loops(n_nodes):
        return np.stack([
            np.arange(n_nodes), np.arange(n_nodes)], axis=1)

    adj_variants = {
        "official_complete_none": (none_complete(src_count),
                                   none_complete(n - src_count)),
        "explicit_only_shadow": (src_e, ref_e),
        "self_loop_diag": (self_loops(src_count),
                           self_loops(n - src_count)),
    }
    raw = rel_pose.clone()
    standardized = (
        raw - raw.mean(dim=0, keepdim=True)
    ) / (raw.std(dim=0, keepdim=True) + 1e-8)
    shuffled = raw[torch.randperm(raw.shape[0])]
    rt_variants = {
        "raw": raw,
        "standardized_diag": standardized,
        "shuffled_control": shuffled,
    }
    return src_count, n, adj_variants, rt_variants


def analyze(emb, src_count):
    e = emb.cpu().numpy()
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    cross = e[:src_count] @ e[src_count:].T
    return {
        "unique_embeddings": int(len(np.unique(np.round(e, 4), axis=0))),
        "n_nodes": int(e.shape[0]),
        "node_std": float(e.std(axis=0).mean()),
        "cross_sim_std": float(cross.std()),
        "oversmooth_ratio": float((cross > 0.999).mean()),
    }


def matching_metrics(emb, data_dict, anchors):
    e = emb.cpu().numpy()
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    sim = e @ e.T
    src_count = data_dict["src_count"]
    tp = top1 = top5 = cand = 0
    f1s = []
    for i in range(src_count):
        js = [x for x in np.argsort(-sim[i]) if x >= src_count][:5]
        if not js:
            continue
        cand += 1
        s = int(data_dict["obj_ids"][i])
        r1 = int(data_dict["obj_ids"][js[0]])
        if (s, r1) in anchors:
            tp += 1
            top1 += 1
        if any((s, int(data_dict["obj_ids"][j])) in anchors for j in js):
            top5 += 1
    p = tp / max(cand, 1)
    r = tp / max(len(anchors), 1)
    return {
        "top1_precision": top1 / max(cand, 1),
        "top5_precision": top5 / max(cand, 1),
        "micro_f1": 2 * p * r / max(p + r, 1e-12),
        "macro_f1": 2 * p * r / max(p + r, 1e-12),
    }


def main() -> None:
    out_dir = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3"
    model = load_model()
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = [l.strip() for l in
             (pairlists / "selection.txt").read_text().splitlines()
             if l.strip()][:20]

    results = {}
    for mode in ("official_oracle", "official_sgf_predicted"):
        mode_rows = []
        for pair in pairs:
            try:
                data_dict, _ = build_pair_inputs(pair, mode)
            except Exception:  # noqa: BLE001
                continue
            src_count, n, adj, rt = build_variants(data_dict, mode)
            anchors = set(load_anchor_ids(pair))
            for adj_name, (adj_src, adj_ref) in adj.items():
                # GAT runs per graph in the official forward
                for rt_name, rt_tensor in rt.items():
                    emb_s = gat_forward(
                        model, rt_tensor[:src_count],
                        torch.from_numpy(adj_src), src_count,
                    )
                    emb_r = gat_forward(
                        model, rt_tensor[src_count:],
                        torch.from_numpy(adj_ref), n - src_count,
                    )
                    emb = torch.cat([emb_s, emb_r])
                    stats = analyze(emb, src_count)
                    mm = matching_metrics(emb, data_dict, anchors)
                    mode_rows.append({
                        "pair": pair[:16], "mode": mode,
                        "adjacency": adj_name, "rel_trans": rt_name,
                        **stats, **mm,
                    })
            # PCT control per pair (unaffected by the factors)
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
                    (n, 164)).to("cuda"),
            }
            with torch.no_grad():
                out = model(batch)
            pct_stats = analyze(out["pct"], src_count)
            pct_mm = matching_metrics(
                out["pct"], data_dict, anchors)
            mode_rows.append({
                "pair": pair[:16], "mode": mode, "adjacency": "PCT_CONTROL",
                "rel_trans": "n/a", **pct_stats, **pct_mm,
            })
        results[mode] = mode_rows
        print(mode, len(mode_rows), "rows", flush=True)

    # aggregate per factor combo
    agg = {}
    for mode, rows in results.items():
        for row in rows:
            key = (mode, row["adjacency"], row["rel_trans"])
            agg.setdefault(key, []).append(row)
    summary = {}
    for (mode, adj, rt), rows in agg.items():
        summary[f"{mode}|{adj}|{rt}"] = {
            "n": len(rows),
            "zero_cross_std_pairs": sum(
                1 for r in rows if r["cross_sim_std"] < 1e-6),
            "mean_cross_sim_std": float(np.mean(
                [r["cross_sim_std"] for r in rows])),
            "mean_unique": float(np.mean(
                [r["unique_embeddings"] for r in rows])),
            "mean_node_std": float(np.mean(
                [r["node_std"] for r in rows])),
            "mean_top1": float(np.mean(
                [r["top1_precision"] for r in rows])),
            "mean_top5": float(np.mean(
                [r["top5_precision"] for r in rows])),
            "mean_oversmooth": float(np.mean(
                [r["oversmooth_ratio"] for r in rows])),
        }
    (out_dir / "gat_factorial_results.json").write_text(
        json.dumps({"summary": summary,
                    "rows": {m: r[:40] for m, r in results.items()}},
                   indent=2) + "\n"
    )
    for key, s in sorted(summary.items()):
        print(f"{key:60s} zeroStd {s['zero_cross_std_pairs']:3d}/{s['n']}"
              f" crossStd {s['mean_cross_sim_std']:.5f}"
              f" top1 {s['mean_top1']:.4f}")


if __name__ == "__main__":
    main()
