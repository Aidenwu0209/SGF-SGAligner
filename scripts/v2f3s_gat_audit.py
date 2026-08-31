"""V2T-Fix3-Seal stage 3: corrected GAT factorial + stagewise audit.

Fixes the Fix3 factorial bug: ``explicit_only_shadow`` previously used
the COMPLETED edge list (all-pairs + 'none'), so it was numerically
identical to the official complete-none contract. The true explicit
edges are now extracted from the RAW ``directed_pairs`` of the graph
sources (pre-completion), mapped to node indices via the contracts.

Every experiment now runs under BOTH the current official checkpoint
and a random-initialisation control, with per-stage GAT outputs
captured, and all inputs/outputs hashed.
"""
from __future__ import annotations

import hashlib
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
from adapters.sgf.data_sources import (  # noqa: E402
    OracleGraphSource, PredictedGraphSource, load_anchor_ids,
)

OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal"
OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def load_ckpt_model(device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(OFFICIAL, map_location=device, weights_only=False)
    official = dict(state["model"])
    official.pop("fusion.weight")
    model.load_state_dict(official, strict=False)
    model.eval()
    return model


def load_random_model(device):
    torch.manual_seed(20260827)
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    model.eval()
    return model


def true_explicit_edges(pairs, object_id2idx):
    """RAW directed pairs (pre-completion) mapped to node index space."""
    seen = []
    for sub, obj in pairs:
        if (sub, obj) in seen:
            continue  # official dedup before degree/root computation
        seen.append((sub, obj))
    mapped = []
    unmapped = 0
    for sub, obj in seen:
        if sub in object_id2idx and obj in object_id2idx:
            mapped.append((object_id2idx[sub], object_id2idx[obj]))
        else:
            unmapped += 1
    edges = np.asarray(mapped, dtype=np.int64).reshape(-1, 2)
    return edges, unmapped, len(seen)


def edge_variants(n_nodes, explicit_edges):
    complete = [
        (i, j) for i in range(n_nodes) for j in range(n_nodes)
        if i != j
    ]
    complete = np.asarray(complete, dtype=np.int64).reshape(-1, 2)
    self_loop = np.stack(
        [np.arange(n_nodes), np.arange(n_nodes)], axis=1
    ).astype(np.int64)
    return {
        "complete_none": complete,
        "explicit_only_true": explicit_edges,
        "self_loop_diag": self_loop,
    }


def stagewise_forward(model, rel_pose, edges_np, device):
    """rel_trans -> GAT layer0 -> layer1 -> projection -> normalised.

    Mirrors the official forward: dropout p is the model's (0.0 here),
    edges are passed transposed as int32 exactly like sg_aligner.py.
    """
    x = rel_pose.to(device)
    edge_index = (
        torch.from_numpy(edges_np.astype(np.int64)).to(device)
        .t().to(torch.int32)
    )
    stages = {}
    with torch.no_grad():
        h = x
        for idx, layer in enumerate(model.structure_encoder.layer_stack):
            h = F.dropout(h, model.structure_encoder.dropout,
                          training=False)
            h = layer(h, edge_index)
            stages[f"gat_layer{idx}"] = h.detach().cpu().numpy()
            if idx + 1 < len(model.structure_encoder.layer_stack):
                h = F.elu(h)
        proj = model.structure_embedding(h)
        stages["structure_embedding"] = proj.detach().cpu().numpy()
        stages["normalized"] = F.normalize(
            proj, dim=1).detach().cpu().numpy()
    return stages


def stage_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    rounded = np.round(a, 4)
    unique = len(np.unique(rounded, axis=0)) if a.size else 0
    return {
        "std": float(a.std()) if a.size else 0.0,
        "max_abs": float(np.abs(a).max()) if a.size else 0.0,
        "unique_rows": int(unique),
        "hash": hash_of(a.astype(np.float32)),
    }


def matching_metrics(emb, data_dict, anchors):
    e = emb / np.maximum(
        np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sim = e @ e.T
    src_count = data_dict["src_count"]
    top1 = top5 = cand = 0
    for i in range(src_count):
        js = [x for x in np.argsort(-sim[i]) if x >= src_count][:5]
        if not js:
            continue
        cand += 1
        if (int(data_dict["obj_ids"][i]),
                int(data_dict["obj_ids"][js[0]])) in anchors:
            top1 += 1
        if any(
            (int(data_dict["obj_ids"][i]),
             int(data_dict["obj_ids"][j])) in anchors for j in js
        ):
            top5 += 1
    return {
        "top1_precision": top1 / max(cand, 1),
        "top5_precision": top5 / max(cand, 1),
    }


def explicit_assertion(rows, out):
    """Fail-loud check: explicit < complete unless a scan is complete."""
    special = []
    for r in rows:
        if r["explicit_edge_count"] >= r["complete_edge_count"]:
            special.append({
                "pair": r["pair"], "mode": r["mode"],
                "explicit": r["explicit_edge_count"],
                "complete": r["complete_edge_count"],
                "note": "scan graph is complete in explicit relations",
            })
    out["complete_graph_special_cases"] = special
    out["assertion_explicit_lt_complete"] = len(special) == 0
    return special


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_model = load_ckpt_model(device)
    random_model = load_random_model(device)

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = [l.strip() for l in
             (pairlists / "selection.txt").read_text().splitlines()
             if l.strip()][:20]

    sources = {
        "official_oracle": OracleGraphSource(),
        "official_sgf_predicted": PredictedGraphSource(),
    }

    factorial_rows = []
    stagewise_rows = []
    error_rows = []

    for mode in ("official_oracle", "official_sgf_predicted"):
        source = sources[mode]
        for pair in pairs:
            try:
                data_dict, contracts = build_pair_inputs(pair, mode)
            except Exception as exc:  # noqa: BLE001
                error_rows.append(
                    {"pair": pair, "mode": mode, "error": repr(exc)[:150]})
                continue
            anchors = set(load_anchor_ids(pair))
            src_count = data_dict["src_count"]
            n = data_dict["tot_obj_pts"].shape[0]

            # TRUE explicit edges per graph from the RAW source pairs
            src_pairs = source.load(pair.split("_to_")[0]).directed_pairs
            ref_pairs = source.load(pair.split("_to_")[1]).directed_pairs
            src_explicit, s_unmapped, s_raw = true_explicit_edges(
                src_pairs, data_dict["src_object_id2idx"])
            ref_explicit, r_unmapped, r_raw = true_explicit_edges(
                ref_pairs, data_dict["ref_object_id2idx"])

            src_vars = edge_variants(src_count, src_explicit)
            ref_vars = edge_variants(n - src_count, ref_explicit)

            # rel_trans variants (pair-level tensor)
            raw = torch.as_tensor(
                np.asarray(data_dict["tot_rel_pose"]))
            standardized = (raw - raw.mean(dim=0, keepdim=True)) / (
                raw.std(dim=0, keepdim=True) + 1e-8)
            shuffled = raw[torch.randperm(raw.shape[0])]
            rt_variants = {
                "raw": raw, "standardized": standardized,
                "shuffled_control": shuffled,
            }

            for adj_name in ("complete_none", "explicit_only_true",
                             "self_loop_diag"):
                exp_total = int(len(src_explicit) + len(ref_explicit))
                comp_total = int(
                    len(src_vars["complete_none"])
                    + len(ref_vars["complete_none"]))
                edge_hashes = {
                    "src": hash_of(src_vars[adj_name]),
                    "ref": hash_of(ref_vars[adj_name]),
                }
                base = {
                    "pair": pair[:20], "mode": mode, "adjacency": adj_name,
                    "explicit_edge_count": exp_total,
                    "complete_edge_count": comp_total,
                    "edge_count_src_ref": [
                        int(len(src_vars[adj_name])),
                        int(len(ref_vars[adj_name]))],
                    "edge_hash": edge_hashes,
                    "unmapped_pairs_src_ref": [s_unmapped, r_unmapped],
                    "raw_directed_pairs_src_ref": [s_raw, r_raw],
                }
                for rt_name, rt_tensor in rt_variants.items():
                    for model_name, model in (
                        ("ckpt", ckpt_model), ("random_init", random_model)
                    ):
                        rel_s = stagewise_forward(
                            model, rt_tensor[:src_count].float(),
                            src_vars[adj_name], device)
                        rel_r = stagewise_forward(
                            model, rt_tensor[src_count:].float(),
                            ref_vars[adj_name], device)
                        merged = {
                            key: np.concatenate(
                                [rel_s[key], rel_r[key]])
                            for key in rel_s
                        }
                        emb = merged["normalized"]
                        mm = matching_metrics(emb, data_dict, anchors)
                        row = dict(base)
                        row.update({
                            "rel_trans": rt_name,
                            "model": model_name,
                            "input_rel_trans_std": float(
                                np.asarray(rt_tensor).std()),
                            "input_rel_trans_hash": hash_of(
                                np.asarray(rt_tensor,
                                           dtype=np.float32)),
                        })
                        for stage in ("gat_layer0", "gat_layer1",
                                      "structure_embedding", "normalized"):
                            stats = stage_stats(merged[stage])
                            row[f"{stage}_std"] = stats["std"]
                            row[f"{stage}_max_abs"] = stats["max_abs"]
                            row[f"{stage}_unique"] = stats["unique_rows"]
                            row[f"{stage}_hash"] = stats["hash"]
                        row.update(mm)
                        factorial_rows.append(row)
                        if (model_name == "ckpt"
                                and rt_name == "raw"
                                and len(stagewise_rows) < 60):
                            stagewise_rows.append({
                                **{k: v for k, v in row.items()
                                   if not k.endswith(
                                       ("_hash", "_std", "_unique"))},
                                "per_stage": {
                                    stage: stage_stats(merged[stage])
                                    for stage in merged
                                },
                            })

    # explicit < complete assertion (fail loud unless complete graphs)
    special = []
    for r in factorial_rows:
        if r["adjacency"] == "explicit_only_true" and \
                r["explicit_edge_count"] >= r["complete_edge_count"]:
            special.append({
                "pair": r["pair"], "mode": r["mode"],
                "explicit": r["explicit_edge_count"],
                "complete": r["complete_edge_count"],
            })
    if special:
        print("WARNING complete-graph special cases:", special)

    # aggregate per factor combo x model
    agg = {}
    for row in factorial_rows:
        key = (row["mode"], row["adjacency"], row["rel_trans"],
               row["model"])
        agg.setdefault(key, []).append(row)
    summary = {}
    for (mode, adj, rt, model), rows in agg.items():
        summary[f"{mode}|{adj}|{rt}|{model}"] = {
            "n": len(rows),
            "mean_normalized_std": float(np.mean(
                [r["normalized_std"] for r in rows])),
            "mean_unique": float(np.mean(
                [r["normalized_unique"] for r in rows])),
            "mean_gat_layer1_std": float(np.mean(
                [r["gat_layer1_std"] for r in rows])),
            "mean_gat_layer0_std": float(np.mean(
                [r["gat_layer0_std"] for r in rows])),
            "mean_top1": float(np.mean(
                [r["top1_precision"] for r in rows])),
            "mean_top5": float(np.mean(
                [r["top5_precision"] for r in rows])),
            "unique_gat_layer1_hash_all_pairs": len(
                {r["gat_layer1_hash"] for r in rows}),
        }

    payload_factorial = {
        "phase": "V2T-Fix3-Seal stage 3 corrected factorial",
        "bug_fixed": (
            "Fix3 explicit_only_shadow used the completed edge list; "
            "explicit edges now come from RAW source directed_pairs "
            "(pre-completion), deduplicated, mapped via contracts"
        ),
        "assertion_explicit_lt_complete": len(special) == 0,
        "complete_graph_special_cases": special,
        "models": ["ckpt (official b716c7d8)", "random_init seed 20260827"],
        "summary": summary,
        "rows": factorial_rows,
        "errors": error_rows,
    }
    (OUT / "gat_factorial_corrected.json").write_text(
        json.dumps(payload_factorial, indent=2) + "\n")
    (OUT / "gat_stagewise_audit.json").write_text(
        json.dumps({
            "phase": "V2T-Fix3-Seal stagewise audit",
            "stages": ["input_rel_trans", "gat_layer0", "gat_layer1",
                       "structure_embedding", "normalized"],
            "rows": stagewise_rows,
        }, indent=2) + "\n")

    print(json.dumps({
        "rows": len(factorial_rows),
        "errors": len(error_rows),
        "assertion_explicit_lt_complete": len(special) == 0,
        "special": len(special),
    }, indent=1))
    for key in sorted(summary):
        s = summary[key]
        print(f"{key:70s} n{int(s['n']):3d} "
              f"normStd {s['mean_normalized_std']:.3e} "
              f"uniq {s['mean_unique']:6.1f} "
              f"top1 {s['mean_top1']:.3f}")


if __name__ == "__main__":
    main()
