"""Fix2 stage 4: GAT modality-collapse audit — oracle vs predicted,
official checkpoint vs fix1 epoch-21, on selection-89 + calibration-90."""
from __future__ import annotations

import argparse
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

OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
FIX1 = (ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix1/"
        "training_B/epoch_00021.pt")


def load_model(ckpt_path, device="cuda"):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ms = state["model"] if "model" in state else state
    official = dict(ms)
    if official.get("fusion.weight").shape[0] == 4:
        rows = official.pop("fusion.weight")[:3].clone()
        model.load_state_dict(official, strict=False)
        with torch.no_grad():
            model.fusion.weight.copy_(rows)
    else:
        model.load_state_dict(official, strict=False)
    model.eval()
    return model


def audit_pair(model, pair_id, mode, device="cuda"):
    data_dict, _ = build_pair_inputs(pair_id, mode)
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
    }
    if mode == "official_oracle":
        batch["tot_bow_vec_object_attr_feats"] = torch.from_numpy(
            data_dict["tot_bow_vec_object_attr_feats"]
        ).to(device)
    else:
        batch["tot_bow_vec_object_attr_feats"] = torch.zeros(
            (data_dict["tot_obj_pts"].shape[0], 164)).to(device)

    with torch.no_grad():
        out = model(batch)

    result = {}
    for modality in ("pct", "gat", "rel", "joint"):
        emb = out[modality].cpu().numpy()
        emb_n = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12
        )
        # per-graph node variance (how much nodes differ within a graph)
        src_count = data_dict["src_count"]
        node_std = float(emb_n.std(axis=0).mean())
        src_std = float(emb_n[:src_count].std(axis=0).mean())
        ref_std = float(emb_n[src_count:].std(axis=0).mean())
        # cross-graph similarity stats
        cross = emb_n[:src_count] @ emb_n[src_count:].T
        # unique embeddings (rounded to 4dp)
        unique = int(
            len(np.unique(np.round(emb_n, 4), axis=0))
        )
        # oversmoothing: fraction of cross-graph sims > 0.999
        oversmooth = float((cross > 0.999).mean())
        result[modality] = {
            "node_std_mean": node_std,
            "src_std": src_std,
            "ref_std": ref_std,
            "cross_sim_std": float(cross.std()),
            "cross_sim_min": float(cross.min()),
            "cross_sim_max": float(cross.max()),
            "unique_embeddings": unique,
            "n_nodes": int(emb_n.shape[0]),
            "oversmooth_ratio": oversmooth,
        }
    # graph shape stats
    prov = data_dict["provenance"]
    result["graph"] = {
        "src_objects": int(data_dict["graph_per_obj_count"][0]),
        "ref_objects": int(data_dict["graph_per_obj_count"][1]),
        "src_edges": int(data_dict["graph_per_edge_count"][0]),
        "ref_edges": int(data_dict["graph_per_edge_count"][1]),
        "src_directed_pairs_before_none": prov["src"][
            "directed_pairs_before_none"
        ],
        "ref_directed_pairs_before_none": prov["ref"][
            "directed_pairs_before_none"
        ],
        "src_none_edges": int(data_dict["graph_per_edge_count"][0])
        - prov["src"]["directed_pairs_before_none"],
        "ref_none_edges": int(data_dict["graph_per_edge_count"][1])
        - prov["ref"]["directed_pairs_before_none"],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    splits = {
        "selection": [l.strip() for l in
                      (pairlists / "selection.txt").read_text().splitlines()
                      if l.strip()],
        "calibration": [l.strip() for l in
                        (pairlists / "calibration.txt").read_text()
                        .splitlines() if l.strip()],
    }
    # for oracle mode only pairs whose scans exist in 3DSSG PLY form work;
    # use the same pair lists (oracle loader raises if missing -> record)
    checkpoints = {"official_epoch6": OFFICIAL, "fix1_epoch21": FIX1}

    summary = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        model = load_model(ckpt_path)
        for split, pairs in splits.items():
            if args.limit:
                pairs = pairs[: args.limit]
            rows = []
            for pair in pairs:
                for mode in ("official_sgf_predicted", "official_oracle"):
                    try:
                        r = audit_pair(model, pair, mode)
                    except Exception as exc:  # noqa: BLE001
                        r = {"error": repr(exc)[:120], "mode": mode}
                    r["pair_id"] = pair
                    r["mode"] = mode
                    rows.append(r)
            # aggregate
            agg = {}
            for modality in ("gat", "pct", "rel", "joint"):
                ok = [r for r in rows if modality in r]
                if not ok:
                    continue
                agg[modality] = {
                    "mean_node_std": float(np.mean(
                        [r[modality]["node_std_mean"] for r in ok])),
                    "mean_cross_sim_std": float(np.mean(
                        [r[modality]["cross_sim_std"] for r in ok])),
                    "mean_oversmooth_ratio": float(np.mean(
                        [r[modality]["oversmooth_ratio"] for r in ok])),
                    "zero_cross_sim_std_pairs": int(sum(
                        1 for r in ok
                        if r[modality]["cross_sim_std"] < 1e-6)),
                    "mean_unique_embeddings": float(np.mean(
                        [r[modality]["unique_embeddings"] for r in ok])),
                    "mean_n_nodes": float(np.mean(
                        [r[modality]["n_nodes"] for r in ok])),
                }
            mode_split = {}
            for mode in ("official_sgf_predicted", "official_oracle"):
                ok = [r for r in rows if r.get("mode") == mode and "gat" in r]
                if ok:
                    mode_split[mode] = {
                        "gat_mean_cross_sim_std": float(np.mean(
                            [r["gat"]["cross_sim_std"] for r in ok])),
                        "gat_zero_std_pairs": int(sum(
                            1 for r in ok
                            if r["gat"]["cross_sim_std"] < 1e-6)),
                        "gat_mean_oversmooth": float(np.mean(
                            [r["gat"]["oversmooth_ratio"] for r in ok])),
                        "pct_mean_cross_sim_std": float(np.mean(
                            [r["pct"]["cross_sim_std"] for r in ok])),
                        "rel_mean_cross_sim_std": float(np.mean(
                            [r["rel"]["cross_sim_std"] for r in ok])),
                        "joint_mean_cross_sim_std": float(np.mean(
                            [r["joint"]["cross_sim_std"] for r in ok])),
                        "errors": sum(1 for r in ok if "error" in r),
                    }
            summary[f"{ckpt_name}/{split}"] = {
                "aggregated": agg,
                "by_mode": mode_split,
                "graph_shape": {
                    "mean_none_edge_share": float(np.mean([
                        r["graph"]["src_none_edges"]
                        / max(r["graph"]["src_edges"], 1)
                        for r in rows if "graph" in r
                    ])) if any("graph" in r for r in rows) else None,
                },
            }
            print(f"{ckpt_name}/{split} done", flush=True)
        del model
        torch.cuda.empty_cache()

    (args.out / "gat_collapse_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    # headline
    for key, v in summary.items():
        for mode, s in v["by_mode"].items():
            print(key, mode, "gat_zero_std:", s["gat_zero_std_pairs"],
                  "gat_cross_std:", round(s["gat_mean_cross_sim_std"], 5),
                  "pct_cross_std:", round(s["pct_mean_cross_sim_std"], 5),
                  "rel_cross_std:", round(s["rel_mean_cross_sim_std"], 5))


if __name__ == "__main__":
    main()
