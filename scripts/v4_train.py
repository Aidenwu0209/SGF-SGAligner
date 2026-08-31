"""V4 Healthy-GAT research training (arms B complete / C explicit).

Pre-registered protocol (outputs/official_sgaligner_v4_healthy_gat_
20260827/protocol.md, commit 29824f8) — core rules frozen before
training:

- PCT fully frozen (object_encoder params + ALL BN buffers +
  num_batches_tracked, object_embedding, always eval());
- relation frozen (meta_embedding_rel);
- structure_encoder + structure_embedding re-initialised with seed
  20260827 (IDENTICAL initial state for arms B and C);
- trainable = structure_encoder.*, structure_embedding.*, fusion row
  1 (GAT) only — rows 0 (pct) and 2 (rel) gradient-masked;
- inputs use official_mt19937 sampling; arm B consumes official
  complete-none edges, arm C consumes SGF native explicit edges (the
  ONLY difference);
- loss = the verified cross-graph-only bidirectional overlap-weighted
  multi-positive InfoNCE;
- exact-resume protocol proven in V2T-Fix3-Seal (total-horizon
  T_max, fail-closed, full RNG state).

The resulting model is an "official-architecture SGF-predicted
healthy-GAT research candidate" — never an official checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get(
    "SGALIGNER_CODE_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from adapters.sgf.data_sources import (  # noqa: E402
    PredictedGraphSource, load_pair_record, load_anchor_ids,
)
from adapters.sgf.object_adapter import adapt_objects  # noqa: E402
from adapters.sgf.graph_adapter import (  # noqa: E402
    adapt_graph, merge_pair_contracts,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from safety.cross_graph_loss import cross_graph_infonce  # noqa: E402

OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
OUT = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"

SEED = 4242
INIT_SEED = 20260827
MAX_EPOCHS = 60
EVAL_INTERVAL = 5
PATIENCE_EVALS = 6
LR = 5e-4
GRAD_CLIP = 5.0
MIN_POSITIVE_WEIGHT = 0.10

FROZEN_PREFIXES = (
    "object_encoder", "object_embedding", "meta_embedding_rel",
)
TRAINABLE_PREFIXES = ("structure_encoder", "structure_embedding")


def sha256_of_tensor_bytes(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def build_labels(pair_id, src_segments, ref_segments):
    from scipy.spatial import cKDTree

    payload = load_pair_record(pair_id)
    gt = np.asarray(payload["gt_transform"], dtype=np.float64).reshape(4, 4)
    weighted = []
    ref_trees = {label: cKDTree(seg) for label, seg in ref_segments.items()}
    for s_label, seg_s in src_segments.items():
        moved = seg_s @ gt[:3, :3].T + gt[:3, 3]
        centre = moved.mean(axis=0)
        for r_label, seg_r in ref_segments.items():
            if np.linalg.norm(seg_r.mean(axis=0) - centre) > 3.0:
                continue
            d1 = ref_trees[r_label].query(moved, k=1)[0]
            d2 = cKDTree(moved).query(seg_r, k=1)[0]
            w = float((np.mean(d2 <= 0.10) + np.mean(d1 <= 0.10)) / 2)
            if w >= MIN_POSITIVE_WEIGHT:
                weighted.append((s_label, r_label, w))
    return weighted


def explicit_edges_for(pairs, object_id2idx):
    seen = []
    for sub, obj in pairs:
        if (sub, obj) in seen:
            continue
        if sub in object_id2idx and obj in object_id2idx:
            seen.append((sub, obj))
    return np.asarray(
        [(object_id2idx[s], object_id2idx[o]) for s, o in seen],
        dtype=np.int64).reshape(-1, 2)


def build_split_samples(pairs, predicted):
    """One shared sample build for both arms: complete-none contract
    tensors PLUS the explicit-only edge variant per graph."""
    samples = []
    skipped = []
    for index, pair_id in enumerate(pairs):
        try:
            src_scan, ref_scan = pair_id.split("_to_")
            src_pred = predicted.load(src_scan)
            ref_pred = predicted.load(ref_scan)
            labels = build_labels(
                pair_id, src_pred.segments, ref_pred.segments)
            if not labels:
                skipped.append(pair_id)
                continue
            contracts = []
            explicit = []
            for pred in (src_pred, ref_pred):
                objects = adapt_objects(
                    pred.segments, sampling_mode="official_mt19937",
                    scan_seed=0,
                )
                contract = adapt_graph(
                    objects, mode="sgf_predicted",
                    directed_pairs=pred.directed_pairs,
                    relation_triples=pred.relation_triples,
                )
                contract.scene_ids = [src_scan if pred is src_pred
                                      else ref_scan]
                contracts.append(contract)
                explicit.append(explicit_edges_for(
                    pred.directed_pairs, objects.object_id2idx))
            center = np.concatenate(
                [contracts[0].tot_obj_pts.reshape(-1, 3),
                 contracts[1].tot_obj_pts.reshape(-1, 3)]
            ).mean(axis=0)
            data_dict = merge_pair_contracts(
                contracts[0], contracts[1], center)
            # per-graph LOCAL edge indices (no offset): the
            # official forward slices merged edges per graph
            data_dict["edges_explicit"] = np.concatenate([
                explicit[0],
                explicit[1]]) if (
                len(explicit[0]) or len(explicit[1])) else \
                np.zeros((0, 2), dtype=np.int64)
            data_dict["graph_per_edge_count_explicit"] = np.asarray(
                [len(explicit[0]), len(explicit[1])], dtype=np.int64)
            samples.append((pair_id, data_dict, labels))
        except Exception:  # noqa: BLE001 - recorded, never silent
            skipped.append(pair_id)
        if (index + 1) % 100 == 0:
            print(f"built {index+1}/{len(pairs)}", flush=True)
    return samples, skipped


def batch_for(data_dict, arm, device):
    edges = (data_dict["edges"] if arm == "complete"
             else data_dict["edges_explicit"])
    counts = (data_dict["graph_per_edge_count"]
              if arm == "complete"
              else data_dict["graph_per_edge_count_explicit"])
    return {
        "tot_obj_pts": torch.from_numpy(
            data_dict["tot_obj_pts"]).to(device),
        "tot_bow_vec_object_edge_feats": torch.from_numpy(
            data_dict["tot_bow_vec_object_edge_feats"]).to(device),
        "tot_rel_pose": torch.from_numpy(
            data_dict["tot_rel_pose"]).to(device),
        "edges": torch.from_numpy(
            np.asarray(edges).astype(np.int64)).to(device),
        "graph_per_obj_count": [np.asarray(
            data_dict["graph_per_obj_count"], dtype=np.int64)],
        "graph_per_edge_count": [np.asarray(
            counts, dtype=np.int64)],
        "batch_size": 1,
        "tot_bow_vec_object_attr_feats": torch.zeros(
            (data_dict["tot_obj_pts"].shape[0], 164)).to(device),
    }


def build_model(device):
    """Official-architecture model with healthy-GAT re-init."""
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(OFFICIAL, map_location=device, weights_only=False)
    official = dict(state["model"])
    fusion3 = official.pop("fusion.weight")[:3].clone()
    model.load_state_dict(official, strict=False)
    with torch.no_grad():
        model.fusion.weight.copy_(fusion3)
    # --- healthy re-initialisation (identical for arms B and C) ----
    torch.manual_seed(INIT_SEED)
    for module in (model.structure_encoder, model.structure_embedding):
        for layer in module.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    # --- freezes ---------------------------------------------------
    for name, p in model.named_parameters():
        p.requires_grad = bool(
            name.startswith(TRAINABLE_PREFIXES)
            or name == "fusion.weight")
    model.object_encoder.eval()
    model.meta_embedding_rel.requires_grad_(False)
    return model


def fusion_grad_mask(model):
    """Zero gradients of the frozen fusion rows (pct, rel)."""
    w = model.fusion.weight.grad
    if w is not None:
        w[0] = 0
        w[2] = 0


def frozen_snapshot(model):
    snap = {}
    for name, t in list(model.named_parameters()) + list(
            model.named_buffers()):
        if name.startswith(FROZEN_PREFIXES):
            snap[name] = t.detach().clone()
    # fusion rows 0 and 2 are frozen values inside a trainable tensor
    snap["fusion.row0"] = model.fusion.weight.data[0].clone()
    snap["fusion.row2"] = model.fusion.weight.data[2].clone()
    return snap


def gat_health(model, data_dict, arm, device):
    """Non-constant / input-sensitivity / shuffle-sensitivity probes."""
    model.eval()
    with torch.no_grad():
        batch = batch_for(data_dict, arm, device)
        h_full = model(batch)["gat"].cpu().numpy()
    n_nodes = h_full.shape[0]
    rounded = np.round(h_full, 4)
    unique = len(np.unique(rounded, axis=0))
    # edge shuffle sensitivity on the full joint graph input
    with torch.no_grad():
        edges = np.asarray(
            data_dict["edges"] if arm == "complete"
            else data_dict["edges_explicit"]).astype(np.int64)
        counts = np.asarray(
            data_dict["graph_per_edge_count"] if arm == "complete"
            else data_dict["graph_per_edge_count_explicit"])
        if len(edges):
            # shuffle WITHIN each graph — the official forward slices
            # the merged edge list per graph, so a global row shuffle
            # would feed ref-local indices to the src-graph GAT
            c0 = int(counts[0])
            rng = np.random.default_rng(777)
            shuffled = np.concatenate([
                edges[:c0][rng.permutation(c0)],
                edges[c0:][rng.permutation(len(edges) - c0)],
            ])
            batch2 = dict(batch)
            batch2["edges"] = torch.from_numpy(shuffled).to(device)
            h2 = model(batch2)["gat"].cpu().numpy()
            shuffle_delta = float(np.abs(h_full - h2).max())
        else:
            shuffle_delta = 0.0
    gat_params = torch.cat([
        p.detach().flatten() for n, p in model.named_parameters()
        if n.startswith("structure_encoder")])
    subnormal_fraction = float(
        ((gat_params.abs() > 0)
         & (gat_params.abs() < torch.finfo(torch.float32).tiny)
         ).float().mean())
    return {
        "unique_node_embeddings": int(unique),
        "n_nodes": int(n_nodes),
        "non_constant": unique > 1,
        "edge_shuffle_delta": shuffle_delta,
        "gat_param_norm": float(gat_params.norm()),
        "subnormal_fraction": subnormal_fraction,
        "embedding_std": float(h_full.std()),
    }


def evaluate(model, samples, arm, device, anchors_by_pair):
    """Deterministic selection metrics (embedding level only)."""
    model.eval()
    f1s, top1s, top5s = [], [], []
    margins = []
    tp_all = pred_all = anchor_all = 0
    for pair_id, data_dict, _labels in samples:
        with torch.no_grad():
            batch = batch_for(data_dict, arm, device)
            emb = model(batch)["joint"].cpu().numpy()
        emb = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = emb @ emb.T
        src_count = data_dict["src_count"]
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        anchors = anchors_by_pair.get(pair_id, set())
        anchor_idx = {
            (src_map[s], ref_map[r] + src_count)
            for s, r in anchors if s in src_map and r in ref_map}
        rank = np.argsort(-sim, axis=1)
        pred = set()
        for i in range(src_count):
            refs = [x for x in rank[i] if x >= src_count][:3]
            for r in refs:
                pred.add((i, int(r)))
        tp = len(pred & anchor_idx)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(anchor_idx) if anchor_idx else 0.0
        f1s.append(2 * p * r / max(p + r, 1e-12))
        tp_all += tp
        pred_all += len(pred)
        anchor_all += len(anchor_idx)
        # top1 / top5 / margin
        pos_sims, neg_sims = [], []
        for i in range(src_count):
            refs = [x for x in rank[i] if x >= src_count][:5]
            if not refs:
                continue
            if (i, int(refs[0])) in anchor_idx:
                top1s.append(1.0)
            else:
                top1s.append(0.0)
            for x in refs:
                if (i, int(x)) in anchor_idx:
                    pos_sims.append(float(sim[i, x]))
                else:
                    neg_sims.append(float(sim[i, x]))
        top5s.append(sum(
            1 for a in anchor_idx
            if int(a[1]) in [
                int(x) for x in rank[a[0]]
                if x >= src_count][:5]) / max(len(anchor_idx), 1))
        if pos_sims and neg_sims:
            margins.append(
                float(np.mean(pos_sims) - np.mean(neg_sims)))
    micro_p = tp_all / pred_all if pred_all else 0.0
    micro_r = tp_all / anchor_all if anchor_all else 0.0
    return {
        "macro_node_f1": float(np.mean(f1s)),
        "top1_precision": float(np.mean(top1s)),
        "top5_recall": float(np.mean(top5s)),
        "margin": float(np.mean(margins)),
        "micro_node_f1": 2 * micro_p * micro_r / max(
            micro_p + micro_r, 1e-12),
    }


def selection_key(metrics):
    """Pre-registered lexicographic key (higher better, epoch asc)."""
    return (metrics["macro_node_f1"], metrics["top1_precision"],
            metrics["top5_recall"], metrics["margin"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("complete", "explicit"),
                        required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override output dir (smoke tests only)")
    args = parser.parse_args()

    out_root = args.out_dir if args.out_dir else OUT
    arm_dir = out_root / "training" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass

    model = build_model(device)

    predicted = PredictedGraphSource()
    man = json.loads(Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/"
        "dataset_three_way.json"
    ).read_text())
    pair_root = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/pairs"
    )
    train_pairs = [pair_root / Path(r).parent.name
                   for r in man["train_pairs"]]

    # ---- selection89 eval tensors (built once, embedding-level) ----
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    sel_pairs = [l.strip() for l in
                 (pairlists / "selection.txt").read_text().splitlines()
                 if l.strip()]
    print("building selection89 tensors...", flush=True)
    sel_samples, _ = build_split_samples(sel_pairs, predicted)
    anchors_by_pair = {
        pid: set(load_anchor_ids(pid))
        for pid, _dd, _lb in sel_samples
    }

    print("building train437 samples...", flush=True)
    samples, skipped = build_split_samples(
        [p.name for p in train_pairs], predicted)
    (arm_dir / "dataset_build.json").write_text(json.dumps({
        "train_pairs": len(train_pairs), "used": len(samples),
        "skipped": len(skipped), "skipped_ids": skipped[:20],
        "sampling_mode": "official_mt19937",
    }, indent=2) + "\n")
    dataset_fingerprint = f"{len(samples)}-{args.arm}-mt19937-pairs"

    # ---- initialization audits ------------------------------------
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS)
    frozen_before = frozen_snapshot(model)
    health0 = gat_health(model, samples[0][1], args.arm, device)
    init_audit = {
        "arm": args.arm,
        "init_seed": INIT_SEED,
        "gat_health": health0,
        "trainable_param_groups": {
            name: {
                "trainable": bool(p.requires_grad),
                "numel": int(p.numel()),
            } for name, p in model.named_parameters()
        },
        "frozen_tensor_hashes": {
            k: sha256_of_tensor_bytes(v)
            for k, v in frozen_before.items()},
    }
    (out_root / f"initialization_audit_{args.arm}.json").write_text(
        json.dumps(init_audit, indent=2) + "\n")
    if not health0["non_constant"] or health0["subnormal_fraction"] > 0:
        raise RuntimeError(f"unhealthy GAT init: {health0}")
    opt_names = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(
        id(p) not in opt_names
        for n, p in model.named_parameters()
        if n.startswith(("object_encoder", "object_embedding",
                         "meta_embedding_rel"))), \
        "optimizer must not contain PCT/relation parameters"

    # ---- resume ----------------------------------------------------
    start_epoch = 1
    pending_fp = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        if ckpt.get("total_epochs") != MAX_EPOCHS:
            raise RuntimeError("resume total_epochs mismatch: refusing")
        pending_fp = ckpt.get("dataset_fingerprint")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"resumed at epoch {ckpt['epoch']}", flush=True)
    if pending_fp is not None and pending_fp != dataset_fingerprint:
        raise RuntimeError("resume fingerprint mismatch: refusing")

    history_path = arm_dir / "history.jsonl"
    best_key = None
    best_epoch = 0
    bad_evals = 0
    if args.resume is not None and history_path.exists():
        for line in history_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "eval":
                    key = selection_key(rec["metrics"])
                    if best_key is None or key > best_key:
                        best_key, best_epoch = key, rec["epoch"]
                    bad_evals = rec.get("bad_evals_after", 0)

    metrics_csv = arm_dir / "epoch_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text(
            "epoch,loss,pos_sim,neg_sim,margin,grad_gat,"
            "gat_param_norm,subnormal_frac,unique_emb,shuffle_delta,"
            "emb_std,frozen_ok,lr,gpu_peak_mb,epoch_s\n")

    final_limit = (args.stop_after_epoch
                   if args.stop_after_epoch is not None
                   else args.epochs)
    stop_reason = "max_epochs"
    ckpt = None
    for epoch in range(start_epoch, final_limit + 1):
        t0 = time.monotonic()
        rng = np.random.default_rng(91000 + epoch)
        order = rng.permutation(len(samples))
        model.train()
        model.object_encoder.eval()
        model.meta_embedding_rel.eval()
        losses, pos_sims, neg_sims = [], [], []
        grad_sq = 0.0
        for idx in order.tolist():
            pair_id, data_dict, labels = samples[idx]
            batch = batch_for(data_dict, args.arm, device)
            out = model(batch)
            emb = out["joint"]
            src_count = data_dict["src_count"]
            n_src, n_ref = src_count, emb.shape[0] - src_count
            src_map = data_dict["src_object_id2idx"]
            ref_map = data_dict["ref_object_id2idx"]
            positives = torch.zeros(
                n_src, n_ref, dtype=torch.bool, device=device)
            weights = torch.zeros(n_src, n_ref, device=device)
            for s, r, w in labels:
                if s in src_map and r in ref_map:
                    positives[src_map[s], ref_map[r]] = True
                    weights[src_map[s], ref_map[r]] = w
            if not positives.any():
                continue
            loss, _diag = cross_graph_infonce(
                emb[:src_count], emb[src_count:], positives, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            fusion_grad_mask(model)
            for n, p in model.named_parameters():
                if p.grad is not None and n.startswith(
                        "structure_encoder"):
                    grad_sq += float(p.grad.norm()) ** 2
            torch.nn.utils.clip_grad_norm_(
                [p for p in optimizer.param_groups[0]["params"]],
                GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss))
            with torch.no_grad():
                e = F.normalize(emb, dim=1).cpu().numpy()
                sim = e[:src_count] @ e[src_count:].T
                pm = positives.cpu().numpy()
                if pm.any():
                    pos_sims.append(float(sim[pm].mean()))
                    neg_sims.append(float(sim[~pm].mean()))
        scheduler.step()

        # ---- frozen integrity guard ------------------------------
        frozen_now = frozen_snapshot(model)
        frozen_ok = all(
            torch.equal(frozen_now[k], frozen_before[k])
            for k in frozen_before)
        if not frozen_ok:
            raise RuntimeError("frozen tensors drifted: aborting")

        health = gat_health(model, samples[0][1], args.arm, device)
        gpu_peak = (torch.cuda.max_memory_allocated()
                    if device == "cuda" else 0.0)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lr_now = optimizer.param_groups[0]["lr"]
        row = [
            epoch,
            float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(pos_sims)) if pos_sims else float("nan"),
            float(np.mean(neg_sims)) if neg_sims else float("nan"),
            (float(np.mean(pos_sims)) - float(np.mean(neg_sims)))
            if pos_sims and neg_sims else float("nan"),
            float(np.sqrt(grad_sq)),
            health["gat_param_norm"],
            health["subnormal_fraction"],
            health["unique_node_embeddings"],
            health["edge_shuffle_delta"],
            health["embedding_std"],
            int(frozen_ok),
            lr_now,
            gpu_peak / 1e6,
            time.monotonic() - t0,
        ]
        metrics_csv.open("a").write(
            ",".join(str(x) for x in row) + "\n")
        # immediate-fail conditions
        if health["subnormal_fraction"] > 0:
            raise RuntimeError(f"GAT subnormal at epoch {epoch}")
        if health["unique_node_embeddings"] <= 1:
            raise RuntimeError(f"GAT collapsed at epoch {epoch}")
        if not np.isfinite(row[1]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        print(
            f"epoch {epoch}: loss {row[1]:.4f} "
            f"margin {row[4]:.4f} gatNorm {row[6]:.3f} "
            f"uniq {row[8]} frozen_ok {frozen_ok}", flush=True)

        # ---- periodic selection eval + checkpoint ------------------
        if epoch % EVAL_INTERVAL == 0 or epoch == final_limit:
            metrics = evaluate(
                model, sel_samples, args.arm, device, anchors_by_pair)
            key = selection_key(metrics)
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                bad_evals = 0
            else:
                bad_evals += 1
            with history_path.open("a") as fh:
                fh.write(json.dumps({
                    "kind": "eval", "epoch": epoch,
                    "metrics": metrics,
                    "bad_evals_after": bad_evals,
                }) + "\n")
            print(f"  eval@{epoch}: {json.dumps(metrics)} "
                  f"best_epoch={best_epoch} bad={bad_evals}",
                  flush=True)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "next_epoch": epoch + 1,
                "total_epochs": MAX_EPOCHS,
                "training_config": {
                    "arm": args.arm, "lr": LR, "seed": SEED,
                    "init_seed": INIT_SEED,
                    "model_naming": (
                        "official-architecture SGF-predicted "
                        "healthy-GAT research candidate"),
                },
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
                "dataset_fingerprint": dataset_fingerprint,
                "history": metrics_csv.read_text(),
                "selection_metrics": metrics,
            }
            tmp = arm_dir / f".epoch_{epoch:05d}.pt.tmp"
            torch.save(ckpt, tmp)
            tmp.replace(arm_dir / f"epoch_{epoch:05d}.pt")
            if bad_evals >= PATIENCE_EVALS:
                stop_reason = (
                    f"early_stop_patience_{PATIENCE_EVALS}_evals")
                print(stop_reason, flush=True)
                break
    if ckpt is not None:
        torch.save(ckpt, arm_dir / "last.pt")
    (arm_dir / "run_summary.json").write_text(json.dumps({
        "arm": args.arm, "stop_reason": stop_reason,
        "epochs_run": final_limit,
        "best_epoch": best_epoch,
        "best_key": list(best_key) if best_key else None,
        "dataset_fingerprint": dataset_fingerprint,
    }, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
