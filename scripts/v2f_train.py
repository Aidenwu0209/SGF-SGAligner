"""Stage 2/3: controlled training with the CORRECT cross-graph loss.

Strategies (pre-registered before looking at selection results):
  A: official epoch-6 fully frozen (baseline; no training)
  B: freeze PCT, train GAT + relation + fusion (lr 5e-4, cosine, 30 ep)
  C: full model low-lr (2e-5) — only if B diagnostics look sane
Labels: overlap-weighted multi-positive; >=1 positive participates in
descriptor training; <3 positives only skips registration supervision
bookkeeping, never silently drops the pair.  Per-pair label audit
written for all 437 pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from adapters.sgf.data_sources import (  # noqa: E402
    PredictedGraphSource, load_pair_record,
)
from adapters.sgf.graph_adapter import (  # noqa: E402
    adapt_graph, merge_pair_contracts,
)
from adapters.sgf.object_adapter import adapt_objects  # noqa: E402
from adapters.sgf.relation_mapper import RelationMapper  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from safety.cross_graph_loss import cross_graph_infonce  # noqa: E402

OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"


def build_synthetic_samples(n_pairs: int, seed: int = 20260827):
    """Deterministic synthetic pair samples (exact CPU resume tests).

    Same dict structure as the real adapter path (merge_pair_contracts
    output keys consumed by the training loop), so the checkpoint /
    RNG / optimizer / scheduler mechanics are exercised identically.
    Labels use overlap weights in [0.5, 1.0] so the loss has positives.
    """
    rng = np.random.default_rng(seed)
    samples = []

    def complete_edges(n):
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    for k in range(n_pairs):
        n_src = int(rng.integers(4, 9))
        n_ref = int(rng.integers(4, 9))
        pts = rng.normal(0.0, 1.0,
                         size=(n_src + n_ref, 512, 3)).astype(np.float32)
        bow = np.zeros((n_src + n_ref, 41), dtype=np.float32)
        for row in range(n_src + n_ref):
            bow[row, rng.choice(41, size=3, replace=False)] = 1.0
        rel = rng.normal(0.0, 2.0,
                         size=(n_src + n_ref, 3)).astype(np.float32)
        labels = []
        for i in range(min(n_src, n_ref)):
            if rng.random() < 0.8:
                labels.append((1000 + i, 2000 + i,
                               float(rng.random()) * 0.5 + 0.5))
        data_dict = {
            "tot_obj_pts": pts,
            "tot_bow_vec_object_edge_feats": bow,
            "tot_rel_pose": rel,
            "edges": np.concatenate(
                [complete_edges(n_src), complete_edges(n_ref)]),
            "graph_per_obj_count": np.asarray(
                [n_src, n_ref], dtype=np.int64),
            "graph_per_edge_count": np.asarray(
                [n_src * (n_src - 1), n_ref * (n_ref - 1)], dtype=np.int64),
            "src_object_id2idx": {1000 + i: i for i in range(n_src)},
            "ref_object_id2idx": {2000 + i: i for i in range(n_ref)},
            "src_count": n_src,
        }
        samples.append((f"synthetic_{k:03d}", data_dict, labels))
    return samples


def build_labels(pair_id, src_segments, ref_segments):
    """Overlap-weighted multi-positive labels from true surfaces."""
    payload = load_pair_record(pair_id)
    gt = np.asarray(payload["gt_transform"], dtype=np.float64).reshape(4, 4)
    weighted = []
    ref_trees = {
        label: cKDTree(seg) for label, seg in ref_segments.items()
    }
    for s_label, seg_s in src_segments.items():
        moved = seg_s @ gt[:3, :3].T + gt[:3, 3]
        centre = moved.mean(axis=0)
        for r_label, seg_r in ref_segments.items():
            # cheap centroid prefilter, then exact bidirectional NN
            if np.linalg.norm(seg_r.mean(axis=0) - centre) > 3.0:
                continue
            d1 = ref_trees[r_label].query(moved, k=1)[0]
            d2 = cKDTree(moved).query(seg_r, k=1)[0]
            w = float((np.mean(d2 <= 0.10) + np.mean(d1 <= 0.10)) / 2)
            if w >= 0.10:
                weighted.append((s_label, r_label, w))
    return weighted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"),
                        default="B")
    parser.add_argument("--epochs", type=int, default=30,
                        help="TOTAL training horizon (scheduler T_max)")
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="simulate an interruption at this epoch")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda",
                        help="device for the training run (cpu is only "
                        "meaningful together with --synthetic)")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="if > 0: use N deterministic synthetic "
                        "pairs instead of the real training dataset "
                        "(exact CPU resume-equivalence testing)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device
    # stage-2 resume determinism: full seeding + deterministic attempt
    seed = 4242
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        deterministic = "warn_only"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim
        deterministic = f"unavailable: {exc!r}"
    (args.out / "determinism.json").write_text(json.dumps({
        "mode": deterministic,
        "seed": seed,
        "note": ("warn_only: unsupported ops raise a warning naming the "
                 "op; collected in the training log"),
    }, indent=2) + "\n")

    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(OFFICIAL, map_location=device, weights_only=False)
    official = dict(state["model"])
    rows = official.pop("fusion.weight")[:3].clone()
    model.load_state_dict(official, strict=False)
    with torch.no_grad():
        model.fusion.weight.copy_(rows)

    # strategy parameter groups (frozen BEFORE seeing selection)
    lr = {"A": 0.0, "B": 5e-4, "C": 2e-5}[args.strategy]
    # FROZEN POINT PATH (strategy B): object_encoder parameters AND the
    # point projection (object_embedding) stay frozen; the encoder is
    # pinned to eval() so BN running stats/dropout never move.
    frozen_prefixes = ("object_encoder", "object_embedding")
    params = list(model.parameters())
    if args.strategy == "B":
        params = [
            p for name, p in model.named_parameters()
            if not name.startswith(frozen_prefixes)
        ]
        for name, p in model.named_parameters():
            p.requires_grad = not name.startswith(frozen_prefixes)
        model.object_encoder.eval()
    elif args.strategy == "A":
        for p in model.parameters():
            p.requires_grad = False
        params = []

    param_groups = {
        name: {
            "trainable": bool(p.requires_grad),
            "numel": int(p.numel()),
            "lr": lr if p.requires_grad else 0.0,
        }
        for name, p in model.named_parameters()
    }
    (args.out / "parameter_groups.json").write_text(
        json.dumps(param_groups, indent=2) + "\n"
    )

    total_epochs = args.epochs  # horizon is ALWAYS total, even if we
    # stop early: interrupted and continuous runs share T_max
    optimizer = torch.optim.Adam(params, lr=lr) if params else None
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
        if optimizer else None
    )

    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        # fail-closed resume validation
        if ckpt.get("total_epochs") != total_epochs:
            raise RuntimeError(
                f"resume total_epochs {ckpt.get('total_epochs')} != "
                f"{total_epochs}: refusing approximate resume"
            )
        pending_fingerprint_ckpt = ckpt.get("dataset_fingerprint")
        model.load_state_dict(ckpt["model"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng"):
            raw = ckpt["cuda_rng"]
            if isinstance(raw, torch.Tensor):
                raw = [raw]
            states = [
                s.clone().to(torch.uint8).cpu()
                if isinstance(s, torch.Tensor)
                else torch.tensor(
                    bytearray(s) if isinstance(s, (bytes, bytearray))
                    else s, dtype=torch.uint8)
                for s in raw
            ]
            torch.cuda.set_rng_state_all(states)
        print(f"resumed at epoch {ckpt['epoch']}", flush=True)

    if args.synthetic > 0:
        samples = build_synthetic_samples(args.synthetic)
        skipped = []
        label_audit = [
            {"pair_id": pid, "status": "ok",
             "positives": len(labels),
             "weight_mean": (
                 float(np.mean([w for _s, _r, w in labels]))
                 if labels else None),
             "skip_reason": None}
            for pid, _dd, labels in samples
        ]
        dataset_fingerprint = f"{len(samples)}-synthetic-pairs"
    else:
        man = json.loads(Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/training_dataset/"
            "dataset_three_way.json"
        ).read_text())
        pair_root = Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/training_dataset/pairs"
        )
        train_pairs = [
            pair_root / Path(r).parent.name for r in man["train_pairs"]
        ]
        predicted = PredictedGraphSource()
        relation_mapper = RelationMapper()

        samples = []
        label_audit = []
        skipped = []
        for index, pair_dir in enumerate(train_pairs):
            pair_id = pair_dir.name
            entry = {"pair_id": pair_id, "status": "ok",
                     "positives": 0, "weight_mean": None, "skip_reason": None}
            try:
                src_pred = predicted.load(pair_id.split("_to_")[0])
                ref_pred = predicted.load(pair_id.split("_to_")[1])
                labels = build_labels(
                    pair_id, src_pred.segments, ref_pred.segments
                )
                entry["positives"] = len(labels)
                if labels:
                    entry["weight_mean"] = float(
                        np.mean([w for _s, _r, w in labels])
                    )
                if len(labels) < 1:
                    entry["status"] = "skipped"
                    entry["skip_reason"] = "no positives >= 0.10 overlap"
                    skipped.append(pair_id)
                    label_audit.append(entry)
                    continue
                src_obj = adapt_objects(src_pred.segments, seed=42)
                ref_obj = adapt_objects(ref_pred.segments, seed=42)
                src_c = adapt_graph(
                    src_obj, mode="sgf_predicted",
                    directed_pairs=src_pred.directed_pairs,
                    relation_triples=src_pred.relation_triples,
                    relation_mapper=relation_mapper,
                )
                ref_c = adapt_graph(
                    ref_obj, mode="sgf_predicted",
                    directed_pairs=ref_pred.directed_pairs,
                    relation_triples=ref_pred.relation_triples,
                    relation_mapper=relation_mapper,
                )
                center = np.concatenate(
                    [src_c.tot_obj_pts.reshape(-1, 3),
                     ref_c.tot_obj_pts.reshape(-1, 3)]
                ).mean(axis=0)
                data_dict = merge_pair_contracts(src_c, ref_c, center)
                samples.append((pair_id, data_dict, labels))
                label_audit.append(entry)
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                entry["status"] = "skipped"
                entry["skip_reason"] = repr(exc)[:150]
                skipped.append(pair_id)
                label_audit.append(entry)
            if (index + 1) % 100 == 0:
                print(f"built {index+1}/{len(train_pairs)}", flush=True)

        dataset_fingerprint = f"{len(samples)}-pairs"
    if args.resume is not None and pending_fingerprint_ckpt is not None:
        if pending_fingerprint_ckpt != dataset_fingerprint:
            raise RuntimeError(
                "resume dataset fingerprint mismatch: refusing"
            )
    (args.out / "pair_label_audit.json").write_text(
        json.dumps({
            "total": len(label_audit),
            "used": len(samples),
            "skipped": len(skipped),
            "entries": label_audit,
        }, indent=2) + "\n"
    )
    (args.out / "skipped_pairs.json").write_text(
        json.dumps({"count": len(skipped), "pair_ids": skipped}, indent=2)
        + "\n"
    )
    print(f"samples {len(samples)} skipped {len(skipped)}", flush=True)

    if args.strategy == "A" or not samples:
        print("strategy A or no samples: no training", flush=True)
        return

    metrics_path = args.out / "epoch_metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "epoch,loss,pos_sim,neg_sim,margin,"
            "top1_overlap_precision,top5_overlap_precision,"
            "fwd_recall,rev_recall,grad_pct,grad_gat,grad_rel\n"
        )

    def evaluate_diagnostics(model, samples, device):
        model.eval()
        if args.strategy == "B":
            pass  # eval() already; encoder stays eval regardless
        pos_sims, neg_sims = [], []
        top1_hits = top5_hits = top1_total = 0
        fwd_hits = rev_hits = rev_total = queries = 0
        for pair_id, data_dict, labels in samples:
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
                out = model(batch)
            emb = out["joint"].cpu()
            emb = F.normalize(emb, dim=1)
            sim = (emb @ emb.T).numpy()
            src_count = data_dict["src_count"]
            src_map = data_dict["src_object_id2idx"]
            ref_map = data_dict["ref_object_id2idx"]
            pos_set = {}
            for s, r, w in labels:
                if s in src_map and r in ref_map:
                    pos_set.setdefault(src_map[s], set()).add(
                        ref_map[r]  # LOCAL ref index within the block
                    )
            if not pos_set:
                continue
            sim_block = sim[:src_count, src_count:]
            n_ref_local = sim_block.shape[1]
            for i, refs in pos_set.items():
                refs = {j for j in refs if 0 <= j < n_ref_local}
                if not refs:
                    continue
                order = np.argsort(-sim_block[i])
                top1 = order[0]
                top5 = order[:5]
                top1_total += 1
                if int(top1) in refs:
                    top1_hits += 1
                    fwd_hits += 1
                if any(int(t) in refs for t in top5):
                    top5_hits += 1
                queries += 1
                pos_sims.extend(sim_block[i, sorted(refs)])
                neg = [j for j in range(n_ref_local) if j not in refs]
                neg_sims.extend(sim_block[i, neg])
                # reverse recall: for every (src, positive-ref) pair,
                # does src rank first among sources for that ref?
                rev_total_local = 0
                for j in refs:
                    rorder = np.argsort(-sim_block[:, j])
                    rev_total += 1
                    if int(rorder[0]) == i:
                        rev_hits += 1
        return {
            "pos_sim": float(np.mean(pos_sims)),
            "neg_sim": float(np.mean(neg_sims)),
            "margin": float(np.mean(pos_sims) - np.mean(neg_sims)),
            "top1": top1_hits / max(top1_total, 1),
            "top5": top5_hits / max(top1_total, 1),
            "fwd_recall": fwd_hits / max(queries, 1),
            "rev_recall": (
                rev_hits / rev_total if rev_total else 0.0
            ),
        }

    def frozen_snapshot():
        snap = {}
        for name, t in list(model.named_parameters()) + list(
            model.named_buffers()
        ):
            if name.startswith(frozen_prefixes):
                snap[name] = t.detach().clone()
        return snap

    frozen_before = frozen_snapshot()

    final_epoch_limit = (
        args.stop_after_epoch
        if args.stop_after_epoch is not None else total_epochs
    )
    for epoch in range(start_epoch, final_epoch_limit + 1):
        rng = np.random.default_rng(91000 + epoch)
        order = rng.permutation(len(samples))
        g_pct_acc = g_gat_acc = g_rel_acc = 0.0
        model.train()
        if args.strategy == "B":
            model.object_encoder.eval()  # keep BN/dropout frozen
        losses = []
        for idx in order.tolist():
            pair_id, data_dict, labels = samples[idx]
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
            output = model(batch)
            emb = output["joint"]
            src_count = data_dict["src_count"]
            src_emb = emb[:src_count]
            ref_emb = emb[src_count:]
            n_src, n_ref = src_emb.shape[0], ref_emb.shape[0]
            src_map = data_dict["src_object_id2idx"]
            ref_map = data_dict["ref_object_id2idx"]
            positives = torch.zeros(n_src, n_ref, dtype=torch.bool,
                                    device=device)
            weights = torch.zeros(n_src, n_ref, device=device)
            for s, r, w in labels:
                if s in src_map and r in ref_map:
                    si = src_map[s]
                    rj = ref_map[r]
                    positives[si, rj] = True
                    weights[si, rj] = w
            try:
                loss, _diag = cross_graph_infonce(
                    src_emb, ref_emb, positives, weights
                )
            except ValueError:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                n = float(p.grad.norm())
                if name.startswith("object_encoder"):
                    g_pct_acc += n * n
                elif name.startswith("structure_encoder"):
                    g_gat_acc += n * n
                elif name.startswith("meta_embedding_rel"):
                    g_rel_acc += n * n
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()

        diag = evaluate_diagnostics(model, samples, device)
        current_lr = (
            optimizer.param_groups[0]["lr"] if optimizer else float("nan")
        )
        (args.out / f"lr_epoch_{epoch:05d}").write_text(
            f"{current_lr!r}\n"
        )
        metrics_path.open("a").write(
            f"{epoch},{np.mean(losses) if losses else float('nan'):.5f},"
            f"{diag['pos_sim']:.5f},{diag['neg_sim']:.5f},"
            f"{diag['margin']:.5f},{diag['top1']:.5f},"
            f"{diag['top5']:.5f},{diag['fwd_recall']:.5f},"
            f"{diag['rev_recall']:.5f},{np.sqrt(g_pct_acc):.4f},"
            f"{np.sqrt(g_gat_acc):.4f},{np.sqrt(g_rel_acc):.4f}\n"
        )
        print(
            f"epoch {epoch}: loss {np.mean(losses):.4f} "
            f"pos {diag['pos_sim']:.4f} neg {diag['neg_sim']:.4f} "
            f"margin {diag['margin']:.4f} top1 {diag['top1']:.4f} "
            f"top5 {diag['top5']:.4f}", flush=True,
        )

        drift = {
            name: float((t.detach() - frozen_before[name]).abs().max())
            for name, t in list(model.named_parameters())
            + list(model.named_buffers())
            if name in frozen_before
        }
        max_drift = max(drift.values()) if drift else 0.0
        if max_drift != 0.0:
            raise RuntimeError(
                f"frozen point path drifted by {max_drift}"
            )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "total_epochs": total_epochs,
            "training_config": {"strategy": args.strategy, "lr": lr},
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
            "dataset_fingerprint": dataset_fingerprint,
            "history": metrics_path.read_text(),
        }
        tmp = args.out / f".epoch_{epoch:05d}.pt.tmp"
        torch.save(ckpt, tmp)
        tmp.replace(args.out / f"epoch_{epoch:05d}.pt")
    torch.save(ckpt, args.out / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
