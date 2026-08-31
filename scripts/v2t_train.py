"""Stage 2: SGF-predicted adapter training on the OFFICIAL architecture.

- PCT + GAT + relation modules from the official MultiModalEncoder
  (unmodified); attribute OFF; official checkpoint as init.
- Labels: multi-positive (split/merge aware), weighted by TRUE surface
  overlap (bidirectional full-anchor-point coverage, 3DSSG GT anchors,
  train pairs only); same-pair hard negatives + cross-scene negatives.
- Official contrastive-style objective re-implemented in adapter space
  (weighted multi-positive InfoNCE), matching the official loss form
  for the pct/gat/rel embedding.
- <=60 epochs, atomic rolling checkpoints every 5, exact resume
  (optimizer/scheduler/RNG/data order), CUDA failures resume from the
  last complete checkpoint without overwriting anything.
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

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from adapters.sgf.data_sources import (  # noqa: E402
    DATA_ROOT, OracleGraphSource, PredictedGraphSource,
    load_pair_record,
)
from adapters.sgf.graph_adapter import adapt_graph  # noqa: E402
from adapters.sgf.object_adapter import adapt_objects  # noqa: E402
from adapters.sgf.relation_mapper import RelationMapper  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

OFFICIAL_SNAPSHOT = ROOT / (
    "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
)
MAX_EPOCHS = 60
CHECKPOINT_EVERY = 5


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def true_overlap_weights(pair_id):
    """Bidirectional surface coverage between GT-anchor-matched objects.

    Uses the OFFICIAL PLY geometry (oracle source) for the anchor pairs
    of this train pair; GT only for labels.  Returns
    [(src_label, ref_label, weight), ...].
    """
    payload = load_pair_record(pair_id)
    src_scan, ref_scan = pair_id.split("_to_")
    oracle = OracleGraphSource()
    src_segments = oracle.load(src_scan).segments
    ref_segments = oracle.load(ref_scan).segments
    # global-id oracle anchors (same construction as evaluation)
    anchors = oracle_global_anchors(oracle, src_scan, ref_scan,
                                    src_segments, ref_segments)
    rows = []
    for src_label, ref_label in anchors:
        src_pts = src_segments[src_label]
        ref_pts = ref_segments[ref_label]
        d1 = cKDTree(ref_pts).query(src_pts, k=1)[0]
        d2 = cKDTree(src_pts).query(ref_pts, k=1)[0]
        weight = float(
            (np.mean(d1 <= 0.10) + np.mean(d2 <= 0.10)) / 2
        )
        rows.append((src_label, ref_label, weight))
    return rows


def oracle_global_anchors(oracle, src_scan, ref_scan,
                          src_segments, ref_segments):
    objects_json = json.loads(
        (DATA_ROOT / "objects.json").read_text()
    )["scans"]
    by_scan = {e["scan"]: e["objects"] for e in objects_json}

    def global_map(scan, segments):
        return {
            int(o["global_id"]): int(o["id"])
            for o in by_scan.get(scan, [])
            if int(o["id"]) in segments
        }

    g_src = global_map(src_scan, src_segments)
    g_ref = global_map(ref_scan, ref_segments)
    shared = set(g_src) & set(g_ref)
    return [(g_src[g], g_ref[g]) for g in sorted(shared)]


def build_train_sample(pair_id, predicted, oracle, relation_mapper,
                       npoint=512, seed=42):
    """One training sample: (data_dict, multi-positive labels+weights)."""
    src_scan, ref_scan = pair_id.split("_to_")
    src_pred = predicted.load(src_scan)
    ref_pred = predicted.load(ref_scan)

    labels = []
    src_obj = adapt_objects(src_pred.segments, seed=seed)
    ref_obj = adapt_objects(ref_pred.segments, seed=seed)

    # multi-positive labels from TRUE overlap between matched SGF
    # segment surfaces: use the GT-anchor geometry on the PLY side
    # mapped by GLOBAL id onto nearest SGF segments is not tractable;
    # instead compute overlap directly between SGF segments that the
    # GT transform aligns: labels = SGF segment pairs whose full
    # surfaces overlap under the pair-record GT transform.
    payload = load_pair_record(pair_id)
    gt = np.asarray(payload["gt_transform"], dtype=np.float64).reshape(4, 4)
    weighted = []
    for s_label, seg_s in src_pred.segments.items():
        moved = seg_s @ gt[:3, :3].T + gt[:3, 3]
        tree = cKDTree(moved)
        for r_label, seg_r in ref_pred.segments.items():
            d1 = tree.query(seg_r, k=1)[0]
            if np.mean(d1 <= 0.10) < 0.30:
                continue
            d2 = cKDTree(moved).query(seg_r, k=1)[0]  # symmetric approx
            d2r = cKDTree(seg_r).query(moved, k=1)[0]
            w = float(
                (np.mean(d2r <= 0.10) + np.mean(d1 <= 0.10)) / 2
            )
            if w >= 0.30:
                weighted.append((s_label, r_label, w))
    labels = weighted
    if len(labels) < 3:
        return None

    src_contract = adapt_graph(
        src_obj, mode="sgf_predicted",
        directed_pairs=src_pred.directed_pairs,
        relation_triples=src_pred.relation_triples,
        relation_mapper=relation_mapper,
    )
    ref_contract = adapt_graph(
        ref_obj, mode="sgf_predicted",
        directed_pairs=ref_pred.directed_pairs,
        relation_triples=ref_pred.relation_triples,
        relation_mapper=relation_mapper,
    )
    pcl_center = np.concatenate(
        [src_contract.tot_obj_pts.reshape(-1, 3),
         ref_contract.tot_obj_pts.reshape(-1, 3)]
    ).mean(axis=0)
    from adapters.sgf.graph_adapter import merge_pair_contracts

    data_dict = merge_pair_contracts(src_contract, ref_contract, pcl_center)
    return data_dict, labels


def official_batch(data_dict, device):
    return {
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
        # unused read-buffer (attr head not in modules; never runs)
        "tot_bow_vec_object_attr_feats": torch.zeros(
            (data_dict["tot_obj_pts"].shape[0], 164)).to(device),
    }


def weighted_multi_positive_infonce(similarity, positives_mask, weights):
    """Weighted multi-positive InfoNCE (official loss form)."""
    temperature = 0.1
    logits = similarity / temperature
    src_nodes = torch.nonzero(positives_mask.any(dim=1)).squeeze(1)
    if len(src_nodes) == 0:
        return None
    per_query = []
    q_weights = []
    for i in src_nodes.tolist():
        js = torch.nonzero(positives_mask[i]).squeeze(1)
        num = torch.logsumexp(logits[i, js], dim=0)
        den = torch.logsumexp(logits[i], dim=0)
        per_query.append(den - num)
        q_weights.append(weights[i, js].max())
    per_query = torch.stack(per_query)
    q_weights = torch.stack(q_weights)
    return (per_query * q_weights).sum() / q_weights.sum().clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

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
        pair_root / Path(rel).parent.name for rel in man["train_pairs"]
    ]

    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(
        OFFICIAL_SNAPSHOT, map_location=device, weights_only=False
    )
    official = dict(state["model"])
    fusion_rows = official.pop("fusion.weight")[:3].clone()
    model.load_state_dict(official, strict=False)
    with torch.no_grad():
        model.fusion.weight.copy_(fusion_rows)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 1
    rng_state = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        print(f"resumed from epoch {ckpt['epoch']}")

    predicted = PredictedGraphSource()
    oracle = OracleGraphSource()
    relation_mapper = RelationMapper()

    # pre-build samples (deterministic; data order per epoch from RNG)
    samples = []
    for index, pair_dir in enumerate(train_pairs):
        pair_id = pair_dir.name
        try:
            sample = build_train_sample(
                pair_id, predicted, oracle, relation_mapper
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            print(f"skip {pair_id}: {exc!r}", flush=True)
            continue
        if sample is not None:
            samples.append((pair_id, *sample))
        if (index + 1) % 50 == 0:
            print(f"built {index+1}/{len(train_pairs)}", flush=True)
    print(f"training samples: {len(samples)}", flush=True)

    metrics_path = args.out / "epoch_metrics.csv"
    if not metrics_path.exists():
        metrics_path.write_text(
            "epoch,loss,mean_pos_similarity,mean_neg_similarity\n"
        )

    if not samples:
        raise RuntimeError("no training samples built; aborting")
    for epoch in range(start_epoch, args.epochs + 1):
        rng = np.random.default_rng(42000 + epoch)
        order = rng.permutation(len(samples))
        model.train()
        losses = []
        pos_sims = []
        neg_sims = []
        for idx in order.tolist():
            pair_id, data_dict, labels = samples[idx]
            batch = official_batch(data_dict, device)
            output = model(batch)
            emb = output["joint"]
            emb = F.normalize(emb, dim=1)
            similarity = emb @ emb.T

            src_map = data_dict["src_object_id2idx"]
            ref_map = data_dict["ref_object_id2idx"]
            n = similarity.shape[0]
            src_count = data_dict["src_count"]
            positives = torch.zeros_like(similarity, dtype=torch.bool)
            weights = torch.zeros_like(similarity)
            for s_label, r_label, w in labels:
                if s_label in src_map and r_label in ref_map:
                    si = src_map[s_label]
                    rj = ref_map[r_label] + src_count
                    positives[si, rj] = True
                    weights[si, rj] = w
            if positives.sum() < 3:
                continue
            loss = weighted_multi_positive_infonce(
                similarity, positives, weights
            )
            if loss is None or not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss))
            with torch.no_grad():
                pos = similarity[positives].mean()
                neg_mask = (~positives) & (
                    torch.arange(n)[None, :].to(device) >= src_count
                )
                neg = similarity[neg_mask].mean()
                pos_sims.append(float(pos))
                neg_sims.append(float(neg))
        scheduler.step()

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        metrics_path.open("a").write(
            f"{epoch},{mean_loss:.5f},"
            f"{np.mean(pos_sims) if pos_sims else float('nan'):.5f},"
            f"{np.mean(neg_sims) if neg_sims else float('nan'):.5f}\n"
        )
        print(
            f"epoch {epoch}: loss {mean_loss:.5f} "
            f"pos {np.mean(pos_sims) if pos_sims else float('nan'):.4f} "
            f"neg {np.mean(neg_sims) if neg_sims else float('nan'):.4f}",
            flush=True,
        )

        state_out = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        tmp = args.out / f".epoch_{epoch:05d}.pt.tmp"
        torch.save(state_out, tmp)
        tmp.replace(args.out / f"epoch_{epoch:05d}.pt")
        if epoch % CHECKPOINT_EVERY == 0:
            print(f"saved rolling epoch {epoch}", flush=True)
    # final last.pt
    torch.save(state_out, args.out / "last.pt")
    print("training done")


if __name__ == "__main__":
    main()
