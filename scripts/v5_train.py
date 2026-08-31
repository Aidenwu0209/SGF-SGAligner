"""V5 training (pre-registered protocol 26f2023, BEFORE training).

Arms (all initialised from the sealed V4 C-epoch25 checkpoint
SHA cd53b956...):
  B relation adaptation — PCT + GAT fully frozen; trains
    meta_embedding_rel + fusion row `rel`; explicit edges.
  C relation+GAT joint  — PCT frozen; trains healthy GAT
    (structure_encoder/structure_embedding) + relation head + fusion
    rows `gat`/`rel`; explicit edges (complete-none FORBIDDEN).

Canonical inputs via scripts/canonical_inputs.py (production centre,
official_mt19937).  Objective = the verified cross-graph-only
bidirectional overlap-weighted multi-positive InfoNCE.  Per-epoch
fail-closed guards; exact resume; deterministic algorithms armed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
for p in (str(ROOT), str(ROOT / "src"),
          str(ROOT / "src/inference/sgf_official"),
          str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from canonical_inputs import build_canonical_pair  # noqa: E402
from v4_train import batch_for  # noqa: E402
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import (  # noqa: E402
    PredictedGraphSource, load_anchor_ids,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from safety.cross_graph_loss import cross_graph_infonce  # noqa: E402
from v4seal_metrics import per_pair_node_metrics, aggregate  # noqa: E402

OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"
INIT_CKPT = (
    ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827/"
    "training/explicit/epoch_00025.pt"
)
SEED = 4242
MAX_EPOCHS = 60
EVAL_INTERVAL = 5
PATIENCE_EVALS = 6
LR = 5e-4
GRAD_CLIP = 5.0
PCT_PREFIXES = ("object_encoder", "object_embedding")
ARM_CONFIG = {
    "B": {
        "frozen_prefixes": (
            "object_encoder", "object_embedding",
            "structure_encoder", "structure_embedding"),
        "fusion_trainable_rows": [2],
    },
    "C": {
        "frozen_prefixes": (
            "object_encoder", "object_embedding"),
        "fusion_trainable_rows": [1, 2],
    },
}


def sha16(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def build_model(arm: str, device: str):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(INIT_CKPT, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    cfg = ARM_CONFIG[arm]
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(cfg["frozen_prefixes"])
    # fusion handled by row-level grad masking below
    model.fusion.weight.requires_grad = True
    model.object_encoder.eval()
    if arm == "B":
        model.structure_encoder.eval()
        model.structure_embedding.eval()
    return model


def fusion_grad_mask(model, arm):
    w = model.fusion.weight.grad
    if w is not None:
        frozen_rows = [i for i in range(3)
                       if i not in ARM_CONFIG[arm][
                           "fusion_trainable_rows"]]
        for i in frozen_rows:
            w[i] = 0


def frozen_snapshot(model, arm):
    snap = {}
    prefixes = ARM_CONFIG[arm]["frozen_prefixes"]
    for name, t in list(model.named_parameters()) + list(
            model.named_buffers()):
        if name.startswith(prefixes):
            snap[name] = t.detach().clone()
    frozen_rows = [i for i in range(3)
                   if i not in ARM_CONFIG[arm][
                       "fusion_trainable_rows"]]
    for i in frozen_rows:
        snap[f"fusion.row{i}"] = model.fusion.weight.data[i].clone()
    return snap


def build_train_samples(predicted):
    man = json.loads(Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/"
        "dataset_three_way.json"
    ).read_text())
    pair_root = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/pairs"
    )
    pairs = [pair_root / Path(r).parent.name
             for r in man["train_pairs"]]
    samples = []
    skipped = []
    for index, pair_dir in enumerate(pairs):
        pair_id = pair_dir.name
        try:
            dd, labels = build_canonical_pair(
                pair_id, with_labels=True, predicted=predicted)
            if not labels:
                skipped.append({"pair_id": pair_id,
                                "reason": "no positives"})
                continue
            samples.append((pair_id, dd, labels))
        except Exception as exc:  # noqa: BLE001 — recorded, never silent
            skipped.append({"pair_id": pair_id,
                            "reason": repr(exc)[:150]})
        if (index + 1) % 100 == 0:
            print(f"built {index+1}/{len(pairs)}", flush=True)
    return samples, skipped


def eval_selection(model, arm, samples, anchors_by_pair, device):
    model.eval()
    per_pair = []
    for pair_id, dd, _labels in samples:
        with torch.no_grad():
            batch = batch_for(dd, arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(
                np.float32)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        anchor_idx = anchors_by_pair[pair_id]
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count, anchor_idx, sim=sim)
        pp["pair_id"] = pair_id
        per_pair.append(pp)
    agg = aggregate([
        {"tp": p["tp"], "pred_count": p["pred_count"],
         "anchor_count": p["anchor_count"], "f1": p["f1"],
         "top1_hit": p["top1_hit"], "top1_total": p["top1_total"],
         "top5_hits": p["top5_hits"], "margin": p["margin"]}
        for p in per_pair])
    return agg, per_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="smoke override")
    args = parser.parse_args()

    out_root = args.out_dir if args.out_dir else OUT
    arm_dir = out_root / "training" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = build_model(args.arm, device)
    predicted = PredictedGraphSource()

    print("building selection89 canonical inputs...", flush=True)
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/selection.txt")
    sel_pairs = [l.strip() for l in pl.read_text().splitlines()
                 if l.strip()]
    sel_samples = []
    anchors_by_pair = {}
    for pair_id in sel_pairs:
        dd, _ = build_canonical_pair(
            pair_id, with_labels=False, predicted=predicted)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        anchors_by_pair[pair_id] = anchor_idx
        sel_samples.append((pair_id, dd, None))

    print("building train437 canonical samples...", flush=True)
    samples, skipped = build_train_samples(predicted)
    (arm_dir / "dataset_build.json").write_text(json.dumps({
        "train_pairs": 437, "used": len(samples),
        "skipped": skipped}, indent=2) + "\n")
    dataset_fingerprint = (
        f"{len(samples)}-v5-{args.arm}-canonical-mt19937")

    cfg = ARM_CONFIG[args.arm]
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if p.requires_grad],
        lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS)
    frozen_before = frozen_snapshot(model, args.arm)
    # audit outputs
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    init_audit = {
        "arm": args.arm,
        "init_checkpoint": str(INIT_CKPT.relative_to(ROOT)),
        "init_checkpoint_sha256": hashlib.sha256(
            INIT_CKPT.read_bytes()).hexdigest(),
        "frozen_tensor_hashes": {
            k: sha16(v) for k, v in frozen_before.items()},
        "param_groups": {
            n: bool(p.requires_grad)
            for n, p in model.named_parameters()},
        "fusion_trainable_rows": cfg["fusion_trainable_rows"],
        "code_head": code_head,
    }
    (out_root / f"frozen_tensor_audit_{args.arm}.json").write_text(
        json.dumps(init_audit, indent=2) + "\n")
    opt_param_ids = {
        id(p) for g in optimizer.param_groups for p in g["params"]}
    assert all(
        id(p) not in opt_param_ids
        for n, p in model.named_parameters()
        if n.startswith(cfg["frozen_prefixes"])), \
        "optimizer contains frozen parameters"

    start_epoch = 1
    pending_fp = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device,
                          weights_only=False)
        if ckpt.get("total_epochs") != MAX_EPOCHS:
            raise RuntimeError("resume total_epochs mismatch")
        pending_fp = ckpt.get("dataset_fingerprint")
        model.load_state_dict(ckpt["model"])
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
            torch.cuda.set_rng_state_all([
                s.to(torch.uint8).cpu()
                if isinstance(s, torch.Tensor)
                else torch.tensor(
                    bytearray(s), dtype=torch.uint8)
                for s in raw])
        print(f"resumed at epoch {ckpt['epoch']}", flush=True)
    if pending_fp is not None and pending_fp != dataset_fingerprint:
        raise RuntimeError("resume fingerprint mismatch")

    history_path = arm_dir / "history.jsonl"
    best_key = None
    best_epoch = 0
    bad_evals = 0
    if args.resume is not None and history_path.exists():
        for line in history_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "eval":
                    key = rec["metrics"]["macro_node_f1"]
                    if best_key is None or key > best_key:
                        best_key, best_epoch = key, rec["epoch"]
                    bad_evals = rec.get("bad_evals_after", 0)

    metrics_csv = arm_dir / "epoch_metrics.csv"
    if not metrics_csv.exists():
        metrics_csv.write_text(
            "epoch,loss,pos_sim,neg_sim,margin,grad_gat,grad_rel,"
            "fusion_softmax_max,frozen_ok,unique_emb,subnormal_frac,"
            "lr,gpu_peak_mb,epoch_s\n")

    final_limit = (args.stop_after_epoch
                   if args.stop_after_epoch is not None
                   else args.epochs)
    stop_reason = "max_epochs"
    ckpt_out = None
    grad_log = []
    for epoch in range(start_epoch, final_limit + 1):
        t0 = time.monotonic()
        rng = np.random.default_rng(91000 + epoch)
        order = rng.permutation(len(samples))
        model.train()
        model.object_encoder.eval()
        if args.arm == "B":
            model.structure_encoder.eval()
            model.structure_embedding.eval()
        losses, pos_sims, neg_sims = [], [], []
        grad_gat_sq = grad_rel_sq = 0.0
        for idx in order.tolist():
            pair_id, dd, labels = samples[idx]
            batch = batch_for(dd, "explicit", device)
            out = model(batch)
            emb = out["joint"]
            src_count = dd["src_count"]
            n_src, n_ref = src_count, emb.shape[0] - src_count
            src_map = dd["src_object_id2idx"]
            ref_map = dd["ref_object_id2idx"]
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
            fusion_grad_mask(model, args.arm)
            for n, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = float(p.grad.norm())
                if n.startswith("structure_"):
                    grad_gat_sq += g * g
                elif n.startswith("meta_embedding_rel"):
                    grad_rel_sq += g * g
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
        grad_gat = float(np.sqrt(grad_gat_sq))
        grad_rel = float(np.sqrt(grad_rel_sq))
        grad_log.append({
            "epoch": epoch, "grad_gat": grad_gat,
            "grad_rel": grad_rel})

        frozen_now = frozen_snapshot(model, args.arm)
        frozen_ok = all(
            torch.equal(frozen_now[k], frozen_before[k])
            for k in frozen_before)
        if not frozen_ok:
            raise RuntimeError("frozen tensors drifted — aborting")
        with torch.no_grad():
            softmax = torch.softmax(model.fusion.weight, dim=0)
            softmax_max = float(softmax.max())
        with torch.no_grad():
            probe = model(batch_for(samples[0][1], "explicit",
                                    device))["gat"]
            unique_emb = int(len(np.unique(
                np.round(probe.cpu().numpy(), 4), axis=0)))
        gat_params = torch.cat([
            p.detach().flatten()
            for n, p in model.named_parameters()
            if n.startswith("structure_encoder")])
        subnormal = float((
            (gat_params.abs() > 0)
            & (gat_params.abs()
               < torch.finfo(torch.float32).tiny)
            ).float().mean())
        gpu_peak = (torch.cuda.max_memory_allocated()
                    if device == "cuda" else 0.0)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        row = [
            epoch,
            float(np.mean(losses)) if losses else float("nan"),
            float(np.mean(pos_sims)) if pos_sims else float("nan"),
            float(np.mean(neg_sims)) if neg_sims else float("nan"),
            (float(np.mean(pos_sims)) - float(np.mean(neg_sims)))
            if pos_sims and neg_sims else float("nan"),
            grad_gat, grad_rel, softmax_max,
            int(frozen_ok), unique_emb, subnormal,
            optimizer.param_groups[0]["lr"],
            gpu_peak / 1e6, time.monotonic() - t0,
        ]
        metrics_csv.open("a").write(
            ",".join(str(x) for x in row) + "\n")
        # fail-closed guards
        if not np.isfinite(row[1]):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        if not np.isfinite(softmax_max) or softmax_max > 0.995:
            raise RuntimeError(
                f"fusion softmax monopolisation at epoch {epoch}: "
                f"{softmax_max}")
        if subnormal > 0:
            raise RuntimeError(f"GAT subnormal at epoch {epoch}")
        if unique_emb <= 1:
            raise RuntimeError(f"GAT collapsed at epoch {epoch}")
        if epoch > 1:
            trainable_key = (
                grad_gat if args.arm == "C" else grad_rel)
            if trainable_key == 0.0:
                raise RuntimeError(
                    f"trainable gradients zero at epoch {epoch}")
        print(f"epoch {epoch}: loss {row[1]:.4f} margin {row[4]:.4f}"
              f" gGat {grad_gat:.4f} gRel {grad_rel:.4f}"
              f" fusedMax {softmax_max:.3f} uniq {unique_emb}"
              f" frozen_ok {frozen_ok}", flush=True)

        if epoch % EVAL_INTERVAL == 0 or epoch == final_limit:
            agg, per_pair = eval_selection(
                model, "explicit", sel_samples, anchors_by_pair,
                device)
            key = agg["macro_node_f1"]
            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                bad_evals = 0
            else:
                bad_evals += 1
            with history_path.open("a") as fh:
                fh.write(json.dumps({
                    "kind": "eval", "epoch": epoch,
                    "metrics": agg,
                    "bad_evals_after": bad_evals}) + "\n")
            print(f"  eval@{epoch}: macroF1 "
                  f"{agg['macro_node_f1']:.4f} top1 "
                  f"{agg['macro_top1']:.4f} top5 "
                  f"{agg['macro_top5']:.4f} best={best_epoch}"
                  f" bad={bad_evals}", flush=True)
            ckpt_out = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "next_epoch": epoch + 1,
                "total_epochs": MAX_EPOCHS,
                "training_config": {
                    "arm": args.arm, "lr": LR, "seed": SEED,
                    "protocol_commit": "26f2023",
                    "model_naming": (
                        "official-architecture SGF-predicted "
                        "relation+GAT research candidate")},
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
                "dataset_fingerprint": dataset_fingerprint,
                "selection_metrics": agg,
            }
            tmp = arm_dir / f".epoch_{epoch:05d}.pt.tmp"
            torch.save(ckpt_out, tmp)
            tmp.replace(arm_dir / f"epoch_{epoch:05d}.pt")
            if bad_evals >= PATIENCE_EVALS:
                stop_reason = (
                    f"early_stop_patience_{PATIENCE_EVALS}_evals")
                print(stop_reason, flush=True)
                break
    if ckpt_out is not None:
        torch.save(ckpt_out, arm_dir / "last.pt")
    (out_root / f"gradient_health_{args.arm}.json").write_text(
        json.dumps(grad_log, indent=2) + "\n")
    (arm_dir / "run_summary.json").write_text(json.dumps({
        "arm": args.arm, "stop_reason": stop_reason,
        "epochs_run": final_limit, "best_epoch": best_epoch,
        "best_macro_f1": best_key,
        "dataset_fingerprint": dataset_fingerprint,
    }, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
