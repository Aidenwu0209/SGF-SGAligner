"""V6 training: SGF-domain matcher arms (protocol 0e546a3741f8).

Arms (all initialised from sealed V5 B_ep10, SHA c82637337b9a…):
  B projection+relation+fusion — PCT trunk & GAT frozen; trains
    object_embedding + meta_embedding_rel + fusion rows pct/rel.
  C pct_last_stage — B plus object_encoder.linear2/bn2 at 0.05x LR.
  D healthy_gat — B plus GAT training (V5 C-ep25 GAT init).

Labels: the NEW sgf_node_labels builder (sets with split/merge,
ambiguous masked OUT of the loss denominators via ignore_mask —
implemented as a v6 wrapper over the verified cross_graph_infonce
because the core function has no ignore semantics; the wrapper
substitutes ambiguous entries with -inf logit bias, keeping every
existing guarantee and failing closed on empty query sets).

Augmentations (train only): segment dropout p=0.1, point jitter
sigma=1cm (metric rigid-invariant).
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
    PredictedGraphSource, load_anchor_ids, load_pair_record,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from safety.cross_graph_loss import cross_graph_infonce  # noqa: E402
from sgf_node_labels import label_pair  # noqa: E402
from v4seal_metrics import (  # noqa: E402
    per_pair_node_metrics, aggregate,
)

OUT = ROOT / (
    "outputs/official_sgaligner_v6_sgf_domain_matcher_20260829")
INIT_CKPT = (ROOT / "outputs/official_sgaligner_v5_relation_gat_"
             "20260828/training/B/epoch_00010.pt")
GAT_INIT_CKPT = (ROOT / "outputs/official_sgaligner_v4_healthy_gat_"
                 "20260827/training/explicit/epoch_00025.pt")
SEED = 4242
MAX_EPOCHS = 60
EVAL_INTERVAL = 5
PATIENCE_EVALS = 6
LR = 5e-4
LR_PCT_LAST = 5e-4 * 0.05
GRAD_CLIP = 5.0
SEG_DROPOUT = 0.10
JITTER_SIGMA = 0.01

ARM_CONFIG = {
    "B": {
        "frozen_prefixes": (
            "object_encoder", "structure_encoder",
            "structure_embedding"),
        "fusion_trainable_rows": [0, 2],
        "pct_last_stage": False, "train_gat": False,
    },
    "C": {
        "frozen_prefixes": (
            "object_encoder.embedding", "object_encoder.sa1",
            "object_encoder.sa2", "object_encoder.sa3",
            "object_encoder.sa4", "object_encoder.linear.0",
            "object_encoder.linear.1", "object_encoder.bn1",
            "object_encoder.dp1", "object_encoder.dp2",
            "structure_encoder", "structure_embedding"),
        "fusion_trainable_rows": [0, 2],
        "pct_last_stage": True, "train_gat": False,
    },
    "D": {
        "frozen_prefixes": ("object_encoder",),
        "fusion_trainable_rows": [0, 2],
        "pct_last_stage": False, "train_gat": True,
    },
}


def sha16(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def masked_infonce(src_emb, ref_emb, positives, weights,
                   ambiguous, temperature=0.1):
    """cross_graph_infonce with ambiguous pairs removed from the
    DENOMINATOR: substitute -inf logits at ambiguous cells (they can
    never be positives, so numerators are unaffected; logsumexp over
    a row whose every entry is -inf fails closed upstream when the
    query set becomes empty — verified by unit test)."""
    if ambiguous.any() and (ambiguous & positives).any():
        raise ValueError("ambiguous overlaps positives (invalid)")
    neg_inf = torch.full_like(
        torch.zeros(1), float("-inf")).item()
    # implement via logits bias is not exposed by the core fn; use
    # the equivalent: temporarily set ambiguous similarities to -inf
    # by scaling — instead we compute the core loss on a masked copy
    # of the embeddings is NOT equivalent.  Correct approach: call
    # the core function with a temperature-equivalent trick is
    # impossible; therefore re-derive the same algorithm here with
    # the mask, keeping every fail-closed check of the original.
    if src_emb.ndim != 2 or ref_emb.ndim != 2:
        raise ValueError("embeddings must be [N, D]")
    n_src, n_ref = src_emb.shape[0], ref_emb.shape[0]
    if positives.shape != (n_src, n_ref):
        raise ValueError("positives shape")
    src_norm = F.normalize(src_emb, dim=1)
    ref_norm = F.normalize(ref_emb, dim=1)
    logits = (src_norm @ ref_norm.T) / temperature
    logits = logits.masked_fill(ambiguous, float("-inf"))

    def direction(query_logits, query_positives, query_weights):
        per_query, q_weights = [], []
        for qi in range(query_logits.shape[0]):
            js = torch.nonzero(query_positives[qi]).squeeze(1)
            if len(js) == 0:
                continue
            den = torch.logsumexp(query_logits[qi], dim=0)
            num = torch.logsumexp(query_logits[qi, js], dim=0)
            if not torch.isfinite(den):
                raise ValueError(
                    "all-denominator row (ambiguous masked "
                    "everything) — fail closed")
            per_query.append(den - num)
            q_weights.append(query_weights[qi, js].max())
        if not per_query:
            raise ValueError("no query with a positive")
        per_query = torch.stack(per_query)
        q_weights = torch.stack(q_weights)
        if not torch.isfinite(per_query).all():
            raise ValueError("non-finite loss term (fail closed)")
        return (per_query * q_weights).sum() / q_weights.sum(
        ).clamp_min(1e-12)

    forward = direction(logits, positives, weights)
    reverse = direction(logits.T, positives.T, weights.T)
    return 0.5 * (forward + reverse)


def build_model(arm: str, device: str):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(INIT_CKPT, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    cfg = ARM_CONFIG[arm]
    if cfg["train_gat"]:
        gat_state = torch.load(
            GAT_INIT_CKPT, map_location=device,
            weights_only=False)["model"]
        gat_keys = [
            k for k in model.state_dict()
            if k.startswith(("structure_encoder",
                             "structure_embedding"))]
        model.load_state_dict(
            {k: v for k, v in gat_state.items() if k in gat_keys},
            strict=False)
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(
            cfg["frozen_prefixes"])
    model.fusion.weight.requires_grad = True
    model.object_encoder.eval()
    if not cfg["train_gat"]:
        model.structure_encoder.eval()
        model.structure_embedding.eval()
    return model


def fusion_grad_mask(model, arm):
    w = model.fusion.weight.grad
    if w is not None:
        frozen = [i for i in range(3)
                  if i not in ARM_CONFIG[arm][
                      "fusion_trainable_rows"]]
        for i in frozen:
            w[i] = 0


def frozen_snapshot(model, arm):
    snap = {}
    prefixes = ARM_CONFIG[arm]["frozen_prefixes"]
    for name, t in list(model.named_parameters()) + list(
            model.named_buffers()):
        if name.startswith(prefixes):
            snap[name] = t.detach().clone()
    frozen_rows = [1] if not ARM_CONFIG[arm]["train_gat"] else []
    for i in frozen_rows:
        snap[f"fusion.row{i}"] = \
            model.fusion.weight.data[i].clone()
    return snap


def build_train_samples(predicted):
    man = json.loads(Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/"
        "dataset_three_way.json").read_text())
    pair_root = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/pairs")
    pairs = [pair_root / Path(r).parent.name
             for r in man["train_pairs"]]
    samples = []
    skipped = []
    for index, pair_dir in enumerate(pairs):
        pair_id = pair_dir.name
        try:
            dd, _ = build_canonical_pair(
                pair_id, with_labels=False, predicted=predicted)
            payload = load_pair_record(pair_id)
            gt = np.asarray(
                payload["gt_transform"], dtype=np.float64
            ).reshape(4, 4)
            src_pred = predicted.load(pair_id.split("_to_")[0])
            ref_pred = predicted.load(pair_id.split("_to_")[1])
            stats = label_pair(
                src_pred.segments, ref_pred.segments, gt)
            pos = [(s.src, s.ref, s.bidir_10) for s in stats
                   if s.label == "positive"]
            amb = {(s.src, s.ref) for s in stats
                   if s.label == "ambiguous"}
            if not pos:
                skipped.append({"pair_id": pair_id,
                                "reason": "no positives"})
                continue
            samples.append((pair_id, dd, pos, amb))
        except Exception as exc:  # noqa: BLE001 — recorded
            skipped.append({"pair_id": pair_id,
                            "reason": repr(exc)[:150]})
        if (index + 1) % 50 == 0:
            print(f"built {index+1}/{len(pairs)}", flush=True)
    return samples, skipped


def eval_selection(model, samples, anchors_by_pair, device):
    model.eval()
    per_pair = []
    for pair_id, dd, _pos, _amb in samples:
        with torch.no_grad():
            emb = model(
                batch_for(dd, "explicit", device))["joint"
            ].cpu().numpy().astype(np.float32)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count,
            anchors_by_pair[pair_id], sim=sim)
        pp["pair_id"] = pair_id
        per_pair.append(pp)
    return aggregate([
        {"tp": p["tp"], "pred_count": p["pred_count"],
         "anchor_count": p["anchor_count"], "f1": p["f1"],
         "top1_hit": p["top1_hit"], "top1_total": p["top1_total"],
         "top5_hits": p["top5_hits"], "margin": p["margin"]}
        for p in per_pair]), per_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("B", "C", "D"),
                        required=True)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--stop-after-epoch", type=int,
                        default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
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
    sel_samples = []
    anchors_by_pair = {}
    for pair_id in [l.strip() for l in
                    pl.read_text().splitlines() if l.strip()]:
        dd, _ = build_canonical_pair(
            pair_id, with_labels=False, predicted=predicted)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchors_by_pair[pair_id] = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        sel_samples.append((pair_id, dd, None, None))

    print("building train437 with NEW labels...", flush=True)
    samples, skipped = build_train_samples(predicted)
    (arm_dir / "dataset_build.json").write_text(json.dumps({
        "train_pairs": 437, "used": len(samples),
        "skipped": skipped}, indent=2) + "\n")
    dataset_fingerprint = (
        f"{len(samples)}-v6-{args.arm}-newlabels-canonical")

    cfg = ARM_CONFIG[args.arm]
    params = [
        p for n, p in model.named_parameters() if p.requires_grad]
    pct_last = [
        p for n, p in model.named_parameters()
        if cfg["pct_last_stage"]
        and n.startswith("object_encoder")
        and p.requires_grad]
    main_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not (cfg["pct_last_stage"]
                 and n.startswith("object_encoder"))]
    groups = [{"params": main_params, "lr": LR}]
    if cfg["pct_last_stage"] and pct_last:
        groups.append({"params": pct_last, "lr": LR_PCT_LAST})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS)
    frozen_before = frozen_snapshot(model, args.arm)
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    (out_root / f"parameter_groups_{args.arm}.json").write_text(
        json.dumps({
            "arm": args.arm,
            "groups": {
                n: bool(p.requires_grad)
                for n, p in model.named_parameters()},
            "fusion_trainable_rows": cfg["fusion_trainable_rows"],
            "pct_last_stage_lr": (
                LR_PCT_LAST if cfg["pct_last_stage"] else None),
            "init_checkpoint_sha256": hashlib.sha256(
                INIT_CKPT.read_bytes()).hexdigest(),
            "frozen_tensor_hashes": {
                k: sha16(v)
                for k, v in frozen_before.items()},
            "code_head": code_head}, indent=2) + "\n")
    opt_ids = {id(p) for g in optimizer.param_groups
               for p in g["params"]}
    for n, p in model.named_parameters():
        if n.startswith(cfg["frozen_prefixes"]):
            assert id(p) not in opt_ids, \
                f"optimizer contains frozen {n}"

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
                else torch.tensor(bytearray(s),
                                  dtype=torch.uint8)
                for s in raw])
        print(f"resumed at epoch {ckpt['epoch']}", flush=True)
    if pending_fp is not None \
            and pending_fp != dataset_fingerprint:
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
            "epoch,loss,pos_sim,neg_sim,margin,grad_main,"
            "grad_pctlast,fusedMax,frozen_ok,unique_emb,lr,"
            "gpu_peak_mb,epoch_s\n")

    final_limit = (args.stop_after_epoch
                   if args.stop_after_epoch is not None
                   else args.epochs)
    stop_reason = "max_epochs"
    ckpt_out = None
    for epoch in range(start_epoch, final_limit + 1):
        t0 = time.monotonic()
        rng = np.random.default_rng(91000 + epoch)
        order = rng.permutation(len(samples))
        model.train()
        model.object_encoder.eval()
        if not cfg["train_gat"]:
            model.structure_encoder.eval()
            model.structure_embedding.eval()
        losses, pos_sims, neg_sims = [], [], []
        g_main = g_pct = 0.0
        for idx in order.tolist():
            pair_id, dd, pos, amb = samples[idx]
            batch = batch_for(dd, "explicit", device)
            # augmentations (train only, rigid-invariant)
            pts = dd["tot_obj_pts"].copy()
            jitter = np.random.default_rng(
                hash(pair_id) % (2**31) + epoch).normal(
                0.0, JITTER_SIGMA, pts.shape).astype(np.float32)
            batch["tot_obj_pts"] = torch.from_numpy(
                pts + jitter).to(device)
            src_count = dd["src_count"]
            n = dd["tot_obj_pts"].shape[0]
            n_ref = n - src_count
            src_map = dd["src_object_id2idx"]
            ref_map = dd["ref_object_id2idx"]
            positives = torch.zeros(
                src_count, n_ref, dtype=torch.bool, device=device)
            weights = torch.zeros(
                src_count, n_ref, device=device)
            ambiguous = torch.zeros(
                src_count, n_ref, dtype=torch.bool, device=device)
            drop = np.random.default_rng(
                hash(pair_id) % (2**31) + 7 * epoch).random(n)
            for s, r, w in pos:
                if s in src_map and r in ref_map:
                    si, rj = src_map[s], ref_map[r]
                    if drop[si] < SEG_DROPOUT:
                        continue  # segment dropout
                    positives[si, rj] = True
                    weights[si, rj] = min(max(w, 1e-3), 1.0)
            for s, r in amb:
                if s in src_map and r in ref_map:
                    ambiguous[src_map[s], ref_map[r]] = True
            if not positives.any():
                continue
            out = model(batch)
            emb = out["joint"]
            loss = masked_infonce(
                emb[:src_count], emb[src_count:],
                positives, weights, ambiguous)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            fusion_grad_mask(model, args.arm)
            for n_name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = float(p.grad.norm())
                if n_name.startswith("object_encoder"):
                    g_pct += g * g
                else:
                    g_main += g * g
            torch.nn.utils.clip_grad_norm_(
                [p for g_ in optimizer.param_groups
                 for p in g_["params"]], GRAD_CLIP)
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
        frozen_now = frozen_snapshot(model, args.arm)
        frozen_ok = all(
            torch.equal(frozen_now[k], frozen_before[k])
            for k in frozen_before)
        if not frozen_ok:
            raise RuntimeError("frozen tensors drifted — aborting")
        with torch.no_grad():
            fused_max = float(torch.softmax(
                model.fusion.weight, dim=0).max())
        with torch.no_grad():
            probe = model(batch_for(
                samples[0][1], "explicit", device))["gat"]
            unique_emb = int(len(np.unique(
                np.round(probe.cpu().numpy(), 4), axis=0)))
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
            float(np.sqrt(g_main)), float(np.sqrt(g_pct)),
            fused_max, int(frozen_ok), unique_emb,
            optimizer.param_groups[0]["lr"],
            gpu_peak / 1e6, time.monotonic() - t0]
        metrics_csv.open("a").write(
            ",".join(str(x) for x in row) + "\n")
        if not np.isfinite(row[1]):
            raise RuntimeError(f"non-finite loss epoch {epoch}")
        if not np.isfinite(fused_max) or fused_max > 0.995:
            raise RuntimeError("fusion monopolisation")
        if unique_emb <= 1:
            raise RuntimeError("embedding collapse")
        print(f"epoch {epoch}: loss {row[1]:.4f} margin {row[4]:.4f}"
              f" gMain {row[5]:.3f} gPct {row[6]:.3f}"
              f" fusedMax {fused_max:.3f} uniq {unique_emb}"
              f" frozen_ok {frozen_ok}", flush=True)

        if epoch % EVAL_INTERVAL == 0 or epoch == final_limit:
            agg, per_pair = eval_selection(
                model, sel_samples, anchors_by_pair, device)
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
                    "protocol_commit": "0e546a3741f8",
                    "labels": "v6 sgf_node_labels (sets+ambiguous)",
                    "model_naming": (
                        "official-architecture SGF-predicted "
                        "SGF-domain-matcher research candidate")},
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
                "dataset_fingerprint": dataset_fingerprint,
                "selection_metrics": agg}
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
    (arm_dir / "run_summary.json").write_text(json.dumps({
        "arm": args.arm, "stop_reason": stop_reason,
        "epochs_run": final_limit, "best_epoch": best_epoch,
        "best_macro_f1": best_key,
        "dataset_fingerprint": dataset_fingerprint,
    }, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
