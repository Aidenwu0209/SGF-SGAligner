"""V4-Fix Part 2: OFFICIAL-SEMANTIC fair checkpoint reselection.

Evaluates EVERY saved checkpoint (B complete: epochs 5..60; C explicit:
epochs 5..50) on selection89 ONLY, using the REAL official matcher —
``inference.official_matching`` imported and called verbatim (top-3 of
the FULL ranking excluding self, THEN keep cross-graph candidates).
No approximate reimplementation is permitted or used.

Determinism: the whole evaluation runs TWICE; per checkpoint the
embedding hash, similarity hash, node-match list and metric dict must
be identical across the two runs (byte/numeric strict) — any mismatch
fails closed BEFORE any calibration step.

Env pinned before torch import: CUBLAS_WORKSPACE_CONFIG=:4096:8,
OMP_NUM_THREADS=1.
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import torch  # noqa: E402  (determinism must be armed BEFORE any forward)

torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from v4_train import build_split_samples, batch_for  # noqa: E402
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

OLD = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
NEW = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"
MATCHER_SOURCE = ROOT / "src/inference/sgf_official/inference.py"

ARMS = {
    "B": ("complete", range(5, 61, 5)),
    "C": ("explicit", range(5, 51, 5)),
}
OLD_WINNERS = {"B": 40, "C": 20}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()


def pair_input_sha(data_dict):
    fields = [
        data_dict["tot_obj_pts"], data_dict["tot_rel_pose"],
        data_dict["tot_bow_vec_object_edge_feats"],
        np.asarray(data_dict["edges"]), data_dict["obj_ids"],
    ]
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(np.asarray(f)).tobytes()
        for f in fields)).hexdigest()


def evaluate_checkpoint(ckpt_path, arm, samples, anchors_by_pair,
                        device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    per_pair = []
    f1s = []
    tp_all = pred_all = anchor_all = 0
    total_candidates = 0
    zero_candidate_pairs = 0
    for pair_id, data_dict, _labels in samples:
        with torch.no_grad():
            batch = batch_for(data_dict, arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(np.float32)
        src_count = data_dict["src_count"]
        # THE official matcher — called verbatim
        node_corrs, rank_list, _sim_full = official_matching(
            emb, src_count)
        anchors = anchors_by_pair[pair_id]
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + src_count)
            for s, r in anchors if s in src_map and r in ref_map}
        pred = set(node_corrs)
        tp = len(pred & anchor_idx)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(anchor_idx) if anchor_idx else 0.0
        f1 = 2 * p * r / max(p + r, 1e-12)
        f1s.append(f1)
        tp_all += tp
        pred_all += len(pred)
        anchor_all += len(anchor_idx)
        total_candidates += len(pred)
        if not pred:
            zero_candidate_pairs += 1
        # top1/top5 from the full cross-graph ranking (official
        # matching semantics for ranking metrics, as in the V3/V4
        # caches) + margin over anchor queries
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        top1_hit = top1_total = 0
        top5_hits = 0
        pos_sims, neg_sims = [], []
        for i in range(src_count):
            refs = [x for x in rank_list[i] if x >= src_count][:5]
            if not refs:
                continue
            top1_total += 1
            if (i, int(refs[0])) in anchor_idx:
                top1_hit += 1
            top5_hits += sum(
                1 for x in refs[:5] if (i, int(x)) in anchor_idx)
            for x in refs[:5]:
                if (i, int(x)) in anchor_idx:
                    pos_sims.append(float(sim[i, int(x)]))
                else:
                    neg_sims.append(float(sim[i, int(x)]))
        per_pair.append({
            "pair_id": pair_id,
            "node_corrs": [[int(a), int(b)] for a, b in node_corrs],
            "candidates": len(pred),
            "tp": tp, "anchors": len(anchor_idx),
            "f1": f1,
            "top1_hit": top1_hit, "top1_total": top1_total,
            "top5_hits": top5_hits,
            "pos_sim_mean": float(np.mean(pos_sims))
            if pos_sims else None,
            "neg_sim_mean": float(np.mean(neg_sims))
            if neg_sims else None,
            "embedding_sha": hash_of(emb),
            "similarity_sha": hash_of(sim.astype(np.float32)),
        })
    micro_p = tp_all / pred_all if pred_all else 0.0
    micro_r = tp_all / anchor_all if anchor_all else 0.0
    metrics = {
        "macro_node_f1": float(np.mean(f1s)),
        "micro_node_f1": 2 * micro_p * micro_r / max(
            micro_p + micro_r, 1e-12),
        "top1_precision": sum(
            pp["top1_hit"] for pp in per_pair) / max(
            sum(pp["top1_total"] for pp in per_pair), 1),
        "top5_recall": sum(
            pp["top5_hits"] for pp in per_pair) / max(anchor_all, 1),
        "margin": float(np.mean([
            pp["pos_sim_mean"] - pp["neg_sim_mean"]
            for pp in per_pair
            if pp["pos_sim_mean"] is not None
            and pp["neg_sim_mean"] is not None]))
        if any(pp["pos_sim_mean"] is not None
               and pp["neg_sim_mean"] is not None
               for pp in per_pair) else 0.0,
        "total_valid_candidates": total_candidates,
        "zero_candidate_pairs": zero_candidate_pairs,
    }
    digest = {
        "embedding_hashes": [pp["embedding_sha"] for pp in per_pair],
        "similarity_hashes": [pp["similarity_sha"] for pp in per_pair],
        "node_matches": [pp["node_corrs"] for pp in per_pair],
        "metrics": metrics,
    }
    return metrics, per_pair, digest


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    matcher_sha = sha256_file(MATCHER_SOURCE)

    print("building selection89 tensors (official_mt19937)...",
          flush=True)
    from adapters.sgf.data_sources import PredictedGraphSource

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    sel_pairs = [l.strip() for l in
                 (pairlists / "selection.txt").read_text().splitlines()
                 if l.strip()]
    samples, _skipped = build_split_samples(
        sel_pairs, PredictedGraphSource())
    anchors_by_pair = {
        pid: set(load_anchor_ids(pid)) for pid, _d, _l in samples}
    input_shas = {
        pid: pair_input_sha(dd) for pid, dd, _l in samples}
    dataset_fingerprint = hashlib.sha256(
        json.dumps(input_shas, sort_keys=True).encode()).hexdigest()

    results = {}
    determinism = {"runs": ["run1", "run2"], "pairs": []}
    for arm_label, (arm, epochs) in ARMS.items():
        rows = []
        for epoch in epochs:
            ckpt = OLD / "training" / arm / f"epoch_{epoch:05d}.pt"
            assert ckpt.exists(), ckpt
            run_metrics = {}
            for run in ("run1", "run2"):
                metrics, per_pair, digest = evaluate_checkpoint(
                    ckpt, arm, samples, anchors_by_pair, device)
                run_metrics[run] = (metrics, per_pair, digest)
            m1, p1, d1 = run_metrics["run1"]
            m2, p2, d2 = run_metrics["run2"]
            identical = (
                d1["embedding_hashes"] == d2["embedding_hashes"]
                and d1["similarity_hashes"] == d2["similarity_hashes"]
                and d1["node_matches"] == d2["node_matches"]
                and d1["metrics"] == d2["metrics"])
            if not identical:
                first_bad = next(
                    (pp1["pair_id"] for pp1, pp2 in zip(p1, p2)
                     if pp1["embedding_sha"] != pp2["embedding_sha"]),
                    None)
                print(
                    f"DIFF arm={arm_label} epoch={epoch} "
                    f"emb_hashes_equal="
                    f"{d1['embedding_hashes'] == d2['embedding_hashes']} "
                    f"sim_equal="
                    f"{d1['similarity_hashes'] == d2['similarity_hashes']} "
                    f"matches_equal="
                    f"{d1['node_matches'] == d2['node_matches']} "
                    f"metrics_equal={d1['metrics'] == d2['metrics']} "
                    f"first_embedding_diff_pair={first_bad}",
                    flush=True)
            determinism["pairs"].append({
                "arm": arm_label, "epoch": epoch,
                "identical": identical})
            if not identical:
                (NEW / "determinism_replay.json").write_text(
                    json.dumps(determinism, indent=2) + "\n")
                raise RuntimeError(
                    f"FAIL-CLOSED: {arm_label} epoch {epoch} is not "
                    "deterministic across reruns")
            rows.append({
                "arm": arm_label, "epoch": epoch,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(ckpt),
                "metrics": m1,
                "per_pair": p1,
            })
            print(f"{arm_label} ep{epoch}: F1 {m1['macro_node_f1']:.4f}"
                  f" top1 {m1['top1_precision']:.4f}"
                  f" top5 {m1['top5_recall']:.4f}"
                  f" marg {m1['margin']:.4f}"
                  f" zero-cand {m1['zero_candidate_pairs']}",
                  flush=True)
        results[arm_label] = rows

    # ranking + corrected selection (pre-registered lexicographic)
    for arm_label, rows in results.items():
        ranked = sorted(
            rows,
            key=lambda r: (-r["metrics"]["macro_node_f1"],
                           -r["metrics"]["top1_precision"],
                           -r["metrics"]["top5_recall"],
                           -r["metrics"]["margin"], r["epoch"]))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        winner = ranked[0]
        (NEW / f"official_semantic_checkpoint_ranking_{arm_label}.json"
         ).write_text(json.dumps({
            "arm": arm_label,
            "matcher": {
                "implementation": "inference.official_matching "
                                  "(imported and called verbatim)",
                "source": str(MATCHER_SOURCE.relative_to(ROOT)),
                "sha256": matcher_sha},
            "selection_split": "selection89 ONLY",
            "dataset_fingerprint": dataset_fingerprint,
            "input_shas": input_shas,
            "code_head": code_head,
            "env": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "OMP_NUM_THREADS": "1"},
            "ranking": [
                {"rank": r["rank"], "epoch": r["epoch"],
                 "checkpoint": r["checkpoint"],
                 "checkpoint_sha256": r["checkpoint_sha256"],
                 "metrics": r["metrics"]} for r in ranked],
        }, indent=2) + "\n")
        (NEW / f"checkpoint_selection_corrected_{arm_label}.json"
         ).write_text(json.dumps({
            "arm": arm_label,
            "winner_epoch": winner["epoch"],
            "winner_checkpoint": winner["checkpoint"],
            "winner_checkpoint_sha256": winner["checkpoint_sha256"],
            "winner_metrics": winner["metrics"],
            "old_winner_epoch": OLD_WINNERS[arm_label],
            "changed": winner["epoch"] != OLD_WINNERS[arm_label],
            "lexicographic_key": ["macro_node_f1", "top1_precision",
                                  "top5_recall", "margin",
                                  "earlier_epoch"],
        }, indent=2) + "\n")
        print(f"arm {arm_label}: winner epoch {winner['epoch']} "
              f"(old {OLD_WINNERS[arm_label]}, "
              f"changed={winner['epoch'] != OLD_WINNERS[arm_label]})",
              flush=True)

    determinism["all_identical"] = all(
        p["identical"] for p in determinism["pairs"])
    determinism["checkpoints_checked"] = len(determinism["pairs"])
    (NEW / "determinism_replay.json").write_text(
        json.dumps(determinism, indent=2) + "\n")

    md = ["# Old vs corrected checkpoint selection (official matcher "
          "semantics, selection89 only)", ""]
    for arm_label, rows in results.items():
        sel = json.loads(
            (NEW / f"checkpoint_selection_corrected_{arm_label}.json"
             ).read_text())
        old_row = next(
            r for r in rows if r["epoch"] == OLD_WINNERS[arm_label])
        new_row = next(
            r for r in rows if r["epoch"] == sel["winner_epoch"])
        md.append(
            f"## Arm {arm_label} ({ARMS[arm_label][0]})\n\n"
            f"- old winner: epoch {OLD_WINNERS[arm_label]} "
            f"(F1 {old_row['metrics']['macro_node_f1']:.4f}, "
            f"top1 {old_row['metrics']['top1_precision']:.4f}, "
            f"top5 {old_row['metrics']['top5_recall']:.4f}, "
            f"margin {old_row['metrics']['margin']:.4f}, "
            f"zero-cand {old_row['metrics']['zero_candidate_pairs']})\n"
            f"- corrected winner: epoch {sel['winner_epoch']} "
            f"(F1 {new_row['metrics']['macro_node_f1']:.4f}, "
            f"top1 {new_row['metrics']['top1_precision']:.4f}, "
            f"top5 {new_row['metrics']['top5_recall']:.4f}, "
            f"margin {new_row['metrics']['margin']:.4f}, "
            f"zero-cand {new_row['metrics']['zero_candidate_pairs']})\n"
            f"- changed: {sel['changed']}\n")
    (NEW / "old_vs_corrected_selection.md").write_text(
        "\n".join(md) + "\n")
    print("determinism all identical:",
          determinism["all_identical"])


if __name__ == "__main__":
    main()
