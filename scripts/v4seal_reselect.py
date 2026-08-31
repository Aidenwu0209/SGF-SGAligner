"""V4-Fix-Seal Part 8: fair reselection of ALL 22 checkpoints on
selection89 with the CANONICAL production inputs and the frozen
macro/micro metric semantics.  Every checkpoint runs TWICE; any
embedding/similarity/match/metric divergence fails closed.

Ranking keys (pre-registered, unchanged): macro_node_f1 ->
macro_top1 -> macro_top5 -> margin -> earlier epoch.  micro values
are reported alongside, never substituted.
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

torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
for p in (str(ROOT), str(ROOT / "src"),
          str(ROOT / "src/inference/sgf_official"),
          str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from canonical_inputs import (  # noqa: E402
    build_canonical_pair, arm_edges, arm_fingerprint,
    MATCHER_SHA, BUILDER_SHA,
)
from v4_train import batch_for  # noqa: E402
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from v4seal_metrics import per_pair_node_metrics, aggregate  # noqa: E402

OUT = ROOT / "outputs/official_sgaligner_v4_fix_seal_20260828"
V4 = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
ARMS = {"B": ("complete", range(5, 61, 5)),
        "C": ("explicit", range(5, 51, 5))}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()


def evaluate(ckpt_path, arm, samples, device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    per_pair = []
    for pair_id, dd, anchors, anchor_idx, sim_cache in samples:
        with torch.no_grad():
            batch = batch_for(dd, arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(
                np.float32)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count, anchor_idx,
            sim=sim_cache(emb))
        pp["pair_id"] = pair_id
        pp["node_corrs"] = [[int(a), int(b)] for a, b in node_corrs]
        pp["embedding_sha"] = hash_of(emb)
        per_pair.append(pp)
    agg = aggregate(per_pair)
    return agg, per_pair


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/selection.txt")
    pairs = [l.strip() for l in pl.read_text().splitlines()
             if l.strip()]

    print("building canonical selection89 inputs...", flush=True)
    samples = []
    for pair_id in pairs:
        dd, _ = build_canonical_pair(pair_id, with_labels=False)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}

        def sim_cache(emb, _a=anchor_idx):
            normed = emb / np.maximum(
                np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            return normed @ normed.T

        samples.append((pair_id, dd, anchors, anchor_idx, sim_cache))
    input_fps = {
        label: {
            pid: hash_of(arm_edges(dd, arm)[0])
            for pid, dd, *_ in samples}
        for label, (arm, _) in ARMS.items()}

    determinism = {"pairs": []}
    for label, (arm, epochs) in ARMS.items():
        ranked_rows = []
        for epoch in epochs:
            ckpt = V4 / "training" / arm / f"epoch_{epoch:05d}.pt"
            runs = []
            for run in ("run1", "run2"):
                agg, per_pair = evaluate(
                    ckpt, arm, samples, device)
                runs.append((agg, per_pair))
            (a1, p1), (a2, p2) = runs
            identical = (
                a1 == a2
                and [p["embedding_sha"] for p in p1]
                == [p["embedding_sha"] for p in p2]
                and [p["node_corrs"] for p in p1]
                == [p["node_corrs"] for p in p2])
            determinism["pairs"].append({
                "arm": label, "epoch": epoch,
                "identical": identical})
            if not identical:
                (OUT / "determinism_replay.json").write_text(
                    json.dumps(determinism, indent=2) + "\n")
                raise RuntimeError(
                    f"FAIL-CLOSED {label} ep{epoch} nondeterministic")
            ckpt_sha = sha256_file(ckpt)
            fps = [
                arm_fingerprint(dd, pid, arm, ckpt_sha)
                for pid, dd, *_ in samples]
            ranked_rows.append({
                "arm": label, "epoch": epoch,
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "checkpoint_sha256": ckpt_sha,
                "metrics_macro_micro": a1,
                "per_pair": [
                    {k: v for k, v in p.items() if k != "node_corrs"}
                    | {"node_corrs": p["node_corrs"]}
                    for p in p1],
                "arm_fingerprints": fps,
            })
            print(f"{label} ep{epoch}: macroF1 {a1['macro_node_f1']:.4f}"
                  f" macroTop1 {a1['macro_top1']:.4f}"
                  f" macroTop5 {a1['macro_top5']:.4f}"
                  f" microF1 {a1['micro_node_f1']:.4f}"
                  f" zero-cand {a1['zero_candidate_pairs']}",
                  flush=True)
        ranked = sorted(
            ranked_rows,
            key=lambda r: (-r["metrics_macro_micro"]["macro_node_f1"],
                           -r["metrics_macro_micro"]["macro_top1"],
                           -r["metrics_macro_micro"]["macro_top5"],
                           -r["metrics_macro_micro"]["margin"],
                           r["epoch"]))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
        winner = ranked[0]
        (OUT / f"checkpoint_ranking_{label}.json").write_text(
            json.dumps({
                "arm": label,
                "matcher": {
                    "implementation": "inference.official_matching",
                    "sha256": MATCHER_SHA},
                "builder_sha256": BUILDER_SHA,
                "code_head": code_head,
                "selection_split": "selection89 ONLY",
                "inputs": "canonical production builder",
                "input_fingerprints": input_fps[label],
                "ranking": [
                    {"rank": r["rank"], "epoch": r["epoch"],
                     "checkpoint_sha256": r["checkpoint_sha256"],
                     "metrics": r["metrics_macro_micro"]}
                    for r in ranked],
                "full_per_pair": {
                    str(r["epoch"]): r["per_pair"] for r in ranked},
            }, indent=2) + "\n")
        (OUT / f"winner_{label}.json").write_text(
            json.dumps({
                "arm": label, "winner_epoch": winner["epoch"],
                "winner_checkpoint": winner["checkpoint"],
                "winner_checkpoint_sha256":
                    winner["checkpoint_sha256"],
                "winner_metrics": winner["metrics_macro_micro"],
                "ranking_key": ["macro_node_f1", "macro_top1",
                                "macro_top5", "margin", "epoch_asc"],
            }, indent=2) + "\n")
        print(f"arm {label} WINNER epoch {winner['epoch']}",
              flush=True)
    determinism["all_identical"] = all(
        p["identical"] for p in determinism["pairs"])
    determinism["checkpoints_checked"] = len(determinism["pairs"])
    (OUT / "determinism_replay.json").write_text(
        json.dumps(determinism, indent=2) + "\n")
    print("determinism:", determinism["all_identical"])


if __name__ == "__main__":
    main()
