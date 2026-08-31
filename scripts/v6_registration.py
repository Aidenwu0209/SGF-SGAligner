"""V6 registration-aware selection: node metrics for ALL checkpoints
(arms B/C/D) + full registration for top-3 per arm + A baseline,
with the spatial-consistency layer ON (pre-registered main config).

Selection key (pre-registered): accepted-strict-error==0 (hard) ->
max raw strict -> max raw relaxed -> max accepted-correct -> macro
F1 -> macro top1 -> macro top5 -> min epoch.  Hard gates mirror the
protocol.
"""
from __future__ import annotations

import hashlib
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

from canonical_inputs import build_canonical_pair  # noqa: E402
from v4_train import batch_for  # noqa: E402
from inference import (  # noqa: E402
    official_matching, geotransformer_forward, STRICT, RELAXED,
)
from adapters.sgf.data_sources import (  # noqa: E402
    load_anchor_ids, load_gt_transform,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from v3b_cache_runner import (  # noqa: E402
    combo_registration, combo_decision,
)
from v4seal_metrics import (  # noqa: E402
    per_pair_node_metrics, aggregate,
)
from spatial_consistency import (  # noqa: E402
    cluster_candidates, hypothesis_rank,
)

OUT = ROOT / (
    "outputs/official_sgaligner_v6_sgf_domain_matcher_20260829")
BASELINE_CKPT = (
    ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828/"
    "training/B/epoch_00010.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def node_eval(ckpt_path, samples, device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    per_pair = []
    for pair_id, dd, anchor_idx, _centres in samples:
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
            node_corrs, rank_list, src_count, anchor_idx, sim=sim)
        pp["pair_id"] = pair_id
        per_pair.append(pp)
    return aggregate([
        {"tp": p["tp"], "pred_count": p["pred_count"],
         "anchor_count": p["anchor_count"], "f1": p["f1"],
         "top1_hit": p["top1_hit"], "top1_total": p["top1_total"],
         "top5_hits": p["top5_hits"], "margin": p["margin"]}
        for p in per_pair]), per_pair


def registration_with_consistency(ckpt_path, samples, device):
    """Full pipeline; candidates clustered by the spatial-
    consistency layer; EVERY cluster runs the unchanged
    GeoT/RANSAC/ICP/decision; best hypothesis chosen by the
    pre-registered GT-free rank."""
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    counts = {"requested": len(samples), "structured": 0,
              "completed": 0, "raw_strict": 0, "raw_relaxed": 0,
              "accepted": 0, "accepted_strict_correct": 0,
              "accepted_strict_error": 0, "failed": 0,
              "zero_candidate": 0, "hypotheses_run": 0,
              "consistency_rescue_vs_flat": 0}
    rows = []
    for pair_id, dd, anchor_idx, centres in samples:
        counts["structured"] += 1
        with torch.no_grad():
            emb = model(
                batch_for(dd, "explicit", device))["joint"
            ].cpu().numpy().astype(np.float32)
        src_count = dd["src_count"]
        node_corrs, _rank, _sim = official_matching(emb, src_count)
        row = {"pair_id": pair_id, "candidates": len(node_corrs)}
        if not node_corrs:
            counts["zero_candidate"] += 1
            row["outcome"] = "zero_candidate"
            rows.append(row)
            continue
        objects = dd["registration_pts"]
        centres_src, centres_ref = centres
        # official_matching uses GLOBAL ref indices (local+src_count);
        # the consistency layer works in per-graph LOCAL indices
        local_cands = [
            (int(a), int(b) - src_count) for a, b in node_corrs]
        clusters_local = cluster_candidates(
            local_cands, centres_src, centres_ref)
        clusters = [
            [(a, b + src_count) for a, b in cl]
            for cl in clusters_local]
        gt = np.asarray(load_gt_transform(pair_id),
                        dtype=np.float64).reshape(4, 4)
        best = None
        best_rank = -1e18
        any_ok = False
        for cluster in clusters:
            geot = {}
            for src_idx, ref_idx in cluster:
                sp = objects.get(int(src_idx))
                rp = objects.get(int(ref_idx))
                if sp is None or rp is None or len(sp) < 50 \
                        or len(rp) < 50:
                    geot[(src_idx, ref_idx)] = {
                        "status": "insufficient"}
                    continue
                status, output = geotransformer_forward(
                    sp, rp, device=device)
                if status != "ok" or len(
                        output["src_corr_points"]) == 0:
                    geot[(src_idx, ref_idx)] = {"status": status}
                    continue
                geot[(src_idx, ref_idx)] = {
                    "status": "ok",
                    "src_corr": output["src_corr_points"].astype(
                        np.float32),
                    "ref_corr": output["ref_corr_points"].astype(
                        np.float32),
                    "scores": output["corr_scores"].astype(
                        np.float32)}
            counts["hypotheses_run"] += 1
            try:
                registration, _u, _f = combo_registration(
                    geot, cluster)
            except RuntimeError:
                continue
            if registration is None:
                continue
            any_ok = True
            feat, decision, icp = combo_decision(
                dd, registration, pair_id)
            transform = registration["transform"]
            cos_r = (np.trace(
                transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(
                np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(
                transform[:3, 3] - gt[:3, 3]))
            rank = hypothesis_rank(
                cluster, registration["inlier_ratio"],
                icp.fitness,
                1.0 if feat["bidirectional_available"] else 0.0,
                feat["overlap_10cm"])
            if rank > best_rank:
                best_rank = rank
                best = {
                    "rre": rre, "rte": rte,
                    "strict": rre <= STRICT[0]
                    and rte <= STRICT[1],
                    "relaxed": rre <= RELAXED[0]
                    and rte <= RELAXED[1],
                    "accepted":
                        decision["usable_for_reconstruction"],
                    "cluster_size": len(cluster),
                    "n_clusters": len(clusters)}
        if best is None:
            counts["failed"] += 1
            row["outcome"] = "failed"
            rows.append(row)
            continue
        counts["completed"] += 1
        counts["raw_strict"] += int(best["strict"])
        counts["raw_relaxed"] += int(best["relaxed"])
        counts["accepted"] += int(best["accepted"])
        if best["accepted"] and best["strict"]:
            counts["accepted_strict_correct"] += 1
            row["outcome"] = "accepted_strict_correct"
        elif best["accepted"]:
            counts["accepted_strict_error"] += 1
            row["outcome"] = "accepted_strict_error"
        else:
            row["outcome"] = "rejected"
        row.update(best)
        rows.append(row)
    return counts, rows


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/selection.txt")
    pairs = [l.strip() for l in pl.read_text().splitlines()
             if l.strip()]
    samples = []
    for pair_id in pairs:
        dd, _ = build_canonical_pair(pair_id, with_labels=False)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        # object centres (GT-free, from registration surfaces)
        objects = dd["registration_pts"]
        n = dd["tot_obj_pts"].shape[0]
        src_count = dd["src_count"]
        centres_src = {
            i: objects[i].mean(axis=0) for i in range(src_count)}
        centres_ref = {
            j - src_count: objects[j].mean(axis=0)
            for j in range(src_count, n)}
        samples.append((pair_id, dd, anchor_idx,
                        (centres_src, centres_ref)))

    results = {"arms": {}}
    # node metrics all checkpoints
    for arm in ("B", "C", "D"):
        rows = []
        for ck in sorted((OUT / "training" / arm).glob(
                "epoch_*.pt")):
            epoch = int(ck.stem.split("_")[1])
            agg1, pp1 = node_eval(ck, samples, device)
            agg2, pp2 = node_eval(ck, samples, device)
            if agg1 != agg2:
                raise RuntimeError(
                    f"nondeterministic {arm} ep{epoch}")
            rows.append({
                "arm": arm, "epoch": epoch,
                "checkpoint": str(ck.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(ck),
                "metrics": agg1})
            print(f"{arm} ep{epoch}: macroF1 "
                  f"{agg1['macro_node_f1']:.4f} top1 "
                  f"{agg1['macro_top1']:.4f} top5 "
                  f"{agg1['macro_top5']:.4f}", flush=True)
        ranked = sorted(
            rows, key=lambda r: (
                -r["metrics"]["macro_node_f1"],
                -r["metrics"]["macro_top1"],
                -r["metrics"]["macro_top5"],
                -r["metrics"]["margin"], r["epoch"]))
        for rank, row in enumerate(ranked, 1):
            row["node_rank"] = rank
        results["arms"][arm] = ranked
    (OUT / "selection_node_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n")

    # full registration: baseline + top3 per arm
    to_run = [{
        "label": "A_baseline_V5B_ep10", "arm": "A",
        "checkpoint": str(BASELINE_CKPT.relative_to(ROOT)),
        "epoch": 10}]
    for arm in ("B", "C", "D"):
        for row in results["arms"][arm]:
            if row["node_rank"] <= 3:
                to_run.append({
                    "label": f"{arm}_ep{row['epoch']}", "arm": arm,
                    "checkpoint": row["checkpoint"],
                    "epoch": row["epoch"],
                    "node_metrics": row["metrics"]})
    reg = {"runs": {}}
    for entry in to_run:
        print(f"registration {entry['label']}...", flush=True)
        counts, rows = registration_with_consistency(
            ROOT / entry["checkpoint"], samples, device)
        counts["checkpoint_sha256"] = sha256_file(
            ROOT / entry["checkpoint"])
        reg["runs"][entry["label"]] = {
            **entry, "counts": counts, "rows": rows}
        print(json.dumps(counts), flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    (OUT / "selection_registration_metrics.json").write_text(
        json.dumps(reg, indent=2) + "\n")

    base = reg["runs"]["A_baseline_V5B_ep10"]["counts"]
    candidates = []
    for label, run in reg["runs"].items():
        if run["arm"] == "A":
            continue
        c = run["counts"]
        m = run.get("node_metrics", {})
        candidates.append({
            "label": label, "arm": run["arm"],
            "epoch": run["epoch"],
            "key": [0 if c["accepted_strict_error"] == 0 else 1,
                    -c["raw_strict"], -c["raw_relaxed"],
                    -c["accepted_strict_correct"],
                    -m.get("macro_node_f1", 0.0),
                    -m.get("macro_top1", 0.0),
                    -m.get("macro_top5", 0.0), run["epoch"]],
            "counts": c, "node_metrics": m})
    candidates.sort(key=lambda x: x["key"])
    for i, cand in enumerate(candidates, 1):
        cand["selection_rank"] = i
    gates = {}
    for cand in candidates:
        c = cand["counts"]
        m = cand["node_metrics"]
        g = {
            "accepted_strict_error_zero":
                c["accepted_strict_error"] == 0,
            "macro_f1_ge_00844":
                m.get("macro_node_f1", 0) >= 0.0844,
            "macro_top1_ge_00675":
                m.get("macro_top1", 0) >= 0.0675,
            "macro_top5_ge_04330":
                m.get("macro_top5", 0) >= 0.4330,
            "raw_strict_not_below_A":
                c["raw_strict"] >= base["raw_strict"],
            "registration_improvement_vs_A": (
                c["raw_strict"] > base["raw_strict"]
                or c["raw_relaxed"] > base["raw_relaxed"]
                or c["accepted_strict_correct"]
                > base["accepted_strict_correct"]),
            "zero_candidate_not_worse":
                c["zero_candidate"] <= base["zero_candidate"] + 2,
        }
        g["all_pass"] = all(v for v in g.values()
                            if isinstance(v, bool))
        gates[cand["label"]] = g
    (OUT / "checkpoint_ranking.json").write_text(json.dumps({
        "A_baseline_counts": base,
        "candidates": candidates, "gates": gates}, indent=2)
        + "\n")
    print(json.dumps({
        "A_baseline": {k: base[k] for k in (
            "raw_strict", "raw_relaxed", "accepted",
            "accepted_strict_correct",
            "accepted_strict_error")},
        "best": candidates[0]["label"],
        "best_all_pass": gates[candidates[0]["label"]][
            "all_pass"]}, indent=1))


if __name__ == "__main__":
    main()
