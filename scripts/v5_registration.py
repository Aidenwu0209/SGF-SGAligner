"""V5 Part 8b (registration): full pipeline
SGAligner -> GeoTransformer -> RANSAC -> Segment ICP ->
RegistrationDecision (frozen rule B) on selection89 for:
  - arm A baseline (sealed C-ep25 as-is),
  - the top-3 checkpoints of each trained arm (from
    selection_node_metrics.json).

Pre-registered selection key (lexicographic):
  accepted-strict-error==0 (hard) -> max raw strict RR -> max raw
  relaxed RR -> max accepted-strict-correct -> max macro F1 -> max
  macro top1 -> max macro top5 -> min epoch.

Outputs selection_registration_metrics.json with per-pair rows and
gate evaluation vs the A baseline.
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
    combo_registration, combo_decision, effective_decision_config,
)

OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"
INIT_CKPT = (
    ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827/"
    "training/explicit/epoch_00025.pt"
)


def run_selection_registration(ckpt_path, label, samples, device):
    """One full registration pass over selection89."""
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = []
    counts = {
        "requested": len(samples), "structured": 0, "completed": 0,
        "raw_strict": 0, "raw_relaxed": 0, "accepted": 0,
        "accepted_strict_correct": 0, "accepted_strict_error": 0,
        "failed": 0, "zero_candidate": 0}
    for pair_id, dd, anchor_idx in samples:
        with torch.no_grad():
            emb = model(
                batch_for(dd, "explicit", device))["joint"
            ].cpu().numpy().astype(np.float32)
        src_count = dd["src_count"]
        node_corrs, _rank, _sim = official_matching(emb, src_count)
        counts["structured"] += 1
        row = {"pair_id": pair_id,
               "candidates": len(node_corrs)}
        if not node_corrs:
            counts["zero_candidate"] += 1
            row["outcome"] = "zero_candidate"
            rows.append(row)
            continue
        objects = dd["registration_pts"]
        geot = {}
        for src_idx, ref_idx in node_corrs:
            sp = objects.get(int(src_idx))
            rp = objects.get(int(ref_idx))
            if sp is None or rp is None or len(sp) < 50 \
                    or len(rp) < 50:
                geot[(src_idx, ref_idx)] = {
                    "status": "insufficient_raw_points"}
                continue
            status, output = geotransformer_forward(
                sp, rp, device=device)
            if status != "ok" or len(output["src_corr_points"]) == 0:
                geot[(src_idx, ref_idx)] = {"status": status}
                continue
            geot[(src_idx, ref_idx)] = {
                "status": "ok",
                "src_corr": output["src_corr_points"].astype(
                    np.float32),
                "ref_corr": output["ref_corr_points"].astype(
                    np.float32),
                "scores": output["corr_scores"].astype(np.float32)}
        try:
            registration, _used, failures = combo_registration(
                geot, node_corrs)
        except RuntimeError:
            counts["failed"] += 1
            row["outcome"] = "ransac_failure"
            rows.append(row)
            continue
        if registration is None:
            counts["failed"] += 1
            row["outcome"] = "no_correspondences"
            rows.append(row)
            continue
        counts["completed"] += 1
        transform = registration["transform"]
        gt = np.asarray(load_gt_transform(pair_id),
                        dtype=np.float64).reshape(4, 4)
        cos_r = (np.trace(
            transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
        rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
        rte = float(np.linalg.norm(transform[:3, 3] - gt[:3, 3]))
        strict = rre <= STRICT[0] and rte <= STRICT[1]
        relaxed = rre <= RELAXED[0] and rte <= RELAXED[1]
        counts["raw_strict"] += int(strict)
        counts["raw_relaxed"] += int(relaxed)
        _f, decision, icp = combo_decision(dd, registration, pair_id)
        accepted = decision["usable_for_reconstruction"]
        counts["accepted"] += int(accepted)
        if accepted and strict:
            counts["accepted_strict_correct"] += 1
            row["outcome"] = "accepted_strict_correct"
        elif accepted:
            counts["accepted_strict_error"] += 1
            row["outcome"] = "accepted_strict_error"
        else:
            row["outcome"] = "rejected"
        row.update({
            "rre": rre, "rte": rte, "strict": strict,
            "relaxed": relaxed, "accepted": accepted,
            "icp_fitness": icp.fitness,
            "ransac_inliers": registration["inliers"],
            "rejection_reasons": decision["rejection_reasons"]})
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
        samples.append((pair_id, dd, anchor_idx))

    node = json.loads(
        (OUT / "selection_node_metrics.json").read_text())
    to_run = [{
        "label": "A_baseline_Cep25",
        "checkpoint": str(INIT_CKPT.relative_to(ROOT)),
        "epoch": 25, "arm": "A"}]
    for arm in ("B", "C"):
        for row in node["arms"][arm]:
            if row["rank"] <= 3:
                to_run.append({
                    "label": f"{arm}_ep{row['epoch']}",
                    "checkpoint": row["checkpoint"],
                    "epoch": row["epoch"], "arm": arm,
                    "node_metrics": row["metrics"]})

    results = {"decision_config": effective_decision_config(),
               "runs": {}}
    for entry in to_run:
        print(f"running {entry['label']} full registration...",
              flush=True)
        counts, rows = run_selection_registration(
            ROOT / entry["checkpoint"], entry["label"], samples,
            device)
        counts["checkpoint_sha256"] = hashlib.sha256(
            (ROOT / entry["checkpoint"]).read_bytes()).hexdigest()
        results["runs"][entry["label"]] = {
            **entry, "counts": counts, "rows": rows}
        print(json.dumps(counts), flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    (OUT / "selection_registration_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n")

    # pre-registered selection key over B/C candidates
    base = results["runs"]["A_baseline_Cep25"]["counts"]
    candidates = []
    for label, run in results["runs"].items():
        if run["arm"] == "A":
            continue
        c = run["counts"]
        m = run.get("node_metrics", {})
        candidates.append({
            "label": label, "arm": run["arm"],
            "epoch": run["epoch"],
            "key": [
                0 if c["accepted_strict_error"] == 0 else 1,
                -c["raw_strict"], -c["raw_relaxed"],
                -c["accepted_strict_correct"],
                -m.get("macro_node_f1", 0.0),
                -m.get("macro_top1", 0.0),
                -m.get("macro_top5", 0.0), run["epoch"]],
            "counts": c, "node_metrics": m})
    candidates.sort(key=lambda x: x["key"])
    for i, cand in enumerate(candidates, 1):
        cand["selection_rank"] = i
    # hard gates vs baseline
    gates = {}
    for cand in candidates:
        c = cand["counts"]
        m = cand["node_metrics"]
        gates[cand["label"]] = {
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
                c["zero_candidate"]
                <= base["zero_candidate"] + 2,
        }
        gates[cand["label"]]["all_pass"] = all(
            v for k, v in gates[cand["label"]].items()
            if isinstance(v, bool))
    (OUT / "checkpoint_ranking.json").write_text(json.dumps({
        "selection_key": [
            "accepted_strict_error==0", "max raw strict RR",
            "max raw relaxed RR", "max accepted-strict-correct",
            "max macro F1", "max macro top1", "max macro top5",
            "min epoch"],
        "A_baseline_counts": base,
        "candidates": candidates, "gates": gates}, indent=2) + "\n")
    print(json.dumps({
        "A_baseline": {
            k: base[k] for k in (
                "raw_strict", "raw_relaxed", "accepted",
                "accepted_strict_correct",
                "accepted_strict_error", "zero_candidate")},
        "best_candidate": candidates[0]["label"],
        "best_all_pass": gates[candidates[0]["label"]]["all_pass"]},
        indent=1))


if __name__ == "__main__":
    main()
