"""V4-Fix-Seal Parts 9+10: one calibration per frozen winner (with the
incumbent recomputed under the SAME frozen semantics on the SAME
canonical inputs — no口径 mixing), paired comparison, and the
controlled fixed12 safety reproduction for the final winner.

fixed12 categories are STRICTLY separated (accepted-strict-correct /
accepted-strict-error / rejected / failed / zero-candidate).  With no
error-side samples the separation capacity is NOT_EVALUABLE and
ready_for_veto stays false — "0 errors" only means "not observed this
round", never a proof of veto generalisability.
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
from inference import official_matching  # noqa: E402
from adapters.sgf.data_sources import load_anchor_ids  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from v3b_cache_runner import fusion_offline  # noqa: E402
from v4seal_metrics import per_pair_node_metrics, aggregate  # noqa: E402

OUT = ROOT / "outputs/official_sgaligner_v4_fix_seal_20260828"
OFFICIAL_CKPT = ROOT / (
    "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar")


def evaluate_embedding(emb, dd, anchors, anchor_idx):
    src_count = dd["src_count"]
    node_corrs, rank_list, _ = official_matching(emb, src_count)
    normed = emb / np.maximum(
        np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sim = normed @ normed.T
    pp = per_pair_node_metrics(
        node_corrs, rank_list, src_count, anchor_idx, sim=sim)
    pp["pair_id"] = dd_pair_id(dd)
    return pp


def dd_pair_id(dd):
    return f"{dd['scene_ids'][0][0]}_to_{dd['scene_ids'][1][0]}" \
        if isinstance(dd["scene_ids"][0], list) else "unknown"


def build_split_cache(pairs):
    cache = {}
    for pair_id in pairs:
        dd, _ = build_canonical_pair(pair_id, with_labels=False)
        anchors = set(load_anchor_ids(pair_id))
        src_map = dd["src_object_id2idx"]
        ref_map = dd["ref_object_id2idx"]
        anchor_idx = {
            (src_map[s], ref_map[r] + dd["src_count"])
            for s, r in anchors if s in src_map and r in ref_map}
        cache[pair_id] = (dd, anchors, anchor_idx)
    return cache


def incumbent_pct_rel(dd, device):
    """Official ckpt, pct+rel fusion, on canonical inputs — the
    incumbent under the frozen semantics."""
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(
        OFFICIAL_CKPT, map_location=device, weights_only=False)
    official = dict(state["model"])
    fusion4 = official.pop("fusion.weight").clone()
    model.load_state_dict(official, strict=False)
    model.eval()
    with torch.no_grad():
        batch = batch_for(dd, "complete", device)
        out = model(batch)
    mods = {m: out[m].cpu().numpy().astype(np.float32)
            for m in ("pct", "rel")}
    return fusion_offline(
        mods, fusion4.cpu().numpy(), "pct+rel", "cpu").astype(
        np.float32)


def evaluate_split(ckpt_path, arm, cache, device, split):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    per_pair = []
    for pair_id, (dd, anchors, anchor_idx) in cache.items():
        with torch.no_grad():
            batch = batch_for(dd, arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(
                np.float32)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count, anchor_idx, sim=sim)
        pp["pair_id"] = pair_id
        per_pair.append(pp)
    return per_pair


def aggregate_of(per_pair):
    return aggregate([
        {"tp": p["tp"], "pred_count": p["pred_count"],
         "anchor_count": p["anchor_count"], "f1": p["f1"],
         "top1_hit": p["top1_hit"], "top1_total": p["top1_total"],
         "top5_hits": p["top5_hits"], "margin": p["margin"]}
        for p in per_pair])


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = ROOT / ("outputs/official_sgaligner_migration_fix2_pairlists"
                 "/calibration.txt")
    cal_pairs = [l.strip() for l in pl.read_text().splitlines()
                 if l.strip()]
    print("building canonical calibration90 inputs...", flush=True)
    cal_cache = build_split_cache(cal_pairs)

    winners = {}
    for label in ("B", "C"):
        w = json.loads((OUT / f"winner_{label}.json").read_text())
        winners[label] = (w["winner_epoch"], w["winner_checkpoint"])

    result = {}
    # incumbent under frozen semantics
    inc_pp = []
    for pair_id, (dd, anchors, anchor_idx) in cal_cache.items():
        emb = incumbent_pct_rel(dd, device)
        src_count = dd["src_count"]
        node_corrs, rank_list, _ = official_matching(emb, src_count)
        normed = emb / np.maximum(
            np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        sim = normed @ normed.T
        pp = per_pair_node_metrics(
            node_corrs, rank_list, src_count, anchor_idx, sim=sim)
        pp["pair_id"] = pair_id
        inc_pp.append(pp)
    inc_cal = aggregate_of(inc_pp)
    result["incumbent_pct_rel"] = {
        "definition": ("official checkpoint, pct+rel fusion, "
                       "canonical inputs, frozen macro/micro "
                       "semantics — same口径 as the candidates"),
        "calibration90": inc_cal}
    print("incumbent cal:", json.dumps(inc_cal))

    for label, (epoch, ckpt_rel) in winners.items():
        arm = "complete" if label == "B" else "explicit"
        per_pair = evaluate_split(
            ROOT / ckpt_rel, arm, cal_cache, device, "calibration90")
        agg = aggregate_of(per_pair)
        result[f"winner_{label}"] = {
            "arm": arm, "epoch": epoch, "checkpoint": ckpt_rel,
            "calibration90": agg}
        (OUT / f"calibration_winner_{label}.json").write_text(
            json.dumps({
                "winner": label, "epoch": epoch,
                "calibration90": agg,
                "per_pair": per_pair}, indent=2) + "\n")
        print(label, "winner cal:", json.dumps(agg))

        # paired vs incumbent (per-pair f1)
        inc_f1 = {p["pair_id"]: p["f1"] for p in inc_pp}
        deltas = [p["f1"] - inc_f1[p["pair_id"]]
                  for p in per_pair]
        result[f"winner_{label}"]["paired_vs_incumbent"] = {
            "common_pairs": len(deltas),
            "node_f1_delta_mean": float(np.mean(deltas)),
            "improved": sum(1 for d in deltas if d > 0),
            "regressed": sum(1 for d in deltas if d < 0),
            "flat": sum(1 for d in deltas if d == 0)}

    (OUT / "calibration_paired_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n")

    # ---------------- fixed12 safety for the FINAL winner ----------
    # final winner = better of B/C on the pre-registered selection key
    wB = json.loads((OUT / "winner_B.json").read_text())[
        "winner_metrics"]
    wC = json.loads((OUT / "winner_C.json").read_text())[
        "winner_metrics"]
    final_label, final_arm = (
        ("C", "explicit")
        if (wC["macro_node_f1"], wC["macro_top1"],
            wC["macro_top5"], wC["margin"])
        >= (wB["macro_node_f1"], wB["macro_top1"],
            wB["macro_top5"], wB["margin"])
        else ("B", "complete"))
    w = json.loads((OUT / f"winner_{final_label}.json").read_text())
    print("final winner:", final_label, w["winner_epoch"])

    smoke = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/phase6_registration_aware_closure/"
        "smoke12/native")
    fixed_pairs = sorted(
        d.name for d in smoke.iterdir()
        if d.is_dir() and "_to_" in d.name)
    fixed_cache = build_split_cache(fixed_pairs)

    # reuse ONE inference (embedding + GeoT via v3b machinery) then
    # >=3 registration replays with full fields
    from v3b_cache_runner import (
        combo_registration, combo_decision, geotransformer_forward,
    )
    from adapters.sgf.data_sources import load_gt_transform
    from inference import STRICT, RELAXED

    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41,
        attr_dim=164).to(device)
    state = torch.load(ROOT / w["winner_checkpoint"],
                       map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    REPEATS = 3
    categories = {"accepted_strict_correct": 0,
                  "accepted_strict_error": 0, "rejected": 0,
                  "failed": 0, "zero_candidate": 0}
    rows = []
    error_pairs = []
    for pair_id in fixed_pairs:
        dd, anchors, anchor_idx = fixed_cache[pair_id]
        with torch.no_grad():
            batch = batch_for(dd, final_arm, device)
            emb = model(batch)["joint"].cpu().numpy().astype(
                np.float32)
        src_count = dd["src_count"]
        node_corrs, _rank, _sim = official_matching(emb, src_count)
        if not node_corrs:
            categories["zero_candidate"] += REPEATS
            rows.append({"pair_id": pair_id,
                         "category": "zero_candidate",
                         "repeats": REPEATS})
            continue
        objects = dd["registration_pts"]
        id2oid = dd["registration_id2oid"]
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
        gt = np.asarray(load_gt_transform(pair_id),
                        dtype=np.float64).reshape(4, 4)
        outcomes = []
        for k in range(REPEATS):
            try:
                registration, _used, _fail = combo_registration(
                    geot, node_corrs)
            except RuntimeError:
                categories["failed"] += 1
                outcomes.append({"repeat": k, "category": "failed"})
                continue
            if registration is None:
                categories["failed"] += 1
                outcomes.append(
                    {"repeat": k, "category": "failed"})
                continue
            transform = registration["transform"]
            cos_r = (np.trace(
                transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(
                np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(
                transform[:3, 3] - gt[:3, 3]))
            strict = rre <= STRICT[0] and rte <= STRICT[1]
            _f, decision, icp = combo_decision(
                dd, registration, pair_id)
            accepted = decision["usable_for_reconstruction"]
            if accepted and strict:
                cat = "accepted_strict_correct"
            elif accepted:
                cat = "accepted_strict_error"
                error_pairs.append({
                    "pair_id": pair_id, "repeat": k,
                    "rre": rre, "rte": rte})
            else:
                cat = "rejected"
            categories[cat] += 1
            outcomes.append({
                "repeat": k, "category": cat, "rre": rre, "rte": rte,
                "accepted": accepted, "strict": strict,
                "icp_fitness": icp.fitness,
                "rejection_reasons": decision["rejection_reasons"]})
        rows.append({"pair_id": pair_id, "outcomes": outcomes})
    separation = (
        "NOT_EVALUABLE" if categories["accepted_strict_error"] == 0
        else "SEE_ERROR_ROWS")
    safety = {
        "final_winner": {
            "label": final_label, "arm": final_arm,
            "epoch": w["winner_epoch"],
            "checkpoint": w["winner_checkpoint"],
            "checkpoint_sha256": w["winner_checkpoint_sha256"]},
        "repeats_per_pair": REPEATS,
        "summary": {
            **categories,
            "separation_capacity": separation,
            "ready_for_veto": False,
            "wording_rule": ("zero error-accepts means 'not observed "
                             "this round' — NEVER a proof of GT-free "
                             "veto generalisability; no veto training "
                             "or tuning is authorised")},
        "error_rows": error_pairs,
        "rows": rows,
    }
    (OUT / "fixed12_safety.json").write_text(
        json.dumps(safety, indent=2) + "\n")
    print(json.dumps(categories))


if __name__ == "__main__":
    main()
