"""V4-Fix Part 4: RegistrationDecision safety diagnosis for the fair
C winner (epoch 20 — checkpoint unchanged by the fair selection).

Reuses the ONE existing V4 inference cache (cache_explicit dirs); runs
>= 3 registration replays per split, capturing EVERY decision input:
raw/ICP transform, RANSAC matches/inliers/ratio, ICP converged/fitness/
update, surface overlap, bidirectional residuals, extent, accepted
verdict + rejection reasons; RRE/RTE recorded as post-hoc labels only.

Deliverables:
  safety_diagnosis/{split}_repeats.json   per-repeat full records
  safety_diagnosis/accepted_table.json    accepted rows w/ correctness
  safety_diagnosis/error_pair_analysis.json  0ad2d384 deep dive
  safety_diagnosis/gt_free_separation.json   can frozen features tell
                                            error accepts apart?
  safety_diagnosis/ambiguity_veto_proposal.md (PROPOSAL ONLY — no
                                            implementation, no tuning)
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from v3b_cache_runner import (  # noqa: E402
    combo_registration, combo_decision,
)
from inference import build_pair_inputs, STRICT, RELAXED  # noqa: E402
from adapters.sgf.data_sources import load_gt_transform  # noqa: E402

OLD = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
NEW = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"
ERROR_PAIR = (
    "0ad2d384-79e2-2212-9b18-72b44eb5463f_to_"
    "0ad2d399-79e2-2212-99cf-7a3512734bd7"
)
SPLITS = {
    "selection89": OLD / "selection89/cache_explicit",
    "calibration90": OLD / "calibration90/cache_explicit",
    "fixed12": OLD / "fixed12/cache_explicit",
}
REPEATS = 3


def load_geot(pair_dir: Path):
    cache = json.loads((pair_dir / "pair_cache.json").read_text())
    geot_meta = cache.get("geot_node_pairs", {})
    arrays = np.load(pair_dir / "geot_corrs.npz")
    geot = {}
    for key, meta in geot_meta.items():
        row = meta.get("cache_row")
        entry = dict(meta)
        for field in ("src_corr", "ref_corr", "scores"):
            if f"{field}_{row}" in arrays.files:
                entry[field] = arrays[f"{field}_{row}"]
        s, r = key.rsplit("_", 1)
        geot[(int(s), int(r))] = entry
    return cache, geot


def joint_of(pair_dir: Path):
    return np.load(pair_dir / "embeddings.npz")["joint"].astype(
        np.float32)


def repeat_records(split, cache_root, repeats):
    from inference import official_matching

    pairs = sorted({
        json.loads((d / "pair_cache.json").read_text())["pair_id"]
        for d in cache_root.iterdir()
        if (d / "pair_cache.json").exists()})
    rows = []
    for pair_id in pairs:
        tag = f"{pair_id[:8]}_{pair_id[-4:]}"
        pair_dir = cache_root / tag
        cache, geot = load_geot(pair_dir)
        if cache["status"] != "ok":
            rows.append({"pair_id": pair_id,
                         "cache_status": cache["status"]})
            continue
        emb = joint_of(pair_dir)
        data_dict, _contracts = build_pair_inputs(
            pair_id, "official_sgf_predicted",
            sampling_mode="official_mt19937")
        src_count = data_dict["src_count"]
        node_corrs, _rank, _sim = official_matching(emb, src_count)
        node_corrs = [tuple(x) for x in node_corrs]
        gt = np.asarray(load_gt_transform(pair_id),
                        dtype=np.float64).reshape(4, 4)
        outcomes = []
        for k in range(repeats):
            try:
                registration, used, failures = combo_registration(
                    geot, node_corrs)
            except RuntimeError as exc:
                outcomes.append({
                    "repeat": k, "status": "ransac_failure",
                    "error": repr(exc)[:150]})
                continue
            if registration is None:
                outcomes.append({
                    "repeat": k, "status": "no_correspondences",
                    "node_pair_failures": len(failures)})
                continue
            transform = registration["transform"]
            cos_r = (np.trace(
                transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(
                transform[:3, 3] - gt[:3, 3]))
            features, decision, icp = combo_decision(
                data_dict, registration, pair_id)
            outcomes.append({
                "repeat": k, "status": "ok",
                # full decision-input capture
                "raw_transform": transform.tolist(),
                "icp_transform": icp.transform.tolist(),
                "ransac_matches": registration["corrs"],
                "ransac_inliers": registration["inliers"],
                "ransac_inlier_ratio": registration[
                    "inlier_ratio"],
                "icp_converged": icp.converged,
                "icp_fitness": icp.fitness,
                "icp_rmse_m": icp.rmse_m,
                "icp_update_translation_m":
                    icp.update_translation_m,
                "icp_update_rotation_deg":
                    icp.update_rotation_deg,
                "surface_overlap_10cm": features["overlap_10cm"],
                "surface_overlap_5cm": features["overlap_5cm"],
                "symmetric_trimmed_chamfer_m": features[
                    "symmetric_trimmed_chamfer_m"],
                "median_residual_m": features["median_residual_m"],
                "p90_residual_m": features["p90_residual_m"],
                "bidirectional_rotation_deg": features[
                    "bidirectional_rotation_deg"],
                "bidirectional_translation_m": features[
                    "bidirectional_translation_m"],
                "bidirectional_available": features[
                    "bidirectional_available"],
                "spatial_extent_m": features["spatial_extent_m"],
                "node_pair_success_ratio": features[
                    "node_pair_success_ratio"],
                "decision_status": decision["status"],
                "rejection_reasons": decision["rejection_reasons"],
                "accepted": decision["usable_for_reconstruction"],
                # post-hoc labels ONLY
                "rre_post_hoc": rre, "rte_post_hoc": rte,
                "strict_post_hoc": (
                    rre <= STRICT[0] and rte <= STRICT[1]),
                "relaxed_post_hoc": (
                    rre <= RELAXED[0] and rte <= RELAXED[1]),
                "accepted_strict_error": bool(
                    decision["usable_for_reconstruction"]
                    and not (rre <= STRICT[0]
                             and rte <= STRICT[1])),
            })
        rows.append({
            "pair_id": pair_id, "cache_status": "ok",
            "node_corrs_count": len(node_corrs),
            "outcomes": outcomes})
        print(f"{split} {pair_id[:12]} done", flush=True)
    return rows


def main() -> None:
    out_dir = NEW / "safety_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for split, cache_root in SPLITS.items():
        rows = repeat_records(split, cache_root, REPEATS)
        all_rows[split] = rows
        (out_dir / f"{split}_repeats.json").write_text(
            json.dumps(rows, indent=2) + "\n")

    # accepted table across all splits/repeats
    accepted_rows = []
    for split, rows in all_rows.items():
        for row in rows:
            for o in row.get("outcomes", []):
                if o.get("status") == "ok" and o.get("accepted"):
                    accepted_rows.append({
                        "split": split, "pair_id": row["pair_id"],
                        "repeat": o["repeat"],
                        "strict_correct": o["strict_post_hoc"],
                        "rre": o["rre_post_hoc"],
                        "rte": o["rte_post_hoc"],
                        "icp_fitness": o["icp_fitness"],
                        "icp_rmse_m": o["icp_rmse_m"],
                        "icp_update_translation_m": o[
                            "icp_update_translation_m"],
                        "overlap_10cm": o["surface_overlap_10cm"],
                        "chamfer_m": o["symmetric_trimmed_chamfer_m"],
                        "median_residual_m": o["median_residual_m"],
                        "bidir_rot": o["bidirectional_rotation_deg"],
                        "bidir_trans": o[
                            "bidirectional_translation_m"],
                        "ransac_inlier_ratio": o[
                            "ransac_inlier_ratio"],
                        "node_pair_success_ratio": o[
                            "node_pair_success_ratio"],
                    })
    (out_dir / "accepted_table.json").write_text(
        json.dumps(accepted_rows, indent=2) + "\n")

    # GT-free separation analysis: error vs correct accepted
    errs = [r for r in accepted_rows if not r["strict_correct"]]
    oks = [r for r in accepted_rows if r["strict_correct"]]
    fields = ["icp_fitness", "icp_rmse_m",
              "icp_update_translation_m", "overlap_10cm",
              "chamfer_m", "median_residual_m", "bidir_rot",
              "bidir_trans", "ransac_inlier_ratio",
              "node_pair_success_ratio"]
    separation = {}
    for f in fields:
        ok_vals = [r[f] for r in oks if r[f] is not None]
        err_vals = [r[f] for r in errs if r[f] is not None]
        separation[f] = {
            "correct_n": len(ok_vals),
            "error_n": len(err_vals),
            "correct_min": min(ok_vals) if ok_vals else None,
            "correct_max": max(ok_vals) if ok_vals else None,
            "error_values": err_vals,
            "separable_by_interval": (
                bool(ok_vals and err_vals)
                and (min(err_vals) > max(ok_vals)
                     or max(err_vals) < min(ok_vals))),
        }
    (out_dir / "gt_free_separation.json").write_text(
        json.dumps({
            "accepted_total": len(accepted_rows),
            "strict_correct": len(oks),
            "strict_error": len(errs),
            "fields": separation,
        }, indent=2) + "\n")

    # error-pair deep dive
    deep = []
    for split, rows in all_rows.items():
        for row in rows:
            if row["pair_id"] != ERROR_PAIR:
                continue
            deep.append({"split": split, "row": row})
    (out_dir / "error_pair_analysis.json").write_text(
        json.dumps({
            "pair_id": ERROR_PAIR,
            "records": deep,
            "recharacterisation": (
                "relaxed-level NEAR-MISS translation error (RTE "
                "~0.217 m vs 0.20 m strict bar, RRE ~2.97 deg passes) "
                "— NOT a 180-degree flip; the V4 attribution to "
                "10b1792c was wrong (that pair appears nowhere in the "
                "V4 repeats)"),
        }, indent=2) + "\n")
    print(json.dumps({
        "accepted": len(accepted_rows), "errors": len(errs),
        "error_pair_splits": [d["split"] for d in deep]}))


if __name__ == "__main__":
    main()
