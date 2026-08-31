"""V4 registration repeats: sample the pygcransac estimator variance.

For each pair in a cache: recompute the (deterministic) matching from
the cached embeddings, pool the cached per-node-pair GeoT
correspondences, then repeat k times: RANSAC -> ICP -> surface
evidence -> RegistrationDecision (frozen rule B).  The model and
GeoTransformer are NEVER re-run; ICP re-executes BY DESIGN because the
repeats exist to quantify the unseedable RANSAC estimator distribution
(pre-registered Phase 7 rule).

Works for arm A (incumbent; V3 official cache, combo recomputed via
official fusion slicing) and arms B/C (candidate caches with a direct
joint embedding).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from v3b_cache_runner import (  # noqa: E402
    combo_registration, combo_decision, fusion_offline, COMBOS,
)
from inference import (  # noqa: E402
    build_pair_inputs, official_matching, STRICT, RELAXED,
)
from adapters.sgf.data_sources import (  # noqa: E402
    load_anchor_ids, load_gt_transform,
)


def load_geot_cache(pair_dir: Path):
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


def embedding_for(cache: dict, pair_dir: Path, combo: str):
    emb = np.load(pair_dir / "embeddings.npz")
    if combo == "candidate":
        return emb["joint"].astype(np.float32)
    if combo in COMBOS:
        mods = {m: emb[m].astype(np.float32)
                for m in combo.split("+")}
        fusion4 = emb["fusion_weight4"]
        return fusion_offline(mods, fusion4, combo, "cpu")
    raise ValueError(f"unknown combo {combo}")


def repeat_pair(pair_id, pair_dir, combo, repeats, decision_cfg):
    cache, geot = load_geot_cache(pair_dir)
    if cache["status"] != "ok":
        return {"pair_id": pair_id, "status": cache["status"],
                "failure_type": cache.get("failure_type")}
    emb = embedding_for(cache, pair_dir, combo)
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
        except RuntimeError:
            outcomes.append({
                "repeat": k, "status": "ransac_failure",
                "strict": False, "relaxed": False, "accepted": False,
                "rre": None, "rte": None})
            continue
        if registration is None:
            outcomes.append({
                "repeat": k, "status": "no_correspondences",
                "strict": False, "relaxed": False, "accepted": False,
                "rre": None, "rte": None})
            continue
        transform = registration["transform"]
        cos_r = (np.trace(
            transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
        rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
        rte = float(np.linalg.norm(transform[:3, 3] - gt[:3, 3]))
        _features, decision, _icp = combo_decision(
            data_dict, registration, pair_id)
        outcomes.append({
            "repeat": k, "status": "ok", "rre": rre, "rte": rte,
            "strict": rre <= STRICT[0] and rte <= STRICT[1],
            "relaxed": rre <= RELAXED[0] and rte <= RELAXED[1],
            "accepted": decision["usable_for_reconstruction"],
            "accepted_strict_error": bool(
                decision["usable_for_reconstruction"]
                and not (rre <= STRICT[0] and rte <= STRICT[1])),
        })
    strict_flags = [o["strict"] for o in outcomes]
    return {
        "pair_id": pair_id, "status": "ok",
        "node_corrs_count": len(node_corrs),
        "outcomes": outcomes,
        "strict_flips": len(set(strict_flags)) > 1,
        "accepted_strict_errors": sum(
            1 for o in outcomes if o.get("accepted_strict_error")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--combo", required=True,
                        help="candidate | pct+rel | pct+gat+rel | ...")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True,
                        help="A_incumbent | B_complete | C_explicit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # enumerate pair dirs (the actual caches); pairs_run.txt may list
    # only a surgical-rerun subset and must not shrink the split
    pairs = sorted({
        json.loads((d / "pair_cache.json").read_text())["pair_id"]
        for d in args.cache_dir.iterdir()
        if (d / "pair_cache.json").exists()
    })
    rows = []
    for index, pair in enumerate(pairs):
        tag = f"{pair[:8]}_{pair[-4:]}"
        row = repeat_pair(
            pair, args.cache_dir / tag, args.combo, args.repeats, None)
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"{index+1}/{len(pairs)}", flush=True)

    per_repeat = []
    for k in range(args.repeats):
        oks = [r for r in rows if r["status"] == "ok"]
        per_repeat.append({
            "repeat": k,
            "completed": sum(
                1 for r in oks if r["outcomes"][k]["status"] == "ok"),
            "strict": sum(
                1 for r in oks if r["outcomes"][k].get("strict")),
            "relaxed": sum(
                1 for r in oks if r["outcomes"][k].get("relaxed")),
            "accepted": sum(
                1 for r in oks if r["outcomes"][k].get("accepted")),
            "accepted_strict_error": sum(
                r["outcomes"][k].get("accepted_strict_error", False)
                for r in oks if r["status"] == "ok"),
        })
    oks = [r for r in rows if r["status"] == "ok"]
    strict_counts = [p["strict"] for p in per_repeat]
    relaxed_counts = [p["relaxed"] for p in per_repeat]
    accepted_counts = [p["accepted"] for p in per_repeat]
    rel_cache = str(args.cache_dir.resolve().relative_to(ROOT)) \
        if str(args.cache_dir.resolve()).startswith(str(ROOT)) \
        else str(args.cache_dir)
    summary = {
        "label": args.label, "combo": args.combo,
        "cache_dir": rel_cache,
        "repeats": args.repeats, "requested": len(rows),
        "structured_failures": len(rows) - len(oks),
        "per_repeat": per_repeat,
        "strict": {
            "min": min(strict_counts), "median": float(
                np.median(strict_counts)),
            "max": max(strict_counts)},
        "relaxed": {
            "min": min(relaxed_counts), "median": float(
                np.median(relaxed_counts)),
            "max": max(relaxed_counts)},
        "accepted": {
            "min": min(accepted_counts), "median": float(
                np.median(accepted_counts)),
            "max": max(accepted_counts)},
        "accepted_strict_error_total": sum(
            p["accepted_strict_error"] for p in per_repeat),
        "ambiguity_pairs": [
            r["pair_id"] for r in oks if r["strict_flips"]],
        "rre_distribution": {
            "median_of_ok": float(np.median([
                o["rre"] for r in oks for o in r["outcomes"]
                if o.get("rre") is not None])) if any(
                o.get("rre") is not None
                for r in oks for o in r["outcomes"]) else None,
            "p90_of_ok": float(np.percentile([
                o["rre"] for r in oks for o in r["outcomes"]
                if o.get("rre") is not None], 90)) if any(
                o.get("rre") is not None
                for r in oks for o in r["outcomes"]) else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
