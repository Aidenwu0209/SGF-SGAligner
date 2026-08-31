"""Fix-2 Seal: single-process cached-inference runner + offline rule replay.

Each pair runs the official SGAligner + GeoTransformer pipeline ONCE
(single process, models loaded once); every raw feature the three
RegistrationDecision rules need is persisted to a per-pair cache.  The
A/B/C comparison then happens purely offline over that one cache — no
model, RANSAC, GeoT or ICP re-execution per rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from inference import (  # noqa: E402
    OFFICIAL_SNAPSHOT, build_pair_inputs, official_forward,
    official_matching, official_registration, geotransformer_forward,
    OracleGraphSource_anchor_segments,
)
from adapters.sgf.data_sources import (  # noqa: E402
    load_anchor_ids, load_gt_transform, load_oracle_anchor_ids,
    oracle_gt_transform,
)
from safety import decision_features as dfx  # noqa: E402
from safety.registration_decision import spatial_support  # noqa: E402

STRICT = (5.0, 0.20)
RELAXED = (10.0, 0.30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


CHECKPOINT_OVERRIDE = os.environ.get("SGALIGNER_CKPT")


def _load_official_snapshot():
    if CHECKPOINT_OVERRIDE:
        return Path(CHECKPOINT_OVERRIDE)
    return Path(OFFICIAL_SNAPSHOT)


def run_pair_cached(pair_id: str, mode: str, out_dir: Path,
                    device: str = "cuda") -> dict:
    """One inference per pair; persist ALL raw decision evidence."""
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {
        "pair_id": pair_id, "mode": mode,
        "checkpoint_sha256": sha256(_load_official_snapshot()),
        "seed": 42, "device": device,
    }
    try:
        data_dict, contracts = build_pair_inputs(pair_id, mode)
        (out_dir / "graph_input.json").write_text(json.dumps({
            "src_objects": int(data_dict["graph_per_obj_count"][0]),
            "ref_objects": int(data_dict["graph_per_obj_count"][1]),
            "provenance": data_dict["provenance"],
        }, indent=2, default=str) + "\n")

        import inference as _inf
        _saved = _inf.OFFICIAL_SNAPSHOT
        _inf.OFFICIAL_SNAPSHOT = str(_load_official_snapshot())
        try:
            embedding, ckpt_epoch = official_forward(data_dict, mode, device)
        finally:
            _inf.OFFICIAL_SNAPSHOT = _saved
        np.savez_compressed(
            out_dir / "embeddings.npz", embedding=embedding
        )
        node_corrs, rank_list, sim = official_matching(
            embedding, data_dict["src_count"]
        )
        (out_dir / "node_matches.json").write_text(json.dumps({
            "node_corrs": [[int(a), int(b)] for a, b in node_corrs],
        }) + "\n")

        registration, used_pairs, node_failures = official_registration(
            data_dict, node_corrs, mode, device=device, pair_id=pair_id
        )

        if registration is not None:
            np.savetxt(out_dir / "raw_transform.txt",
                       registration["transform"], fmt="%.10f")
            np.savez_compressed(
                out_dir / "geot_corrs.npz",
                src=registration["src_corr_points"].astype(np.float32),
                ref=registration["ref_corr_points"].astype(np.float32),
            )

            # ---- full raw evidence for offline A/B/C replay --------
            objects = data_dict["registration_pts"]
            src_surface = np.concatenate(
                [objects[int(a)] for a, _b in used_pairs]
            )
            ref_surface = np.concatenate(
                [objects[int(b)] for _a, b in used_pairs]
            )
            src_bary = np.asarray(
                [objects[int(a)].mean(axis=0) for a, _b in used_pairs]
            )
            extent, second = spatial_support(src_bary)
            evidence = dfx.surface_evidence(
                src_surface, ref_surface, registration["transform"],
                seed=42,
            )
            icp = dfx.segment_icp(
                src_surface, ref_surface, registration["transform"],
                seed=42,
            )
            np.savetxt(out_dir / "icp_transform.txt",
                       icp.transform, fmt="%.10f")
            bidir_available = True
            try:
                t_rs = dfx.segment_icp(
                    ref_surface, src_surface,
                    np.linalg.inv(registration["transform"]), seed=43,
                ).transform
                bidir_r, bidir_t = dfx.transform_discrepancy(
                    registration["transform"], t_rs
                )
            except Exception as exc:  # noqa: BLE001
                bidir_available = False
                bidir_r = bidir_t = None
                cache["bidirectional_error"] = repr(exc)

            successful = len(used_pairs)
            failed = len(node_failures)
            total = successful + failed
            raw_features = {
                "ransac_inliers": registration["inliers"],
                "ransac_inlier_ratio": registration["inlier_ratio"],
                "spatial_extent_m": float(extent),
                "spatial_second_axis_m": float(second),
                "icp_update_translation_m": icp.update_translation_m,
                "icp_update_rotation_deg": icp.update_rotation_deg,
                "icp_converged": icp.converged,
                "icp_fitness": icp.fitness,
                "icp_rmse_m": icp.rmse_m,
                "overlap_5cm": evidence.overlap_5cm,
                "overlap_10cm": evidence.overlap_10cm,
                "overlap_ratio": evidence.overlap_10cm,
                "symmetric_trimmed_chamfer_m":
                    evidence.symmetric_trimmed_chamfer_m,
                "median_residual_m": evidence.median_residual_m,
                "p90_residual_m": evidence.p90_residual_m,
                "bidirectional_available": bidir_available,
                "bidirectional_rotation_deg": bidir_r,
                "bidirectional_translation_m": bidir_t,
                "node_pair_success_ratio": (
                    successful / total if total else 0.0
                ),
                "successful_node_pairs": successful,
                "failed_node_pairs": failed,
                "_provenance": {
                    "surfaces": "matched-object FULL registration points",
                    "surface_points": {
                        "src": evidence.n_src_points,
                        "ref": evidence.n_ref_points,
                    },
                    "surface_seed": evidence.seed,
                    "units": "metres / degrees / ratios",
                },
            }

            # post-hoc labels (GT only here)
            if mode == "official_oracle":
                src_scan, ref_scan = pair_id.split("_to_")
                src_segments = OracleGraphSource_anchor_segments(src_scan)
                ref_segments = OracleGraphSource_anchor_segments(ref_scan)
                gt = oracle_gt_transform(
                    src_scan, ref_scan, src_segments, ref_segments
                )
            else:
                gt = load_gt_transform(pair_id)
            t_world = registration["transform"]
            cos_r = (np.trace(t_world[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(t_world[:3, 3] - gt[:3, 3]))
            cache.update({
                "status": "ok",
                "registration_status": "hypothesis_generated",
                "raw_transform_sha": sha256(
                    out_dir / "raw_transform.txt"
                ),
                "rre": rre, "rte": rte,
                "strict": bool(rre <= STRICT[0] and rte <= STRICT[1]),
                "relaxed": bool(rre <= RELAXED[0] and rte <= RELAXED[1]),
                "raw_features": raw_features,
                "node_pair_failures": node_failures,
                "node_pairs_used": [
                    [int(a), int(b)] for a, b in used_pairs
                ],
            })
        else:
            cache.update({
                "status": "ok",
                "registration_status": "no_hypothesis",
                "strict": False, "relaxed": False,
                "failure_stage_counts": {
                    stage: sum(
                        1 for f in node_failures if f["stage"] == stage
                    )
                    for stage in sorted({
                        f["stage"] for f in node_failures
                    })
                },
                "node_pair_failures": node_failures,
            })
        cache["elapsed_s"] = time.monotonic() - started
        (out_dir / "pair_cache.json").write_text(
            json.dumps(cache, indent=2, default=_json_default) + "\n"
        )
        return {"status": "ok", "pair_id": pair_id}
    except torch.cuda.OutOfMemoryError as exc:
        cache.update({
            "status": "structured_failure",
            "failure_type": "cuda_oom",
            "error": repr(exc)[:300],
            "strict": False, "relaxed": False,
            "elapsed_s": time.monotonic() - started,
        })
        (out_dir / "pair_cache.json").write_text(
            json.dumps(cache, indent=2, default=_json_default) + "\n"
        )
        return {"status": "structured_failure", "failure_type": "cuda_oom"}
    except Exception as exc:  # noqa: BLE001 - typed + traceback retained
        import traceback

        cache.update({
            "status": "structured_failure",
            "failure_type": type(exc).__name__,
            "error": repr(exc)[:300],
            "traceback": traceback.format_exc()[-2000:],
            "strict": False, "relaxed": False,
            "elapsed_s": time.monotonic() - started,
        })
        (out_dir / "pair_cache.json").write_text(
            json.dumps(cache, indent=2, default=_json_default) + "\n"
        )
        return {"status": "structured_failure",
                "failure_type": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("selection", "calibration",
                                            "fixed12"), required=True)
    parser.add_argument("--mode", default="official_sgf_predicted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="DEBUG ONLY; seal runs must not use it")
    args = parser.parse_args()

    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    if args.split == "fixed12":
        smoke = Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/phase6_registration_aware_closure/"
            "smoke12/native"
        )
        pairs = sorted(
            d.name for d in smoke.iterdir()
            if d.is_dir() and "_to_" in d.name
        )
    else:
        pairs = [
            line.strip()
            for line in (pairlists / f"{args.split}.txt").read_text()
            .splitlines() if line.strip()
        ]
    if args.limit:
        pairs = pairs[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs_run.txt").write_text("\n".join(pairs) + "\n")
    (args.out / "pairs_run.sha256").write_text(
        sha256(args.out / "pairs_run.txt") + "\n"
    )
    counters = {"ok": 0, "structured_failure": 0, "oom": 0}
    for index, pair in enumerate(pairs):
        tag = f"{pair[:8]}_{pair[-4:]}"
        result = run_pair_cached(
            pair, args.mode, args.out / tag, device="cuda"
        )
        counters[result["status"]] += 1
        if result.get("failure_type") == "cuda_oom":
            counters["oom"] += 1
        print(f"[{index+1}/{len(pairs)}] {tag} {result['status']}"
              f"{'/' + result.get('failure_type', '') if result.get('failure_type') else ''}",
              flush=True)
        torch.cuda.empty_cache()
    (args.out / "run_summary.json").write_text(
        json.dumps({
            "split": args.split, "mode": args.mode,
            "requested": len(pairs), **counters,
        }, indent=2) + "\n"
    )
    print(json.dumps({
        "split": args.split, "mode": args.mode,
        "requested": len(pairs), **counters,
    }))


if __name__ == "__main__":
    main()
