"""V3-B: single-inference cache runner (SGF-predicted, official_mt19937).

ONE expensive backbone per pair:
  - ONE SGAligner forward (all modality embeddings cached);
  - ONE GeoTransformer pass per NODE PAIR over the UNION of all
    ablation combos' candidate node pairs;
  - per combo, ONCE each: pooled RANSAC, segment ICP, surface
    evidence, bidirectional ICP, RegistrationDecision (frozen rule B).

The offline ablation replay (v3b_replay.py) only READS this cache —
it never re-runs the model, GeoTransformer, or ICP.  Decision rule
conflict resolution (task section 7): seal final_report.md says
chosen_rule=B, final_report_data.json field says A with basis "B
retained as the stricter superset" — the effective config actually
executed here is rule B with the current RULE_THRESHOLDS, recorded
verbatim with a SHA and shared by every combo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get(
    "SGALIGNER_CODE_ROOT", Path(__file__).resolve().parents[1])).resolve()
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

from inference import (  # noqa: E402
    OFFICIAL_SNAPSHOT, build_pair_inputs, official_matching,
    geotransformer_forward, REG_K, NUM_P2P, STRICT, RELAXED,
)
from safety import decision_features as dfx  # noqa: E402
from safety.registration_decision import spatial_support  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

COMBOS = ("pct", "rel", "gat", "pct+rel", "pct+gat+rel")
MODULE_INDEX = {"pct": 0, "gat": 1, "rel": 2, "attr": 3}
DECISION_RULE = "B"
SEED = 42


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
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


def effective_decision_config() -> dict:
    cfg = {
        "rule": DECISION_RULE,
        "rule_evaluator": f"evaluate_rule_{DECISION_RULE.lower()}",
        "thresholds": dict(dfx.RULE_THRESHOLDS),
        "conflict_resolution": (
            "seal final_report.md chosen_rule=B vs "
            "final_report_data.json field A with basis 'B retained as "
            "the stricter superset' -> effective executed config = B"
        ),
    }
    cfg["sha256"] = sha256_bytes(
        json.dumps(cfg, sort_keys=True).encode())
    return cfg


def load_model_and_fusion(device):
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(
        OFFICIAL_SNAPSHOT, map_location=device, weights_only=False)
    official = dict(state["model"])
    fusion4 = official.pop("fusion.weight").clone()  # [4,1] official
    model.load_state_dict(official, strict=False)
    with torch.no_grad():
        model.fusion.weight.copy_(fusion4[:3])
    model.eval()
    return model, fusion4.cpu().numpy(), state.get("epoch")


def fusion_offline(embs: dict, fusion4: np.ndarray, combo: str,
                   device) -> np.ndarray:
    """Official MultiModalFusion semantics over a module subset."""
    mods = combo.split("+")
    rows = [MODULE_INDEX[m] for m in mods]
    w = torch.from_numpy(fusion4[rows]).float().to(device)
    wn = torch.softmax(w, dim=0)
    outs = []
    for i, mod in enumerate(mods):
        emb = torch.from_numpy(embs[mod]).float().to(device)
        outs.append(wn[i] * F.normalize(emb, dim=1))
    return torch.cat(outs, dim=1).cpu().numpy()


def ransac_from_pooled(src_all: np.ndarray, ref_all: np.ndarray):
    """Exact official_registration pooling + pygcransac composition."""
    import pygcransac

    corrs = np.concatenate([src_all, ref_all], axis=1).astype(np.float64)
    shifted = corrs - corrs.min(axis=0)
    est_transform, _inliers = pygcransac.findRigidTransform(
        np.ascontiguousarray(shifted),
        probabilities=[],
        threshold=0.05, neighborhood_size=4.0, sampler=1,
        min_iters=1000, max_iters=10000,
        spatial_coherence_weight=0.0, use_space_partitioning=True,
        neighborhood=0, conf=0.999, use_sprt=False,
    )
    if not isinstance(est_transform, np.ndarray) \
            or est_transform.shape != (4, 4):
        raise RuntimeError("pygcransac returned no rigid transform")
    min_coordinates = np.min(corrs, axis=0)
    T1 = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [-min_coordinates[0], -min_coordinates[1],
         -min_coordinates[2], 1],
    ])
    T2inv = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [min_coordinates[3], min_coordinates[4],
         min_coordinates[5], 1],
    ])
    transform = (T1 @ est_transform @ T2inv).T
    residual = np.linalg.norm(
        src_all @ transform[:3, :3].T + transform[:3, 3] - ref_all,
        axis=1,
    )
    inliers = int((residual <= 0.10).sum())
    return transform, inliers


def combo_registration(cached_geot: dict, node_corrs: list):
    """Pool cached per-node-pair corrs (official capping) + RANSAC."""
    point_src, point_ref = [], []
    used, failures = [], []
    cap = max(NUM_P2P // max(len(node_corrs), 1), 1)
    for src_idx, ref_idx in node_corrs:
        entry = cached_geot.get((int(src_idx), int(ref_idx)))
        if entry is None or entry["status"] != "ok":
            failures.append({
                "src_index": int(src_idx), "ref_index": int(ref_idx),
                "stage": entry["status"] if entry
                else "not_in_union_cache",
            })
            continue
        src_corr = entry["src_corr"]
        ref_corr = entry["ref_corr"]
        scores = entry["scores"]
        if len(src_corr) > cap:
            keep = np.argsort(-scores)[:cap]
            src_corr, ref_corr, scores = (
                src_corr[keep], ref_corr[keep], scores[keep])
        point_src.append(src_corr)
        point_ref.append(ref_corr)
        used.append((int(src_idx), int(ref_idx)))
    if not point_src:
        return None, used, failures
    src_all = np.concatenate(point_src)
    ref_all = np.concatenate(point_ref)
    transform, inliers = ransac_from_pooled(src_all, ref_all)
    return {
        "transform": transform, "inliers": inliers,
        "corrs": int(len(src_all)),
        "inlier_ratio": inliers / max(len(src_all), 1),
        "node_pairs_used": used,
        "node_pair_failures": failures,
    }, used, failures


def combo_decision(data_dict, registration, pair_id):
    """decision_features_full semantics for one combo (rule B)."""
    objects = data_dict["registration_pts"]
    used = registration["node_pairs_used"]
    successful_pairs = len(used)
    failed_pairs = len(registration.get("node_pair_failures", []))
    total_pairs = successful_pairs + failed_pairs
    success_ratio = (
        successful_pairs / total_pairs if total_pairs else 0.0
    )
    src_surface = np.concatenate(
        [objects[int(a)] for a, _b in used])
    ref_surface = np.concatenate(
        [objects[int(b)] for _a, b in used])
    src_bary = np.asarray(
        [objects[int(a)].mean(axis=0) for a, _b in used])
    extent, second = spatial_support(src_bary)
    transform = registration["transform"]
    evidence = dfx.surface_evidence(
        src_surface, ref_surface, transform, seed=SEED)
    icp = dfx.segment_icp(
        src_surface, ref_surface, transform, seed=SEED)
    bidir_rotation = bidir_translation = None
    bidirectional_available = False
    try:
        t_rs = dfx.segment_icp(
            ref_surface, src_surface, np.linalg.inv(transform),
            seed=43,
        ).transform
        bidir_rotation, bidir_translation = dfx.transform_discrepancy(
            transform, t_rs)
        bidirectional_available = True
    except Exception:  # noqa: BLE001 - recorded as unavailable
        bidirectional_available = False
    features = {
        "ransac_inliers": registration["inliers"],
        "ransac_inlier_ratio": registration["inlier_ratio"],
        "spatial_extent_m": float(extent),
        "spatial_second_axis_m": float(second),
        "icp_update_translation_m": icp.update_translation_m,
        "icp_update_rotation_deg": icp.update_rotation_deg,
        "bidirectional_rotation_deg": bidir_rotation,
        "bidirectional_translation_m": bidir_translation,
        "overlap_ratio": evidence.overlap_10cm,
        "icp_converged": icp.converged,
        "overlap_10cm": evidence.overlap_10cm,
        "overlap_5cm": evidence.overlap_5cm,
        "symmetric_trimmed_chamfer_m":
            evidence.symmetric_trimmed_chamfer_m,
        "median_residual_m": evidence.median_residual_m,
        "p90_residual_m": evidence.p90_residual_m,
        "icp_fitness": icp.fitness,
        "icp_rmse_m": icp.rmse_m,
        "node_pair_success_ratio": success_ratio,
        "successful_node_pairs": successful_pairs,
        "failed_node_pairs": failed_pairs,
        "bidirectional_available": bidirectional_available,
    }
    rule_features = dict(features)
    if not bidirectional_available:
        rule_features["bidirectional_rotation_deg"] = 1e9
        rule_features["bidirectional_translation_m"] = 1e9
    violations = dfx.RULE_EVALUATORS[DECISION_RULE](rule_features)
    return features, {
        "status": "accepted" if not violations else "rejected",
        "usable_for_reconstruction": not violations,
        "rejection_reasons": violations,
        "rule": f"fix2-{DECISION_RULE}",
    }, icp


def node_metrics(embedding, src_count, anchors, src_map, ref_map):
    node_corrs, rank_list, _sim = official_matching(
        embedding, src_count)
    anchor_idx = {
        (src_map[s], ref_map[r] + src_count)
        for s, r in anchors if s in src_map and r in ref_map
    }
    pred = set(node_corrs)
    tp = len(pred & anchor_idx)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(anchor_idx) if anchor_idx else 0.0
    # top-1 precision / top-5 recall from the FULL ranking
    top1_hits = top1_total = 0
    top5_hits = 0
    for i in range(src_count):
        ref_ranks = [
            x for x in rank_list[i] if x >= src_count][:5]
        if not ref_ranks:
            continue
        top1_total += 1
        if (i, int(ref_ranks[0])) in anchor_idx:
            top1_hits += 1
        top5_hits += sum(
            1 for x in ref_ranks[:5] if (i, int(x)) in anchor_idx)
    return {
        "node_corrs": [[int(a), int(b)] for a, b in node_corrs],
        "precision": p, "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-12),
        "top1_precision": top1_hits / max(top1_total, 1),
        "top5_recall": (
            top5_hits / len(anchor_idx) if anchor_idx else 0.0),
        "tp": tp, "pred_count": len(pred),
        "anchor_count": len(anchor_idx),
    }


def run_pair_cached(pair_id, out_dir, model, fusion4, ckpt_epoch,
                    device, code_head, model_config, decision_config):
    # GT-labelled anchor/pose loaders are deliberately scoped to the legacy
    # evaluation runner. Importing combo_registration/combo_decision for a
    # predicted inference path must not import a GT transform symbol.
    from adapters.sgf.data_sources import (  # noqa: PLC0415
        load_anchor_ids, load_gt_transform,
    )
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha = sha256_file(Path(OFFICIAL_SNAPSHOT))
    cache = {
        "pair_id": pair_id, "mode": "official_sgf_predicted",
        "sampling_mode": "official_mt19937", "scan_seed": 0,
        "checkpoint_sha256": ckpt_sha, "checkpoint_epoch": ckpt_epoch,
        "seed": SEED, "device": device, "code_head": code_head,
        "model_config": model_config,
        "decision_config": decision_config,
    }
    try:
        data_dict, _contracts = build_pair_inputs(
            pair_id, "official_sgf_predicted",
            sampling_mode="official_mt19937",
        )
        # ---- cache key material ------------------------------------
        input_tensors = {
            "tot_obj_pts": data_dict["tot_obj_pts"],
            "tot_rel_pose": data_dict["tot_rel_pose"],
            "tot_bow_vec_object_edge_feats":
                data_dict["tot_bow_vec_object_edge_feats"],
            "edges": data_dict["edges"],
            "obj_ids": data_dict["obj_ids"],
        }
        input_sha = sha256_bytes(
            b"".join(
                np.ascontiguousarray(v).tobytes()
                for v in input_tensors.values()))
        pair_record_sha = sha256_bytes(pair_id.encode())
        cache["cache_key"] = {
            "pair_id": pair_id,
            "input_tensor_sha256": input_sha,
            "checkpoint_sha256": ckpt_sha,
            "sampling_mode": "official_mt19937",
            "model_config_sha256": model_config["sha256"],
            "code_head": code_head,
        }
        cache["pair_record_sha256"] = pair_record_sha
        np.savez_compressed(
            out_dir / "input_tensors.npz", **input_tensors)

        # ---- ONE SGAligner forward, all modalities -----------------
        src_count = data_dict["src_count"]
        batch = {
            "tot_obj_pts": torch.from_numpy(
                data_dict["tot_obj_pts"]).to(device),
            "tot_bow_vec_object_edge_feats": torch.from_numpy(
                data_dict["tot_bow_vec_object_edge_feats"]).to(device),
            "tot_rel_pose": torch.from_numpy(
                data_dict["tot_rel_pose"]).to(device),
            "edges": torch.from_numpy(
                data_dict["edges"].astype(np.int64)).to(device),
            "graph_per_obj_count": [np.asarray(
                data_dict["graph_per_obj_count"], dtype=np.int64)],
            "graph_per_edge_count": [np.asarray(
                data_dict["graph_per_edge_count"], dtype=np.int64)],
            "batch_size": 1,
            "tot_bow_vec_object_attr_feats": torch.zeros(
                (data_dict["tot_obj_pts"].shape[0], 164)).to(device),
        }
        with torch.no_grad():
            embs_model = model(batch)
        embs = {
            m: embs_model[m].cpu().numpy().astype(np.float32)
            for m in ("pct", "gat", "rel")
        }
        joint_model = embs_model["joint"].cpu().numpy().astype(
            np.float32)
        np.savez_compressed(
            out_dir / "embeddings.npz", fusion_weight4=fusion4,
            joint_model=joint_model, **embs)

        # joint consistency: offline fusion == model joint (pct+gat+rel)
        joint_offline = fusion_offline(
            embs, fusion4, "pct+gat+rel", device)
        cache["joint_online_offline_consistent"] = bool(
            np.array_equal(joint_offline, joint_model))

        # ---- per-combo matching + UNION node pairs -----------------
        combo_embeddings = {
            combo: fusion_offline(embs, fusion4, combo, device)
            for combo in COMBOS
        }
        anchors = set(load_anchor_ids(pair_id))
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        combo_matching = {}
        union_pairs = []
        seen_union = set()
        for combo in COMBOS:
            emb = combo_embeddings[combo]
            nm = node_metrics(emb, src_count, anchors, src_map, ref_map)
            combo_matching[combo] = nm
            for a, b in nm["node_corrs"]:
                key = (int(a), int(b))
                if key not in seen_union:
                    seen_union.add(key)
                    union_pairs.append(key)

        # ---- ONE GeoTransformer pass per UNION node pair -----------
        objects = data_dict["registration_pts"]
        id2oid = data_dict["registration_id2oid"]
        geot_cache = {}
        for (src_idx, ref_idx) in union_pairs:
            src_pts = objects.get(int(src_idx))
            ref_pts = objects.get(int(ref_idx))
            head = {
                "src_object_id": id2oid.get(int(src_idx)),
                "ref_object_id": id2oid.get(int(ref_idx)),
            }
            if src_pts is None or ref_pts is None \
                    or len(src_pts) < 50 or len(ref_pts) < 50:
                geot_cache[(src_idx, ref_idx)] = {
                    "status": "insufficient_raw_points", **head}
                continue
            status, output = geotransformer_forward(
                src_pts, ref_pts, device=device)
            if status != "ok":
                geot_cache[(src_idx, ref_idx)] = {
                    "status": status, **head,
                    **({"detail": output} if output else {}),
                }
                continue
            src_corr = output["src_corr_points"]
            ref_corr = output["ref_corr_points"]
            if len(src_corr) == 0:
                geot_cache[(src_idx, ref_idx)] = {
                    "status": "empty_point_correspondence", **head}
                continue
            geot_cache[(src_idx, ref_idx)] = {
                "status": "ok", **head,
                "src_corr": src_corr.astype(np.float32),
                "ref_corr": ref_corr.astype(np.float32),
                "scores": output["corr_scores"].astype(np.float32),
                "geot_correspondences": int(len(src_corr)),
                "src_input_points": int(len(src_pts)),
                "ref_input_points": int(len(ref_pts)),
            }
        # ---- per combo: RANSAC + ICP + decision, ONCE each ---------
        gt = load_gt_transform(pair_id)  # evaluation only
        gt = np.asarray(gt, dtype=np.float64).reshape(4, 4)
        combo_results = {}
        for combo in COMBOS:
            node_corrs = [tuple(x) for x in
                          combo_matching[combo]["node_corrs"]]
            try:
                registration, used, failures = combo_registration(
                    geot_cache, node_corrs)
            except RuntimeError as exc:
                # per-combo typed RANSAC failure (official semantics:
                # ransac_failure stage), NOT a pair-level pipeline error
                combo_results[combo] = {
                    "status": "failed",
                    "failed_stage": "ransac_failure",
                    "error": repr(exc)[:200],
                    "node_metrics": combo_matching[combo],
                    "strict": False, "relaxed": False,
                }
                continue
            entry = {
                "node_metrics": combo_matching[combo],
            }
            if registration is None:
                entry.update({
                    "status": "failed",
                    "failed_stage": "registration",
                    "node_pair_failures": failures,
                    "failure_stage_counts": {
                        stage: sum(1 for f in failures
                                   if f["stage"] == stage)
                        for stage in sorted({
                            f["stage"] for f in failures})
                    } if failures else {},
                    "strict": False, "relaxed": False,
                })
            else:
                transform = registration["transform"]
                cos_r = (np.trace(
                    transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
                rre = float(np.degrees(
                    np.arccos(np.clip(cos_r, -1, 1))))
                rte = float(np.linalg.norm(
                    transform[:3, 3] - gt[:3, 3]))
                strict = rre <= STRICT[0] and rte <= STRICT[1]
                relaxed = rre <= RELAXED[0] and rte <= RELAXED[1]
                features, decision, icp = combo_decision(
                    data_dict, registration, pair_id)
                entry.update({
                    "status": "ok",
                    "rre": rre, "rte": rte,
                    "strict": strict, "relaxed": relaxed,
                    "ransac_inliers": registration["inliers"],
                    "ransac_corrs": registration["corrs"],
                    "node_pairs_used": [
                        [int(a), int(b)]
                        for a, b in registration["node_pairs_used"]],
                    "node_pair_failures": failures,
                    "decision": decision,
                    "decision_features": features,
                    "icp_converged": icp.converged,
                    "icp_fitness": icp.fitness,
                    "icp_rmse_m": icp.rmse_m,
                    "accepted": decision["usable_for_reconstruction"],
                    "raw_transform": transform.tolist(),
                    "icp_transform": icp.transform.tolist(),
                })
            combo_results[combo] = entry
        # strip corr arrays from the geot entries into the npz AFTER
        # every combo consumed them; JSON stays small
        geot_arrays = {}
        for i, (key, entry) in enumerate(geot_cache.items()):
            for field in ("src_corr", "ref_corr", "scores"):
                if field in entry:
                    geot_arrays[f"{field}_{i}"] = entry.pop(field)
            entry["cache_row"] = i
        np.savez_compressed(
            out_dir / "geot_corrs.npz", **geot_arrays)
        cache["geot_node_pairs"] = {
            f"{s}_{r}": v for (s, r), v in geot_cache.items()}
        cache["combos"] = combo_results
        cache["status"] = "ok"
        cache["requested_structured_completed"] = {
            "requested": 1, "structured": 1, "completed": sum(
                1 for c in combo_results.values()
                if c.get("status") == "ok"),
        }
    except Exception as exc:  # noqa: BLE001 - typed failure
        cache.update({
            "status": "structured_failure",
            "failure_type": type(exc).__name__,
            "error": repr(exc)[:300],
            "traceback": traceback.format_exc()[-2000:],
        })
    cache["elapsed_s"] = time.monotonic() - started
    cache["gpu_peak_bytes"] = (
        torch.cuda.max_memory_allocated()
        if device == "cuda" else None)
    (out_dir / "pair_cache.json").write_text(
        json.dumps(cache, indent=2, default=_json_default) + "\n")
    return {
        "status": cache["status"],
        "failure_type": cache.get("failure_type"),
        "elapsed_s": cache["elapsed_s"],
        "joint_consistent": cache.get("joint_online_offline_consistent"),
    }


def split_pairs(split: str):
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    if split == "fixed12":
        smoke = Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/phase6_registration_aware_closure/"
            "smoke12/native"
        )
        return sorted(
            d.name for d in smoke.iterdir()
            if d.is_dir() and "_to_" in d.name)
    return [
        line.strip() for line in
        (pairlists / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=(
        "selection", "calibration", "fixed12"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="DEBUG ONLY; formal runs must not use it")
    parser.add_argument("--pairs-file", type=Path, default=None,
                        help="rerun ONLY the pairs listed in this file "
                        "(one pair id per line); used to repair typed "
                        "failures without touching healthy caches")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    model, fusion4, ckpt_epoch = load_model_and_fusion(device)
    model_config = {
        "modules": ["pct", "gat", "rel"], "rel_dim": 41,
        "attr_dim": 164, "hidden_units": [3, 128, 128],
        "heads": [2, 2], "emb_dim": 100, "pt_out_dim": 256,
        "attribute_available": False,
        "fusion": "official 4-row weight, per-combo row slicing",
    }
    model_config["sha256"] = sha256_bytes(
        json.dumps(model_config, sort_keys=True).encode())
    decision_config = effective_decision_config()

    pairs = split_pairs(args.split)
    if args.pairs_file:
        wanted = [l.strip() for l in
                  args.pairs_file.read_text().splitlines() if l.strip()]
        pairs = [p for p in pairs if p in wanted]
        assert pairs, "pairs-file has no intersection with this split"
    if args.limit:
        pairs = pairs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs_run.txt").write_text("\n".join(pairs) + "\n")
    (args.out / "pairs_run.sha256").write_text(
        sha256_file(args.out / "pairs_run.txt") + "\n")
    (args.out / "effective_config.json").write_text(
        json.dumps({
            "model_config": model_config,
            "decision_config": decision_config,
            "combos": list(COMBOS),
            "sampling": {
                "mode": "official_mt19937", "scan_seed": 0,
                "predicted_iteration_order": "sorted object id",
            },
        }, indent=2) + "\n")

    counters = {"ok": 0, "structured_failure": 0}
    joint_ok = 0
    for index, pair in enumerate(pairs):
        tag = f"{pair[:8]}_{pair[-4:]}"
        result = run_pair_cached(
            pair, args.out / tag, model, fusion4, ckpt_epoch,
            device, code_head, model_config, decision_config)
        counters[result["status"]] = counters.get(result["status"], 0) + 1
        if result.get("joint_consistent"):
            joint_ok += 1
        print(f"[{index+1}/{len(pairs)}] {tag} {result['status']}"
              f" {result['elapsed_s']:.1f}s"
              f" joint={'Y' if result.get('joint_consistent') else 'N'}",
              flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    (args.out / "run_summary.json").write_text(
        json.dumps({
            "split": args.split, "mode": "official_sgf_predicted",
            "sampling_mode": "official_mt19937",
            "requested": len(pairs), **counters,
            "joint_online_offline_consistent": joint_ok,
        }, indent=2) + "\n")
    print(json.dumps({
        "split": args.split, "requested": len(pairs), **counters,
        "joint_consistent": joint_ok,
    }))


if __name__ == "__main__":
    main()
