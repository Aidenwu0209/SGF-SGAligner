"""V4 candidate cache runner: v3b-style single-inference cache for a
healthy-GAT research candidate checkpoint (arm B/C), plus the incumbent
arm A path that REUSES the V3 official caches unchanged.

Same GeoT / RANSAC / ICP / RegistrationDecision code and frozen rule B
as V3 (imported from v3b_cache_runner) — identical configuration across
arms. The candidate's own joint embedding (pct+gat+rel with its trained
GAT and fusion row) drives matching; GeoT runs once per node pair.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
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
    sha256_file, _json_default, effective_decision_config,
    combo_registration, combo_decision, node_metrics, split_pairs,
    COMBOS,
)
from inference import build_pair_inputs  # noqa: E402
from adapters.sgf.data_sources import (  # noqa: E402
    load_anchor_ids, load_gt_transform,
)
from inference import geotransformer_forward, STRICT, RELAXED  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

OUT = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
SEED = 42
CANDIDATE_COMBO = "candidate"


def explicit_edges_of(pair_id, data_dict):
    """SGF native explicit edges per graph, LOCAL node indices."""
    from v4_train import explicit_edges_for
    from adapters.sgf.data_sources import PredictedGraphSource

    predicted = PredictedGraphSource()
    src_scan, ref_scan = pair_id.split("_to_")
    ex = [
        explicit_edges_for(
            predicted.load(src_scan).directed_pairs,
            data_dict["src_object_id2idx"]),
        explicit_edges_for(
            predicted.load(ref_scan).directed_pairs,
            data_dict["ref_object_id2idx"]),
    ]
    return (
        np.concatenate([ex[0], ex[1]])
        if (len(ex[0]) or len(ex[1]))
        else np.zeros((0, 2), dtype=np.int64),
        np.asarray([len(ex[0]), len(ex[1])], dtype=np.int64))


def batch_for_arm(data_dict, arm, device, pair_id=None):
    if arm == "complete":
        edges = data_dict["edges"]
        counts = data_dict["graph_per_edge_count"]
    else:
        edges, counts = explicit_edges_of(pair_id, data_dict)
    return {
        "tot_obj_pts": torch.from_numpy(
            data_dict["tot_obj_pts"]).to(device),
        "tot_bow_vec_object_edge_feats": torch.from_numpy(
            data_dict["tot_bow_vec_object_edge_feats"]).to(device),
        "tot_rel_pose": torch.from_numpy(
            data_dict["tot_rel_pose"]).to(device),
        "edges": torch.from_numpy(
            np.asarray(edges).astype(np.int64)).to(device),
        "graph_per_obj_count": [np.asarray(
            data_dict["graph_per_obj_count"], dtype=np.int64)],
        "graph_per_edge_count": [np.asarray(
            counts, dtype=np.int64)],
        "batch_size": 1,
        "tot_bow_vec_object_attr_feats": torch.zeros(
            (data_dict["tot_obj_pts"].shape[0], 164)).to(device),
    }


def run_pair_candidate(pair_id, arm, ckpt_path, out_dir, device,
                       code_head, model_config, decision_config):
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_sha = sha256_file(ckpt_path)
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    cache = {
        "pair_id": pair_id, "mode": "official_sgf_predicted",
        "arm": arm, "sampling_mode": "official_mt19937",
        "checkpoint_sha256": ckpt_sha,
        "model_naming": (
            "official-architecture SGF-predicted healthy-GAT research "
            "candidate"),
        "seed": SEED, "device": device, "code_head": code_head,
        "model_config": model_config,
        "decision_config": decision_config,
    }
    try:
        data_dict, _contracts = build_pair_inputs(
            pair_id, "official_sgf_predicted",
            sampling_mode="official_mt19937",
        )
        input_tensors = {
            "tot_obj_pts": data_dict["tot_obj_pts"],
            "tot_rel_pose": data_dict["tot_rel_pose"],
            "tot_bow_vec_object_edge_feats":
                data_dict["tot_bow_vec_object_edge_feats"],
            "edges": data_dict["edges"],
            "obj_ids": data_dict["obj_ids"],
        }
        import hashlib

        input_sha = hashlib.sha256(b"".join(
            np.ascontiguousarray(v).tobytes()
            for v in input_tensors.values())).hexdigest()
        cache["cache_key"] = {
            "pair_id": pair_id,
            "input_tensor_sha256": input_sha,
            "checkpoint_sha256": ckpt_sha,
            "sampling_mode": "official_mt19937",
            "model_config_sha256": model_config["sha256"],
            "code_head": code_head,
            "arm": arm,
        }
        np.savez_compressed(
            out_dir / "input_tensors.npz", **input_tensors)

        with torch.no_grad():
            batch = batch_for_arm(data_dict, arm, device, pair_id)
            embs_model = model(batch)
        embs = {
            m: embs_model[m].cpu().numpy().astype(np.float32)
            for m in ("pct", "gat", "rel")
        }
        joint = embs_model["joint"].cpu().numpy().astype(np.float32)
        np.savez_compressed(
            out_dir / "embeddings.npz", joint=joint, **embs)

        src_count = data_dict["src_count"]
        anchors = set(load_anchor_ids(pair_id))
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        nm = node_metrics(joint, src_count, anchors, src_map, ref_map)
        node_corrs = [tuple(x) for x in nm["node_corrs"]]

        objects = data_dict["registration_pts"]
        id2oid = data_dict["registration_id2oid"]
        geot_cache = {}
        for src_idx, ref_idx in node_corrs:
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
                    "status": status, **head}
                continue
            if len(output["src_corr_points"]) == 0:
                geot_cache[(src_idx, ref_idx)] = {
                    "status": "empty_point_correspondence", **head}
                continue
            geot_cache[(src_idx, ref_idx)] = {
                "status": "ok", **head,
                "src_corr": output["src_corr_points"].astype(np.float32),
                "ref_corr": output["ref_corr_points"].astype(np.float32),
                "scores": output["corr_scores"].astype(np.float32),
                "geot_correspondences": int(
                    len(output["src_corr_points"])),
            }

        entry = {"node_metrics": nm}
        try:
            registration, used, failures = combo_registration(
                geot_cache, node_corrs)
        except RuntimeError as exc:
            entry.update({
                "status": "failed", "failed_stage": "ransac_failure",
                "error": repr(exc)[:200],
                "strict": False, "relaxed": False,
            })
            registration = None
        if registration is not None:
            gt = np.asarray(
                load_gt_transform(pair_id), dtype=np.float64
            ).reshape(4, 4)
            transform = registration["transform"]
            cos_r = (np.trace(
                transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(
                transform[:3, 3] - gt[:3, 3]))
            features, decision, icp = combo_decision(
                data_dict, registration, pair_id)
            entry.update({
                "status": "ok", "rre": rre, "rte": rte,
                "strict": rre <= STRICT[0] and rte <= STRICT[1],
                "relaxed": rre <= RELAXED[0] and rte <= RELAXED[1],
                "ransac_inliers": registration["inliers"],
                "ransac_corrs": registration["corrs"],
                "node_pairs_used": [
                    [int(a), int(b)]
                    for a, b in registration["node_pairs_used"]],
                "node_pair_failures": failures,
                "decision": decision,
                "decision_features": features,
                "icp_converged": icp.converged,
                "accepted": decision["usable_for_reconstruction"],
                "raw_transform": transform.tolist(),
                "icp_transform": icp.transform.tolist(),
            })
        else:
            entry.setdefault("status", "failed")
            entry.setdefault("failed_stage", "registration")
            entry.setdefault("node_pair_failures", [])
            entry.update({"strict": False, "relaxed": False})

        geot_arrays = {}
        for i, (key, e2) in enumerate(geot_cache.items()):
            for field in ("src_corr", "ref_corr", "scores"):
                if field in e2:
                    geot_arrays[f"{field}_{i}"] = e2.pop(field)
            e2["cache_row"] = i
        np.savez_compressed(out_dir / "geot_corrs.npz", **geot_arrays)
        cache["geot_node_pairs"] = {
            f"{s}_{r}": v for (s, r), v in geot_cache.items()}
        cache["combos"] = {CANDIDATE_COMBO: entry}
        cache["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
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
    return {"status": cache["status"],
            "elapsed_s": cache["elapsed_s"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("complete", "explicit"),
                        required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=(
        "selection", "calibration", "fixed12"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import subprocess

    code_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    import hashlib

    model_config = {
        "modules": ["pct", "gat", "rel"], "rel_dim": 41,
        "attr_dim": 164, "arm": args.arm,
        "attribute_available": False,
        "candidate": (
            "official-architecture SGF-predicted healthy-GAT "
            "research candidate"),
    }
    model_config["sha256"] = hashlib.sha256(
        json.dumps(model_config, sort_keys=True).encode()).hexdigest()
    decision_config = effective_decision_config()

    pairs = split_pairs(args.split)
    if args.limit:
        pairs = pairs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pairs_run.txt").write_text("\n".join(pairs) + "\n")
    (args.out / "effective_config.json").write_text(
        json.dumps({"model_config": model_config,
                    "decision_config": decision_config,
                    "sampling": {
                        "mode": "official_mt19937", "scan_seed": 0}},
                   indent=2) + "\n")
    counters = {"ok": 0, "structured_failure": 0}
    for index, pair in enumerate(pairs):
        tag = f"{pair[:8]}_{pair[-4:]}"
        result = run_pair_candidate(
            pair, args.arm, args.checkpoint, args.out / tag,
            device, code_head, model_config, decision_config)
        counters[result["status"]] = counters.get(result["status"], 0) + 1
        print(f"[{index+1}/{len(pairs)}] {tag} {result['status']} "
              f"{result['elapsed_s']:.1f}s", flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()
    ckpt_rel = (
        str(args.checkpoint.resolve().relative_to(ROOT))
        if str(args.checkpoint.resolve()).startswith(str(ROOT))
        else str(args.checkpoint))
    (args.out / "run_summary.json").write_text(
        json.dumps({
            "arm": args.arm, "split": args.split,
            "checkpoint": ckpt_rel,
            "requested": len(pairs), **counters}, indent=2) + "\n")
    print(json.dumps({"requested": len(pairs), **counters}))


if __name__ == "__main__":
    main()
