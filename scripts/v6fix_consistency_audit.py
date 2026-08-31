"""V6-Fix F/C0/C1 paired registration audit.

The expensive SGAligner and GeoTransformer outputs are built once per
pair/checkpoint and shared by exact-flat (F), the historical buggy V6
wrapper (C0), and the corrected shadow wrapper (C1).  GT is loaded only
after every path has selected a transform from GT-free evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for path in (CODE_ROOT / "scripts", CODE_ROOT,
             CODE_ROOT / "src", CODE_ROOT / "src/inference/sgf_official"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from canonical_inputs import build_canonical_pair  # noqa: E402
from v4_train import batch_for  # noqa: E402
from inference import (  # noqa: E402
    official_matching, geotransformer_forward, STRICT, RELAXED,
    REG_K, NUM_P2P,
)
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402
from v3b_cache_runner import combo_registration, combo_decision  # noqa: E402
from spatial_consistency import (  # noqa: E402
    cluster_candidates, cluster_candidates_corrected,
    hypothesis_rank, rank_hypotheses_corrected,
)

ASSET_ROOT = Path(os.environ.get(
    "SGALIGNER_ASSET_ROOT",
    "/home/aidenwu/Documents/sgaligner-sgf-official")).resolve()
DEFAULT_OUT = CODE_ROOT / (
    "outputs/official_sgaligner_v6_fix_consistency_audit_20260829/formal_v2")
CACHE_SCHEMA = "v6fix-inference-cache-v2"
RESULT_SCHEMA = "v6fix-consistency-audit-v2"
CHECKPOINTS = {
    "A": ASSET_ROOT / ("outputs/official_sgaligner_v5_relation_gat_20260828/"
                       "training/B/epoch_00010.pt"),
    "B": ASSET_ROOT / ("outputs/official_sgaligner_v6_sgf_domain_matcher_"
                       "20260829/training/B/epoch_00020.pt"),
    "D": ASSET_ROOT / ("outputs/official_sgaligner_v6_sgf_domain_matcher_"
                       "20260829/training/D/epoch_00055.pt"),
}
EXPECTED_SHA = {
    "A": "c82637337b9a3d79693383a54c8a92013dd7f8f10f4d964c7cc196e74f3411c6",
    "B": "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200",
    "D": "ea9584a0d102ec4ad3a8d73b7f25a419fa7818cf8b2ef4e784151d7466d2e26d",
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value):
    payload = json.dumps(value, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def source_sha256(path):
    return sha256_file(CODE_ROOT / path)


def git_state():
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
        check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=CODE_ROOT).returncode != 0
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=CODE_ROOT
    ).returncode != 0
    if dirty or staged:
        raise RuntimeError("formal audit requires a clean tracked worktree")
    return {"head": head, "tracked_dirty": False}


def configure_rng(repeat):
    seed = 424200 + int(repeat)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed": seed if torch.cuda.is_available() else None,
        "torch_deterministic_warn_only": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "pygcransac_seed_api_available": False,
        "pygcransac_note": (
            "no seed parameter; repeat distribution remains mandatory"),
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def split_pairs(name):
    if name in ("selection", "calibration"):
        path = ASSET_ROOT / (
            "outputs/official_sgaligner_migration_fix2_pairlists/"
            f"{name}.txt")
        return [line.strip() for line in path.read_text().splitlines()
                if line.strip()]
    if name == "fixed12":
        root = Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/phase6_registration_aware_closure/"
            "smoke12/native")
        return sorted(p.name for p in root.iterdir()
                      if p.is_dir() and "_to_" in p.name)
    raise ValueError(name)


def split_manifest(name):
    pairs = split_pairs(name)
    expected = {"selection": 89, "calibration": 90, "fixed12": 12}[name]
    if len(pairs) != expected or len(set(pairs)) != expected:
        raise RuntimeError(
            f"{name} manifest expected {expected} unique pairs, got "
            f"{len(pairs)}/{len(set(pairs))}")
    payload = "\n".join(pairs) + "\n"
    return pairs, {
        "name": name, "expected": expected, "actual": len(pairs),
        "unique": len(set(pairs)),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def object_geometry(dd):
    objects = dd["registration_pts"]
    src_count = int(dd["src_count"])
    total = int(dd["tot_obj_pts"].shape[0])
    centres_src, centres_ref = {}, {}
    extents_src, extents_ref = {}, {}
    for idx in range(total):
        pts = np.asarray(objects[idx])
        centre = pts.mean(axis=0)
        extent = float(np.linalg.norm(np.ptp(pts, axis=0)))
        if idx < src_count:
            centres_src[idx] = centre
            extents_src[idx] = extent
        else:
            centres_ref[idx - src_count] = centre
            extents_ref[idx - src_count] = extent
    counts = np.asarray(dd["graph_per_edge_count_explicit"], dtype=int)
    edges = np.asarray(dd["edges_explicit"], dtype=int)
    src_edges = edges[:counts[0]]
    ref_edges = edges[counts[0]:counts[0] + counts[1]]
    def adjacency(n, graph_edges):
        result = {i: set() for i in range(n)}
        for a, b in graph_edges:
            result[int(a)].add(int(b))
            result[int(b)].add(int(a))
        return result
    return {
        "centres_src": centres_src,
        "centres_ref": centres_ref,
        "extents_src": extents_src,
        "extents_ref": extents_ref,
        "adjacency_src": adjacency(src_count, src_edges),
        "adjacency_ref": adjacency(total - src_count, ref_edges),
        "semantic_available": False,
    }


def full_input_provenance(dd, pair_id, ckpt_id, geometry):
    h = hashlib.sha256()

    def add(label, value):
        h.update(label.encode())
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            h.update(str(array.dtype).encode())
            h.update(json.dumps(array.shape).encode())
            if array.dtype.kind in ("O", "U", "S"):
                h.update(json.dumps(jsonable(array), sort_keys=True,
                                    separators=(",", ":")).encode())
            else:
                h.update(array.tobytes())
        else:
            h.update(json.dumps(jsonable(value), sort_keys=True,
                                separators=(",", ":")).encode())

    add("pair_id", pair_id)
    add("checkpoint_id", ckpt_id)
    add("checkpoint_sha256", EXPECTED_SHA[ckpt_id])
    for key in (
            "obj_ids", "tot_obj_pts", "tot_rel_pose",
            "tot_bow_vec_object_edge_feats", "edges_explicit",
            "graph_per_edge_count_explicit", "graph_per_obj_count",
            "pcl_center"):
        add(key, np.asarray(dd[key]))
    add("src_count", int(dd["src_count"]))
    surfaces = []
    for key in sorted(dd["registration_pts"], key=int):
        points = np.asarray(dd["registration_pts"][key], dtype=np.float32)
        digest = hashlib.sha256(
            np.ascontiguousarray(points).tobytes()).hexdigest()
        surfaces.append({"index": int(key), "points": len(points),
                         "sha256": digest})
        add(f"registration_pts:{int(key)}", points)
    source_hashes = {
        name: source_sha256(name) for name in (
            "scripts/canonical_inputs.py", "scripts/v4_train.py",
            "scripts/v3b_cache_runner.py", "scripts/spatial_consistency.py",
            "scripts/v6fix_consistency_audit.py",
            "src/inference/sgf_official/inference.py")
    }
    add("source_hashes", source_hashes)
    add("matcher_contract", {
        "function": "official_matching",
        "model_modules": ["pct", "gat", "rel"],
        "rel_dim": 41, "attr_dim": 164,
        "registration_top_k": REG_K,
        "point_correspondence_cap": NUM_P2P,
        "sampling": "official_mt19937", "scan_seed": 0,
    })
    return {
        "cache_schema": CACHE_SCHEMA,
        "cache_key": h.hexdigest(),
        "pair_id": pair_id,
        "checkpoint_id": ckpt_id,
        "checkpoint_sha256": EXPECTED_SHA[ckpt_id],
        "object_ids_order": jsonable(np.asarray(dd["obj_ids"])),
        "src_count": int(dd["src_count"]),
        "unit": "metres",
        "registration_surfaces": surfaces,
        "semantic_state": "unknown_unavailable",
        "adjacency_state": "explicit_edges_available",
        "centres": jsonable({
            "src": geometry["centres_src"],
            "ref": geometry["centres_ref"]}),
        "extents": jsonable({
            "src": geometry["extents_src"],
            "ref": geometry["extents_ref"]}),
        "source_hashes": source_hashes,
        "matcher_contract": {
            "function": "official_matching",
            "model_modules": ["pct", "gat", "rel"],
            "rel_dim": 41, "attr_dim": 164,
            "registration_top_k": REG_K,
            "point_correspondence_cap": NUM_P2P,
            "sampling": "official_mt19937", "scan_seed": 0,
        },
    }


def build_or_load_cache(pair_id, ckpt_id, model, device, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pair_id}.pt"
    dd, _ = build_canonical_pair(pair_id, with_labels=False)
    geometry = object_geometry(dd)
    provenance = full_input_provenance(
        dd, pair_id, ckpt_id, geometry)
    input_hash = provenance["cache_key"]
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached.get("cache_schema") != CACHE_SCHEMA
                or cached["checkpoint_sha256"] != EXPECTED_SHA[ckpt_id]
                or cached["input_sha256"] != input_hash
                or cached.get("provenance") != provenance):
            raise RuntimeError(f"stale cache {path}")
        return dd, cached
    with torch.no_grad():
        embedding = model(batch_for(dd, "explicit", device))["joint"]
        embedding = embedding.cpu().numpy().astype(np.float32)
    src_count = int(dd["src_count"])
    node_corrs, rank_list, similarity = official_matching(
        embedding, src_count)
    objects = dd["registration_pts"]
    geot = {}
    for src_idx, ref_idx in sorted(set(
            (int(a), int(b)) for a, b in node_corrs)):
        sp, rp = objects.get(src_idx), objects.get(ref_idx)
        if sp is None or rp is None or len(sp) < 50 or len(rp) < 50:
            geot[(src_idx, ref_idx)] = {"status": "insufficient"}
            continue
        status, output = geotransformer_forward(sp, rp, device=device)
        if status != "ok" or len(output["src_corr_points"]) == 0:
            geot[(src_idx, ref_idx)] = {"status": status}
            continue
        geot[(src_idx, ref_idx)] = {
            "status": "ok",
            "src_corr": output["src_corr_points"].astype(np.float32),
            "ref_corr": output["ref_corr_points"].astype(np.float32),
            "scores": output["corr_scores"].astype(np.float32),
        }
        entry = geot[(src_idx, ref_idx)]
        entry["sha256"] = hashlib.sha256(b"".join(
            np.ascontiguousarray(entry[key]).tobytes()
            for key in ("src_corr", "ref_corr", "scores"))).hexdigest()
    cached = {
        "cache_schema": CACHE_SCHEMA,
        "pair_id": pair_id,
        "checkpoint_id": ckpt_id,
        "checkpoint_sha256": EXPECTED_SHA[ckpt_id],
        "input_sha256": input_hash,
        "embedding_sha256": hashlib.sha256(
            np.ascontiguousarray(embedding).tobytes()).hexdigest(),
        "node_corrs": [(int(a), int(b)) for a, b in node_corrs],
        "rank_list": jsonable(rank_list),
        "similarity_sha256": hashlib.sha256(
            np.ascontiguousarray(similarity).tobytes()).hexdigest(),
        "provenance": provenance,
        "geot": geot,
    }
    torch.save(cached, path)
    return dd, cached


def hypothesis_record(dd, geot, members, kind, memo):
    ordered_members = tuple((int(a), int(b)) for a, b in members)
    canonical_members = tuple(sorted(ordered_members))
    signature = stable_hash(canonical_members)
    execution_signature = stable_hash(ordered_members)
    if execution_signature in memo:
        record = dict(memo[execution_signature])
        record["kind"] = kind
        return record
    record = {
        "kind": kind, "members": ordered_members,
        "canonical_members": canonical_members,
        "cluster_size": len(ordered_members),
        "stable_signature": signature,
        "execution_signature": execution_signature,
        "registration_valid": False,
    }
    try:
        registration, used, failures = combo_registration(
            geot, list(ordered_members))
    except RuntimeError as exc:
        record["failure"] = f"{type(exc).__name__}: {exc}"
        memo[execution_signature] = dict(record)
        return record
    record["used"] = used
    record["node_pair_failures"] = failures
    if registration is None:
        record["failure"] = "no_registration"
        memo[execution_signature] = dict(record)
        return record
    try:
        features, decision, _icp = combo_decision(dd, registration, "audit")
    except Exception as exc:  # fail closed, preserved in evidence
        record["failure"] = f"decision:{type(exc).__name__}: {exc}"
        memo[execution_signature] = dict(record)
        return record
    record.update({
        "registration_valid": True,
        "transform": registration["transform"],
        "ransac_support": registration["inlier_ratio"],
        "ransac_inliers": registration["inliers"],
        "corrs": registration["corrs"],
        "icp_fitness": features["icp_fitness"],
        "surface_overlap": features["overlap_10cm"],
        "bidirectional_available": features["bidirectional_available"],
        "bidir_rotation_deg": features["bidirectional_rotation_deg"],
        "bidir_translation_m": features["bidirectional_translation_m"],
        "spatial_extent_m": features["spatial_extent_m"],
        "decision": decision,
        "features": features,
    })
    memo[execution_signature] = dict(record)
    return record


def infer_paths(dd, cached):
    node_corrs = list(cached["node_corrs"])
    src_count = int(dd["src_count"])
    if not node_corrs:
        return {name: None for name in ("F", "C0", "C1")}, {
            "semantic_available": False, "zero_candidate": True,
            "hypotheses": {}}
    geom = object_geometry(dd)
    local = [(a, b - src_count) for a, b in node_corrs]
    c0_local = cluster_candidates(
        local, geom["centres_src"], geom["centres_ref"])
    c1_local = cluster_candidates_corrected(
        local, geom["centres_src"], geom["centres_ref"],
        geom["extents_src"], geom["extents_ref"],
        semantic_src=None, semantic_ref=None,
        adjacency_src=geom["adjacency_src"],
        adjacency_ref=geom["adjacency_ref"])
    c0 = [[(a, b + src_count) for a, b in cluster]
          for cluster in c0_local]
    c1 = [[(a, b + src_count) for a, b in cluster]
          for cluster in c1_local]
    memo = {}
    flat = hypothesis_record(dd, cached["geot"], node_corrs, "flat", memo)
    c0_records = [hypothesis_record(
        dd, cached["geot"], members, "cluster_current", memo)
        for members in c0]
    c1_records = [flat] + [hypothesis_record(
        dd, cached["geot"], members, "cluster_corrected", memo)
        for members in c1]
    valid_c0 = [r for r in c0_records if r.get("registration_valid")]
    for record in valid_c0:
        record["c0_score"] = hypothesis_rank(
            record["members"], record["ransac_support"],
            record["icp_fitness"],
            1.0 if record["bidirectional_available"] else 0.0,
            record["surface_overlap"])
    c0_winner = max(valid_c0, key=lambda r: r["c0_score"]) \
        if valid_c0 else None
    c1_winner, c1_ranked = rank_hypotheses_corrected(c1_records)
    return {"F": flat if flat.get("registration_valid") else None,
            "C0": c0_winner, "C1": c1_winner}, {
        "semantic_available": False, "zero_candidate": False,
        "current_cluster_count": len(c0),
        "corrected_cluster_count": len(c1),
        "hypotheses": {
            "flat": flat, "C0": c0_records, "C1": c1_ranked},
    }


def evaluate_posthoc(pair_id, winners):
    from adapters.sgf.data_sources import load_gt_transform
    gt = np.asarray(load_gt_transform(pair_id), dtype=np.float64).reshape(4, 4)
    rows = {}
    for name, winner in winners.items():
        if winner is None:
            rows[name] = {"valid": False, "strict": False,
                          "relaxed": False, "accepted": False}
            continue
        transform = np.asarray(winner["transform"], dtype=np.float64)
        cos_r = (np.trace(transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
        rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
        rte = float(np.linalg.norm(transform[:3, 3] - gt[:3, 3]))
        accepted = bool(winner["decision"]["usable_for_reconstruction"])
        strict = rre <= STRICT[0] and rte <= STRICT[1]
        relaxed = rre <= RELAXED[0] and rte <= RELAXED[1]
        rows[name] = {
            "valid": True, "rre": rre, "rte": rte,
            "strict": strict, "relaxed": relaxed,
            "accepted": accepted,
            "accepted_correct": accepted and strict,
            "accepted_error": accepted and not strict,
            "transform": transform,
            "selected_kind": winner.get("kind"),
            "stable_signature": winner.get("stable_signature"),
            "decision": winner.get("decision"),
            "features": winner.get("features"),
        }
    return rows


def aggregate(rows, path):
    values = [row["paths"][path] for row in rows]
    zero_candidate = [bool(row["audit"].get("zero_candidate", False))
                      for row in rows]
    return {
        "requested": len(values),
        "completed": sum(v["valid"] for v in values),
        "raw_strict": sum(v.get("strict", False) for v in values),
        "raw_relaxed": sum(v.get("relaxed", False) for v in values),
        "accepted_correct": sum(v.get("accepted_correct", False)
                                for v in values),
        "accepted_error": sum(v.get("accepted_error", False)
                              for v in values),
        "rejected": sum(v["valid"] and not v.get("accepted", False)
                        for v in values),
        "zero_candidate": sum(zero_candidate),
        "failed": sum(not v["valid"] and not zero
                      for v, zero in zip(values, zero_candidate)),
    }


def load_results(out, split, checkpoint, repeats):
    results = []
    for repeat in repeats:
        path = out / split / checkpoint / f"repeat_{repeat:02d}.json"
        if not path.is_file():
            raise RuntimeError(f"missing prerequisite result {path}")
        data = json.loads(path.read_text())
        if (data.get("schema") != RESULT_SCHEMA
                or data.get("split") != split
                or data.get("checkpoint") != checkpoint
                or data.get("repeat") != repeat):
            raise RuntimeError(f"invalid prerequisite result {path}")
        results.append(data)
    return results


def gate1_passed(out):
    results = load_results(out, "selection", "A", (0, 1, 2))
    counts = [result["counts"]["F"] for result in results]
    passed = (
        statistics.median(c["raw_strict"] for c in counts) >= 9
        and statistics.median(c["accepted_correct"] for c in counts) >= 7
        and all(c["accepted_error"] == 0 for c in counts)
        and all(c["failed"] == 0 for c in counts))
    if not passed:
        raise RuntimeError("Gate 1 BASELINE_NOT_REPRODUCED")
    return True


def enforce_gate_order(out, split, checkpoint, repeat, limit):
    if limit is not None:
        raise RuntimeError(
            "formal v2 refuses partial --limit runs; use the diagnostic v1 "
            "directory for smoke tests")
    if split == "selection" and checkpoint == "A":
        if repeat not in (0, 1, 2):
            raise RuntimeError("Gate 1 allows only A selection repeats 0,1,2")
        return
    gate1_passed(out)
    if split == "selection":
        if checkpoint not in ("B", "D") or repeat not in (0, 1, 2):
            raise RuntimeError("Gate 2 allows B/D selection repeats 0,1,2")
        return
    frozen_selection = out / "frozen_selection.json"
    if not frozen_selection.is_file():
        raise RuntimeError(
            "calibration/fixed12 forbidden before frozen_selection.json")
    freeze = json.loads(frozen_selection.read_text())
    if checkpoint != freeze.get("checkpoint"):
        raise RuntimeError("checkpoint differs from frozen selection winner")
    if split == "calibration":
        if repeat != 0:
            raise RuntimeError("Gate 3 calibration is exactly one run")
        return
    frozen_calibration = out / "frozen_calibration.json"
    if not frozen_calibration.is_file():
        raise RuntimeError("fixed12 forbidden before frozen_calibration.json")
    if repeat not in (0, 1, 2):
        raise RuntimeError("Gate 4 fixed12 allows repeats 0,1,2 only")


def compare(rows, candidate):
    stats = {key: 0 for key in (
        "rescue", "destruction", "accepted_rescue",
        "accepted_destruction", "wrong_accept_introduced",
        "wrong_accept_removed", "selected_hypothesis_changed",
        "flat_fallback_selected")}
    per_pair = []
    for row in rows:
        flat, other = row["paths"]["F"], row["paths"][candidate]
        flags = {
            "rescue": other.get("strict", False) and not flat.get("strict", False),
            "destruction": flat.get("strict", False) and not other.get("strict", False),
            "accepted_rescue": other.get("accepted_correct", False)
                               and not flat.get("accepted_correct", False),
            "accepted_destruction": flat.get("accepted_correct", False)
                                    and not other.get("accepted_correct", False),
            "wrong_accept_introduced": other.get("accepted_error", False)
                                       and not flat.get("accepted_error", False),
            "wrong_accept_removed": flat.get("accepted_error", False)
                                    and not other.get("accepted_error", False),
            "selected_hypothesis_changed": flat.get("stable_signature")
                                           != other.get("stable_signature"),
            "flat_fallback_selected": other.get("selected_kind") == "flat",
        }
        for key, value in flags.items():
            stats[key] += int(value)
        per_pair.append({"pair_id": row["pair_id"], **flags})
    return {"aggregate": stats, "pairs": per_pair}


def load_model(ckpt_id, device):
    ckpt = CHECKPOINTS[ckpt_id]
    actual = sha256_file(ckpt)
    if actual != EXPECTED_SHA[ckpt_id]:
        raise RuntimeError(f"checkpoint SHA mismatch {actual}")
    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("selection", "calibration", "fixed12"),
                        default="selection")
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS), default="A")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rng = configure_rng(args.repeat)
    repository = git_state()
    out = args.out.resolve()
    historical = ASSET_ROOT / "outputs"
    if out == historical or historical in out.parents:
        raise RuntimeError("audit OUT must not be inside historical asset outputs")
    enforce_gate_order(
        out, args.split, args.checkpoint, args.repeat, args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    pairs, manifest = split_manifest(args.split)
    if args.limit:
        pairs = pairs[:args.limit]
    rows = []
    cache_dir = out / "cache_v2" / args.checkpoint / args.split
    run_dir = out / args.split / args.checkpoint
    run_dir.mkdir(parents=True, exist_ok=True)
    for index, pair_id in enumerate(pairs, 1):
        dd, cached = build_or_load_cache(
            pair_id, args.checkpoint, model, device, cache_dir)
        winners, audit = infer_paths(dd, cached)
        paths = evaluate_posthoc(pair_id, winners)
        rows.append({"pair_id": pair_id, "paths": paths,
                     "audit": jsonable(audit)})
        print(f"[{index}/{len(pairs)}] {pair_id} " + " ".join(
            f"{name}:{int(paths[name].get('strict', False))}/"
            f"{int(paths[name].get('accepted_correct', False))}"
            for name in ("F", "C0", "C1")), flush=True)
    result = {
        "schema": RESULT_SCHEMA,
        "code_root": str(CODE_ROOT), "asset_root": str(ASSET_ROOT),
        "repository": repository,
        "split": args.split, "checkpoint": args.checkpoint,
        "checkpoint_sha256": EXPECTED_SHA[args.checkpoint],
        "repeat": args.repeat, "device": device,
        "rng": rng, "split_manifest": manifest,
        "counts": {name: aggregate(rows, name)
                   for name in ("F", "C0", "C1")},
        "C0_vs_F": compare(rows, "C0"),
        "C1_vs_F": compare(rows, "C1"),
        "rows": rows,
    }
    dest = run_dir / f"repeat_{args.repeat:02d}.json"
    if dest.exists():
        raise RuntimeError(f"refusing to overwrite existing result {dest}")
    dest.write_text(json.dumps(jsonable(result), indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))
    print(f"WROTE {dest}")


if __name__ == "__main__":
    main()
