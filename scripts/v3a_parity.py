"""V3-A: official sampling + PCT end-to-end parity (oracle mode, fixed12).

Stream A closure: with the new ``official_mt19937`` sampling mode the
adapter must reproduce the official preprocessing 512-point inputs
BYTE-EXACTLY, and every downstream PCT artefact (embeddings,
normalised embeddings, similarity matrix, top-k candidates, Node
metrics) must match the official-loader reference on the fixed12
pairs.

Official reference: in-process reconstruction of the exact official
preprocessing (process_scan with the global MT19937 seeded to 0 per
scan, official sources untouched), reusing the V2T-Fix3-Seal cache.

Outputs (per the phase deliverables):
  sampling_parity/scan_<tag>.json      per-scan 512-pt + raw parity
  pct_embedding_parity/pair_<tag>.json per-pair PCT embeddings
  similarity_parity/pair_<tag>.json    per-pair similarity matrices
  topk_parity/pair_<tag>.json          per-pair top-1/top-5 + Node F1
  parity_digest.json                   run-to-run determinism digest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "preprocessing/scan3r"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

DATA = Path("/home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full")
FILES = Path("/tmp/official_scan3r_files")
CACHE = Path("/tmp/official_scan3r_contract_seal")
OUT = ROOT / "outputs/official_sgaligner_v3_pct_parity_baseline_20260827"
OFFICIAL_SEED = 0

FILES.mkdir(parents=True, exist_ok=True)
for name in ("relationships.txt", "objects.json", "3RScan.json"):
    src = DATA / name
    if src.exists() and not (FILES / name).exists():
        shutil.copy(src, FILES / name)
if not (FILES / "relationships.txt").exists():
    shutil.copy(
        ROOT / "checkpoints/release/relationships.txt",
        FILES / "relationships.txt",
    )
import os  # noqa: E402

os.makedirs(str(FILES / "files"), exist_ok=True)
for name in ("relationships.txt", "objects.json", "3RScan.json"):
    _src, _dst = FILES / name, FILES / "files" / name
    if _src.exists() and not _dst.exists():
        shutil.copy(_src, _dst)
import utils.define as define  # noqa: E402

define.SCAN3R_ORIG_DIR = str(FILES)
from preprocessing.scan3r import preprocess as pp  # noqa: E402

from inference import (  # noqa: E402
    build_pair_inputs, official_matching,
)
from adapters.sgf.data_sources import (  # noqa: E402
    OracleGraphSource, load_oracle_anchor_ids, oracle_gt_transform,
)

OFFICIAL_CKPT = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()


def official_data_npy(scan, out_dir):
    from plyfile import PlyData

    out = Path(out_dir) / "scans" / scan
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data.npy").exists():
        return out
    ply = PlyData.read(DATA / scan / "labels.instances.annotated.v2.ply")
    v = ply["vertex"]
    vertices = np.empty(
        len(v["x"]),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"),
               ("green", "u1"), ("blue", "u1"), ("objectId", "h"),
               ("globalId", "h"), ("NYU40", "u1"), ("Eigen13", "u1"),
               ("RIO27", "u1")],
    )
    for k in ("x", "y", "z"):
        vertices[k] = v[k]
    for k in ("objectId", "globalId", "NYU40", "Eigen13", "RIO27"):
        vertices[k] = v[k]
    np.save(out / "data.npy", vertices)
    return out


def official_contract(scan):
    """Exact official preprocessing payload (seeded MT19937 draw)."""
    cache = CACHE / f"{scan}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    tmp = Path("/tmp/official_scan3r_work_seal")
    official_data_npy(scan, tmp)
    args = types.SimpleNamespace(
        remove_node=False, remove_edge=False,
        change_node_semantic=False, change_edge_semantic=False,
        mode="orig",
    )
    rel2idx = pp.common.name2idx(FILES / "files" / "relationships.txt")
    rel_json = json.loads(
        (DATA / "relationships.json").read_text())["scans"]
    obj_json = json.loads((DATA / "objects.json").read_text())["scans"]
    rel_data = [r for r in rel_json if r["scan"] == scan][0]
    obj_data = [o for o in obj_json if o["scan"] == scan][0]
    cfg = types.SimpleNamespace(
        preprocess=types.SimpleNamespace(pc_resolutions=[512],
                                         min_obj_points=50))
    write_dir = tmp / "files" / "orig" / "data"
    write_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(OFFICIAL_SEED)
    dd = pp.process_scan(str(tmp), rel_data, obj_data, args, cfg, rel2idx)
    pp.common.write_pkl_data(dd, str(write_dir / f"{scan}.pkl"))
    pp.calculate_bow_node_edge_feats(str(tmp / "files" / "orig"), rel2idx)
    import pickle

    with open(write_dir / f"{scan}.pkl", "rb") as fh:
        dd = pickle.load(fh)
    payload = {
        "objects_id": np.asarray(dd["objects_id"], dtype=np.int64),
        "obj_points_512": np.asarray(
            dd["obj_points"][512], dtype=np.float32),
        "rel_trans": np.asarray(dd["rel_trans"], dtype=np.float32),
        "root_obj_id": np.int64(dd["root_obj_id"]),
        "edges": np.asarray(dd["edges"], dtype=np.int64),
        "bow_edge": np.asarray(
            dd["bow_vec_object_edge_feats"], dtype=np.float32),
        "objects_count": np.int64(dd["objects_count"]),
        "edges_count": np.int64(dd["edges_count"]),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **payload)
    return payload


def ply_raw_points(scan):
    """Official raw per-object point sets (ply row order, float32)."""
    from plyfile import PlyData

    ply = PlyData.read(DATA / scan / "labels.instances.annotated.v2.ply")
    vertex = ply["vertex"]
    points32 = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    )  # already float32 from the ply
    labels = np.asarray(vertex["objectId"])
    out = {}
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        mask = labels == label
        if int(mask.sum()) >= 50:
            out[int(label)] = points32[mask]
    return out


def load_pct_model(device):
    from aligner.sg_aligner import MultiModalEncoder

    model = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    ).to(device)
    state = torch.load(
        OFFICIAL_CKPT, map_location=device, weights_only=False)
    official = dict(state["model"])
    fusion_rows = official.pop("fusion.weight")[:3].clone()
    model.load_state_dict(official, strict=False)
    with torch.no_grad():
        model.fusion.weight.copy_(fusion_rows)
    model.eval()
    return model


def pct_branch(model, pts32, device):
    """(encoder_256, projected_100, normalized_100) for a [N,512,3] f32.

    Mirrors the official forward: tot_obj_pts.permute(0, 2, 1) before
    the encoder ([N,3,512] conv1d convention).
    """
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(pts32)).to(device)
        enc = model.object_encoder(t.permute(0, 2, 1))
        proj = model.object_embedding(enc)
        norm = F.normalize(proj, dim=1)
    return (enc.cpu().numpy(), proj.cpu().numpy(), norm.cpu().numpy())


def scan_parity_row(scan, adapter_contracts, segments_by_scan):
    off = official_contract(scan)
    off_ids = [int(x) for x in off["objects_id"]]
    raw32 = ply_raw_points(scan)
    ada = adapter_contracts[scan]

    # canonical order: ascending object id both sides
    ids_sorted = sorted(off_ids)
    off_order = np.argsort(np.asarray(off_ids))
    off_pts = off["obj_points_512"][off_order]

    # adapter contract rows are already canonical (sorted oid)
    ada_ids = [int(x) for x in ada.obj_ids]
    assert ada_ids == ids_sorted, f"{scan}: adapter id order mismatch"
    ada_pts = ada.tot_obj_pts

    row = {
        "scan": scan,
        "objects": len(ids_sorted),
        "object_id_set_equal": set(off_ids) == set(ada_ids),
    }
    # raw point sets (hash + count per object)
    seg_f32 = {
        int(k): np.asarray(v, dtype=np.float64).astype(np.float32)
        for k, v in segments_by_scan[scan].items()
    }
    raw_rows = []
    for oid in ids_sorted:
        raw_rows.append({
            "object_id": oid,
            "official_raw_count": int(len(raw32[oid])),
            "adapter_raw_count": int(len(seg_f32[oid])),
            "official_raw_hash": hash_of(raw32[oid]),
            "adapter_raw_hash": hash_of(seg_f32[oid]),
        })
    row["raw_points_all_equal"] = all(
        r["official_raw_count"] == r["adapter_raw_count"]
        and r["official_raw_hash"] == r["adapter_raw_hash"]
        for r in raw_rows
    )
    row["raw_points"] = {
        "n_objects": len(raw_rows),
        "first_mismatch": next(
            (r["object_id"] for r in raw_rows
             if r["official_raw_hash"] != r["adapter_raw_hash"]),
            None),
    }
    # 512-point inputs
    diff = np.abs(
        off_pts.astype(np.float64) - ada_pts.astype(np.float64))
    nz = np.flatnonzero(diff.ravel() > 0)
    row["points_512"] = {
        "shape_official": list(off_pts.shape),
        "shape_adapter": list(ada_pts.shape),
        "dtype_official": str(off_pts.dtype),
        "dtype_adapter": str(ada_pts.dtype),
        "units": "metres (raw 3RScan PLY coordinates)",
        "official_hash": hash_of(off_pts),
        "adapter_hash": hash_of(ada_pts),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "first_mismatch": (
            str(np.unravel_index(int(nz[0]), off_pts.shape))
            if nz.size else None),
        "equal": bool(nz.size == 0),
    }
    row["scan_sha_pair"] = f"{hash_of(off_pts)[:12]}:{hash_of(ada_pts)[:12]}"
    return row, raw_rows


def pair_parity_rows(pair, model, device, segments_by_scan,
                     adapter_sampling="official_mt19937"):
    """Pair-level PCT/similarity/top-k parity on the FIXED12 pair."""
    src_scan, ref_scan = pair.split("_to_")
    # ---- official side: reconstructed tensors + official centering --
    off_src = official_contract(src_scan)
    off_ref = official_contract(ref_scan)
    from plyfile import PlyData

    ply = PlyData.read(
        DATA / src_scan / "labels.instances.annotated.v2.ply")
    vertex = ply["vertex"]
    pcl_center = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float64).mean(axis=0)

    off_src_ids = sorted(int(x) for x in off_src["objects_id"])
    off_ref_ids = sorted(int(x) for x in off_ref["objects_id"])
    off_src_pts = off_src["obj_points_512"][
        np.argsort(np.asarray(
            [int(x) for x in off_src["objects_id"]]))]
    off_ref_pts = off_ref["obj_points_512"][
        np.argsort(np.asarray(
            [int(x) for x in off_ref["objects_id"]]))]
    off_pts = np.concatenate([off_src_pts, off_ref_pts]).astype(np.float32)
    off_centered = (off_pts.astype(np.float64) - pcl_center).astype(
        np.float32)

    # ---- adapter side: production pair path, official_mt19937 -------
    data_dict, contracts = build_pair_inputs(
        pair, "official_oracle", sampling_mode=adapter_sampling)
    ada_centered = data_dict["tot_obj_pts"]
    src_count = data_dict["src_count"]
    ada_src_ids = sorted(int(x) for x in contracts[0].obj_ids)
    ada_ref_ids = sorted(int(x) for x in contracts[1].obj_ids)

    emb_row = {
        "pair": pair,
        "id_sets_equal": (
            off_src_ids == ada_src_ids and off_ref_ids == ada_ref_ids),
        "centered_input": {
            "official_hash": hash_of(off_centered),
            "adapter_hash": hash_of(ada_centered),
            "max_abs_diff": float(np.abs(
                off_centered.astype(np.float64)
                - ada_centered.astype(np.float64)).max()),
            "shape_equal": off_centered.shape == ada_centered.shape,
            "equal": bool(np.array_equal(off_centered, ada_centered)),
        },
        "uncentered_input": {
            "official_hash": hash_of(off_pts),
            "adapter_hash": hash_of(np.concatenate([
                contracts[0].tot_obj_pts, contracts[1].tot_obj_pts])),
            "equal": bool(np.array_equal(off_pts, np.concatenate([
                contracts[0].tot_obj_pts,
                contracts[1].tot_obj_pts]))),
        },
    }

    off_enc, off_proj, off_norm = pct_branch(model, off_centered, device)
    ada_enc, ada_proj, ada_norm = pct_branch(model, ada_centered, device)

    def maxdiff(a, b):
        return float(np.abs(
            a.astype(np.float64) - b.astype(np.float64)).max())

    cos = float(np.mean(np.sum(off_norm * ada_norm, axis=1)))
    emb_row["pct_encoder_256"] = {
        "official_hash": hash_of(off_enc.astype(np.float32)),
        "adapter_hash": hash_of(ada_enc.astype(np.float32)),
        "max_abs_diff": maxdiff(off_enc, ada_enc),
    }
    emb_row["pct_projected_100"] = {
        "official_hash": hash_of(off_proj.astype(np.float32)),
        "adapter_hash": hash_of(ada_proj.astype(np.float32)),
        "max_abs_diff": maxdiff(off_proj, ada_proj),
    }
    emb_row["pct_normalized_100"] = {
        "official_hash": hash_of(off_norm.astype(np.float32)),
        "adapter_hash": hash_of(ada_norm.astype(np.float32)),
        "max_abs_diff": maxdiff(off_norm, ada_norm),
        "mean_cosine": cos,
    }

    # ---- similarity + top-k via the SAME official matching code -----
    off_corrs, off_rank, _ = official_matching(off_norm, src_count)
    ada_corrs, ada_rank, _ = official_matching(ada_norm, src_count)
    off_sim = off_norm @ off_norm.T
    ada_sim = ada_norm @ ada_norm.T
    sim_row = {
        "pair": pair,
        "similarity_max_abs_diff": maxdiff(off_sim, ada_sim),
        "similarity_official_hash": hash_of(
            off_sim.astype(np.float32)),
        "similarity_adapter_hash": hash_of(ada_sim.astype(np.float32)),
        "equal": bool(np.array_equal(
            off_sim.astype(np.float32), ada_sim.astype(np.float32))),
    }

    # node metrics vs oracle anchors (GT post-hoc evaluation only)
    anchors = set(load_oracle_anchor_ids(
        src_scan, ref_scan,
        segments_by_scan[src_scan], segments_by_scan[ref_scan]))

    def node_metrics(corrs, src_map, ref_map):
        anchor_idx = {
            (src_map[s], ref_map[r] + src_count)
            for s, r in anchors if s in src_map and r in ref_map
        }
        pred = set(corrs)
        tp = len(pred & anchor_idx)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(anchor_idx) if anchor_idx else 0.0
        return {
            "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-12),
        }

    off_map = {oid: i for i, oid in enumerate(off_src_ids)}
    off_ref_map = {oid: i for i, oid in enumerate(off_ref_ids)}
    ada_src_map = data_dict["src_object_id2idx"]
    ada_ref_map = data_dict["ref_object_id2idx"]
    topk_row = {
        "pair": pair,
        "node_corrs_official": [[int(a), int(b)] for a, b in off_corrs],
        "node_corrs_adapter": [[int(a), int(b)] for a, b in ada_corrs],
        "topk_equal": [list(map(int, a)) == list(map(int, b))
                       for a, b in zip(off_corrs, ada_corrs)]
        if len(off_corrs) == len(ada_corrs) else [],
        "all_topk_equal": (
            len(off_corrs) == len(ada_corrs) and all(
                list(map(int, a)) == list(map(int, b))
                for a, b in zip(off_corrs, ada_corrs))),
        "node_metrics_official": node_metrics(
            off_corrs, off_map, off_ref_map),
        "node_metrics_adapter": node_metrics(
            ada_corrs, ada_src_map, ada_ref_map),
    }
    return emb_row, sim_row, topk_row


def fixed12_pairs():
    smoke = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/phase6_registration_aware_closure/"
        "smoke12/native"
    )
    return sorted(
        d.name for d in smoke.iterdir()
        if d.is_dir() and "_to_" in d.name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="run1")
    parser.add_argument(
        "--adapter-sampling", default="official_mt19937",
        choices=("official_mt19937", "deterministic_pcg64"),
        help="pcg64 = the PREVIOUS production sampling; used once to "
        "quantify the historical sampling gap (Q4)")
    args = parser.parse_args()
    for sub in ("sampling_parity", "pct_embedding_parity",
                "similarity_parity", "topk_parity"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pct_model(device)
    source = OracleGraphSource()

    pairs = fixed12_pairs()
    scans = []
    for pair in pairs:
        for scan in pair.split("_to_"):
            if scan not in scans:
                scans.append(scan)
    print(f"pairs {len(pairs)} scans {len(scans)}", flush=True)

    # build adapter contracts once per scan (official_mt19937)
    from adapters.sgf.object_adapter import adapt_objects
    from adapters.sgf.graph_adapter import adapt_graph
    import pickle

    with open(ROOT / "checkpoints/release/obj_attr.pkl", "rb") as fh:
        vocab = pickle.load(fh)
    adapter_contracts = {}
    segments_by_scan = {}
    for scan in scans:
        result = source.load(scan)
        segments_by_scan[scan] = result.segments
        order = source.official_object_order(scan)
        objects = adapt_objects(
            result.segments,
            sampling_mode=args.adapter_sampling,
            scan_seed=OFFICIAL_SEED, iteration_order=order,
        )
        contract = adapt_graph(
            objects, mode="oracle",
            directed_pairs=result.directed_pairs,
            relation_triples=result.relation_triples,
            attributes_per_object=result.attributes_per_object,
            attribute_vocab={k: int(v) for k, v in vocab.items()},
        )
        adapter_contracts[scan] = contract

    scan_rows = []
    for scan in scans:
        row, raw_rows = scan_parity_row(
            scan, adapter_contracts, segments_by_scan)
        scan_rows.append(row)
        (OUT / "sampling_parity" / f"scan_{scan[:8]}.json").write_text(
            json.dumps({"row": row, "raw_points": raw_rows}, indent=2)
            + "\n")
        print("scan", scan[:8], "pts512 equal:",
              row["points_512"]["equal"], flush=True)

    emb_rows, sim_rows, topk_rows = [], [], []
    for pair in pairs:
        emb, sim, topk = pair_parity_rows(
            pair, model, device, segments_by_scan,
            adapter_sampling=args.adapter_sampling)
        emb_rows.append(emb)
        sim_rows.append(sim)
        topk_rows.append(topk)
        tag = f"{pair[:8]}_{pair[-4:]}"
        (OUT / "pct_embedding_parity" / f"pair_{tag}.json").write_text(
            json.dumps(emb, indent=2) + "\n")
        (OUT / "similarity_parity" / f"pair_{tag}.json").write_text(
            json.dumps(sim, indent=2) + "\n")
        (OUT / "topk_parity" / f"pair_{tag}.json").write_text(
            json.dumps(topk, indent=2) + "\n")
        print("pair", tag,
              "input equal:", emb["centered_input"]["equal"],
              "enc diff:", emb["pct_encoder_256"]["max_abs_diff"],
              "topk equal:", topk["all_topk_equal"], flush=True)

    digest = {
        "run_tag": args.run_tag,
        "adapter_sampling": args.adapter_sampling,
        "scans_512_equal": sum(
            1 for r in scan_rows if r["points_512"]["equal"]),
        "scans_total": len(scan_rows),
        "raw_sets_all_equal": all(
            r["raw_points_all_equal"] for r in scan_rows),
        "pairs_centered_input_equal": sum(
            1 for r in emb_rows if r["centered_input"]["equal"]),
        "pairs_total": len(emb_rows),
        "max_encoder_diff": max(
            r["pct_encoder_256"]["max_abs_diff"] for r in emb_rows),
        "min_normalized_cosine": min(
            r["pct_normalized_100"]["mean_cosine"] for r in emb_rows),
        "max_similarity_diff": max(
            r["similarity_max_abs_diff"] for r in sim_rows),
        "pairs_topk_all_equal": sum(
            1 for r in topk_rows if r["all_topk_equal"]),
        "input_hashes": sorted(
            {r["centered_input"]["adapter_hash"]
             for r in emb_rows} | {
             r["centered_input"]["official_hash"] for r in emb_rows}),
    }
    (OUT / f"parity_digest_{args.run_tag}.json").write_text(
        json.dumps(digest, indent=2) + "\n")
    print(json.dumps(digest, indent=1))


if __name__ == "__main__":
    main()
