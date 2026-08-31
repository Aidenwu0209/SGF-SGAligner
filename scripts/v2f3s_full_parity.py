"""V2T-Fix3-Seal stage 4: FULL official-vs-adapter tensor parity.

Extends the Fix3 parity to complete per-field tensor comparison under a
canonical object-ID reordering. Every field reports
equal / max_abs_diff / first_mismatch / official hash / adapter hash.

Official side: exact official preprocessing (process_scan +
calculate_bow_node_edge_feats) re-run in-process with a FIXED numpy
seed (the official FPS consumes the global RNG; seeding makes the
official draw reproducible run-to-run — the official code itself is
untouched). Adapter side: the production adapter contract path.

The 512-point descriptor coordinate arrays CANNOT be byte-equal by
construction: the official sampler draws from the global MT19937
stream while the adapter uses a fixed-seed PCG64 generator
(determinism is an adapter requirement the official does not satisfy).
Those fields are reported honestly (equal=false) together with
per-object geometry-agreement statistics proving both sample the same
underlying surfaces.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "preprocessing/scan3r"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

DATA = Path("/home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full")
FILES = Path("/tmp/official_scan3r_files")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal"
CACHE = Path("/tmp/official_scan3r_contract_seal")

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

OFFICIAL_SEED = 0  # fixed draw of the official global-RNG sampler


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


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
    vertices["red"], vertices["green"], vertices["blue"] = (
        v["red"], v["green"], v["blue"])
    for k in ("objectId", "globalId", "NYU40", "Eigen13", "RIO27"):
        vertices[k] = v[k]
    np.save(out / "data.npy", vertices)
    return out


def official_contract(scan):
    """Exact official pkl payload with FULL tensors, seeded RNG."""
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
    np.random.seed(OFFICIAL_SEED)  # official FPS draws global RNG
    dd = pp.process_scan(str(tmp), rel_data, obj_data, args, cfg, rel2idx)
    pp.common.write_pkl_data(dd, str(write_dir / f"{scan}.pkl"))
    pp.calculate_bow_node_edge_feats(str(tmp / "files" / "orig"), rel2idx)
    import pickle

    with open(write_dir / f"{scan}.pkl", "rb") as fh:
        dd = pickle.load(fh)

    # explicit pairs (pre-supplement) reconstructed exactly like the
    # official loop: dedup on (sub,obj), relationship.json order
    seen = []
    for triple in rel_data["relationships"]:
        sub, obj = int(triple[0]), int(triple[1])
        if sub in dd["objects_id"] and obj in dd["objects_id"]:
            if [sub, obj] not in [[s, o] for s, o in seen]:
                seen.append((sub, obj))

    import pickle as _pkl

    with open(ROOT / "checkpoints/release/obj_attr.pkl", "rb") as fh:
        attr_vocab = _pkl.load(fh)
    bow_attr = np.zeros((dd["objects_count"], 164), dtype=np.float32)
    for i, obj_attr in enumerate(dd["object_attributes"]):
        for a in obj_attr:
            if a in attr_vocab:
                bow_attr[i, attr_vocab[a]] += 1

    payload = {
        "objects_id": np.asarray(dd["objects_id"], dtype=np.int64),
        "obj_points_512": np.asarray(
            dd["obj_points"][512], dtype=np.float32),
        "rel_trans": np.asarray(dd["rel_trans"], dtype=np.float32),
        "root_obj_id": np.int64(dd["root_obj_id"]),
        "explicit_pairs": np.asarray(seen, dtype=np.int64).reshape(-1, 2),
        "edges": np.asarray(dd["edges"], dtype=np.int64),
        "bow_edge": np.asarray(
            dd["bow_vec_object_edge_feats"], dtype=np.float32),
        "bow_attr": bow_attr,
        "objects_count": np.int64(dd["objects_count"]),
        "edges_count": np.int64(dd["edges_count"]),
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **payload)
    return payload


def adapter_contract(scan):
    from adapters.sgf.data_sources import OracleGraphSource
    from adapters.sgf.object_adapter import adapt_objects
    from adapters.sgf.graph_adapter import adapt_graph
    import pickle

    src = OracleGraphSource().load(scan)
    objects = adapt_objects(src.segments)
    with open(ROOT / "checkpoints/release/obj_attr.pkl", "rb") as fh:
        vocab = pickle.load(fh)
    contract = adapt_graph(
        objects, mode="oracle",
        directed_pairs=src.directed_pairs,
        relation_triples=src.relation_triples,
        attributes_per_object=src.attributes_per_object,
        attribute_vocab={k: int(v) for k, v in vocab.items()},
    )
    return contract, src.directed_pairs


def field_cmp(field, a, b, note=None, float_tol=0.0):
    """equal / max_abs_diff / first_mismatch / hashes."""
    a = np.asarray(a)
    b = np.asarray(b)
    out = {
        "field": field,
        "shape_official": list(a.shape),
        "shape_adapter": list(b.shape),
        "official_hash": hash_of(a),
        "adapter_hash": hash_of(b),
    }
    if note:
        out["note"] = note
    if a.shape != b.shape:
        out.update({
            "equal": False, "max_abs_diff": None,
            "first_mismatch": "shape mismatch",
        })
        return out
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        diff = np.abs(
            a.astype(np.float64) - b.astype(np.float64))
        out["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
        nz = np.flatnonzero(diff.ravel() > float_tol)
        out["first_mismatch"] = (
            str(np.unravel_index(int(nz[0]), a.shape)) if nz.size
            else None)
        out["equal"] = bool(nz.size == 0)
    else:
        eq = bool(np.array_equal(a, b))
        out["equal"] = eq
        out["max_abs_diff"] = 0.0 if eq else None
        if eq:
            out["first_mismatch"] = None
        else:
            bad = np.flatnonzero(
                (a != b).ravel())
            out["first_mismatch"] = str(
                np.unravel_index(int(bad[0]), a.shape))
    return out


def canonical(points_side, ids):
    order = np.argsort(np.asarray(ids))
    return order


def parity_row(scan):
    off = official_contract(scan)
    ada, ada_raw_pairs = adapter_contract(scan)

    off_ids = off["objects_id"].tolist()
    ada_ids = [int(x) for x in ada.obj_ids]
    row = {"scan": scan}

    # --- object identity ---
    row["object_id_set"] = field_cmp(
        "object_id_set", sorted(off_ids), sorted(ada_ids))
    row["object_id_order_raw"] = field_cmp(
        "object_id_order_raw", off_ids, ada_ids,
        note="raw order (official=objects.json; adapter=segment order)")

    if set(off_ids) != set(ada_ids):
        row["fatal"] = "object id sets differ; downstream skipped"
        return row

    # canonical reordering: ascending object id on BOTH sides
    off_order = np.argsort(np.asarray(off_ids))
    ada_order = np.argsort(np.asarray(ada_ids))
    n = len(off_ids)

    def reorder_rows(arr):
        return np.asarray(arr)[off_order]

    def reorder_rows_ada(arr):
        return np.asarray(arr)[ada_order]

    # --- 512-point descriptor coordinates ---
    off_pts = reorder_rows(off["obj_points_512"])
    ada_pts = reorder_rows_ada(ada.tot_obj_pts)
    pts_note = (
        "cannot be byte-equal: official FPS draws the global MT19937 "
        "stream (seeded here for reproducibility), adapter uses a "
        "fixed-seed PCG64 generator; geometry agreement reported via "
        "per-object nearest-neighbour distances"
    )
    row["points_512_exact"] = field_cmp(
        "points_512_exact", off_pts, ada_pts, note=pts_note)
    row["points_512_shape_dtype"] = {
        "official": [str(off_pts.dtype), list(off_pts.shape)],
        "adapter": [str(ada_pts.dtype), list(ada_pts.shape)],
        "units": "metres (raw 3RScan PLY coordinates, uncentered)",
    }
    # row-sorted (sampling-order-insensitive) canonical comparison
    def rows_sorted(pts):
        sorted_rows = []
        for i in range(pts.shape[0]):
            p = pts[i]
            order = np.lexsort((p[:, 2], p[:, 1], p[:, 0]))
            sorted_rows.append(p[order])
        return np.stack(sorted_rows)

    row["points_512_rowsort_hash"] = field_cmp(
        "points_512_rowsort", rows_sorted(off_pts), rows_sorted(ada_pts),
        note="per-object point rows sorted lexicographically before "
             "hashing; still RNG-dependent on both sides")
    from scipy.spatial import cKDTree

    nn_mean, nn_max, centroid_max = [], [], 0.0
    for i in range(n):
        tree = cKDTree(ada_pts[i])
        d, _ = tree.query(off_pts[i], k=1)
        nn_mean.append(float(d.mean()))
        nn_max.append(float(d.max()))
        centroid_max = max(
            centroid_max,
            float(np.linalg.norm(
                off_pts[i].mean(axis=0) - ada_pts[i].mean(axis=0))))
    row["points_512_geometry_agreement"] = {
        "per_object_mean_nn_dist_mean": float(np.mean(nn_mean)),
        "per_object_mean_nn_dist_max": float(np.max(nn_mean)),
        "per_object_max_nn_dist_max": float(np.max(nn_max)),
        "per_object_centroid_dist_max": centroid_max,
        "official_scale_mean_abs": float(np.abs(off_pts).mean()),
        "adapter_scale_mean_abs": float(np.abs(ada_pts).mean()),
    }

    # --- root ---
    row["root_obj_id"] = field_cmp(
        "root_obj_id",
        [int(off["root_obj_id"])],
        [int(ada.provenance["root_obj_id"])],
    )

    # --- rel_trans (canonical) ---
    off_rt = reorder_rows(off["rel_trans"])
    ada_rt = reorder_rows_ada(ada.tot_rel_pose)
    row["rel_trans"] = field_cmp("rel_trans", off_rt, ada_rt)
    row["rel_trans_direction_check"] = field_cmp(
        "rel_trans_sign_flip", off_rt, -ada_rt,
        note="if THIS is equal, directions are flipped",
    )

    # --- explicit edges (object-id space, canonical sorted) ---
    off_exp = np.asarray(off["explicit_pairs"], dtype=np.int64)
    seen = []
    for sub, obj in ada_raw_pairs:
        if [sub, obj] not in seen:
            seen.append((int(sub), int(obj)))
    ada_exp = np.asarray(seen, dtype=np.int64).reshape(-1, 2)
    off_exp_s = off_exp[np.lexsort((off_exp[:, 1], off_exp[:, 0]))] \
        if len(off_exp) else off_exp
    ada_exp_s = ada_exp[np.lexsort((ada_exp[:, 1], ada_exp[:, 0]))] \
        if len(ada_exp) else ada_exp
    row["explicit_edges"] = field_cmp(
        "explicit_edges", off_exp_s, ada_exp_s)
    row["explicit_edges_raw_order"] = field_cmp(
        "explicit_edges_raw_order", off_exp, ada_exp,
        note="official=relationship.json first-occurrence order",
    )

    # --- complete-none edges ---
    off_ids_sorted = sorted(off_ids)
    off_edges = off["edges"]
    off_complete_ids = np.stack([
        [off_ids_sorted[src] for src in off_edges[:, 0]],
        [off_ids_sorted[dst] for dst in off_edges[:, 1]],
    ], axis=1).astype(np.int64)
    ada_edges = np.asarray(ada.edges, dtype=np.int64)
    ada_ids_sorted = sorted(ada_ids)
    ada_complete_ids = np.stack([
        [ada_ids_sorted[src] for src in ada_edges[:, 0]],
        [ada_ids_sorted[dst] for dst in ada_edges[:, 1]],
    ], axis=1).astype(np.int64)

    def canon_edges(edges):
        if not len(edges):
            return edges
        return edges[np.lexsort((edges[:, 1], edges[:, 0]))]

    row["complete_none_edges_raw_order"] = field_cmp(
        "complete_none_edges_raw_order",
        off_complete_ids, ada_complete_ids,
        note="official order = explicit pairs first, then i,j loop",
    )
    row["complete_none_edges_canonical"] = field_cmp(
        "complete_none_edges_canonical",
        canon_edges(off_complete_ids), canon_edges(ada_complete_ids),
    )
    row["edge_order_equal_after_canonical_objids"] = bool(
        np.array_equal(off_complete_ids, ada_complete_ids))

    # --- BOWs (canonical) ---
    row["bow_edge_41d"] = field_cmp(
        "bow_edge_41d",
        reorder_rows(off["bow_edge"]), reorder_rows_ada(
            ada.tot_bow_vec_object_edge_feats),
    )
    row["bow_attr_164d"] = field_cmp(
        "bow_attr_164d",
        reorder_rows(off["bow_attr"]), reorder_rows_ada(
            ada.tot_bow_vec_object_attr_feats),
    )

    # --- counts ---
    row["graph_counts"] = field_cmp(
        "graph_counts",
        [int(off["objects_count"]), len(off_exp),
         int(off["edges_count"])],
        [len(ada_ids), len(ada_exp), len(ada_edges)],
        note="[objects, explicit_edges, complete_edges]",
    )
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rel_scans = {
        e["scan"] for e in json.loads(
            (DATA / "relationships.json").read_text())["scans"]
    }
    pairlists = ROOT / "outputs/official_sgaligner_migration_fix2_pairlists"
    pairs = []
    for split in ("selection", "calibration"):
        pairs += [l.strip() for l in
                  (pairlists / f"{split}.txt").read_text().splitlines()
                  if l.strip()][:6]
    scans = []
    for pair in pairs:
        for scan in pair.split("_to_"):
            if scan in rel_scans and scan not in scans:
                scans.append(scan)
    scans = scans[:12]
    print("scans:", len(scans), flush=True)

    rows = []
    for scan in scans:
        try:
            row = parity_row(scan)
            rows.append(row)
            digest = {
                k: row[k].get("equal") if isinstance(
                    row.get(k), dict) and "equal" in row[k] else None
                for k in ("object_id_set", "root_obj_id", "rel_trans",
                          "explicit_edges", "complete_none_edges_canonical",
                          "bow_edge_41d", "bow_attr_164d", "graph_counts",
                          "points_512_exact")
            }
            print(scan[:8], digest, flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"scan": scan, "error": repr(exc)[:300]})
            print(scan[:8], "ERROR", repr(exc)[:120], flush=True)

    summary_fields = [
        "object_id_set", "object_id_order_raw", "root_obj_id",
        "rel_trans", "explicit_edges", "explicit_edges_raw_order",
        "complete_none_edges_raw_order", "complete_none_edges_canonical",
        "bow_edge_41d", "bow_attr_164d", "graph_counts",
        "points_512_exact", "points_512_rowsort_hash",
    ]
    summary = {}
    for field in summary_fields:
        vals = [r[field] for r in rows if field in r and isinstance(
            r.get(field), dict) and "equal" in r[field]]
        summary[field] = {
            "n": len(vals),
            "equal": sum(1 for v in vals if v["equal"]),
            "max_abs_diff_overall": max(
                (v.get("max_abs_diff") or 0.0) for v in vals
            ) if vals else None,
        }
    payload = {
        "phase": "V2T-Fix3-Seal full tensor parity",
        "official_seed": OFFICIAL_SEED,
        "canonical_rule": "both sides reordered by ascending object id",
        "scans": scans,
        "summary": summary,
        "rows": rows,
    }
    (OUT / "official_adapter_full_tensor_parity.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
