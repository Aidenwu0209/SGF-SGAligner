"""Fix3 stage 3+4: exact-official-loader reconstruction + GAT factorial.

Builds the OFFICIAL pkl contract by running the official process_scan
on real 3DSSG scans (with define.py paths redirected to a local files/
symlink farm — official sources untouched), then:
  (3) tensor-by-tensor parity vs the adapters on the same scans;
  (4) GAT factorial: input path x adjacency x rel_trans, frozen ckpt,
      no training, with PCT control.
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "preprocessing/scan3r"))

DATA = Path("/home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full")
FILES = Path("/tmp/official_scan3r_files")
RELATIONSHIPS = (ROOT / "checkpoints/release/relationships.txt").resolve()

# local files/ farm so the official define.py paths resolve without
# touching official sources
FILES.mkdir(parents=True, exist_ok=True)
for name in ("relationships.txt", "objects.json", "3RScan.json"):
    src = DATA / name
    if src.exists() and not (FILES / name).exists():
        shutil.copy(src, FILES / name)
if not (FILES / "relationships.txt").exists():
    shutil.copy(RELATIONSHIPS, FILES / "relationships.txt")
import utils.define as define  # noqa: E402

define.SCAN3R_ORIG_DIR = str(FILES)
import os
os.makedirs(str(FILES / "files"), exist_ok=True)
for name in ("relationships.txt", "objects.json", "3RScan.json"):
    _src = FILES / name
    _dst = FILES / "files" / name
    if _src.exists() and not _dst.exists():
        shutil.copy(_src, _dst)

from preprocessing.scan3r import preprocess as pp  # noqa: E402
from aligner.sg_aligner import MultiModalEncoder  # noqa: E402

OFFICIAL = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"


def official_data_npy(scan, out_dir):
    from plyfile import PlyData

    out = Path(out_dir) / "scans" / scan
    out.mkdir(parents=True, exist_ok=True)
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


import hashlib


def hashlib_of(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def official_process_scan(scan):
    import pickle

    cache = Path("/tmp/official_scan3r_contract") / f"{scan}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    tmp = Path("/tmp/official_scan3r_work")
    official_data_npy(scan, tmp)
    args = types.SimpleNamespace(
        remove_node=False, remove_edge=False,
        change_node_semantic=False, change_edge_semantic=False,
        mode="orig",
    )
    rel2idx = pp.common.name2idx(FILES / "files" / "relationships.txt")
    rel_json = json.loads((DATA / "relationships.json").read_text())["scans"]
    obj_json = json.loads((DATA / "objects.json").read_text())["scans"]
    rel_data = [r for r in rel_json if r["scan"] == scan][0]
    obj_data = [o for o in obj_json if o["scan"] == scan][0]
    cfg = types.SimpleNamespace(
        preprocess=types.SimpleNamespace(pc_resolutions=[512],
                                         min_obj_points=50))
    write_dir = tmp / "files" / "orig" / "data"
    write_dir.mkdir(parents=True, exist_ok=True)
    dd = pp.process_scan(str(tmp), rel_data, obj_data, args, cfg, rel2idx)
    pp.common.write_pkl_data(dd, str(write_dir / f"{scan}.pkl"))
    # official post-hoc BOW calculators (exact code path)
    pp.calculate_bow_node_edge_feats(str(tmp / "files" / "orig"),
                                      rel2idx)
    with open(write_dir / f"{scan}.pkl", "rb") as fh:
        dd = pickle.load(fh)
    payload = {
        "objects_id": np.asarray(dd["objects_id"]).tolist(),
        "edges": np.asarray(dd["edges"]).tolist(),
        "rel_trans": np.asarray(dd["rel_trans"]).tolist(),
        "root_obj_id": int(dd["root_obj_id"]),
        "bow_edge": np.asarray(
            dd["bow_vec_object_edge_feats"]).tolist(),
        "objects_count": int(dd["objects_count"]),
        "edges_count": int(dd["edges_count"]),
        "obj_points_512_hash": hashlib_of(
            np.asarray(dd["obj_points"][512]).astype(np.float32)),
        "obj_points_512_shape": list(
            np.asarray(dd["obj_points"][512]).shape),
        "obj_points_512_mean_abs": float(np.abs(
            np.asarray(dd["obj_points"][512])).mean()),
        "explicit_pairs": len(dd["pairs"]) - sum(
            1 for t in dd["triples"] if t[2] == rel2idx["none"]),
    }
    try:
        import pickle as _pkl

        with open(ROOT / "checkpoints/release/obj_attr.pkl", "rb") as fh:
            attr_vocab = _pkl.load(fh)
        attributes = dd["object_attributes"]
        bow_attr = np.zeros((dd["objects_count"], 164), dtype=np.float32)
        for i, obj_attr in enumerate(attributes):
            for a in obj_attr:
                if a in attr_vocab:
                    bow_attr[i, attr_vocab[a]] += 1
        payload["bow_attr"] = bow_attr.tolist()
    except Exception as exc:  # noqa: BLE001
        payload["bow_attr_error"] = repr(exc)[:200]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def adapter_contract(scan):
    sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
    from adapters.sgf.data_sources import OracleGraphSource
    from adapters.sgf.object_adapter import adapt_objects
    from adapters.sgf.graph_adapter import adapt_graph

    src = OracleGraphSource().load(scan)
    objects = adapt_objects(src.segments)
    contract = adapt_graph(
        objects, mode="oracle",
        directed_pairs=src.directed_pairs,
        relation_triples=src.relation_triples,
        attributes_per_object=src.attributes_per_object,
        attribute_vocab=_attr_vocab(),
    )
    return contract


def _attr_vocab():
    import pickle

    with open(ROOT / "checkpoints/release/obj_attr.pkl", "rb") as fh:
        vocab = pickle.load(fh)
    return {k: int(v) for k, v in vocab.items()}


def parity_row(scan):
    off = official_process_scan(scan)
    ada = adapter_contract(scan)
    off_ids = [int(x) for x in off["objects_id"]]
    ada_ids = [int(x) for x in ada.obj_ids]
    id_set_equal = set(off_ids) == set(ada_ids)
    off_edges = np.asarray(off["edges"])
    row = {
        "scan": scan,
        "official_objects": off["objects_count"],
        "adapter_objects": len(ada_ids),
        "object_id_set_equal": id_set_equal,
        "object_id_order_equal": off_ids == ada_ids,
        "official_root": off["root_obj_id"],
        "adapter_root": ada.provenance["root_obj_id"],
        "root_equal": (
            off["root_obj_id"] == ada.provenance["root_obj_id"]
        ),
        "official_edges": off["edges_count"],
        "adapter_edges": int(len(ada.edges)),
        "edges_count_equal": off["edges_count"] == len(ada.edges),
        "official_pts_hash": off["obj_points_512_hash"],
        "official_pts_scale_mean_abs": off["obj_points_512_mean_abs"],
        # rel_trans comparison only meaningful when the root matches
        "rel_trans_max_abs_diff": None,
        "bow_edge_shape": (off["objects_count"], 41),
        "adapter_bow_shape": list(
            ada.tot_bow_vec_object_edge_feats.shape),
    }
    if id_set_equal and row["root_equal"]:
        idx = {oid: i for i, oid in enumerate(ada_ids)}
        off_rt = np.asarray(off["rel_trans"], dtype=np.float64)
        ada_rt = np.asarray(ada.tot_rel_pose, dtype=np.float64)
        mapped = np.stack([
            ada_rt[idx[int(oid)]] for oid in off_ids])
        row["rel_trans_max_abs_diff"] = float(
            np.abs(off_rt - mapped).max())
        row["rel_trans_direction_check"] = float(
            np.abs(off_rt + mapped).max())  # if sign flipped ~0
    return row


def main() -> None:
    out_dir = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick 12 fixed oracle pairs across selection+calibration whose
    # scans appear in 3DSSG relationships.json
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

    rows = []
    for scan in scans:
        try:
            rows.append(parity_row(scan))
            print("parity", scan[:8], rows[-1][
                "object_id_set_equal"], rows[-1]["root_equal"],
                rows[-1]["rel_trans_max_abs_diff"], flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append({"scan": scan, "error": repr(exc)[:200]})
            print("parity", scan[:8], "ERROR", repr(exc)[:80], flush=True)
    (out_dir / "official_adapter_tensor_parity.json").write_text(
        json.dumps({"scans": scans, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps({
        "scans": len(scans),
        "id_set_equal": sum(1 for r in rows if r.get("object_id_set_equal")),
        "root_equal": sum(1 for r in rows if r.get("root_equal")),
        "edges_equal": sum(1 for r in rows if r.get("edges_count_equal")),
    }))


if __name__ == "__main__":
    main()
