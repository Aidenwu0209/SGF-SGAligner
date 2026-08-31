"""SGF graph -> official SGAligner graph data contract.

Produces exactly the fields the official ``Scan3RDataset.__getitem__``
builds (see src/datasets/scan3r.py at upstream 51cd572):

    tot_obj_pts [N,512,3] (metres, later centred by pcl_center)
    edges [E,2]           directed pairs in index space incl. 'none'
    tot_rel_pose [N,3]    root barycentre - object barycentre
    tot_bow_vec_object_edge_feats [N,41]
    tot_bow_vec_object_attr_feats [N,164] (oracle mode only)
    graph_per_obj_count / graph_per_edge_count
    obj_ids / object_id2idx / scene_ids / pcl_center
    modality_mask + provenance

Root selection follows the official algorithm exactly: the object with
the highest TOTAL degree over directed relationship pairs (before
'none' supplementation), ``np.argmax(np.bincount(flattened pairs))``
(ties resolved by np.argmax taking the smallest id).

Two modes:

- ``oracle``: 3DSSG GT relationships + attributes (164-D BoW).
- ``sgf_predicted``: SGF GraphPredictor relations only; attribute
  module disabled — the 164-D attribute field is None with
  attribute_available=false (NEVER a zero vector passed off as real).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .object_adapter import ObjectAdapterResult
from .relation_mapper import RelationMapper


@dataclass
class GraphContract:
    tot_obj_pts: np.ndarray
    registration_pts: dict           # idx -> [K,3] full unique world pts
    registration_id2oid: dict        # idx -> original object id
    edges: np.ndarray
    tot_rel_pose: np.ndarray
    tot_bow_vec_object_edge_feats: np.ndarray
    tot_bow_vec_object_attr_feats: np.ndarray | None
    graph_per_obj_count: np.ndarray
    graph_per_edge_count: np.ndarray
    obj_ids: np.ndarray
    object_id2idx: dict
    scene_ids: list
    pcl_center: np.ndarray
    modality_mask: dict
    provenance: dict = field(default_factory=dict)
    root_obj_id: int | None = None


def select_root_official(
    directed_pairs: list[tuple[int, int]],
    object_ids: list[int],
) -> int:
    """Official root: highest total degree, np.argmax tie-break."""
    if not directed_pairs:
        return int(object_ids[0])
    flattened = np.asarray(directed_pairs, dtype=np.int64).reshape(-1)
    counts = np.bincount(flattened)
    root_id = int(np.argmax(counts))
    return root_id


def build_none_supplemented_edges(
    directed_pairs: list[tuple[int, int]],
    object_id2idx: dict,
    relation_mapper: RelationMapper,
    triples: list[tuple[int, int, str]],
) -> np.ndarray:
    """Directed edges incl. official all-pairs 'none' supplementation."""
    pairs = [tuple(pair) for pair in directed_pairs]
    ids = list(object_id2idx.keys())
    existing = set(pairs)
    for i in ids:
        for j in ids:
            if i == j or (i, j) in existing:
                continue
            pairs.append((i, j))
            triples.append((i, j, "none"))
    src = np.asarray([object_id2idx[i] for i, _ in pairs], dtype=np.int64)
    dst = np.asarray([object_id2idx[j] for _, j in pairs], dtype=np.int64)
    return np.stack([src, dst], axis=1)


def build_relation_bow(
    triples: list[tuple[int, int, str]],
    object_id2idx: dict,
    relation_mapper: RelationMapper,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-object outgoing relation-name BoW (official 41-D)."""
    n = len(object_id2idx)
    outgoing: list[list[str]] = [[] for _ in range(n)]
    for sub, _obj, name in triples:
        index = object_id2idx.get(int(sub))
        if index is not None:
            outgoing[index].append(name)
    bow = np.zeros((n, 41), dtype=np.float32)
    mask = np.zeros(41, dtype=np.float32)
    all_names = set()
    for index, names in enumerate(outgoing):
        vector, slot_mask = relation_mapper.bow_vector(names)
        bow[index] = vector
        mask = np.maximum(mask, slot_mask)
        all_names.update(names)
    return bow, mask, sorted(all_names)


def build_attribute_bow_oracle(
    attributes_per_object: dict[int, list[str]], object_id2idx: dict,
    attribute_vocab: dict[str, int],
) -> np.ndarray:
    """Per-object attribute-name BoW over the official 164-D vocab."""
    n = len(object_id2idx)
    bow = np.zeros((n, len(attribute_vocab)), dtype=np.float32)
    for oid, index in object_id2idx.items():
        for attr in attributes_per_object.get(int(oid), []):
            position = attribute_vocab.get(attr)
            if position is not None:
                bow[index, position] += 1.0
    if bow.shape[1] != 164:
        raise ValueError(
            f"attribute vocabulary must be 164-D, got {bow.shape[1]}"
        )
    return bow


def adapt_graph(
    objects: ObjectAdapterResult,
    *,
    mode: str,
    directed_pairs: list[tuple[int, int]] | None = None,
    relation_triples: list[tuple[int, int, str]] | None = None,
    attributes_per_object: dict[int, list[str]] | None = None,
    attribute_vocab: dict[str, int] | None = None,
    relation_mapper: RelationMapper | None = None,
    pcl_center: np.ndarray | None = None,
) -> GraphContract:
    """Assemble the official contract for one scan in ``mode``."""
    if mode == "official_oracle":
        mode = "oracle"
    elif mode == "official_sgf_predicted":
        mode = "sgf_predicted"
    if mode not in {"oracle", "sgf_predicted"}:
        raise ValueError(f"unknown mode {mode!r}")
    relation_mapper = relation_mapper or RelationMapper()
    directed_pairs = directed_pairs or []
    triples = [tuple(t) for t in (relation_triples or [])]

    root_id = select_root_official(
        directed_pairs, objects.obj_ids.tolist()
    )
    if root_id not in objects.object_id2idx:
        root_id = int(objects.obj_ids[0])
    root_index = objects.object_id2idx[root_id]

    rel_pose = objects.barycenters[root_index] - objects.barycenters

    edges = build_none_supplemented_edges(
        directed_pairs, objects.object_id2idx, relation_mapper, triples
    )
    bow, rel_mask, used_names = build_relation_bow(
        triples, objects.object_id2idx, relation_mapper
    )

    attr_bow = None
    if mode == "oracle":
        if attribute_vocab is None or attributes_per_object is None:
            raise ValueError("oracle mode requires attribute vocabulary")
        attr_bow = build_attribute_bow_oracle(
            attributes_per_object, objects.object_id2idx, attribute_vocab
        )
        modules = ["pct", "gat", "rel", "attr"]
        mask = {"pct": True, "gat": True, "rel": True, "attr": True,
                "relation_slots": rel_mask}
    else:
        modules = ["pct", "gat", "rel"]
        mask = {"pct": True, "gat": True, "rel": True, "attr": False,
                "relation_slots": rel_mask}

    if pcl_center is None:
        pcl_center = np.zeros(3, dtype=np.float64)

    return GraphContract(
        tot_obj_pts=objects.obj_pts,
        registration_pts=objects.registration_pts,
        registration_id2oid=objects.idx_to_object_id,
        edges=edges,
        tot_rel_pose=rel_pose.astype(np.float32),
        tot_bow_vec_object_edge_feats=bow,
        tot_bow_vec_object_attr_feats=attr_bow,
        graph_per_obj_count=np.asarray(
            [len(objects.obj_ids)], dtype=np.int64
        ),
        graph_per_edge_count=np.asarray([len(edges)], dtype=np.int64),
        obj_ids=objects.obj_ids,
        object_id2idx=objects.object_id2idx,
        scene_ids=[],
        pcl_center=pcl_center,
        modality_mask=mask,
        provenance={
            "mode": mode,
            "modules": modules,
            "root_obj_id": root_id,
            "directed_pairs_before_none": len(directed_pairs),
            "relation_names_used": used_names,
            "attribute_available": mode == "oracle",
            "objects": [
                {
                    "original_id": p.original_id,
                    "stable_surfel_count": p.stable_surfel_count,
                    "used_replacement": p.used_replacement,
                    "full_point_count": p.full_point_count,
                    "unique_point_count": p.unique_point_count,
                }
                for p in objects.provenance
            ],
        },
    )


def merge_pair_contracts(
    src: GraphContract, ref: GraphContract, pcl_center: np.ndarray
) -> dict:
    """Build the official pair data_dict (getitem semantics)."""
    if src.tot_bow_vec_object_attr_feats is not None:
        attr = np.concatenate(
            [src.tot_bow_vec_object_attr_feats,
             ref.tot_bow_vec_object_attr_feats]
        )
    else:
        attr = None
    src_count = int(src.graph_per_obj_count[0])
    ref_reg = {
        index + src_count: pts
        for index, pts in ref.registration_pts.items()
    }
    reg = dict(src.registration_pts)
    reg.update(ref_reg)
    return {
        "registration_pts": reg,
        "registration_id2oid": {
            **{i + src_count: oid for i, oid in
               ref.registration_id2oid.items()},
            **src.registration_id2oid,
        },
        "obj_ids": np.concatenate([src.obj_ids, ref.obj_ids]),
        "tot_obj_pts": np.concatenate(
            [
                src.tot_obj_pts - pcl_center,
                ref.tot_obj_pts - pcl_center,
            ]
        ).astype(np.float32),
        "graph_per_obj_count": np.asarray(
            [src_count, int(ref.graph_per_obj_count[0])], dtype=np.int64
        ),
        "graph_per_edge_count": np.asarray(
            [len(src.edges), len(ref.edges)], dtype=np.int64
        ),
        "edges": np.concatenate([src.edges, ref.edges]),
        "tot_rel_pose": np.concatenate(
            [src.tot_rel_pose, ref.tot_rel_pose]
        ).astype(np.float32),
        "tot_bow_vec_object_edge_feats": np.concatenate(
            [src.tot_bow_vec_object_edge_feats,
             ref.tot_bow_vec_object_edge_feats]
        ),
        "tot_bow_vec_object_attr_feats": attr,
        "scene_ids": [src.scene_ids, ref.scene_ids],
        "pcl_center": pcl_center,
        "src_object_id2idx": src.object_id2idx,
        "ref_object_id2idx": ref.object_id2idx,
        "src_count": src_count,
        "modality_mask": {
            "src": src.modality_mask, "ref": ref.modality_mask,
        },
        "provenance": {"src": src.provenance, "ref": ref.provenance},
    }
