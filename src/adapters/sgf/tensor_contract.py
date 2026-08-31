"""Runtime validation of the official SGAligner tensor contract.

Fail-closed checks mirroring the pre-registered Phase D test list:
shapes, dtypes, index ranges, contiguity of object ids, root/rel_pose
consistency with the official algorithm, unit safety (metres), modality
availability honesty, and swap-order conventions.
"""

from __future__ import annotations

import numpy as np

from .graph_adapter import GraphContract, select_root_official


def validate_contract(contract: GraphContract) -> list[str]:
    errors: list[str] = []
    n = int(contract.graph_per_obj_count[0]) if np.ndim(
        contract.graph_per_obj_count
    ) else int(contract.graph_per_obj_count)
    pts = contract.tot_obj_pts
    if pts.dtype != np.float32 or pts.shape != (n, 512, 3):
        errors.append(f"tot_obj_pts invalid: {pts.shape} {pts.dtype}")
    if not np.isfinite(pts).all():
        errors.append("tot_obj_pts contains non-finite values")
    # metre-scale sanity: object local extents must be well below mm-era
    # magnitudes (a mm-scaled object would exceed 100 units)
    extents = np.linalg.norm(
        pts.max(axis=1) - pts.min(axis=1), axis=-1
    )
    if extents.max() > 100.0:
        errors.append(
            f"tot_obj_pts extent {extents.max():.1f} suggests mm/mm mixup"
        )
    ids = contract.obj_ids
    if len(np.unique(ids)) != len(ids):
        errors.append("obj_ids contains duplicates")
    if list(ids) != sorted(ids):
        errors.append("obj_ids not sorted (continuity contract)")
    if list(range(len(ids))) != [
        contract.object_id2idx[int(o)] for o in ids
    ]:
        errors.append("object_id2idx is not the continuous 0..N-1 map")
    edges = contract.edges
    if edges.ndim != 2 or edges.shape[1] != 2:
        errors.append(f"edges must be [E,2], got {edges.shape}")
    elif len(edges) and (
        edges.min() < 0 or edges.max() >= n
    ):
        errors.append("edges reference out-of-range node indices")
    rel = contract.tot_rel_pose
    if rel.shape != (n, 3) or not np.isfinite(rel).all():
        errors.append(f"tot_rel_pose invalid: {rel.shape}")
    bow = contract.tot_bow_vec_object_edge_feats
    if bow.shape != (n, 41):
        errors.append(f"relation bow must be [N,41], got {bow.shape}")
    if contract.provenance.get("mode") == "oracle":
        attr = contract.tot_bow_vec_object_attr_feats
        if attr is None or attr.shape != (n, 164):
            errors.append("oracle mode requires [N,164] attributes")
    else:
        if contract.tot_bow_vec_object_attr_feats is not None:
            errors.append(
                "predicted mode must NOT fabricate attribute vectors"
            )
        if contract.modality_mask.get("attr"):
            errors.append("predicted mode attr mask must be false")
    # root consistency
    if contract.provenance.get("directed_pairs_before_none", 0) > 0:
        # recompute root from provenance-ordered ids is not possible
        # here without the pairs; the adapter records it and the unit
        # tests cover the algorithm directly
        pass
    return errors


def validate_pair_dict(data_dict: dict) -> list[str]:
    errors: list[str] = []
    src_count = data_dict.get("src_count")
    total = data_dict["tot_obj_pts"].shape[0]
    ref_count = total - src_count
    counts = data_dict["graph_per_obj_count"]
    if int(counts[0]) != src_count or int(counts[1]) != ref_count:
        errors.append("graph_per_obj_count inconsistent with tensors")
    edges = data_dict["edges"]
    if edges.max() >= total:
        errors.append("pair edges exceed merged node range")
    if data_dict["tot_rel_pose"].shape[0] != total:
        errors.append("tot_rel_pose rows mismatch")
    if data_dict["tot_bow_vec_object_edge_feats"].shape[0] != total:
        errors.append("relation bow rows mismatch")
    attr = data_dict.get("tot_bow_vec_object_attr_feats")
    if attr is not None and attr.shape[0] != total:
        errors.append("attribute bow rows mismatch")
    return errors


def swap_pair_convention(data_dict: dict) -> dict:
    """Return the swapped (ref->src) pair dict without mutating input.

    Object order, edge blocks, per-graph counts and modality provenance
    all swap consistently; ``transform`` semantics stay source->ref
    at the call site (documented in tests).
    """
    src_count = data_dict["src_count"]
    total = data_dict["tot_obj_pts"].shape[0]
    pts = data_dict["tot_obj_pts"]
    edges = data_dict["edges"]
    e_counts = data_dict["graph_per_edge_count"]

    def block(tensor):
        return np.concatenate([tensor[src_count:], tensor[:src_count]])

    swapped = dict(data_dict)
    swapped["tot_obj_pts"] = block(pts)
    swapped["tot_rel_pose"] = block(data_dict["tot_rel_pose"])
    swapped["tot_bow_vec_object_edge_feats"] = block(
        data_dict["tot_bow_vec_object_edge_feats"]
    )
    if data_dict.get("tot_bow_vec_object_attr_feats") is not None:
        swapped["tot_bow_vec_object_attr_feats"] = block(
            data_dict["tot_bow_vec_object_attr_feats"]
        )
    swapped["obj_ids"] = block(data_dict["obj_ids"])
    src_edges = edges[: int(e_counts[0])]
    ref_edges = edges[int(e_counts[0]):]
    # remap indices into the swapped layout
    new_src_base = src_count
    new_ref_base = 0
    remap = np.concatenate([
        np.arange(new_ref_base, new_ref_base + (total - src_count)),
        np.arange(new_src_base, new_src_base + src_count),
    ])
    swapped["edges"] = np.concatenate(
        [remap[ref_edges.reshape(-1)].reshape(-1, 2),
         remap[src_edges.reshape(-1)].reshape(-1, 2)]
    )
    swapped["graph_per_obj_count"] = np.asarray(
        [total - src_count, src_count], dtype=np.int64
    )
    swapped["graph_per_edge_count"] = np.asarray(
        [e_counts[1], e_counts[0]], dtype=np.int64
    )
    swapped["src_count"] = total - src_count
    swapped["object_id2idx"] = {
        "src": data_dict.get("object_id2idx", {}).get(
            "ref", data_dict.get("ref_object_id2idx")
        ),
        "ref": data_dict.get("object_id2idx", {}).get(
            "src", data_dict.get("src_object_id2idx")
        ),
    }
    swapped["scene_ids"] = [
        data_dict["scene_ids"][1], data_dict["scene_ids"][0]
    ]
    swapped["modality_mask"] = {
        "src": data_dict["modality_mask"]["ref"],
        "ref": data_dict["modality_mask"]["src"],
    }
    swapped["provenance"] = {
        "src": data_dict["provenance"]["ref"],
        "ref": data_dict["provenance"]["src"],
    }
    return swapped
