"""Canonical predicted-input builder (V4-Fix-Seal Part 3).

ONE builder for training, evaluation and cache running — the
PRODUCTION contract of ``inference.build_pair_inputs`` is the single
source of truth:

  - pcl_center = source scan's FULL stable InSeg surfel cloud mean
    (NEVER the descriptor-sample mean that v4_train used before —
    that deviation caused the 6.55 cm tot_obj_pts divergence);
  - official_mt19937 sampling, scan seed 0, canonical sorted-id
    object order — identical to the production caches;
  - attachments: training labels, complete-none edges (already in the
    contract), SGF explicit edges per graph in LOCAL indices, and
    graph_per_edge_count_explicit.

Every consumer (v4_train semantics, evaluators, cache runners) must
use THIS module; re-deriving inputs with a private copy is forbidden.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get(
    "SGALIGNER_CODE_ROOT", Path(__file__).resolve().parents[1])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src/inference/sgf_official") not in sys.path:
    sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from inference import build_pair_inputs  # noqa: E402
from v4_train import build_labels, explicit_edges_for  # noqa: E402
from adapters.sgf.data_sources import PredictedGraphSource  # noqa: E402

BUILDER_PATH = Path(__file__).resolve()
MATCHER_PATH = ROOT / "src/inference/sgf_official/inference.py"
MATCHER_SHA = hashlib.sha256(MATCHER_PATH.read_bytes()).hexdigest()
BUILDER_SHA = hashlib.sha256(BUILDER_PATH.read_bytes()).hexdigest()


def build_canonical_pair(pair_id: str, with_labels: bool = True,
                         predicted: PredictedGraphSource | None = None):
    """Production input contract + training/evaluation attachments.

    Returns (data_dict, labels).  ``data_dict`` is EXACTLY what
    build_pair_inputs returns (production centering/sampling/order),
    plus edges_explicit / graph_per_edge_count_explicit.  Skips are
    NEVER silent: exceptions propagate (fail-closed) and label-free
    pairs return labels=[] with the pair still present.
    """
    predicted = predicted or PredictedGraphSource()
    data_dict, _contracts = build_pair_inputs(
        pair_id, "official_sgf_predicted",
        sampling_mode="official_mt19937")
    src_scan, ref_scan = pair_id.split("_to_")
    src_pairs = predicted.load(src_scan).directed_pairs
    ref_pairs = predicted.load(ref_scan).directed_pairs
    explicit = [
        explicit_edges_for(src_pairs, data_dict["src_object_id2idx"]),
        explicit_edges_for(ref_pairs, data_dict["ref_object_id2idx"]),
    ]
    data_dict["edges_explicit"] = (
        np.concatenate([explicit[0], explicit[1]])
        if (len(explicit[0]) or len(explicit[1]))
        else np.zeros((0, 2), dtype=np.int64))
    data_dict["graph_per_edge_count_explicit"] = np.asarray(
        [len(explicit[0]), len(explicit[1])], dtype=np.int64)
    labels = []
    if with_labels:
        labels = build_labels(
            pair_id,
            predicted.load(src_scan).segments,
            predicted.load(ref_scan).segments)
    return data_dict, labels


def arm_edges(data_dict: dict, arm: str):
    """The edges an arm ACTUALLY consumes at forward time."""
    if arm == "complete":
        return (np.asarray(data_dict["edges"]),
                np.asarray(data_dict["graph_per_edge_count"]))
    if arm == "explicit":
        return (np.asarray(data_dict["edges_explicit"]),
                np.asarray(data_dict["graph_per_edge_count_explicit"]))
    raise ValueError(f"unknown arm {arm}")


def arm_fingerprint(data_dict: dict, pair_id: str, arm: str,
                    checkpoint_sha: str) -> dict:
    """Complete arm-specific input fingerprint (Part 5).

    Covers: pair/scans, object ids AND order, every consumed tensor,
    the ARM-SPECIFIC edges + counts (so B and C can never share a
    fingerprint), object counts, pcl_center + definition, sampling,
    checkpoint, matcher and builder source hashes.
    """
    src_scan, ref_scan = pair_id.split("_to_")
    edges, edge_counts = arm_edges(data_dict, arm)
    payload = {
        "pair_id": pair_id,
        "src_scan": src_scan, "ref_scan": ref_scan,
        "obj_ids": np.asarray(data_dict["obj_ids"]).tolist(),
        "tot_obj_pts": np.ascontiguousarray(
            data_dict["tot_obj_pts"]).tobytes(),
        "tot_rel_pose": np.ascontiguousarray(
            data_dict["tot_rel_pose"]).tobytes(),
        "relation_bow_41d": np.ascontiguousarray(
            data_dict["tot_bow_vec_object_edge_feats"]).tobytes(),
        "arm_edges": np.ascontiguousarray(edges).tobytes(),
        "arm_edge_counts": np.ascontiguousarray(
            edge_counts).tobytes(),
        "graph_obj_counts": np.ascontiguousarray(
            data_dict["graph_per_obj_count"]).tobytes(),
        "pcl_center": np.ascontiguousarray(
            np.asarray(data_dict["pcl_center"], dtype=np.float64)
        ).tobytes(),
        "sampling_mode": "official_mt19937",
        "scan_seed": 0,
    }
    h = hashlib.sha256()
    for key in sorted(payload):
        h.update(key.encode())
        h.update(payload[key] if isinstance(payload[key], bytes)
                 else json_dumps(payload[key]).encode())
    return {
        "pair_id": pair_id, "arm": arm,
        "input_sha256": h.hexdigest(),
        "pcl_center_definition":
            data_dict["pcl_center_definition"],
        "checkpoint_sha256": checkpoint_sha,
        "matcher_source_sha256": MATCHER_SHA,
        "builder_source_sha256": BUILDER_SHA,
    }


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, sort_keys=True, default=str)
