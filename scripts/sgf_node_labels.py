"""V6 label builder: auditable SGF cross-scan node labels.

GT transforms are used OFFLINE ONLY to align object surfaces for
statistics.  Per (src, ref) object pair the builder computes
bidirectional NN coverage at 5cm/10cm, voxel IoU (5cm voxels),
centroid residual, extent ratio, semantic confidence and adjacency
Jaccard, then applies the pre-registered thresholds:

  positive   : bidir10 >= 0.30 OR (unidir >= 0.45 AND IoU >= 0.10
               AND same semantic)
  negative   : bidir10 < 0.10 AND IoU < 0.02
  ambiguous  : the band in between (masked out of the loss)
  hard-neg   : negative subset with same-semantic OR extent ratio in
               [0.7, 1.4] OR centroid residual < 0.5 m (recorded)

Labels are SETS (split/merge native): one src may have many ref
positives and vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

POS_BIDIR = 0.30
POS_UNIDIR = 0.45
POS_IOU = 0.10
NEG_BIDIR = 0.10
NEG_IOU = 0.02
AMBIG_UNIDIR_LOW = 0.30


@dataclass
class PairStats:
    src: int
    ref: int
    cov_5_src2ref: float
    cov_5_ref2src: float
    cov_10_src2ref: float
    cov_10_ref2src: float
    bidir_10: float
    unidir_10: float
    voxel_iou: float
    centroid_residual: float
    extent_ratio: float
    semantic_conf: float
    adjacency_jaccard: float
    label: str = ""          # positive / negative / ambiguous
    hard_negative: bool = False


def voxel_iou(a: np.ndarray, b: np.ndarray, voxel: float = 0.05):
    key_a = np.unique(
        np.floor(np.asarray(a) / voxel).astype(np.int64), axis=0)
    key_b = np.unique(
        np.floor(np.asarray(b) / voxel).astype(np.int64), axis=0)
    sa = {tuple(k) for k in key_a}
    sb = {tuple(k) for k in key_b}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def pair_statistics(src_pts, ref_pts, src_centroid, ref_centroid,
                    src_extent, ref_extent, semantic_conf,
                    adj_jaccard) -> PairStats:
    tree_r = cKDTree(ref_pts)
    tree_s = cKDTree(src_pts)
    d_sr = tree_r.query(src_pts, k=1)[0]
    d_rs = tree_s.query(ref_pts, k=1)[0]
    cov10_sr = float(np.mean(d_sr <= 0.10))
    cov10_rs = float(np.mean(d_rs <= 0.10))
    cov5_sr = float(np.mean(d_sr <= 0.05))
    cov5_rs = float(np.mean(d_rs <= 0.05))
    return PairStats(
        src=0, ref=0,
        cov_5_src2ref=cov5_sr, cov_5_ref2src=cov5_rs,
        cov_10_src2ref=cov10_sr, cov_10_ref2src=cov10_rs,
        bidir_10=min(cov10_sr, cov10_rs),
        unidir_10=max(cov10_sr, cov10_rs),
        voxel_iou=voxel_iou(src_pts, ref_pts),
        centroid_residual=float(np.linalg.norm(
            np.asarray(src_centroid) - np.asarray(ref_centroid))),
        extent_ratio=(
            float(min(src_extent, ref_extent)
                  / max(src_extent, ref_extent))
            if max(src_extent, ref_extent) > 0 else 0.0),
        semantic_conf=float(semantic_conf),
        adjacency_jaccard=float(adj_jaccard))


def classify(st: PairStats) -> PairStats:
    if st.bidir_10 >= POS_BIDIR or (
            st.unidir_10 >= POS_UNIDIR
            and st.voxel_iou >= POS_IOU
            and st.semantic_conf >= 0.999):
        st.label = "positive"
        # (surface evidence alone is sufficient for positives; the
        # semantic conjunct only guards the weak unidir path)
    elif st.bidir_10 < NEG_BIDIR and st.voxel_iou < NEG_IOU:
        st.label = "negative"
        st.hard_negative = bool(
            st.semantic_conf >= 0.999
            or 0.7 <= st.extent_ratio <= 1.4
            or st.centroid_residual < 0.5)
        # with the neutral default (0.5) the semantic path never
        # fires; hard negatives then come from extent/centroid
        # geometry only — non-degenerate by construction
    else:
        st.label = "ambiguous"
    return st


def label_pair(src_segments, ref_segments, gt_transform,
               src_adj=None, ref_adj=None, semantic_fn=None):
    """Full label computation for one training pair.

    src_segments/ref_segments: {oid: [K,3] world points}
    gt_transform: 4x4 mapping SRC world -> REF world (offline only)
    semantic_fn: (src_oid, ref_oid) -> [0,1] confidence; default is
    NEUTRAL 0.5 (semantic unknown for SGF-predicted segments — a
    constant 1.0 would mark every negative 'same-semantic' and make
    the hard-negative class degenerate, as the first audit showed).
    """
    semantic_fn = semantic_fn or (lambda a, b: 0.5)
    moved = {
        oid: pts @ gt_transform[:3, :3].T + gt_transform[:3, 3]
        for oid, pts in src_segments.items()}
    stats = []
    for s_oid, s_pts in moved.items():
        for r_oid, r_pts in ref_segments.items():
            adj = 1.0
            if src_adj is not None and ref_adj is not None:
                adj = adjacency_jaccard(
                    src_adj, s_oid, ref_adj, r_oid)
            st = pair_statistics(
                s_pts, r_pts,
                s_pts.mean(axis=0), r_pts.mean(axis=0),
                extent_of(s_pts), extent_of(r_pts),
                semantic_fn(s_oid, r_oid), adj)
            st.src = int(s_oid)
            st.ref = int(r_oid)
            stats.append(classify(st))
    return stats


def extent_of(pts: np.ndarray) -> float:
    d = np.asarray(pts).max(axis=0) - np.asarray(pts).min(axis=0)
    return float(np.linalg.norm(d))


def adjacency_jaccard(src_adj, s_oid, ref_adj, r_oid):
    a = set(src_adj.get(s_oid, ()))
    b = set(ref_adj.get(r_oid, ()))
    return len(a & b) / len(a | b) if (a or b) else 0.0


def audit(stats):
    pos = [s for s in stats if s.label == "positive"]
    neg = [s for s in stats if s.label == "negative"]
    amb = [s for s in stats if s.label == "ambiguous"]
    src_count = {}
    ref_count = {}
    for s in pos:
        src_count[s.src] = src_count.get(s.src, 0) + 1
        ref_count[s.ref] = ref_count.get(s.ref, 0) + 1
    one2one = sum(
        1 for s in pos
        if src_count[s.src] == 1 and ref_count[s.ref] == 1)
    split = sum(1 for c in src_count.values() if c > 1)
    merge = sum(1 for c in ref_count.values() if c > 1)
    return {
        "pairs_total": len(stats),
        "positive": len(pos), "negative": len(neg),
        "ambiguous": len(amb),
        "hard_negative": sum(1 for s in neg if s.hard_negative),
        "one_to_one_positive_objects": one2one,
        "split_sources": split, "merged_refs": merge,
        "mean_bidir10_positive": float(np.mean(
            [s.bidir_10 for s in pos])) if pos else 0.0,
    }
