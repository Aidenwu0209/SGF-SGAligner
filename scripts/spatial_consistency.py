"""V6 spatial consistency layer: deterministic, GT-free candidate
consistency BEFORE GeoTransformer/RANSAC.

Takes per-node top-k cross-graph candidates plus object geometry
(centres, extents, semantics, adjacency), builds a compatibility
graph over candidate PAIRS (a,b) — (c,d) and extracts up to
max_clusters rigid-consistent clusters.  Each cluster then enters
the existing GeoT/RANSAC/ICP pipeline independently; hypothesis
selection uses GT-free features only (equal-weight rank sum —
pre-registered, no tuning).

Compatibility (pre-registered):
  |d_src(a,c) - d_ref(b,d)| <= 0.35 m   (centre-distance preservation)
  extent ratio in [0.5, 2.0]
  semantic compatible (same label if available)
Candidate pair (a,b),(c,d) requires BOTH endpoint compatibilities.
"""
from __future__ import annotations

import numpy as np

CENTRE_TOL = 0.35
EXTENT_LO = 0.5
EXTENT_HI = 2.0
MAX_CLUSTERS = 3


def _compatible(a, b, c, d, centres_src, centres_ref,
                extents_src=None, extents_ref=None,
                semantics=None):
    ds = float(np.linalg.norm(
        centres_src[a] - centres_src[c]))
    dr = float(np.linalg.norm(
        centres_ref[b] - centres_ref[d]))
    if abs(ds - dr) > CENTRE_TOL:
        return False
    if extents_src is not None and extents_ref is not None:
        ratio = (
            extents_src[a] * extents_ref[d])
        denom = extents_src[c] * extents_ref[b]
        if denom <= 0 or ratio <= 0:
            return False
        ratio = (extents_src[a] / max(extents_src[c], 1e-9)) * (
            extents_ref[d] / max(extents_ref[b], 1e-9))
        ratio = np.sqrt(ratio)
        if not (EXTENT_LO <= ratio <= EXTENT_HI):
            return False
    if semantics is not None:
        if semantics.get((a, b), 1.0) < 1.0 and \
                semantics.get((c, d), 1.0) < 1.0:
            # both cross-semantic -> incompatible pairing
            if semantics[(a, b)] != semantics[(c, d)]:
                return False
    return True


def cluster_candidates(candidates, centres_src, centres_ref,
                       extents_src=None, extents_ref=None,
                       semantics=None, max_clusters=MAX_CLUSTERS):
    """Greedy maximal-cluster extraction over the compatibility
    graph (deterministic: candidates processed in given order).

    Returns a list of clusters (lists of (src, ref) candidates),
    largest first, up to max_clusters.  Every input candidate ends
    up in exactly one cluster (singletons allowed) — no candidate is
    dropped (fail-open prohibited; the RANSAC/ICP/decision layers
    remain the safety gate).
    """
    n = len(candidates)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a, b = candidates[i]
        for j in range(i + 1, n):
            c, d = candidates[j]
            ok = _compatible(
                a, b, c, d, centres_src, centres_ref,
                extents_src, extents_ref, semantics)
            adj[i, j] = adj[j, i] = bool(ok)

    assigned = set()
    clusters = []
    order = np.arange(n)
    for start in order:
        if start in assigned:
            continue
        # grow a cluster greedily from `start`
        cluster = [start]
        assigned.add(start)
        changed = True
        while changed:
            changed = False
            for k in order:
                if k in assigned:
                    continue
                if all(adj[k, m] for m in cluster):
                    cluster.append(k)
                    assigned.add(k)
                    changed = True
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    out = [[candidates[i] for i in c] for c in clusters]
    # full coverage: candidates not in the top clusters still get
    # their own (small) clusters — we return top max_clusters as
    # primary hypotheses and the remainder merged as one residual
    # cluster only if non-empty (residual still goes to the pipeline)
    primary = out[:max_clusters]
    residual = [cand for cl in out[max_clusters:]
                for cand in cl]
    if residual:
        primary = primary + [residual]
    return primary


def hypothesis_rank(cluster, ransac_support, icp_fitness,
                    bidir_consistency, surface_overlap):
    """GT-free hypothesis score: equal-weight rank inputs
    (pre-registered; higher better).  All inputs are already
    GT-free quantities computed by the existing pipeline."""
    return (ransac_support + icp_fitness
            + bidir_consistency + surface_overlap
            + len(cluster))


def _edge_state(adjacency, u, v):
    """Return True/False/None for edge present/absent/unknown."""
    if adjacency is None:
        return None
    if u not in adjacency or v not in adjacency:
        return None
    return v in adjacency[u] or u in adjacency[v]


def _compatible_corrected(a, b, c, d, centres_src, centres_ref,
                          extents_src=None, extents_ref=None,
                          semantic_src=None, semantic_ref=None,
                          adjacency_src=None, adjacency_ref=None):
    """Three-state corrected compatibility used only by the C1 shadow.

    Missing semantic/adjacency data is neutral and explicitly audited;
    it is never fabricated as a positive match.
    """
    if not _compatible(a, b, c, d, centres_src, centres_ref,
                       extents_src=None, extents_ref=None,
                       semantics=None):
        return False
    if extents_src is not None and extents_ref is not None:
        for src_idx, ref_idx in ((a, b), (c, d)):
            src_extent = float(extents_src[src_idx])
            ref_extent = float(extents_ref[ref_idx])
            if src_extent <= 0 or ref_extent <= 0:
                return False
            endpoint_ratio = src_extent / ref_extent
            if not (EXTENT_LO <= endpoint_ratio <= EXTENT_HI):
                return False
    if semantic_src is not None and semantic_ref is not None:
        sa, sc = semantic_src.get(a), semantic_src.get(c)
        rb, rd = semantic_ref.get(b), semantic_ref.get(d)
        if None not in (sa, sc, rb, rd):
            if (sa == sc) != (rb == rd):
                return False
    es = _edge_state(adjacency_src, a, c)
    er = _edge_state(adjacency_ref, b, d)
    if es is not None and er is not None and es != er:
        return False
    return True


def cluster_candidates_corrected(
        candidates, centres_src, centres_ref,
        extents_src=None, extents_ref=None,
        semantic_src=None, semantic_ref=None,
        adjacency_src=None, adjacency_ref=None):
    """Deterministic C1 clusters without residual merging.

    Candidates are canonicalised so a permutation of the matcher output
    cannot change the resulting partition.  Every returned cluster is a
    clique under the corrected compatibility relation and no incompatible
    tail clusters are flattened together.
    """
    canonical = sorted({(int(a), int(b)) for a, b in candidates})
    n = len(canonical)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a, b = canonical[i]
        for j in range(i + 1, n):
            c, d = canonical[j]
            ok = _compatible_corrected(
                a, b, c, d, centres_src, centres_ref,
                extents_src, extents_ref, semantic_src, semantic_ref,
                adjacency_src, adjacency_ref)
            adj[i, j] = adj[j, i] = bool(ok)
    assigned = set()
    clusters = []
    degree_order = sorted(range(n), key=lambda i: (-int(adj[i].sum()),
                                                    canonical[i]))
    for start in degree_order:
        if start in assigned:
            continue
        cluster = [start]
        assigned.add(start)
        for idx in degree_order:
            if idx in assigned:
                continue
            if all(adj[idx, other] for other in cluster):
                cluster.append(idx)
                assigned.add(idx)
        clusters.append(sorted(canonical[i] for i in cluster))
    clusters.sort(key=lambda c: (-len(c), tuple(c)))
    return clusters


def _ordinal(values, higher_better=True):
    """Dense pair-local ordinal scores in [0, 1]."""
    def finite_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    converted = [finite_float(value) for value in values]
    finite = sorted({value for value in converted if value is not None})
    if not finite:
        return [0.0 for _ in values]
    denom = max(1, len(finite) - 1)
    index = {v: i / denom for i, v in enumerate(finite)}
    out = []
    for value in converted:
        if value is None:
            out.append(0.0)
            continue
        else:
            score = index[value]
        out.append(score if higher_better else 1.0 - score)
    return out


def rank_hypotheses_corrected(records):
    """Select C1 by the exact pre-registered GT-free lexicographic keys.

    The order is known-success, bidirectional usability, pair-local ordinal
    RANSAC support, ICP fitness, surface overlap, then cluster size.  A stable
    signature is used only to make a complete tie deterministic.
    """
    if not records:
        return None, []
    features = {
        "ransac_support": _ordinal([
            r.get("ransac_support", np.nan) for r in records]),
        "icp_fitness": _ordinal([
            r.get("icp_fitness", np.nan) for r in records]),
        "surface_overlap": _ordinal([
            r.get("surface_overlap", np.nan) for r in records]),
    }
    ranked = []
    for i, record in enumerate(records):
        ranks = {name: values[i] for name, values in features.items()}
        signature = str(record.get("stable_signature", ""))
        required = (
            record.get("ransac_support"), record.get("icp_fitness"),
            record.get("surface_overlap"))
        known_success = bool(record.get("registration_valid", False))
        for value in required:
            try:
                known_success = known_success and value is not None \
                    and bool(np.isfinite(float(value)))
            except (TypeError, ValueError):
                known_success = False
        key = (
            int(known_success),
            int(record.get("bidirectional_available", False)),
            ranks["ransac_support"],
            ranks["icp_fitness"],
            ranks["surface_overlap"],
            int(record.get("cluster_size", 0)),
            signature,
        )
        enriched = dict(record)
        enriched["known_success"] = known_success
        enriched["ordinal_ranks"] = ranks
        enriched["selection_key"] = list(key[:-1]) + [signature]
        ranked.append(enriched)
    winner = max(ranked, key=lambda r: tuple(r["selection_key"]))
    return winner, ranked
