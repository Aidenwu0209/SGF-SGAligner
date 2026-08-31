"""Stage 2: cross-graph-only, bidirectional, overlap-weighted
multi-positive InfoNCE for the SGF-predicted adapter.

S = normalize(src_emb) @ normalize(ref_emb).T   [Nsrc, Nref]
forward: each SOURCE query's denominator covers REFERENCE nodes only
reverse: each REFERENCE query's denominator covers SOURCE nodes only
positives are never negatives; no self-similarity can occur (the two
node sets are disjoint graphs); NaN/Inf/empty fail closed.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_graph_infonce(
    src_emb: torch.Tensor,
    ref_emb: torch.Tensor,
    positives: torch.Tensor,   # [Nsrc, Nref] bool
    weights: torch.Tensor,     # [Nsrc, Nref] float in (0, 1]
    *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if src_emb.ndim != 2 or ref_emb.ndim != 2:
        raise ValueError("embeddings must be [N, D]")
    n_src, n_ref = src_emb.shape[0], ref_emb.shape[0]
    if n_src < 1 or n_ref < 1:
        raise ValueError("both graphs must be non-empty")
    if positives.shape != (n_src, n_ref):
        raise ValueError("positives must be [Nsrc, Nref]")
    if weights.shape != (n_src, n_ref):
        raise ValueError("weights must be [Nsrc, Nref]")
    if not torch.isfinite(weights).all():
        raise ValueError("weights contain non-finite values")
    positive_weights = weights[positives]
    if ((positive_weights <= 0) | (positive_weights > 1)).any():
        raise ValueError("positive weights must lie in (0, 1]")
    if (weights[~positives] != 0).any():
        raise ValueError("weights must be zero outside positives")
    if (weights[positives] <= 0).any():
        raise ValueError("every positive must carry a positive weight")
    src_queries = torch.nonzero(positives.any(dim=1)).squeeze(1)
    ref_queries = torch.nonzero(positives.any(dim=0)).squeeze(1)
    if len(src_queries) == 0 or len(ref_queries) == 0:
        raise ValueError("no query has a positive (empty positives)")

    src_norm = F.normalize(src_emb, dim=1)
    ref_norm = F.normalize(ref_emb, dim=1)
    similarity = src_norm @ ref_norm.T  # [Nsrc, Nref], no self terms

    logits = similarity / temperature

    def direction(query_logits, query_positives, query_weights):
        per_query = []
        q_weights = []
        for qi in range(query_logits.shape[0]):
            js = torch.nonzero(query_positives[qi]).squeeze(1)
            if len(js) == 0:
                continue  # this row has no positive: not a query here
            num = torch.logsumexp(query_logits[qi, js], dim=0)
            den = torch.logsumexp(query_logits[qi], dim=0)
            per_query.append(den - num)
            q_weights.append(query_weights[qi, js].max())
        if not per_query:
            raise ValueError("no query with a positive in this direction")
        per_query = torch.stack(per_query)
        q_weights = torch.stack(q_weights)
        if not torch.isfinite(per_query).all():
            raise ValueError("non-finite loss term (fail closed)")
        return (per_query * q_weights).sum() / q_weights.sum().clamp_min(
            1e-12
        )

    forward = direction(logits, positives, weights)
    reverse = direction(logits.T, positives.T, weights.T)
    loss = 0.5 * (forward + reverse)
    with torch.no_grad():
        pos_sim = similarity[positives]
        neg_mask = ~positives
        neg_sim = similarity[neg_mask] if neg_mask.any() else torch.zeros(1)
        margin = (pos_sim.mean() - neg_sim.mean()) if len(neg_sim) else pos_sim.mean()
    diagnostics = {
        "positive_similarity": float(pos_sim.mean()),
        "cross_graph_negative_similarity": (
            float(neg_sim.mean()) if len(neg_sim) else None
        ),
        "margin": float(margin),
        "forward_queries": int(len(src_queries)),
        "reverse_queries": int(len(ref_queries)),
        "positive_pairs": int(positives.sum()),
    }
    return loss, diagnostics


# ----------------------------------------------------------------------
# explicit-loop reference implementation (independent; shares no code)
# ----------------------------------------------------------------------

def reference_loss(
    src_emb, ref_emb, positives, weights, temperature=0.1
):
    import numpy as np

    src = np.asarray(
        src_emb.detach().cpu().numpy(), dtype=np.float64
    )
    ref = np.asarray(
        ref_emb.detach().cpu().numpy(), dtype=np.float64
    )
    pos = np.asarray(positives.detach().cpu().numpy(), dtype=bool)
    w = np.asarray(weights.detach().cpu().numpy(), dtype=np.float64)
    src = src / np.linalg.norm(src, axis=1, keepdims=True)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    sim = src @ ref.T

    def one_direction(logits, P, W):
        per = []
        qw = []
        for i in range(logits.shape[0]):
            js = [j for j in range(logits.shape[1]) if P[i, j]]
            if not js:
                continue
            mx = max(logits[i, j] for j in js)
            num = mx + math.log(
                sum(math.exp(logits[i, j] - mx) for j in js)
            )
            mall = max(logits[i])
            den = mall + math.log(
                sum(math.exp(v - mall) for v in logits[i])
            )
            per.append(den - num)
            qw.append(max(W[i, j] for j in js))
        total = sum(p * q for p, q in zip(per, qw))
        return total / sum(qw)

    fwd = one_direction(sim / temperature, pos, w)
    rev = one_direction((sim / temperature).T, pos.T, w.T)
    return 0.5 * (fwd + rev)
