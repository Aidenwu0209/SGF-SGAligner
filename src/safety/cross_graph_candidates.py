"""Deterministic GT-free cross-graph candidate generation.

The official embedding and full rank list remain unchanged.  This adapter
changes only where the fixed top-k is applied: each node first filters to the
opposite graph, then mutual rank and a global resource cap are enforced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence


@dataclass(frozen=True)
class CrossGraphCandidateConfig:
    cross_graph_k: int = 5
    require_mutual: bool = True
    max_candidates_per_pair: int = 48


class CrossGraphCandidateError(RuntimeError):
    """The rank-list contract is malformed or cannot be evaluated."""


def _validate(rank_list: Sequence[Sequence[int]], src_count: int) -> None:
    size = len(rank_list)
    if not (0 < src_count < size):
        raise CrossGraphCandidateError("src_count must split two nonempty graphs")
    expected = set(range(size))
    for index, ranking in enumerate(rank_list):
        if (len(ranking) != size or set(ranking) != expected
                or any(type(value) is not int for value in ranking)):
            raise CrossGraphCandidateError(
                f"rank_list[{index}] is not a full integer permutation")


def cross_graph_candidates(
    rank_list: Sequence[Sequence[int]],
    src_count: int,
    config: CrossGraphCandidateConfig = CrossGraphCandidateConfig(),
) -> list[dict[str, int]]:
    """Return a stable, resource-bounded mutual cross-graph candidate list."""
    _validate(rank_list, src_count)
    if config.cross_graph_k < 1 or config.max_candidates_per_pair < 1:
        raise CrossGraphCandidateError("candidate limits must be positive")
    size = len(rank_list)
    forward = {
        source: [target for target in rank_list[source]
                 if target >= src_count][:config.cross_graph_k]
        for source in range(src_count)
    }
    reverse = {
        reference: [source for source in rank_list[reference]
                    if source < src_count][:config.cross_graph_k]
        for reference in range(src_count, size)
    }
    rows = []
    for source, references in forward.items():
        for forward_offset, reference in enumerate(references, 1):
            reverse_sources = reverse[reference]
            if source not in reverse_sources:
                if config.require_mutual:
                    continue
                reverse_rank = config.cross_graph_k + 1
            else:
                reverse_rank = reverse_sources.index(source) + 1
            rows.append({
                "source_index": source,
                "reference_index": reference,
                "forward_cross_rank": forward_offset,
                "reverse_cross_rank": reverse_rank,
                "worst_cross_rank": max(forward_offset, reverse_rank),
                "rank_sum": forward_offset + reverse_rank,
            })
    rows.sort(key=lambda row: (
        row["worst_cross_rank"], row["rank_sum"],
        row["forward_cross_rank"], row["reverse_cross_rank"],
        row["source_index"], row["reference_index"]))
    return rows[:config.max_candidates_per_pair]


def candidate_fingerprint(candidates: Sequence[dict[str, Any]],
                          config: CrossGraphCandidateConfig) -> str:
    payload = {
        "config": {name: getattr(config, name)
                   for name in config.__dataclass_fields__},
        "candidates": list(candidates),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
