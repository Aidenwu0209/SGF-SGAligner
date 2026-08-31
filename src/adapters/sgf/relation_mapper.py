"""SGF relation vocabulary -> official 3DSSG 41-D relation mapping.

Official vocabulary (checkpoints/release/relationships.txt, 41 entries):
none + 40 relation names.  SGF GraphPredictor predicts a 9-relation
subset (3DSSG-derived).  Only exact string matches are mapped; anything
else stays mask=false.  No semantically-adjacent substitutions.

This module is project adaptation code, clearly separated from the
official repository sources (which are never modified).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OFFICIAL_VOCAB_FILE = (
    "/home/aidenwu/Documents/sgaligner-sgf-official/checkpoints/release/"
    "relationships.txt"
)

# SGF GraphPredictor relation outputs (from /home/aidenwu/opt/traced/
# relationships.txt, 9 entries incl. 'none') — all exact substrings of
# official names except the listed aliases which we do NOT auto-map.
SGF_RELATIONS = (
    "attached to", "build in", "connected to", "hanging on",
    "part of", "same part", "standing on", "supported by", "none",
)


def load_official_vocab(path: str | Path = OFFICIAL_VOCAB_FILE) -> list[str]:
    lines = Path(path).read_text().splitlines()
    vocab = [line.strip() for line in lines if line.strip()]
    if len(vocab) != 41:
        raise ValueError(
            f"official relation vocabulary must have 41 entries, got {len(vocab)}"
        )
    return vocab


class RelationMapper:
    """Deterministic SGF->official relation-name mapping (exact only)."""

    def __init__(self, official_vocab_path: str | Path = OFFICIAL_VOCAB_FILE):
        self.official_vocab = load_official_vocab(official_vocab_path)
        self.name_to_index = {
            name: index for index, name in enumerate(self.official_vocab)
        }
        # 'same part' is predicted by SGF but absent from the official
        # 41-word vocabulary -> stays unmapped (mask=false), never
        # silently mapped to 'same as'/'same object type'.
        self.mapped = {}
        self.unmapped = set()
        for relation in SGF_RELATIONS:
            if relation in self.name_to_index:
                self.mapped[relation] = self.name_to_index[relation]
            else:
                self.unmapped.add(relation)

    def bow_vector(self, relation_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """41-D outgoing-relation BoW + per-slot availability mask."""
        vector = np.zeros(41, dtype=np.float32)
        mask = np.zeros(41, dtype=np.float32)
        for name in relation_names:
            index = self.name_to_index.get(name)
            if index is not None:
                vector[index] += 1.0
                mask[index] = 1.0
        return vector, mask

    def coverage_report(self, used_relation_names: list[str]) -> dict:
        used = set(used_relation_names)
        return {
            "official_vocab_size": len(self.official_vocab),
            "sgf_relations": list(SGF_RELATIONS),
            "exact_mapped": sorted(self.mapped),
            "unmapped": sorted(self.unmapped),
            "no_relation_index": self.name_to_index.get("none"),
            "used_names": sorted(used),
            "used_mapped_share": (
                sum(1 for n in used if n in self.mapped) / len(used)
                if used else None
            ),
        }

    def dump_mapping(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "official_vocab": self.official_vocab,
                    "sgf_to_official": self.mapped,
                    "unmapped_sgf_relations": sorted(self.unmapped),
                    "rule": "exact string match only; no semantic substitution",
                },
                indent=2,
            )
            + "\n"
        )
