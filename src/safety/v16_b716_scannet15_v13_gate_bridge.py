"""Exact15 prepared-surface identity loader beside the frozen V13 gate.

The module validates provenance and exposes only current-coordinate surfaces.
It contains no registration thresholds, Rule-B logic, solver, ICP, or decision
code; the reviewed V13 gate remains byte-for-byte unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from safety.v16_b716_scannet15_identity import (
    ScanNet15IdentityError, sha256_file, validate_prepared_npz,
    validate_preregister,
)


def load_verified_surfaces(
    prepared_path: Path, *, pair_id: str, arm: str,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if arm != "sgf_selected_union":
        raise ScanNet15IdentityError("exact15 gate bridge accepts primary arm only")
    validate_preregister(preregister)
    validate_prepared_npz(
        prepared_path, pair_id=pair_id, preregister=preregister)
    prepared_path = Path(prepared_path).resolve()
    before = sha256_file(prepared_path)
    with np.load(prepared_path, allow_pickle=False) as data:
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        values = {
            side: {
                key: np.asarray(data[f"{arm}_{side}_{key}"])
                for key in ("xyz", "labels")}
            for side in ("source", "reference")}
    if sha256_file(prepared_path) != before:
        raise ScanNet15IdentityError("prepared NPZ changed while loading surfaces")
    return {
        "path": str(prepared_path), "sha256": before,
        "manifest": manifest,
        "source": values["source"]["xyz"].astype(np.float64),
        "reference": values["reference"]["xyz"].astype(np.float64),
        "source_labels": values["source"]["labels"].astype(np.int64),
        "reference_labels": values["reference"]["labels"].astype(np.int64),
        "execution_authorized": False,
    }
