"""Provenance records for the official-migration project.

Tracks upstream pins, downloaded checkpoint hashes, and adapter build
inputs so every artifact is reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class UpstreamProvenance:
    repository: str = "https://github.com/sayands/sgaligner"
    commit: str = "51cd5723513d7e59145d26433ceb3a0b10298748"
    branch: str = "wu/official-sgf-adapter-v1"
    geotransformer_submodule: str = "9bba3040d2a258b9cb4272293a4eed87d24a9202"
    license: str = "MIT (official repository); GeoTransformer Apache-2.0"
    note: str = (
        "official sources are never modified; project adaptation code "
        "lives under src/adapters, src/safety, src/reconstruction"
    )


CHECKPOINT_SOURCES = {
    "sgaligner_pct_gat_rel_attr.pth.tar": (
        "https://drive.google.com/drive/folders/"
        "10-JNxWLxPFQ2q6_zY-9HXIO-Qx-vhmmT (official release folder)"
    ),
    "geotransformer-3dmatch.pth.tar": (
        "https://github.com/qinzheng93/GeoTransformer/releases/tag/1.0.0"
    ),
    "relationships.json (3DSSG GT)": (
        "https://campar.in.tum.de/public_datasets/3DSSG/3DSSG/"
        "relationships.json"
    ),
}


def checkpoint_manifest(
    checkpoints_dir: str | Path = (
        "/home/aidenwu/Documents/sgaligner-sgf-official/checkpoints"
    ),
) -> dict:
    root = Path(checkpoints_dir)
    entries = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".tar", ".pkl", ".txt"}:
            entries[str(path.relative_to(root))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "source": CHECKPOINT_SOURCES.get(
                    path.name, "official release folder"
                ),
            }
    return {"upstream": asdict(UpstreamProvenance()), "files": entries}


def git_state(repo: str | Path) -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True,
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_clean": not run("status", "--short"),
        "log": run("log", "--oneline", "-8"),
    }
