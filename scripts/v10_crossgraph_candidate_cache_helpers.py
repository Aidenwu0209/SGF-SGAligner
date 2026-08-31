"""Small atomic writer isolated for unit testing and import safety."""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import torch


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
