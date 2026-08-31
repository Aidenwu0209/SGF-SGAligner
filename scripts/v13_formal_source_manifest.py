#!/usr/bin/env python3
"""Single authoritative manifest for code that can affect a formal V13 run.

The one-pair CLI records this exact map, and the fixed4 orchestrator recomputes
it for fresh execution, receipt creation, resume, and summary revalidation.
Adding a formal runtime component therefore requires changing one list only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


FORMAL_SOURCE_PATHS = {
    "formal_source_manifest": "scripts/v13_formal_source_manifest.py",
    "matrix_driver": "scripts/v13_fixed4_matrix_driver.py",
    "driver": "scripts/v13_fixed4_driver.py",
    "cli": "scripts/v13_dual_solver_cli.py",
    "sentinel_subprocess": "scripts/v13_colorpcr_sentinel_subprocess.py",
    "official_worker": "scripts/v13_colorpcr_official_worker.py",
    "converter": "scripts/v13_corr_cache_converter.py",
    "preflight_builder": "scripts/v13_colorpcr_pointdsc_preflight.py",
    "strict_pair_gate": "src/safety/v13_strict_pair_gate.py",
    "dual_solver_runtime": "src/safety/v13_dual_solver_runtime.py",
    "fixed4_aggregate": "src/safety/v13_fixed4_aggregate.py",
}


class FormalSourceContractError(RuntimeError):
    """A formal source is missing or unreadable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_source_sha256(repo: Path) -> dict[str, str]:
    """Return the exact, centrally-defined formal V13 source hash map."""
    repo = Path(repo).resolve()
    paths = {name: repo / relative
             for name, relative in FORMAL_SOURCE_PATHS.items()}
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FormalSourceContractError(f"formal source missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}
