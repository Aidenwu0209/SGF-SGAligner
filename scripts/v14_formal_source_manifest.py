#!/usr/bin/env python3
"""Central exact source map for any future V14 fixed4 research run."""
from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.v13_formal_source_manifest import (
    FORMAL_SOURCE_PATHS as V13_FORMAL_SOURCE_PATHS,
    FormalSourceContractError,
    _sha256_file,
)


FORMAL_SOURCE_PATHS = {
    **{f"v13_{name}": relative
       for name, relative in V13_FORMAL_SOURCE_PATHS.items()},
    "v14_formal_source_manifest": "scripts/v14_formal_source_manifest.py",
    "v14_module": "src/safety/v14_rigid_multihypothesis.py",
    "v14_builder": "scripts/v14_rigid_multihypothesis_builder.py",
    "v14_candidate_strict_runner": "scripts/v14_candidate_strict_runner.py",
    "v14_fixed4_orchestrator": "scripts/v14_fixed4_research_orchestrator.py",
    "v14_fixed4_input_builder": "scripts/v14_fixed4_input_builder.py",
}


def formal_source_sha256(repo: Path) -> dict[str, str]:
    """Rehash all inherited V13 and V14 code that can affect evidence."""
    repo = Path(repo).resolve()
    paths = {name: repo / relative
             for name, relative in FORMAL_SOURCE_PATHS.items()}
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise FormalSourceContractError(f"formal source missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def verify_reviewed_source_authorization(
    repo: Path, preregister: dict,
) -> dict[str, str]:
    """Require the current formal code to equal one exact reviewed commit."""
    repo = Path(repo).resolve()
    commit = preregister.get("reviewed_source_commit")
    reviewed = preregister.get("reviewed_formal_source_sha256")
    current = formal_source_sha256(repo)
    if (not isinstance(commit, str) or len(commit) != 40
            or not isinstance(reviewed, dict) or reviewed != current
            or set(reviewed) != set(FORMAL_SOURCE_PATHS)):
        raise FormalSourceContractError(
            "V14 reviewed formal source authorization is missing or stale")
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=repo,
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FormalSourceContractError(
            "V14 reviewed source commit is not locally verifiable") from exc
    if resolved != commit:
        raise FormalSourceContractError("V14 reviewed source commit is not exact")
    for name, relative in FORMAL_SOURCE_PATHS.items():
        try:
            payload = subprocess.run(
                ["git", "show", f"{commit}:{relative}"], cwd=repo,
                check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FormalSourceContractError(
                f"reviewed source missing at commit: {name}") from exc
        import hashlib
        if hashlib.sha256(payload).hexdigest() != reviewed[name]:
            raise FormalSourceContractError(
                f"reviewed source commit/hash mismatch: {name}")
    return current
