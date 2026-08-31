#!/usr/bin/env python3
"""Build one task's canonical production input/execution manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from safety.v13_dual_solver_runtime import sha256_file  # noqa: E402
from safety.v16_b716_fixed4_production_manifest_builder import (  # noqa: E402
    ProductionManifestBuilderError, load_canonical_upstream_results,
    materialize_production_manifests,
)


def _load(path: Path, expected: str, role: str) -> dict:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise ProductionManifestBuilderError(f"{role} path/SHA invalid")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ProductionManifestBuilderError(f"{role} must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-sha256", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--production-assets-manifest", required=True)
    parser.add_argument("--production-assets-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    task = _load(Path(args.task), args.task_sha256, "task")
    preflight = _load(Path(args.preflight), args.preflight_sha256, "preflight")
    assets = _load(Path(args.production_assets_manifest),
                   args.production_assets_manifest_sha256, "assets manifest")
    runtime = _load(Path(args.runtime_manifest), args.runtime_manifest_sha256,
                    "runtime manifest")
    root = Path(args.output_root).resolve()
    parents = load_canonical_upstream_results(task, root)
    receipt = materialize_production_manifests(
        task=task, preflight=preflight, output_root=root,
        upstream_results=parents, production_assets_manifest=assets,
        runtime_manifest=runtime)
    print(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProductionManifestBuilderError, OSError, ValueError, json.JSONDecodeError):
        raise SystemExit(70)
