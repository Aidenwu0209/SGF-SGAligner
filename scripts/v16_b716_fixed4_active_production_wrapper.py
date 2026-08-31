#!/usr/bin/env python3
"""Code-pinned CLI for one active fixed4 production task.

The caller supplies the already hash-bound interpreter and dependency closure.
This entry point writes only a wrapper receipt; RESULT-v5 remains a candidate
until the parent validation/release gate accepts it create-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_pinned(path: Path, expected_sha256: str) -> bytes:
    """Read once, then use the same bytes for both the hash and parser."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("pinned input is not a regular file")
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise ValueError("pinned input SHA drift")
    return value


def _validated_controlled_sys_path(value: object, repo: Path) -> list[str]:
    expected_prefix = [str(repo), str(repo / "src"), str(repo / "scripts")]
    if (not isinstance(value, list)
            or value[:len(expected_prefix)] != expected_prefix
            or len(value) != len(set(value))
            or any(not isinstance(item, str) or not Path(item).is_absolute()
                   for item in value)):
        raise ValueError("controlled sys.path contract mismatch")
    return list(value)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-sha256", required=True)
    parser.add_argument("--execution-manifest", required=True)
    parser.add_argument("--execution-manifest-sha256", required=True)
    parser.add_argument("--production-manifest-commit", required=True)
    parser.add_argument("--production-manifest-commit-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runner-source-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    task_path = Path(args.task).resolve()
    manifest_path = Path(args.execution_manifest).resolve()
    commit_path = Path(args.production_manifest_commit).resolve()
    output = Path(args.output).resolve()
    module = repo / "src/safety/v16_b716_fixed4_active_production_wrapper.py"
    runner = repo / "scripts/v16_b716_fixed4_active_stage_runner.sh"
    try:
        task_bytes = _read_pinned(task_path, args.task_sha256)
        manifest_bytes = _read_pinned(
            manifest_path, args.execution_manifest_sha256)
        _read_pinned(commit_path, args.production_manifest_commit_sha256)
        _read_pinned(runner, args.runner_source_sha256)
    except (OSError, ValueError):
        return 70
    if module.is_symlink() or not module.is_file():
        return 70
    try:
        execution_manifest = json.loads(manifest_bytes)
        controlled = _validated_controlled_sys_path(
            execution_manifest["controlled_sys_path"], repo)
        # The active runner enters CPython with -I -S.  Only this signed,
        # hash-bound path list is installed; ambient editable .pth files and
        # an older checkout can never influence imports.
        sys.path[:] = controlled
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 70
    from safety.v16_b716_fixed4_active_production_wrapper import (  # noqa: E402
        execute_active_production_wrapper,
    )
    try:
        task = json.loads(task_bytes)
        result = execute_active_production_wrapper(
            task=task, execution_manifest_path=manifest_path,
            execution_manifest_sha256=args.execution_manifest_sha256,
            output_root=Path(args.output_root),
            validator_source_sha256=_sha(module),
            runner_source_sha256=args.runner_source_sha256)
        encoded = (json.dumps(result, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode()
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = output.parent.open("xb")
        try:
            descriptor.write(encoded); descriptor.flush()
        finally:
            descriptor.close()
    except Exception:
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
