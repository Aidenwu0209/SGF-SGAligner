#!/usr/bin/python3.12
"""Fresh-process fixed4 executor; metadata-only and permanently disabled."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from safety.v16_b716_fixed4_subprocess_contract import (  # noqa: E402
    DISABLED_EXIT_CODE, Fixed4SubprocessContractError, build_subprocess_registry,
    execute_disabled_stage, no_symlink_file_row, read_no_symlink_bytes,
    verify_fixed_signed_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("repo", "task", "preflight", "authorization", "task-manifest",
                 "task-root"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("task", "preflight", "authorization", "task-manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    controls = {
        "task": (Path(args.task), args.task_sha256),
        "preflight": (Path(args.preflight), args.preflight_sha256),
        "authorization": (Path(args.authorization), args.authorization_sha256),
        "task_manifest": (Path(args.task_manifest), args.task_manifest_sha256),
    }
    for role, (path, expected) in controls.items():
        if no_symlink_file_row(path, role)["sha256"] != expected:
            raise Fixed4SubprocessContractError(f"{role} SHA drift in sealed executor")
    task = json.loads(read_no_symlink_bytes(controls["task"][0], "task"))
    preflight = json.loads(read_no_symlink_bytes(controls["preflight"][0], "preflight"))
    authorization = json.loads(read_no_symlink_bytes(
        controls["authorization"][0], "authorization"))
    if authorization.get("signer_private_key_not_on_execution_host") is not True:
        raise Fixed4SubprocessContractError("off-host signer assertion missing")
    verify_fixed_signed_document(
        authorization, repo_root=Path(args.repo),
        output_root=controls["preflight"][0].parent,
        purpose="sealed executor authorization")
    rows, digest = build_subprocess_registry(Path(args.repo))
    if (preflight.get("runner_registry") != rows
            or preflight.get("runner_registry_closure_sha256") != digest):
        raise Fixed4SubprocessContractError("sealed executor registry drift")
    receipt = execute_disabled_stage(
        repo=Path(args.repo), task=task, task_path=controls["task"][0],
        preflight_path=controls["preflight"][0],
        authorization_path=controls["authorization"][0],
        task_manifest_path=controls["task_manifest"][0],
        task_root=Path(args.task_root), registry_rows=rows,
        registry_sha256=digest)
    if receipt.get("failure_type") != "CHECKED_IN_RUNNER_EXECUTION_DISABLED":
        raise Fixed4SubprocessContractError("sealed child did not fail in disabled state")
    return DISABLED_EXIT_CODE


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fixed4SubprocessContractError:
        raise SystemExit(70)
