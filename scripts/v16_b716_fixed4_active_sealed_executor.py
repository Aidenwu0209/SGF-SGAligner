#!/usr/bin/python3.12
"""Fresh-process active fixed4 boundary; production remains fail closed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "src/safety/v16_b716_fixed4_subprocess_contract.py"
CONTRACT_SHA256 = "1c6bb539b0f7be5ca48228148761acbb5450fc9fc566b1cd8d8eceb052dda788"
if (CONTRACT.is_symlink() or not CONTRACT.is_file()
        or hashlib.sha256(CONTRACT.read_bytes()).hexdigest() != CONTRACT_SHA256):
    raise SystemExit(70)
sys.path.insert(0, str(REPO / "src"))

from safety.v16_b716_fixed4_subprocess_contract import (  # noqa: E402
    CONTRACT_FIXTURE_STAGE, Fixed4SubprocessContractError,
    RUNNER_MODE_ACTIVE, build_subprocess_registry, execute_active_stage,
    no_symlink_file_row, read_no_symlink_bytes,
    validate_active_control_bindings, validate_active_production_control_bindings,
    verify_fixed_signed_document,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("repo", "task", "preflight", "authorization", "task-manifest",
                 "task-root"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("task", "preflight", "authorization", "task-manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--execution-manifest")
    parser.add_argument("--execution-manifest-sha256")
    parser.add_argument("--runner-mode", required=True, choices=(RUNNER_MODE_ACTIVE,))
    parser.add_argument("--allow-contract-fixture", action="store_true")
    return parser


def _json(path: Path, role: str) -> dict:
    try:
        value = json.loads(read_no_symlink_bytes(path, role))
    except Exception as exc:
        raise Fixed4SubprocessContractError(f"invalid {role} JSON") from exc
    if not isinstance(value, dict):
        raise Fixed4SubprocessContractError(f"invalid {role} document")
    return value


def _validate_authorization_ttl(value: dict) -> None:
    try:
        issued = datetime.fromisoformat(str(value["issued_at"]).replace("Z", "+00:00"))
        expires = datetime.fromisoformat(
            str(value["expires_at"]).replace("Z", "+00:00"))
        if issued.tzinfo is None or expires.tzinfo is None:
            raise ValueError
        issued = issued.astimezone(timezone.utc)
        expires = expires.astimezone(timezone.utc)
    except Exception as exc:
        raise Fixed4SubprocessContractError(
            "active authorization TTL malformed") from exc
    now = datetime.now(timezone.utc)
    if (issued.timestamp() > now.timestamp() + 60
            or expires <= now or expires <= issued
            or (expires - issued).total_seconds() > 3600):
        raise Fixed4SubprocessContractError(
            "active authorization TTL invalid/expired")


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
            raise Fixed4SubprocessContractError(f"{role} SHA drift in active executor")
    task = _json(controls["task"][0], "task")
    preflight = _json(controls["preflight"][0], "preflight")
    authorization = _json(controls["authorization"][0], "authorization")
    if authorization.get("signer_private_key_not_on_execution_host") is not True:
        raise Fixed4SubprocessContractError("off-host signer assertion missing")
    verify_fixed_signed_document(
        authorization, repo_root=Path(args.repo),
        output_root=controls["preflight"][0].parent,
        purpose="active sealed executor authorization")
    _validate_authorization_ttl(authorization)
    rows, digest = build_subprocess_registry(
        Path(args.repo), runner_mode=RUNNER_MODE_ACTIVE,
        include_contract_fixture=args.allow_contract_fixture)
    if task.get("stage") == CONTRACT_FIXTURE_STAGE:
        if args.execution_manifest or args.execution_manifest_sha256:
            raise Fixed4SubprocessContractError(
                "contract fixture cannot consume a production manifest")
        validate_active_control_bindings(
            task=task, preflight=preflight, authorization=authorization,
            registry_rows=rows, registry_sha256=digest,
            include_contract_fixture=args.allow_contract_fixture)
    else:
        if (args.allow_contract_fixture or not args.execution_manifest
                or not args.execution_manifest_sha256):
            raise Fixed4SubprocessContractError(
                "production-v2 execution manifest binding missing")
        execution_manifest = _json(
            Path(args.execution_manifest), "production execution manifest")
        validate_active_production_control_bindings(
            task=task, preflight=preflight, authorization=authorization,
            execution_manifest=execution_manifest,
            execution_manifest_path=Path(args.execution_manifest), repo=Path(args.repo),
            registry_rows=rows, registry_sha256=digest)
    receipt = execute_active_stage(
        repo=Path(args.repo), task=task, task_path=controls["task"][0],
        preflight_path=controls["preflight"][0],
        authorization_path=controls["authorization"][0],
        task_manifest_path=controls["task_manifest"][0],
        task_root=Path(args.task_root), registry_rows=rows,
        registry_sha256=digest,
        execution_manifest_path=(Path(args.execution_manifest)
                                 if args.execution_manifest else None),
        execution_manifest_sha256=args.execution_manifest_sha256,
        include_contract_fixture=args.allow_contract_fixture)
    if receipt.get("failure_type") is None:
        return 0
    raise Fixed4SubprocessContractError("active child result classification drift")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Fixed4SubprocessContractError:
        raise SystemExit(70)
