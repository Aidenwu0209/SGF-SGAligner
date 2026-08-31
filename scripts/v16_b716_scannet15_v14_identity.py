#!/usr/bin/env python3
"""Identity-only exact15 bridge for the frozen V14 builder/strict runner.

This sibling entry point deliberately does not import V14 execution code.  It
validates a pair or a prepared/preflight closure and always reports execution
as unauthorized.  The reviewed fixed4 V14 sources therefore remain byte-for-
byte unchanged until a separate authorization reviews an execution adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for item in (REPO, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from safety.v16_b716_scannet15_identity import (  # noqa: E402
    ScanNet15IdentityError, pair_row, sha256_file, validate_preregister,
    verify_preflight_closure, verify_source_closure,
)


def load_identity(path: Path) -> dict:
    path = Path(path).resolve()
    before = sha256_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise ScanNet15IdentityError("identity JSON changed while reading")
    validate_preregister(value)
    verify_source_closure(value)
    return value


def validate_pair(path: Path, pair_id: str) -> dict:
    value = load_identity(path)
    row = pair_row(value, pair_id)
    return {
        "schema": "v16-b716-scannet15-v14-identity-validation-v1",
        "pair_id": pair_id,
        "scene_id": row["scene_id"],
        "identity_payload_sha256": row["identity_payload_sha256"],
        "preregister_sha256": sha256_file(Path(path).resolve()),
        "identity_valid": True,
        "execution_authorized": False,
        "gpu_authorized": False,
        "solver_or_icp_executed": False,
    }


def validate_preflight(
    preregister: Path, preflight: Path, prepared: Path, pair_id: str,
) -> dict:
    load_identity(preregister)
    receipt = verify_preflight_closure(
        preregister_path=preregister, preflight_path=preflight,
        prepared_path=prepared, pair_id=pair_id)
    return {
        "schema": "v16-b716-scannet15-v14-strict-preflight-v1",
        "pair_id": pair_id,
        "identity_receipt": receipt,
        "preflight_valid": True,
        "execution_authorized": False,
        "gpu_authorized": False,
        "solver_or_icp_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("validate-builder-identity")
    pair.add_argument("--preregister", required=True, type=Path)
    pair.add_argument("--pair-id", required=True)
    strict = sub.add_parser("validate-strict-preflight")
    strict.add_argument("--preregister", required=True, type=Path)
    strict.add_argument("--preflight", required=True, type=Path)
    strict.add_argument("--prepared", required=True, type=Path)
    strict.add_argument("--pair-id", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate-builder-identity":
            result = validate_pair(args.preregister, args.pair_id)
        else:
            result = validate_preflight(
                args.preregister, args.preflight, args.prepared, args.pair_id)
    except (OSError, ValueError, json.JSONDecodeError,
            ScanNet15IdentityError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
