#!/usr/bin/env python3
"""Prepare or verify the fail-closed b716 fixed4 execution pilot.

This command never executes a model, GPU worker, rigid solver, ICP, refusion,
GT evaluation, or official92.  An independent reviewer must create the
authorization receipt; this command only verifies it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from safety.v13_dual_solver_runtime import sha256_file  # noqa: E402
from safety.v16_b716_fixed4_execution_pilot import (  # noqa: E402
    build_preflight,
    materialize_preflight,
    validate_authorization,
)
from safety.v16_b716_fixed4_subprocess_contract import (  # noqa: E402
    RUNNER_MODE_ACTIVE, RUNNER_MODE_DISABLED,
)


def _prepare(args: argparse.Namespace) -> dict:
    value = build_preflight(
        repo=ROOT,
        preregister_path=args.preregister,
        exact191_path=args.exact191_manifest,
        exact191_sha256=args.exact191_manifest_sha256,
        exact72_lineage_path=args.exact72_lineage_manifest,
        exact72_lineage_sha256=args.exact72_lineage_manifest_sha256,
        prepared_path=args.prepared_manifest,
        prepared_sha256=args.prepared_manifest_sha256,
        output_root=args.output_root,
        runner_mode=args.runner_mode,
        active_preregister_path=args.active_preregister,
    )
    result = materialize_preflight(args.output_root, value)
    public = json.loads(Path(result["preflight"]).read_text())
    active_contract = public.get("active_subprocess_contract")
    production_ready = bool(
        isinstance(active_contract, dict)
        and active_contract.get("schema")
            == "v16-b716-fixed4-active-subprocess-preflight-v2"
        and active_contract.get("production_adapter_protocol_ready") is True)
    return {
        "status": ("PREPARED_ACTIVE_CANDIDATE_NOT_AUTHORIZED"
                   if args.runner_mode == RUNNER_MODE_ACTIVE
                   else "PREPARED_EXECUTION_DISABLED"),
        **result,
        "preflight_file_sha256": sha256_file(Path(result["preflight"])),
        "preflight_payload_sha256": public["payload_sha256"],
        "operational_stage_counts": public["operational_stage_counts"],
        "full_evidence_dag_node_count": public["dag"]["node_count"],
        "execution_authorized": False,
        "runner_mode": args.runner_mode,
        "production_adapter_protocol_ready": production_ready,
        "reconstruction_authorized": False,
        "registration_defense_guard_status": public[
            "registration_defense_guard"]["status"],
    }


def _verify_authorization(args: argparse.Namespace) -> dict:
    value = validate_authorization(
        args.authorization,
        args.authorization_sha256,
        args.preflight,
        args.preflight_sha256,
    )
    return {
        "status": "EXECUTION_AUTHORIZATION_VALID",
        "authorization_file_sha256": sha256_file(args.authorization),
        "authorization_payload_sha256": value["payload_sha256"],
        "allowed_stages": value["allowed_stages"],
        "execution_authorized": True,
        "reconstruction_authorized": False,
        "note": (
            "Any later execution must pass the fixed external signature gate "
            "and hash-bound independent subprocess boundary. This commit's "
            "checked-in runner remains disabled."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--preregister", type=Path,
        default=ROOT / "manifests/v16_b716_fixed4_orchestrator_preregister.json")
    prepare.add_argument("--exact191-manifest", type=Path, required=True)
    prepare.add_argument("--exact191-manifest-sha256", required=True)
    prepare.add_argument("--exact72-lineage-manifest", type=Path, required=True)
    prepare.add_argument("--exact72-lineage-manifest-sha256", required=True)
    prepare.add_argument("--prepared-manifest", type=Path, required=True)
    prepare.add_argument("--prepared-manifest-sha256", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--runner-mode", choices=(RUNNER_MODE_DISABLED,
                                                   RUNNER_MODE_ACTIVE),
                         default=RUNNER_MODE_DISABLED)
    prepare.add_argument("--active-preregister", type=Path,
        default=ROOT / "manifests/v16_b716_fixed4_active_execution_ready_v2_preregister.json")
    prepare.set_defaults(handler=_prepare)

    verify = sub.add_parser("verify-authorization")
    verify.add_argument("--preflight", type=Path, required=True)
    verify.add_argument("--preflight-sha256", required=True)
    verify.add_argument("--authorization", type=Path, required=True)
    verify.add_argument("--authorization-sha256", required=True)
    verify.set_defaults(handler=_verify_authorization)

    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
