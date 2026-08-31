#!/usr/bin/env python3
"""CPU-only contract generator for the fixed4 production-stage adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v16_b716_fixed4_production_adapters import (
    build_stage_adapter_contract,
    expand_verified_candidate_slots,
    finalize_v15_from_slot_results,
    load_bound_input_manifest,
    materialize_contract_create_only,
)
from safety.v13_dual_solver_runtime import sha256_file


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("contract", "expand-slots", "finalize-v15"),
                        default="contract")
    parser.add_argument("--candidate-set-sha256")
    parser.add_argument("--slot-expansion", type=Path)
    parser.add_argument("--slot-expansion-sha256")
    parser.add_argument("--slot-results", type=Path)
    parser.add_argument("--slot-results-sha256")
    args = parser.parse_args()
    task = _json(args.task)
    allowed = (() if args.mode == "contract" else
        ("forward_candidate_dir", "reverse_candidate_dir", "candidate_set")
        if args.mode == "expand-slots" else
        ("forward_candidate_dir", "reverse_candidate_dir", "candidate_set",
         "slot_root"))
    manifest = load_bound_input_manifest(
        args.input_manifest, args.input_manifest_sha256, task, args.output_root,
        allowed_existing_output_roles=allowed)
    if args.mode == "contract":
        contract = build_stage_adapter_contract(task, manifest, args.output_root)
    elif args.mode == "expand-slots":
        if not args.candidate_set_sha256:
            parser.error("--candidate-set-sha256 is required for expand-slots")
        contract = expand_verified_candidate_slots(
            task, manifest, args.candidate_set_sha256, args.output_root)
    else:
        if (args.slot_expansion is None or args.slot_results is None
                or not args.slot_expansion_sha256
                or not args.slot_results_sha256):
            parser.error("finalize-v15 requires slot expansion/results paths and SHAs")
        if sha256_file(args.slot_expansion) != args.slot_expansion_sha256:
            parser.error("slot expansion SHA mismatch")
        expansion = _json(args.slot_expansion)
        contract = finalize_v15_from_slot_results(
            task, manifest, expansion, args.slot_results,
            args.slot_results_sha256, args.output_root)
    receipt = materialize_contract_create_only(
        args.output_root, args.output, contract)
    print(json.dumps({"status": "ARTIFACT_GENERATED_NOT_AUTHORIZED",
                      "mode": args.mode,
                      "contract": receipt}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
