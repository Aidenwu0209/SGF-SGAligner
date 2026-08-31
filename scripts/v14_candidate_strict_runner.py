#!/usr/bin/env python3
"""Run one sealed V14 hypothesis through the unchanged V13 strict stack.

This is CPU-only: ColorPCR correspondence generation is outside this runner.
The script is implemented for review and tests; the pre-registration does not
authorize a real fixed4 pilot yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v13_dual_solver_runtime import atomic_json, run_matrix, sha256_file
from safety.v13_strict_pair_gate import strict_pair_gate
from safety.v14_rigid_multihypothesis import load_candidate_contract
from scripts.v13_dual_solver_cli import (
    _load_json, _verify_fixed_identity, _verify_preflight_closure,
    _verify_safety_authority,
)
from scripts.v7_registration_pilot import segment_icp_with_trace, rule_b_features
from scripts.v14_formal_source_manifest import (
    formal_source_sha256, verify_reviewed_source_authorization,
)


def verify_v14_authorization(path: Path, contract: dict) -> dict:
    path = Path(path).resolve()
    value = _load_json(path)
    if (value.get("schema")
            != "v14-rigid-multihypothesis-preregister-v1"
            or value.get("allow_real_pilot") is not True
            or value.get("allow_gpu_pilot") is not False
            or value.get("gt_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("posthoc_allowed") is not False):
        raise RuntimeError("V14 real CPU pilot is not explicitly authorized")
    verify_reviewed_source_authorization(Path(__file__).resolve().parents[1],
                                         value)
    if (contract["candidate"].get("pair_id")
            not in value.get("fixed_pair_order", ())
            or contract["candidate"].get("arm")
            not in (value.get("primary_arm"), value.get("control_arm"))
            or Path(str(contract.get("preregister_path", ""))).resolve() != path
            or contract.get("preregister_sha256") != sha256_file(path)):
        raise RuntimeError("candidate set is not bound to V14 preregistration")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-set", required=True, type=Path)
    parser.add_argument("--candidate-index", required=True, type=int)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--arm", required=True,
                        choices=("sgf_selected_union", "fullscan"))
    parser.add_argument("--prepared-input", required=True, type=Path)
    parser.add_argument("--v13-preregister", required=True, type=Path)
    parser.add_argument("--v14-preregister", required=True, type=Path)
    parser.add_argument("--preflight-manifest", required=True, type=Path)
    parser.add_argument("--pointdsc-root", required=True, type=Path)
    parser.add_argument("--pointdsc-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    contract = load_candidate_contract(args.candidate_set,
                                       args.candidate_index)
    v14_preregister = verify_v14_authorization(args.v14_preregister, contract)
    candidate = contract["candidate"]
    if candidate.get("pair_id") != args.pair_id or candidate.get("arm") != args.arm:
        raise RuntimeError("candidate CLI identity mismatch")
    preregister = _load_json(args.v13_preregister)
    preflight = _load_json(args.preflight_manifest)
    _verify_fixed_identity(preregister, preflight, args.pair_id)
    closure = _verify_preflight_closure(
        preregister_path=args.v13_preregister,
        preflight_path=args.preflight_manifest,
        prepared_path=args.prepared_input, pair_id=args.pair_id,
        preregister=preregister, preflight=preflight)
    safety_sha = _verify_safety_authority(repo)
    formal_sources = formal_source_sha256(repo)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forward = Path(candidate["forward_candidate_cache_path"])
    reverse = Path(candidate["reverse_candidate_cache_path"])
    raw = run_matrix(forward, reverse, output / "raw", args.pointdsc_root,
                     args.pointdsc_checkpoint, device="cpu", known_bad=False)
    atomic_json(output / "raw_summary.json", raw)
    strict = strict_pair_gate(
        pair_id=args.pair_id, arm=args.arm,
        prepared_path=args.prepared_input,
        forward_cache_path=forward, reverse_cache_path=reverse,
        dual_summary=raw, preregistration=preregister,
        icp_fn=segment_icp_with_trace, rule_features_fn=rule_b_features)
    strict.update({
        "v13_strict_schema": strict["schema"],
        "v14_candidate_evidence_schema": "v14-candidate-strict-evidence-v1",
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_index": args.candidate_index,
        "pair_id": args.pair_id,
        "arm": args.arm,
        "candidate_set_path": contract["candidate_set_path"],
        "candidate_set_sha256": contract["candidate_set_sha256"],
        "candidate_receipt_sha256": contract["candidate_receipt_sha256"],
        "candidate_receipt_path": {
            direction: candidate[f"{direction}_candidate_receipt_path"]
            for direction in ("forward", "reverse")},
        "cache_sha256": contract["cache_sha256"],
        "candidate_cache_path": {
            direction: candidate[f"{direction}_candidate_cache_path"]
            for direction in ("forward", "reverse")},
        "preflight_closure": closure,
        "safety_authority_sha256": safety_sha,
        "v14_runner_sha256": sha256_file(Path(__file__)),
        "v14_preregister_sha256": sha256_file(args.v14_preregister),
        "v14_preregister_path": str(args.v14_preregister.resolve()),
        "v14_selection_rule": v14_preregister["selection_rule"],
        "formal_source_sha256": formal_sources,
        "gt_consumed": False, "fallback_used": False,
    })
    atomic_json(output / "summary.json", strict)
    print(json.dumps(strict, indent=2, sort_keys=True))
    return 0 if strict["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
