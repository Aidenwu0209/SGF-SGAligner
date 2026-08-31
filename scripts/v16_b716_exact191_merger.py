#!/usr/bin/env python3
"""Seal an authorized exact72 GeoT batch into the frozen b716 exact191 plan.

The merger imports no model runtime, performs no GPU work and exposes no key,
pair, selector, score, outcome, GT or official92 option.  Every input SHA is an
explicit command-line requirement so a caller cannot silently consume a
different execution or candidate plan.
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

from safety.v16_b716_candidate_plan import sha256_file  # noqa: E402
from safety.v16_b716_exact191_merger import merge_exact191  # noqa: E402


def _sha(value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise argparse.ArgumentTypeError("expected lowercase SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-sha256", type=_sha, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--preflight-sha256", type=_sha, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--preregister-sha256", type=_sha, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", type=_sha, required=True)
    parser.add_argument("--batch-result", type=Path, required=True)
    parser.add_argument("--batch-result-sha256", type=_sha, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_exact191(
        candidate_path=args.candidate_manifest,
        candidate_sha256=args.candidate_sha256,
        preflight_path=args.preflight_manifest,
        preflight_sha256=args.preflight_sha256,
        preregister_path=args.preregister,
        preregister_sha256=args.preregister_sha256,
        authorization_path=args.authorization,
        authorization_sha256=args.authorization_sha256,
        batch_path=args.batch_result,
        batch_sha256=args.batch_result_sha256,
        output_root=args.output_root,
    )
    path = args.output_root / "exact191_manifest.json"
    print(json.dumps({
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "candidate_count": manifest["candidate_count"],
        "existing_count": manifest["existing_count"],
        "new_authorized_count": manifest["new_authorized_count"],
        "hypothesis_count": manifest["hypothesis_count"],
        "consumer_scope": manifest["consumer_scope"],
        "official92_allowed": manifest["official92_allowed"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
