#!/usr/bin/env python3
"""CPU-only V14 candidate builder; it never invokes ColorPCR or a solver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v14_rigid_multihypothesis import (
    build_direction_candidates, seal_bidirectional_candidate_set,
)
from scripts.v14_formal_source_manifest import verify_reviewed_source_authorization


def load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def preregister(path: Path) -> dict:
    value = load_json(path)
    if (value.get("schema") != "v14-rigid-multihypothesis-preregister-v1"
            or value.get("allow_real_pilot") is not True
            or value.get("allow_gpu_pilot") is not False
            or value.get("gt_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("posthoc_allowed") is not False):
        raise RuntimeError("V14 real CPU pilot is not explicitly authorized")
    verify_reviewed_source_authorization(Path(__file__).resolve().parents[1],
                                         value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-direction")
    build.add_argument("--cache", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--pair-id", required=True)
    build.add_argument("--arm", required=True,
                       choices=("sgf_selected_union", "fullscan"))
    build.add_argument("--direction", required=True,
                       choices=("forward", "reverse"))
    build.add_argument("--preregister", required=True, type=Path)
    pair = subparsers.add_parser("pair-directions")
    pair.add_argument("--forward-manifest", required=True, type=Path)
    pair.add_argument("--reverse-manifest", required=True, type=Path)
    pair.add_argument("--output", required=True, type=Path)
    pair.add_argument("--preregister", required=True, type=Path)
    args = parser.parse_args()
    frozen = preregister(args.preregister)
    if args.command == "build-direction":
        if args.pair_id not in frozen.get("fixed_pair_order", []):
            raise RuntimeError("pair is outside frozen fixed4")
        value = build_direction_candidates(
            args.cache, args.output, pair_id=args.pair_id, arm=args.arm,
            direction=args.direction, preregister_path=args.preregister)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    value = seal_bidirectional_candidate_set(
        args.forward_manifest, args.reverse_manifest, args.output,
        args.preregister)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
