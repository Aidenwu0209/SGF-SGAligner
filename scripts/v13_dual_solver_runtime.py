#!/usr/bin/env python3
"""CLI for the V13 frozen-correspondence dual-solver runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v13_dual_solver_runtime import run_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-cache", required=True, type=Path,
                        help="independently generated forward src->ref cache")
    parser.add_argument("--reverse-cache", required=True, type=Path,
                        help="independently generated reverse ref->src cache")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pointdsc-root", required=True, type=Path)
    parser.add_argument("--pointdsc-checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--known-bad", action="store_true",
                        help="force an explicit known-bad veto after solver consensus")
    args = parser.parse_args()
    summary = run_matrix(args.forward_cache, args.reverse_cache,
                         args.output_dir, args.pointdsc_root,
                         args.pointdsc_checkpoint, device=args.device,
                         known_bad=args.known_bad)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
