"""Narrow subprocess wrapper for one manifest-bound calibration90 worker.

The V7 worker core is reused byte-for-byte.  The only adaptation is replacing
its selection-only cache-root constant inside this isolated process; the
controller has already bound every file SHA in the frozen calibration manifest.
No labels are imported here.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_pilot as pilot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--direction", choices=("forward", "reverse"),
                        required=True)
    parser.add_argument("--replicate", type=int, choices=range(5), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--worker-out", type=Path, required=True)
    args = parser.parse_args()
    args.cache_root = args.cache_root.resolve()
    args.worker_out = args.worker_out.resolve()
    # ``cache_path`` compares against this constant at call time.  The wrapper
    # is a fresh subprocess, so this cannot alter another split or process.
    pilot.DEFAULT_CACHE_ROOT = args.cache_root
    return pilot.run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
