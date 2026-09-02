#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pose_pipeline.sgad_shadow import import_sgad_shadow


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a no-GT SGAD-SLAM trajectory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    report = import_sgad_shadow(
        manifest_path=args.manifest, matrices_path=args.matrices,
        provenance_path=args.provenance, output_path=args.output,
        audit_path=args.audit,
    )
    print(f"imported {report['frame_count']} SGAD-SLAM poses")


if __name__ == "__main__":
    main()
