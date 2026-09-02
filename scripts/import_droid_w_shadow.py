#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pose_pipeline.droid_w_shadow import import_droid_w_shadow


def main() -> None:
    parser = argparse.ArgumentParser(description="Import raw no-GT DROID-W shadow trajectory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--maximum-step-translation-m", type=float, default=1.5)
    args = parser.parse_args()
    report = import_droid_w_shadow(
        manifest_path=args.manifest, trajectory_path=args.trajectory,
        provenance_path=args.provenance, output_path=args.output,
        audit_path=args.audit,
        maximum_step_translation_m=args.maximum_step_translation_m,
    )
    print(f"imported {report['frame_count']} DROID-W poses")


if __name__ == "__main__":
    main()
