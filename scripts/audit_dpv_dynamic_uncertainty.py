#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_pipeline.dpv_uncertainty import DynamicUncertaintyConfig, DynamicUncertaintyStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-static-confidence", type=float, default=0.20)
    parser.add_argument("--depth-prior-sigma-m", type=float, default=0.10)
    args = parser.parse_args()
    store = DynamicUncertaintyStore(
        manifest_path=args.manifest,
        artifact_path=args.artifact,
        config=DynamicUncertaintyConfig(
            minimum_static_confidence=args.minimum_static_confidence,
            depth_prior_sigma_m=args.depth_prior_sigma_m,
        ),
    )
    print(json.dumps(store.write_audit(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

