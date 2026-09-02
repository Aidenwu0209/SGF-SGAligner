#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_pipeline.gcvo_refinement import GCVORefinementConfig, apply_gcvo_refinement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--gcvo-result", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--max-disagreement-translation", type=float, default=0.08)
    parser.add_argument("--max-disagreement-rotation", type=float, default=8.0)
    parser.add_argument("--max-step-translation", type=float, default=0.20)
    parser.add_argument("--max-step-rotation", type=float, default=20.0)
    parser.add_argument("--minimum-accepted-fraction", type=float, default=0.80)
    args = parser.parse_args()
    result = apply_gcvo_refinement(
        manifest_path=args.manifest,
        baseline_trajectory_path=args.baseline_trajectory,
        gcvo_result_path=args.gcvo_result,
        output_trajectory_path=args.output_trajectory,
        output_audit_path=args.output_audit,
        config=GCVORefinementConfig(
            maximum_baseline_disagreement_translation_m=args.max_disagreement_translation,
            maximum_baseline_disagreement_rotation_deg=args.max_disagreement_rotation,
            maximum_step_translation_m=args.max_step_translation,
            maximum_step_rotation_deg=args.max_step_rotation,
            minimum_accepted_fraction=args.minimum_accepted_fraction,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

