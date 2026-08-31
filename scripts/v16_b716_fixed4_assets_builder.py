#!/usr/bin/env python3
"""Create the fixed4 stage-asset and production-runtime manifest bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from safety.v16_b716_fixed4_assets_builder import (  # noqa: E402
    COLORPCR_COMMIT, COLORPCR_EXTENSION_SHA256, COLORPCR_WEIGHT_SHA256,
    POINTDSC_CHECKPOINT_SHA256, POINTDSC_COMMIT,
    ProductionAssetsBuilderError, build_documents, materialize_documents,
)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--colorpcr-repo", type=Path,
        default=Path("/home/aidenwu/Documents/colorpcr-clean-d579a80-audit"))
    parser.add_argument("--colorpcr-weights", type=Path,
        default=Path("/home/aidenwu/Documents/jojo_pipeline_eval_20260830/third_party/ColorPCR/weights/weights.pth.tar"))
    parser.add_argument("--colorpcr-extension", type=Path,
        default=Path("/home/aidenwu/Documents/colorpcr-clean-d579a80-audit/geotransformer/ext.cpython-310-x86_64-linux-gnu.so"))
    parser.add_argument("--pointdsc-root", type=Path,
        default=Path("/home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC"))
    parser.add_argument("--pointdsc-checkpoint", type=Path,
        default=Path("/home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC/snapshot/PointDSC_3DMatch_release/models/model_best.pkl"))
    parser.add_argument("--sgaligner-python", type=Path,
        default=Path("/home/aidenwu/miniconda3/envs/sgaligner/bin/python"))
    parser.add_argument("--jojo-python", type=Path,
        default=Path("/home/aidenwu/miniconda3/envs/jojo2026/bin/python"))
    parser.add_argument("--preflight-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--color-device", choices=("cuda:0", "cpu"),
                        default="cuda:0")
    args = parser.parse_args()
    documents = build_documents(repo=args.repo.resolve(),
        expected_repo_commit=args.expected_repo_commit,
        colorpcr_repo=args.colorpcr_repo.resolve(),
        colorpcr_weights=args.colorpcr_weights.resolve(),
        colorpcr_extension=args.colorpcr_extension.resolve(),
        pointdsc_root=args.pointdsc_root.resolve(),
        pointdsc_checkpoint=args.pointdsc_checkpoint.resolve(),
        sgaligner_python=args.sgaligner_python,
        jojo_python=args.jojo_python,
        preflight_manifest=args.preflight_manifest.resolve(),
        color_device=args.color_device)
    result = materialize_documents(args.output_dir.resolve(), documents)
    result["verified_pins"] = {
        "colorpcr_commit": COLORPCR_COMMIT,
        "colorpcr_extension_sha256": COLORPCR_EXTENSION_SHA256,
        "colorpcr_weight_sha256": COLORPCR_WEIGHT_SHA256,
        "pointdsc_commit": POINTDSC_COMMIT,
        "pointdsc_checkpoint_sha256": POINTDSC_CHECKPOINT_SHA256,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProductionAssetsBuilderError, OSError, ValueError, json.JSONDecodeError):
        raise SystemExit(70)
