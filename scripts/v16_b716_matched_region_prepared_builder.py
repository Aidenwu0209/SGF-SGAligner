#!/usr/bin/env python3
"""Build exactly 34 CPU-only b716 matched-region prepared inputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src", ROOT / "scripts",
              ROOT / "src/inference/sgf_official"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from safety.v16_b716_prepared_builder import (  # noqa: E402
    EXPECTED_HYPOTHESIS_COUNTS,
    OFFICIAL_RELEASE_SHA256,
    V16ContractError,
    build_fixed4_prepared_inputs,
    sha256_file,
)
from safety.v16_matched_region_colorpcr import (  # noqa: E402
    load_raw_inseg,
    resolve_unique_inseg_path,
)


DEFAULT_CANDIDATE_MANIFEST = (
    ROOT / "outputs/v16_b716_candidate_plan_fixed4_20260830/fixed4_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST_SHA256 = (
    "774d4b49624e495412fcb72d1c79716d7b1b2b2840de72ce303ee8c70fd4ca68"
)
RAW_ROOTS = (
    Path("/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
         "delivery_stage1_20260823/training_dataset/cache"),
    Path("/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
         "official_sgaligner_migration_20260825_235139/supplementary_scan_cache"),
)


def _validate_preregister(path: Path) -> dict:
    value = json.loads(path.read_text())
    if (value.get("schema") != "v16-b716-prepared-builder-preregister-v2"
            or value.get("frozen") is not True
            or value.get("cpu_builder_allowed") is not True
            or value.get("worker_execution_allowed") is not False
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or value.get("legacy_B_ep20_or_89ed_allowed") is not False
            or value.get("expected_hypothesis_distribution")
            != list(EXPECTED_HYPOTHESIS_COUNTS)
            or value.get("expected_existing_typed_failure_count") != 16
            or value.get("expected_new_typed_failure_count") != 12
            or value.get("expected_typed_failure_total_count") != 28
            or value.get("expected_all_typed_failure_hypothesis_count") != 10
            or value.get("typed_failure_hypothesis_voting_allowed") is not False
            or value.get("gt_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("geot_result_filtering_allowed") is not False):
        raise V16ContractError("prepared builder preregistration mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path,
                        default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--candidate-manifest-sha256",
                        default=DEFAULT_CANDIDATE_MANIFEST_SHA256)
    parser.add_argument("--exact191-manifest", type=Path, required=True)
    parser.add_argument("--exact191-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-synthetic-test-fixture", action="store_true",
                        help="TEST ONLY: never authorizes workers")
    args = parser.parse_args()

    prereg_path = ROOT / "manifests/v16_b716_prepared_builder_preregister.json"
    _validate_preregister(prereg_path)
    sources = {
        "manifests/v16_b716_prepared_builder_preregister.json":
            sha256_file(prereg_path),
        "scripts/v16_b716_matched_region_prepared_builder.py":
            sha256_file(Path(__file__)),
        "scripts/canonical_inputs.py":
            sha256_file(ROOT / "scripts/canonical_inputs.py"),
        "src/safety/v16_b716_prepared_builder.py":
            sha256_file(ROOT / "src/safety/v16_b716_prepared_builder.py"),
        "src/safety/v16_matched_region_colorpcr.py":
            sha256_file(ROOT / "src/safety/v16_matched_region_colorpcr.py"),
        "src/safety/v13_colorpcr_pointdsc_shadow.py":
            sha256_file(ROOT / "src/safety/v13_colorpcr_pointdsc_shadow.py"),
    }

    result = build_fixed4_prepared_inputs(
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
        exact191_manifest_path=args.exact191_manifest,
        exact191_manifest_sha256=args.exact191_manifest_sha256,
        output_root=args.output_root,
        raw_roots=RAW_ROOTS,
        canonical_builder=lambda pair_id: build_canonical_pair(
            pair_id, with_labels=False),
        raw_loader=lambda path, scan_id, side: load_raw_inseg(
            path, scan_id=scan_id, side=side),
        raw_resolver=resolve_unique_inseg_path,
        source_hashes=sources,
        allow_test_fixture=args.allow_synthetic_test_fixture,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
