#!/usr/bin/env python3
"""Verify and materialize the CPU-only exact72 execution-lineage seal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety.v13_dual_solver_runtime import sha256_file
from safety.v16_b716_exact72_lineage_seal import (
    build_lineage_seal, materialize_lineage_seal,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact191-manifest", type=Path, required=True)
    parser.add_argument("--exact191-manifest-sha256", required=True)
    parser.add_argument("--delivery-seal", type=Path, required=True)
    parser.add_argument("--delivery-seal-sha256", required=True)
    parser.add_argument("--delivery-signature", type=Path, required=True)
    parser.add_argument("--delivery-signature-sha256", required=True)
    parser.add_argument("--exact72-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = build_lineage_seal(
        exact191_manifest_path=args.exact191_manifest,
        exact191_manifest_sha256=args.exact191_manifest_sha256,
        exact191_delivery_seal_path=args.delivery_seal,
        exact191_delivery_seal_sha256=args.delivery_seal_sha256,
        exact191_delivery_signature_path=args.delivery_signature,
        exact191_delivery_signature_sha256=args.delivery_signature_sha256,
        exact72_root=args.exact72_root)
    manifest_path, state = materialize_lineage_seal(args.output_root, value)
    print(json.dumps({
        "status": "PASS", "state": state,
        "output": str(manifest_path),
        "output_sha256": sha256_file(manifest_path),
        "task_count": value["task_count"],
        "ok_count": value["ok_count"],
        "typed_failure_count": value["typed_failure_count"],
        "typed_failure_correspondence_absence_verified":
            value["typed_failure_correspondence_absence_verified"],
        "model_imported": value["model_imported"],
        "gpu_used": value["gpu_used"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
