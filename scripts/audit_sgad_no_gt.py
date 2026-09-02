#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_pipeline.contracts import stable_json_sha256
from pose_pipeline.sgad_shadow import audit_sgad_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SGAD-SLAM for inference-time GT pose reads")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unsigned = audit_sgad_source(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")
    if not unsigned["passes_no_gt_input_gate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
