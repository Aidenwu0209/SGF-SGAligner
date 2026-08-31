#!/usr/bin/env python3
"""Create the current ready-v2 protocol preregistration, never authorization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.v16_b716_fixed4_execution_pilot import (  # noqa: E402
    build_active_execution_preregister_v2,
)
from safety.v16_b716_fixed4_subprocess_contract import (  # noqa: E402
    create_only_bytes_beneath,
)


DEFAULT_OUTPUT = (ROOT / "manifests" /
    "v16_b716_fixed4_active_execution_ready_v2_preregister.json")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError("ready-v2 preregistration output path is fixed")
    value = build_active_execution_preregister_v2(ROOT)
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    _row, state = create_only_bytes_beneath(
        ROOT, output, encoded, create_parents=False,
        resume_identical=True)
    print(json.dumps({
        "status": "READY_V2_PROTOCOL_PREREGISTERED_NOT_AUTHORIZED",
        "state": state,
        "path": str(output),
        "payload_sha256": value["payload_sha256"],
        "formal_execution_authorized": False,
        "private_key_used": False,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
