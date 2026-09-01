#!/usr/bin/env python3
"""Create a transform-free 3RScan split manifest for inference processes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-groups", type=int, default=8)
    parser.add_argument("--sentinel")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    development_groups = sorted(
        (group for group in metadata if group.get("type") == "train"),
        key=lambda group: hashlib.sha256(group["reference"].encode()).hexdigest(),
    )[:args.development_groups]
    selected_groups = []
    for split, groups in (
        ("development", development_groups),
        ("validation", [group for group in metadata if group.get("type") == "validation"]),
    ):
        for group in groups:
            selected_groups.append({
                "split": split,
                "reference": group["reference"],
                "scans": [scan["reference"] for scan in group.get("scans", [])],
            })
    sequences, seen = [], set()
    for group in selected_groups:
        for sequence_id in [group["reference"], *group["scans"]]:
            if sequence_id in seen:
                continue
            seen.add(sequence_id)
            sequence = args.data_root / sequence_id / "sequence"
            sequences.append({
                "sequence_id": sequence_id,
                "split": group["split"],
                "present": (sequence / "_info.txt").is_file(),
            })
    if args.sentinel and args.sentinel not in seen:
        sequence = args.data_root / args.sentinel / "sequence"
        sequences.append({
            "sequence_id": args.sentinel,
            "split": "train_failure_sentinel",
            "present": (sequence / "_info.txt").is_file(),
        })
    unsigned = {
        "schema": "scan3r_pose_selection.v1",
        "source_metadata_sha256": sha256_file(args.metadata),
        "source_metadata_role": "selection_builder_only_not_inference",
        "development_group_references": [
            group["reference"] for group in development_groups
        ],
        "groups": selected_groups,
        "sequences": sequences,
        "contains_transforms": False,
        "gt_at_inference": False,
    }
    payload = {
        **unsigned,
        "payload_sha256": hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "groups": len(selected_groups),
        "sequences": len(sequences),
        "present": sum(bool(row["present"]) for row in sequences),
        "payload_sha256": payload["payload_sha256"],
    }))


if __name__ == "__main__":
    main()
