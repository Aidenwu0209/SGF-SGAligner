#!/usr/bin/env python3
"""Create a no-pose DROID-W image window from an audited SGF manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pose_pipeline.contracts import (
    SequenceManifest, load_manifest, sha256_file, stable_json_sha256,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument(
        "--reuse-audited-output", action="store_true",
        help="reuse an existing window only after its input audit matches",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    frames = manifest.frames[args.start:args.start + args.count]
    if len(frames) != args.count or args.count < 2:
        raise ValueError("DROID-W manifest window is invalid or incomplete")
    reuse = args.output_dir.exists() and args.reuse_audited_output
    if args.output_dir.exists() and not reuse:
        raise FileExistsError(f"create-only output exists: {args.output_dir}")
    if args.output_manifest.exists():
        raise FileExistsError(
            f"create-only manifest exists: {args.output_manifest}",
        )
    if not reuse:
        args.output_dir.mkdir(parents=True)
    rows = []
    for ordinal, frame in enumerate(frames):
        suffix = frame.color_path.suffix.lower()
        if suffix not in {".jpg", ".jpeg"}:
            raise ValueError("RGB_NoPose shadow adapter currently requires JPEG inputs")
        link = args.output_dir / f"frame{ordinal:06d}.jpg"
        if reuse:
            if not link.is_symlink() or link.resolve() != frame.color_path.resolve():
                raise ValueError(f"existing DROID-W input link mismatch: {link}")
        else:
            os.symlink(frame.color_path.resolve(), link)
        rows.append({
            "ordinal": ordinal,
            "frame_id": frame.frame_id,
            "timestamp_us": frame.timestamp_us,
            "link": str(link),
            "source": str(frame.color_path),
            "source_sha256": sha256_file(frame.color_path),
        })
    unsigned = {
        "schema": "droid_w_manifest_window.v1",
        "sequence_id": manifest.sequence_id,
        "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
        "start_ordinal": args.start,
        "count": args.count,
        "rows": rows,
        "pose_files_exposed": False,
        "gt_consumed": False,
    }
    audit = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
    audit_path = args.output_dir / "input_audit.json"
    if reuse:
        if json.loads(audit_path.read_text()) != audit:
            raise ValueError("existing DROID-W input audit mismatch")
    else:
        with audit_path.open("x", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
    write_manifest(args.output_manifest, SequenceManifest(
        dataset=manifest.dataset,
        sequence_id=f"{manifest.sequence_id}:frames-{args.start}-{args.start + args.count - 1}",
        root=manifest.root,
        depth_scale=manifest.depth_scale,
        frames=tuple(frames),
        source=f"{manifest.source}; DROID-W shadow window",
    ))


if __name__ == "__main__":
    main()
