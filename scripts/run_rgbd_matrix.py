"""Run the same raw RGB-D recipe over an explicit list of dataset manifests.

Input JSON is a list of {"key": "dataset/scene", "manifest": "/path/..."}.
Every scene is attempted once. Failures and partial artifacts are retained;
the final summary separates process completion from reconstruction quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import signal
import time

from pose_pipeline.contracts import load_manifest, sha256_file
from pose_pipeline.rgbd_mapping import run_rgbd_mapping


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--gpu-python", type=Path, required=True)
    parser.add_argument("--cpu-python", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--stage-timeout", type=float, default=7200)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("Matrix must contain at least one scene")
    keys = set()
    for row in matrix:
        key = row["key"]
        path = PurePosixPath(key)
        if (path.is_absolute() or len(path.parts) != 2 or ".." in path.parts
                or str(path) != key or key in keys
                or path.parts[0] not in {"scannet", "3rscan", "orbbec"}):
            raise ValueError(f"Invalid or duplicate scene key: {key}")
        keys.add(key)
        row["manifest"] = str(Path(row["manifest"]).resolve(strict=True))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "raw_rgbd_matrix.v1", "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_sha256": sha256_file(args.matrix), "scene_count": len(matrix),
        "completed": [], "current": None,
    }
    save(output / "input_matrix.json", matrix)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, row in enumerate(matrix):
            summary["current"] = row["key"]
            save(output / "status.json", summary)
            started = time.monotonic()
            result = {"key": row["key"], "manifest": row["manifest"]}
            try:
                manifest = load_manifest(Path(row["manifest"]))
                if row["key"] != manifest.dataset + "/" + manifest.sequence_id:
                    raise ValueError("Matrix dataset/scene differs from manifest")
                result["raw_frame_count"] = len(manifest.frames)
                mapping = run_rgbd_mapping(
                    manifest_path=Path(row["manifest"]), output_dir=output / "scenes" / row["key"],
                    provider_root=args.provider_root, gpu_python=args.gpu_python,
                    cpu_python=args.cpu_python, stage_timeout_s=args.stage_timeout,
                    device=args.device, threads=args.threads,
                )
                result.update(
                    status="completed", final_pose_count=mapping["final_pose_count"],
                    final_cloud=mapping["final_cloud"],
                    final_cloud_sha256=mapping["final_cloud_sha256"],
                )
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
            result["seconds"] = time.monotonic() - started
            summary["completed"].append(result)
            summary["current"] = None
            save(output / "status.json", summary)
            print(json.dumps({"index": index + 1, "total": len(matrix), **result}), flush=True)
        summary["status"] = "complete_with_failures" if any(
            r["status"] != "completed" for r in summary["completed"]
        ) else "completed"
        summary["quality_acceptance"] = "requires_separate_evaluation"
    except BaseException as error:
        summary.update(status="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "status.json", summary)
    save(output / "batch_complete.json", summary)


if __name__ == "__main__":
    main()
