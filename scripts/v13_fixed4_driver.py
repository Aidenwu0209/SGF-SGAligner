#!/usr/bin/env python3
"""Formal one-pair V13 driver: independent ColorPCR directions -> dual solvers.

This is the only authorized aggregate path.  It never calls the legacy
``independent_solver_q4`` helper and never accepts a manual known-bad flag.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from v13_corr_cache_converter import convert


JOJO_PYTHON = Path("/home/aidenwu/miniconda3/envs/jojo2026/bin/python")
SGALIGNER_PYTHON = Path("/home/aidenwu/miniconda3/envs/sgaligner/bin/python")
COLORPCR_REPO = Path("/home/aidenwu/Documents/colorpcr-clean-d579a80-audit")
COLORPCR_COMMIT = "d579a80d71c3d6ae37ee58ba5a3943fe81e8427d"
COLORPCR_WEIGHTS = Path("/home/aidenwu/Documents/jojo_pipeline_eval_20260830/third_party/ColorPCR/weights/weights.pth.tar")
COLORPCR_WEIGHT_SHA = "b4900863c86629c24386189094691f159c1ff437b5623510a11c9468bc8cb814"
COLORPCR_PYTHON_TREE_SHA = "26f732740d70433324f7e3a2368b9f7bf1670fb3e7a95945f80d7af6ae50958d"
EMPTY_DIFF_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
COLORPCR_EXTENSION = COLORPCR_REPO / "geotransformer/ext.cpython-310-x86_64-linux-gnu.so"
COLORPCR_EXTENSION_SHA = "33160b284931483c570b32fd73513ceb5603cce3d30d4e0903937ce30c8b594f"
POINTDSC_ROOT = Path("/home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC")
POINTDSC_CHECKPOINT = POINTDSC_ROOT / "snapshot/PointDSC_3DMatch_release/models/model_best.pkl"
NEIGHBOR_LIMITS = "38,36,36,38"


def run_pair(repo: Path, prepared: Path, pair_id: str, arm: str,
             output: Path, device: str, *, preregister: Path,
             preflight_manifest: Path) -> dict:
    repo = repo.resolve(); prepared = prepared.resolve(); output = output.resolve()
    preregister = preregister.resolve()
    preflight_manifest = preflight_manifest.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sentinel_script = repo / "scripts/v13_colorpcr_sentinel_subprocess.py"
    worker_script = repo / "scripts/v13_colorpcr_official_worker.py"
    solver_script = repo / "scripts/v13_dual_solver_cli.py"
    solver_caches = {}
    solver_receipts = {}
    for direction in ("forward", "reverse"):
        corr_cache = output / "colorpcr" / f"{direction}.sentinel_invariant.npz"
        command = [
            sys.executable, str(sentinel_script), "--python", str(JOJO_PYTHON),
            "--worker", str(worker_script), "--repo", str(COLORPCR_REPO),
            "--expected-commit", COLORPCR_COMMIT, "--weights", str(COLORPCR_WEIGHTS),
            "--expected-weight-sha256", COLORPCR_WEIGHT_SHA,
            "--input", str(prepared),
            "--expected-python-tree-sha256", COLORPCR_PYTHON_TREE_SHA,
            "--expected-tracked-diff-sha256", EMPTY_DIFF_SHA,
            "--extension", str(COLORPCR_EXTENSION),
            "--expected-extension-sha256", COLORPCR_EXTENSION_SHA,
            "--arm", arm, "--direction", direction,
            "--neighbor-limits", NEIGHBOR_LIMITS,
            "--sampling", "voxel10", "--output", str(corr_cache),
            "--evidence-dir", str(output / "colorpcr" / "sentinel_artifacts"),
            "--device", device,
        ]
        subprocess.run(command, check=True)
        solver_cache = output / "solver_cache" / f"{direction}.three_key.npz"
        receipt_path = solver_cache.with_suffix(".receipt.json")
        convert(corr_cache, prepared, solver_cache, receipt_path,
                pair_id=pair_id, arm=arm, direction=direction)
        solver_caches[direction] = solver_cache
        solver_receipts[direction] = receipt_path
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "src"), str(repo / "scripts")])
    command = [
        str(SGALIGNER_PYTHON), str(solver_script),
        "--forward-cache", str(solver_caches["forward"]),
        "--reverse-cache", str(solver_caches["reverse"]),
        "--output-dir", str(output / "dual_solver"),
        "--pointdsc-root", str(POINTDSC_ROOT),
        "--pointdsc-checkpoint", str(POINTDSC_CHECKPOINT),
        "--prepared-input", str(prepared), "--arm", arm,
        "--pair-id", pair_id,
        "--preregister", str(preregister),
        "--preflight-manifest", str(preflight_manifest),
        "--forward-receipt", str(solver_receipts["forward"]),
        "--reverse-receipt", str(solver_receipts["reverse"]),
        "--driver-source", str(Path(__file__).resolve()),
        # Both rigid solvers are frozen to CPU; only isolated ColorPCR uses GPU.
        "--device", "cpu",
    ]
    completed = subprocess.run(command, check=False, env=env)
    summary_path = output / "dual_solver" / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"dual solver produced no summary (rc={completed.returncode})")
    summary = json.loads(summary_path.read_text())
    if completed.returncode not in (0, 2):
        raise RuntimeError(f"dual solver runtime failed rc={completed.returncode}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--arm", choices=("sgf_selected_union", "fullscan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    args = parser.parse_args()
    summary = run_pair(args.repo, args.prepared, args.pair_id, args.arm,
                       args.output, args.device,
                       preregister=args.preregister,
                       preflight_manifest=args.preflight_manifest)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
