"""V2T-Fix3-Seal stage 5: resume reproducibility protocol (in-repo).

Migrated from the Fix3 /tmp/resume_protocol.py into the repository so
commands.sh only ever references repo scripts. Adds a CPU synthetic
exact-equivalence leg (fast, hermetic) alongside the retained GPU
real-subset byte-exact equivalence leg and the fail-closed checks.

Legs:
  cpu  : --device cpu --synthetic 6; interrupted@2 / resume vs
         continuous, 4-epoch horizon. Exact equality expected.
  gpu  : real training subset, corrected protocol (total horizon
         always 4). Byte-exact equality expected (Fix3 result).
  failclosed: total_epochs mismatch and dataset-fingerprint mismatch
         must BOTH be refused (non-zero exit + explicit message).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal/resume"
OUT.mkdir(parents=True, exist_ok=True)

ENV = dict(os.environ)
ENV["PYTHONPATH"] = "src:."
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(args, extra_env=None):
    e = dict(ENV)
    if extra_env:
        e.update(extra_env)
    return subprocess.run(
        [sys.executable, "scripts/v2f_train.py", *args],
        capture_output=True, text=True, cwd=ROOT, env=e,
    )


def compare(sa, sb):
    import torch

    rows = []
    for k in sa:
        if k not in sb:
            continue
        ta, tb = sa[k], sb[k]
        if ta.shape != tb.shape:
            rows.append({"key": k, "identical": False, "max_diff": None})
            continue
        if ta.dtype in (torch.int64, torch.long, torch.bool):
            eq = bool(torch.equal(ta, tb))
            rows.append({"key": k, "identical": eq,
                         "max_diff": 0.0 if eq else None})
        else:
            d = float((ta.float() - tb.float()).abs().max())
            rows.append({"key": k, "identical": d == 0.0, "max_diff": d})
    changed = [r for r in rows if not r["identical"]]
    return rows, changed


def report(tag, a_path, b_path):
    import torch

    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    model_rows, model_changed = compare(a["model"], b["model"])
    opt_rows, opt_changed = [], []
    if a.get("optimizer") and b.get("optimizer"):
        for state_key in a["optimizer"].get("state", {}):
            as_ = a["optimizer"]["state"][state_key]
            bs = b["optimizer"]["state"][state_key]
            r, c = compare(as_, bs)
            opt_rows.extend(r)
            opt_changed.extend(c)
    return {
        "tag": tag,
        "final_epoch_split": a["epoch"],
        "final_epoch_cont": b["epoch"],
        "total_epochs": a.get("total_epochs"),
        "model": {
            "compared": len(model_rows),
            "changed": len(model_changed),
            "max_diff": max(
                (r["max_diff"] or 0.0) for r in model_rows
            ) if model_rows else 0.0,
            "changed_keys": [r["key"] for r in model_changed][:10],
        },
        "optimizer": {
            "compared": len(opt_rows),
            "changed": len(opt_changed),
            "max_diff": max(
                (r["max_diff"] or 0.0) for r in opt_rows
            ) if opt_rows else 0.0,
        },
        "scheduler_identical": a.get("scheduler") == b.get("scheduler"),
        "history_identical": a.get("history") == b.get("history"),
        "rng_states_identical": bool(
            torch.equal(a["torch_rng"], b["torch_rng"])),
        "numpy_rng_identical": bool(
            np.array_equal(np.asarray(a["numpy_rng"][1]),
                           np.asarray(b["numpy_rng"][1]))
            and a["numpy_rng"][2] == b["numpy_rng"][2]),
        "next_epoch": a.get("next_epoch"),
    }


def lr_trajectory(d):
    return {
        f: (d / f).read_text().strip()
        for f in sorted(x.name for x in d.glob("lr_epoch_*"))
    }


def equivalence_leg(tag, split_dir, cont_dir, train_args):
    for d in (split_dir, cont_dir):
        if d.exists():
            shutil.rmtree(d)
    r1 = train([*train_args, "--out", str(split_dir),
                "--epochs", "4", "--stop-after-epoch", "2"])
    r2 = train([*train_args, "--out", str(split_dir),
                "--epochs", "4",
                "--resume", str(split_dir / "epoch_00002.pt")])
    r3 = train([*train_args, "--out", str(cont_dir), "--epochs", "4"])
    result = report(tag, split_dir / "last.pt", cont_dir / "last.pt")
    result["leg_return_codes"] = {
        "interrupted": r1.returncode,
        "resume": r2.returncode,
        "continuous": r3.returncode,
    }
    result["resume_stderr_tail"] = (
        r2.stderr.strip().splitlines()[-1][:200]
        if r2.stderr.strip() else "")
    result["lr_split"] = lr_trajectory(split_dir)
    result["lr_cont"] = lr_trajectory(cont_dir)
    result["lr_identical"] = result["lr_split"] == result["lr_cont"]
    result["exact"] = (
        result["model"]["changed"] == 0
        and result["model"]["max_diff"] == 0.0
        and result["optimizer"]["changed"] == 0
        and result["scheduler_identical"]
        and result["history_identical"]
        and result["lr_identical"]
    )
    return result


def cpu_leg():
    args = ["--strategy", "B", "--device", "cpu", "--synthetic", "6"]
    return equivalence_leg(
        "cpu_synthetic_exact",
        OUT / "cpu_split", OUT / "cpu_cont", args,
    )


def gpu_leg():
    args = ["--strategy", "B"]
    return equivalence_leg(
        "gpu_real_subset_byte_exact",
        OUT / "gpu_split", OUT / "gpu_cont", args,
    )


def failclosed_leg():
    out = {}
    # (a) total_epochs horizon mismatch must be refused
    d = OUT / "fc_mismatch"
    if d.exists():
        shutil.rmtree(d)
    train(["--out", str(d), "--strategy", "B", "--epochs", "2",
           "--stop-after-epoch", "1"])
    fc = train(["--out", str(d), "--strategy", "B", "--epochs", "4",
                "--resume", str(d / "epoch_00001.pt")])
    out["total_epochs_mismatch"] = {
        "rc": fc.returncode,
        "refused": fc.returncode != 0
        and "total_epochs" in fc.stderr,
        "stderr_tail": fc.stderr.strip().splitlines()[-1][:200]
        if fc.stderr.strip() else "",
    }
    # (b) dataset fingerprint mismatch must be refused (cpu synthetic 6
    # vs cpu synthetic 5 → different fingerprint, same horizon)
    d6 = OUT / "fc_fp6"
    d5 = OUT / "fc_fp5"
    for dd in (d6, d5):
        if dd.exists():
            shutil.rmtree(dd)
    train(["--out", str(d6), "--strategy", "B", "--device", "cpu",
           "--synthetic", "6", "--epochs", "4", "--stop-after-epoch", "2"])
    fp = train(["--out", str(d6), "--strategy", "B", "--device", "cpu",
                "--synthetic", "5", "--epochs", "4",
                "--resume", str(d6 / "epoch_00002.pt")])
    out["dataset_fingerprint_mismatch"] = {
        "rc": fp.returncode,
        "refused": fp.returncode != 0
        and "fingerprint" in fp.stderr,
        "stderr_tail": fp.stderr.strip().splitlines()[-1][:200]
        if fp.stderr.strip() else "",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("cpu", "gpu", "failclosed", "all"),
                        default="all")
    args = parser.parse_args()

    if args.part in ("cpu", "all"):
        result = cpu_leg()
        (OUT / "resume_equivalence_cpu.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print("CPU:", json.dumps({
            "model_changed": result["model"]["changed"],
            "model_max_diff": result["model"]["max_diff"],
            "optimizer_changed": result["optimizer"]["changed"],
            "lr_identical": result["lr_identical"],
            "exact": result["exact"],
        }))

    if args.part in ("gpu", "all"):
        result = gpu_leg()
        (OUT / "resume_equivalence_gpu.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print("GPU:", json.dumps({
            "model_changed": result["model"]["changed"],
            "model_max_diff": result["model"]["max_diff"],
            "optimizer_changed": result["optimizer"]["changed"],
            "lr_identical": result["lr_identical"],
            "exact": result["exact"],
        }))

    if args.part in ("failclosed", "all"):
        result = failclosed_leg()
        (OUT / "resume_failclosed.json").write_text(
            json.dumps(result, indent=2) + "\n")
        print("FAILCLOSED:", json.dumps(result))


if __name__ == "__main__":
    main()
