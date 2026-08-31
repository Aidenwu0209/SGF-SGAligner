"""V3-Seal-Fix manifest + environment + git_state (new dir only)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    import torch

    branch = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    log = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--oneline", "-3"],
        capture_output=True, text=True).stdout
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--short"],
        capture_output=True, text=True).stdout
    (OUT / "git_state.txt").write_text(
        log + f"branch: {branch}\nHEAD: {head}\n--- status ---\n{status}")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    (OUT / "environment.json").write_text(
        json.dumps({
            "branch": branch, "head_at_evidence": head,
            "conda_env": "sgaligner",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "gpu": gpu, "services_touched": "NONE",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")

    manifest = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            manifest[str(path.relative_to(ROOT))] = {
                "sha256": sha256(path), "size": path.stat().st_size}
    (OUT / "artifact_manifest.json").write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "files": manifest}, indent=2, sort_keys=True) + "\n")
    print("manifest files:", len(manifest), "head:", head[:12])


if __name__ == "__main__":
    main()
