"""V3 checkpoint provenance: re-verify the official checkpoint identity
for the V3 evidence package (byte-compare vs the V2T-Fix3-Seal fresh
download kept in checkpoints/redownload_verify/, SHA vs the recorded
sgaligner_checkpoint_sha256.txt)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v3_pct_parity_baseline_20260827"
CURRENT = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
SEAL_FRESH = (
    ROOT / "checkpoints/redownload_verify/"
    "sgaligner_pct_gat_rel_attr.pth.tar"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = subprocess.run(
        ["cmp", str(CURRENT), str(SEAL_FRESH)], capture_output=True)
    state = torch_meta()
    payload = {
        "phase": "V3-A/B",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "current_checkpoint": {
            "path": str(CURRENT),
            "size_bytes": CURRENT.stat().st_size,
            "sha256": sha256(CURRENT),
        },
        "vs_seal_fresh_download": {
            "path": str(SEAL_FRESH),
            "cmp_byte_identical": cmp.returncode == 0,
            "sha256": sha256(SEAL_FRESH),
        },
        "recorded_sha256_file": (
            ROOT / "checkpoints/sgaligner_checkpoint_sha256.txt"
        ).read_text().splitlines()[-1].split()[0],
        "torch_metadata": state,
    }
    payload["all_three_agree"] = (
        payload["current_checkpoint"]["sha256"]
        == payload["vs_seal_fresh_download"]["sha256"]
        == payload["recorded_sha256_file"])
    (OUT / "checkpoint_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print("byte-identical:",
          payload["vs_seal_fresh_download"]["cmp_byte_identical"],
          "all agree:", payload["all_three_agree"],
          "epoch:", state.get("epoch"))


def torch_meta() -> dict:
    import torch

    state = torch.load(CURRENT, map_location="cpu", weights_only=False)
    return {
        "top_level_keys": sorted(state.keys()),
        "epoch": state.get("epoch"),
        "iteration": state.get("iteration"),
        "model_num_tensors": len(state["model"]),
    }


if __name__ == "__main__":
    main()
