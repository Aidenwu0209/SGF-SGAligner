#!/usr/bin/env python3
"""Bounded formal V13 fixed4 orchestrator.

Runs the exact frozen 4-pair x 2-arm matrix sequentially.  A pair may resume
only through a complete, hash-verified receipt; the only global verdict is the
audited :func:`safety.v13_fixed4_aggregate.aggregate_fixed4` result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from safety.v13_fixed4_aggregate import aggregate_fixed4
from v13_formal_source_manifest import (
    FormalSourceContractError,
    formal_source_sha256,
)


PAIR_SCHEMA = "v13-strict-pair-gate-v1"
PAIR_RECEIPT_SCHEMA = "v13-fixed4-pair-receipt-v1"
RUN_SCHEMA = "v13-fixed4-formal-run-v1"
ARMS = ("sgf_selected_union", "fullscan")


class Fixed4DriverError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".json",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise Fixed4DriverError(f"JSON object required: {path}")
    return value


def _frozen_pairs(preregister: dict, preflight: dict) -> list[str]:
    normals = [str(value) for value in preregister.get("normal_pair_ids", ())]
    known_bad = str(preregister.get("known_bad_pair_id", ""))
    frozen = normals + [known_bad]
    if len(normals) != 3 or len(set(frozen)) != 4 or not known_bad:
        raise Fixed4DriverError("pre-registration is not exact fixed4")
    if preflight.get("pair_ids") != frozen:
        raise Fixed4DriverError("preflight pair order differs from pre-registration")
    return frozen


def _prepared_by_pair(preflight: dict, pairs: list[str]) -> dict[str, tuple[Path, str]]:
    records = preflight.get("pairs", ())
    if not isinstance(records, list) or len(records) != 4:
        raise Fixed4DriverError("preflight must contain exact four pair records")
    result = {}
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        path = Path(record.get("prepared_npz_path", "")).resolve()
        expected = str(record.get("prepared_npz_sha256", ""))
        if pair_id not in pairs or pair_id in result or not path.is_file() \
                or len(expected) != 64 or sha256_file(path) != expected:
            raise Fixed4DriverError(f"prepared preflight closure failed: {pair_id}")
        result[pair_id] = (path, expected)
    if set(result) != set(pairs):
        raise Fixed4DriverError("prepared pair set mismatch")
    return result


def _formal_source_sha256(repo: Path) -> dict[str, str]:
    """Compatibility wrapper around the sole formal source manifest."""
    try:
        return formal_source_sha256(repo)
    except FormalSourceContractError as exc:
        raise Fixed4DriverError(str(exc)) from exc


def _pair_receipt_valid(receipt_path: Path, *, pair_id: str, arm: str,
                        summary_path: Path, prepared_path: Path,
                        expected: dict[str, Any]) -> tuple[bool, dict | None]:
    if not receipt_path.is_file():
        return False, None
    receipt = _load_json(receipt_path)
    exact = {
        "schema": PAIR_RECEIPT_SCHEMA, "pair_id": pair_id, "arm": arm,
        "summary_path": str(summary_path),
        "prepared_input_sha256": sha256_file(prepared_path),
        **expected,
    }
    if any(receipt.get(key) != value for key, value in exact.items()):
        raise Fixed4DriverError(f"stale pair receipt dependency: {pair_id}/{arm}")
    if not summary_path.is_file() \
            or sha256_file(summary_path) != receipt.get("summary_sha256"):
        raise Fixed4DriverError(f"pair summary hash mismatch: {pair_id}/{arm}")
    row = _load_json(summary_path)
    if row.get("schema") != PAIR_SCHEMA or row.get("pair_id") != pair_id \
            or row.get("arm") != arm \
            or row.get("runtime_receipt", {}).get("mode") != "SEALED_FORMAL_RUNTIME":
        raise Fixed4DriverError(f"pair strict authority mismatch: {pair_id}/{arm}")
    if row.get("formal_source_sha256") != expected.get("formal_source_sha256"):
        raise Fixed4DriverError(f"pair formal source mismatch: {pair_id}/{arm}")
    return True, row


def _artifact_manifest(root: Path) -> dict:
    excluded = {"artifact_manifest.json", "closure.json"}
    files = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"schema": "v13-fixed4-artifact-manifest-v1",
            "files": files, "file_count": len(files)}


def run_fixed4(*, repo: Path, preregister_path: Path, preflight_path: Path,
               output_root: Path, python: Path, device: str) -> dict:
    repo = repo.resolve(); output_root = output_root.resolve()
    preregister_path = preregister_path.resolve(); preflight_path = preflight_path.resolve()
    preregister = _load_json(preregister_path)
    preflight = _load_json(preflight_path)
    pairs = _frozen_pairs(preregister, preflight)
    prepared = _prepared_by_pair(preflight, pairs)
    driver = repo / "scripts/v13_fixed4_driver.py"
    orchestrator = Path(__file__).resolve()
    formal_sources = _formal_source_sha256(repo)
    dependency = {
        "driver_sha256": formal_sources["driver"],
        "orchestrator_sha256": sha256_file(orchestrator),
        "preregister_sha256": sha256_file(preregister_path),
        "preflight_sha256": sha256_file(preflight_path),
        "formal_source_sha256": formal_sources,
    }
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "src"),
                                         str(repo / "scripts")])
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    rows = []
    commands = []
    resources = []
    for pair_id in pairs:
        prepared_path, prepared_sha = prepared[pair_id]
        for arm in ARMS:
            pair_root = output_root / "pairs" / pair_id / arm
            summary_path = pair_root / "dual_solver" / "summary.json"
            receipt_path = output_root / "pair_receipts" / f"{pair_id}.{arm}.json"
            complete, row = _pair_receipt_valid(
                receipt_path, pair_id=pair_id, arm=arm,
                summary_path=summary_path, prepared_path=prepared_path,
                expected=dependency)
            if complete:
                rows.append(row)
                prior = _load_json(receipt_path)
                commands.append({"pair_id": pair_id, "arm": arm,
                                 "status": "resumed_hash_verified"})
                resources.append({"pair_id": pair_id, "arm": arm,
                                  "wall_seconds": prior.get("wall_seconds"),
                                  "resumed_hash_verified": True})
                continue
            command = [str(python), str(driver), "--repo", str(repo),
                       "--prepared", str(prepared_path), "--pair-id", pair_id,
                       "--arm", arm, "--output", str(pair_root),
                       "--preregister", str(preregister_path),
                       "--preflight-manifest", str(preflight_path),
                       "--device", device]
            started = time.monotonic()
            completed = subprocess.run(command, env=env, check=False)
            elapsed = time.monotonic() - started
            if completed.returncode not in (0, 2) or not summary_path.is_file():
                raise Fixed4DriverError(
                    f"pair driver failed without strict summary: {pair_id}/{arm} rc={completed.returncode}")
            row = _load_json(summary_path)
            if row.get("schema") != PAIR_SCHEMA or row.get("pair_id") != pair_id \
                    or row.get("arm") != arm \
                    or row.get("runtime_receipt", {}).get("mode") != "SEALED_FORMAL_RUNTIME":
                raise Fixed4DriverError(f"pair strict summary invalid: {pair_id}/{arm}")
            if row.get("formal_source_sha256") != formal_sources:
                raise Fixed4DriverError(
                    f"pair formal source mismatch: {pair_id}/{arm}")
            receipt = {"schema": PAIR_RECEIPT_SCHEMA, "pair_id": pair_id, "arm": arm,
                       "summary_path": str(summary_path),
                       "summary_sha256": sha256_file(summary_path),
                       "prepared_input_sha256": prepared_sha,
                       "wall_seconds": elapsed,
                       **dependency}
            _atomic_json(receipt_path, receipt)
            rows.append(row)
            commands.append({"pair_id": pair_id, "arm": arm,
                             "status": "executed", "returncode": completed.returncode,
                             "argv": command})
            resources.append({"pair_id": pair_id, "arm": arm,
                              "wall_seconds": elapsed})
    if len(rows) != 8:
        raise Fixed4DriverError("exact 8 strict pair summaries required")
    aggregate = aggregate_fixed4(rows, preregister)
    aggregate.update({"run_schema": RUN_SCHEMA, "dependency_sha256": dependency,
                      "gt_consumed": False, "official92_run": False})
    _atomic_json(output_root / "summary.json", aggregate)
    _atomic_json(output_root / "commands.json",
                 {"schema": "v13-fixed4-command-manifest-v1", "commands": commands})
    _atomic_json(output_root / "environment.json", {
        "schema": "v13-fixed4-environment-v1",
        "python": str(python.resolve()), "python_version": sys.version,
        "platform": {name: getattr(os.uname(), name)
                     for name in ("sysname", "nodename", "release",
                                  "version", "machine")},
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": env.get("CUBLAS_WORKSPACE_CONFIG"),
    })
    _atomic_json(output_root / "resource_usage.json", {
        "schema": "v13-fixed4-resource-usage-v1", "pair_runs": resources,
        "colorpcr_workers_expected": 32,
        "dual_solver_workers_expected": 160,
    })
    artifact_path = output_root / "artifact_manifest.json"
    _atomic_json(artifact_path, _artifact_manifest(output_root))
    closure = {"schema": "v13-fixed4-closure-v1",
               "summary_sha256": sha256_file(output_root / "summary.json"),
               "artifact_manifest_sha256": sha256_file(artifact_path),
               "preregister_sha256": dependency["preregister_sha256"],
               "preflight_sha256": dependency["preflight_sha256"]}
    _atomic_json(output_root / "closure.json", closure)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path,
                        default=Path("/home/aidenwu/miniconda3/envs/sgaligner/bin/python"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run_fixed4(repo=args.repo, preregister_path=args.preregister,
                         preflight_path=args.preflight_manifest,
                         output_root=args.output, python=args.python,
                         device=args.device)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
