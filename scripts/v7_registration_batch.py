"""GT-free 12-pair orchestration for the V7 registration-veto research pilot.

This controller deliberately reuses the frozen V7 worker and aggregation
implementation.  It never imports labels or ground truth.  The optional
posthoc phase is launched as a separate process only after all pair receipts
have been frozen and revalidated.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
os.environ["SGALIGNER_CODE_ROOT"] = str(CODE_ROOT)
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_pilot as pilot  # noqa: E402


MANIFEST_SCHEMA = "v7-registration-veto-batch-manifest-v1"
BATCH_SCHEMA = "v7-registration-veto-batch-receipt-v1"
POSTHOC_BATCH_SCHEMA = "v7-registration-veto-batch-posthoc-receipt-v1"
FORMAL_EVIDENCE_MODE = "PREREGISTERED_FORMAL"
RESEARCH_EVIDENCE_MODE = "NON_PREREGISTERED_RESEARCH"
PAIR_COUNT = 12
PAIR_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_to_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
PAIR_RECEIPT_SCHEMA = "v7-registration-veto-pilot-receipt-v1"
PAIR_POSTHOC_SCHEMA = "v7-registration-veto-posthoc-v1"
DEFAULT_MANIFEST = (
    CODE_ROOT / "outputs/v7_pilot_manifest_seal_20260830/"
    "v7_pilot_manifest.json")
DEFAULT_MANIFEST_SHA256 = (
    "9b3ea1f4ffa9361fe990a8dc21f370f73c6415624dc7b220765b7a88948b37aa")

ALLOWED_MANIFEST_KEYS = {
    "schema", "status", "pair_count", "pairs", "pair_ids_sha256",
    "checkpoint_id", "checkpoint_sha256", "protocol_sha256",
    # Explicitly non-decisional audit metadata admitted by the frozen schema.
    "audit_metadata", "reason", "created_at", "projection", "selector",
    "source", "inputs", "input_sha256", "source_sha256",
    "projection_sha256", "selector_sha256", "decision_sha256",
    "leakage_audit", "gt_free_contract", "population_decision_patterns",
    "projection_schema", "selection_receipt", "source_files",
}
ALLOWED_PAIR_KEYS = {
    "pair_id", "cache_sha256", "role", "audit_metadata", "reason",
    "input_sha256", "source_sha256",
}
SOURCE_FILES = {
    "batch_runner": "scripts/v7_registration_batch.py",
    "pilot_runner": "scripts/v7_registration_pilot.py",
    "posthoc_runner": "scripts/v7_registration_posthoc.py",
    "consensus": "src/safety/registration_consensus.py",
    "decision_features": "src/safety/decision_features.py",
    "registration_decision": "src/safety/registration_decision.py",
    "canonical_inputs": "scripts/canonical_inputs.py",
    "v3b_cache_runner": "scripts/v3b_cache_runner.py",
    "pilot_gate": "scripts/v7_pilot_gate.py",
}
CPU_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
)


class BatchEvidenceError(RuntimeError):
    """Frozen inputs or batch evidence failed a fail-closed check."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def pair_ids_sha256(pair_ids: list[str]) -> str:
    payload = "".join(f"{pair_id}\n" for pair_id in pair_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_optional_hashes(record: Mapping[str, Any]) -> None:
    for key, value in record.items():
        if key.endswith("_sha256") and not _is_sha256(value):
            raise BatchEvidenceError(f"invalid SHA-256 field {key}")


def validate_manifest(path: Path, expected_sha256: str, *,
                      allow_non_preregistered: bool = False) -> dict[str, Any]:
    path = path.resolve()
    formal_identity = (
        path == DEFAULT_MANIFEST.resolve()
        and expected_sha256 == DEFAULT_MANIFEST_SHA256
    )
    if not formal_identity and not allow_non_preregistered:
        raise BatchEvidenceError(
            "formal pilot mode requires the committed default manifest path "
            "and its frozen SHA; pass --research-non-preregistered for an "
            "alternative research manifest")
    if not _is_sha256(expected_sha256):
        raise BatchEvidenceError("manifest SHA-256 argument is malformed")
    if not path.is_file() or pilot.sha256_file(path) != expected_sha256:
        raise BatchEvidenceError("frozen manifest file SHA mismatch")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError("frozen manifest is not valid JSON") from exc
    if not isinstance(data, dict):
        raise BatchEvidenceError("manifest must be an object")
    unknown = set(data) - ALLOWED_MANIFEST_KEYS
    if unknown:
        raise BatchEvidenceError(
            f"unknown decision-affecting manifest fields: {sorted(unknown)}")
    required = {
        "schema", "status", "pair_count", "pairs", "pair_ids_sha256",
        "checkpoint_id", "checkpoint_sha256", "protocol_sha256",
    }
    if not required.issubset(data):
        raise BatchEvidenceError("manifest required fields missing")
    if data["schema"] != MANIFEST_SCHEMA or data["status"] != "FROZEN":
        raise BatchEvidenceError("manifest is not a frozen V7 batch manifest")
    if data["pair_count"] != PAIR_COUNT:
        raise BatchEvidenceError("manifest pair_count must be exactly 12")
    if (data["checkpoint_id"] != pilot.CHECKPOINT_ID
            or data["checkpoint_sha256"] != pilot.CHECKPOINT_SHA256):
        raise BatchEvidenceError("manifest checkpoint does not match frozen B")
    current_protocol = pilot.protocol_sha256()
    if data["protocol_sha256"] != current_protocol:
        raise BatchEvidenceError("manifest protocol SHA differs from repository")
    if not isinstance(data["pairs"], list) or len(data["pairs"]) != PAIR_COUNT:
        raise BatchEvidenceError("manifest must contain exactly 12 pair rows")
    pair_ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(data["pairs"]):
        if not isinstance(row, dict):
            raise BatchEvidenceError(f"pair row {index} is not an object")
        unknown_pair = set(row) - ALLOWED_PAIR_KEYS
        if unknown_pair:
            raise BatchEvidenceError(
                f"unknown pair fields at row {index}: {sorted(unknown_pair)}")
        if not {"pair_id", "cache_sha256", "role"}.issubset(row):
            raise BatchEvidenceError(f"pair row {index} fields missing")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not PAIR_ID_RE.fullmatch(pair_id):
            raise BatchEvidenceError(f"unsafe pair_id at row {index}")
        if not _is_sha256(row["cache_sha256"]):
            raise BatchEvidenceError(f"cache SHA malformed at row {index}")
        if not isinstance(row["role"], str) or not row["role"].strip():
            raise BatchEvidenceError(f"role missing at row {index}")
        _validate_optional_hashes(row)
        pair_ids.append(pair_id)
        normalized.append(dict(row))
    if len(set(pair_ids)) != PAIR_COUNT:
        raise BatchEvidenceError("manifest pair ids must be unique")
    if data["pair_ids_sha256"] != pair_ids_sha256(pair_ids):
        raise BatchEvidenceError("manifest ordered pair-id SHA mismatch")
    _validate_optional_hashes(data)
    result = dict(data)
    result["pairs"] = normalized
    result["_path"] = str(path)
    result["_file_sha256"] = expected_sha256
    result["_evidence_mode"] = (
        FORMAL_EVIDENCE_MODE if formal_identity
        else RESEARCH_EVIDENCE_MODE)
    result["_formal_preregistered"] = bool(formal_identity)
    for row in result["pairs"]:
        row["_evidence_mode"] = result["_evidence_mode"]
    return result


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_output_root(output_root: Path) -> Path:
    """Prevent a broad in-repository output allowance from hiding source files."""
    resolved = output_root.resolve()
    repository_outputs = (CODE_ROOT / "outputs").resolve()
    if _within(resolved, CODE_ROOT.resolve()):
        if resolved == repository_outputs or not _within(
                resolved, repository_outputs):
            raise BatchEvidenceError(
                "in-repository batch output must be a named outputs/ child")
    if (resolved == pilot.FORMAL_ROOT.resolve()
            or _within(resolved, pilot.FORMAL_ROOT.resolve())):
        raise BatchEvidenceError("batch output must not enter immutable V6 outputs")
    return resolved


def parse_porcelain_z(payload: bytes) -> list[tuple[str, str]]:
    """Parse the path-bearing subset of porcelain-v1 -z conservatively."""
    tokens = payload.split(b"\0")
    rows: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        text = token.decode("utf-8", errors="strict")
        if len(text) < 4:
            raise BatchEvidenceError("malformed git porcelain record")
        status, path = text[:2], text[3:]
        rows.append((status, path))
        if "R" in status or "C" in status:
            if index >= len(tokens) or not tokens[index]:
                raise BatchEvidenceError("malformed git rename record")
            rows.append((status, tokens[index].decode("utf-8", errors="strict")))
            index += 1
    return rows


def repository_state(output_root: Path) -> dict[str, Any]:
    output_root = validate_output_root(output_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()
    if not HEAD_RE.fullmatch(head):
        raise BatchEvidenceError("repository HEAD is not a full lowercase SHA")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=CODE_ROOT, check=True, capture_output=True).stdout
    allowed_root = output_root.resolve()
    for state, relative in parse_porcelain_z(status):
        candidate = (CODE_ROOT / relative).resolve()
        if state == "??" and _within(candidate, allowed_root):
            continue
        raise BatchEvidenceError(
            f"repository contains non-output change: {state} {relative}")
    return {"head": head, "tracked_dirty": False,
            "untracked_outside_output": False}


def _package_record(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"available": False, "version": None, "module_file": None}
    version = None
    candidates = (name, "pygcransac") if name == "pygcransac" else (name,)
    for candidate in candidates:
        try:
            version = importlib.metadata.version(candidate)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if version is None:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", None)
        except Exception:
            version = None
    return {"available": True, "version": str(version) if version else None,
            "module_file": spec.origin}


def source_snapshot(repository: Mapping[str, Any]) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for name, relative in SOURCE_FILES.items():
        path = CODE_ROOT / relative
        if not path.is_file():
            raise BatchEvidenceError(f"required source file missing: {relative}")
        hashes[name] = {"path": relative, "sha256": pilot.sha256_file(path)}
    packages = {name: _package_record(name)
                for name in ("numpy", "scipy", "torch", "pygcransac")}
    snapshot = {
        "repository": dict(repository),
        "source_files": hashes,
        "packages": packages,
        "controller_cpu_environment": {
            key: os.environ.get(key) for key in CPU_ENV_KEYS
        },
        "worker_environment": {
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
    }
    snapshot["snapshot_sha256"] = pilot.stable_json_hash(snapshot)
    return snapshot


def preflight_caches(manifest: Mapping[str, Any]) -> None:
    """Bind all twelve immutable cache files before starting any worker."""
    for row in manifest["pairs"]:
        path = pilot.cache_path(pilot.DEFAULT_CACHE_ROOT, row["pair_id"])
        if pilot.sha256_file(path) != row["cache_sha256"]:
            raise BatchEvidenceError(
                f"immutable cache SHA mismatch {row['pair_id']}")


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    return environment


def _load_batch_worker(path: Path, *, pair_id: str, direction: str,
                       replicate: int, cache_sha: str,
                       protocol_sha: str) -> dict[str, Any]:
    row = pilot.load_worker(
        path, pair_id=pair_id, direction=direction, replicate=replicate,
        cache_sha=cache_sha, protocol_sha=protocol_sha)
    cache = row.get("cache", {})
    if (row.get("status") != "ok"
            or cache.get("checkpoint_id") != pilot.CHECKPOINT_ID
            or cache.get("checkpoint_sha256") != pilot.CHECKPOINT_SHA256):
        raise BatchEvidenceError(f"worker checkpoint/status mismatch {path}")
    for field in ("raw_transform", "final_transform"):
        transform = np.asarray(row.get(field), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise BatchEvidenceError(f"worker non-finite transform {path}")
    count = row.get("correspondence_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise BatchEvidenceError(f"worker correspondence count invalid {path}")
    permutation, provenance = pilot.stable_row_permutation(
        count, pair_id=pair_id, direction=direction, replicate=replicate,
        protocol_sha=protocol_sha)
    permutation_sha = hashlib.sha256(
        np.ascontiguousarray(permutation.astype(np.int64)).tobytes()
    ).hexdigest()
    if (row.get("permutation_provenance_sha256") != provenance
            or row.get("permutation_sha256") != permutation_sha):
        raise BatchEvidenceError(f"worker permutation binding mismatch {path}")
    return row


def _worker_permutation_binding(worker: Mapping[str, Any]) -> dict[str, Any]:
    """Return the direction/replicate-keyed deterministic permutation proof."""
    return {
        "direction": worker["direction"],
        "replicate": worker["replicate"],
        "correspondence_count": worker["correspondence_count"],
        "permutation_provenance_sha256": worker[
            "permutation_provenance_sha256"],
        "permutation_sha256": worker["permutation_sha256"],
    }


def _expected_outer_files() -> set[str]:
    return {
        *(f"{direction}_{replicate:02d}.json"
          for direction in ("forward", "reverse") for replicate in range(5)),
        "gt_free_aggregate.json",
    }


def _validate_directory_shape(run_dir: Path, *, complete: bool) -> None:
    observed = {path.name for path in run_dir.iterdir()}
    expected = _expected_outer_files()
    if complete:
        if observed != expected:
            raise BatchEvidenceError(
                f"complete outer directory has unexpected files {run_dir}")
    elif observed:
        # Partial outers are intentionally not guessed/recovered: their exact
        # elapsed-time provenance was not atomically frozen.
        raise BatchEvidenceError(
            f"dirty/incomplete outer evidence cannot be resumed {run_dir}")


def _aggregate_binding(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": aggregate.get("schema"),
        "pair_id": aggregate.get("pair_id"),
        "cache": aggregate.get("cache"),
        "protocol": aggregate.get("protocol"),
        "manifest_sha256": aggregate.get("batch", {}).get("manifest_sha256"),
        "source_snapshot_sha256": aggregate.get("batch", {}).get(
            "source_snapshot_sha256"),
        "policy_names": sorted(aggregate.get("policies", {})),
        "worker_shape": {
            "requested": aggregate.get("workers", {}).get("requested"),
            "completed": aggregate.get("workers", {}).get("completed"),
        },
        "worker_permutation_bindings": aggregate.get(
            "worker_permutation_bindings"),
    }


def validate_aggregate(path: Path, *, pair: Mapping[str, Any], outer: int,
                       manifest_sha: str, source_snapshot_sha: str) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError(f"invalid aggregate JSON {path}") from exc
    expected_hash = data.pop("evidence_sha256", None)
    actual_hash = pilot.stable_json_hash(data)
    data["evidence_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise BatchEvidenceError(f"aggregate evidence SHA mismatch {path}")
    if (data.get("schema") != pilot.SCHEMA
            or data.get("status") != "GT_FREE_COMPLETE"
            or data.get("pair_id") != pair["pair_id"]
            or data.get("outer_repeat") != outer
            or data.get("cache", {}).get("sha256") != pair["cache_sha256"]
            or data.get("cache", {}).get("checkpoint_id") != pilot.CHECKPOINT_ID
            or data.get("cache", {}).get("checkpoint_sha256")
            != pilot.CHECKPOINT_SHA256
            or data.get("protocol", {}).get("sha256")
            != pilot.protocol_sha256()
            or data.get("batch", {}).get("manifest_sha256") != manifest_sha
            or data.get("batch", {}).get("source_snapshot_sha256")
            != source_snapshot_sha):
        raise BatchEvidenceError(f"aggregate provenance mismatch {path}")
    if data.get("batch", {}).get("evidence_mode") != pair.get(
            "_evidence_mode"):
        raise BatchEvidenceError(f"aggregate evidence mode mismatch {path}")
    workers = data.get("workers", {})
    if (workers.get("requested") != 10 or workers.get("completed") != 10
            or workers.get("exceptions") != 0
            or workers.get("nonfinite_transforms") != 0
            or workers.get("cache_mismatches") != 0):
        raise BatchEvidenceError(f"aggregate worker gate failed {path}")
    timing = workers.get("timing")
    expected_worker_keys = {
        (direction, replicate) for direction in ("forward", "reverse")
        for replicate in range(5)
    }
    if (not isinstance(timing, list) or len(timing) != 10
            or {(row.get("direction"), row.get("replicate"))
                for row in timing} != expected_worker_keys
            or any(row.get("returncode") != 0
                   or not _is_sha256(row.get("command_sha256"))
                   or not isinstance(row.get("elapsed_wall_s"), (int, float))
                   or not np.isfinite(row["elapsed_wall_s"])
                   or row["elapsed_wall_s"] < 0 for row in timing)):
        raise BatchEvidenceError(f"aggregate worker timing malformed {path}")
    if (not isinstance(data.get("elapsed_wall_s"), (int, float))
            or not np.isfinite(data["elapsed_wall_s"])
            or data["elapsed_wall_s"] < 0):
        raise BatchEvidenceError(f"aggregate elapsed time malformed {path}")
    observed_worker_hashes = []
    observed_permutation_bindings = []
    for direction, replicate in sorted(expected_worker_keys):
        worker_path = path.parent / f"{direction}_{replicate:02d}.json"
        worker = _load_batch_worker(
            worker_path, pair_id=pair["pair_id"], direction=direction,
            replicate=replicate, cache_sha=pair["cache_sha256"],
            protocol_sha=pilot.protocol_sha256())
        observed_worker_hashes.append(worker["evidence_sha256"])
        observed_permutation_bindings.append(
            _worker_permutation_binding(worker))
    if sorted(observed_worker_hashes) != data.get("worker_evidence_sha256"):
        raise BatchEvidenceError(f"aggregate worker hash set mismatch {path}")
    if observed_permutation_bindings != data.get(
            "worker_permutation_bindings"):
        raise BatchEvidenceError(
            f"aggregate worker permutation bindings mismatch {path}")
    expected_policies = {pilot.policy_name(config) for config in pilot.POLICIES}
    policies = data.get("policies")
    if not isinstance(policies, dict) or set(policies) != expected_policies:
        raise BatchEvidenceError(f"aggregate policy grid mismatch {path}")
    if any(not isinstance(row.get("usable_for_reconstruction"), bool)
           for row in policies.values()):
        raise BatchEvidenceError(f"aggregate policy decision malformed {path}")
    return data


def run_outer(pair: Mapping[str, Any], *, outer: int, out_root: Path,
              manifest_sha: str, snapshot: Mapping[str, Any],
              resume: bool) -> Path:
    pair_id = pair["pair_id"]
    cache = pilot.cache_path(pilot.DEFAULT_CACHE_ROOT, pair_id)
    if pilot.sha256_file(cache) != pair["cache_sha256"]:
        raise BatchEvidenceError(f"immutable cache SHA mismatch {pair_id}")
    protocol_sha = pilot.protocol_sha256()
    run_dir = out_root / "pairs" / pair_id / f"outer_{outer:02d}"
    aggregate_path = run_dir / "gt_free_aggregate.json"
    if run_dir.exists():
        if not resume:
            raise BatchEvidenceError(f"refusing to reuse outer {run_dir}")
        if aggregate_path.is_file():
            _validate_directory_shape(run_dir, complete=True)
            # Worker records are validated before accepting resumed aggregate.
            for direction in ("forward", "reverse"):
                for replicate in range(5):
                    _load_batch_worker(
                        run_dir / f"{direction}_{replicate:02d}.json",
                        pair_id=pair_id, direction=direction,
                        replicate=replicate, cache_sha=pair["cache_sha256"],
                        protocol_sha=protocol_sha)
            validate_aggregate(
                aggregate_path, pair=pair, outer=outer,
                manifest_sha=manifest_sha,
                source_snapshot_sha=snapshot["snapshot_sha256"])
            return aggregate_path
        _validate_directory_shape(run_dir, complete=False)
    else:
        run_dir.mkdir(parents=True)

    workers: list[dict[str, Any]] = []
    timing: list[dict[str, Any]] = []
    outer_started = time.monotonic()
    python = Path(sys.executable).resolve()
    runner = (CODE_ROOT / "scripts/v7_registration_pilot.py").resolve()
    for direction in ("forward", "reverse"):
        for replicate in range(5):
            worker_out = run_dir / f"{direction}_{replicate:02d}.json"
            command = [
                str(python), str(runner), "--worker", "--pair", pair_id,
                "--direction", direction, "--replicate", str(replicate),
                "--cache-root", str(pilot.DEFAULT_CACHE_ROOT),
                "--cache-sha256", pair["cache_sha256"],
                "--protocol-sha256", protocol_sha,
                "--worker-out", str(worker_out),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command, cwd=CODE_ROOT, env=_worker_environment(),
                capture_output=True, text=True)
            elapsed = time.monotonic() - started
            timing.append({
                "direction": direction, "replicate": replicate,
                "elapsed_wall_s": elapsed,
                "command_sha256": hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode()
                ).hexdigest(),
                "returncode": completed.returncode,
            })
            if completed.returncode != 0:
                raise BatchEvidenceError(
                    f"worker failed {pair_id} {direction}/{replicate}: "
                    f"{completed.stderr[-4000:]}")
            workers.append(_load_batch_worker(
                worker_out, pair_id=pair_id, direction=direction,
                replicate=replicate, cache_sha=pair["cache_sha256"],
                protocol_sha=protocol_sha))
    if pilot.sha256_file(cache) != pair["cache_sha256"]:
        raise BatchEvidenceError(f"cache changed during outer {pair_id}")
    policies = {
        pilot.policy_name(config): pilot.aggregate_policy(workers, config)
        for config in pilot.POLICIES
    }
    aggregate = {
        "schema": pilot.SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "research_only": True,
        "pair_id": pair_id,
        "known_near_miss": pair_id == pilot.NEAR_MISS_PAIR,
        "outer_repeat": outer,
        "repository": dict(snapshot["repository"]),
        "cache": {
            "path": str(cache), "sha256": pair["cache_sha256"],
            "checkpoint_id": pilot.CHECKPOINT_ID,
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
        },
        "protocol": {"path": str(pilot.PROTOCOL), "sha256": protocol_sha},
        "batch": {
            "manifest_sha256": manifest_sha,
            "source_snapshot_sha256": snapshot["snapshot_sha256"],
            "pair_role": pair["role"],
            "evidence_mode": snapshot["manifest_evidence_mode"],
        },
        "worker_evidence_sha256": sorted(
            row["evidence_sha256"] for row in workers),
        "worker_permutation_bindings": [
            _worker_permutation_binding(row)
            for row in sorted(
                workers, key=lambda item: (
                    item["direction"], item["replicate"]))
        ],
        "workers": {
            "requested": 10, "completed": 10, "exceptions": 0,
            "nonfinite_transforms": 0, "cache_mismatches": 0,
            "environment": dict(snapshot["worker_environment"]),
            "timing": timing,
        },
        "elapsed_wall_s": time.monotonic() - outer_started,
        "policies": policies,
    }
    aggregate["evidence_sha256"] = pilot.stable_json_hash(aggregate)
    pilot.atomic_create_json(aggregate_path, aggregate)
    return aggregate_path


def policy_repeatability(aggregates: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(aggregates) != 2:
        raise BatchEvidenceError("repeatability requires exactly two outers")
    bindings = [_aggregate_binding(row) for row in aggregates]
    binding_hashes = [pilot.stable_json_hash(row) for row in bindings]
    if binding_hashes[0] != binding_hashes[1]:
        raise BatchEvidenceError("outer input/schema bindings differ")
    output: dict[str, Any] = {}
    for name in sorted(aggregates[0]["policies"]):
        states = [bool(row["policies"][name]["usable_for_reconstruction"])
                  for row in aggregates]
        output[name] = {
            "outer_usable": states,
            "repeatable": states[0] == states[1],
            "outcome": ("usable" if states == [True, True]
                        else "veto" if states == [False, False]
                        else "mixed"),
        }
    return {
        "input_and_schema_repeatable": True,
        "binding_sha256": binding_hashes[0],
        "aggregate_evidence_hashes_valid": [True, True],
        # Random transform hashes and per-worker Rule-B patterns are
        # intentionally absent from this comparison.
        "policies": output,
    }


def _validate_pair_receipt(path: Path, *, pair: Mapping[str, Any],
                           manifest_sha: str,
                           source_snapshot_sha: str) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError(f"invalid pair receipt {path}") from exc
    expected = receipt.pop("evidence_sha256", None)
    actual = pilot.stable_json_hash(receipt)
    receipt["evidence_sha256"] = expected
    if expected != actual:
        raise BatchEvidenceError(f"pair receipt evidence SHA mismatch {path}")
    if (receipt.get("schema") != PAIR_RECEIPT_SCHEMA
            or receipt.get("status") != "GT_FREE_COMPLETE"
            or receipt.get("pair_id") != pair["pair_id"]
            or receipt.get("outer_repeats") != 2
            or receipt.get("posthoc_not_run") is not True
            or receipt.get("batch", {}).get("manifest_sha256") != manifest_sha
            or receipt.get("batch", {}).get("source_snapshot_sha256")
            != source_snapshot_sha
            or receipt.get("batch", {}).get("evidence_mode")
            != pair.get("_evidence_mode")
            or len(receipt.get("aggregates", [])) != 2):
        raise BatchEvidenceError(f"pair receipt provenance mismatch {path}")
    if (not isinstance(receipt.get("elapsed_wall_s"), (int, float))
            or not np.isfinite(receipt["elapsed_wall_s"])
            or receipt["elapsed_wall_s"] < 0):
        raise BatchEvidenceError(f"pair receipt elapsed time malformed {path}")
    aggregates = []
    for outer, row in enumerate(receipt["aggregates"]):
        aggregate_path = Path(row["path"]).resolve()
        if not _is_sha256(row.get("sha256")):
            raise BatchEvidenceError("pair aggregate SHA malformed")
        if pilot.sha256_file(aggregate_path) != row["sha256"]:
            raise BatchEvidenceError("pair aggregate file SHA mismatch")
        aggregates.append(validate_aggregate(
            aggregate_path, pair=pair, outer=outer,
            manifest_sha=manifest_sha,
            source_snapshot_sha=source_snapshot_sha))
    if receipt.get("repeatability") != policy_repeatability(aggregates):
        raise BatchEvidenceError("pair repeatability report mismatch")
    return receipt


def run_pair(pair: Mapping[str, Any], *, out_root: Path,
             manifest_sha: str, snapshot: Mapping[str, Any],
             resume: bool) -> Path:
    receipt_path = out_root / "pair_receipts" / f"{pair['pair_id']}.json"
    if receipt_path.exists():
        if not resume:
            raise BatchEvidenceError(f"refusing to overwrite {receipt_path}")
        _validate_pair_receipt(
            receipt_path, pair=pair, manifest_sha=manifest_sha,
            source_snapshot_sha=snapshot["snapshot_sha256"])
        return receipt_path
    started = time.monotonic()
    destinations = [
        run_outer(pair, outer=outer, out_root=out_root,
                  manifest_sha=manifest_sha, snapshot=snapshot,
                  resume=resume)
        for outer in range(2)
    ]
    aggregates = [validate_aggregate(
        path, pair=pair, outer=outer, manifest_sha=manifest_sha,
        source_snapshot_sha=snapshot["snapshot_sha256"])
        for outer, path in enumerate(destinations)]
    receipt = {
        "schema": PAIR_RECEIPT_SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "pair_id": pair["pair_id"],
        "outer_repeats": 2,
        "aggregates": [
            {"path": str(path), "sha256": pilot.sha256_file(path)}
            for path in destinations
        ],
        "posthoc_not_run": True,
        "batch": {
            "manifest_sha256": manifest_sha,
            "source_snapshot_sha256": snapshot["snapshot_sha256"],
            "role": pair["role"],
            "evidence_mode": snapshot["manifest_evidence_mode"],
        },
        "repeatability": policy_repeatability(aggregates),
        "elapsed_wall_s": time.monotonic() - started,
    }
    receipt["evidence_sha256"] = pilot.stable_json_hash(receipt)
    pilot.atomic_create_json(receipt_path, receipt)
    return receipt_path


def _batch_policy_summary(pair_receipts: list[Mapping[str, Any]]) -> dict:
    names = sorted(pair_receipts[0]["repeatability"]["policies"])
    summary: dict[str, Any] = {}
    for name in names:
        rows = []
        for receipt in pair_receipts:
            row = receipt["repeatability"]["policies"][name]
            rows.append({
                "pair_id": receipt["pair_id"],
                "outer_usable": row["outer_usable"],
                "repeatable": row["repeatable"],
                "outcome": row["outcome"],
            })
        summary[name] = {
            "usable_pairs": sum(row["outcome"] == "usable" for row in rows),
            "vetoed_pairs": sum(row["outcome"] == "veto" for row in rows),
            "mixed_pairs": sum(row["outcome"] == "mixed" for row in rows),
            "all_pair_outcomes_repeatable": all(
                row["repeatable"] for row in rows),
            "pairs": rows,
        }
    return summary


def validate_batch_receipt(path: Path, *, manifest: Mapping[str, Any],
                           snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError(f"invalid batch receipt {path}") from exc
    expected = receipt.pop("evidence_sha256", None)
    actual = pilot.stable_json_hash(receipt)
    receipt["evidence_sha256"] = expected
    if expected != actual:
        raise BatchEvidenceError("batch receipt evidence SHA mismatch")
    if (receipt.get("schema") != BATCH_SCHEMA
            or receipt.get("status") != "GT_FREE_COMPLETE"
            or receipt.get("pair_count") != PAIR_COUNT
            or receipt.get("posthoc_not_run") is not True
            or receipt.get("manifest", {}).get("sha256")
            != manifest["_file_sha256"]
            or receipt.get("evidence_mode") != manifest["_evidence_mode"]
            or receipt.get("formal_preregistered")
            is not manifest["_formal_preregistered"]
            or receipt.get("source_snapshot", {}).get("snapshot_sha256")
            != snapshot["snapshot_sha256"]
            or len(receipt.get("pair_receipts", [])) != PAIR_COUNT
            or receipt.get("global_fail_closed_counts") != {
                "exceptions": 0, "nonfinite_transforms": 0,
                "cache_mismatches": 0,
            }):
        raise BatchEvidenceError("batch receipt provenance/gates mismatch")
    if (not isinstance(receipt.get("elapsed_wall_s"), (int, float))
            or not np.isfinite(receipt["elapsed_wall_s"])
            or receipt["elapsed_wall_s"] < 0):
        raise BatchEvidenceError("batch elapsed time malformed")
    pair_by_id = {row["pair_id"]: row for row in manifest["pairs"]}
    expected_pair_order = [row["pair_id"] for row in manifest["pairs"]]
    if ([row.get("pair_id") for row in receipt["pair_receipts"]]
            != expected_pair_order):
        raise BatchEvidenceError("batch receipt pair order differs from manifest")
    loaded = []
    for row in receipt["pair_receipts"]:
        pair = pair_by_id.get(row.get("pair_id"))
        if pair is None or not _is_sha256(row.get("sha256")):
            raise BatchEvidenceError("batch pair receipt row malformed")
        pair_path = Path(row["path"]).resolve()
        if pilot.sha256_file(pair_path) != row["sha256"]:
            raise BatchEvidenceError("batch pair receipt file SHA mismatch")
        loaded.append(_validate_pair_receipt(
            pair_path, pair=pair,
            manifest_sha=manifest["_file_sha256"],
            source_snapshot_sha=snapshot["snapshot_sha256"]))
    if {row["pair_id"] for row in loaded} != set(pair_by_id):
        raise BatchEvidenceError("batch receipt pair coverage mismatch")
    if receipt.get("policy_pair_summary") != _batch_policy_summary(loaded):
        raise BatchEvidenceError("batch policy summary mismatch")
    return receipt


def run_gt_free(manifest: Mapping[str, Any], *, out_root: Path,
                resume: bool, pair_concurrency: int) -> dict[str, Any]:
    receipt_path = out_root / "gt_free_batch_receipt.json"
    if out_root.exists() and not resume:
        raise BatchEvidenceError(f"refusing existing batch output {out_root}")
    repository = repository_state(out_root)
    snapshot = source_snapshot(repository)
    snapshot["manifest_evidence_mode"] = manifest["_evidence_mode"]
    snapshot["snapshot_sha256"] = pilot.stable_json_hash(
        {key: value for key, value in snapshot.items()
         if key != "snapshot_sha256"})
    preflight_caches(manifest)
    if receipt_path.exists():
        receipt = validate_batch_receipt(
            receipt_path, manifest=manifest, snapshot=snapshot)
        if not resume:
            raise BatchEvidenceError("refusing existing batch receipt")
        return receipt
    started = time.monotonic()
    pair_paths: list[Path] = []
    kwargs = {
        "out_root": out_root,
        "manifest_sha": manifest["_file_sha256"],
        "snapshot": snapshot,
        "resume": resume,
    }
    if pair_concurrency == 1:
        pair_paths = [run_pair(pair, **kwargs) for pair in manifest["pairs"]]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=pair_concurrency) as executor:
            futures = [executor.submit(run_pair, pair, **kwargs)
                       for pair in manifest["pairs"]]
            pair_paths = [future.result() for future in futures]
    pairs_by_id = {row["pair_id"]: row for row in manifest["pairs"]}
    loaded = [_validate_pair_receipt(
        path, pair=pairs_by_id[json.loads(path.read_text())["pair_id"]],
        manifest_sha=manifest["_file_sha256"],
        source_snapshot_sha=snapshot["snapshot_sha256"])
        for path in pair_paths]
    receipt = {
        "schema": BATCH_SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "research_only": True,
        "evidence_mode": manifest["_evidence_mode"],
        "formal_preregistered": manifest["_formal_preregistered"],
        "manifest": {"path": manifest["_path"],
                     "sha256": manifest["_file_sha256"]},
        "pair_count": PAIR_COUNT,
        "outer_repeats_per_pair": 2,
        "replicates_per_outer": {"forward": 5, "reverse": 5},
        "pair_receipts": [
            {"pair_id": row["pair_id"], "path": str(path),
             "sha256": pilot.sha256_file(path),
             "elapsed_wall_s": row["elapsed_wall_s"]}
            for path, row in zip(pair_paths, loaded)
        ],
        "policy_pair_summary": _batch_policy_summary(loaded),
        "global_fail_closed_counts": {
            "exceptions": 0, "nonfinite_transforms": 0,
            "cache_mismatches": 0,
        },
        "source_snapshot": snapshot,
        "elapsed_wall_s": time.monotonic() - started,
        "posthoc_not_run": True,
    }
    receipt["evidence_sha256"] = pilot.stable_json_hash(receipt)
    pilot.atomic_create_json(receipt_path, receipt)
    return receipt


def _validate_pair_posthoc(path: Path, pair_receipt: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError(f"invalid pair posthoc {path}") from exc
    expected = data.pop("evidence_sha256", None)
    actual = pilot.stable_json_hash(data)
    data["evidence_sha256"] = expected
    if expected != actual or data.get("schema") != PAIR_POSTHOC_SCHEMA:
        raise BatchEvidenceError(f"pair posthoc evidence invalid {path}")
    pair_receipt_data = json.loads(pair_receipt.read_text())
    receipt_evidence = pair_receipt_data.pop("evidence_sha256", None)
    if receipt_evidence != pilot.stable_json_hash(pair_receipt_data):
        raise BatchEvidenceError(
            f"pair receipt evidence invalid for posthoc {pair_receipt}")
    pair_receipt_data["evidence_sha256"] = receipt_evidence
    aggregate_rows = pair_receipt_data.get("aggregates", [])
    if (data.get("status") != "POSTHOC_COMPLETE"
            or data.get("receipt", {}).get("sha256")
            != pilot.sha256_file(pair_receipt)
            or data.get("pair_id") != pair_receipt_data.get("pair_id")
            or data.get("outer_repeats") != 2
            or data.get("manifest_sha256")
            != pair_receipt_data.get("batch", {}).get("manifest_sha256")
            or data.get("source_snapshot_sha256")
            != pair_receipt_data.get("batch", {}).get(
                "source_snapshot_sha256")
            or data.get("evidence_mode")
            != pair_receipt_data.get("batch", {}).get("evidence_mode")
            or data.get("source_sha256")
            != pilot.sha256_file(CODE_ROOT / "scripts/v7_registration_posthoc.py")
            or len(aggregate_rows) != 2
            or data.get("aggregate_bindings") != [
                {"outer_repeat": outer, "path": row.get("path"),
                 "sha256": row.get("sha256")}
                for outer, row in enumerate(aggregate_rows)]):
        raise BatchEvidenceError(f"pair posthoc receipt binding mismatch {path}")
    runs = data.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise BatchEvidenceError(f"pair posthoc run shape mismatch {path}")
    for outer, (run, aggregate_row) in enumerate(zip(runs, aggregate_rows)):
        aggregate_path = Path(aggregate_row["path"]).resolve()
        if (not _is_sha256(aggregate_row.get("sha256"))
                or pilot.sha256_file(aggregate_path)
                != aggregate_row["sha256"]):
            raise BatchEvidenceError(
                f"pair posthoc aggregate file binding mismatch {path}")
        aggregate = json.loads(aggregate_path.read_text())
        aggregate_expected = aggregate.pop("evidence_sha256", None)
        aggregate_actual = pilot.stable_json_hash(aggregate)
        aggregate["evidence_sha256"] = aggregate_expected
        if (run.get("pair_id") != data["pair_id"]
                or run.get("outer_repeat") != outer
                or run.get("gt_free_evidence_sha256")
                != aggregate.get("evidence_sha256")
                or aggregate_expected != aggregate_actual
                or set(run.get("policies", {}))
                != set(aggregate.get("policies", {}))):
            raise BatchEvidenceError(f"pair posthoc run binding mismatch {path}")
    return data


def validate_posthoc_batch_receipt(
        path: Path, *, batch_receipt_path: Path,
        batch: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchEvidenceError(f"invalid posthoc batch receipt {path}") from exc
    expected = receipt.pop("evidence_sha256", None)
    actual = pilot.stable_json_hash(receipt)
    receipt["evidence_sha256"] = expected
    if expected != actual:
        raise BatchEvidenceError("posthoc batch evidence SHA mismatch")
    if (receipt.get("schema") != POSTHOC_BATCH_SCHEMA
            or receipt.get("status") != "POSTHOC_COMPLETE"
            or receipt.get("pair_count") != PAIR_COUNT
            or receipt.get("gt_free_batch_receipt", {}).get("sha256")
            != pilot.sha256_file(batch_receipt_path)
            or receipt.get("source_snapshot_sha256")
            != snapshot["snapshot_sha256"]
            or receipt.get("evidence_mode") != batch.get("evidence_mode")
            or len(receipt.get("posthoc", [])) != PAIR_COUNT):
        raise BatchEvidenceError("posthoc batch receipt provenance mismatch")
    pair_receipts = {
        row["pair_id"]: Path(row["path"]).resolve()
        for row in batch["pair_receipts"]
    }
    observed = set()
    if ([row.get("pair_id") for row in receipt["posthoc"]]
            != [row["pair_id"] for row in batch["pair_receipts"]]):
        raise BatchEvidenceError("posthoc pair order differs from GT-free receipt")
    for row in receipt["posthoc"]:
        pair_id = row.get("pair_id")
        pair_receipt = pair_receipts.get(pair_id)
        if pair_receipt is None or not _is_sha256(row.get("sha256")):
            raise BatchEvidenceError("posthoc pair row malformed")
        output = Path(row["path"]).resolve()
        if pilot.sha256_file(output) != row["sha256"]:
            raise BatchEvidenceError("posthoc pair file SHA mismatch")
        _validate_pair_posthoc(output, pair_receipt)
        observed.add(pair_id)
    if observed != set(pair_receipts):
        raise BatchEvidenceError("posthoc pair coverage mismatch")
    return receipt


def run_posthoc(manifest: Mapping[str, Any], *, batch_receipt_path: Path,
                out_root: Path, resume: bool) -> dict[str, Any]:
    repository = repository_state(out_root)
    snapshot = source_snapshot(repository)
    snapshot["manifest_evidence_mode"] = manifest["_evidence_mode"]
    snapshot["snapshot_sha256"] = pilot.stable_json_hash(
        {key: value for key, value in snapshot.items()
         if key != "snapshot_sha256"})
    preflight_caches(manifest)
    batch = validate_batch_receipt(
        batch_receipt_path, manifest=manifest, snapshot=snapshot)
    destination = out_root / "posthoc_batch_receipt.json"
    if destination.exists():
        if not resume:
            raise BatchEvidenceError(f"refusing to overwrite {destination}")
        return validate_posthoc_batch_receipt(
            destination, batch_receipt_path=batch_receipt_path,
            batch=batch, snapshot=snapshot)
    started = time.monotonic()
    posthoc_rows = []
    runner = CODE_ROOT / "scripts/v7_registration_posthoc.py"
    for row in batch["pair_receipts"]:
        pair_receipt = Path(row["path"]).resolve()
        output = out_root / "posthoc" / f"{row['pair_id']}.json"
        pair_started = time.monotonic()
        if output.exists():
            if not resume:
                raise BatchEvidenceError(f"refusing to overwrite {output}")
            _validate_pair_posthoc(output, pair_receipt)
        else:
            completed = subprocess.run(
                [str(Path(sys.executable).resolve()), str(runner),
                 "--receipt", str(pair_receipt), "--out", str(output)],
                cwd=CODE_ROOT, env=_worker_environment(),
                capture_output=True, text=True)
            if completed.returncode != 0:
                raise BatchEvidenceError(
                    f"posthoc failed {row['pair_id']}: "
                    f"{completed.stderr[-4000:]}")
            _validate_pair_posthoc(output, pair_receipt)
        posthoc_rows.append({
            "pair_id": row["pair_id"], "path": str(output),
            "sha256": pilot.sha256_file(output),
            "elapsed_wall_s": time.monotonic() - pair_started,
        })
    receipt = {
        "schema": POSTHOC_BATCH_SCHEMA,
        "status": "POSTHOC_COMPLETE",
        "gt_free_batch_receipt": {
            "path": str(batch_receipt_path),
            "sha256": pilot.sha256_file(batch_receipt_path),
        },
        "pair_count": PAIR_COUNT,
        "posthoc": posthoc_rows,
        "gt_scope": "separate process after all GT-free pair receipts frozen",
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "evidence_mode": manifest["_evidence_mode"],
        "elapsed_wall_s": time.monotonic() - started,
    }
    receipt["evidence_sha256"] = pilot.stable_json_hash(receipt)
    pilot.atomic_create_json(destination, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-sha256", default=DEFAULT_MANIFEST_SHA256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pair-concurrency", type=int, choices=(1, 2),
                        default=1)
    parser.add_argument("--posthoc", action="store_true")
    parser.add_argument("--batch-receipt", type=Path)
    parser.add_argument(
        "--research-non-preregistered", action="store_true",
        help=("explicitly classify a non-default manifest as "
              "NON_PREREGISTERED_RESEARCH; never accepted by formal gates"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = validate_manifest(
        args.manifest, args.manifest_sha256,
        allow_non_preregistered=args.research_non_preregistered)
    out_root = validate_output_root(args.out)
    if args.posthoc:
        if args.batch_receipt is None:
            raise BatchEvidenceError("--posthoc requires --batch-receipt")
        result = run_posthoc(
            manifest, batch_receipt_path=args.batch_receipt.resolve(),
            out_root=out_root, resume=args.resume)
    else:
        if args.batch_receipt is not None:
            raise BatchEvidenceError("--batch-receipt is posthoc-only")
        result = run_gt_free(
            manifest, out_root=out_root, resume=args.resume,
            pair_concurrency=args.pair_concurrency)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
