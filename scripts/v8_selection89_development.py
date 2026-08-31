"""Freeze and execute the GT-free V8 selection89 development worker cache.

This module has deliberately narrow responsibilities:

* seal the existing B/selection inference caches in the canonical 89-pair order;
* run two outer repeats of five forward and five true-reverse workers exactly once;
* make the immutable worker records available to an offline V8 stage-order replay;
* never import labels, calibration90, fixed12, or official92.

The manifest is development evidence, not a preregistered confirmation set.  GT
evaluation belongs to a separate posthoc program after the worker receipt is
frozen.  The CLI defaults to ``--dry-run`` and requires an explicit flag before
starting the expensive worker batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_batch as v7_batch  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402


MANIFEST_SCHEMA = "v8-selection89-development-manifest-v1"
PLAN_SCHEMA = "v8-selection89-development-plan-v1"
OUTER_SCHEMA = "v8-selection89-worker-outer-v1"
PAIR_SCHEMA = "v8-selection89-worker-pair-v1"
BATCH_SCHEMA = "v8-selection89-worker-cache-receipt-v1"
EVIDENCE_CLASS = "NON_PREREGISTERED_DEVELOPMENT"
SPLIT = "selection89"
PAIR_COUNT = 89
OUTER_REPEATS = 2
DIRECTIONS = ("forward", "reverse")
REPLICATES_PER_DIRECTION = 5
WORKERS_PER_OUTER = len(DIRECTIONS) * REPLICATES_PER_DIRECTION
TOTAL_WORKERS = PAIR_COUNT * OUTER_REPEATS * WORKERS_PER_OUTER
PAIR_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_to_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_PAIRLIST = (
    CODE_ROOT / "outputs/official_sgaligner_migration_fix2_pairlists/selection.txt")
DEFAULT_CACHE_ROOT = pilot.DEFAULT_CACHE_ROOT
DEFAULT_DECISION_PROTOCOL = CODE_ROOT / "docs/V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md"
DEFAULT_MANIFEST_OUT = (
    CODE_ROOT / "outputs/v8_selection89_manifest_seal_v2_20260830/"
    "v8_selection89_manifest.json")
DEFAULT_PLAN_OUT = (
    CODE_ROOT / "outputs/v8_selection89_manifest_seal_v2_20260830/"
    "dry_run_plan.json")
EXCLUDED_V6_REPEAT_ROOT = pilot.FORMAL_ROOT / "selection/B"

TOP_KEYS = frozenset({
    "schema", "status", "evidence_class", "split", "pair_count", "pairs",
    "pair_ids_sha256", "pairlist", "cache_contract", "worker_contract",
    "decision_protocol", "gt_separation", "payload_sha256",
})
PAIR_KEYS = frozenset({
    "pair_id", "cache_basename", "cache_sha256", "cache_bytes",
    "input_sha256", "node_corr_count", "geot_ok_pairs", "geot_failed_pairs",
})
PAIRLIST_KEYS = frozenset({"path", "sha256", "bytes"})
CACHE_CONTRACT_KEYS = frozenset({
    "root", "schema", "checkpoint_id", "checkpoint_sha256", "cache_count",
})
WORKER_CONTRACT_KEYS = frozenset({
    "generator", "generator_protocol_sha256", "outer_repeats", "directions",
    "replicates_per_direction", "workers_per_outer", "total_workers", "reuse",
})
DECISION_PROTOCOL_KEYS = frozenset({"path", "sha256", "policy", "stage"})
GT_SEPARATION_KEYS = frozenset({
    "gt_free_workers", "labels_in_manifest", "evaluation",
    "confirmatory_claim_allowed", "forbidden_splits",
})
FORBIDDEN_MANIFEST_KEYS = frozenset({
    "gt", "gt_transform", "label", "labels", "rre", "rte", "strict",
    "relaxed", "accepted", "accepted_correct", "accepted_error",
    "calibration", "fixed12", "official92",
})


class Selection89EvidenceError(RuntimeError):
    """A frozen development input or worker-cache binding is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"))
            + "\n").encode()


def stable_json_hash(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def pair_ids_sha256(pair_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in pair_ids).encode()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(CODE_ROOT.resolve()))
    except ValueError:
        # Tests may supply an isolated fixture. Production inputs are all under
        # CODE_ROOT except the intentionally absolute immutable cache root.
        return str(resolved)


def _resolve_portable_path(value: str) -> Path:
    path = Path(value)
    return ((CODE_ROOT / path).resolve() if not path.is_absolute()
            else path.resolve())


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Selection89EvidenceError(
                f"refusing to overwrite frozen evidence: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Selection89EvidenceError(f"{where} is not a lowercase SHA-256")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                        where: str) -> None:
    if not isinstance(value, Mapping):
        raise Selection89EvidenceError(f"{where} must be an object")
    observed = frozenset(str(key) for key in value)
    if observed != expected:
        raise Selection89EvidenceError(
            f"{where} schema mismatch missing={sorted(expected-observed)} "
            f"unknown={sorted(observed-expected)}")
    forbidden = observed & FORBIDDEN_MANIFEST_KEYS
    if forbidden:
        raise Selection89EvidenceError(
            f"{where} leaks forbidden posthoc keys: {sorted(forbidden)}")


def read_pairlist(path: Path) -> list[str]:
    path = path.resolve()
    if path != DEFAULT_PAIRLIST.resolve() or not path.is_file():
        raise Selection89EvidenceError(
            "selection89 freezer requires the canonical committed pairlist")
    pair_ids = [row.strip() for row in path.read_text().splitlines()
                if row.strip()]
    if len(pair_ids) != PAIR_COUNT or len(set(pair_ids)) != PAIR_COUNT:
        raise Selection89EvidenceError(
            "canonical selection pairlist must contain 89 unique rows")
    for index, pair_id in enumerate(pair_ids):
        if PAIR_ID_RE.fullmatch(pair_id) is None:
            raise Selection89EvidenceError(
                f"unsafe selection pair id at row {index}")
    return pair_ids


def _cache_summary(cache: Mapping[str, Any], path: Path) -> dict[str, Any]:
    geot = cache.get("geot")
    if not isinstance(geot, Mapping):
        raise Selection89EvidenceError("cache geot payload is not a mapping")
    ok = sum(isinstance(row, Mapping) and row.get("status") == "ok"
             for row in geot.values())
    failed = len(geot) - ok
    return {
        "pair_id": cache["pair_id"],
        "cache_basename": path.name,
        "cache_sha256": cache["_file_sha256"],
        "cache_bytes": path.stat().st_size,
        "input_sha256": _require_sha(cache["input_sha256"], "input_sha256"),
        "node_corr_count": len(cache["_members"]),
        "geot_ok_pairs": ok,
        "geot_failed_pairs": failed,
    }


def build_manifest(
        *, pairlist: Path = DEFAULT_PAIRLIST,
        cache_root: Path = DEFAULT_CACHE_ROOT,
        decision_protocol: Path = DEFAULT_DECISION_PROTOCOL,
        cache_loader: Callable[[Path, str, str | None], dict] | None = None,
) -> dict[str, Any]:
    """Build a label-free manifest in memory; the caller freezes it once."""
    pair_ids = read_pairlist(pairlist)
    cache_root = cache_root.resolve()
    if cache_root != DEFAULT_CACHE_ROOT.resolve():
        raise Selection89EvidenceError("cache root must be immutable formal B/selection")
    decision_protocol = decision_protocol.resolve()
    if not decision_protocol.is_file():
        raise Selection89EvidenceError(
            f"missing frozen V8 decision protocol: {decision_protocol}")
    loader = cache_loader or pilot.load_validated_cache
    cache_files = sorted(cache_root.glob("*.pt"))
    if ({path.stem for path in cache_files} != set(pair_ids)
            or len(cache_files) != PAIR_COUNT):
        raise Selection89EvidenceError(
            "B/selection cache directory must exactly match the 89-pair list")
    pairs = []
    for pair_id in pair_ids:
        path = pilot.cache_path(cache_root, pair_id)
        before = sha256_file(path)
        cache = loader(path, pair_id, before)
        row = _cache_summary(cache, path)
        _require_exact_keys(row, PAIR_KEYS, f"pair[{pair_id}]")
        if sha256_file(path) != before:
            raise Selection89EvidenceError(f"cache changed while sealing {pair_id}")
        pairs.append(row)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "FROZEN",
        "evidence_class": EVIDENCE_CLASS,
        "split": SPLIT,
        "pair_count": PAIR_COUNT,
        "pairs": pairs,
        "pair_ids_sha256": pair_ids_sha256(pair_ids),
        "pairlist": {
            "path": _portable_path(pairlist),
            "sha256": sha256_file(pairlist),
            "bytes": pairlist.stat().st_size,
        },
        "cache_contract": {
            "root": str(cache_root),
            "schema": pilot.CACHE_SCHEMA,
            "checkpoint_id": pilot.CHECKPOINT_ID,
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
            "cache_count": PAIR_COUNT,
        },
        "worker_contract": {
            "generator": "scripts/v7_registration_pilot.py",
            "generator_protocol_sha256": pilot.protocol_sha256(),
            "outer_repeats": OUTER_REPEATS,
            "directions": list(DIRECTIONS),
            "replicates_per_direction": REPLICATES_PER_DIRECTION,
            "workers_per_outer": WORKERS_PER_OUTER,
            "total_workers": TOTAL_WORKERS,
            "reuse": "persist once; V8 policies replay offline without rerunning workers",
        },
        "decision_protocol": {
            "path": _portable_path(decision_protocol),
            "sha256": sha256_file(decision_protocol),
            "policy": "fixed q=4 rotation=5deg translation=0.10m",
            "stage": "final transforms cluster before medoid Rule-B",
        },
        "gt_separation": {
            "gt_free_workers": True,
            "labels_in_manifest": False,
            "evaluation": "separate posthoc process after batch receipt is frozen",
            "confirmatory_claim_allowed": False,
            "forbidden_splits": ["calibration90", "fixed12", "official92"],
        },
    }
    manifest["payload_sha256"] = stable_json_hash(manifest)
    _require_exact_keys(manifest, TOP_KEYS, "manifest")
    return manifest


def validate_manifest(path: Path, expected_sha256: str, *,
                      verify_caches: bool = True) -> dict[str, Any]:
    _require_sha(expected_sha256, "manifest file SHA")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise Selection89EvidenceError("manifest file SHA mismatch")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Selection89EvidenceError("manifest is not valid JSON") from exc
    _require_exact_keys(manifest, TOP_KEYS, "manifest")
    payload_sha = manifest.pop("payload_sha256", None)
    if payload_sha != stable_json_hash(manifest):
        raise Selection89EvidenceError("manifest payload SHA mismatch")
    manifest["payload_sha256"] = payload_sha
    if (manifest["schema"] != MANIFEST_SCHEMA
            or manifest["status"] != "FROZEN"
            or manifest["evidence_class"] != EVIDENCE_CLASS
            or manifest["split"] != SPLIT
            or manifest["pair_count"] != PAIR_COUNT):
        raise Selection89EvidenceError("manifest identity mismatch")
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
        raise Selection89EvidenceError("manifest pair count mismatch")
    pair_ids = []
    for index, row in enumerate(pairs):
        _require_exact_keys(row, PAIR_KEYS, f"pairs[{index}]")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or PAIR_ID_RE.fullmatch(pair_id) is None:
            raise Selection89EvidenceError(f"unsafe pair id at row {index}")
        _require_sha(row["cache_sha256"], f"pairs[{index}].cache_sha256")
        _require_sha(row["input_sha256"], f"pairs[{index}].input_sha256")
        pair_ids.append(pair_id)
    if (len(set(pair_ids)) != PAIR_COUNT
            or manifest["pair_ids_sha256"] != pair_ids_sha256(pair_ids)):
        raise Selection89EvidenceError("ordered pair-id binding mismatch")
    pairlist = manifest["pairlist"]
    _require_exact_keys(pairlist, PAIRLIST_KEYS, "pairlist")
    if (_resolve_portable_path(pairlist["path"]) != DEFAULT_PAIRLIST.resolve()
            or sha256_file(DEFAULT_PAIRLIST) != pairlist["sha256"]
            or read_pairlist(DEFAULT_PAIRLIST) != pair_ids):
        raise Selection89EvidenceError("canonical pairlist binding mismatch")
    cache_contract = manifest["cache_contract"]
    _require_exact_keys(
        cache_contract, CACHE_CONTRACT_KEYS, "cache_contract")
    if (Path(cache_contract["root"]).resolve() != DEFAULT_CACHE_ROOT.resolve()
            or cache_contract["schema"] != pilot.CACHE_SCHEMA
            or cache_contract["checkpoint_id"] != pilot.CHECKPOINT_ID
            or cache_contract["checkpoint_sha256"] != pilot.CHECKPOINT_SHA256
            or cache_contract["cache_count"] != PAIR_COUNT):
        raise Selection89EvidenceError("cache contract mismatch")
    worker = manifest["worker_contract"]
    _require_exact_keys(worker, WORKER_CONTRACT_KEYS, "worker_contract")
    expected_worker = {
        "outer_repeats": OUTER_REPEATS,
        "directions": list(DIRECTIONS),
        "replicates_per_direction": REPLICATES_PER_DIRECTION,
        "workers_per_outer": WORKERS_PER_OUTER,
        "total_workers": TOTAL_WORKERS,
    }
    if any(worker.get(key) != value for key, value in expected_worker.items()):
        raise Selection89EvidenceError("worker shape contract mismatch")
    if worker.get("generator_protocol_sha256") != pilot.protocol_sha256():
        raise Selection89EvidenceError("worker generator protocol drift")
    protocol = manifest["decision_protocol"]
    _require_exact_keys(
        protocol, DECISION_PROTOCOL_KEYS, "decision_protocol")
    protocol_path = _resolve_portable_path(protocol["path"])
    if (protocol_path != DEFAULT_DECISION_PROTOCOL.resolve()
            or not protocol_path.is_file()
            or sha256_file(protocol_path) != protocol["sha256"]):
        raise Selection89EvidenceError("V8 decision protocol drift")
    separation = manifest["gt_separation"]
    _require_exact_keys(
        separation, GT_SEPARATION_KEYS, "gt_separation")
    if (separation.get("gt_free_workers") is not True
            or separation.get("labels_in_manifest") is not False
            or separation.get("confirmatory_claim_allowed") is not False
            or separation.get("forbidden_splits")
            != ["calibration90", "fixed12", "official92"]):
        raise Selection89EvidenceError("GT separation contract mismatch")
    if verify_caches:
        observed = {path.stem for path in DEFAULT_CACHE_ROOT.glob("*.pt")}
        if observed != set(pair_ids):
            raise Selection89EvidenceError("cache directory membership drift")
        for row in pairs:
            cache_path = pilot.cache_path(DEFAULT_CACHE_ROOT, row["pair_id"])
            if (sha256_file(cache_path) != row["cache_sha256"]
                    or cache_path.stat().st_size != row["cache_bytes"]):
                raise Selection89EvidenceError(
                    f"cache drift {row['pair_id']}")
    manifest["_file_sha256"] = expected_sha256
    manifest["_path"] = str(path)
    return manifest


def excluded_prior_evidence() -> list[dict[str, Any]]:
    """Hash, but never parse, GT-tainted V6 repeats excluded from replay."""
    rows = []
    for repeat in range(3):
        path = EXCLUDED_V6_REPEAT_ROOT / f"repeat_{repeat:02d}.json"
        if path.is_file():
            rows.append({
                "path": str(path), "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "eligible_for_v8_worker_replay": False,
                "reason": (
                    "contains posthoc-labelled final path rows; no 5x forward/"
                    "reverse per-worker transforms"),
            })
    return rows


def build_dry_run_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in manifest["pairs"]]
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "DRY_RUN_COMPLETE",
        "evidence_class": EVIDENCE_CLASS,
        "manifest": {
            "path": manifest["_path"],
            "sha256": manifest["_file_sha256"],
            "payload_sha256": manifest["payload_sha256"],
        },
        "validated_inputs": {
            "pair_count": len(pair_ids),
            "cache_count": len(list(DEFAULT_CACHE_ROOT.glob("*.pt"))),
            "pair_ids_sha256": pair_ids_sha256(pair_ids),
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
        },
        "execution_shape": {
            "outer_repeats": OUTER_REPEATS,
            "directions": list(DIRECTIONS),
            "replicates_per_direction": REPLICATES_PER_DIRECTION,
            "workers_per_outer": WORKERS_PER_OUTER,
            "total_workers": TOTAL_WORKERS,
            "worker_generation": "GT-free; immutable JSON records",
            "offline_replay": (
                "all V8 policies consume the same frozen workers; no rerun"),
        },
        "stage_order": [
            "generate K=5 forward and K=5 reverse workers",
            "cluster all finite final ICP transforms per direction",
            "select observed medoid and apply unchanged Rule-B",
            "require forward/reverse final consensus q=4 r=5deg t=0.10m",
            "freeze GT-free receipt",
            "evaluate labels only in separate posthoc process",
        ],
        "prior_evidence": {
            "reusable": "89 immutable B/selection inference caches",
            "excluded_from_replay": excluded_prior_evidence(),
        },
        "authorization": {
            "selection89": "development only",
            "calibration90": False,
            "fixed12": False,
            "official92": False,
            "confirmatory_claim_allowed": False,
        },
    }
    plan["evidence_sha256"] = stable_json_hash(plan)
    return plan


def _source_snapshot() -> dict[str, Any]:
    paths = {
        "selection89_controller": Path(__file__).resolve(),
        "worker_generator": CODE_ROOT / "scripts/v7_registration_pilot.py",
        "v7_batch_validator": CODE_ROOT / "scripts/v7_registration_batch.py",
        "decision_protocol": DEFAULT_DECISION_PROTOCOL,
    }
    files = {}
    for name, path in paths.items():
        if not path.is_file():
            raise Selection89EvidenceError(f"missing source {path}")
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
        check=True, text=True, capture_output=True).stdout.strip()
    snapshot = {"head": head, "files": files}
    snapshot["snapshot_sha256"] = stable_json_hash(snapshot)
    return snapshot


def _validate_output_root(path: Path) -> Path:
    path = path.resolve()
    outputs = (CODE_ROOT / "outputs").resolve()
    try:
        relative = path.relative_to(outputs)
    except ValueError as exc:
        raise Selection89EvidenceError(
            "worker output must be a named child of repository outputs/") from exc
    if not relative.parts:
        raise Selection89EvidenceError("broad outputs/ root is forbidden")
    return path


def _outer_worker_names() -> set[str]:
    return {
        f"{direction}_{replicate:02d}.json"
        for direction in DIRECTIONS
        for replicate in range(REPLICATES_PER_DIRECTION)
    }


def _validate_worker(path: Path, *, pair: Mapping[str, Any], direction: str,
                     replicate: int) -> dict[str, Any]:
    return v7_batch._load_batch_worker(  # intentionally reuse hardened validator
        path, pair_id=pair["pair_id"], direction=direction,
        replicate=replicate, cache_sha=pair["cache_sha256"],
        protocol_sha=pilot.protocol_sha256())


def _validate_outer(path: Path, *, pair: Mapping[str, Any], outer: int,
                    manifest_sha: str, snapshot_sha: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    expected = data.pop("evidence_sha256", None)
    if expected != stable_json_hash(data):
        raise Selection89EvidenceError(f"outer receipt SHA mismatch {path}")
    data["evidence_sha256"] = expected
    if (data.get("schema") != OUTER_SCHEMA
            or data.get("status") != "GT_FREE_WORKERS_COMPLETE"
            or data.get("pair_id") != pair["pair_id"]
            or data.get("outer_repeat") != outer
            or data.get("manifest_sha256") != manifest_sha
            or data.get("source_snapshot_sha256") != snapshot_sha
            or data.get("posthoc_not_run") is not True
            or data.get("worker_count") != WORKERS_PER_OUTER):
        raise Selection89EvidenceError(f"outer receipt provenance mismatch {path}")
    expected_rows = [
        (direction, replicate) for direction in DIRECTIONS
        for replicate in range(REPLICATES_PER_DIRECTION)]
    if [(row.get("direction"), row.get("replicate"))
            for row in data.get("workers", [])] != expected_rows:
        raise Selection89EvidenceError(f"outer worker shape mismatch {path}")
    for row in data["workers"]:
        worker_path = Path(row["path"]).resolve()
        if sha256_file(worker_path) != row["sha256"]:
            raise Selection89EvidenceError(f"worker file SHA mismatch {worker_path}")
        worker = _validate_worker(
            worker_path, pair=pair, direction=row["direction"],
            replicate=row["replicate"])
        if worker["evidence_sha256"] != row["evidence_sha256"]:
            raise Selection89EvidenceError(
                f"worker evidence SHA mismatch {worker_path}")
    return data


def _run_outer(pair: Mapping[str, Any], *, outer: int, output_root: Path,
               manifest_sha: str, snapshot: Mapping[str, Any]) -> Path:
    run_dir = output_root / "pairs" / pair["pair_id"] / f"outer_{outer:02d}"
    receipt_path = run_dir / "worker_receipt.json"
    if run_dir.exists():
        observed = {path.name for path in run_dir.iterdir()}
        if observed != _outer_worker_names() | {"worker_receipt.json"}:
            raise Selection89EvidenceError(
                f"partial/unexpected worker outer cannot resume: {run_dir}")
        _validate_outer(
            receipt_path, pair=pair, outer=outer,
            manifest_sha=manifest_sha,
            snapshot_sha=snapshot["snapshot_sha256"])
        return receipt_path
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    worker_rows = []
    runner = (CODE_ROOT / "scripts/v7_registration_pilot.py").resolve()
    for direction in DIRECTIONS:
        for replicate in range(REPLICATES_PER_DIRECTION):
            output = run_dir / f"{direction}_{replicate:02d}.json"
            command = [
                str(Path(sys.executable).resolve()), str(runner), "--worker",
                "--pair", pair["pair_id"], "--direction", direction,
                "--replicate", str(replicate), "--cache-root",
                str(DEFAULT_CACHE_ROOT), "--cache-sha256", pair["cache_sha256"],
                "--protocol-sha256", pilot.protocol_sha256(),
                "--worker-out", str(output),
            ]
            completed = subprocess.run(
                command, cwd=CODE_ROOT, env=v7_batch._worker_environment(),
                text=True, capture_output=True)
            if completed.returncode != 0:
                raise Selection89EvidenceError(
                    f"worker failed {pair['pair_id']} {direction}/{replicate}: "
                    f"{completed.stderr[-4000:]}")
            worker = _validate_worker(
                output, pair=pair, direction=direction, replicate=replicate)
            worker_rows.append({
                "direction": direction, "replicate": replicate,
                "path": str(output), "sha256": sha256_file(output),
                "evidence_sha256": worker["evidence_sha256"],
            })
    receipt = {
        "schema": OUTER_SCHEMA,
        "status": "GT_FREE_WORKERS_COMPLETE",
        "pair_id": pair["pair_id"],
        "outer_repeat": outer,
        "manifest_sha256": manifest_sha,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "worker_count": WORKERS_PER_OUTER,
        "workers": worker_rows,
        "elapsed_wall_s": time.monotonic() - started,
        "posthoc_not_run": True,
    }
    receipt["evidence_sha256"] = stable_json_hash(receipt)
    _atomic_create_json(receipt_path, receipt)
    return receipt_path


def _run_pair(pair: Mapping[str, Any], *, output_root: Path,
              manifest_sha: str, snapshot: Mapping[str, Any]) -> Path:
    started = time.monotonic()
    outer_paths = [
        _run_outer(
            pair, outer=outer, output_root=output_root,
            manifest_sha=manifest_sha, snapshot=snapshot)
        for outer in range(OUTER_REPEATS)]
    destination = output_root / "pair_receipts" / f"{pair['pair_id']}.json"
    if destination.exists():
        return destination
    receipt = {
        "schema": PAIR_SCHEMA,
        "status": "GT_FREE_WORKERS_COMPLETE",
        "pair_id": pair["pair_id"],
        "outer_repeats": OUTER_REPEATS,
        "outers": [
            {"outer_repeat": outer, "path": str(path),
             "sha256": sha256_file(path)}
            for outer, path in enumerate(outer_paths)],
        "elapsed_wall_s": time.monotonic() - started,
        "posthoc_not_run": True,
    }
    receipt["evidence_sha256"] = stable_json_hash(receipt)
    _atomic_create_json(destination, receipt)
    return destination


def run_worker_batch(manifest: Mapping[str, Any], *, output_root: Path,
                     pair_concurrency: int) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    if output_root.exists():
        raise Selection89EvidenceError(
            "worker batch requires a fresh output root; old evidence is immutable")
    repository = v7_batch.repository_state(output_root)
    if repository["head"] != subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
            check=True, text=True, capture_output=True).stdout.strip():
        raise Selection89EvidenceError("repository HEAD changed during preflight")
    snapshot = _source_snapshot()
    output_root.mkdir(parents=True)
    started = time.monotonic()
    kwargs = {
        "output_root": output_root,
        "manifest_sha": manifest["_file_sha256"],
        "snapshot": snapshot,
    }
    if pair_concurrency == 1:
        paths = [_run_pair(pair, **kwargs) for pair in manifest["pairs"]]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=pair_concurrency) as executor:
            futures = [executor.submit(_run_pair, pair, **kwargs)
                       for pair in manifest["pairs"]]
            paths = [future.result() for future in futures]
    receipt = {
        "schema": BATCH_SCHEMA,
        "status": "GT_FREE_WORKERS_COMPLETE",
        "evidence_class": EVIDENCE_CLASS,
        "manifest": {"path": manifest["_path"],
                     "sha256": manifest["_file_sha256"]},
        "source_snapshot": snapshot,
        "pair_count": PAIR_COUNT,
        "outer_repeats_per_pair": OUTER_REPEATS,
        "workers_per_outer": WORKERS_PER_OUTER,
        "total_workers": TOTAL_WORKERS,
        "pair_receipts": [
            {"pair_id": pair["pair_id"], "path": str(path),
             "sha256": sha256_file(path)}
            for pair, path in zip(manifest["pairs"], paths)],
        "posthoc_not_run": True,
        "policy_not_applied": True,
        "elapsed_wall_s": time.monotonic() - started,
    }
    receipt["evidence_sha256"] = stable_json_hash(receipt)
    destination = output_root / "gt_free_worker_batch_receipt.json"
    _atomic_create_json(destination, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-manifest", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-workers", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_DECISION_PROTOCOL)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pair-concurrency", type=int, choices=(1, 2, 4),
                        default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze_manifest:
        if args.manifest_sha256 is not None:
            raise Selection89EvidenceError(
                "--manifest-sha256 is validation-only")
        manifest = build_manifest(decision_protocol=args.protocol)
        _atomic_create_json(args.manifest.resolve(), manifest)
        result = {
            "manifest": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest.resolve()),
            "pair_count": PAIR_COUNT,
            "evidence_class": EVIDENCE_CLASS,
        }
    else:
        if args.manifest_sha256 is None:
            raise Selection89EvidenceError(
                "--dry-run/--execute-workers require --manifest-sha256")
        manifest = validate_manifest(
            args.manifest, args.manifest_sha256, verify_caches=True)
        if args.dry_run:
            result = build_dry_run_plan(manifest)
            _atomic_create_json(args.plan_out.resolve(), result)
        else:
            if args.out is None:
                raise Selection89EvidenceError("--execute-workers requires --out")
            result = run_worker_batch(
                manifest, output_root=args.out,
                pair_concurrency=args.pair_concurrency)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
