"""Locked, label-free infrastructure for the one-shot V8 calibration90 gate.

This module deliberately does not import a GT loader.  It freezes the unique
selection89 winner and immutable B/calibration caches, then can generate one
fresh 5-forward + 5-reverse worker batch twice per pair.  A durable claim file
is created before the first worker and is never removed, even after failure.
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
from typing import Any, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v7_registration_batch as v7_batch  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402
from safety import decision_features  # noqa: E402
from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config, evaluate_stage_order,
)

MANIFEST_SCHEMA = "v8-calibration90-locked-manifest-v1"
WINNER_SCHEMA = "v8-selection89-winner-freeze-v1"
OUTER_SCHEMA = "v8-calibration90-outer-v1"
PAIR_SCHEMA = "v8-calibration90-pair-v1"
BATCH_SCHEMA = "v8-calibration90-gt-free-batch-v1"
AUDIT_SCHEMA = "v8-calibration90-input-audit-v1"
CACHE_PREPARE_RECEIPT_SCHEMA = "v8-calibration90-B-cache-receipt-v1"
EVIDENCE_CLASS = "LOCKED_FIRST_V8_GATE"
SPLIT = "calibration90"
PAIR_COUNT = 90
OUTER_REPEATS = 2
DIRECTIONS = ("forward", "reverse")
REPLICATES = 5
TOTAL_WORKERS = PAIR_COUNT * OUTER_REPEATS * len(DIRECTIONS) * REPLICATES
CONFIG = V8Config(repeats=5, quorum=4, max_rotation_deg=5.0,
                  max_translation_m=0.10)
PAIR_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_to_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_PAIRLIST = (CODE_ROOT / "outputs/official_sgaligner_migration_"
                    "fix2_pairlists/calibration.txt")
CANONICAL_PAIRLIST_SHA256 = (
    "0d82256933c9e6dbac55a4ccc85a902ec49941f5dc5b600f3ff9150ef51a88fd")
DEFAULT_LEGACY_CACHE = (CODE_ROOT / "outputs/official_sgaligner_v3_pct_"
                        "parity_baseline_20260827/final_inference_cache/"
                        "calibration90")
OFFICIAL_EPOCH6_SHA256 = (
    "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e")
DEFAULT_CLAIM_ROOT = Path(
    "/home/aidenwu/.cache/sgaligner_v8_calibration90_single_use")
PROTOCOL = CODE_ROOT / "docs/V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md"
GATE_PROTOCOL = CODE_ROOT / "docs/V8_CALIBRATION90_LOCKED_GATE.md"

# Frozen before labels are opened.  These are the historical sealed Rule-A
# calibration floor (6 strict / 8 relaxed / 5 correct accepted) plus zero-error,
# full accounting and repeatability requirements.  No CLI can change them.
THRESHOLDS = {
    "completed": 90,
    "strict_min": 6,
    "relaxed_min": 8,
    "accepted_correct_min": 5,
    "accepted_error_max": 0,
    "repeatable_pairs": 90,
    "exceptions_max": 0,
    "nonfinite_max": 0,
    "cache_mismatches_max": 0,
}

FORBIDDEN_KEYS = frozenset({
    "gt", "gt_transform", "label", "labels", "rre", "rte", "strict",
    "relaxed", "accepted", "accepted_correct", "accepted_error",
    "fixed12", "official92",
})


class Calibration90Error(RuntimeError):
    """A pre-registration, provenance or single-use invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return pilot.stable_json_hash(value)


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise Calibration90Error(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Calibration90Error(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Calibration90Error(f"invalid JSON {path}") from exc
    if not isinstance(value, dict):
        raise Calibration90Error(f"JSON object required {path}")
    return value


def read_pairlist(path: Path = DEFAULT_PAIRLIST) -> list[str]:
    path = path.resolve()
    if path != DEFAULT_PAIRLIST.resolve() or not path.is_file():
        raise Calibration90Error("only canonical calibration90 pairlist allowed")
    if sha256_file(path) != CANONICAL_PAIRLIST_SHA256:
        raise Calibration90Error("canonical calibration90 pairlist SHA drift")
    pair_ids = [line.strip() for line in path.read_text().splitlines()
                if line.strip()]
    if (len(pair_ids) != PAIR_COUNT or len(set(pair_ids)) != PAIR_COUNT
            or any(PAIR_ID_RE.fullmatch(pair_id) is None
                   for pair_id in pair_ids)):
        raise Calibration90Error("calibration90 pairlist must be 90 unique IDs")
    return pair_ids


def pair_ids_sha256(pair_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in pair_ids).encode()).hexdigest()


def audit_legacy_cache(root: Path = DEFAULT_LEGACY_CACHE) -> dict[str, Any]:
    """Audit only GT-free metadata; never load arrays or posthoc labels."""
    root = root.resolve()
    pairs = read_pairlist()
    rows = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = directory / "pair_cache.json"
        if not metadata.is_file():
            continue
        value = _payload(metadata)
        allowed = {"pair_id", "checkpoint_sha256", "status", "cache_key"}
        row = {key: value.get(key) for key in allowed}
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or PAIR_ID_RE.fullmatch(pair_id) is None:
            raise Calibration90Error("legacy cache has unsafe pair identity")
        files = []
        for name in ("pair_cache.json", "input_tensors.npz",
                     "embeddings.npz", "geot_corrs.npz"):
            path = directory / name
            if not path.is_file():
                raise Calibration90Error(f"legacy cache missing {path}")
            files.append({"name": name, "sha256": sha256_file(path),
                          "bytes": path.stat().st_size})
        rows.append({"pair_id": pair_id,
                     "checkpoint_sha256": row["checkpoint_sha256"],
                     "files": files})
    identities = [row["pair_id"] for row in rows]
    inventory = stable_hash(rows)
    compatible = (
        identities == pairs
        and all(row["checkpoint_sha256"] == pilot.CHECKPOINT_SHA256
                for row in rows))
    return {
        "schema": AUDIT_SCHEMA,
        "pair_count": len(rows),
        "ordered_pair_ids_match": identities == pairs,
        "pair_ids_sha256": pair_ids_sha256(identities),
        "inventory_sha256": inventory,
        "checkpoint_sha256_values": sorted(set(
            str(row["checkpoint_sha256"]) for row in rows)),
        "eligible_as_v8_B_worker_cache": compatible,
        "finding": ("READY" if compatible else
                    "NOT_READY: legacy 90-pair cache is official epoch-6, "
                    "not the frozen SGF-domain B checkpoint cache"),
        "arrays_or_labels_opened": False,
    }


def validate_winner(path: Path, expected_sha: str) -> dict[str, Any]:
    _require_sha(expected_sha, "winner receipt SHA")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise Calibration90Error("selection89 winner receipt SHA mismatch")
    value = _payload(path)
    evidence = value.pop("evidence_sha256", None)
    if evidence != stable_hash(value):
        raise Calibration90Error("winner embedded evidence SHA mismatch")
    value["evidence_sha256"] = evidence
    if (value.get("schema") != WINNER_SCHEMA
            or value.get("status") != "FROZEN_UNIQUE_WINNER"
            or value.get("split") != "selection89"
            or value.get("candidate_id")
            != "V8_FINAL_FIRST_Q4_R5_T0.10_FIXED_TRACE"
            or value.get("checkpoint_sha256") != pilot.CHECKPOINT_SHA256
            or value.get("config") != {
                "repeats": 5, "quorum": 4,
                "max_rotation_deg": 5.0, "max_translation_m": 0.10}
            or value.get("selection89_gate_passed") is not True
            or value.get("unique_winner") is not True):
        raise Calibration90Error("selection89 winner identity/gate mismatch")
    for field in ("selection_manifest_sha256", "worker_batch_sha256",
                  "posthoc_sha256", "cache_inventory_sha256",
                  "code_inventory_sha256"):
        _require_sha(value.get(field), f"winner.{field}")
    if set(value) & FORBIDDEN_KEYS:
        raise Calibration90Error("winner leaks calibration/fixed/official fields")
    return value


def _cache_inventory(cache_root: Path, pair_ids: list[str]) -> tuple[
        list[dict[str, Any]], str]:
    root = cache_root.resolve()
    paths = sorted(root.glob("*.pt")) if root.is_dir() else []
    if len(paths) != PAIR_COUNT or {path.stem for path in paths} != set(pair_ids):
        raise Calibration90Error(
            "B/calibration cache root must contain exactly the 90 pair caches")
    rows = []
    # Deliberately validate the GT-free cache schema only.
    for pair_id in pair_ids:
        path = root / f"{pair_id}.pt"
        before = sha256_file(path)
        cache = pilot.load_validated_cache(path, pair_id, before)
        if cache["checkpoint_sha256"] != pilot.CHECKPOINT_SHA256:
            raise Calibration90Error("calibration cache checkpoint drift")
        rows.append({
            "pair_id": pair_id, "cache_sha256": before,
            "cache_bytes": path.stat().st_size,
            "input_sha256": cache["input_sha256"],
            "node_corr_count": len(cache["_members"]),
        })
        if sha256_file(path) != before:
            raise Calibration90Error("cache changed during freeze")
    return rows, stable_hash(rows)


def _cache_preparation_receipt(
        cache_root: Path, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify the Gate-0 receipt without importing its builder module.

    The receipt is deliberately a JSON hand-off: the controller never imports
    the cache-preparation program and therefore cannot accidentally gain an
    inference or label-loading side path.
    """
    path = cache_root.resolve() / "cache_receipt.json"
    value = _payload(path)
    evidence = value.pop("evidence_sha256", None)
    if evidence != stable_hash(value):
        raise Calibration90Error("B/calibration cache receipt evidence drift")
    value["evidence_sha256"] = evidence
    pair_ids = [row["pair_id"] for row in pairs]
    receipt_pairs = value.get("pairs", [])
    if (value.get("schema") != CACHE_PREPARE_RECEIPT_SCHEMA
            or value.get("status") != "GT_FREE_B_CACHE_COMPLETE"
            or value.get("split") != SPLIT
            or value.get("checkpoint_sha256") != pilot.CHECKPOINT_SHA256
            or value.get("pair_count") != PAIR_COUNT
            or value.get("pair_ids_sha256") != pair_ids_sha256(pair_ids)
            or value.get("gt_ast_audit", {}).get("status") != "PASS"
            or value.get("labels_loaded") is not False
            or value.get("workers_run") is not False
            or value.get("posthoc_run") is not False
            or [row.get("pair_id") for row in receipt_pairs] != pair_ids):
        raise Calibration90Error("B/calibration cache receipt contract mismatch")
    by_pair = {row["pair_id"]: row for row in receipt_pairs}
    for frozen in pairs:
        prepared = by_pair[frozen["pair_id"]]
        for field in ("cache_sha256", "cache_bytes", "input_sha256",
                      "node_corr_count"):
            if prepared.get(field) != frozen[field]:
                raise Calibration90Error(
                    f"B/calibration cache receipt mismatch {field}")
        for field in ("embedding_sha256", "similarity_sha256"):
            _require_sha(prepared.get(field), f"cache receipt {field}")
    plan = value.get("plan", {})
    for field in ("file_sha256", "payload_sha256"):
        _require_sha(plan.get(field), f"cache plan {field}")
    if not isinstance(plan.get("path"), str):
        raise Calibration90Error("cache plan path missing")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "evidence_sha256": evidence,
        "plan": dict(plan),
        "cache_inventory_sha256": value.get("cache_inventory_sha256"),
    }


def freeze_manifest(*, winner_path: Path, winner_sha256: str,
                    cache_root: Path) -> dict[str, Any]:
    winner = validate_winner(winner_path, winner_sha256)
    pair_ids = read_pairlist()
    pairs, inventory_sha = _cache_inventory(cache_root, pair_ids)
    preparation = _cache_preparation_receipt(cache_root, pairs)
    source_files = {}
    for name, path in {
        "calibration_controller": Path(__file__).resolve(),
        "calibration_worker": (
            CODE_ROOT / "scripts/v8_calibration90_worker.py"),
        "calibration_posthoc": (
            CODE_ROOT / "scripts/v8_calibration90_posthoc_gate.py"),
        "v8_core": CODE_ROOT / "src/safety/v8_stage_order_consensus.py",
        "worker_core": CODE_ROOT / "scripts/v7_registration_pilot.py",
        "rule_b": CODE_ROOT / "src/safety/decision_features.py",
        "protocol": PROTOCOL,
        "gate_protocol": GATE_PROTOCOL,
    }.items():
        if not path.is_file():
            raise Calibration90Error(f"missing frozen source {path}")
        source_files[name] = {"path": str(path.relative_to(CODE_ROOT)),
                              "sha256": sha256_file(path)}
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "FROZEN",
        "evidence_class": EVIDENCE_CLASS,
        "split": SPLIT,
        "pair_count": PAIR_COUNT,
        "pair_ids_sha256": pair_ids_sha256(pair_ids),
        "pairlist": {"path": str(DEFAULT_PAIRLIST.relative_to(CODE_ROOT)),
                     "sha256": CANONICAL_PAIRLIST_SHA256,
                     "bytes": DEFAULT_PAIRLIST.stat().st_size},
        "pairs": pairs,
        "cache_contract": {
            "root": str(cache_root.resolve()),
            "schema": pilot.CACHE_SCHEMA,
            "checkpoint_id": pilot.CHECKPOINT_ID,
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
            "inventory_sha256": inventory_sha,
            "preparation_receipt": preparation,
        },
        "selection89_winner": {
            "path": str(winner_path.resolve()),
            "file_sha256": winner_sha256,
            "evidence_sha256": winner["evidence_sha256"],
            "candidate_id": winner["candidate_id"],
            "selection_cache_inventory_sha256": winner[
                "cache_inventory_sha256"],
            "selection_code_inventory_sha256": winner[
                "code_inventory_sha256"],
        },
        "config": dict(winner["config"]),
        "source_files": source_files,
        "worker_contract": {
            "outer_repeats": OUTER_REPEATS,
            "directions": list(DIRECTIONS),
            "replicates_per_direction": REPLICATES,
            "total_workers": TOTAL_WORKERS,
            "fixed_trace_required": True,
            "raw_consensus_decisional": False,
        },
        "thresholds": dict(THRESHOLDS),
        "single_use": {
            "claim_root": str(DEFAULT_CLAIM_ROOT),
            "claim_before_worker": True,
            "claim_removed_on_failure": False,
            "rerun_allowed": False,
        },
        "gt_separation": {
            "labels_loaded_by_this_module": False,
            "posthoc_program": "scripts/v8_calibration90_posthoc_gate.py",
            "fixed12_authorized": False,
            "official92_authorized": False,
        },
    }
    manifest["payload_sha256"] = stable_hash(manifest)
    return manifest


def validate_manifest(path: Path, expected_sha: str,
                      *, verify_caches: bool = True) -> dict[str, Any]:
    _require_sha(expected_sha, "manifest SHA")
    path = path.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise Calibration90Error("manifest file SHA mismatch")
    value = _payload(path)
    payload_sha = value.pop("payload_sha256", None)
    if payload_sha != stable_hash(value):
        raise Calibration90Error("manifest payload SHA mismatch")
    value["payload_sha256"] = payload_sha
    if (value.get("schema") != MANIFEST_SCHEMA
            or value.get("status") != "FROZEN"
            or value.get("evidence_class") != EVIDENCE_CLASS
            or value.get("split") != SPLIT
            or value.get("pair_count") != PAIR_COUNT
            or value.get("config") != {
                "repeats": 5, "quorum": 4,
                "max_rotation_deg": 5.0, "max_translation_m": 0.10}
            or value.get("thresholds") != THRESHOLDS):
        raise Calibration90Error("manifest identity/config/threshold drift")
    if set(value) & FORBIDDEN_KEYS:
        raise Calibration90Error("manifest leaks posthoc fields")
    pair_ids = read_pairlist()
    if ([row.get("pair_id") for row in value.get("pairs", [])] != pair_ids
            or value.get("pair_ids_sha256") != pair_ids_sha256(pair_ids)):
        raise Calibration90Error("manifest pair binding mismatch")
    winner_row = value.get("selection89_winner", {})
    validate_winner(Path(winner_row["path"]), winner_row["file_sha256"])
    for name, row in value.get("source_files", {}).items():
        source = CODE_ROOT / row["path"]
        if not source.is_file() or sha256_file(source) != row["sha256"]:
            raise Calibration90Error(f"source drift {name}")
    if verify_caches:
        rows, inventory = _cache_inventory(
            Path(value["cache_contract"]["root"]), pair_ids)
        if (rows != value["pairs"]
                or inventory != value["cache_contract"]["inventory_sha256"]):
            raise Calibration90Error("calibration cache inventory drift")
        preparation = _cache_preparation_receipt(
            Path(value["cache_contract"]["root"]), rows)
        if preparation != value["cache_contract"].get("preparation_receipt"):
            raise Calibration90Error("cache preparation receipt drift")
    value["_path"] = str(path)
    value["_file_sha256"] = expected_sha
    return value


def claim_single_use(manifest: Mapping[str, Any], *,
                     claim_root: Path = DEFAULT_CLAIM_ROOT) -> Path:
    if claim_root.resolve() != Path(
            manifest["single_use"]["claim_root"]).resolve():
        raise Calibration90Error("single-use claim root differs from manifest")
    claim_root.mkdir(parents=True, exist_ok=True)
    destination = claim_root / f"{manifest['_file_sha256']}.started.json"
    value = {
        "schema": "v8-calibration90-single-use-claim-v1",
        "status": "CONSUMED",
        "manifest_path": manifest["_path"],
        "manifest_sha256": manifest["_file_sha256"],
        "selection_winner_sha256": manifest["selection89_winner"][
            "file_sha256"],
        "rerun_allowed": False,
    }
    _atomic_create(destination, value)
    return destination


def _worker(path: Path, *, pair: Mapping[str, Any], direction: str,
            replicate: int, cache_root: Path) -> dict[str, Any]:
    command = [
        str(Path(sys.executable).resolve()),
        str((CODE_ROOT / "scripts/v8_calibration90_worker.py").resolve()),
        "--pair", pair["pair_id"], "--direction", direction,
        "--replicate", str(replicate), "--cache-root", str(cache_root),
        "--cache-sha256", pair["cache_sha256"],
        "--protocol-sha256", pilot.protocol_sha256(),
        "--worker-out", str(path),
    ]
    completed = subprocess.run(
        command, cwd=CODE_ROOT, env=v7_batch._worker_environment(),
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise Calibration90Error(
            f"worker failed {pair['pair_id']} {direction}/{replicate}: "
            f"{completed.stderr[-4000:]}")
    return v7_batch._load_batch_worker(
        path, pair_id=pair["pair_id"], direction=direction,
        replicate=replicate, cache_sha=pair["cache_sha256"],
        protocol_sha=pilot.protocol_sha256())


def _run_outer(pair: Mapping[str, Any], *, outer: int, out: Path,
               manifest: Mapping[str, Any]) -> Path:
    root = out / "pairs" / pair["pair_id"] / f"outer_{outer:02d}"
    if root.exists():
        raise Calibration90Error(f"refusing existing outer {root}")
    root.mkdir(parents=True)
    workers = []
    bindings = []
    for direction in DIRECTIONS:
        for replicate in range(REPLICATES):
            path = root / f"{direction}_{replicate:02d}.json"
            row = _worker(
                path, pair=pair, direction=direction, replicate=replicate,
                cache_root=Path(manifest["cache_contract"]["root"]))
            workers.append(row)
            bindings.append({
                "direction": direction, "replicate": replicate,
                "path": str(path), "file_sha256": sha256_file(path),
                "evidence_sha256": row["evidence_sha256"],
            })
    result = evaluate_stage_order(
        workers, CONFIG, decision_features.evaluate_rule_b,
        require_fixed_trace=True)
    receipt = {
        "schema": OUTER_SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "pair_id": pair["pair_id"],
        "outer_repeat": outer,
        "manifest_sha256": manifest["_file_sha256"],
        "cache_sha256": pair["cache_sha256"],
        "worker_count": 10,
        "workers": bindings,
        "v8_result": result,
        "posthoc_not_run": True,
    }
    receipt["evidence_sha256"] = stable_hash(receipt)
    destination = root / "gt_free_outer_receipt.json"
    _atomic_create(destination, receipt)
    return destination


def _run_pair(pair: Mapping[str, Any], *, out: Path,
              manifest: Mapping[str, Any]) -> Path:
    outer_paths = [_run_outer(pair, outer=outer, out=out,
                              manifest=manifest)
                   for outer in range(OUTER_REPEATS)]
    outcomes = []
    for path in outer_paths:
        row = _payload(path)
        outcomes.append(bool(row["v8_result"]["usable_for_reconstruction"]))
    receipt = {
        "schema": PAIR_SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "pair_id": pair["pair_id"],
        "outers": [{"outer_repeat": index, "path": str(path),
                    "file_sha256": sha256_file(path)}
                   for index, path in enumerate(outer_paths)],
        "outer_usable": outcomes,
        "repeatable": outcomes[0] == outcomes[1],
        "posthoc_not_run": True,
    }
    receipt["evidence_sha256"] = stable_hash(receipt)
    destination = out / "pair_receipts" / f"{pair['pair_id']}.json"
    _atomic_create(destination, receipt)
    return destination


def run_batch(manifest: Mapping[str, Any], *, out: Path,
              pair_concurrency: int = 1) -> dict[str, Any]:
    out = out.resolve()
    if out.exists():
        raise Calibration90Error("calibration batch requires a fresh output root")
    claim = claim_single_use(manifest)
    repository = v7_batch.repository_state(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    kwargs = {"out": out, "manifest": manifest}
    if pair_concurrency == 1:
        paths = [_run_pair(pair, **kwargs) for pair in manifest["pairs"]]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=pair_concurrency) as executor:
            futures = [executor.submit(_run_pair, pair, **kwargs)
                       for pair in manifest["pairs"]]
            paths = [future.result() for future in futures]
    pairs = [_payload(path) for path in paths]
    receipt = {
        "schema": BATCH_SCHEMA,
        "status": "GT_FREE_COMPLETE",
        "evidence_class": EVIDENCE_CLASS,
        "manifest": {"path": manifest["_path"],
                     "file_sha256": manifest["_file_sha256"],
                     "payload_sha256": manifest["payload_sha256"]},
        "single_use_claim": {"path": str(claim),
                             "file_sha256": sha256_file(claim)},
        "repository": repository,
        "pair_count": len(pairs),
        "total_workers": TOTAL_WORKERS,
        "pair_receipts": [
            {"pair_id": row["pair_id"], "path": str(path),
             "file_sha256": sha256_file(path)}
            for row, path in zip(pairs, paths)],
        "repeatable_pairs": sum(row["repeatable"] for row in pairs),
        "global_fail_closed_counts": {
            "exceptions": 0, "nonfinite": 0, "cache_mismatches": 0},
        "posthoc_not_run": True,
        "fixed12_not_run": True,
        "official92_not_run": True,
        "elapsed_wall_s": time.monotonic() - started,
    }
    receipt["evidence_sha256"] = stable_hash(receipt)
    _atomic_create(out / "gt_free_batch_receipt.json", receipt)
    return receipt


def validate_batch_receipt(path: Path, manifest: Mapping[str, Any]) \
        -> dict[str, Any]:
    value = _payload(path.resolve())
    evidence = value.pop("evidence_sha256", None)
    if evidence != stable_hash(value):
        raise Calibration90Error("batch embedded evidence SHA mismatch")
    value["evidence_sha256"] = evidence
    if (value.get("schema") != BATCH_SCHEMA
            or value.get("status") != "GT_FREE_COMPLETE"
            or value.get("pair_count") != PAIR_COUNT
            or value.get("total_workers") != TOTAL_WORKERS
            or value.get("repeatable_pairs") != PAIR_COUNT
            or value.get("global_fail_closed_counts") != {
                "exceptions": 0, "nonfinite": 0, "cache_mismatches": 0}
            or value.get("posthoc_not_run") is not True
            or value.get("manifest", {}).get("file_sha256")
            != manifest["_file_sha256"]):
        raise Calibration90Error("batch completeness/provenance gate failed")
    expected_ids = [row["pair_id"] for row in manifest["pairs"]]
    if [row.get("pair_id") for row in value.get("pair_receipts", [])] \
            != expected_ids:
        raise Calibration90Error("batch pair order mismatch")
    for row in value["pair_receipts"]:
        receipt = Path(row["path"]).resolve()
        if sha256_file(receipt) != row["file_sha256"]:
            raise Calibration90Error("pair receipt hash mismatch")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-inputs", action="store_true")
    mode.add_argument("--freeze-manifest", action="store_true")
    mode.add_argument("--execute-workers", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--winner", type=Path)
    parser.add_argument("--winner-sha256")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--pair-concurrency", type=int, choices=(1, 2, 4),
                        default=1)
    args = parser.parse_args()
    if args.audit_inputs:
        result = audit_legacy_cache()
    elif args.freeze_manifest:
        if None in (args.winner, args.winner_sha256, args.cache_root):
            raise Calibration90Error(
                "freeze requires winner SHA and prepared B/calibration cache")
        result = freeze_manifest(
            winner_path=args.winner, winner_sha256=args.winner_sha256,
            cache_root=args.cache_root)
    else:
        if args.manifest is None or args.manifest_sha256 is None:
            raise Calibration90Error("execute requires frozen manifest SHA")
        manifest = validate_manifest(args.manifest, args.manifest_sha256)
        result = run_batch(
            manifest, out=args.out,
            pair_concurrency=args.pair_concurrency)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result["evidence_sha256"] = stable_hash(result)
    _atomic_create(args.out.resolve(), result)
    print(json.dumps({"out": str(args.out.resolve()),
                      "file_sha256": sha256_file(args.out.resolve()),
                      "evidence_sha256": result["evidence_sha256"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
