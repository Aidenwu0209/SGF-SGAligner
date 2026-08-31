#!/usr/bin/env python3
"""Fail-closed Gate-1/Gate-2 verifier and selection freezer for V6-Fix.

This program is deliberately independent from ``v6fix_consistency_audit.py``.
It never imports or runs SGAligner, GeoTransformer, RANSAC, ICP, or the formal
runner.  Its only experiment inputs are the nine immutable formal-v2 selection
JSON files (A/B/D x repeats 0/1/2) and the pre-registered protocol JSON.

Node metrics are recomputed only from three independently generated sidecars
(``A.json``, ``B.json``, ``D.json``).  Every sidecar pair carries::

    "node_evidence": {
      "tp": int, "predicted": int, "anchors": int,
      "top1_hits": int, "top1_total": int,
      "top5_hits": int, "top5_total": int
    }

``top5_total`` must equal ``anchors``.  The sidecar SHA, cache manifest and GT
anchor manifest are bound into the freeze.  Missing or malformed evidence produces
an explicit BLOCKED result on stdout and no ``frozen_selection.json``.  This is
intentional: registration outcomes and post-hoc RRE/RTE cannot be used to guess
matcher precision/recall.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


RESULT_SCHEMA = "v6fix-consistency-audit-v2"
FREEZE_SCHEMA = "v6fix-selection-freeze-v1"
NODE_SIDECAR_SCHEMA = "v6fix-node-evidence-v1"
CHECKPOINTS = ("A", "B", "D")
REPEATS = (0, 1, 2)
PATHS = ("F", "C0", "C1")
EXPECTED_PAIRS = 89
NODE_FIELDS = (
    "tp", "predicted", "anchors", "top1_hits", "top1_total",
    "top5_hits", "top5_total",
)
COUNT_FIELDS = (
    "requested", "completed", "raw_strict", "raw_relaxed",
    "accepted_correct", "accepted_error", "rejected", "zero_candidate",
    "failed",
)


class Blocked(RuntimeError):
    """A fail-closed protocol/evidence failure."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    _need(path.is_file(), f"missing input JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Blocked(f"invalid JSON {path}: {exc}") from exc
    _need(isinstance(value, dict), f"top-level JSON object required: {path}")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _need(type(value) is int, f"{label} must be an integer")
    _need(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    _need(type(value) is bool, f"{label} must be boolean")
    return value


def _median(values: Iterable[float]) -> float:
    values = list(values)
    _need(len(values) == 3, "exactly three values are required for a median")
    return float(statistics.median(values))


def _pair_manifest_sha(pair_ids: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(pair_ids) + "\n").encode("utf-8"))


def _recompute_counts(rows: Sequence[Mapping[str, Any]], path: str,
                      label: str) -> Dict[str, int]:
    requested = len(rows)
    completed = raw_strict = raw_relaxed = 0
    accepted_correct = accepted_error = rejected = zero = failed = 0
    for index, row in enumerate(rows):
        prefix = f"{label}.rows[{index}]"
        audit = row.get("audit")
        paths = row.get("paths")
        _need(isinstance(audit, dict), f"{prefix}.audit must be an object")
        _need(isinstance(paths, dict) and path in paths,
              f"{prefix}.paths.{path} missing")
        item = paths[path]
        _need(isinstance(item, dict), f"{prefix}.paths.{path} must be object")
        is_zero = _exact_bool(audit.get("zero_candidate"),
                              f"{prefix}.audit.zero_candidate")
        valid = _exact_bool(item.get("valid"), f"{prefix}.{path}.valid")
        strict = _exact_bool(item.get("strict", False),
                             f"{prefix}.{path}.strict")
        relaxed = _exact_bool(item.get("relaxed", False),
                              f"{prefix}.{path}.relaxed")
        accepted = _exact_bool(item.get("accepted", False),
                               f"{prefix}.{path}.accepted")
        correct = _exact_bool(item.get("accepted_correct", False),
                              f"{prefix}.{path}.accepted_correct")
        wrong = _exact_bool(item.get("accepted_error", False),
                            f"{prefix}.{path}.accepted_error")
        _need(correct == (accepted and strict),
              f"{prefix}.{path} accepted_correct is inconsistent")
        _need(wrong == (accepted and not strict),
              f"{prefix}.{path} accepted_error is inconsistent")
        _need(not strict or relaxed,
              f"{prefix}.{path} strict must imply relaxed")
        _need(valid or not (strict or relaxed or accepted),
              f"{prefix}.{path} invalid result cannot carry an outcome")
        _need(not (valid and is_zero),
              f"{prefix} cannot be both valid and zero_candidate")
        if valid:
            decision = item.get("decision")
            _need(isinstance(decision, dict),
                  f"{prefix}.{path}.decision missing for valid result")
            _need(decision.get("usable_for_reconstruction") == accepted,
                  f"{prefix}.{path}.decision usable flag is inconsistent")
        completed += int(valid)
        raw_strict += int(strict)
        raw_relaxed += int(relaxed)
        accepted_correct += int(correct)
        accepted_error += int(wrong)
        rejected += int(valid and not accepted)
        zero += int(is_zero)
        failed += int(not valid and not is_zero)
    return {
        "requested": requested,
        "completed": completed,
        "raw_strict": raw_strict,
        "raw_relaxed": raw_relaxed,
        "accepted_correct": accepted_correct,
        "accepted_error": accepted_error,
        "rejected": rejected,
        "zero_candidate": zero,
        "failed": failed,
    }


def _node_evidence(row: Mapping[str, Any], label: str) -> Dict[str, int]:
    evidence = row.get("node_evidence")
    _need(isinstance(evidence, dict),
          f"{label}.node_evidence missing; matcher metrics are not inferable "
          "from registration outcomes")
    unknown = set(evidence) - set(NODE_FIELDS)
    missing = set(NODE_FIELDS) - set(evidence)
    _need(not missing, f"{label}.node_evidence missing {sorted(missing)}")
    _need(not unknown, f"{label}.node_evidence unknown fields {sorted(unknown)}")
    values = {
        key: _exact_int(evidence[key], f"{label}.node_evidence.{key}")
        for key in NODE_FIELDS
    }
    _need(values["tp"] <= values["predicted"],
          f"{label}.node_evidence tp exceeds predicted")
    _need(values["tp"] <= values["anchors"],
          f"{label}.node_evidence tp exceeds anchors")
    _need(values["top1_hits"] <= values["top1_total"],
          f"{label}.node_evidence top1 hits exceed total")
    _need(values["top5_hits"] <= values["top5_total"],
          f"{label}.node_evidence top5 hits exceed total")
    _need(values["top5_total"] == values["anchors"],
          f"{label}.node_evidence top5_total must equal anchors")
    return values


def aggregate_node_metrics(rows: Sequence[Mapping[str, Any]],
                           label: str) -> Dict[str, float]:
    per_pair: List[Dict[str, float]] = []
    pooled = {key: 0 for key in NODE_FIELDS}
    for index, row in enumerate(rows):
        value = _node_evidence(row, f"{label}.rows[{index}]")
        for key in NODE_FIELDS:
            pooled[key] += value[key]
        precision = value["tp"] / value["predicted"] \
            if value["predicted"] else 0.0
        recall = value["tp"] / value["anchors"] \
            if value["anchors"] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) \
            if precision + recall else 0.0
        top1 = value["top1_hits"] / value["top1_total"] \
            if value["top1_total"] else 0.0
        top5 = value["top5_hits"] / value["top5_total"] \
            if value["top5_total"] else 0.0
        per_pair.append({
            "node_precision": precision, "node_recall": recall,
            "node_f1": f1, "top1": top1, "top5": top5,
        })
    _need(len(per_pair) == EXPECTED_PAIRS,
          f"{label}: node metric denominator must be {EXPECTED_PAIRS}")
    _need(pooled["predicted"] > 0 and pooled["anchors"] > 0
          and pooled["top1_total"] > 0 and pooled["top5_total"] > 0,
          f"{label}: pooled node metric denominators must be non-zero")
    micro_p = pooled["tp"] / pooled["predicted"]
    micro_r = pooled["tp"] / pooled["anchors"]
    micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p + micro_r else 0.0)
    return {
        "macro_node_precision": statistics.fmean(
            item["node_precision"] for item in per_pair),
        "macro_node_recall": statistics.fmean(
            item["node_recall"] for item in per_pair),
        "macro_node_f1": statistics.fmean(
            item["node_f1"] for item in per_pair),
        "micro_node_precision": micro_p,
        "micro_node_recall": micro_r,
        "micro_node_f1": micro_f1,
        "macro_top1": statistics.fmean(item["top1"] for item in per_pair),
        "micro_top1": pooled["top1_hits"] / pooled["top1_total"],
        "macro_top5": statistics.fmean(item["top5"] for item in per_pair),
        "micro_top5": pooled["top5_hits"] / pooled["top5_total"],
        "pair_denominator": len(per_pair),
        **{f"pooled_{key}": value for key, value in pooled.items()},
    }


def _validate_one(data: Mapping[str, Any], *, checkpoint: str, repeat: int,
                  expected_sha: str, label: str) -> Tuple[List[str],
                                                                 Dict[str, Any]]:
    _need(data.get("schema") == RESULT_SCHEMA,
          f"{label}: schema must be {RESULT_SCHEMA}")
    _need(data.get("split") == "selection", f"{label}: split must be selection")
    _need(data.get("checkpoint") == checkpoint,
          f"{label}: checkpoint mismatch")
    _need(data.get("repeat") == repeat, f"{label}: repeat mismatch")
    _need(data.get("checkpoint_sha256") == expected_sha,
          f"{label}: checkpoint SHA mismatch")
    repository = data.get("repository")
    _need(isinstance(repository, dict), f"{label}: repository object missing")
    code_root = data.get("code_root")
    asset_root = data.get("asset_root")
    _need(isinstance(code_root, str) and Path(code_root).is_absolute(),
          f"{label}: absolute code_root missing")
    _need(isinstance(asset_root, str) and Path(asset_root).is_absolute(),
          f"{label}: absolute asset_root missing")
    _need(repository.get("tracked_dirty") is False,
          f"{label}: evidence was produced from a tracked-dirty worktree")
    repository_head = repository.get("head")
    _need(isinstance(repository_head, str)
          and len(repository_head) == 40
          and all(character in "0123456789abcdef"
                  for character in repository_head),
          f"{label}: repository HEAD must be 40 lowercase hex")
    rows = data.get("rows")
    _need(isinstance(rows, list) and len(rows) == EXPECTED_PAIRS,
          f"{label}: rows denominator must be {EXPECTED_PAIRS}")
    pair_ids = []
    for index, row in enumerate(rows):
        _need(isinstance(row, dict), f"{label}.rows[{index}] must be object")
        pair_id = row.get("pair_id")
        _need(isinstance(pair_id, str) and pair_id,
              f"{label}.rows[{index}].pair_id invalid")
        pair_ids.append(pair_id)
    _need(len(set(pair_ids)) == EXPECTED_PAIRS,
          f"{label}: pair IDs must be unique")
    manifest = data.get("split_manifest")
    _need(isinstance(manifest, dict), f"{label}: split_manifest missing")
    expected_manifest = {
        "name": "selection", "expected": EXPECTED_PAIRS,
        "actual": EXPECTED_PAIRS, "unique": EXPECTED_PAIRS,
        "sha256": _pair_manifest_sha(pair_ids),
    }
    _need(manifest == expected_manifest,
          f"{label}: pair manifest or manifest SHA mismatch")
    stored_counts = data.get("counts")
    _need(isinstance(stored_counts, dict), f"{label}: counts missing")
    recomputed = {}
    for path in PATHS:
        counts = _recompute_counts(rows, path, label)
        _need(path in stored_counts and isinstance(stored_counts[path], dict),
              f"{label}: counts.{path} missing")
        for field in COUNT_FIELDS:
            _need(stored_counts[path].get(field) == counts[field],
                  f"{label}: counts.{path}.{field} is not recomputable")
        recomputed[path] = counts
    return pair_ids, {
        "registration": recomputed, "repository": repository,
        "code_root": code_root,
    }


def _result_paths(formal_root: Path, checkpoint: str) -> List[Path]:
    directory = formal_root / "selection" / checkpoint
    expected = [directory / f"repeat_{repeat:02d}.json" for repeat in REPEATS]
    discovered = sorted(directory.glob("repeat_*.json")) if directory.is_dir() else []
    _need(discovered == expected,
          f"{checkpoint}: require exactly repeat_00/01/02.json, got "
          f"{[path.name for path in discovered]}")
    return expected


def load_and_validate(formal_root: Path, protocol: Mapping[str, Any]) \
        -> Tuple[Dict[str, List[Mapping[str, Any]]], Dict[str, Any],
                 List[Dict[str, Any]]]:
    protocol_checkpoints = protocol.get("checkpoints")
    _need(isinstance(protocol_checkpoints, dict), "protocol.checkpoints missing")
    results: Dict[str, List[Mapping[str, Any]]] = {}
    computed: Dict[str, Any] = {}
    inputs: List[Dict[str, Any]] = []
    canonical_pair_ids: Sequence[str] | None = None
    canonical_code_root: str | None = None
    for checkpoint in CHECKPOINTS:
        spec = protocol_checkpoints.get(checkpoint)
        _need(isinstance(spec, dict), f"protocol.checkpoints.{checkpoint} missing")
        expected_sha = spec.get("sha256")
        _need(isinstance(expected_sha, str) and len(expected_sha) == 64,
              f"protocol checkpoint {checkpoint} SHA invalid")
        paths = _result_paths(formal_root, checkpoint)
        results[checkpoint] = []
        computed[checkpoint] = []
        for repeat, path in zip(REPEATS, paths):
            data = _load_json(path)
            pair_ids, derived = _validate_one(
                data, checkpoint=checkpoint, repeat=repeat,
                expected_sha=expected_sha,
                label=f"{checkpoint}/repeat_{repeat:02d}")
            if canonical_pair_ids is None:
                canonical_pair_ids = pair_ids
                canonical_code_root = derived["code_root"]
            _need(pair_ids == canonical_pair_ids,
                  f"{checkpoint}/repeat_{repeat:02d}: pair order differs")
            _need(derived["code_root"] == canonical_code_root,
                  f"{checkpoint}/repeat_{repeat:02d}: code_root differs")
            results[checkpoint].append(data)
            computed[checkpoint].append(derived)
            inputs.append({
                "checkpoint": checkpoint, "repeat": repeat,
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return results, computed, inputs


def _canonical_json_sha(value: Any) -> str:
    return _sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8"))


def _validate_hex_sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64
          and all(character in "0123456789abcdef" for character in value),
          f"{label} must be a lowercase SHA-256")
    return value


def _validate_node_sidecar(data: Mapping[str, Any], *, checkpoint: str,
                           expected_sha: str, pair_ids: Sequence[str],
                           label: str) -> Dict[str, Any]:
    _need(data.get("schema") == NODE_SIDECAR_SCHEMA,
          f"{label}: schema must be {NODE_SIDECAR_SCHEMA}")
    _need(data.get("checkpoint") == checkpoint,
          f"{label}: checkpoint mismatch")
    _need(data.get("checkpoint_sha256") == expected_sha,
          f"{label}: checkpoint SHA mismatch")
    _need(data.get("split") == "selection", f"{label}: split mismatch")
    manifest = data.get("pair_manifest")
    expected_manifest = {
        "name": "selection", "expected": EXPECTED_PAIRS,
        "actual": EXPECTED_PAIRS, "unique": EXPECTED_PAIRS,
        "sha256": _pair_manifest_sha(pair_ids),
    }
    _need(manifest == expected_manifest, f"{label}: pair manifest mismatch")
    pairs = data.get("pairs")
    _need(isinstance(pairs, list) and len(pairs) == EXPECTED_PAIRS,
          f"{label}: pairs denominator must be {EXPECTED_PAIRS}")
    _need([pair.get("pair_id") for pair in pairs
           if isinstance(pair, dict)] == list(pair_ids),
          f"{label}: pair IDs/order differ from formal evidence")
    cache_entries = []
    anchor_entries = []
    for index, pair in enumerate(pairs):
        prefix = f"{label}.pairs[{index}]"
        _need(isinstance(pair, dict), f"{prefix} must be an object")
        cache = pair.get("cache")
        anchors = pair.get("anchors")
        _need(isinstance(cache, dict), f"{prefix}.cache missing")
        _need(isinstance(anchors, dict), f"{prefix}.anchors missing")
        for field in ("sha256", "cache_key", "input_sha256",
                      "embedding_sha256", "similarity_sha256"):
            _validate_hex_sha(cache.get(field), f"{prefix}.cache.{field}")
        _exact_int(cache.get("bytes"), f"{prefix}.cache.bytes", minimum=1)
        _need(isinstance(cache.get("path"), str) and cache["path"],
              f"{prefix}.cache.path missing")
        raw = anchors.get("raw_object_ids")
        mapped = anchors.get("mapped_indices")
        unmapped = anchors.get("unmapped_object_ids")
        source = anchors.get("source_pair_json")
        _need(isinstance(raw, list) and isinstance(mapped, list)
              and isinstance(unmapped, list),
              f"{prefix}.anchors lists missing")
        _need(isinstance(source, dict),
              f"{prefix}.anchors.source_pair_json missing")
        _need(isinstance(source.get("path"), str) and source["path"],
              f"{prefix}.anchors source path missing")
        _validate_hex_sha(source.get("sha256"),
                          f"{prefix}.anchors.source_pair_json.sha256")
        for field, rows in (("raw_object_ids", raw),
                            ("mapped_indices", mapped),
                            ("unmapped_object_ids", unmapped)):
            _need(all(isinstance(item, list) and len(item) == 2
                      and all(type(value) is int for value in item)
                      for item in rows),
                  f"{prefix}.anchors.{field} must contain integer pairs")
        _need(len({tuple(item) for item in raw}) == len(raw),
              f"{prefix}: duplicate raw anchors")
        _need(len({tuple(item) for item in mapped}) == len(mapped),
              f"{prefix}: duplicate mapped anchors")
        node = _node_evidence(pair, prefix)
        _need(node["anchors"] == len(mapped),
              f"{prefix}: node anchor denominator differs from manifest")
        _need((node["predicted"] == 0)
              == bool(pair.get("zero_candidate")),
              f"{prefix}: zero-candidate flag differs from node evidence")
        cache_entries.append({
            "pair_id": pair["pair_id"], "sha256": cache["sha256"],
            "cache_key": cache["cache_key"],
        })
        anchor_entries.append({
            "pair_id": pair["pair_id"],
            "source_pair_json": source,
            "raw_object_ids": raw, "mapped_indices": mapped,
            "unmapped_object_ids": unmapped,
        })
    cache_manifest = data.get("cache_manifest")
    expected_cache_manifest = {
        "count": EXPECTED_PAIRS, "unique": EXPECTED_PAIRS,
        "sha256": _canonical_json_sha(cache_entries),
    }
    _need(cache_manifest == expected_cache_manifest,
          f"{label}: cache manifest mismatch")
    gt_manifest = data.get("gt_anchor_manifest")
    _need(isinstance(gt_manifest, dict), f"{label}: GT anchor manifest missing")
    expected_gt = {
        "loader": "adapters.sgf.data_sources.load_anchor_ids",
        "loader_source_sha256": gt_manifest.get("loader_source_sha256"),
        "pair_count": EXPECTED_PAIRS,
        "sha256": _canonical_json_sha(anchor_entries),
    }
    _validate_hex_sha(expected_gt["loader_source_sha256"],
                      f"{label}.gt_anchor_manifest.loader_source_sha256")
    _need(gt_manifest == expected_gt, f"{label}: GT anchor manifest mismatch")
    provenance = data.get("provenance")
    _need(isinstance(provenance, dict)
          and provenance.get("gt_posthoc_only") is True,
          f"{label}: post-hoc GT provenance missing")
    _need(isinstance(provenance.get("source_sha256"), dict)
          and provenance["source_sha256"],
          f"{label}: source hashes missing")
    return aggregate_node_metrics(pairs, label)


def load_node_sidecars(evidence_dir: Path,
                       protocol: Mapping[str, Any],
                       pair_ids: Sequence[str]) \
        -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    expected_paths = [evidence_dir / f"{checkpoint}.json"
                      for checkpoint in CHECKPOINTS]
    discovered = sorted(evidence_dir.glob("*.json")) \
        if evidence_dir.is_dir() else []
    _need(discovered == expected_paths,
          "node evidence requires exactly A.json, B.json and D.json")
    metrics = {}
    inputs = []
    for checkpoint, path in zip(CHECKPOINTS, expected_paths):
        data = _load_json(path)
        expected_sha = protocol["checkpoints"][checkpoint]["sha256"]
        metrics[checkpoint] = _validate_node_sidecar(
            data, checkpoint=checkpoint, expected_sha=expected_sha,
            pair_ids=pair_ids, label=f"node-evidence/{checkpoint}")
        inputs.append({
            "kind": "node_evidence_sidecar", "checkpoint": checkpoint,
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "gt_anchor_manifest_sha256": data["gt_anchor_manifest"]["sha256"],
            "cache_manifest_sha256": data["cache_manifest"]["sha256"],
        })
    return metrics, inputs


def _distribution(computed: Mapping[str, Any], checkpoint: str,
                  path: str) -> Dict[str, Any]:
    rows = [entry["registration"][path] for entry in computed[checkpoint]]
    return {
        "per_repeat": rows,
        "median": {
            field: _median(entry[field] for entry in rows)
            for field in COUNT_FIELDS
        },
    }


def _sign_stable(values: Sequence[float], *, tolerance: float = 1e-12) -> bool:
    signs = {1 if value > tolerance else -1 if value < -tolerance else 0
             for value in values}
    signs.discard(0)
    return len(signs) <= 1


def _verify_checkpoint_files(protocol: Mapping[str, Any],
                             results: Mapping[str, Sequence[Mapping[str, Any]]]) \
        -> Dict[str, Dict[str, Any]]:
    verified = {}
    for checkpoint in CHECKPOINTS:
        spec = protocol["checkpoints"][checkpoint]
        roots = {row.get("asset_root") for row in results[checkpoint]}
        _need(len(roots) == 1 and all(isinstance(root, str) for root in roots),
              f"{checkpoint}: one asset_root is required")
        path = Path(next(iter(roots))) / spec["path"]
        _need(path.is_file(), f"{checkpoint}: checkpoint file missing: {path}")
        actual = sha256_file(path)
        _need(actual == spec["sha256"],
              f"{checkpoint}: checkpoint file SHA mismatch")
        verified[checkpoint] = {
            "path": str(path.resolve()), "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return verified


def _candidate_path_safety(
        registration: Mapping[str, Mapping[str, Mapping[str, Any]]],
        gate2_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Audit every evaluated B path before choosing the production path.

    The formal result schema partitions a non-completed registration into a
    ``zero_candidate`` outcome or ``failed``.  For the protocol's
    failed/unknown safety bound, a zero-candidate outcome is therefore reported
    as ``unknown`` and the combined count is checked fail-closed.  This report
    deliberately covers F, C0 and C1: selecting C1 cannot hide a safety failure
    observed on another evaluated B path.
    """
    accepted_error_max = _exact_int(
        gate2_spec.get("accepted_error_max"),
        "protocol.gates.selection.accepted_error_max")
    # Selection did not spell this shared hard bound out separately.  The
    # preregistered safety vocabulary uses zero for failed/unknown elsewhere;
    # absence here must not silently weaken safety.
    failed_unknown_max = _exact_int(
        gate2_spec.get("failed_unknown_max", 0),
        "protocol.gates.selection.failed_unknown_max")
    paths: Dict[str, Any] = {}
    violations = []
    for path in PATHS:
        per_repeat = []
        for repeat, counts in zip(REPEATS,
                                  registration["B"][path]["per_repeat"]):
            failed = counts["failed"]
            unknown = counts["zero_candidate"]
            failed_unknown = failed + unknown
            accepted_ok = counts["accepted_error"] <= accepted_error_max
            failed_unknown_ok = failed_unknown <= failed_unknown_max
            row = {
                "repeat": repeat,
                "accepted_error": counts["accepted_error"],
                "failed": failed,
                "unknown": unknown,
                "failed_unknown": failed_unknown,
                "accepted_error_ok": accepted_ok,
                "failed_unknown_ok": failed_unknown_ok,
                "passed": accepted_ok and failed_unknown_ok,
            }
            per_repeat.append(row)
            if not row["passed"]:
                violations.append(
                    f"B/{path}/repeat_{repeat:02d}: "
                    f"accepted_error={counts['accepted_error']}, "
                    f"failed={failed}, unknown={unknown}")
        paths[path] = {
            "passed": all(row["passed"] for row in per_repeat),
            "per_repeat": per_repeat,
        }
    return {
        "checkpoint": "B",
        "scope": list(PATHS),
        "thresholds": {
            "accepted_error_max": accepted_error_max,
            "failed_unknown_max": failed_unknown_max,
            "failed_unknown_max_source": (
                "protocol" if "failed_unknown_max" in gate2_spec
                else "fail_closed_default_zero"),
        },
        "paths": paths,
        "passed": not violations,
        "violations": violations,
    }


def evaluate_gates(protocol: Mapping[str, Any], computed: Mapping[str, Any],
                   node_metrics: Mapping[str, Mapping[str, Any]]) \
        -> Dict[str, Any]:
    gates = protocol.get("gates")
    _need(isinstance(gates, dict), "protocol.gates missing")
    gate1_spec = gates.get("flat_recovery")
    gate2_spec = gates.get("selection")
    _need(isinstance(gate1_spec, dict), "protocol.gates.flat_recovery missing")
    _need(isinstance(gate2_spec, dict), "protocol.gates.selection missing")
    registration = {
        checkpoint: {path: _distribution(computed, checkpoint, path)
                     for path in PATHS}
        for checkpoint in CHECKPOINTS
    }
    _need(set(node_metrics) == set(CHECKPOINTS),
          "node metrics must cover A, B and D")
    node = {checkpoint: {"single_shared_cache": node_metrics[checkpoint]}
            for checkpoint in CHECKPOINTS}

    a_flat = registration["A"]["F"]
    gate1_checks = {
        "median_raw_strict": (
            a_flat["median"]["raw_strict"]
            >= gate1_spec["median_raw_strict_min"]),
        "median_accepted_correct": (
            a_flat["median"]["accepted_correct"]
            >= gate1_spec["median_correct_accepted_min"]),
        "accepted_error_all_repeats": all(
            row["accepted_error"] <= gate1_spec["accepted_error_max"]
            for row in a_flat["per_repeat"]),
        "failed_all_repeats": all(
            row["failed"] <= gate1_spec["failed_unknown_max"]
            for row in a_flat["per_repeat"]),
    }
    gate1_passed = all(gate1_checks.values())
    _need(gate1_passed, f"Gate1 failed: {gate1_checks}")

    # F and C0 are controls.  B is explicitly the primary checkpoint and C1
    # the corrected production candidate; D is diagnostic-only by protocol.
    b_c1 = registration["B"]["C1"]
    candidate_path_safety = _candidate_path_safety(
        registration, gate2_spec)
    a_node = node_metrics["A"]
    b_node = node_metrics["B"]
    metric_deltas = {
        field: b_node[field] - a_node[field]
        for field in ("macro_node_f1", "macro_top1", "macro_top5")
    }
    registration_deltas = {
        field: [
            b_c1["per_repeat"][repeat][field]
            - a_flat["per_repeat"][repeat][field]
            for repeat in REPEATS
        ] for field in ("raw_strict", "accepted_correct", "zero_candidate")
    }
    gate2_checks = {
        "winner_accepted_error_all_repeats": all(
            row["accepted_error"] <= gate2_spec["accepted_error_max"]
            for row in b_c1["per_repeat"]),
        "all_candidate_paths_safe": candidate_path_safety["passed"],
        "macro_f1_shared_cache": (
            b_node["macro_node_f1"]
            >= gate2_spec["macro_f1_min"]),
        "macro_top1_shared_cache": (
            b_node["macro_top1"]
            >= gate2_spec["macro_top1_min"]),
        "macro_top5_shared_cache": (
            b_node["macro_top5"]
            >= gate2_spec["macro_top5_min"]),
        "raw_strict_drop_vs_A": (
            b_c1["median"]["raw_strict"]
            >= a_flat["median"]["raw_strict"]
            - gate2_spec["raw_strict_drop_vs_recovered_A_max"]),
        # The prose says accepted-correct must not be lower but supplies no
        # tolerance.  Fail-closed interpretation: no median regression.
        "accepted_correct_not_lower_than_A": (
            b_c1["median"]["accepted_correct"]
            >= a_flat["median"]["accepted_correct"]),
        "zero_candidate_not_regressed": (
            all(b_c1["per_repeat"][repeat]["zero_candidate"]
                <= a_flat["per_repeat"][repeat]["zero_candidate"]
                for repeat in REPEATS)),
        "three_run_direction_stable": all(
            _sign_stable(values) for values in
            list(registration_deltas.values())),
    }
    gate2_passed = all(gate2_checks.values())
    _need(gate2_passed,
          f"Gate2 failed: {gate2_checks}; candidate path safety: "
          f"{candidate_path_safety}")
    return {
        "gate1": {"passed": True, "checks": gate1_checks,
                  "thresholds": gate1_spec},
        "gate2": {"passed": True, "checks": gate2_checks,
                  "thresholds": gate2_spec,
                  "metric_deltas_B_minus_A": metric_deltas,
                  "registration_deltas_B_C1_minus_A_F": registration_deltas},
        "candidate_path_safety": candidate_path_safety,
        "registration": registration,
        "node_metrics": node,
        "winner": {
            "checkpoint": "B", "path": "C1",
            "reason": "protocol primary checkpoint B and corrected path C1 "
                      "passed every Gate2 hard check",
        },
        "diagnostic_only": ["D"],
    }


def _git_output(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _need(process.returncode == 0,
          f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def collect_provenance(repo_root: Path, protocol_path: Path,
                       checkpoint_files: Mapping[str, Mapping[str, Any]]) \
        -> Dict[str, Any]:
    source_relpaths = (
        "scripts/v6fix_gate_freezer.py",
        "scripts/v6fix_node_evidence.py",
        "scripts/v6fix_consistency_audit.py",
        "scripts/spatial_consistency.py",
    )
    source_hashes = {}
    for relpath in source_relpaths:
        path = repo_root / relpath
        _need(path.is_file(), f"source file missing: {path}")
        source_hashes[relpath] = sha256_file(path)
    protocol_md = protocol_path.with_suffix(".md")
    _need(protocol_md.is_file(), f"protocol markdown missing: {protocol_md}")
    dirty = _git_output(repo_root, "status", "--porcelain",
                        "--untracked-files=no")
    _need(not dirty, "current tracked worktree must be clean before freezing")
    return {
        "repository_root": str(repo_root.resolve()),
        "git_head": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_branch": _git_output(repo_root, "branch", "--show-current"),
        "tracked_dirty": False,
        "source_sha256": source_hashes,
        "protocol_json": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "protocol_md": {
            "path": str(protocol_md.resolve()),
            "sha256": sha256_file(protocol_md),
        },
        "checkpoint_files": checkpoint_files,
    }


def build_freeze(formal_root: Path, protocol_path: Path,
                 repo_root: Path, evidence_dir: Path | None = None) \
        -> Dict[str, Any]:
    protocol = _load_json(protocol_path)
    _need(protocol.get("phase") == "V6-Fix consistency audit",
          "wrong protocol phase")
    results, computed, inputs = load_and_validate(formal_root, protocol)
    evidence_code_root = Path(computed["A"][0]["code_root"])
    _need(repo_root.resolve() == evidence_code_root.resolve(),
          f"repo_root/code_root mismatch: {repo_root.resolve()} != "
          f"{evidence_code_root.resolve()}")
    pair_ids = [row["pair_id"] for row in results["A"][0]["rows"]]
    evidence_dir = evidence_dir or formal_root / "node_evidence"
    node_metrics, sidecar_inputs = load_node_sidecars(
        evidence_dir, protocol, pair_ids)
    inputs.extend(sidecar_inputs)
    checkpoint_files = _verify_checkpoint_files(protocol, results)
    evaluation = evaluate_gates(protocol, computed, node_metrics)
    evaluation["winner"]["checkpoint_sha256"] = \
        protocol["checkpoints"]["B"]["sha256"]
    evidence_heads = {
        checkpoint: sorted({
            row["repository"].get("head") for row in results[checkpoint]
        }) for checkpoint in CHECKPOINTS
    }
    _need(all(all(isinstance(value, str) and value for value in values)
              for values in evidence_heads.values()),
          "evidence repository HEAD missing")
    return {
        "schema": FREEZE_SCHEMA,
        "status": "FROZEN",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "decision": evaluation,
        "inputs": inputs,
        "evidence_repository_heads": evidence_heads,
        "provenance": collect_provenance(
            repo_root, protocol_path, checkpoint_files),
    }


def atomic_write_new(path: Path, document: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                dir=str(path.parent), delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-linking a fully fsynced temporary file gives create-new
            # semantics atomically: unlike os.replace, it can never overwrite
            # a receipt or freeze created by another process after preflight.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Blocked(f"refusing to overwrite artifact: {path}") from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True,
                        help="formal_v2 directory containing selection/A|B|D")
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--node-evidence-dir", type=Path,
                        help="default: FORMAL_ROOT/node_evidence")
    parser.add_argument("--output", type=Path,
                        help="default: FORMAL_ROOT/frozen_selection.json")
    parser.add_argument(
        "--receipt", type=Path,
        help="optional atomic create-new JSON receipt mirroring stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (args.output or
              args.formal_root / "frozen_selection.json").resolve()
    receipt = args.receipt.resolve() if args.receipt else None
    if receipt is not None and (receipt == output or receipt.exists()):
        report = {
            "schema": FREEZE_SCHEMA, "status": "BLOCKED",
            "reason": ("receipt must differ from frozen output" if
                       receipt == output else
                       f"refusing to overwrite artifact: {receipt}"),
            "frozen_selection_written": False,
        }
        print(json.dumps(report, sort_keys=True), file=sys.stdout)
        return 2
    output_written = False
    try:
        document = build_freeze(
            args.formal_root.resolve(), args.protocol_json.resolve(),
            args.repo_root.resolve(),
            (args.node_evidence_dir.resolve()
             if args.node_evidence_dir else None))
        atomic_write_new(output, document)
        output_written = True
    except (Blocked, OSError, ValueError, KeyError, TypeError) as exc:
        report = {
            "schema": FREEZE_SCHEMA, "status": "BLOCKED",
            "reason": str(exc), "frozen_selection_written": False,
        }
        if receipt is not None:
            try:
                atomic_write_new(receipt, report)
            except (Blocked, OSError, ValueError, TypeError) as receipt_exc:
                report["reason"] += f"; receipt write failed: {receipt_exc}"
        print(json.dumps(report, sort_keys=True), file=sys.stdout)
        return 2
    report = {
        "schema": FREEZE_SCHEMA, "status": "FROZEN",
        "output": str(output),
        "winner": document["decision"]["winner"],
    }
    if receipt is not None:
        try:
            atomic_write_new(receipt, report)
        except (Blocked, OSError, ValueError, TypeError) as exc:
            # This run created output, so rolling it back is safe and preserves
            # the invariant that an exit-2 run never creates frozen_selection.
            if output_written:
                try:
                    output.unlink()
                    directory_fd = os.open(output.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError as rollback_exc:
                    exc = OSError(f"{exc}; output rollback failed: "
                                  f"{rollback_exc}")
            blocked = {
                "schema": FREEZE_SCHEMA, "status": "BLOCKED",
                "reason": f"receipt write failed: {exc}",
                "frozen_selection_written": False,
            }
            print(json.dumps(blocked, sort_keys=True), file=sys.stdout)
            return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
