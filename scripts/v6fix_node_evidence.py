#!/usr/bin/env python3
"""Extract post-hoc matcher evidence from immutable V6-Fix caches.

The extractor is deliberately read-only with respect to formal JSON and cache
artifacts.  It loads no model and calls no SGAligner forward, GeoTransformer,
RANSAC, ICP or RegistrationDecision code.  GT anchors are loaded only after
the predicted ``node_corrs`` and ``rank_list`` have been read and validated.

One sidecar is produced for each checkpoint A/B/D.  All three documents are
built and validated in memory before any output is atomically created.
Existing sidecars are never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import torch


RESULT_SCHEMA = "v6fix-consistency-audit-v2"
CACHE_SCHEMA = "v6fix-inference-cache-v2"
SIDECAR_SCHEMA = "v6fix-node-evidence-v1"
CHECKPOINTS = ("A", "B", "D")
REPEATS = (0, 1, 2)
EXPECTED_PAIRS = 89


class ExtractionBlocked(RuntimeError):
    """A fail-closed cache, GT provenance, or output invariant failure."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionBlocked(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _pair_manifest_sha(pair_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(pair_ids) + "\n").encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    _need(path.is_file(), f"missing JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtractionBlocked(f"invalid JSON {path}: {exc}") from exc
    _need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _lower_sha(value: Any, label: str) -> str:
    _need(isinstance(value, str) and len(value) == 64
          and all(character in "0123456789abcdef" for character in value),
          f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum,
          f"{label} must be an integer >= {minimum}")
    return value


def _formal_manifest(formal_root: Path, checkpoint: str,
                     expected_checkpoint_sha: str) \
        -> Tuple[List[str], Mapping[str, Any]]:
    directory = formal_root / "selection" / checkpoint
    expected_paths = [directory / f"repeat_{repeat:02d}.json"
                      for repeat in REPEATS]
    discovered = sorted(directory.glob("repeat_*.json")) \
        if directory.is_dir() else []
    _need(discovered == expected_paths,
          f"{checkpoint}: require exactly formal repeats 00/01/02")
    canonical_pairs = None
    canonical_manifest = None
    canonical_code_root = None
    canonical_asset_root = None
    for repeat, path in zip(REPEATS, expected_paths):
        data = _load_json(path)
        label = f"{checkpoint}/repeat_{repeat:02d}"
        _need(data.get("schema") == RESULT_SCHEMA,
              f"{label}: formal schema mismatch")
        _need(data.get("split") == "selection"
              and data.get("checkpoint") == checkpoint
              and data.get("repeat") == repeat,
              f"{label}: formal identity mismatch")
        _need(data.get("checkpoint_sha256") == expected_checkpoint_sha,
              f"{label}: checkpoint SHA mismatch")
        repository = data.get("repository")
        _need(isinstance(repository, dict)
              and repository.get("tracked_dirty") is False,
              f"{label}: formal repository provenance invalid")
        repository_head = repository.get("head")
        _need(isinstance(repository_head, str)
              and len(repository_head) == 40
              and all(character in "0123456789abcdef"
                      for character in repository_head),
              f"{label}: formal repository HEAD must be 40 lowercase hex")
        code_root = data.get("code_root")
        asset_root = data.get("asset_root")
        _need(isinstance(code_root, str) and Path(code_root).is_absolute(),
              f"{label}: absolute code_root missing")
        _need(isinstance(asset_root, str) and Path(asset_root).is_absolute(),
              f"{label}: absolute asset_root missing")
        rows = data.get("rows")
        _need(isinstance(rows, list) and len(rows) == EXPECTED_PAIRS,
              f"{label}: expected {EXPECTED_PAIRS} rows")
        pair_ids = [row.get("pair_id") for row in rows
                    if isinstance(row, dict)]
        _need(len(pair_ids) == EXPECTED_PAIRS
              and all(isinstance(pair_id, str) and pair_id
                      for pair_id in pair_ids)
              and len(set(pair_ids)) == EXPECTED_PAIRS,
              f"{label}: invalid or duplicate pair IDs")
        manifest = {
            "name": "selection", "expected": EXPECTED_PAIRS,
            "actual": EXPECTED_PAIRS, "unique": EXPECTED_PAIRS,
            "sha256": _pair_manifest_sha(pair_ids),
        }
        _need(data.get("split_manifest") == manifest,
              f"{label}: pair manifest mismatch")
        if canonical_pairs is None:
            canonical_pairs = pair_ids
            canonical_manifest = manifest
            canonical_code_root = code_root
            canonical_asset_root = asset_root
        _need(pair_ids == canonical_pairs,
              f"{label}: pair order changed across repeats")
        _need(code_root == canonical_code_root
              and asset_root == canonical_asset_root,
              f"{label}: roots changed across repeats")
    return list(canonical_pairs), {
        "pair_manifest": canonical_manifest,
        "code_root": canonical_code_root,
        "asset_root": canonical_asset_root,
        "formal_json_sha256": [
            {"repeat": repeat, "path": str(path.resolve()),
             "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for repeat, path in zip(REPEATS, expected_paths)
        ],
    }


def official_anchor_loader(repo_root: Path, pair_id: str) \
        -> Tuple[List[Tuple[int, int]], Path]:
    """Invoke the established anchor loader and identify its source record."""
    for path in (repo_root, repo_root / "src", repo_root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # Deliberately imported here, after predicted cache fields are validated by
    # the caller.  No GT transform loader is imported or called.
    from adapters.sgf.data_sources import (  # pylint: disable=import-outside-toplevel
        LEGACY_PAIR_ROOTS, load_anchor_ids,
    )
    source = next((root / pair_id / "pair.json" for root in LEGACY_PAIR_ROOTS
                   if (root / pair_id / "pair.json").is_file()), None)
    _need(source is not None, f"{pair_id}: GT anchor pair.json not found")
    anchors = load_anchor_ids(pair_id)
    return [(int(src), int(ref)) for src, ref in anchors], source


def _validate_cache(cache: Mapping[str, Any], *, path: Path,
                    pair_id: str, checkpoint: str,
                    checkpoint_sha: str, code_root: Path) -> Dict[str, Any]:
    prefix = f"{checkpoint}/{pair_id}"
    required = {
        "cache_schema", "pair_id", "checkpoint_id", "checkpoint_sha256",
        "input_sha256", "embedding_sha256", "node_corrs", "rank_list",
        "similarity_sha256", "provenance", "geot",
    }
    missing = required - set(cache)
    _need(not missing, f"{prefix}: cache fields missing {sorted(missing)}")
    _need(cache.get("cache_schema") == CACHE_SCHEMA,
          f"{prefix}: cache schema mismatch")
    _need(cache.get("pair_id") == pair_id,
          f"{prefix}: cache pair_id mismatch")
    _need(cache.get("checkpoint_id") == checkpoint
          and cache.get("checkpoint_sha256") == checkpoint_sha,
          f"{prefix}: cache checkpoint identity mismatch")
    for field in ("input_sha256", "embedding_sha256", "similarity_sha256"):
        _lower_sha(cache.get(field), f"{prefix}.{field}")
    provenance = cache.get("provenance")
    _need(isinstance(provenance, dict), f"{prefix}: provenance missing")
    for field in ("cache_key", "pair_id", "checkpoint_id",
                  "checkpoint_sha256", "object_ids_order", "src_count",
                  "source_hashes"):
        _need(field in provenance, f"{prefix}: provenance.{field} missing")
    _lower_sha(provenance["cache_key"], f"{prefix}.cache_key")
    _need(provenance["pair_id"] == pair_id
          and provenance["checkpoint_id"] == checkpoint
          and provenance["checkpoint_sha256"] == checkpoint_sha,
          f"{prefix}: provenance identity mismatch")
    object_ids = provenance["object_ids_order"]
    src_count = _exact_int(provenance["src_count"],
                           f"{prefix}.src_count", minimum=1)
    _need(isinstance(object_ids, list)
          and all(type(value) is int for value in object_ids)
          and src_count < len(object_ids),
          f"{prefix}: object_ids_order invalid")
    src_ids, ref_ids = object_ids[:src_count], object_ids[src_count:]
    _need(len(set(src_ids)) == len(src_ids)
          and len(set(ref_ids)) == len(ref_ids),
          f"{prefix}: duplicate object ID within scan")
    node_corrs = cache["node_corrs"]
    _need(isinstance(node_corrs, list), f"{prefix}: node_corrs must be list")
    normalized_corrs = []
    for index, pair in enumerate(node_corrs):
        _need(isinstance(pair, (list, tuple)) and len(pair) == 2
              and all(type(value) is int for value in pair),
              f"{prefix}: node_corrs[{index}] invalid")
        src, ref = pair
        _need(0 <= src < src_count and src_count <= ref < len(object_ids),
              f"{prefix}: node_corrs[{index}] out of cross-graph range")
        normalized_corrs.append((src, ref))
    _need(len(set(normalized_corrs)) == len(normalized_corrs),
          f"{prefix}: duplicate node_corrs")
    rank_list = cache["rank_list"]
    _need(isinstance(rank_list, list) and len(rank_list) == len(object_ids),
          f"{prefix}: rank_list must cover every node")
    normalized_ranks = []
    expected_indices = set(range(len(object_ids)))
    for index, ranking in enumerate(rank_list):
        _need(isinstance(ranking, (list, tuple))
              and all(type(value) is int for value in ranking),
              f"{prefix}: rank_list[{index}] invalid")
        _need(len(ranking) == len(object_ids)
              and set(ranking) == expected_indices,
              f"{prefix}: rank_list[{index}] is not a full permutation")
        normalized_ranks.append(list(ranking))
    source_hashes = provenance["source_hashes"]
    _need(isinstance(source_hashes, dict) and source_hashes,
          f"{prefix}: provenance source hashes missing")
    for relpath, expected in source_hashes.items():
        _lower_sha(expected, f"{prefix}.source_hashes[{relpath}]")
        source = code_root / relpath
        _need(source.is_file(), f"{prefix}: source missing: {source}")
        _need(sha256_file(source) == expected,
              f"{prefix}: source SHA mismatch: {relpath}")
    _need(isinstance(cache["geot"], dict),
          f"{prefix}: GeoT raw cache object missing")
    return {
        "src_count": src_count, "object_ids": object_ids,
        "node_corrs": normalized_corrs, "rank_list": normalized_ranks,
        "cache_key": provenance["cache_key"],
        "source_hashes": source_hashes,
    }


def _posthoc_pair_evidence(cache: Mapping[str, Any], validated: Mapping[str, Any],
                           raw_anchors: Sequence[Tuple[int, int]]) \
        -> Tuple[Dict[str, int], Dict[str, Any]]:
    # Import the frozen semantics, not a reimplementation.
    from v4seal_metrics import per_pair_node_metrics  # noqa: WPS433

    src_count = validated["src_count"]
    object_ids = validated["object_ids"]
    src_map = {object_id: index
               for index, object_id in enumerate(object_ids[:src_count])}
    ref_map = {object_id: index + src_count
               for index, object_id in enumerate(object_ids[src_count:])}
    raw_unique = sorted(set((int(src), int(ref))
                            for src, ref in raw_anchors))
    mapped, unmapped = [], []
    for src, ref in raw_unique:
        if src in src_map and ref in ref_map:
            mapped.append((src_map[src], ref_map[ref]))
        else:
            unmapped.append((src, ref))
    anchor_idx = set(mapped)
    metrics = per_pair_node_metrics(
        validated["node_corrs"], validated["rank_list"], src_count,
        anchor_idx, sim=None)
    node_evidence = {
        "tp": int(metrics["tp"]),
        "predicted": int(metrics["pred_count"]),
        "anchors": int(metrics["anchor_count"]),
        "top1_hits": int(metrics["top1_hit"]),
        "top1_total": int(metrics["top1_total"]),
        "top5_hits": int(metrics["top5_hits"]),
        "top5_total": int(metrics["anchor_count"]),
    }
    return node_evidence, {
        "raw_object_ids": [[src, ref] for src, ref in raw_unique],
        "mapped_indices": [[src, ref] for src, ref in sorted(anchor_idx)],
        "unmapped_object_ids": [[src, ref] for src, ref in unmapped],
    }


def _git(repo_root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _need(process.returncode == 0,
          f"git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def build_sidecar(formal_root: Path, protocol: Mapping[str, Any],
                  repo_root: Path, checkpoint: str,
                  anchor_loader: Callable[[str], Tuple[
                      Sequence[Tuple[int, int]], Path]]) -> Dict[str, Any]:
    spec = protocol["checkpoints"][checkpoint]
    checkpoint_sha = _lower_sha(spec.get("sha256"),
                                f"protocol.checkpoints.{checkpoint}.sha256")
    pair_ids, formal = _formal_manifest(
        formal_root, checkpoint, checkpoint_sha)
    cache_dir = formal_root / "cache_v2" / checkpoint / "selection"
    expected_paths = [cache_dir / f"{pair_id}.pt" for pair_id in pair_ids]
    discovered = sorted(cache_dir.glob("*.pt")) if cache_dir.is_dir() else []
    _need(set(discovered) == set(expected_paths)
          and len(discovered) == EXPECTED_PAIRS,
          f"{checkpoint}: cache set must contain exactly {EXPECTED_PAIRS} "
          "pair-named .pt files")
    code_root = Path(formal["code_root"])
    _need(repo_root.resolve() == code_root.resolve(),
          f"{checkpoint}: repo_root/code_root mismatch: "
          f"{repo_root.resolve()} != {code_root.resolve()}")
    asset_root = Path(formal["asset_root"])
    checkpoint_path = asset_root / spec["path"]
    _need(checkpoint_path.is_file(),
          f"{checkpoint}: checkpoint file missing: {checkpoint_path}")
    _need(sha256_file(checkpoint_path) == checkpoint_sha,
          f"{checkpoint}: checkpoint file SHA mismatch")
    pair_rows = []
    shared_source_hashes = None
    # Phase 1: predicted evidence is loaded and validated without GT.
    loaded = []
    for pair_id, path in zip(pair_ids, expected_paths):
        try:
            cache = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # torch raises several serialization types
            raise ExtractionBlocked(
                f"{checkpoint}/{pair_id}: cache load failed: {exc}") from exc
        _need(isinstance(cache, dict),
              f"{checkpoint}/{pair_id}: cache must be a dictionary")
        validated = _validate_cache(
            cache, path=path, pair_id=pair_id, checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha, code_root=code_root)
        if shared_source_hashes is None:
            shared_source_hashes = validated["source_hashes"]
        _need(validated["source_hashes"] == shared_source_hashes,
              f"{checkpoint}/{pair_id}: source hashes vary across caches")
        loaded.append((pair_id, path, cache, validated))
    # Phase 2: GT anchors are consumed strictly post-hoc.
    metrics_scripts = str((repo_root / "scripts").resolve())
    if metrics_scripts not in sys.path:
        sys.path.insert(0, metrics_scripts)
    for pair_id, path, cache, validated in loaded:
        raw_anchors, source_pair_json = anchor_loader(pair_id)
        _need(source_pair_json.is_file(),
              f"{pair_id}: GT anchor source missing: {source_pair_json}")
        node_evidence, anchors = _posthoc_pair_evidence(
            cache, validated, raw_anchors)
        anchors["source_pair_json"] = {
            "path": str(source_pair_json.resolve()),
            "sha256": sha256_file(source_pair_json),
        }
        pair_rows.append({
            "pair_id": pair_id,
            "cache": {
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "cache_key": validated["cache_key"],
                "input_sha256": cache["input_sha256"],
                "embedding_sha256": cache["embedding_sha256"],
                "similarity_sha256": cache["similarity_sha256"],
            },
            "anchors": anchors,
            "zero_candidate": node_evidence["predicted"] == 0,
            "node_evidence": node_evidence,
        })
    cache_entries = [{
        "pair_id": row["pair_id"], "sha256": row["cache"]["sha256"],
        "cache_key": row["cache"]["cache_key"],
    } for row in pair_rows]
    anchor_entries = [{
        "pair_id": row["pair_id"],
        "source_pair_json": row["anchors"]["source_pair_json"],
        "raw_object_ids": row["anchors"]["raw_object_ids"],
        "mapped_indices": row["anchors"]["mapped_indices"],
        "unmapped_object_ids": row["anchors"]["unmapped_object_ids"],
    } for row in pair_rows]
    loader_source = repo_root / "src/adapters/sgf/data_sources.py"
    metric_source = repo_root / "scripts/v4seal_metrics.py"
    extractor_source = repo_root / "scripts/v6fix_node_evidence.py"
    for source in (loader_source, metric_source, extractor_source):
        _need(source.is_file(), f"source file missing: {source}")
    return {
        "schema": SIDECAR_SCHEMA,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha,
        "split": "selection", "pair_manifest": formal["pair_manifest"],
        "cache_manifest": {
            "count": EXPECTED_PAIRS, "unique": EXPECTED_PAIRS,
            "sha256": _canonical_json_sha(cache_entries),
        },
        "gt_anchor_manifest": {
            "loader": "adapters.sgf.data_sources.load_anchor_ids",
            "loader_source_sha256": sha256_file(loader_source),
            "pair_count": EXPECTED_PAIRS,
            "sha256": _canonical_json_sha(anchor_entries),
        },
        "provenance": {
            "gt_posthoc_only": True,
            "no_model_or_registration_calls": True,
            "formal_json_sha256": formal["formal_json_sha256"],
            "code_root": str(code_root.resolve()),
            "asset_root": str(asset_root.resolve()),
            "git_head": _git(repo_root, "rev-parse", "HEAD"),
            "source_sha256": {
                "scripts/v6fix_node_evidence.py": sha256_file(extractor_source),
                "scripts/v4seal_metrics.py": sha256_file(metric_source),
                "src/adapters/sgf/data_sources.py": sha256_file(loader_source),
                **shared_source_hashes,
            },
        },
        "pairs": pair_rows,
    }


def atomic_write_new(path: Path, document: Mapping[str, Any]) -> None:
    _need(not path.exists(), f"refusing to overwrite sidecar: {path}")
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
        os.replace(temporary, path)
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
    parser.add_argument("--formal-root", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="default: FORMAL_ROOT/node_evidence")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    formal_root = args.formal_root.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir
                  else formal_root / "node_evidence")
    try:
        protocol = _load_json(args.protocol_json.resolve())
        _need(protocol.get("phase") == "V6-Fix consistency audit",
              "wrong protocol phase")
        dirty = _git(repo_root, "status", "--porcelain",
                     "--untracked-files=no")
        _need(not dirty, "tracked worktree must be clean")
        outputs = [output_dir / f"{checkpoint}.json"
                   for checkpoint in CHECKPOINTS]
        _need(not any(path.exists() for path in outputs),
              "refusing to overwrite an existing sidecar")
        documents = {
            checkpoint: build_sidecar(
                formal_root, protocol, repo_root, checkpoint,
                lambda pair_id, root=repo_root:
                    official_anchor_loader(root, pair_id))
            for checkpoint in CHECKPOINTS
        }
        for checkpoint, path in zip(CHECKPOINTS, outputs):
            atomic_write_new(path, documents[checkpoint])
    except (ExtractionBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({
            "schema": SIDECAR_SCHEMA, "status": "BLOCKED",
            "reason": str(exc), "sidecars_written": False,
        }, sort_keys=True))
        return 2
    print(json.dumps({
        "schema": SIDECAR_SCHEMA, "status": "COMPLETE",
        "outputs": [str(path) for path in outputs],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
