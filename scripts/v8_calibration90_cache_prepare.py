"""Gate-0: prepare fresh B/calibration90 caches without labels.

The actual model/GeoTransformer builder is reused from
``v6fix_consistency_audit``.  This wrapper adds a frozen preflight plan,
reachable-function AST audit, staging-directory publication and a 90/90
cryptographic inventory.  It does not evaluate registration or import GT.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

CODE_ROOT = Path(__file__).resolve().parents[1]
for _path in (CODE_ROOT, CODE_ROOT / "src", CODE_ROOT / "scripts",
              CODE_ROOT / "src/inference/sgf_official"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v6fix_consistency_audit as builder  # noqa: E402
import v7_registration_pilot as pilot  # noqa: E402
import v8_calibration90_locked as locked  # noqa: E402

PLAN_SCHEMA = "v8-calibration90-B-cache-plan-v1"
RECEIPT_SCHEMA = "v8-calibration90-B-cache-receipt-v1"
FORBIDDEN_SYMBOLS = frozenset({"load_gt_transform", "load_anchor_ids"})
REACHABLE_FUNCTIONS: tuple[Callable[..., Any], ...] = (
    builder.split_manifest,
    builder.object_geometry,
    builder.full_input_provenance,
    builder.build_or_load_cache,
    builder.load_model,
    builder.build_canonical_pair,
    builder.batch_for,
    builder.official_matching,
    builder.geotransformer_forward,
)


class CachePrepareError(RuntimeError):
    pass


def _atomic_create(path: Path, value: Mapping[str, Any]) -> None:
    locked._atomic_create(path, value)


def reachable_gt_ast_audit(
        functions: tuple[Callable[..., Any], ...] = REACHABLE_FUNCTIONS) \
        -> dict[str, Any]:
    rows = []
    for function in functions:
        source = inspect.getsource(function)
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute)}
        imports = set()
        with_labels_true = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or "")
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if (keyword.arg == "with_labels"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True):
                        with_labels_true = True
        forbidden = sorted((names | attrs | imports) & FORBIDDEN_SYMBOLS)
        if forbidden or with_labels_true:
            raise CachePrepareError(
                f"GT AST audit failed {function.__module__}.{function.__name__}: "
                f"symbols={forbidden} with_labels_true={with_labels_true}")
        rows.append({
            "function": f"{function.__module__}.{function.__name__}",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        })
    return {
        "status": "PASS",
        "reachable_functions": rows,
        "forbidden_symbols": sorted(FORBIDDEN_SYMBOLS),
        "with_labels_true": False,
        "scope": "reachable cache-generation functions only; no posthoc path",
    }


def source_inventory() -> dict[str, Any]:
    paths = {
        "gate0": Path(__file__).resolve(),
        "locked_controller": (
            CODE_ROOT / "scripts/v8_calibration90_locked.py"),
        "existing_builder": CODE_ROOT / "scripts/v6fix_consistency_audit.py",
        "cache_validator": CODE_ROOT / "scripts/v7_registration_pilot.py",
        "canonical_inputs": CODE_ROOT / "scripts/canonical_inputs.py",
        "batch_builder": CODE_ROOT / "scripts/v4_train.py",
        "registration": CODE_ROOT / "scripts/v3b_cache_runner.py",
        "official_inference": (
            CODE_ROOT / "src/inference/sgf_official/inference.py"),
    }
    rows = {
        name: {"path": str(path.relative_to(CODE_ROOT)),
               "sha256": locked.sha256_file(path)}
        for name, path in paths.items()
    }
    return {"files": rows, "inventory_sha256": locked.stable_hash(rows)}


def build_plan() -> dict[str, Any]:
    pair_ids = locked.read_pairlist()
    checkpoint = builder.CHECKPOINTS["B"].resolve()
    if (not checkpoint.is_file()
            or locked.sha256_file(checkpoint) != pilot.CHECKPOINT_SHA256):
        raise CachePrepareError("frozen B checkpoint is missing or changed")
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "FROZEN",
        "split": locked.SPLIT,
        "pair_count": locked.PAIR_COUNT,
        "pair_ids_sha256": locked.pair_ids_sha256(pair_ids),
        "pairlist": {
            "path": str(locked.DEFAULT_PAIRLIST.relative_to(CODE_ROOT)),
            "sha256": locked.CANONICAL_PAIRLIST_SHA256,
            "bytes": locked.DEFAULT_PAIRLIST.stat().st_size,
        },
        "checkpoint": {"id": "B", "path": str(checkpoint),
                       "sha256": pilot.CHECKPOINT_SHA256},
        "builder": {
            "reuse": "v6fix_consistency_audit.build_or_load_cache",
            "cache_schema": pilot.CACHE_SCHEMA,
            "atomic_publication": "build sibling staging; validate; rename",
            "overwrite": False,
        },
        "source_inventory": source_inventory(),
        "gt_ast_audit": reachable_gt_ast_audit(),
        "authorization": {
            "labels": False, "workers": False, "posthoc": False,
            "fixed12": False, "official92": False,
        },
    }
    plan["payload_sha256"] = locked.stable_hash(plan)
    return plan


def validate_plan(path: Path, expected_sha: str) -> dict[str, Any]:
    locked._require_sha(expected_sha, "cache plan SHA")
    path = path.resolve()
    if not path.is_file() or locked.sha256_file(path) != expected_sha:
        raise CachePrepareError("cache plan file SHA mismatch")
    value = locked._payload(path)
    payload_sha = value.pop("payload_sha256", None)
    if payload_sha != locked.stable_hash(value):
        raise CachePrepareError("cache plan embedded SHA mismatch")
    value["payload_sha256"] = payload_sha
    if (value.get("schema") != PLAN_SCHEMA
            or value.get("status") != "FROZEN"
            or value.get("split") != locked.SPLIT
            or value.get("pair_count") != locked.PAIR_COUNT
            or value.get("pair_ids_sha256")
            != locked.CANONICAL_PAIRLIST_SHA256
            or value.get("checkpoint", {}).get("sha256")
            != pilot.CHECKPOINT_SHA256
            or value.get("checkpoint", {}).get("id") != "B"
            or Path(value.get("checkpoint", {}).get("path", "")).resolve()
            != builder.CHECKPOINTS["B"].resolve()
            or value.get("pairlist") != {
                "path": str(locked.DEFAULT_PAIRLIST.relative_to(CODE_ROOT)),
                "sha256": locked.CANONICAL_PAIRLIST_SHA256,
                "bytes": locked.DEFAULT_PAIRLIST.stat().st_size,
            }
            or value.get("builder") != {
                "reuse": "v6fix_consistency_audit.build_or_load_cache",
                "cache_schema": pilot.CACHE_SCHEMA,
                "atomic_publication":
                    "build sibling staging; validate; rename",
                "overwrite": False,
            }
            or value.get("authorization") != {
                "labels": False, "workers": False, "posthoc": False,
                "fixed12": False, "official92": False,
            }
            or value.get("gt_ast_audit", {}).get("status") != "PASS"
            or value.get("source_inventory") != source_inventory()
            or value.get("gt_ast_audit") != reachable_gt_ast_audit()):
        raise CachePrepareError("cache plan identity/source/AST drift")
    value["_path"] = str(path)
    value["_file_sha256"] = expected_sha
    return value


def _validate_cache(path: Path, pair_id: str) -> dict[str, Any]:
    before = locked.sha256_file(path)
    cache = pilot.load_validated_cache(path, pair_id, before)
    if cache["checkpoint_sha256"] != pilot.CHECKPOINT_SHA256:
        raise CachePrepareError("prepared cache checkpoint mismatch")
    if locked.sha256_file(path) != before:
        raise CachePrepareError("prepared cache changed during validation")
    return {
        "pair_id": pair_id,
        "cache_sha256": before,
        "cache_bytes": path.stat().st_size,
        "input_sha256": cache["input_sha256"],
        "embedding_sha256": cache["embedding_sha256"],
        "similarity_sha256": cache["similarity_sha256"],
        "node_corr_count": len(cache["_members"]),
        "geot_entry_count": len(cache["geot"]),
    }


def execute(plan: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise CachePrepareError("cache output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    if staging.exists():
        raise CachePrepareError("unexpected staging collision")
    staging.mkdir()
    published = False
    try:
        pair_ids = locked.read_pairlist()
        device = "cuda" if builder.torch.cuda.is_available() else "cpu"
        model = builder.load_model("B", device)
        for index, pair_id in enumerate(pair_ids, 1):
            builder.build_or_load_cache(
                pair_id, "B", model, device, staging)
            print(f"[{index}/{len(pair_ids)}] {pair_id}", flush=True)
        paths = sorted(staging.glob("*.pt"))
        if (len(paths) != locked.PAIR_COUNT
                or {path.stem for path in paths} != set(pair_ids)):
            raise CachePrepareError("prepared cache completeness mismatch")
        rows = [_validate_cache(staging / f"{pair_id}.pt", pair_id)
                for pair_id in pair_ids]
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "status": "GT_FREE_B_CACHE_COMPLETE",
            "split": locked.SPLIT,
            "plan": {"path": plan["_path"],
                     "file_sha256": plan["_file_sha256"],
                     "payload_sha256": plan["payload_sha256"]},
            "checkpoint_sha256": pilot.CHECKPOINT_SHA256,
            "pair_count": len(rows),
            "pair_ids_sha256": locked.pair_ids_sha256(pair_ids),
            "pairs": rows,
            "cache_inventory_sha256": locked.stable_hash(rows),
            "gt_ast_audit": reachable_gt_ast_audit(),
            "labels_loaded": False,
            "workers_run": False,
            "posthoc_run": False,
        }
        receipt["evidence_sha256"] = locked.stable_hash(receipt)
        _atomic_create(staging / "cache_receipt.json", receipt)
        # Atomic publication on the same filesystem; no partial final root.
        os.rename(staging, output)
        published = True
        return receipt
    finally:
        if not published and staging.exists():
            # Incomplete staging contains no labels and is never accepted by
            # the freezer.  Remove it to make the failure explicit/recoverable.
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.freeze_plan:
        if args.plan_sha256 is not None or args.out is not None:
            raise CachePrepareError("plan freeze takes only --plan")
        plan = build_plan()
        _atomic_create(args.plan.resolve(), plan)
        print(json.dumps({"plan": str(args.plan.resolve()),
                          "file_sha256": locked.sha256_file(args.plan.resolve()),
                          "payload_sha256": plan["payload_sha256"]}, indent=2))
        return 0
    if args.plan_sha256 is None or args.out is None:
        raise CachePrepareError("execute requires plan SHA and fresh --out")
    plan = validate_plan(args.plan, args.plan_sha256)
    result = execute(plan, args.out)
    print(json.dumps({"status": result["status"],
                      "pair_count": result["pair_count"],
                      "cache_inventory_sha256": result[
                          "cache_inventory_sha256"],
                      "output": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
