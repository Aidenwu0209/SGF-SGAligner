#!/usr/bin/env python3
"""Create the unsigned exact15 identity preregister and preflight closure.

The command performs only file hashing, JSON/NPZ validation, and source
snapshotting.  It never imports a model, invokes a GPU, runs a solver/ICP, or
creates a registration/fused PLY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for item in (REPO, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from safety.v16_b716_scannet15_identity import (  # noqa: E402
    PAIR_COUNT, PREFLIGHT_SCHEMA, PREREGISTER_SCHEMA,
    ScanNet15IdentityError, pair_id_for_scene, sha256_file,
    stable_json_sha256, validate_prepared_npz, validate_preregister,
    verify_preflight_closure,
)


RAW_SCHEMA = "v16-b716-scannet15-raw-pair-inventory-v1"
BRIDGE_SCHEMA = "v16-b716-scannet15-prepared-bridge-summary-v1"
SUMMARY_SCHEMA = "v16-b716-scannet15-identity-migration-summary-v1"
SOURCE_FILES = (
    "src/safety/v16_b716_scannet15_identity.py",
    "scripts/v16_b716_scannet15_corr_cache_converter.py",
    "scripts/v16_b716_scannet15_identity_preflight.py",
    "scripts/v16_b716_scannet15_v14_identity.py",
    "src/safety/v16_b716_scannet15_v13_gate_bridge.py",
    "scripts/v13_corr_cache_converter.py",
    "scripts/v13_dual_solver_cli.py",
    "src/safety/v13_strict_pair_gate.py",
    "src/safety/v14_rigid_multihypothesis.py",
    "scripts/v14_rigid_multihypothesis_builder.py",
    "scripts/v14_candidate_strict_runner.py",
)
POLICY_FALSE = {
    "execution_authorized": False, "gpu_authorized": False,
    "gt_allowed": False, "identity_fallback_allowed": False,
    "threshold_change_allowed": False, "result_selection_allowed": False,
    "reconstruction_authorized": False, "refusion_allowed": False,
    "official92_allowed": False,
}


class BuildError(RuntimeError):
    pass


def sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def load_json(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise BuildError(f"JSON is not a regular file: {path}")
    before = sha256_file(path)
    if expected_sha is not None and before != expected_sha:
        raise BuildError(f"JSON SHA mismatch: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise BuildError(f"JSON changed while reading: {path}")
    return value


def file_binding(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if (not path.is_file() or path.is_symlink()
            or not stat.S_ISREG(path.stat().st_mode)):
        raise BuildError(f"file binding is invalid: {path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise BuildError(f"file binding SHA mismatch: {path}")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest}


def directory_closure(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise BuildError(f"directory closure invalid: {root}")
    rows = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in [*dirnames, *filenames]:
            if (base / name).is_symlink():
                raise BuildError(f"directory closure has symlink: {base / name}")
        for name in filenames:
            path = base / name
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)})
    rows.sort(key=lambda row: row["path"])
    if not rows:
        raise BuildError(f"directory closure is empty: {root}")
    return {"path": str(root), "file_count": len(rows),
            "closure_sha256": stable_json_sha256(rows)}


def copy_sources(output: Path) -> tuple[dict[str, str], dict[str, str]]:
    paths, hashes = {}, {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        if not source.is_file() or source.is_symlink():
            raise BuildError(f"identity source missing: {relative}")
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        paths[relative] = str(target.resolve())
        hashes[relative] = sha256_file(target)
    return paths, hashes


def artifact_manifest(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()
                       and item.name != "artifact_manifest.sha256"):
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    payload = "\n".join(rows) + "\n"
    (root / "artifact_manifest.sha256").write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def embedded_manifest(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise BuildError("prepared manifest_json missing")
        return json.loads(str(data["manifest_json"].item()))


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise BuildError("output root must be absolute and not exist")
    output.mkdir(parents=True)
    (output / "BUILD_INCOMPLETE").write_text(
        "Identity build in progress; never an execution authorization.\n")
    raw_path = args.raw_root.resolve() / "pair_inventory.json"
    bridge_path = args.prepared_root.resolve() / "prepared_summary.json"
    raw = load_json(raw_path, args.raw_inventory_sha256)
    bridge = load_json(bridge_path, args.prepared_summary_sha256)
    if (raw.get("schema") != RAW_SCHEMA or raw.get("scan_count") != PAIR_COUNT
            or len(raw.get("pairs", ())) != PAIR_COUNT
            or not all(row.get("quality_gate_passed") is True
                       for row in raw.get("pairs", ()))
            or bridge.get("schema") != BRIDGE_SCHEMA
            or bridge.get("pair_count") != PAIR_COUNT
            or bridge.get("prepared_count") != PAIR_COUNT
            or bridge.get("formal_execution_authorized") is not False
            or bridge.get("gt_consumed") is not False):
        raise BuildError("raw/prepared bridge is not exact15 unauthorized")
    raw_by_scene = {row["scene_id"]: row for row in raw["pairs"]}
    if len(raw_by_scene) != PAIR_COUNT:
        raise BuildError("raw pair scene identity is not exact15")
    source_paths, source_hashes = copy_sources(output)

    official_sources = {}
    for relative, expected in bridge["official_source_sha256"].items():
        official_sources[relative] = file_binding(
            args.official_repo.resolve() / relative, expected)
    dependencies = {
        "official_repo": {"path": str(args.official_repo.resolve()),
                          "head": bridge["official_repo_head"],
                          "source_files": official_sources},
        "official_sgaligner_checkpoint": file_binding(
            args.official_checkpoint,
            bridge["official_checkpoint_sha256"]),
        "official_geotransformer_checkpoint": file_binding(
            args.geotransformer_checkpoint,
            bridge["geotransformer_checkpoint_sha256"]),
        "colorpcr_repo": directory_closure(args.colorpcr_root),
        "colorpcr_weights": file_binding(args.colorpcr_weights),
        "colorpcr_extension": file_binding(args.colorpcr_extension),
        "pointdsc_repo": directory_closure(args.pointdsc_root),
        "pointdsc_checkpoint": file_binding(args.pointdsc_checkpoint),
        "sgaligner_python": file_binding(args.sgaligner_python),
        "jojo_python": file_binding(args.jojo_python),
    }
    rows = []
    for bridge_row in bridge["pairs"]:
        scene = bridge_row["scene_id"]
        raw_row = raw_by_scene.get(scene)
        if raw_row is None or raw_row.get("quality_gate_passed") is not True:
            raise BuildError(f"raw pair is not quality-gated: {scene}")
        pair_id = pair_id_for_scene(scene)
        prepared_path = args.prepared_root.resolve() / bridge_row["prepared_input"]
        prepared_manifest_path = (args.prepared_root.resolve()
                                  / bridge_row["prepared_manifest"])
        prepared_manifest = load_json(
            prepared_manifest_path, bridge_row["prepared_manifest_sha256"])
        embedded = embedded_manifest(prepared_path)
        raw_receipt = args.raw_root.resolve() / raw_row["pair_receipt"]
        source_ply = args.raw_root.resolve() / raw_row["source_raw_ply"]
        reference_ply = args.raw_root.resolve() / raw_row["reference_raw_ply"]
        graph_path = Path(prepared_manifest["sgf_graph"]["predictions_path"])
        checks = (
            (prepared_path, bridge_row["prepared_input_sha256"]),
            (raw_receipt, raw_row["pair_receipt_sha256"]),
            (source_ply, raw_row["source_raw_ply_sha256"]),
            (reference_ply, raw_row["reference_raw_ply_sha256"]),
            (graph_path, prepared_manifest["sgf_graph"]["predictions_sha256"]),
        )
        for path, expected in checks:
            file_binding(path, expected)
        if (embedded.get("payload_sha256")
                != prepared_manifest["embedded_manifest_payload_sha256"]
                or prepared_manifest["raw_pair"]["inventory_sha256"]
                != args.raw_inventory_sha256
                or prepared_manifest["raw_pair"]["receipt_sha256"]
                != raw_row["pair_receipt_sha256"]):
            raise BuildError(f"prepared/raw embedded binding mismatch: {scene}")
        row = {
            "pair_id": pair_id, "scene_id": scene,
            "prepared_npz_path": str(prepared_path),
            "prepared_npz_sha256": bridge_row["prepared_input_sha256"],
            "prepared_manifest_path": str(prepared_manifest_path),
            "prepared_manifest_sha256": bridge_row["prepared_manifest_sha256"],
            "prepared_manifest_payload_sha256": embedded["payload_sha256"],
            "raw_inventory_path": str(raw_path),
            "raw_inventory_sha256": args.raw_inventory_sha256,
            "raw_pair_receipt_path": str(raw_receipt),
            "raw_pair_receipt_sha256": raw_row["pair_receipt_sha256"],
            "source_raw_ply_path": str(source_ply),
            "source_raw_ply_sha256": raw_row["source_raw_ply_sha256"],
            "reference_raw_ply_path": str(reference_ply),
            "reference_raw_ply_sha256": raw_row["reference_raw_ply_sha256"],
            "sgf_prediction_path": str(graph_path),
            "sgf_prediction_sha256":
                prepared_manifest["sgf_graph"]["predictions_sha256"],
        }
        row["identity_payload_sha256"] = stable_json_sha256(row)
        rows.append(row)
    pair_ids = [row["pair_id"] for row in rows]
    preregister = sealed({
        "schema": PREREGISTER_SCHEMA, "pair_count": PAIR_COUNT,
        "pair_ids": pair_ids, "pairs": rows,
        "raw_inventory_path": str(raw_path),
        "raw_inventory_sha256": args.raw_inventory_sha256,
        "prepared_bridge_summary_path": str(bridge_path),
        "prepared_bridge_summary_sha256": args.prepared_summary_sha256,
        "official_repo_head": bridge["official_repo_head"],
        "official_checkpoint_sha256": bridge["official_checkpoint_sha256"],
        "geotransformer_checkpoint_sha256":
            bridge["geotransformer_checkpoint_sha256"],
        "sgf_model_closure_sha256": bridge["sgf_model_closure_sha256"],
        "bridge_source_sha256": bridge["bridge_source_sha256"],
        "colorpcr_schema_source_sha256":
            bridge["colorpcr_schema_source_sha256"],
        "source_paths": source_paths, "source_sha256": source_hashes,
        "dependency_closure_sha256": stable_json_sha256(dependencies),
        "primary_arm": "sgf_selected_union",
        "selection_rule": "unchanged_v14_exact8_then_exact20_then_v15",
        "allow_real_pilot": False, "allow_gpu_pilot": False,
        "posthoc_allowed": False, "algorithm_or_threshold_change": False,
        **POLICY_FALSE,
    })
    preregister_path = output / "scannet15_identity_preregister.json"
    write_json(preregister_path, preregister)
    validate_preregister(preregister)
    write_json(output / "dependency_closure.json", sealed({
        "schema": "v16-b716-scannet15-dependency-closure-v2",
        "dependencies": dependencies, "source_paths": source_paths,
        "source_sha256": source_hashes, **POLICY_FALSE,
    }))
    preflight = sealed({
        "schema": PREFLIGHT_SCHEMA, "status": "IDENTITY_READY_UNAUTHORIZED",
        "pair_count": PAIR_COUNT, "pair_ids": pair_ids, "pairs": rows,
        "preregister_path": str(preregister_path.resolve()),
        "preregister_sha256": sha256_file(preregister_path),
        "dependency_closure_sha256": stable_json_sha256(dependencies),
        "all_pair_files_rehashed": True,
        "all_prepared_npz_validated": True,
        "solver_or_icp_executed": False, "final_ply_generated": False,
        **POLICY_FALSE,
    })
    preflight_path = output / "scannet15_identity_preflight.json"
    write_json(preflight_path, preflight)
    receipts = []
    for row in rows:
        try:
            receipt = verify_preflight_closure(
                preregister_path=preregister_path,
                preflight_path=preflight_path,
                prepared_path=Path(row["prepared_npz_path"]),
                pair_id=row["pair_id"])
            _, tensor_receipt = validate_prepared_npz(
                Path(row["prepared_npz_path"]), pair_id=row["pair_id"],
                preregister=preregister)
        except ScanNet15IdentityError as exc:
            raise BuildError(f"identity validation failed: {exc}") from exc
        receipts.append(sealed({**receipt, "tensor_validation": tensor_receipt}))
    write_json(output / "pair_validation_receipts.json", sealed({
        "schema": "v16-b716-scannet15-pair-validation-receipts-v1",
        "pair_count": PAIR_COUNT, "receipts": receipts,
        "execution_authorized": False, "gt_consumed": False,
    }))
    migration = sealed({
        "schema": "v16-b716-scannet15-identity-migration-map-v1",
        "legacy_fixed4_unchanged": True,
        "legacy_converter_path": source_paths[
            "scripts/v13_corr_cache_converter.py"],
        "legacy_converter_sha256": source_hashes[
            "scripts/v13_corr_cache_converter.py"],
        "scannet15_converter_path": source_paths[
            "scripts/v16_b716_scannet15_corr_cache_converter.py"],
        "scannet15_converter_sha256": source_hashes[
            "scripts/v16_b716_scannet15_corr_cache_converter.py"],
        "prepared_schema_from": "v13-color-preserving-pair-v2",
        "prepared_schema_to":
            "v16-b716-scannet15-official-colorpcr-input-v1",
        "identity_schema_from": "historical fixed4 pair order",
        "identity_schema_to": PREREGISTER_SCHEMA,
        "v13_identity_bridge_path": source_paths[
            "src/safety/v16_b716_scannet15_v13_gate_bridge.py"],
        "v14_identity_bridge_path": source_paths[
            "scripts/v16_b716_scannet15_v14_identity.py"],
        "v14_config_changed": False, "v13_thresholds_changed": False,
        "algorithm_changed": False, "authorization_granted": False,
    })
    write_json(output / "MIGRATION_MAP.json", migration)
    remaining = sealed({
        "schema": "v16-b716-scannet15-identity-remaining-blockers-v1",
        "status": "IDENTITY_READY_EXECUTION_BLOCKED",
        "blockers": [
            "production color manifest must bind the ScanNet15 sibling converter and preregister argument",
            "active control plane has no reviewed/signed ScanNet15 execution authorization",
            "V14 real pilot remains disabled because allow_real_pilot=false",
            "no ColorPCR cache or real parent-result SHA exists yet",
        ],
        **POLICY_FALSE,
    })
    write_json(output / "remaining_blockers.json", remaining)
    summary = sealed({
        "schema": SUMMARY_SCHEMA, "status": "EXACT15_IDENTITY_READY_UNAUTHORIZED",
        "pair_count": PAIR_COUNT,
        "preregister_path": str(preregister_path),
        "preregister_sha256": sha256_file(preregister_path),
        "preregister_payload_sha256": preregister["payload_sha256"],
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_payload_sha256": preflight["payload_sha256"],
        "validation_receipt_count": len(receipts),
        "legacy_converter_sha256": migration["legacy_converter_sha256"],
        "scannet15_converter_sha256": migration["scannet15_converter_sha256"],
        "source_sha256": source_hashes, "final_ply_generated": False,
        **POLICY_FALSE,
    })
    write_json(output / "identity_summary.json", summary)
    (output / "README.md").write_text(
        "# ScanNet15 identity-only migration\n\n"
        "Exact15 raw, prepared, graph, official source/checkpoint, ColorPCR, "
        "PointDSC, interpreter, converter, V14 runner, and strict-gate identity "
        "bindings are validated. The historical fixed4 converter is unchanged; "
        "the ScanNet15 schema uses a sibling converter. This directory is not "
        "an authorization and contains no GPU/solver/ICP/final PLY result.\n")
    (output / "BUILD_INCOMPLETE").unlink()
    manifest_sha = artifact_manifest(output)
    return {"output_root": str(output), "pair_count": PAIR_COUNT,
            "summary_sha256": sha256_file(output / "identity_summary.json"),
            "artifact_manifest_sha256": manifest_sha}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--raw-inventory-sha256", required=True)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--prepared-summary-sha256", required=True)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--geotransformer-checkpoint", required=True, type=Path)
    parser.add_argument("--colorpcr-root", required=True, type=Path)
    parser.add_argument("--colorpcr-weights", required=True, type=Path)
    parser.add_argument("--colorpcr-extension", required=True, type=Path)
    parser.add_argument("--pointdsc-root", required=True, type=Path)
    parser.add_argument("--pointdsc-checkpoint", required=True, type=Path)
    parser.add_argument("--sgaligner-python", required=True, type=Path)
    parser.add_argument("--jojo-python", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main() -> int:
    try:
        result = build(parse_args())
    except (BuildError, ScanNet15IdentityError, OSError, ValueError,
            json.JSONDecodeError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
