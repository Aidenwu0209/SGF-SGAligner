#!/usr/bin/env python3
"""Dry-run-first exact 72-key official GeoTransformer backfill runner."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO_ROOT = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official")
OFFICIAL_SOURCE_ROOT = OFFICIAL_REPO_ROOT / "src"
for value in (ROOT, ROOT / "src", ROOT / "scripts",
              ROOT / "src/inference/sgf_official"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from safety.v16_b716_candidate_plan import (  # noqa: E402
    atomic_json, canonical_boundary, file_evidence, sha256_file,
    stable_json_sha256,
)
from safety.v16_b716_geot_backfill import (  # noqa: E402
    BackfillError, SCHEMA, build_attempt_receipt, build_frozen_merge_contract,
    compare_exact_missing, derive_authorized_task_view, enforce_cuda_clean_gate,
    enforce_cuda_runtime_gate, expected_missing_rows, extract_observed_missing,
    normalise_result, payload_valid, query_cuda_snapshot, revalidate_pair,
    revalidate_runtime_registration_points,
    validate_attempt_receipt, validate_authorization,
    validate_clean_service_receipt, validate_external_authorities,
    validate_preregister, validate_resumed_result, write_tasks,
)
from safety.v16_matched_region_colorpcr import (  # noqa: E402
    array_sha256 as v16_array_sha256,
    canonical_surface_from_rows, load_raw_inseg, node_object_id,
    verify_canonical_surface,
)


PREREGISTER = ROOT / "manifests/v16_b716_geot_backfill_preregister.json"
RUNTIME_MODULE_PATHS = {
    "v16_b716_frozen_inference": (
        ROOT / "src/inference/sgf_official/inference.py"),
    "GeoTransformer.config": OFFICIAL_SOURCE_ROOT / "GeoTransformer/config.py",
    "GeoTransformer.model": OFFICIAL_SOURCE_ROOT / "GeoTransformer/model.py",
    "GeoTransformer.geotransformer.utils.data": (
        OFFICIAL_SOURCE_ROOT / "GeoTransformer/geotransformer/utils/data.py"),
    "engine.registration_evaluator": (
        OFFICIAL_SOURCE_ROOT / "engine/registration_evaluator.py"),
    "utils.torch_util": OFFICIAL_REPO_ROOT / "utils/torch_util.py",
}


def immutable_runtime_source_bundle() -> list[dict[str, Any]]:
    roots = [
        ("local_project", ROOT),
        ("official_project", OFFICIAL_REPO_ROOT),
    ]
    files: list[tuple[str, Path]] = [
        ("runner", ROOT / "scripts/v16_b716_geot_backfill.py"),
        ("canonical_inputs", ROOT / "scripts/canonical_inputs.py"),
    ]
    for label, directory in roots:
        discovered = sorted(directory.rglob("*.py")) if directory.is_dir() else []
        if not discovered:
            raise BackfillError(f"runtime source bundle root is absent: {directory}")
        files.extend((label, path) for path in discovered)
    rows, seen = [], set()
    for label, path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(file_evidence(
            resolved, f"immutable_runtime_source_bundle:{label}"))
    return sorted(rows, key=lambda row: (row["path"], row["role"]))


def runtime_module_entrypoints() -> list[dict[str, Any]]:
    rows = []
    for module_name, path in sorted(RUNTIME_MODULE_PATHS.items()):
        evidence = file_evidence(path, "runtime_module_entrypoint")
        rows.append({"module": module_name, **evidence})
    return rows


def validate_runtime_module_resolution(
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    expected = preflight.get("runtime_module_entrypoints")
    if (not isinstance(expected, list)
            or len(expected) != len(RUNTIME_MODULE_PATHS)
            or preflight.get("runtime_module_entrypoint_closure_sha256")
            != stable_json_sha256(expected)):
        raise BackfillError("runtime module entrypoint closure is absent")
    expected_by_name = {row.get("module"): row for row in expected}
    if set(expected_by_name) != set(RUNTIME_MODULE_PATHS):
        raise BackfillError("runtime module entrypoint set changed")
    runtime_bundle = preflight.get("runtime_source_bundle")
    bundle_files = {
        (row.get("path"), row.get("bytes"), row.get("sha256"))
        for row in runtime_bundle or []
    }
    for row in expected:
        if ((row.get("path"), row.get("bytes"), row.get("sha256"))
                not in bundle_files):
            raise BackfillError("runtime module entrypoint escaped source bundle")

    # Only after the clean CUDA gate, force official GeoTransformer/engine/utils
    # ahead of identically named local packages. Local safety/adapters are
    # already imported and independently frozen in the immutable bundle.
    for root in (OFFICIAL_REPO_ROOT, OFFICIAL_SOURCE_ROOT):
        while str(root) in sys.path:
            sys.path.remove(str(root))
        sys.path.insert(0, str(root))
    modules = {}
    for name in sorted(RUNTIME_MODULE_PATHS):
        row = expected_by_name[name]
        path = Path(row.get("path", ""))
        if name == "v16_b716_frozen_inference":
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise BackfillError("frozen inference module cannot be loaded")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(name)
        actual = Path(getattr(module, "__file__", "")).resolve()
        if (actual != path.resolve() or actual != RUNTIME_MODULE_PATHS[name].resolve()
                or not actual.is_file()
                or actual.stat().st_size != int(row.get("bytes", -1))
                or sha256_file(actual) != row.get("sha256")):
            raise BackfillError(f"runtime module resolved outside source bundle: {name}")
        modules[name] = module

    allowed_paths = {Path(row[0]).resolve() for row in bundle_files}
    scoped_roots = (ROOT.resolve(), OFFICIAL_REPO_ROOT.resolve())
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path)
        # PyTorch exposes synthetic modules such as torch.classes with a
        # relative pseudo-file; they are not project source files.
        if not path.is_absolute():
            continue
        path = path.resolve()
        if path.suffix not in {".py", ".pyc"}:
            continue
        source_path = path.with_suffix(".py") if path.suffix == ".pyc" else path
        if (any(source_path.is_relative_to(root) for root in scoped_roots)
                and source_path not in allowed_paths):
            raise BackfillError(
                f"executed project module escaped source bundle: {module_name}")
    return modules


def raw_bindings(data: Mapping[str, Any], pair_id: str) -> list[dict[str, Any]]:
    src_count = int(data["src_count"])
    plan_rows = []
    # Candidate plan already froze one unique raw path per side and node.  The
    # caller supplies it via the current canonical data's provenance closure;
    # recover the same authoritative path from the adapter helper.
    from adapters.sgf.data_sources import _source_inseg_cloud

    src_scan, ref_scan = pair_id.split("_to_")
    raws = {
        "source": load_raw_inseg(
            _source_inseg_cloud(src_scan), scan_id=src_scan, side="source"),
        "reference": load_raw_inseg(
            _source_inseg_cloud(ref_scan), scan_id=ref_scan, side="reference"),
    }
    for node in range(len(data["obj_ids"])):
        side = "source" if node < src_count else "reference"
        oid = node_object_id(data, node, side=side)
        indices, reconstructed = canonical_surface_from_rows(raws[side], oid)
        surface_sha = verify_canonical_surface(data, node, reconstructed)
        plan_rows.append({
            "node_index": node, "side": side, "scan_id": raws[side].scan_id,
            "object_id": oid, "raw_inseg_path": str(raws[side].path),
            "raw_inseg_sha256": raws[side].file_sha256,
            "raw_row_count": len(indices),
            "raw_row_indices_sha256": v16_array_sha256(indices),
            "canonical_registration_surface_sha256": surface_sha,
            "canonical_registration_points": len(reconstructed),
        })
    return plan_rows


def build_preflight(output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict]:
    preregister = json.loads(PREREGISTER.read_text())
    expected = validate_preregister(preregister)
    candidate_path = ROOT / preregister["candidate_manifest_path"]
    if sha256_file(candidate_path) != preregister["candidate_manifest_sha256"]:
        raise BackfillError("candidate fixed4 manifest SHA mismatch")
    candidate = json.loads(candidate_path.read_text())
    if (candidate.get("official_release_domain_matched") is not True
            or candidate.get("legacy_B_ep20_or_89ed_consumed") is not False
            or candidate.get("geot_missing_disabled") != 72
            or candidate.get("new_geot_executed") != 0
            or candidate.get("official92_executed") is not False):
        raise BackfillError("candidate manifest is not frozen 72-key b716 input")
    candidate_root = candidate_path.parent
    authority_sources, authority_summary = validate_external_authorities(
        preregister, candidate, candidate_root)
    observed = extract_observed_missing(candidate_root, candidate)
    compare_exact_missing(expected, observed)
    all_tasks, sources = [], [
        file_evidence(PREREGISTER, "frozen_backfill_preregistration"),
        file_evidence(candidate_path, "frozen_candidate_manifest"),
        *authority_sources,
    ]
    by_short: dict[str, list[dict[str, Any]]] = {}
    for row in observed:
        by_short.setdefault(row["short_id"], []).append(row)
    for short_id in preregister["expected_missing_node_pairs_by_short_id"]:
        tasks, pair_sources = revalidate_pair(
            by_short[short_id], candidate_root,
            build_canonical_pair=build_canonical_pair,
            canonical_boundary=canonical_boundary,
            raw_binding_builder=raw_bindings)
        all_tasks.extend(tasks)
        sources.extend(pair_sources)
    if len(all_tasks) != 72:
        raise BackfillError("revalidated task count is not exact 72")
    task_rows = write_tasks(all_tasks, output_root)
    runtime_bundle = immutable_runtime_source_bundle()
    module_entrypoints = runtime_module_entrypoints()
    sources.extend(runtime_bundle)
    sources = sorted(sources, key=lambda row: (row["path"], row["role"]))
    artifacts = sorted([{
        "path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"],
        "role": "per_key_atomic_task",
    } for row in task_rows], key=lambda row: row["path"])
    manifest = {
        "schema": SCHEMA, "stage": "dry_run_preflight",
        "frozen": True, "disabled": preregister["disabled"],
        "real_execution_attempted": False,
        "new_geot_executed": 0, "official92_executed": False,
        "exact_batch_only": True, "key_selection_allowed": False,
        "result_based_selection_allowed": False,
        "candidate_manifest_path": str(candidate_path.resolve()),
        "candidate_manifest_sha256": preregister["candidate_manifest_sha256"],
        "authoritative_upstream_audit": authority_summary,
        "missing_key_count": len(all_tasks),
        "missing_key_closure_sha256": stable_json_sha256(expected),
        "task_count": len(task_rows), "tasks": task_rows,
        "task_closure_sha256": stable_json_sha256(task_rows),
        "source_closure": sources,
        "recursive_source_closure_sha256": stable_json_sha256(sources),
        "immutable_runtime_source_bundle_sha256": stable_json_sha256(
            runtime_bundle),
        "runtime_source_bundle": runtime_bundle,
        "runtime_module_entrypoints": module_entrypoints,
        "runtime_module_entrypoint_closure_sha256": stable_json_sha256(
            module_entrypoints),
        "artifact_closure": artifacts,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifacts),
        "future_merge_contract": build_frozen_merge_contract(
            candidate_root, candidate, preregister),
        "execution_derivation_contract": preregister["execution_contract"][
            "authorization_derivation_contract"],
        "authorization_state": (
            "review_required_execution_disabled" if preregister["disabled"]
            else "enabled_preregister_requires_bound_authorization"),
        "forbidden_inputs": [
            "GT/selection/evaluation labels", "pair combos/node metrics",
            "posthoc", "official92", "fallbacks", "result-based selection",
        ],
    }
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    path = output_root / "preflight_manifest.json"
    atomic_json(path, manifest)
    return manifest, all_tasks, preregister


def current_execution_context(
    *, preflight_path: Path, output_root: Path, authorization_path: Path,
    authorization_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    preregister_sha256 = sha256_file(PREREGISTER)
    preregister = json.loads(PREREGISTER.read_text())
    if sha256_file(PREREGISTER) != preregister_sha256:
        raise BackfillError("preregister changed during per-key validation")
    validate_preregister(preregister)
    preflight_sha256 = sha256_file(preflight_path)
    preflight = json.loads(preflight_path.read_text())
    preregister_sources = [
        row for row in preflight.get("source_closure", [])
        if row.get("role") == "frozen_backfill_preregistration"]
    runtime_bundle = sorted([
        row for row in preflight.get("source_closure", [])
        if str(row.get("role", "")).startswith(
            "immutable_runtime_source_bundle:")
    ], key=lambda row: (row["path"], row["role"]))
    module_entrypoints = preflight.get("runtime_module_entrypoints", [])
    bundle_files = {
        (row.get("path"), row.get("bytes"), row.get("sha256"))
        for row in runtime_bundle
    }
    if (sha256_file(preflight_path) != preflight_sha256
            or preflight.get("schema") != SCHEMA
            or preflight.get("task_count") != 72
            or preflight.get("missing_key_count") != 72
            or preflight.get("disabled") != preregister["disabled"]
            or preflight.get("execution_derivation_contract")
            != preregister["execution_contract"][
                "authorization_derivation_contract"]
            or len(preregister_sources) != 1
            or preregister_sources[0].get("sha256") != preregister_sha256
            or stable_json_sha256(preflight.get("source_closure", []))
            != preflight.get("recursive_source_closure_sha256")
            or stable_json_sha256(preflight.get("artifact_closure", []))
            != preflight.get("recursive_artifact_closure_sha256")
            or stable_json_sha256(preflight.get("tasks", []))
            != preflight.get("task_closure_sha256")
            or stable_json_sha256(runtime_bundle)
            != preflight.get("immutable_runtime_source_bundle_sha256")
            or preflight.get("runtime_source_bundle") != runtime_bundle
            or stable_json_sha256(module_entrypoints)
            != preflight.get("runtime_module_entrypoint_closure_sha256")
            or {row.get("module") for row in module_entrypoints}
            != set(RUNTIME_MODULE_PATHS)
            or any((row.get("path"), row.get("bytes"), row.get("sha256"))
                   not in bundle_files for row in module_entrypoints)
            or not runtime_bundle
            or not payload_valid(preflight)):
        raise BackfillError("preflight changed or is invalid at execution boundary")
    for row in runtime_bundle:
        path = Path(row["path"])
        if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                or sha256_file(path) != row["sha256"]):
            raise BackfillError(f"immutable runtime source changed: {path}")
    authorization = validate_authorization(
        authorization_path, authorization_sha256, preregister,
        candidate_manifest_sha256=preflight["candidate_manifest_sha256"],
        missing_closure_sha256=preflight["missing_key_closure_sha256"],
        preregister_sha256=preregister_sha256,
        preflight_manifest_sha256=preflight_sha256,
        preflight_payload_sha256=preflight["payload_sha256"],
        recursive_source_closure_sha256=preflight[
            "recursive_source_closure_sha256"],
        recursive_artifact_closure_sha256=preflight[
            "recursive_artifact_closure_sha256"],
        task_closure_sha256=preflight["task_closure_sha256"],
        immutable_runtime_source_bundle_sha256=preflight[
            "immutable_runtime_source_bundle_sha256"],
        runtime_module_entrypoint_closure_sha256=preflight[
            "runtime_module_entrypoint_closure_sha256"],
        output_root=output_root)
    binding = {
        "authorization_sha256": authorization_sha256,
        "preregister_sha256": preregister_sha256,
        "preflight_manifest_sha256": preflight_sha256,
        "preflight_payload_sha256": preflight["payload_sha256"],
        "recursive_source_closure_sha256": preflight[
            "recursive_source_closure_sha256"],
        "recursive_artifact_closure_sha256": preflight[
            "recursive_artifact_closure_sha256"],
        "task_closure_sha256": preflight["task_closure_sha256"],
        "immutable_runtime_source_bundle_sha256": preflight[
            "immutable_runtime_source_bundle_sha256"],
        "runtime_module_entrypoint_closure_sha256": preflight[
            "runtime_module_entrypoint_closure_sha256"],
        "cuda_device_uuid": authorization["cuda_device_uuid"],
    }
    return preregister, preflight, authorization, binding


def execute_exact_batch(
    tasks: list[dict[str, Any]], preregister: dict[str, Any], output_root: Path,
    authorization_path: Path, authorization_sha256: str,
) -> dict[str, Any]:
    preflight_path = output_root / "preflight_manifest.json"
    preregister, preflight, authorization, binding = current_execution_context(
        preflight_path=preflight_path, output_root=output_root,
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256)
    snapshot = query_cuda_snapshot()
    enforce_cuda_clean_gate(snapshot, authorization, preregister)
    # Delayed exact-path import: no GeoT/model code is imported before the
    # initial zero-process CUDA hard gate.
    modules = validate_runtime_module_resolution(preflight)
    geotransformer_forward = modules[
        "v16_b716_frozen_inference"].geotransformer_forward

    results = []
    for task in tasks:
        # Authorization and both receipt expiries are re-read before every key.
        preregister, preflight, authorization, binding = current_execution_context(
            preflight_path=preflight_path, output_root=output_root,
            authorization_path=authorization_path,
            authorization_sha256=authorization_sha256)
        validate_clean_service_receipt(
            authorization, preregister,
            expected_cuda_device_uuid=binding["cuda_device_uuid"])
        snapshot = query_cuda_snapshot()
        enforce_cuda_runtime_gate(
            snapshot, authorization, preregister, current_pid=os.getpid())
        node_pair = task["node_pair"]
        task_id = f"{task['short_id']}__{node_pair[0]}_{node_pair[1]}"
        directory = output_root / "tasks" / task_id
        authorized_view = derive_authorized_task_view(task, binding, preregister)
        authorized_view_path = directory / "authorized_task_view.json"
        atomic_json(authorized_view_path, authorized_view)
        authorized_view_sha256 = sha256_file(authorized_view_path)
        result_path = directory / "result.json"
        attempt = directory / "attempt_receipt.json"
        if result_path.exists():
            result = validate_resumed_result(
                result_path, task, attempt_receipt_path=attempt,
                execution_binding=binding,
                authorized_task_view_sha256=authorized_view_sha256)
            results.append({"task_id": task_id, "status": result["status"],
                            "resumed": True,
                            "attempt_receipt_sha256": result[
                                "attempt_receipt_sha256"],
                            "result_sha256": sha256_file(result_path)})
            continue
        if attempt.exists():
            validate_attempt_receipt(
                attempt, task, authorized_view_sha256, binding)
            raise BackfillError(
                f"ambiguous interrupted attempt cannot auto-rerun: {task_id}")
        data, labels = build_canonical_pair(task["pair_id"], with_labels=False)
        if labels:
            raise BackfillError("canonical builder returned prohibited labels")
        fresh_bindings = raw_bindings(data, task["pair_id"])
        source, reference = revalidate_runtime_registration_points(
            task, data, fresh_bindings,
            array_fingerprint=v16_array_sha256)
        refreshed_preregister, _refreshed_preflight, refreshed_authorization, refreshed_binding = (
            current_execution_context(
                preflight_path=preflight_path, output_root=output_root,
                authorization_path=authorization_path,
                authorization_sha256=authorization_sha256))
        if refreshed_binding != binding:
            raise BackfillError("execution binding changed during fresh surface rebuild")
        validate_clean_service_receipt(
            refreshed_authorization, refreshed_preregister,
            expected_cuda_device_uuid=binding["cuda_device_uuid"])
        snapshot = query_cuda_snapshot()
        enforce_cuda_runtime_gate(
            snapshot, refreshed_authorization, refreshed_preregister,
            current_pid=os.getpid())
        receipt = build_attempt_receipt(
            task, authorized_view_sha256, binding, snapshot)
        atomic_json(attempt, receipt)
        attempt_sha256 = sha256_file(attempt)
        try:
            status, value = geotransformer_forward(source, reference, device="cuda")
        except Exception as exc:  # typed fail-closed per-key result
            status, value = "geotransformer_runtime_error", {
                "exception_type": type(exc).__name__, "reason": str(exc)[:500]}
        result = normalise_result(
            status, value, task, directory, execution_binding=binding,
            attempt_receipt_sha256=attempt_sha256,
            authorized_task_view_sha256=authorized_view_sha256)
        atomic_json(result_path, result)
        results.append({"task_id": task_id, "status": result["status"],
                        "resumed": False,
                        "attempt_receipt_sha256": attempt_sha256,
                        "result_sha256": sha256_file(result_path)})
    if len(results) != 72 or [row["task_id"] for row in results] != [
            f"{task['short_id']}__{task['node_pair'][0]}_{task['node_pair'][1]}"
            for task in tasks]:
        raise BackfillError("exact-batch execution/result order changed")
    aggregate = {
        "schema": "v16-b716-geot-backfill-batch-result-v1",
        "exact_batch_count": 72, "selector_eligible": False,
        "result_based_selection_allowed": False, "results": results,
        "execution_binding": binding,
        "attempt_receipt_closure_sha256": stable_json_sha256([
            {"task_id": row["task_id"],
             "attempt_receipt_sha256": row["attempt_receipt_sha256"]}
            for row in results]),
    }
    aggregate["payload_sha256"] = stable_json_sha256(aggregate)
    atomic_json(output_root / "batch_result.json", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args()
    manifest, tasks, preregister = build_preflight(args.output_root)
    if args.execute:
        if args.authorization is None or not args.authorization_sha256:
            raise BackfillError("--execute requires authorization path and SHA")
        execute_exact_batch(tasks, preregister, args.output_root,
                            args.authorization, args.authorization_sha256)
    print(json.dumps({
        "preflight_manifest": str(args.output_root / "preflight_manifest.json"),
        "preflight_manifest_sha256": sha256_file(
            args.output_root / "preflight_manifest.json"),
        "task_count": manifest["task_count"],
        "new_geot_executed": 0 if not args.execute else None,
        "execution_attempted": bool(args.execute),
        "authorization_state": manifest["authorization_state"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
