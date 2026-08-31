"""Fail-closed merger for the frozen b716 fixed4 GeoTransformer closure.

This module never executes a model.  It seals exactly the 119 immutable cache
entries and the 72 independently authorized backfill results in the original
191-candidate order.  The resulting correspondence store is consumable only
through the 34 frozen matched-region hypotheses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v16_b716_candidate_plan import (
    OFFICIAL_RELEASE_SHA256, atomic_json, array_sha256, sha256_file, stable_json_sha256,
    write_deterministic_npz,
)
from safety.v16_b716_geot_backfill import (
    AUTH_SCHEMA, AUTHORIZED_TASK_SCHEMA, RESULT_SCHEMA, TASK_SCHEMA,
    BackfillError, derive_authorized_task_view, payload_valid,
    reject_forbidden_result_fields, validate_attempt_receipt,
    validate_execution_binding, validate_preregister,
    validate_resumed_result,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "v16-b716-exact191-merged-manifest-v1"
PAIR_SCHEMA = "v16-b716-exact191-pair-v1"
HYPOTHESIS_SCHEMA = "v16-b716-frozen-hypothesis-allowlist-v1"
BATCH_SCHEMA = "v16-b716-geot-backfill-batch-result-v1"
ATTEMPT_SCHEMA = "v16-b716-geot-attempt-receipt-v1"
EXPECTED_SHORT_IDS = (
    "09582205_1883", "68bae76c_5364",
    "f38169cf_56fe", "6a36052f_c2b5",
)
EXPECTED_CANDIDATES = (48, 48, 48, 47)
EXPECTED_EXISTING = (46, 27, 27, 19)
EXPECTED_MISSING = (2, 21, 21, 28)
EXPECTED_HYPOTHESES = (12, 8, 2, 12)
EXPECTED_ORDERED_KEY_CLOSURE_SHA256 = (
    "572634917937d79b88a1ba4e99ea34e68e3fa5b0e5401567fec308d3b48ef6b4"
)
EXPECTED_TYPED_FAILURES = 16
EXPECTED_HYPOTHESES_WITH_TYPED_FAILURES = 8
ALLOWED_NEW_TYPED_FAILURES = frozenset({"insufficient_post_voxel_points"})
FORBIDDEN_TOKENS = (
    "gt", "ground_truth", "official92", "selector", "selected",
    "selection", "outcome", "combos", "node_metrics", "posthoc",
)


class Exact191Error(BackfillError):
    """An exact191 source, result, ordering, or consumer contract is invalid."""


def _candidate_repository_root(candidate_path: Path) -> Path:
    """Return the clean execution repository bound by the candidate manifest.

    The merger may be reviewed and executed from a successor worktree after
    exact72 has finished.  Its authorization still belongs to the immutable
    repository that produced the candidate/preflight/result closure.  Resolve
    that repository from the already SHA-bound candidate path instead of
    silently rebinding the authorization to the merger implementation checkout.
    """
    candidate_path = Path(candidate_path)
    if (not candidate_path.is_file() or candidate_path.is_symlink()
            or candidate_path.absolute() != candidate_path.resolve()):
        raise Exact191Error("candidate manifest path is not canonical")
    try:
        raw_root = subprocess.run(
            ["git", "-C", str(candidate_path.parent),
             "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise Exact191Error(
            "candidate execution repository cannot be resolved") from exc
    repo_root = Path(raw_root).resolve()
    try:
        candidate_path.resolve().relative_to(repo_root)
    except ValueError as exc:
        raise Exact191Error(
            "candidate manifest is outside its execution repository") from exc
    return repo_root


def _expected_missing_in_preregistered_order(
        preregister: Mapping[str, Any]) -> list[tuple[str, list[int]]]:
    """Flatten exact72 keys using the explicit preregistered business order.

    JSON object ordering is serialization detail.  The signed contract carries
    ``enable_scope.expected_short_id_order`` specifically so canonical
    ``sort_keys=True`` serialization cannot change exact72 semantics.
    """
    mapping = preregister.get("expected_missing_node_pairs_by_short_id")
    order = preregister.get("enable_scope", {}).get(
        "expected_short_id_order")
    if (not isinstance(mapping, Mapping)
            or order != list(EXPECTED_SHORT_IDS)
            or set(mapping) != set(EXPECTED_SHORT_IDS)
            or len(mapping) != len(EXPECTED_SHORT_IDS)):
        raise Exact191Error("preregistered short-id order/set mismatch")
    flattened: list[tuple[str, list[int]]] = []
    identities = set()
    counts = []
    for short_id in order:
        pairs = mapping.get(short_id)
        if not isinstance(pairs, list):
            raise Exact191Error("preregistered missing-pair table is malformed")
        counts.append(len(pairs))
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int or value < 0
                           for value in pair)):
                raise Exact191Error(
                    "preregistered missing-pair identity is malformed")
            identity = (short_id, tuple(pair))
            if identity in identities:
                raise Exact191Error("duplicate preregistered missing-pair identity")
            identities.add(identity)
            flattened.append((short_id, pair))
    if counts != list(EXPECTED_MISSING) or len(flattened) != 72:
        raise Exact191Error("preregistered exact72 count/distribution mismatch")
    return flattened


def _ordered_task_id_closure_sha256(preflight: Mapping[str, Any]) -> str:
    """Recompute the authorization's ordered72 closure from exact task IDs."""
    rows = preflight.get("tasks")
    if (not isinstance(rows, list) or len(rows) != 72
            or any(not isinstance(row, Mapping)
                   or not isinstance(row.get("task_id"), str)
                   or not row["task_id"] for row in rows)):
        raise Exact191Error("preflight ordered exact72 task IDs are malformed")
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != 72:
        raise Exact191Error("preflight ordered exact72 task IDs contain duplicates")
    return stable_json_sha256(task_ids)


def _validate_bound_execution_authorization(
    *, authorization_path: Path, authorization_sha256: str,
    preregister: Mapping[str, Any], execution_repo_root: Path,
    candidate_manifest_sha256: str, missing_closure_sha256: str,
    preregister_sha256: str, preflight_manifest_sha256: str,
    preflight_payload_sha256: str, recursive_source_closure_sha256: str,
    recursive_artifact_closure_sha256: str, task_closure_sha256: str,
    immutable_runtime_source_bundle_sha256: str,
    runtime_module_entrypoint_closure_sha256: str, output_root: Path,
    future_merge_contract_sha256: str, ordered72_sha256: str,
) -> dict[str, Any]:
    """Validate execution evidence in the repository that signed it.

    Audit-authority contracts intentionally pin the absolute public-key path.
    A reviewed merger successor therefore must not reinterpret an old receipt
    with its own module-local key path.  Run the exact validator from the clean,
    SHA-bound execution repository in an isolated interpreter and compare its
    returned document with an independent immutable read in this process.
    """
    execution_repo_root = Path(execution_repo_root).resolve()
    validator_path = (
        execution_repo_root / "src/safety/v16_b716_geot_backfill.py")
    if (not validator_path.is_file() or validator_path.is_symlink()
            or validator_path.resolve() != validator_path):
        raise Exact191Error("bound authorization validator path is invalid")
    kwargs = {
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "missing_closure_sha256": missing_closure_sha256,
        "preregister_sha256": preregister_sha256,
        "preflight_manifest_sha256": preflight_manifest_sha256,
        "preflight_payload_sha256": preflight_payload_sha256,
        "recursive_source_closure_sha256": recursive_source_closure_sha256,
        "recursive_artifact_closure_sha256": recursive_artifact_closure_sha256,
        "task_closure_sha256": task_closure_sha256,
        "immutable_runtime_source_bundle_sha256": (
            immutable_runtime_source_bundle_sha256),
        "runtime_module_entrypoint_closure_sha256": (
            runtime_module_entrypoint_closure_sha256),
        "output_root": str(Path(output_root).resolve()),
        "repo_root": str(execution_repo_root),
        "future_merge_contract_sha256": future_merge_contract_sha256,
        "ordered72_sha256": ordered72_sha256,
    }
    payload = {
        "authorization_path": str(Path(authorization_path).resolve()),
        "authorization_sha256": authorization_sha256,
        "preregister": preregister,
        "repo_root": str(execution_repo_root),
        "validator_path": str(validator_path),
        "kwargs": kwargs,
    }
    child = r"""
import json
from pathlib import Path
import sys
payload = json.load(sys.stdin)
sys.path.insert(0, str(Path(payload["repo_root"]) / "src"))
from safety import v16_b716_geot_backfill as module
if Path(module.__file__).resolve() != Path(payload["validator_path"]):
    raise SystemExit("foreign authorization validator imported")
kwargs = dict(payload["kwargs"])
kwargs["output_root"] = Path(kwargs["output_root"])
kwargs["repo_root"] = Path(kwargs["repo_root"])
value = module.validate_authorization(
    Path(payload["authorization_path"]),
    payload["authorization_sha256"], payload["preregister"], **kwargs)
sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
"""
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-s", "-c", child],
            input=json.dumps(payload, sort_keys=True), text=True,
            capture_output=True, check=True, cwd=execution_repo_root,
            env=environment,
        )
        validated = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        detail = (getattr(exc, "stderr", "") or "")[-500:]
        raise Exact191Error(
            f"bound execution authorization validation failed: {detail}") from exc
    expected = _load_json(
        authorization_path, authorization_sha256, "execution authorization")
    if validated != expected:
        raise Exact191Error("bound authorization validator returned foreign data")
    return validated


def _load_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    path = Path(path)
    if (not path.is_file() or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or sha256_file(path) != expected_sha256):
        raise Exact191Error(f"{role} SHA/path mismatch")
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise Exact191Error(f"{role} is malformed JSON") from exc
    if sha256_file(path) != before or not isinstance(value, dict):
        raise Exact191Error(f"{role} changed during read")
    return value


def _resolve_inside(root: Path, relative: str, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise Exact191Error(f"{role} relative path is absent")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Exact191Error(f"{role} escapes its frozen root") from exc
    return path


def _verify_file_row(root: Path, row: Mapping[str, Any], role: str,
                     *, absolute: bool = False) -> Path:
    raw = row.get("path")
    path = Path(str(raw)).resolve() if absolute else _resolve_inside(
        root, str(raw), role)
    if (not path.is_file() or type(row.get("bytes")) is not int
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row.get("sha256")):
        raise Exact191Error(f"{role} bytes/SHA mismatch")
    return path


def _verify_source_closure(rows: Any, expected_sha256: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise Exact191Error("source closure is absent")
    if stable_json_sha256(rows) != expected_sha256:
        raise Exact191Error("recursive source closure SHA mismatch")
    identities = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise Exact191Error("source closure row is malformed")
        path = _verify_file_row(Path("/"), row, "source closure", absolute=True)
        identity = (str(path), str(row.get("role")))
        if identity in identities:
            raise Exact191Error("duplicate source closure row")
        identities.add(identity)


def _entry_payload_valid(entry: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in entry.items()
               if key != "entry_sha256"}
    return entry.get("entry_sha256") == stable_json_sha256(payload)


def _load_npz(path: Path, expected_sha256: str,
              expected_bytes: int) -> dict[str, np.ndarray]:
    if (not path.is_file() or path.stat().st_size != int(expected_bytes)
            or sha256_file(path) != expected_sha256):
        raise Exact191Error("immutable NPZ SHA/bytes mismatch")
    before = sha256_file(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.ascontiguousarray(archive[name])
                      for name in archive.files}
    except Exception as exc:
        raise Exact191Error("immutable NPZ cannot be decoded safely") from exc
    if sha256_file(path) != before:
        raise Exact191Error("immutable NPZ changed during read")
    return arrays


def _validate_corr_arrays(arrays: Mapping[str, np.ndarray], prefix: str,
                          declared: Mapping[str, Any] | None = None
                          ) -> dict[str, np.ndarray]:
    names = {field: f"{field}_{prefix}"
             for field in ("src_corr", "ref_corr", "scores")}
    if any(name not in arrays for name in names.values()):
        raise Exact191Error("correspondence arrays are incomplete")
    output = {field: np.ascontiguousarray(arrays[name])
              for field, name in names.items()}
    src, ref, scores = output["src_corr"], output["ref_corr"], output["scores"]
    if (src.ndim != 2 or src.shape[1:] != (3,) or ref.shape != src.shape
            or scores.shape != (len(src),) or len(src) == 0
            or any(not value.dtype == np.float32 for value in output.values())
            or any(not np.isfinite(value).all() for value in output.values())):
        raise Exact191Error("correspondence arrays are malformed")
    if declared is not None:
        if (not isinstance(declared, Mapping)
                or set(declared) != set(output)):
            raise Exact191Error("correspondence array declaration mismatch")
        for field, value in output.items():
            row = declared.get(field)
            if (not isinstance(row, Mapping)
                    or set(row) != {"shape", "dtype", "sha256"}
                    or row.get("shape") != list(value.shape)
                    or row.get("dtype") != str(value.dtype)
                    or row.get("sha256") != array_sha256(value)):
                raise Exact191Error("correspondence array declaration mismatch")
    return output


def validate_preflight(preflight: Mapping[str, Any], root: Path,
                       preregister: Mapping[str, Any],
                       preregister_path: Path,
                       preregister_sha256: str) -> list[tuple[dict, Path]]:
    reject_forbidden_result_fields(preflight)
    # The current enabled-preregister contract deliberately carries
    # ``enable_scope.selected = false`` to prove that authorization did not
    # select a task or outcome.  It is a control-plane invariant validated by
    # validate_preregister(), not a model/result field.  Applying the recursive
    # result-field denylist to the preregister makes every valid enabled
    # exact72 receipt impossible to merge.
    validate_preregister(preregister)
    if (preflight.get("schema") != "v16-b716-geot-backfill-preflight-v1"
            or preflight.get("frozen") is not True
            or preflight.get("disabled") is not False
            or preregister.get("disabled") is not False
            or preregister.get("execution_contract", {}).get(
                "real_execution_allowed") is not True
            or preflight.get("execution_derivation_contract")
            != preregister.get("execution_contract", {}).get(
                "authorization_derivation_contract")
            or preflight.get("exact_batch_only") is not True
            or preflight.get("key_selection_allowed") is not False
            or preflight.get("result_based_selection_allowed") is not False
            or preflight.get("official92_executed") is not False
            or preflight.get("task_count") != 72
            or preflight.get("missing_key_count") != 72
            or not payload_valid(preflight)):
        raise Exact191Error("preflight contract mismatch")
    rows = preflight.get("tasks")
    if (not isinstance(rows, list) or len(rows) != 72
            or stable_json_sha256(rows) != preflight.get("task_closure_sha256")):
        raise Exact191Error("preflight task table is not exact 72")
    _verify_source_closure(
        preflight.get("source_closure"),
        str(preflight.get("recursive_source_closure_sha256", "")))
    prereg_matches = [row for row in preflight["source_closure"]
                       if row.get("role") == "frozen_backfill_preregistration"]
    if (len(prereg_matches) != 1
            or Path(prereg_matches[0]["path"]).resolve()
            != Path(preregister_path).resolve()
            or prereg_matches[0].get("sha256") != preregister_sha256):
        raise Exact191Error("preflight does not bind the supplied preregistration")
    expected = []
    task_ids = set()
    for row in rows:
        path = _verify_file_row(root, row, "preflight task")
        task = _load_json(path, row["sha256"], "preflight task")
        reject_forbidden_result_fields(task)
        if (task.get("schema") != TASK_SCHEMA
                or task.get("state") != "planned_disabled"
                or task.get("execution_authorized") is not False
                or task.get("task_sha256")
                != stable_json_sha256({key: value for key, value in task.items()
                                       if key != "task_sha256"})
                or task.get("task_sha256") != row.get("task_sha256")):
            raise Exact191Error("task contract or task SHA mismatch")
        task_id = f"{task['short_id']}__{task['node_pair'][0]}_{task['node_pair'][1]}"
        if task_id != row.get("task_id") or task_id in task_ids:
            raise Exact191Error("task identity is duplicated or foreign")
        task_ids.add(task_id)
        expected.append((task, path.parent))
    artifact_rows = preflight.get("artifact_closure")
    if (not isinstance(artifact_rows, list)
            or stable_json_sha256(artifact_rows)
            != preflight.get("recursive_artifact_closure_sha256")
            or sorted((row["path"], row["bytes"], row["sha256"])
                      for row in artifact_rows)
            != sorted((row["path"], row["bytes"], row["sha256"])
                      for row in rows)):
        raise Exact191Error("preflight task artifact closure mismatch")
    runtime_bundle = sorted([
        row for row in preflight.get("source_closure", [])
        if str(row.get("role", "")).startswith(
            "immutable_runtime_source_bundle:")
    ], key=lambda row: (row["path"], row["role"]))
    declared_bundle = preflight.get("runtime_source_bundle")
    module_rows = preflight.get("runtime_module_entrypoints")
    bundle_files = {
        (row.get("path"), row.get("bytes"), row.get("sha256"))
        for row in runtime_bundle
    }
    if (not runtime_bundle or declared_bundle != runtime_bundle
            or stable_json_sha256(runtime_bundle)
            != preflight.get("immutable_runtime_source_bundle_sha256")
            or not isinstance(module_rows, list) or not module_rows
            or stable_json_sha256(module_rows)
            != preflight.get("runtime_module_entrypoint_closure_sha256")
            or len({row.get("module") for row in module_rows}) != len(module_rows)
            or any((row.get("path"), row.get("bytes"), row.get("sha256"))
                   not in bundle_files for row in module_rows)):
        raise Exact191Error("runtime source/module closure mismatch")
    expected_missing = _expected_missing_in_preregistered_order(preregister)
    if ([(task["short_id"], task["node_pair"]) for task, _ in expected]
            != expected_missing):
        raise Exact191Error("task order differs from preregistered exact 72")
    return expected


def validate_new_results(
    tasks: Sequence[tuple[dict, Path]], batch: Mapping[str, Any],
    preregister: Mapping[str, Any], execution_binding: Mapping[str, Any],
) -> dict[tuple[str, tuple[int, int]], dict[str, Any]]:
    reject_forbidden_result_fields(batch)
    validate_execution_binding(execution_binding)
    batch_fields = {
        "schema", "exact_batch_count", "selector_eligible",
        "result_based_selection_allowed", "results", "execution_binding",
        "attempt_receipt_closure_sha256", "payload_sha256",
    }
    if (set(batch) != batch_fields
            or batch.get("schema") != BATCH_SCHEMA
            or batch.get("exact_batch_count") != 72
            or batch.get("selector_eligible") is not False
            or batch.get("result_based_selection_allowed") is not False
            or batch.get("execution_binding") != execution_binding
            or not payload_valid(batch)):
        raise Exact191Error("batch result contract mismatch")
    rows = batch.get("results")
    if not isinstance(rows, list) or len(rows) != 72:
        raise Exact191Error("batch result table is not exact 72")
    expected_ids = [f"{task['short_id']}__{task['node_pair'][0]}_{task['node_pair'][1]}"
                    for task, _ in tasks]
    if [row.get("task_id") for row in rows] != expected_ids:
        raise Exact191Error("batch result order/subset differs from exact 72")
    expected_attempt_closure = [{
        "task_id": row.get("task_id"),
        "attempt_receipt_sha256": row.get("attempt_receipt_sha256"),
    } for row in rows]
    if (stable_json_sha256(expected_attempt_closure)
            != batch.get("attempt_receipt_closure_sha256")):
        raise Exact191Error("batch attempt-receipt closure mismatch")
    output = {}
    for (task, directory), row in zip(tasks, rows):
        task_id = row["task_id"]
        if set(row) != {
                "task_id", "status", "resumed", "attempt_receipt_sha256",
                "result_sha256"} or type(row.get("resumed")) is not bool:
            raise Exact191Error("batch result row field set changed")
        view_path = directory / "authorized_task_view.json"
        attempt_path = directory / "attempt_receipt.json"
        result_path = directory / "result.json"
        if (not view_path.is_file() or not attempt_path.is_file()
                or not result_path.is_file()):
            raise Exact191Error(f"task-view/attempt/result missing: {task_id}")
        view_sha = sha256_file(view_path)
        view = _load_json(view_path, view_sha, "authorized task view")
        expected_view = derive_authorized_task_view(
            task, execution_binding, preregister)
        if (view != expected_view or view.get("schema") != AUTHORIZED_TASK_SCHEMA):
            raise Exact191Error("authorized task view binding mismatch")
        _attempt, attempt_sha = validate_attempt_receipt(
            attempt_path, task, view_sha, execution_binding)
        if row.get("attempt_receipt_sha256") != attempt_sha:
            raise Exact191Error("batch/attempt receipt SHA mismatch")
        if sha256_file(result_path) != row.get("result_sha256"):
            raise Exact191Error("batch/result receipt SHA mismatch")
        result = validate_resumed_result(
            result_path, task, attempt_receipt_path=attempt_path,
            execution_binding=execution_binding,
            authorized_task_view_sha256=view_sha)
        status = row.get("status")
        if (result.get("schema") != RESULT_SCHEMA
                or result.get("task_sha256") != task["task_sha256"]
                or result.get("short_id") != task["short_id"]
                or result.get("pair_id") != task["pair_id"]
                or result.get("node_pair") != task["node_pair"]
                or result.get("object_pair") != task["object_pair"]
                or result.get("selector_eligible") is not False
                or result.get("status") != status
                or not payload_valid(result)):
            raise Exact191Error("failed, foreign, or malformed new result")
        if status == "ok":
            corr = result.get("correspondences")
            if (not isinstance(corr, Mapping)
                    or corr.get("path") != "correspondences.npz"):
                raise Exact191Error(
                    "new result correspondence path is not canonical")
            corr_path = _verify_file_row(directory, corr, "new correspondence")
            corr_arrays = _load_npz(corr_path, corr["sha256"], corr["bytes"])
            if set(corr_arrays) != {"src_corr", "ref_corr", "scores"}:
                raise Exact191Error("new correspondence NPZ has foreign arrays")
            canonical = _validate_corr_arrays(
                {f"{name}_new": value for name, value in corr_arrays.items()},
                "new", corr.get("arrays"))
            correspondence_sha256 = corr["sha256"]
            failure = None
        elif status in ALLOWED_NEW_TYPED_FAILURES:
            failure = result.get("failure")
            if (not isinstance(failure, Mapping)
                    or set(failure) != {"status", "detail"}
                    or failure.get("status") != status
                    or not isinstance(failure.get("detail"), Mapping)
                    or "correspondences" in result):
                raise Exact191Error("typed new failure evidence is malformed")
            canonical = {}
            correspondence_sha256 = None
        else:
            raise Exact191Error("non-scientific new failure cannot be merged")
        key = (task["short_id"], tuple(task["node_pair"]))
        if key in output:
            raise Exact191Error("duplicate new result key")
        output[key] = {
            "task": task, "attempt_sha256": attempt_sha,
            "result_sha256": row["result_sha256"],
            "status": status, "failure": failure,
            "correspondence_sha256": correspondence_sha256,
            "arrays": canonical,
            "authorized_task_view_sha256": view_sha,
        }
    return output


def _validate_candidate_manifest(candidate: Mapping[str, Any]) -> None:
    if (candidate.get("schema") != "v16-b716-candidate-plan-manifest-v1"
            or candidate.get("scope") != "fixed4"
            or candidate.get("frozen") is not True
            or candidate.get("pair_count") != 4
            or candidate.get("candidate_count") != 191
            or candidate.get("hypothesis_count") != 34
            or candidate.get("geot_existing_reused") != 119
            or candidate.get("geot_missing_disabled") != 72
            or candidate.get("new_geot_executed") != 0
            or candidate.get("official92_executed") is not False
            or candidate.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or candidate.get("downstream_colorpcr_authorized") is not False
            or candidate.get("legacy_B_ep20_or_89ed_consumed") is not False
            or not payload_valid(candidate)):
        raise Exact191Error("candidate manifest contract mismatch")
    if any(token in json.dumps(candidate.get("forbidden_inputs", [])).lower()
           for token in ("allowed", "authorized")):
        raise Exact191Error("candidate forbidden-input declaration is malformed")
    artifacts = candidate.get("artifact_closure")
    if (not isinstance(artifacts, list)
            or stable_json_sha256(artifacts)
            != candidate.get("recursive_artifact_closure_sha256")):
        raise Exact191Error("candidate artifact closure mismatch")


def reconstruct_ordered_candidate_keys(
    candidate: Mapping[str, Any], candidate_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Independently rebuild the exact 191-key ledger from frozen pair plans."""
    pairs = candidate.get("pairs")
    if (not isinstance(pairs, list) or len(pairs) != 4
            or [row.get("short_id") for row in pairs]
            != list(EXPECTED_SHORT_IDS)):
        raise Exact191Error("candidate pair order cannot reconstruct exact191 keys")
    expected: list[dict[str, Any]] = []
    existing: list[dict[str, Any]] = []
    backfill: list[dict[str, Any]] = []
    identities = set()
    for pair in pairs:
        plan_path = _verify_file_row(candidate_root, {
            "path": pair.get("plan_path"), "bytes": pair.get("plan_bytes"),
            "sha256": pair.get("plan_sha256"),
        }, "candidate plan for ordered-key reconstruction")
        plan = _load_json(
            plan_path, str(pair.get("plan_sha256", "")),
            "candidate plan for ordered-key reconstruction")
        entries = plan.get("geot_entries")
        if (not payload_valid(plan) or plan.get("short_id") != pair["short_id"]
                or plan.get("pair_id") != pair.get("pair_id")
                or not isinstance(entries, list)
                or len(entries) != int(pair.get("candidate_count", -1))):
            raise Exact191Error("candidate plan cannot reconstruct exact191 keys")
        for index, entry in enumerate(entries):
            node_pair = entry.get("node_pair")
            if (entry.get("candidate_index") != index
                    or not isinstance(node_pair, list) or len(node_pair) != 2
                    or any(type(value) is not int for value in node_pair)
                    or not _entry_payload_valid(entry)):
                raise Exact191Error("candidate ordered-key row is malformed")
            row = {
                "short_id": pair["short_id"], "candidate_index": index,
                "node_pair": list(node_pair),
            }
            identity = (row["short_id"], index, tuple(node_pair))
            if identity in identities:
                raise Exact191Error("candidate ordered-key row is duplicated")
            identities.add(identity)
            expected.append(row)
            if (entry.get("origin") == "official_pair_cache"
                    and entry.get("immutable") is True):
                existing.append(row)
            elif (entry.get("origin") == "missing_execution_disabled"
                  and entry.get("immutable") is False):
                backfill.append(row)
            else:
                raise Exact191Error("candidate ordered-key origin is foreign")
    if (len(expected) != 191 or len(existing) != 119 or len(backfill) != 72
            or stable_json_sha256(expected)
            != EXPECTED_ORDERED_KEY_CLOSURE_SHA256):
        raise Exact191Error("independent ordered-key reconstruction mismatch")
    return expected, existing, backfill


def validate_future_merge_contract(
    contract: Any, expected: Sequence[Mapping[str, Any]],
    existing: Sequence[Mapping[str, Any]],
    backfill: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
        "schema", "existing_official_entry_count",
        "required_backfill_entry_count", "total_candidate_entry_count",
        "exact_candidate_key_coverage_required",
        "result_based_selection_allowed", "downstream_authorized",
        "expected_candidate_keys", "expected_candidate_key_closure_sha256",
        "existing_candidate_key_closure_sha256",
        "backfill_candidate_key_closure_sha256", "status",
    }
    if (not isinstance(contract, Mapping) or set(contract) != fields
            or contract.get("schema") != "v16-b716-geot-merge-contract-v1"
            or contract.get("existing_official_entry_count") != 119
            or contract.get("required_backfill_entry_count") != 72
            or contract.get("total_candidate_entry_count") != 191
            or contract.get("exact_candidate_key_coverage_required") is not True
            or contract.get("result_based_selection_allowed") is not False
            or contract.get("downstream_authorized") is not False
            or contract.get("status")
            != "blocked_until_exact_72_completed_receipts"
            or contract.get("expected_candidate_keys") != list(expected)
            or contract.get("expected_candidate_key_closure_sha256")
            != EXPECTED_ORDERED_KEY_CLOSURE_SHA256
            or contract.get("expected_candidate_key_closure_sha256")
            != stable_json_sha256(expected)
            or contract.get("existing_candidate_key_closure_sha256")
            != stable_json_sha256(existing)
            or contract.get("backfill_candidate_key_closure_sha256")
            != stable_json_sha256(backfill)):
        raise Exact191Error("future merge ordered-key contract mismatch")


def merge_exact191(
    *, candidate_path: Path, candidate_sha256: str,
    preflight_path: Path, preflight_sha256: str,
    preregister_path: Path, preregister_sha256: str,
    authorization_path: Path, authorization_sha256: str,
    batch_path: Path, batch_sha256: str, output_root: Path,
) -> dict[str, Any]:
    """Validate and seal one exact191 merger run without model/GPU access."""
    candidate = _load_json(candidate_path, candidate_sha256, "candidate manifest")
    _validate_candidate_manifest(candidate)
    execution_repo_root = _candidate_repository_root(candidate_path)
    candidate_root = Path(candidate_path).parent
    ordered_keys, existing_keys, backfill_keys = reconstruct_ordered_candidate_keys(
        candidate, candidate_root)
    preregister = _load_json(
        preregister_path, preregister_sha256, "backfill preregistration")
    preflight = _load_json(preflight_path, preflight_sha256, "preflight manifest")
    if (preflight.get("candidate_manifest_sha256") != candidate_sha256
            or Path(preflight.get("candidate_manifest_path", "")).resolve()
            != Path(candidate_path).resolve()):
        raise Exact191Error("preflight/candidate closure mismatch")
    validate_future_merge_contract(
        preflight.get("future_merge_contract"), ordered_keys,
        existing_keys, backfill_keys)
    tasks = validate_preflight(
        preflight, preflight_path.parent, preregister,
        preregister_path, preregister_sha256)
    enable_scope = preregister.get("enable_scope", {})
    future_merge_contract_sha256 = stable_json_sha256(
        preflight["future_merge_contract"])
    ordered72_sha256 = _ordered_task_id_closure_sha256(preflight)
    if (enable_scope.get("future_merge_contract_sha256")
            != future_merge_contract_sha256
            or enable_scope.get("ordered72_sha256") != ordered72_sha256):
        raise Exact191Error(
            "enabled preregister does not bind exact191/ordered72 closures")
    authorization = _validate_bound_execution_authorization(
        authorization_path=authorization_path,
        authorization_sha256=authorization_sha256,
        preregister=preregister,
        execution_repo_root=execution_repo_root,
        candidate_manifest_sha256=candidate_sha256,
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
        output_root=preflight_path.parent,
        future_merge_contract_sha256=future_merge_contract_sha256,
        ordered72_sha256=ordered72_sha256)
    # Authorization is a validated control-plane document and necessarily
    # records ``selected = false``.  Result-field recursion remains enforced
    # on the batch, per-task receipts, and result artifacts below.
    execution_binding = {
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
    validate_execution_binding(execution_binding)
    batch = _load_json(batch_path, batch_sha256, "batch result")
    if Path(batch_path).resolve() != (preflight_path.parent / "batch_result.json").resolve():
        raise Exact191Error("batch result is outside the authorized output root")
    new_results = validate_new_results(
        tasks, batch, preregister, execution_binding)
    new_result_closure = []
    for task, _directory in tasks:
        key = (task["short_id"], tuple(task["node_pair"]))
        value = new_results[key]
        new_result_closure.append({
            "short_id": task["short_id"],
            "candidate_index": task["candidate_index"],
            "node_pair": task["node_pair"],
            "object_pair": task["object_pair"],
            "task_sha256": task["task_sha256"],
            "attempt_sha256": value["attempt_sha256"],
            "result_sha256": value["result_sha256"],
            "status": value["status"],
            "correspondence_sha256": value["correspondence_sha256"],
            "authorized_task_view_sha256": value[
                "authorized_task_view_sha256"],
            "authorization_sha256": authorization_sha256,
        })

    pairs = candidate.get("pairs")
    if ([row.get("short_id") for row in pairs] != list(EXPECTED_SHORT_IDS)
            or [row.get("candidate_count") for row in pairs]
            != list(EXPECTED_CANDIDATES)
            or [row.get("existing_reused") for row in pairs]
            != list(EXPECTED_EXISTING)
            or [row.get("missing_disabled") for row in pairs]
            != list(EXPECTED_MISSING)
            or [row.get("hypothesis_count") for row in pairs]
            != list(EXPECTED_HYPOTHESES)):
        raise Exact191Error("fixed4 count/distribution contract mismatch")

    manifest_pairs, artifact_rows = [], []
    consumed_new = set()
    total_existing_ok = total_existing_failed = total_new = 0
    total_new_ok = total_new_failed = 0
    total_hypotheses = 0
    existing_entry_closure = []
    sealed_ordered_keys = []
    hypotheses_with_typed_failure_members = 0
    hypotheses_with_existing_typed_failure_members = 0
    for pair_row in pairs:
        short_id = pair_row["short_id"]
        plan_path = _verify_file_row(candidate_root, {
            "path": pair_row["plan_path"], "bytes": pair_row["plan_bytes"],
            "sha256": pair_row["plan_sha256"],
        }, "candidate plan")
        plan = _load_json(plan_path, pair_row["plan_sha256"], "candidate plan")
        if (plan.get("pair_id") != pair_row["pair_id"]
                or plan.get("short_id") != short_id
                or plan.get("candidate_count") != pair_row["candidate_count"]
                or plan.get("hypothesis_count") != pair_row["hypothesis_count"]
                or plan.get("domain", {}).get("matched") is not True
                or plan.get("domain", {}).get("checkpoint_sha256")
                != OFFICIAL_RELEASE_SHA256
                or plan.get("domain", {}).get(
                    "legacy_B_ep20_or_89ed_consumed") is not False
                or not payload_valid(plan)):
            raise Exact191Error("candidate pair plan identity/count mismatch")
        _verify_source_closure(
            plan.get("source_closure"),
            str(plan.get("recursive_source_closure_sha256", "")))
        official_cache_sources = [row for row in plan["source_closure"]
                                  if row.get("role") == "official_geot_cache"]
        if len(official_cache_sources) != 1:
            raise Exact191Error("official existing GeoT source is not unique")
        official_cache_path = Path(official_cache_sources[0]["path"])
        official_cache = _load_npz(
            official_cache_path, official_cache_sources[0]["sha256"],
            official_cache_sources[0]["bytes"])
        geot_rows = plan.get("geot_entries")
        ranks = plan.get("candidate_rank_records")
        if (not isinstance(geot_rows, list) or not isinstance(ranks, list)
                or len(geot_rows) != pair_row["candidate_count"]
                or len(ranks) != len(geot_rows)):
            raise Exact191Error("candidate entry table length mismatch")
        immutable_path = _verify_file_row(candidate_root, {
            "path": pair_row["geot_npz_path"],
            "bytes": pair_row["geot_npz_bytes"],
            "sha256": pair_row["geot_npz_sha256"],
        }, "immutable existing GeoT NPZ")
        immutable = _load_npz(
            immutable_path, pair_row["geot_npz_sha256"],
            pair_row["geot_npz_bytes"])
        expected_immutable_names = set()
        merged_arrays: dict[str, np.ndarray] = {}
        sealed_entries = []
        by_node_pair = {}
        pair_existing = pair_failed = pair_new = pair_new_failed = 0
        for index, (entry, rank) in enumerate(zip(geot_rows, ranks)):
            if (entry.get("candidate_index") != index
                    or entry.get("node_pair") != [rank["source_index"],
                                                   rank["reference_index"]]
                    or not _entry_payload_valid(entry)
                    or tuple(entry["node_pair"]) in by_node_pair):
                raise Exact191Error("candidate identity/order/entry SHA mismatch")
            by_node_pair[tuple(entry["node_pair"])] = index
            if entry.get("origin") == "official_pair_cache":
                if entry.get("immutable") is not True:
                    raise Exact191Error("existing cache entry is mutable")
                pair_existing += 1
                source_metadata = entry.get("source_metadata")
                cache_row = entry.get("source_cache_row")
                if (not isinstance(source_metadata, Mapping)
                        or source_metadata.get("cache_row") != cache_row
                        or source_metadata.get("status") != entry.get("status")
                        or source_metadata.get("src_object_id")
                        != entry.get("object_pair", [None, None])[0]
                        or source_metadata.get("ref_object_id")
                        != entry.get("object_pair", [None, None])[1]):
                    raise Exact191Error("existing cache metadata binding mismatch")
                if entry.get("status") == "ok":
                    declared = {}
                    for field in ("src_corr", "ref_corr", "scores"):
                        name = f"{field}_{index}"
                        expected_immutable_names.add(name)
                        declared[field] = {
                            "shape": entry.get(f"{field}_shape"),
                            "dtype": "float32",
                            "sha256": entry.get(f"{field}_sha256"),
                        }
                    arrays = _validate_corr_arrays(immutable, str(index), declared)
                    source_arrays = _validate_corr_arrays(
                        official_cache, str(cache_row), declared)
                    if any(not np.array_equal(arrays[field], source_arrays[field])
                           for field in arrays):
                        raise Exact191Error(
                            "immutable existing entry differs from official cache")
                    total_existing_ok += 1
                else:
                    arrays = {}
                    if any(f"{field}_{cache_row}" in official_cache
                           for field in ("src_corr", "ref_corr", "scores")):
                        raise Exact191Error(
                            "failed existing cache entry unexpectedly has arrays")
                    pair_failed += 1
                    total_existing_failed += 1
                origin = {
                    "kind": "frozen_existing", "entry_sha256": entry["entry_sha256"],
                    "source_cache_row": entry.get("source_cache_row"),
                    "status": entry["status"],
                }
                existing_entry_closure.append({
                    "short_id": short_id, "candidate_index": index,
                    "node_pair": entry["node_pair"],
                    "object_pair": entry["object_pair"],
                    "entry_sha256": entry["entry_sha256"],
                    "source_cache_row": entry.get("source_cache_row"),
                    "status": entry["status"],
                })
            elif entry.get("origin") == "missing_execution_disabled":
                key = (short_id, tuple(entry["node_pair"]))
                new = new_results.get(key)
                if (entry.get("immutable") is not False or new is None
                        or new["task"].get("candidate_index") != index
                        or new["task"].get("object_pair") != entry.get("object_pair")):
                    raise Exact191Error("new result does not fill its frozen candidate slot")
                arrays = new["arrays"]
                pair_new += 1
                total_new += 1
                if new["status"] == "ok":
                    total_new_ok += 1
                else:
                    pair_new_failed += 1
                    total_new_failed += 1
                consumed_new.add(key)
                origin = {
                    "kind": "authorized_backfill", "status": new["status"],
                    "task_sha256": new["task"]["task_sha256"],
                    "authorized_task_view_sha256": new[
                        "authorized_task_view_sha256"],
                    "attempt_sha256": new["attempt_sha256"],
                    "result_sha256": new["result_sha256"],
                    "authorization_sha256": authorization_sha256,
                }
                if new["status"] == "ok":
                    origin["correspondence_sha256"] = new[
                        "correspondence_sha256"]
                else:
                    origin["failure"] = new["failure"]
            else:
                raise Exact191Error("foreign candidate origin")
            array_names = {}
            for field, value in arrays.items():
                name = f"{field}_{index}"
                merged_arrays[name] = value
                array_names[field] = {
                    "name": name, "shape": list(value.shape),
                    "dtype": str(value.dtype), "sha256": array_sha256(value),
                }
            sealed = {
                "candidate_index": index, "node_pair": entry["node_pair"],
                "object_pair": entry["object_pair"], "status": origin["status"],
                "origin": origin, "correspondences": array_names,
            }
            sealed["entry_sha256"] = stable_json_sha256(sealed)
            sealed_entries.append(sealed)
            sealed_ordered_keys.append({
                "short_id": short_id, "candidate_index": index,
                "node_pair": list(entry["node_pair"]),
            })
        if set(immutable) != expected_immutable_names:
            raise Exact191Error("immutable existing NPZ has unused/foreign arrays")
        if (pair_existing != pair_row["existing_reused"]
                or pair_failed != pair_row["existing_failed"]
                or pair_new != pair_row["missing_disabled"]):
            raise Exact191Error("per-pair existing/new/failure accounting mismatch")

        hypotheses = plan.get("hypotheses")
        allow_rows = []
        if not isinstance(hypotheses, list) or len(hypotheses) != pair_row["hypothesis_count"]:
            raise Exact191Error("frozen hypothesis table changed")
        for expected_index, hypothesis in enumerate(hypotheses):
            payload = {key: hypothesis[key] for key in (
                "members", "member_rank_records", "member_count")}
            if (hypothesis.get("hypothesis_index") != expected_index
                    or hypothesis.get("hypothesis_sha256")
                    != stable_json_sha256(payload)):
                raise Exact191Error("frozen hypothesis SHA/order mismatch")
            try:
                indices = [by_node_pair[tuple(pair)] for pair in hypothesis["members"]]
            except (KeyError, TypeError) as exc:
                raise Exact191Error("hypothesis refers to a foreign candidate") from exc
            if len(indices) != hypothesis.get("member_count") or len(indices) != len(set(indices)):
                raise Exact191Error("hypothesis member identity is malformed")
            existing_typed_failure_indices = [
                index for index in indices
                if sealed_entries[index]["origin"]["kind"] == "frozen_existing"
                and sealed_entries[index]["status"] != "ok"
            ]
            new_typed_failure_indices = [
                index for index in indices
                if sealed_entries[index]["origin"]["kind"]
                == "authorized_backfill"
                and sealed_entries[index]["status"] != "ok"
            ]
            typed_failure_indices = (
                existing_typed_failure_indices + new_typed_failure_indices)
            row = {
                "hypothesis_index": expected_index,
                "hypothesis_sha256": hypothesis["hypothesis_sha256"],
                "member_candidate_indices": indices,
                "all_members_ok": all(sealed_entries[i]["status"] == "ok"
                                      for i in indices),
                "contains_typed_failure_members": bool(typed_failure_indices),
                "typed_failure_member_candidate_indices": typed_failure_indices,
                "existing_typed_failure_member_candidate_indices": (
                    existing_typed_failure_indices),
                "new_typed_failure_member_candidate_indices": (
                    new_typed_failure_indices),
            }
            row["allowlist_entry_sha256"] = stable_json_sha256(row)
            allow_rows.append(row)
        pair_hypotheses_with_typed_failures = sum(
            row["contains_typed_failure_members"] for row in allow_rows)
        hypotheses_with_typed_failure_members += (
            pair_hypotheses_with_typed_failures)
        pair_hypotheses_with_existing_typed_failures = sum(
            bool(row["existing_typed_failure_member_candidate_indices"])
            for row in allow_rows)
        hypotheses_with_existing_typed_failure_members += (
            pair_hypotheses_with_existing_typed_failures)
        total_hypotheses += len(allow_rows)

        pair_dir = Path(output_root) / "pairs" / short_id
        npz_path = pair_dir / "exact191_correspondences.npz"
        write_deterministic_npz(npz_path, merged_arrays)
        allowlist = {
            "schema": HYPOTHESIS_SCHEMA, "short_id": short_id,
            "pair_id": plan["pair_id"], "hypothesis_count": len(allow_rows),
            "candidate_selection_allowed": False,
            "hypothesis_selection_allowed": False,
            "all_hypotheses_must_be_replayed": True,
            "typed_failure_members_visible_and_never_filtered": True,
            "hypotheses_with_typed_failure_members": (
                pair_hypotheses_with_typed_failures),
            "consumer_scope": "only_the_34_frozen_hypotheses_across_fixed4",
            "hypotheses": allow_rows,
        }
        allowlist["payload_sha256"] = stable_json_sha256(allowlist)
        allow_path = pair_dir / "frozen_hypothesis_allowlist.json"
        atomic_json(allow_path, allowlist)
        pair_value = {
            "schema": PAIR_SCHEMA, "short_id": short_id,
            "pair_id": plan["pair_id"], "candidate_count": len(sealed_entries),
            "existing_count": pair_existing, "new_count": pair_new,
            "new_ok_count": pair_new - pair_new_failed,
            "new_typed_failure_count": pair_new_failed,
            "existing_failed_count": pair_failed,
            "hypotheses_with_typed_failure_members": (
                pair_hypotheses_with_typed_failures),
            "hypotheses_with_existing_typed_failure_members": (
                pair_hypotheses_with_existing_typed_failures),
            "entries": sealed_entries,
            "correspondences": {
                "path": str(Path("pairs") / short_id / npz_path.name),
                "bytes": int(npz_path.stat().st_size),
                "sha256": sha256_file(npz_path),
            },
            "hypothesis_allowlist": {
                "path": str(Path("pairs") / short_id / allow_path.name),
                "bytes": int(allow_path.stat().st_size),
                "sha256": sha256_file(allow_path),
            },
        }
        pair_value["payload_sha256"] = stable_json_sha256(pair_value)
        pair_path = pair_dir / "exact191_entries.json"
        atomic_json(pair_path, pair_value)
        pair_manifest_row = {
            "short_id": short_id, "pair_id": plan["pair_id"],
            "candidate_count": len(sealed_entries),
            "existing_count": pair_existing, "new_count": pair_new,
            "new_ok_count": pair_new - pair_new_failed,
            "new_typed_failure_count": pair_new_failed,
            "existing_failed_count": pair_failed,
            "hypothesis_count": len(allow_rows),
            "hypotheses_with_typed_failure_members": (
                pair_hypotheses_with_typed_failures),
            "hypotheses_with_existing_typed_failure_members": (
                pair_hypotheses_with_existing_typed_failures),
            "entries_path": str(Path("pairs") / short_id / pair_path.name),
            "entries_bytes": int(pair_path.stat().st_size),
            "entries_sha256": sha256_file(pair_path),
            "correspondences_path": str(Path("pairs") / short_id / npz_path.name),
            "correspondences_bytes": int(npz_path.stat().st_size),
            "correspondences_sha256": sha256_file(npz_path),
            "allowlist_path": str(Path("pairs") / short_id / allow_path.name),
            "allowlist_bytes": int(allow_path.stat().st_size),
            "allowlist_sha256": sha256_file(allow_path),
        }
        manifest_pairs.append(pair_manifest_row)
        artifact_rows.extend([
            {"path": pair_manifest_row["entries_path"],
             "bytes": pair_manifest_row["entries_bytes"],
             "sha256": pair_manifest_row["entries_sha256"], "role": "sealed_entries"},
            {"path": pair_manifest_row["correspondences_path"],
             "bytes": pair_manifest_row["correspondences_bytes"],
             "sha256": pair_manifest_row["correspondences_sha256"],
             "role": "sealed_correspondences"},
            {"path": pair_manifest_row["allowlist_path"],
             "bytes": pair_manifest_row["allowlist_bytes"],
             "sha256": pair_manifest_row["allowlist_sha256"],
             "role": "frozen_hypothesis_allowlist"},
        ])
    if consumed_new != set(new_results) or len(consumed_new) != 72:
        raise Exact191Error("new result set has duplicate/missing/extra keys")
    if (sealed_ordered_keys != ordered_keys
            or stable_json_sha256(sealed_ordered_keys)
            != EXPECTED_ORDERED_KEY_CLOSURE_SHA256
            or sum(row["candidate_count"] for row in manifest_pairs) != 191
            or sum(row["existing_count"] for row in manifest_pairs) != 119
            or total_new != 72 or total_hypotheses != 34
            or total_new_ok + total_new_failed != 72
            or total_existing_ok + total_existing_failed != 119
            or total_existing_failed != EXPECTED_TYPED_FAILURES
            or hypotheses_with_existing_typed_failure_members
            != EXPECTED_HYPOTHESES_WITH_TYPED_FAILURES):
        raise Exact191Error("global exact191 accounting mismatch")
    artifact_rows = sorted(artifact_rows, key=lambda row: row["path"])
    input_closure = sorted([
        {"path": str(Path(path).resolve()),
         "bytes": int(Path(path).stat().st_size),
         "sha256": digest, "role": role}
        for path, digest, role in (
            (candidate_path, candidate_sha256, "frozen_candidate_manifest"),
            (preflight_path, preflight_sha256, "authorized_preflight_manifest"),
            (preregister_path, preregister_sha256, "authorized_preregistration"),
            (authorization_path, authorization_sha256, "execution_authorization"),
            (batch_path, batch_sha256, "exact72_batch_result"),
        )
    ], key=lambda row: (row["path"], row["role"]))
    manifest = {
        "schema": SCHEMA, "sealed": True, "candidate_count": 191,
        "existing_count": 119, "new_authorized_count": 72,
        "new_authorized_ok_count": total_new_ok,
        "new_authorized_typed_failure_count": total_new_failed,
        "existing_ok_count": total_existing_ok,
        "existing_failed_count": total_existing_failed,
        "typed_failure_existing_count": total_existing_failed,
        "typed_failure_total_count": total_existing_failed + total_new_failed,
        "hypothesis_count": 34,
        "hypotheses_with_typed_failure_members": (
            hypotheses_with_typed_failure_members),
        "hypotheses_with_existing_typed_failure_members": (
            hypotheses_with_existing_typed_failure_members),
        "typed_failures_visible_and_never_filtered": True,
        "consumer_scope": "only_the_34_frozen_hypotheses_across_fixed4",
        "candidate_selection_allowed": False,
        "result_based_selection_allowed": False,
        "hypothesis_selection_allowed": False,
        "gt_allowed": False, "official92_allowed": False,
        "new_geot_execution_performed_by_merger": False,
        "b716_domain_only": True,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "fixed_hypothesis_distribution": [12, 8, 2, 12],
        "ordered_candidate_key_closure_sha256": (
            EXPECTED_ORDERED_KEY_CLOSURE_SHA256),
        "existing_candidate_key_closure_sha256": stable_json_sha256(
            existing_keys),
        "backfill_candidate_key_closure_sha256": stable_json_sha256(
            backfill_keys),
        "legacy_B_ep20_or_89ed_consumed": False,
        "candidate_manifest_sha256": candidate_sha256,
        "preflight_manifest_sha256": preflight_sha256,
        "preregister_sha256": preregister_sha256,
        "authorization_sha256": authorization_sha256,
        "batch_result_sha256": batch_sha256,
        "execution_binding": execution_binding,
        "input_closure": input_closure,
        "recursive_input_closure_sha256": stable_json_sha256(input_closure),
        "existing_entry_closure": existing_entry_closure,
        "recursive_existing_entry_closure_sha256": stable_json_sha256(
            existing_entry_closure),
        "new_result_closure": new_result_closure,
        "recursive_new_result_closure_sha256": stable_json_sha256(
            new_result_closure),
        "pairs": manifest_pairs, "artifact_closure": artifact_rows,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifact_rows),
    }
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    atomic_json(Path(output_root) / "exact191_manifest.json", manifest)
    return manifest
