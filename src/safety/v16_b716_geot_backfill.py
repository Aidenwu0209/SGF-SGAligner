"""Exact-batch, fail-closed GeoTransformer backfill contracts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from safety.v16_b716_candidate_plan import (
    B716PlanError, OFFICIAL_CHECKPOINT_EPOCH, OFFICIAL_CODE_HEAD,
    OFFICIAL_MODEL_CONFIG_SHA256, OFFICIAL_RELEASE_SHA256, array_sha256,
    atomic_json, file_evidence, load_input_tensors, load_joint_model,
    input_tensor_sha256, safe_pair_metadata, sha256_file, stable_json_sha256,
    validate_pair_metadata, write_deterministic_npz,
)


SCHEMA = "v16-b716-geot-backfill-preflight-v1"
TASK_SCHEMA = "v16-b716-geot-backfill-task-v1"
RESULT_SCHEMA = "v16-b716-geot-backfill-result-v1"
AUTH_SCHEMA = "v16-b716-geot-backfill-authorization-v1"
CLEAN_SCHEMA = "v16-b716-clean-service-receipt-v1"
AUTHORIZED_TASK_SCHEMA = "v16-b716-geot-authorized-task-view-v1"
ATTEMPT_SCHEMA = "v16-b716-geot-attempt-receipt-v1"
GEOT_SHA256 = "5c5ffe352baddd83a12a8077451650235bb68a401367d7061344cd9c4aa3595c"
FORBIDDEN_RESULT_FIELDS = frozenset({
    "gt", "ground_truth", "labels", "combos", "node_metrics", "outcome",
    "selection", "selected",
})
ALLOWED_FAILURE_STATUSES = frozenset({
    "insufficient_post_voxel_points", "geotransformer_runtime_error",
})


class BackfillError(B716PlanError):
    """A backfill preflight, resource gate or resume artifact is invalid."""


def payload_valid(value: Mapping[str, Any]) -> bool:
    payload = {key: item for key, item in value.items() if key != "payload_sha256"}
    return value.get("payload_sha256") == stable_json_sha256(payload)


def _sha256_text(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def authorization_derivation_contract() -> dict[str, Any]:
    return {
        "planned_task_state": "planned_disabled",
        "planned_execution_authorized": False,
        "authorized_view_schema": AUTHORIZED_TASK_SCHEMA,
        "authorized_state": "authorized_pending",
        "requires_disabled_false": True,
        "requires_real_execution_allowed_true": True,
        "planned_task_is_immutable": True,
    }


def cuda_hard_gate_contract() -> dict[str, Any]:
    return {
        "authorization_receipt_required": True,
        "authorization_sha256_required": True,
        "clean_service_receipt_required": True,
        "clean_service_environment_sentinel": "V16_B716_CLEAN_SERVICE=1",
        "isolated_gpu_environment_sentinel": "V16_B716_ISOLATED_GPU=1",
        "single_cuda_visible_device_required": True,
        "max_memory_used_mib": 256,
        "max_utilization_percent": 5,
        "compute_process_count": 0,
        "recheck_before_every_key": True,
        "runtime_max_memory_used_mib": 8192,
        "runtime_max_utilization_percent": 100,
        "runtime_only_current_process_allowed": True,
        "clean_service_receipt_max_age_seconds": 300,
        "required_services_checked": [
            "nvidia_compute_process_table", "sgaligner_python_workers",
            "geotransformer_workers"],
    }


def future_merge_contract() -> dict[str, Any]:
    return {
        "schema": "v16-b716-geot-merge-contract-v1",
        "existing_official_entry_count": 119,
        "required_backfill_entry_count": 72,
        "total_candidate_entry_count": 191,
        "exact_candidate_key_coverage_required": True,
        "result_based_selection_allowed": False,
        "downstream_authorized": False,
    }


def reject_forbidden_result_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_RESULT_FIELDS:
                raise BackfillError(f"forbidden GeoT result field: {key}")
            reject_forbidden_result_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_forbidden_result_fields(item)


def expected_missing_rows(preregister: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = preregister.get("expected_missing_node_pairs_by_short_id")
    if not isinstance(mapping, Mapping):
        raise BackfillError("preregister exact missing-key table is absent")
    rows = []
    for short_id, pairs in mapping.items():
        if not isinstance(short_id, str) or not isinstance(pairs, list):
            raise BackfillError("preregister missing-key table is malformed")
        for raw in pairs:
            if (not isinstance(raw, list) or len(raw) != 2
                    or any(type(value) is not int for value in raw)):
                raise BackfillError("preregister node pair is malformed")
            rows.append({"short_id": short_id, "node_pair": list(raw)})
    if len(rows) != int(preregister.get("expected_missing_key_count", -1)):
        raise BackfillError("preregister missing-key count mismatch")
    if stable_json_sha256(rows) != preregister.get(
            "expected_missing_key_closure_sha256"):
        raise BackfillError("preregister missing-key closure mismatch")
    if len({(row["short_id"], *row["node_pair"]) for row in rows}) != len(rows):
        raise BackfillError("preregister missing-key table has duplicates")
    return rows


def validate_preregister(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    execution = value.get("execution_contract", {})
    gate = value.get("cuda_hard_gate", {})
    authorities = value.get("authoritative_v3_manifests", {})
    formal = value.get("formal_v13_fixed4_preregister", {})
    audit = value.get("selection89_readonly_audit", {})
    forbidden = value.get("forbidden_field_declarations", {})
    execution_mode = (value.get("disabled"),
                      execution.get("real_execution_allowed"))
    if (value.get("schema") != "v16-b716-geot-backfill-preregister-v1"
            or value.get("frozen") is not True
            or execution_mode not in {(True, False), (False, True)}
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or value.get("official_geotransformer_checkpoint_sha256")
            != GEOT_SHA256
            or execution.get("default_mode") != "dry-run"
            or execution.get("key_selection_allowed") is not False
            or execution.get("exact_batch_only") is not True
            or execution.get("result_based_selection_allowed") is not False
            or execution.get("authorization_derivation_contract")
            != authorization_derivation_contract()
            or gate != cuda_hard_gate_contract()
            or value.get("official92_allowed") is not False
            or value.get("gt_allowed") is not False
            or value.get("pair_combos_allowed") is not False
            or value.get("outcome_or_result_selection_allowed") is not False
            or value.get("future_merge_contract") != future_merge_contract()
            or set(authorities) != {"artifact_manifest", "cache_manifest"}
            or len(formal.get("full_pair_ids", [])) != 4
            or audit.get("row_count") != 89
            or audit.get("unique_tag_count") != 89
            or audit.get("unique_pair_id_count") != 89
            or audit.get("checkpoint_epoch") != OFFICIAL_CHECKPOINT_EPOCH
            or audit.get("checkpoint_sha256") != OFFICIAL_RELEASE_SHA256
            or audit.get("sampling_mode") != "official_mt19937"
            or audit.get("model_config_sha256")
            != OFFICIAL_MODEL_CONFIG_SHA256
            or audit.get("code_head") != OFFICIAL_CODE_HEAD
            or forbidden != {
                "pair_cache_top_level_fields_lexically_skipped": ["combos"],
                "nested_fields_never_decoded": ["node_metrics"],
                "canonical_builder_with_labels": False,
                "gt_or_label_inputs_allowed": False,
                "posthoc_inputs_allowed": False,
                "official92_inputs_allowed": False,
                "result_or_outcome_selection_allowed": False,
            }):
        raise BackfillError("backfill preregistration contract mismatch")
    for label, row in [*authorities.items(),
                       ("formal_v13_fixed4_preregister", formal)]:
        path = Path(row.get("path", ""))
        if (not path.is_absolute() or not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64
                or type(row.get("bytes")) is not int or row["bytes"] <= 0):
            raise BackfillError(f"malformed frozen authority declaration: {label}")
    return expected_missing_rows(value)


def _verify_declared_file(row: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = Path(row.get("path", ""))
    if (not path.is_absolute() or not path.is_file()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")):
        raise BackfillError(f"frozen external authority changed: {role}")
    return file_evidence(path, role)


def _read_frozen_json(source: Mapping[str, Any], role: str) -> Any:
    path = Path(source["path"])
    before = sha256_file(path)
    if before != source["sha256"]:
        raise BackfillError(f"frozen JSON changed before read: {role}")
    text = path.read_text()
    if sha256_file(path) != before:
        raise BackfillError(f"frozen JSON changed during read: {role}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackfillError(f"frozen JSON is malformed: {role}") from exc


def _validate_artifact_entry(files: Mapping[str, Any], repo_root: Path,
                             path: Path, expected_sha256: str,
                             expected_bytes: int) -> str:
    try:
        key = str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise BackfillError("V3 artifact lies outside authoritative repository") from exc
    entry = files.get(key)
    if (not isinstance(entry, Mapping)
            or entry.get("sha256") != expected_sha256
            or int(entry.get("size", -1)) != int(expected_bytes)):
        raise BackfillError(f"V3 artifact manifest mismatch: {key}")
    return key


def validate_external_authorities(
    preregister: Mapping[str, Any], candidate: Mapping[str, Any],
    candidate_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind fixed4 to V3 manifests and the formal V13 full-pair prereg.

    The selection89 manifest is audited read-only.  Only the safe lexical view
    of each fixed4 pair_cache is decoded; ``combos`` and node metrics remain
    skipped by :func:`safe_pair_metadata`.
    """
    declared = preregister["authoritative_v3_manifests"]
    artifact_source = _verify_declared_file(
        declared["artifact_manifest"], "authoritative_v3_artifact_manifest")
    cache_source = _verify_declared_file(
        declared["cache_manifest"], "authoritative_v3_cache_manifest")
    formal_source = _verify_declared_file(
        preregister["formal_v13_fixed4_preregister"],
        "formal_v13_fixed4_preregister")
    artifact_path = Path(artifact_source["path"])
    cache_path = Path(cache_source["path"])
    if artifact_path.parent != cache_path.parent:
        raise BackfillError("V3 authority manifests do not share one run root")
    v3_run_root = artifact_path.parent
    if v3_run_root.parent.name != "outputs":
        raise BackfillError("V3 authority run is not under an outputs directory")
    repo_root = v3_run_root.parents[1]
    artifact = _read_frozen_json(artifact_source, "V3 artifact_manifest")
    cache = _read_frozen_json(cache_source, "V3 cache_manifest")
    files = artifact.get("files")
    if not isinstance(files, Mapping):
        raise BackfillError("V3 artifact_manifest files table is absent")
    cache_key_path = _validate_artifact_entry(
        files, repo_root, cache_path, cache_source["sha256"],
        cache_source["bytes"])

    selection = cache.get("splits", {}).get("selection89")
    audit = preregister["selection89_readonly_audit"]
    if not isinstance(selection, list):
        raise BackfillError("V3 cache manifest selection89 is absent")
    tags: list[str] = []
    pair_ids: list[str] = []
    expected_cache_fields = {
        "pair_id", "input_tensor_sha256", "checkpoint_sha256",
        "sampling_mode", "model_config_sha256", "code_head",
    }
    for row in selection:
        cache_key = row.get("cache_key") if isinstance(row, Mapping) else None
        if (not isinstance(row, Mapping)
                or set(row) != {"pair_id", "tag", "cache_key", "status",
                                "pair_cache_sha256"}
                or not isinstance(cache_key, Mapping)
                or set(cache_key) != expected_cache_fields
                or row.get("status") != "ok"
                or cache_key.get("pair_id") != row.get("pair_id")
                or cache_key.get("checkpoint_sha256")
                != OFFICIAL_RELEASE_SHA256
                or cache_key.get("sampling_mode") != "official_mt19937"
                or cache_key.get("model_config_sha256")
                != OFFICIAL_MODEL_CONFIG_SHA256
                or cache_key.get("code_head") != OFFICIAL_CODE_HEAD
                or not isinstance(cache_key.get("input_tensor_sha256"), str)
                or len(cache_key["input_tensor_sha256"]) != 64
                or not isinstance(row.get("pair_cache_sha256"), str)
                or len(row["pair_cache_sha256"]) != 64):
            raise BackfillError("V3 selection89 cache row contract mismatch")
        tags.append(str(row["tag"]))
        pair_ids.append(str(row["pair_id"]))
    selection_summary = {
        "row_count": len(selection),
        "unique_tag_count": len(set(tags)),
        "unique_pair_id_count": len(set(pair_ids)),
        "stable_rows_sha256": stable_json_sha256(selection),
        "checkpoint_epoch": OFFICIAL_CHECKPOINT_EPOCH,
        "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "sampling_mode": "official_mt19937",
        "model_config_sha256": OFFICIAL_MODEL_CONFIG_SHA256,
        "code_head": OFFICIAL_CODE_HEAD,
    }
    if selection_summary != audit:
        raise BackfillError("V3 selection89 read-only audit digest/count mismatch")
    if len(tags) != len(set(tags)) or len(pair_ids) != len(set(pair_ids)):
        raise BackfillError("V3 selection89 identities are not unique")

    formal = _read_frozen_json(formal_source, "formal V13 fixed4 preregister")
    full_pair_ids = preregister["formal_v13_fixed4_preregister"]["full_pair_ids"]
    if (formal.get("schema") != "v13-colorpcr-pointdsc-preregister-v1"
            or formal.get("normal_pair_ids") != full_pair_ids[:3]
            or formal.get("known_bad_pair_id") != full_pair_ids[3]
            or formal.get("append_known_bad") is not True
            or formal.get("gpu_authorized") is not False):
        raise BackfillError("formal V13 fixed4 full-pair preregistration mismatch")

    pairs = candidate.get("pairs")
    expected_short_ids = list(
        preregister["expected_missing_node_pairs_by_short_id"])
    if (not isinstance(pairs, list) or len(pairs) != 4
            or [row.get("pair_id") for row in pairs] != full_pair_ids
            or [row.get("short_id") for row in pairs] != expected_short_ids):
        raise BackfillError("candidate fixed4 identities differ from formal V13 prereg")
    selection_by_tag = {row["tag"]: row for row in selection}
    fixed_artifacts: list[dict[str, Any]] = []
    role_files = {
        "official_pair_metadata": "pair_cache.json",
        "official_input_tensors": "input_tensors.npz",
        "official_embeddings": "embeddings.npz",
        "official_geot_cache": "geot_corrs.npz",
    }
    external_sources = [artifact_source, cache_source, formal_source]
    for pair in pairs:
        tag = pair["short_id"]
        cache_row = selection_by_tag.get(tag)
        if cache_row is None or cache_row["pair_id"] != pair["pair_id"]:
            raise BackfillError("fixed4 pair is absent from V3 selection89")
        plan_path = candidate_root / pair["plan_path"]
        if (not plan_path.is_file()
                or sha256_file(plan_path) != pair["plan_sha256"]
                or plan_path.stat().st_size != int(pair["plan_bytes"])):
            raise BackfillError("candidate plan changed before authority validation")
        plan = json.loads(plan_path.read_text())
        source_rows = verify_source_closure(plan)
        by_role = {row["role"]: row for row in source_rows}
        if any(role not in by_role for role in role_files):
            raise BackfillError("candidate plan lacks an authoritative V3 cache role")
        expected_dir = v3_run_root / "final_inference_cache" / "selection89" / tag
        for role, filename in role_files.items():
            source = by_role[role]
            path = Path(source["path"])
            if path.resolve() != (expected_dir / filename).resolve():
                raise BackfillError("candidate source is outside canonical V3 tag path")
            artifact_key = _validate_artifact_entry(
                files, repo_root, path, source["sha256"], source["bytes"])
            fixed_artifacts.append({
                "tag": tag, "pair_id": pair["pair_id"], "filename": filename,
                "artifact_key": artifact_key, "bytes": int(source["bytes"]),
                "sha256": source["sha256"],
            })
            external_sources.append(file_evidence(
                path, f"authoritative_v3_fixed4:{tag}:{filename}"))
        meta = safe_pair_metadata(Path(by_role["official_pair_metadata"]["path"]))
        validate_pair_metadata(meta)
        input_semantic_sha256 = input_tensor_sha256(load_input_tensors(
            Path(by_role["official_input_tensors"]["path"])))
        if (meta["pair_id"] != pair["pair_id"]
                or meta["cache_key"] != cache_row["cache_key"]
                or meta["checkpoint_epoch"] != audit["checkpoint_epoch"]
                or meta["code_head"] != audit["code_head"]
                or cache_row["pair_cache_sha256"]
                != by_role["official_pair_metadata"]["sha256"]
                or cache_row["cache_key"]["input_tensor_sha256"]
                != input_semantic_sha256):
            raise BackfillError("fixed4 pair_cache/cache_manifest inner binding mismatch")
    authority_summary = {
        "artifact_manifest_sha256": artifact_source["sha256"],
        "cache_manifest_sha256": cache_source["sha256"],
        "cache_manifest_artifact_key": cache_key_path,
        "formal_v13_preregister_sha256": formal_source["sha256"],
        "formal_fixed4_full_pair_ids": full_pair_ids,
        "selection89_readonly_audit": selection_summary,
        "fixed4_artifact_file_count": len(fixed_artifacts),
        "fixed4_artifact_closure_sha256": stable_json_sha256(fixed_artifacts),
    }
    return external_sources, authority_summary


def build_frozen_merge_contract(
    candidate_root: Path, candidate: Mapping[str, Any],
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    base = preregister["future_merge_contract"]
    if base != future_merge_contract():
        raise BackfillError("future merge preregistration changed")
    expected_keys, existing_keys, missing_keys = [], [], []
    for pair in candidate.get("pairs", []):
        plan_path = candidate_root / pair["plan_path"]
        if (sha256_file(plan_path) != pair["plan_sha256"]
                or plan_path.stat().st_size != int(pair["plan_bytes"])):
            raise BackfillError("candidate plan changed during merger freeze")
        plan = json.loads(plan_path.read_text())
        entries = plan.get("geot_entries")
        if (not isinstance(entries, list)
                or len(entries) != int(pair["candidate_count"])
                or [entry.get("candidate_index") for entry in entries]
                != list(range(len(entries)))):
            raise BackfillError("candidate plan cannot freeze exact merger keys")
        for entry in entries:
            key = {
                "short_id": pair["short_id"],
                "candidate_index": int(entry["candidate_index"]),
                "node_pair": [int(value) for value in entry["node_pair"]],
            }
            expected_keys.append(key)
            if (entry.get("origin") == "official_pair_cache"
                    and entry.get("immutable") is True):
                existing_keys.append(key)
            elif (entry.get("origin") == "missing_execution_disabled"
                  and entry.get("immutable") is False):
                missing_keys.append(key)
            else:
                raise BackfillError("candidate entry has no merger-safe origin")
    if (len(expected_keys) != base["total_candidate_entry_count"]
            or len(existing_keys) != base["existing_official_entry_count"]
            or len(missing_keys) != base["required_backfill_entry_count"]
            or len({(row["short_id"], row["candidate_index"])
                    for row in expected_keys}) != len(expected_keys)):
        raise BackfillError("frozen 119+72 merger accounting mismatch")
    return {
        **base,
        "expected_candidate_keys": expected_keys,
        "expected_candidate_key_closure_sha256": stable_json_sha256(expected_keys),
        "existing_candidate_key_closure_sha256": stable_json_sha256(existing_keys),
        "backfill_candidate_key_closure_sha256": stable_json_sha256(missing_keys),
        "status": "blocked_until_exact_72_completed_receipts",
    }


def merge_completed_geot_ledger(
    contract: Mapping[str, Any], existing_rows: Sequence[Mapping[str, Any]],
    backfill_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically account for all 119+72 entries without selection."""
    expected = contract.get("expected_candidate_keys")
    if (not isinstance(expected, list)
            or stable_json_sha256(expected)
            != contract.get("expected_candidate_key_closure_sha256")
            or len(existing_rows) != contract.get("existing_official_entry_count")
            or len(backfill_rows) != contract.get("required_backfill_entry_count")
            or len(expected) != contract.get("total_candidate_entry_count")
            or contract.get("result_based_selection_allowed") is not False
            or contract.get("downstream_authorized") is not False):
        raise BackfillError("merger contract/count mismatch")
    def key(row: Mapping[str, Any]) -> tuple[str, int, tuple[int, int]]:
        raw_pair = row.get("node_pair")
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise BackfillError("merger row node key is malformed")
        return (str(row.get("short_id")), int(row.get("candidate_index", -1)),
                (int(raw_pair[0]), int(raw_pair[1])))
    expected_order = [key(row) for row in expected]
    existing_by_key, backfill_by_key = {}, {}
    for row in existing_rows:
        reject_forbidden_result_fields(row)
        row_key = key(row)
        if (row.get("origin") != "official_pair_cache"
                or row.get("immutable") is not True
                or not _sha256_text(row.get("entry_sha256"))
                or row_key in existing_by_key):
            raise BackfillError("existing merger row contract mismatch")
        existing_by_key[row_key] = dict(row)
    for row in backfill_rows:
        reject_forbidden_result_fields(row)
        row_key = key(row)
        if (row.get("origin") != "official_geotransformer_backfill"
                or row.get("selector_eligible") is not False
                or not _sha256_text(row.get("task_sha256"))
                or not _sha256_text(row.get("attempt_receipt_sha256"))
                or not _sha256_text(row.get("result_sha256"))
                or row.get("status") not in ({"ok"} | ALLOWED_FAILURE_STATUSES)
                or row_key in backfill_by_key):
            raise BackfillError("backfill merger row contract mismatch")
        backfill_by_key[row_key] = dict(row)
    if (set(existing_by_key) & set(backfill_by_key)
            or set(existing_by_key) | set(backfill_by_key) != set(expected_order)):
        raise BackfillError("merger rows do not cover exact 191 candidate keys")
    return [existing_by_key.get(row_key, backfill_by_key.get(row_key))
            for row_key in expected_order]


def extract_observed_missing(candidate_root: Path,
                             manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair in manifest.get("pairs", []):
        path = candidate_root / pair["plan_path"]
        if (sha256_file(path) != pair["plan_sha256"]
                or int(path.stat().st_size) != int(pair["plan_bytes"])):
            raise BackfillError("candidate structural plan SHA/bytes mismatch")
        plan = json.loads(path.read_text())
        if not payload_valid(plan):
            raise BackfillError("candidate structural plan payload hash mismatch")
        if (plan.get("domain", {}).get("matched") is not True
                or plan.get("domain", {}).get("checkpoint_sha256")
                != OFFICIAL_RELEASE_SHA256
                or plan.get("domain", {}).get(
                    "legacy_B_ep20_or_89ed_consumed") is not False):
            raise BackfillError("candidate plan is not matched b716 domain")
        missing = [entry for entry in plan.get("geot_entries", [])
                   if entry.get("origin") == "missing_execution_disabled"]
        if len(missing) != int(pair["missing_disabled"]):
            raise BackfillError("candidate missing-key count differs from manifest")
        for entry in missing:
            if (entry.get("immutable") is not False
                    or entry.get("status") != "disabled_missing_geotransformer"):
                raise BackfillError("candidate missing key is not fail-closed")
            rows.append({
                "short_id": pair["short_id"], "pair_id": pair["pair_id"],
                "candidate_index": int(entry["candidate_index"]),
                "node_pair": [int(x) for x in entry["node_pair"]],
                "object_pair": [int(x) for x in entry["object_pair"]],
                "candidate_plan_path": str(path.resolve()),
                "candidate_plan_sha256": pair["plan_sha256"],
            })
    return rows


def compare_exact_missing(expected: Sequence[Mapping[str, Any]],
                          observed: Sequence[Mapping[str, Any]]) -> None:
    public = [{"short_id": row["short_id"],
               "node_pair": list(row["node_pair"])} for row in observed]
    if list(expected) != public:
        raise BackfillError("observed missing keys/order differ from exact 72 preregistration")


def verify_source_closure(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = plan.get("source_closure")
    if not isinstance(rows, list) or not rows:
        raise BackfillError("candidate source closure is absent")
    for row in rows:
        path = Path(row["path"])
        if (not path.is_absolute() or not path.is_file()
                or path.stat().st_size != int(row["bytes"])
                or sha256_file(path) != row["sha256"]):
            raise BackfillError(f"candidate source closure mismatch: {path}")
    if stable_json_sha256(rows) != plan.get("recursive_source_closure_sha256"):
        raise BackfillError("candidate recursive source closure mismatch")
    return rows


def source_by_role(rows: Sequence[Mapping[str, Any]], role: str) -> Path:
    matches = [Path(row["path"]) for row in rows if row.get("role") == role]
    if len(matches) != 1:
        raise BackfillError(f"source closure role is not unique: {role}")
    return matches[0]


def revalidate_pair(
    task_rows: Sequence[Mapping[str, Any]], candidate_root: Path,
    *, build_canonical_pair: Callable[..., tuple[Mapping[str, Any], Sequence[Any]]],
    canonical_boundary: Callable[[Mapping[str, Any], Mapping[str, np.ndarray]],
                                 Mapping[str, Any]],
    raw_binding_builder: Callable[[Mapping[str, Any], str], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not task_rows:
        raise BackfillError("pair has no missing tasks")
    plan_path = Path(task_rows[0]["candidate_plan_path"])
    plan = json.loads(plan_path.read_text())
    sources = verify_source_closure(plan)
    pair_meta_path = source_by_role(sources, "official_pair_metadata")
    meta = safe_pair_metadata(pair_meta_path)
    validate_pair_metadata(meta)
    if meta["pair_id"] != plan["pair_id"]:
        raise BackfillError("pair metadata identity mismatch")
    checkpoint = source_by_role(sources, "official_release_checkpoint")
    geot_checkpoint = source_by_role(sources, "official_geotransformer_checkpoint")
    if sha256_file(checkpoint) != OFFICIAL_RELEASE_SHA256:
        raise BackfillError("official release checkpoint changed")
    if sha256_file(geot_checkpoint) != GEOT_SHA256:
        raise BackfillError("official GeoTransformer checkpoint changed")
    tensors = load_input_tensors(source_by_role(sources, "official_input_tensors"))
    if input_tensor_sha256(tensors) != meta["cache_key"]["input_tensor_sha256"]:
        raise BackfillError("pair cache semantic input_tensor_sha256 changed")
    data, labels = build_canonical_pair(plan["pair_id"], with_labels=False)
    if labels:
        raise BackfillError("canonical input builder returned prohibited labels")
    boundary = dict(canonical_boundary(data, tensors))
    if boundary != plan["src_count_evidence"]:
        raise BackfillError("canonical object/src boundary changed")
    joint = load_joint_model(
        source_by_role(sources, "official_embeddings"), boundary["total_objects"])
    if (list(joint.shape) != plan["joint_model"]["shape"]
            or str(joint.dtype) != plan["joint_model"]["dtype"]
            or array_sha256(joint) != plan["joint_model"]["sha256"]):
        raise BackfillError("official joint_model binding changed")
    bindings = list(raw_binding_builder(data, plan["pair_id"]))
    if bindings != plan["canonical_surface_bindings"]:
        raise BackfillError("canonical raw/object/surface binding changed")
    by_node = {int(row["node_index"]): row for row in bindings}
    output = []
    for raw in task_rows:
        source_index, reference_index = map(int, raw["node_pair"])
        if (not 0 <= source_index < boundary["src_count"]
                or not boundary["src_count"] <= reference_index
                < boundary["total_objects"]):
            raise BackfillError("missing key crosses an invalid object boundary")
        if ([by_node[source_index]["object_id"],
             by_node[reference_index]["object_id"]] != raw["object_pair"]):
            raise BackfillError("missing key object identity changed")
        task = {
            "schema": TASK_SCHEMA, "state": "planned_disabled",
            "execution_authorized": False,
            "execution_transition_contract": authorization_derivation_contract(),
            "short_id": raw["short_id"], "pair_id": raw["pair_id"],
            "candidate_index": raw["candidate_index"],
            "node_pair": raw["node_pair"], "object_pair": raw["object_pair"],
            "source_surface": by_node[source_index],
            "reference_surface": by_node[reference_index],
            "candidate_plan_sha256": raw["candidate_plan_sha256"],
            "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "official_geotransformer_checkpoint_sha256": GEOT_SHA256,
            "canonical_boundary_sha256": stable_json_sha256(boundary),
            "canonical_surface_binding_sha256": stable_json_sha256(bindings),
            "forbidden_inputs": [
                "GT/selection/evaluation labels", "pair combos/node metrics",
                "posthoc", "official92", "fallbacks", "result-based selection",
            ],
        }
        task["task_sha256"] = stable_json_sha256(task)
        output.append(task)
    short_id = str(task_rows[0]["short_id"])
    current_sources = [file_evidence(
        plan_path, f"candidate_structural_plan:{short_id}")]
    current_sources.extend(file_evidence(
        Path(row["path"]), f"candidate_source:{short_id}:{row['role']}")
        for row in sources)
    return output, current_sources


def revalidate_runtime_registration_points(
    task: Mapping[str, Any], data: Mapping[str, Any],
    fresh_bindings: Sequence[Mapping[str, Any]], *,
    array_fingerprint: Callable[[np.ndarray], str],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild and bind the exact two surfaces immediately before GeoT."""
    by_node = {int(row["node_index"]): dict(row) for row in fresh_bindings}
    surfaces = data.get("registration_pts")
    if not isinstance(surfaces, Mapping):
        raise BackfillError("fresh canonical registration_pts are absent")
    output = []
    for side, node_position in (("source", 0), ("reference", 1)):
        node = int(task["node_pair"][node_position])
        frozen = task[f"{side}_surface"]
        if by_node.get(node) != frozen or node not in surfaces:
            raise BackfillError("fresh raw/object/surface binding changed before GeoT")
        points = np.ascontiguousarray(np.asarray(surfaces[node]))
        if (points.ndim != 2 or points.shape[1:] != (3,)
                or len(points) != int(frozen["canonical_registration_points"])
                or not np.issubdtype(points.dtype, np.floating)
                or not np.isfinite(points).all()
                or array_fingerprint(points)
                != frozen["canonical_registration_surface_sha256"]):
            raise BackfillError("fresh registration_pts array changed before GeoT")
        output.append(points.copy())
    return output[0], output[1]


def write_tasks(tasks: Sequence[Mapping[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        pair = task["node_pair"]
        task_id = f"{task['short_id']}__{pair[0]}_{pair[1]}"
        path = output_root / "tasks" / task_id / "task.json"
        atomic_json(path, task)
        rows.append({
            "task_id": task_id, "short_id": task["short_id"],
            "node_pair": task["node_pair"], "task_sha256": task["task_sha256"],
            "path": str(Path("tasks") / task_id / "task.json"),
            "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
            "state": "planned_disabled",
        })
    return rows


def validate_execution_binding(binding: Mapping[str, Any]) -> None:
    sha_fields = {
        "authorization_sha256", "preregister_sha256",
        "preflight_manifest_sha256", "preflight_payload_sha256",
        "recursive_source_closure_sha256",
        "recursive_artifact_closure_sha256", "task_closure_sha256",
        "immutable_runtime_source_bundle_sha256",
        "runtime_module_entrypoint_closure_sha256",
    }
    if (set(binding) != sha_fields | {"cuda_device_uuid"}
            or any(not _sha256_text(binding.get(field)) for field in sha_fields)
            or not isinstance(binding.get("cuda_device_uuid"), str)
            or not binding["cuda_device_uuid"]):
        raise BackfillError("execution binding is malformed")


def derive_authorized_task_view(
    task: Mapping[str, Any], binding: Mapping[str, Any],
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive an enable-time view without mutating the frozen planned task."""
    validate_execution_binding(binding)
    validate_preregister(preregister)
    transition = authorization_derivation_contract()
    task_payload = {key: value for key, value in task.items()
                    if key != "task_sha256"}
    if (preregister.get("disabled") is not False
            or preregister["execution_contract"].get(
                "real_execution_allowed") is not True
            or task.get("schema") != TASK_SCHEMA
            or task.get("state") != transition["planned_task_state"]
            or task.get("execution_authorized") is not False
            or task.get("execution_transition_contract") != transition
            or stable_json_sha256(task_payload) != task.get("task_sha256")):
        raise BackfillError("planned task cannot derive an authorized execution view")
    view = {
        "schema": AUTHORIZED_TASK_SCHEMA,
        "state": transition["authorized_state"],
        "execution_authorized": True,
        "planned_task_immutable": True,
        "planned_task_sha256": task["task_sha256"],
        "short_id": task["short_id"], "pair_id": task["pair_id"],
        "node_pair": task["node_pair"], "object_pair": task["object_pair"],
        "execution_binding": dict(binding),
    }
    view["payload_sha256"] = stable_json_sha256(view)
    return view


def build_attempt_receipt(
    task: Mapping[str, Any], authorized_task_view_sha256: str,
    binding: Mapping[str, Any], cuda_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    validate_execution_binding(binding)
    if (not _sha256_text(authorized_task_view_sha256)
            or cuda_snapshot.get("uuid") != binding["cuda_device_uuid"]):
        raise BackfillError("attempt receipt task-view/GPU binding mismatch")
    receipt = {
        "schema": ATTEMPT_SCHEMA,
        "task_sha256": task["task_sha256"],
        "authorized_task_view_sha256": authorized_task_view_sha256,
        **dict(binding),
        "cuda_snapshot": dict(cuda_snapshot),
        "cuda_snapshot_sha256": stable_json_sha256(cuda_snapshot),
    }
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    return receipt


def validate_attempt_receipt(
    path: Path, task: Mapping[str, Any], authorized_task_view_sha256: str,
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    validate_execution_binding(binding)
    path = Path(path)
    if not path.is_file():
        raise BackfillError("completed result lacks its attempt receipt")
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BackfillError("attempt receipt JSON is malformed") from exc
    if sha256_file(path) != before:
        raise BackfillError("attempt receipt changed during validation")
    expected_fields = {
        "schema", "task_sha256", "authorized_task_view_sha256",
        "cuda_snapshot", "cuda_snapshot_sha256", "payload_sha256",
    } | set(binding)
    snapshot = value.get("cuda_snapshot")
    if (set(value) != expected_fields
            or value.get("schema") != ATTEMPT_SCHEMA
            or value.get("task_sha256") != task.get("task_sha256")
            or value.get("authorized_task_view_sha256")
            != authorized_task_view_sha256
            or any(value.get(field) != expected
                   for field, expected in binding.items())
            or not isinstance(snapshot, Mapping)
            or snapshot.get("uuid") != binding["cuda_device_uuid"]
            or value.get("cuda_snapshot_sha256")
            != stable_json_sha256(snapshot)
            or not payload_valid(value)):
        raise BackfillError("attempt receipt binding/contract mismatch")
    return value, before


def validate_authorization(
    path: Path, expected_sha256: str, preregister: Mapping[str, Any],
    *, candidate_manifest_sha256: str, missing_closure_sha256: str,
    preregister_sha256: str, preflight_manifest_sha256: str,
    preflight_payload_sha256: str, recursive_source_closure_sha256: str,
    recursive_artifact_closure_sha256: str, task_closure_sha256: str,
    immutable_runtime_source_bundle_sha256: str, output_root: Path,
    runtime_module_entrypoint_closure_sha256: str,
) -> dict[str, Any]:
    execution = preregister["execution_contract"]
    if (preregister.get("disabled") is not False
            or execution.get("real_execution_allowed") is not True):
        raise BackfillError("reviewed preregistration still disables real execution")
    if (not Path(path).is_file() or not _sha256_text(expected_sha256)
            or sha256_file(path) != expected_sha256):
        raise BackfillError("authorization receipt SHA mismatch")
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BackfillError("authorization receipt JSON is malformed") from exc
    if sha256_file(path) != before:
        raise BackfillError("authorization receipt changed during validation")
    expires = value.get("expires_utc", "")
    try:
        if not isinstance(expires, str):
            raise TypeError
        expiry = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise BackfillError("authorization expiry is invalid") from exc
    if (value.get("schema") != AUTH_SCHEMA or value.get("authorized") is not True
            or value.get("candidate_manifest_sha256") != candidate_manifest_sha256
            or value.get("missing_key_closure_sha256") != missing_closure_sha256
            or value.get("preregister_sha256") != preregister_sha256
            or value.get("preflight_manifest_sha256")
            != preflight_manifest_sha256
            or value.get("preflight_payload_sha256") != preflight_payload_sha256
            or value.get("recursive_source_closure_sha256")
            != recursive_source_closure_sha256
            or value.get("recursive_artifact_closure_sha256")
            != recursive_artifact_closure_sha256
            or value.get("task_closure_sha256") != task_closure_sha256
            or value.get("immutable_runtime_source_bundle_sha256")
            != immutable_runtime_source_bundle_sha256
            or value.get("runtime_module_entrypoint_closure_sha256")
            != runtime_module_entrypoint_closure_sha256
            or value.get("exact_batch_count") != 72
            or value.get("key_selection_allowed") is not False
            or value.get("result_selection_allowed") is not False
            or value.get("gt_allowed") is not False
            or value.get("official92_allowed") is not False
            or Path(value.get("output_root", "")).resolve() != output_root.resolve()
            or expiry.utcoffset() is None
            or expiry <= datetime.now(timezone.utc)):
        raise BackfillError("authorization receipt scope/expiry mismatch")
    return value


def query_cuda_snapshot(run: Callable[..., subprocess.CompletedProcess] = subprocess.run
                        ) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or "," in visible:
        raise BackfillError("exactly one numeric CUDA_VISIBLE_DEVICES is required")
    gpu_cmd = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    proc_cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits"]
    gpu = run(gpu_cmd, check=True, capture_output=True, text=True)
    processes = run(proc_cmd, check=True, capture_output=True, text=True)
    rows = []
    for line in gpu.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 4:
            rows.append({"index": int(parts[0]), "uuid": parts[1],
                         "memory_used_mib": int(parts[2]),
                         "utilization_percent": int(parts[3])})
    matches = [row for row in rows if row["index"] == int(visible)]
    if len(matches) != 1:
        raise BackfillError("CUDA_VISIBLE_DEVICES does not resolve uniquely")
    compute = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            raise BackfillError("nvidia-smi compute-process row is malformed")
        try:
            process = {
                "gpu_uuid": parts[0], "pid": int(parts[1]),
                "process_name": parts[2], "used_gpu_memory_mib": int(parts[3]),
            }
        except ValueError as exc:
            raise BackfillError("nvidia-smi compute-process value is invalid") from exc
        if process["gpu_uuid"] == matches[0]["uuid"]:
            compute.append(process)
    return {**matches[0], "compute_processes": compute}


def validate_clean_service_receipt(
    authorization: Mapping[str, Any], preregister: Mapping[str, Any],
    *, expected_cuda_device_uuid: str,
) -> dict[str, Any]:
    gate = preregister["cuda_hard_gate"]
    clean_path = Path(authorization.get("clean_service_receipt_path", ""))
    expected = authorization.get("clean_service_receipt_sha256", "")
    if (not clean_path.is_absolute() or not clean_path.is_file()
            or not _sha256_text(expected) or sha256_file(clean_path) != expected):
        raise BackfillError("clean-service receipt SHA/path mismatch")
    before = sha256_file(clean_path)
    try:
        clean = json.loads(clean_path.read_text())
    except json.JSONDecodeError as exc:
        raise BackfillError("clean-service receipt JSON is malformed") from exc
    if sha256_file(clean_path) != before:
        raise BackfillError("clean-service receipt changed during validation")
    try:
        checked = datetime.fromisoformat(
            str(clean.get("checked_utc", "")).replace("Z", "+00:00"))
        expires = datetime.fromisoformat(
            str(clean.get("expires_utc", "")).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackfillError("clean-service receipt time scope is invalid") from exc
    now = datetime.now(timezone.utc)
    if (clean.get("schema") != CLEAN_SCHEMA or clean.get("clean") is not True
            or clean.get("cuda_device_uuid") != expected_cuda_device_uuid
            or clean.get("services_checked") != gate["required_services_checked"]
            or clean.get("compute_process_count") != 0
            or checked.utcoffset() is None or expires.utcoffset() is None
            or checked > now or expires <= now or expires <= checked
            or (expires - checked).total_seconds()
            > int(gate["clean_service_receipt_max_age_seconds"])):
        raise BackfillError("clean-service receipt contract mismatch")
    return clean


def enforce_cuda_clean_gate(snapshot: Mapping[str, Any], authorization: Mapping[str, Any],
                            preregister: Mapping[str, Any]) -> None:
    gate = preregister["cuda_hard_gate"]
    if (os.environ.get("V16_B716_CLEAN_SERVICE") != "1"
            or os.environ.get("V16_B716_ISOLATED_GPU") != "1"):
        raise BackfillError("clean-service/isolated-GPU sentinels are absent")
    if (snapshot.get("uuid") != authorization.get("cuda_device_uuid")
            or int(snapshot.get("memory_used_mib", 10**9))
            > int(gate["max_memory_used_mib"])
            or int(snapshot.get("utilization_percent", 100))
            > int(gate["max_utilization_percent"])
            or snapshot.get("compute_processes") != []):
        raise BackfillError("CUDA is not isolated, idle and process-clean")
    validate_clean_service_receipt(
        authorization, preregister,
        expected_cuda_device_uuid=str(snapshot["uuid"]))


def enforce_cuda_runtime_gate(
    snapshot: Mapping[str, Any], authorization: Mapping[str, Any],
    preregister: Mapping[str, Any], *, current_pid: int,
) -> None:
    """Recheck isolation before every key while allowing only this runner.

    The zero-process clean gate is evaluated before model import.  Once the
    model is resident, subsequent checks must permit the authorized runner's
    own CUDA context but no foreign process.
    """
    gate = preregister["cuda_hard_gate"]
    if (os.environ.get("V16_B716_CLEAN_SERVICE") != "1"
            or os.environ.get("V16_B716_ISOLATED_GPU") != "1"):
        raise BackfillError("clean-service/isolated-GPU sentinels are absent")
    processes = snapshot.get("compute_processes")
    if (not isinstance(processes, list)
            or snapshot.get("uuid") != authorization.get("cuda_device_uuid")
            or int(snapshot.get("memory_used_mib", 10**9))
            > int(gate["runtime_max_memory_used_mib"])
            or int(snapshot.get("utilization_percent", 100))
            > int(gate["runtime_max_utilization_percent"])
            or len(processes) > 1
            or any(row.get("gpu_uuid") != snapshot.get("uuid")
                   or int(row.get("pid", -1)) != int(current_pid)
                   for row in processes)):
        raise BackfillError("CUDA runtime isolation changed before key execution")


def normalise_result(
    status: str, output: Mapping[str, Any] | None,
    task: Mapping[str, Any], output_dir: Path, *,
    execution_binding: Mapping[str, Any], attempt_receipt_sha256: str,
    authorized_task_view_sha256: str,
) -> dict[str, Any]:
    reject_forbidden_result_fields(output or {})
    validate_execution_binding(execution_binding)
    if (not _sha256_text(attempt_receipt_sha256)
            or not _sha256_text(authorized_task_view_sha256)):
        raise BackfillError("result receipt/task-view SHA binding is malformed")
    if status != "ok" and status not in ALLOWED_FAILURE_STATUSES:
        raise BackfillError("GeoTransformer failure status is not whitelisted")
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "task_sha256": task["task_sha256"],
        "short_id": task["short_id"], "pair_id": task["pair_id"],
        "node_pair": task["node_pair"], "object_pair": task["object_pair"],
        "status": str(status), "selector_eligible": False,
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "authorized_task_view_sha256": authorized_task_view_sha256,
        **dict(execution_binding),
    }
    if status == "ok":
        output = output or {}
        required = ("src_corr_points", "ref_corr_points", "corr_scores")
        if any(key not in output for key in required):
            raise BackfillError("GeoTransformer ok result lacks arrays")
        arrays = {
            "src_corr": np.ascontiguousarray(output[required[0]], dtype=np.float32),
            "ref_corr": np.ascontiguousarray(output[required[1]], dtype=np.float32),
            "scores": np.ascontiguousarray(output[required[2]], dtype=np.float32),
        }
        if (arrays["src_corr"].ndim != 2 or arrays["src_corr"].shape[1:] != (3,)
                or arrays["ref_corr"].shape != arrays["src_corr"].shape
                or arrays["scores"].shape != (len(arrays["src_corr"]),)
                or len(arrays["src_corr"]) == 0
                or any(not np.isfinite(value).all() for value in arrays.values())):
            raise BackfillError("GeoTransformer result arrays are malformed")
        npz = output_dir / "correspondences.npz"
        write_deterministic_npz(npz, arrays)
        result["correspondences"] = {
            "path": "correspondences.npz", "bytes": int(npz.stat().st_size),
            "sha256": sha256_file(npz),
            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                              "sha256": array_sha256(value)}
                       for key, value in arrays.items()},
        }
    else:
        result["failure"] = {"status": str(status), "detail": output or {}}
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def validate_resumed_result(
    path: Path, task: Mapping[str, Any], *, attempt_receipt_path: Path,
    execution_binding: Mapping[str, Any], authorized_task_view_sha256: str,
) -> dict[str, Any]:
    _attempt, attempt_sha256 = validate_attempt_receipt(
        attempt_receipt_path, task, authorized_task_view_sha256,
        execution_binding)
    before = sha256_file(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise BackfillError("resumed result JSON is malformed") from exc
    if sha256_file(path) != before:
        raise BackfillError("resumed result changed during validation")
    reject_forbidden_result_fields(value)
    if (value.get("schema") != RESULT_SCHEMA
            or value.get("task_sha256") != task["task_sha256"]
            or value.get("selector_eligible") is not False
            or value.get("attempt_receipt_sha256") != attempt_sha256
            or value.get("authorized_task_view_sha256")
            != authorized_task_view_sha256
            or any(value.get(field) != expected
                   for field, expected in execution_binding.items())
            or not payload_valid(value)):
        raise BackfillError("resumed per-key result contract mismatch")
    status = value.get("status")
    base_fields = {
        "schema", "task_sha256", "short_id", "pair_id", "node_pair",
        "object_pair", "status", "selector_eligible",
        "attempt_receipt_sha256", "authorized_task_view_sha256",
        "payload_sha256", *execution_binding.keys(),
    }
    if status == "ok":
        if set(value) != base_fields | {"correspondences"}:
            raise BackfillError("resumed ok-result field set changed")
        corr = value.get("correspondences", {})
        npz = path.parent / corr.get("path", "")
        if (set(corr) != {"path", "bytes", "sha256", "arrays"}
                or corr.get("path") != "correspondences.npz"
                or not npz.is_file()
                or npz.stat().st_size != int(corr.get("bytes", -1))
                or sha256_file(npz) != corr.get("sha256")):
            raise BackfillError("resumed correspondence artifact mismatch")
        try:
            with np.load(npz, allow_pickle=False) as archive:
                if set(archive.files) != {"src_corr", "ref_corr", "scores"}:
                    raise BackfillError("resumed NPZ field set changed")
                arrays = {name: np.ascontiguousarray(archive[name])
                          for name in archive.files}
        except (OSError, ValueError) as exc:
            raise BackfillError("resumed NPZ cannot be decoded safely") from exc
        metadata = corr.get("arrays")
        if (not isinstance(metadata, Mapping)
                or set(metadata) != set(arrays)
                or arrays["src_corr"].dtype != np.float32
                or arrays["src_corr"].ndim != 2
                or arrays["src_corr"].shape[1:] != (3,)
                or arrays["ref_corr"].dtype != np.float32
                or arrays["ref_corr"].shape != arrays["src_corr"].shape
                or arrays["scores"].dtype != np.float32
                or arrays["scores"].shape != (len(arrays["src_corr"]),)
                or len(arrays["src_corr"]) == 0
                or any(not np.isfinite(array).all() for array in arrays.values())):
            raise BackfillError("resumed NPZ array contract mismatch")
        for name, array in arrays.items():
            row = metadata.get(name)
            if (not isinstance(row, Mapping)
                    or set(row) != {"shape", "dtype", "sha256"}
                    or row.get("shape") != list(array.shape)
                    or row.get("dtype") != str(array.dtype)
                    or row.get("sha256") != array_sha256(array)):
                raise BackfillError("resumed NPZ per-array evidence mismatch")
    elif status in ALLOWED_FAILURE_STATUSES:
        failure = value.get("failure")
        if (set(value) != base_fields | {"failure"}
                or not isinstance(failure, Mapping)
                or set(failure) != {"status", "detail"}
                or failure.get("status") != status):
            raise BackfillError("resumed failure result contract mismatch")
    else:
        raise BackfillError("resumed failure status is not whitelisted")
    return value
