import base64
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from safety.v13_dual_solver_runtime import stable_json_sha256
from safety.v16_b716_fixed4_execution_pilot import (
    ACTIVE_EXECUTION_PREREGISTER_SCHEMA,
    ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA,
    ACTIVE_STAGE_INPUT_DESCRIPTOR_SCHEMA,
    ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA,
    Fixed4ExecutionPilotError,
    PREFLIGHT_SCHEMA,
    RESULT_SCHEMA,
    TASK_SCHEMA,
    active_authorization_time_status,
    build_active_stage_authorization_request,
    build_operational_tasks,
    validate_active_execution_preregister_candidate,
)
from safety.v16_b716_fixed4_subprocess_contract import (
    ACTIVE_PREFLIGHT_SCHEMA, RUNNER_MODE_ACTIVE, SIGNATURE_ALGORITHM,
    Fixed4SubprocessContractError, build_subprocess_registry,
    verify_document_signature,
)
from safety.v16_b716_fixed4_orchestrator_contract import (
    EXPECTED_NODE_COUNT, build_task_dag, synthetic_fixture_bindings,
)

REPO = Path(__file__).resolve().parents[1]


def _seal(value):
    value = dict(value)
    value.pop("payload_sha256", None)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _candidate_preregister():
    return _seal({"schema": ACTIVE_EXECUTION_PREREGISTER_SCHEMA,
        "candidate_only": True, "frozen": True,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "legacy_disabled_preregister_accepted": False,
        "production_adapter_protocol_required": True,
        "production_adapter_ready": False,
        "operational_result_schema": RESULT_SCHEMA,
        "parent_results_derived_by_dispatcher": True,
        "task_stage_input_trusted": False,
        "max_authorization_ttl_seconds": 3600,
        "off_host_signer_required": True,
        "execution_host_private_key_allowed": False,
        "formal_execution_authorized": False,
        "measurement_primitive": {
            "schema": "v16-b716-fixed4-measurement-primitive-v1",
            "stage": "bidirectional_multi_solver_pilot",
            "source": "selected_v15_slot.raw_summary.cross_solver_check",
            "rotation_field": "rotation_deg", "rotation_unit": "degree",
            "translation_field": "translation_m", "translation_unit": "meter",
            "rotation_threshold_source":
                "safety.v15_safe_pose_cluster.ROTATION_MAX_DEG",
            "translation_threshold_source":
                "safety.v15_safe_pose_cluster.TRANSLATION_MAX_M",
            "requires_unique_safe_v15_acceptance": True,
            "direction_checks_allowed": False,
            "v15_compatibility_matrix_allowed": False,
            "gt_allowed": False, "result_selection_allowed": False},
        "active_runner_registry_closure_sha256": "a" * 64,
        "candidate_source_pins": {"src/candidate.py": "b" * 64},
        "gt_allowed": False, "official92_allowed": False,
        "threshold_change_allowed": False, "result_selection_allowed": False,
        "default_checkpoint_replacement_allowed": False,
        "reconstruction_authorized": False, "refusion_allowed": False})


def _ready_v2_preregister():
    value = _candidate_preregister()
    value["schema"] = ACTIVE_EXECUTION_PREREGISTER_V2_SCHEMA
    value["production_adapter_ready"] = True
    return _seal(value)


def _request_case(tmp_path):
    registry_sha = "1" * 64
    executor_sha = "2" * 64
    preflight = _seal({"schema": PREFLIGHT_SCHEMA,
        "execution_authorized": False, "execution_performed": False,
        "reconstruction_authorized": False, "refusion_allowed": False,
        "repo_root": str(tmp_path / "repo"), "git_head": "3" * 40,
        "git_tree": "4" * 40, "output_root": str(tmp_path / "out"),
        "runner_registry_closure_sha256": registry_sha,
        "execution_source_closure_sha256": "5" * 64,
        "runner_registry": [{"runner_mode": RUNNER_MODE_ACTIVE,
            "sealed_executor": {"source": {"sha256": executor_sha}}}],
        "active_subprocess_contract": {"schema": ACTIVE_PREFLIGHT_SCHEMA,
            "runner_mode": RUNNER_MODE_ACTIVE,
            "runner_registry_closure_sha256": registry_sha,
            "sealed_executor_sha256": executor_sha,
            "legacy_disabled_preflight_accepted": False,
            "contract_fixture_allowed": False,
            "operational_result_release_allowed": False}})
    descriptor = _seal({"schema": ACTIVE_STAGE_INPUT_DESCRIPTOR_SCHEMA,
        "task_id": "c0", "stage": "colorpcr_direction",
        "upstream_task_ids": [],
        "input_source": "sealed_preregistered_source_closure",
        "derivation_policy": "dispatcher_only_never_trust_task_runtime_paths",
        "production_input_manifest_schema":
            "v16-b716-fixed4-production-input-manifest-v1",
        "production_adapter_contract_schema":
            "v16-b716-fixed4-production-adapter-contract-v1",
        "operational_result_schema": RESULT_SCHEMA,
        "production_adapter_protocol_ready": False})
    task = _seal({"schema": TASK_SCHEMA, "task_id": "c0",
        "stage": "colorpcr_direction", "upstream_task_ids": [],
        "stage_runner_input_descriptor": descriptor})
    manifest = _seal({"schema": "fixture-manifest", "tasks": []})
    preflight_path = tmp_path / "out" / "execution_preflight.json"
    manifest_path = tmp_path / "out" / "task_manifest.json"
    task_path = tmp_path / "out" / "tasks" / "c0" / "task.json"
    _write(preflight_path, preflight); _write(manifest_path, manifest)
    _write(task_path, task)
    return preflight, preflight_path, manifest, manifest_path, task, task_path


def test_active_candidate_preregister_is_separate_from_legacy_disabled():
    candidate = _candidate_preregister()
    validate_active_execution_preregister_candidate(candidate)
    legacy = dict(candidate)
    legacy["runner_mode"] = "disabled"
    legacy["legacy_disabled_preregister_accepted"] = True
    legacy = _seal(legacy)
    with pytest.raises(Fixed4ExecutionPilotError, match="fail-closed"):
        validate_active_execution_preregister_candidate(legacy)


def test_ready_v2_preregister_only_upgrades_protocol_not_authority():
    value = _ready_v2_preregister()
    validate_active_execution_preregister_candidate(value)
    assert value["production_adapter_ready"] is True
    assert value["formal_execution_authorized"] is False
    assert value["reconstruction_authorized"] is False
    assert value["refusion_allowed"] is False
    downgraded = _seal({**value, "production_adapter_ready": False})
    with pytest.raises(Fixed4ExecutionPilotError, match="fail-closed"):
        validate_active_execution_preregister_candidate(downgraded)
    upgraded_legacy = _seal({**_candidate_preregister(),
                             "production_adapter_ready": True})
    with pytest.raises(Fixed4ExecutionPilotError, match="fail-closed"):
        validate_active_execution_preregister_candidate(upgraded_legacy)


def test_checked_in_active_candidate_is_frozen_historical_and_fail_closed():
    """A future active v2 must pin its newly reviewed candidate commit."""
    value = json.loads((REPO / "manifests" /
        "v16_b716_fixed4_active_execution_candidate_preregister.json").read_text())
    validate_active_execution_preregister_candidate(value)
    assert value["schema"] == ACTIVE_EXECUTION_PREREGISTER_SCHEMA
    assert value["candidate_only"] is True
    assert value["frozen"] is True
    assert value["formal_execution_authorized"] is False
    assert value["production_adapter_ready"] is False
    assert value["legacy_disabled_preregister_accepted"] is False
    assert value["reconstruction_authorized"] is False
    assert value["refusion_allowed"] is False
    pins = value["candidate_source_pins"]
    assert pins
    assert all(isinstance(relative, str) and relative for relative in pins)
    assert all(isinstance(digest, str) and len(digest) == 64
               and set(digest) <= set("0123456789abcdef")
               for digest in pins.values())
    changed = dict(value)
    changed["formal_execution_authorized"] = True
    changed = _seal(changed)
    with pytest.raises(Fixed4ExecutionPilotError, match="fail-closed"):
        validate_active_execution_preregister_candidate(changed)


def test_unsigned_stage_request_binds_descriptor_and_never_contains_signature(tmp_path):
    case = _request_case(tmp_path)
    issued = datetime(2026, 8, 31, tzinfo=timezone.utc)
    request = build_active_stage_authorization_request(
        preflight=case[0], preflight_path=case[1], manifest=case[2],
        manifest_path=case[3], task=case[4], task_path=case[5],
        signing_key_id="offline-key", issued_at=issued, ttl_seconds=3600)
    assert request["unsigned"] is True
    assert request["signer_location"] == "off_host"
    assert "signature_b64" not in request
    body = request["authorization_body"]
    assert body["stage_input_descriptor_payload_sha256"] \
        == case[4]["stage_runner_input_descriptor"]["payload_sha256"]
    assert body["task_stage_input_trusted"] is False
    assert body["production_adapter_protocol_ready"] is False
    assert body["signature_algorithm"] == SIGNATURE_ALGORITHM


def test_stage_request_ttl_over_3600_and_expired_authorization_fail_closed(tmp_path):
    case = _request_case(tmp_path)
    with pytest.raises(Fixed4ExecutionPilotError, match="exceeds 3600"):
        build_active_stage_authorization_request(
            preflight=case[0], preflight_path=case[1], manifest=case[2],
            manifest_path=case[3], task=case[4], task_path=case[5],
            signing_key_id="offline-key", ttl_seconds=3601)
    issued = datetime(2026, 8, 31, tzinfo=timezone.utc)
    expired = {"issued_at": issued.isoformat(),
               "expires_at": (issued + timedelta(seconds=60)).isoformat()}
    assert active_authorization_time_status(
        expired, now=issued + timedelta(seconds=61)) == "EXPIRED"
    malformed = dict(expired)
    malformed["expires_at"] = (issued + timedelta(seconds=3601)).isoformat()
    with pytest.raises(Fixed4ExecutionPilotError, match="TTL invalid"):
        active_authorization_time_status(malformed, now=issued)


def test_unsigned_request_can_only_be_satisfied_by_matching_off_host_ed25519(tmp_path):
    case = _request_case(tmp_path)
    request = build_active_stage_authorization_request(
        preflight=case[0], preflight_path=case[1], manifest=case[2],
        manifest_path=case[3], task=case[4], task_path=case[5],
        signing_key_id="offline-key", ttl_seconds=60)
    private = tmp_path / "offline-private.pem"
    public = tmp_path / "execution-host-public.pem"
    message = tmp_path / "message.bin"; signature = tmp_path / "signature.bin"
    subprocess.run(["/usr/bin/openssl", "genpkey", "-algorithm", "Ed25519",
                    "-out", str(private)], check=True, capture_output=True)
    subprocess.run(["/usr/bin/openssl", "pkey", "-in", str(private), "-pubout",
                    "-out", str(public)], check=True, capture_output=True)
    signed = dict(request["authorization_body"])
    message.write_text(json.dumps(signed, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False))
    subprocess.run(["/usr/bin/openssl", "pkeyutl", "-sign", "-inkey",
                    str(private), "-rawin", "-in", str(message), "-out",
                    str(signature)], check=True, capture_output=True)
    signed["signature_b64"] = base64.b64encode(signature.read_bytes()).decode()
    signed = _seal(signed)
    anchor = {"key_id": "offline-key", "_public_key": str(public)}
    verify_document_signature(signed, anchor, purpose="active test authorization")
    tampered = dict(signed); tampered["task_id"] = "forged-task"
    tampered = _seal(tampered)
    with pytest.raises(Fixed4SubprocessContractError, match="signature rejected"):
        verify_document_signature(
            tampered, anchor, purpose="active test authorization")


def test_all_107_active_tasks_bind_registry_and_dispatcher_only_descriptors():
    bindings = synthetic_fixture_bindings()
    dag = build_task_dag(bindings, "d" * 64, synthetic_fixture=True)
    rows, digest = build_subprocess_registry(REPO, runner_mode=RUNNER_MODE_ACTIVE)
    identity = {"runner_registry_closure_sha256": digest}
    tasks, mapping = build_operational_tasks(
        bindings, identity, dag, rows, runner_mode=RUNNER_MODE_ACTIVE)
    assert len(tasks) == 107
    assert len(mapping) == EXPECTED_NODE_COUNT
    for task in tasks:
        assert task["execution_binding"]["runner_mode"] == RUNNER_MODE_ACTIVE
        assert "stage_runner_input" not in task
        descriptor = task["stage_runner_input_descriptor"]
        assert descriptor["upstream_task_ids"] == task["upstream_task_ids"]
        assert descriptor["derivation_policy"] \
            == "dispatcher_only_never_trust_task_runtime_paths"
        assert descriptor["production_adapter_protocol_ready"] is False


def test_all_107_ready_v2_tasks_use_v2_descriptors_without_authority():
    bindings = synthetic_fixture_bindings()
    dag = build_task_dag(bindings, "d" * 64, synthetic_fixture=True)
    rows, digest = build_subprocess_registry(REPO, runner_mode=RUNNER_MODE_ACTIVE)
    tasks, mapping = build_operational_tasks(
        bindings, {"runner_registry_closure_sha256": digest}, dag, rows,
        runner_mode=RUNNER_MODE_ACTIVE,
        production_adapter_protocol_ready=True)
    assert len(tasks) == 107
    assert len(mapping) == EXPECTED_NODE_COUNT
    for task in tasks:
        assert task["execution_authorized"] is False
        descriptor = task["stage_runner_input_descriptor"]
        assert descriptor["schema"] == ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA
        assert descriptor["production_adapter_protocol_ready"] is True
