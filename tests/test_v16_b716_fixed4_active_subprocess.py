import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from safety.v13_dual_solver_runtime import stable_json_sha256
from safety.v16_b716_fixed4_subprocess_contract import (
    ACTIVE_POLICY_FALSE_FIELDS,
    ACTIVE_AUTHORIZATION_SCHEMA,
    ACTIVE_CONSUMPTION_SCHEMA,
    ACTIVE_PREFLIGHT_SCHEMA,
    ACTIVE_STAGE_INPUT_SCHEMA,
    CONTRACT_FIXTURE_STAGE,
    RUNNER_MODE_ACTIVE,
    RUNNER_MODE_DISABLED,
    TOPOLOGICAL_PARENT_SCHEMA,
    Fixed4SubprocessContractError,
    build_subprocess_registry,
    build_topological_parent_receipt,
    execute_active_stage,
    no_symlink_file_row,
    task_execution_binding,
    validate_active_control_bindings,
)
from safety.v16_b716_fixed4_execution_pilot import POLICY_FALSE_FIELDS


REPO = Path(__file__).resolve().parents[1]


def test_active_sealed_executor_pins_current_subprocess_contract():
    executor = (
        REPO / "scripts/v16_b716_fixed4_active_sealed_executor.py"
    ).read_text()
    match = re.search(r'^CONTRACT_SHA256 = "([0-9a-f]{64})"$', executor,
                      flags=re.MULTILINE)
    assert match is not None
    observed = hashlib.sha256(
        (REPO / "src/safety/v16_b716_fixed4_subprocess_contract.py")
        .read_bytes()
    ).hexdigest()
    assert match.group(1) == observed


def test_active_execution_policy_names_match_operational_documents():
    assert set(ACTIVE_POLICY_FALSE_FIELDS) == set(POLICY_FALSE_FIELDS)


def _seal(value):
    value.pop("payload_sha256", None)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _case(tmp_path, *, stage=CONTRACT_FIXTURE_STAGE, fixture=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows, digest = build_subprocess_registry(
        REPO, runner_mode=RUNNER_MODE_ACTIVE,
        include_contract_fixture=fixture)
    task_id = "fixture-1" if fixture else "pair-1"
    output = tmp_path / "output"
    task_root = output / "tasks" / task_id
    task_path = task_root / "task.json"
    fixture_input = tmp_path / "declared-fixture.bin"
    if fixture:
        fixture_input.write_bytes(b"declared fixture payload\n")
        declared = [no_symlink_file_row(fixture_input, "test fixture input")]
        status = "contract_fixture"
    else:
        declared = []
        status = "production_adapter_unavailable"
    stage_input = _seal({
        "schema": ACTIVE_STAGE_INPUT_SCHEMA,
        "task_id": task_id,
        "stage": stage,
        "declared_read_files": declared,
        "declared_read_closure_sha256": stable_json_sha256(declared),
        "implementation_status": status,
    })
    binding = task_execution_binding(rows, digest, stage, task_id)
    task = _seal({
        "task_id": task_id,
        "stage": stage,
        "execution_binding": binding,
        "stage_runner_input": stage_input,
    })
    row = next(item for item in rows if item["stage"] == stage)
    preflight = {"active_subprocess_contract": {
        "schema": ACTIVE_PREFLIGHT_SCHEMA,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "runner_registry_closure_sha256": digest,
        "sealed_executor_sha256": row["sealed_executor"]["source"]["sha256"],
        "legacy_disabled_preflight_accepted": False,
        "contract_fixture_allowed": fixture,
        "operational_result_release_allowed": False,
    }}
    authorization = {"active_subprocess_authorization": {
        "schema": ACTIVE_AUTHORIZATION_SCHEMA,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "runner_registry_closure_sha256": digest,
        "task_id": task_id,
        "task_payload_sha256": task["payload_sha256"],
        "stage": stage,
        "execution_binding_sha256": stable_json_sha256(binding),
        "stage_input_payload_sha256": stage_input["payload_sha256"],
        "execution_authorized": True,
        "contract_fixture_allowed": fixture,
        "operational_result_release_allowed": False,
    }}
    preflight_path = output / "execution_preflight.json"
    authorization_path = (output / "authorizations" / task_id /
        (stable_json_sha256(authorization) + ".json"))
    manifest_path = output / "task_manifest.json"
    _write_json(task_path, task)
    _write_json(preflight_path, preflight)
    _write_json(authorization_path, authorization)
    _write_json(manifest_path, {"task_ids": [task_id]})
    return {
        "repo": REPO, "task": task, "task_path": task_path,
        "preflight_path": preflight_path,
        "authorization_path": authorization_path,
        "task_manifest_path": manifest_path,
        "task_root": task_root, "registry_rows": rows,
        "registry_sha256": digest, "include_contract_fixture": fixture,
        "fixture_input": fixture_input,
        "preflight": preflight, "authorization": authorization,
    }


def _execute(case):
    return execute_active_stage(**{
        key: value for key, value in case.items()
        if key not in {"fixture_input", "preflight", "authorization"}
    })


def test_active_contract_fixture_executes_with_exact_trace_and_no_result_release(tmp_path):
    case = _case(tmp_path)
    receipt = _execute(case)
    assert receipt["schema"] == ACTIVE_CONSUMPTION_SCHEMA
    assert receipt["failure_type"] is None
    assert receipt["returncode"] == 0
    assert receipt["parent_observed_accesses"]["valid"] is True
    assert receipt["parent_observed_accesses"]["observed_write_paths"] == [
        str(case["task_root"] / "active" / "runner_output.bin")]
    assert (case["task_root"] / "active" / "runner_output.bin").read_bytes() \
        == case["fixture_input"].read_bytes()
    assert receipt["operational_result_emitted"] is False
    assert receipt["operational_result_release_allowed"] is False
    assert not (case["task_root"] / "result.json").exists()


def test_active_root_level_or_wrong_task_authorization_is_rejected(tmp_path):
    case = _case(tmp_path / "root")
    root_auth = case["task_root"].parents[1] / "authorization.json"
    root_auth.write_bytes(case["authorization_path"].read_bytes())
    case["authorization_path"] = root_auth
    with pytest.raises(Fixed4SubprocessContractError,
                       match="control input path/layout mismatch"):
        _execute(case)

    case = _case(tmp_path / "wrong-task")
    wrong = (case["task_root"].parents[1] / "authorizations" / "other-task" /
             case["authorization_path"].name)
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(case["authorization_path"].read_bytes())
    case["authorization_path"] = wrong
    with pytest.raises(Fixed4SubprocessContractError,
                       match="control input path/layout mismatch"):
        _execute(case)


def test_legacy_preflight_and_missing_stage_input_cannot_activate(tmp_path):
    case = _case(tmp_path)
    legacy = {}
    with pytest.raises(Fixed4SubprocessContractError,
                       match="legacy/unbound preflight"):
        validate_active_control_bindings(
            task=case["task"], preflight=legacy,
            authorization=case["authorization"],
            registry_rows=case["registry_rows"],
            registry_sha256=case["registry_sha256"],
            include_contract_fixture=True)
    missing = copy.deepcopy(case["task"])
    missing.pop("stage_runner_input")
    _seal(missing)
    with pytest.raises(Fixed4SubprocessContractError,
                       match="stage_runner_input is absent"):
        validate_active_control_bindings(
            task=missing, preflight=case["preflight"],
            authorization=case["authorization"],
            registry_rows=case["registry_rows"],
            registry_sha256=case["registry_sha256"],
            include_contract_fixture=True)


def test_declared_input_hash_and_symlink_are_rejected(tmp_path):
    case = _case(tmp_path)
    case["fixture_input"].write_bytes(b"tampered\n")
    with pytest.raises(Fixed4SubprocessContractError,
                       match="bytes/SHA drift"):
        _execute(case)

    case = _case(tmp_path / "symlink")
    original = case["fixture_input"]
    real = original.with_name("real.bin")
    original.rename(real)
    original.symlink_to(real)
    with pytest.raises(Fixed4SubprocessContractError,
                       match="symlinked"):
        _execute(case)


def test_active_repeated_execution_and_extra_inventory_fail_closed(tmp_path):
    case = _case(tmp_path / "repeat")
    _execute(case)
    with pytest.raises(Fixed4SubprocessContractError,
                       match="partial active state"):
        _execute(case)

    case = _case(tmp_path / "inventory")
    (case["task_root"] / "unsealed.bin").write_bytes(b"not declared\n")
    with pytest.raises(Fixed4SubprocessContractError,
                       match="extra/unsealed"):
        _execute(case)


def test_active_registry_runner_sha_drift_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("v16_b716_fixed4_active_stage_runner.sh",
                 "v16_b716_fixed4_active_sealed_executor.py"):
        shutil.copy2(REPO / "scripts" / name, scripts / name)
    with (scripts / "v16_b716_fixed4_active_stage_runner.sh").open("ab") as stream:
        stream.write(b"# unauthorized drift\n")
    with pytest.raises(Fixed4SubprocessContractError,
                       match="runner SHA drift"):
        build_subprocess_registry(repo, runner_mode=RUNNER_MODE_ACTIVE)


def test_disabled_default_and_fixture_opt_in_are_separate():
    disabled, _ = build_subprocess_registry(REPO)
    assert all(row["runner_mode"] == RUNNER_MODE_DISABLED for row in disabled)
    assert all(row["stage"] != CONTRACT_FIXTURE_STAGE for row in disabled)
    disabled_only_flags = {
        "--repo", "--output-root", "--execution-manifest",
        "--production-manifest-commit", "--production-python",
        "--production-wrapper", "--runner-source-sha256",
        "--fixture-input", "--fixture-input-sha256",
    }
    assert all(not disabled_only_flags.intersection(row["argv_template"])
               for row in disabled)
    active, _ = build_subprocess_registry(REPO, runner_mode=RUNNER_MODE_ACTIVE)
    assert all(row["runner_mode"] == RUNNER_MODE_ACTIVE for row in active)
    assert all(row["stage"] != CONTRACT_FIXTURE_STAGE for row in active)
    with pytest.raises(Fixed4SubprocessContractError,
                       match="fixture is only valid"):
        build_subprocess_registry(
            REPO, runner_mode=RUNNER_MODE_DISABLED,
            include_contract_fixture=True)


def test_topological_parent_receipt_validates_upstream_order_only():
    rows = [
        {"task_id": "c1", "stage": "colorpcr_direction",
         "upstream_task_ids": [], "task_payload_sha256": "1" * 64},
        {"task_id": "p1", "stage": "v16_pair_hypothesis_cluster",
         "upstream_task_ids": ["c1"], "task_payload_sha256": "2" * 64},
    ]
    value = _seal({
        "schema": TOPOLOGICAL_PARENT_SCHEMA,
        "tasks": rows,
        "task_closure_sha256": stable_json_sha256(rows),
    })
    receipt = build_topological_parent_receipt(value)
    assert receipt["execution_performed"] is False
    assert receipt["production_dispatch_available"] is False
    assert receipt["failure_type"] == "PRODUCTION_STAGE_ADAPTER_UNAVAILABLE"

    invalid_rows = [rows[1], rows[0]]
    invalid = _seal({
        "schema": TOPOLOGICAL_PARENT_SCHEMA,
        "tasks": invalid_rows,
        "task_closure_sha256": stable_json_sha256(invalid_rows),
    })
    with pytest.raises(Fixed4SubprocessContractError,
                       match="non-topological"):
        build_topological_parent_receipt(invalid)
