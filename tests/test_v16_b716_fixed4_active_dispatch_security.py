import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import safety.v16_b716_fixed4_execution_pilot as pilot
import safety.v16_b716_fixed4_subprocess_contract as contract
from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256


def _seal(value):
    result = dict(value)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _chain_case(tmp_path):
    root = tmp_path / "run"
    preflight_path = root / "execution_preflight.json"
    manifest_path = root / "task_manifest.json"
    task_path = root / "tasks/t0/task.json"
    execution_path = root / "tasks/t0/control/production_execution_manifest.json"
    commit_path = execution_path.with_name("COMMITTED.json")
    preflight = _seal({"repo_root": str(tmp_path / "repo"),
                       "output_root": str(root)})
    manifest = _seal({"tasks": []})
    task = _seal({"task_id": "t0", "stage": "colorpcr_direction"})
    execution = _seal({"task_id": "t0"})
    commit = _seal({"transaction_state": "COMMITTED", "task_id": "t0"})
    for path, value in ((preflight_path, preflight), (manifest_path, manifest),
                        (task_path, task), (execution_path, execution),
                        (commit_path, commit)):
        _write(path, value)
    now = datetime.now(timezone.utc)

    def request(*, renewal_of=None, offset=0):
        issued = now + timedelta(seconds=offset)
        body = {"schema": pilot.ACTIVE_STAGE_AUTHORIZATION_SCHEMA,
            "issued_at": issued.isoformat(),
            "expires_at": (issued + timedelta(minutes=30)).isoformat(),
            "task_id": "t0", "task_payload_sha256": task["payload_sha256"],
            "stage": "colorpcr_direction",
            "preflight_path": str(preflight_path.resolve()),
            "preflight_sha256": sha256_file(preflight_path),
            "preflight_payload_sha256": preflight["payload_sha256"],
            "task_manifest_path": str(manifest_path.resolve()),
            "task_manifest_sha256": sha256_file(manifest_path),
            "task_manifest_payload_sha256": manifest["payload_sha256"],
            "execution_manifest_path": str(execution_path.resolve()),
            "execution_manifest_sha256": sha256_file(execution_path),
            "production_manifest_commit_path": str(commit_path.resolve()),
            "production_manifest_commit_sha256": sha256_file(commit_path),
            "production_manifest_commit_payload_sha256":
                commit["payload_sha256"],
            "production_adapter_protocol_ready": True,
            "signer_private_key_not_on_execution_host": True,
            "renewal_of_authorization_payload_sha256": renewal_of,
            **pilot.POLICY_FALSE_FIELDS}
        value = {"schema": pilot.ACTIVE_AUTHORIZATION_REQUEST_SCHEMA,
            "unsigned": True, "signer_location": "off_host",
            "private_key_expected_on_execution_host": False,
            "authorization_body": body,
            "authorization_body_sha256": stable_json_sha256(body)}
        return _seal(value)

    return {"root": root, "preflight": preflight,
        "preflight_path": preflight_path, "manifest": manifest,
        "manifest_path": manifest_path, "task": task,
        "execution_path": execution_path, "request": request, "now": now}


def _store_request(case, request, *, with_auth=True):
    request_path, auth_path = pilot._active_request_paths(
        case["root"], "t0", request)
    _write(request_path, request)
    auth = None
    if with_auth:
        auth = dict(request["authorization_body"])
        auth["signature_b64"] = "test-signature"
        auth = _seal(auth)
        _write(auth_path, auth)
    return request_path, auth_path, auth


def _load(case, monkeypatch):
    monkeypatch.setattr(pilot, "validate_active_stage_authorization",
                        lambda *_args, **_kwargs: None)
    return pilot._load_active_request_chain(
        root=case["root"], task=case["task"],
        preflight=case["preflight"], preflight_path=case["preflight_path"],
        manifest=case["manifest"], manifest_path=case["manifest_path"],
        execution_manifest_path=case["execution_path"], now=case["now"])


def test_versioned_authorization_chain_accepts_unique_renewal(tmp_path, monkeypatch):
    case = _chain_case(tmp_path)
    first = case["request"]()
    _, _, first_auth = _store_request(case, first)
    renewal = case["request"](
        renewal_of=first_auth["payload_sha256"], offset=1)
    renewal_path, renewal_auth_path, renewal_auth = _store_request(case, renewal)
    observed_request, observed_path, observed_auth_path, observed_auth = \
        _load(case, monkeypatch)
    assert observed_request["payload_sha256"] == renewal["payload_sha256"]
    assert observed_path == renewal_path
    assert observed_auth_path == renewal_auth_path
    assert observed_auth["payload_sha256"] == renewal_auth["payload_sha256"]


@pytest.mark.parametrize("attack", ["unknown-parent", "fork", "extra-auth"])
def test_versioned_authorization_chain_rejects_orphans_and_forks(
        tmp_path, monkeypatch, attack):
    case = _chain_case(tmp_path)
    first = case["request"]()
    _, _, first_auth = _store_request(case, first)
    if attack == "unknown-parent":
        _store_request(case, case["request"](renewal_of="f" * 64, offset=1),
                       with_auth=False)
    elif attack == "fork":
        _store_request(case, case["request"](
            renewal_of=first_auth["payload_sha256"], offset=1), with_auth=False)
        _store_request(case, case["request"](
            renewal_of=first_auth["payload_sha256"], offset=2), with_auth=False)
    else:
        extra = case["root"] / "authorizations/t0" / ("e" * 64 + ".json")
        _write(extra, {"unexpected": True})
    with pytest.raises(pilot.Fixed4ExecutionPilotError,
                       match="unknown parent|fork|extra authorization"):
        _load(case, monkeypatch)


def _parent_attempt_case(tmp_path, *, executed_at):
    root = tmp_path / "run"; task_root = root / "tasks/t0"
    task = _seal({"task_id": "t0", "stage": "colorpcr_direction",
                  "upstream_task_ids": []})
    result = _seal({"status": "succeeded"})
    result_path = task_root / "result.json"; _write(result_path, result)
    now = datetime.now(timezone.utc)
    body = {"issued_at": (now - timedelta(hours=3)).isoformat(),
            "expires_at": (now - timedelta(hours=1)).isoformat(),
            "task_id": "t0"}
    request = _seal({"schema": pilot.ACTIVE_AUTHORIZATION_REQUEST_SCHEMA,
        "unsigned": True, "signer_location": "off_host",
        "private_key_expected_on_execution_host": False,
        "authorization_body": body,
        "authorization_body_sha256": stable_json_sha256(body)})
    request_path, auth_path = pilot._active_request_paths(root, "t0", request)
    _write(request_path, request)
    auth = _seal({**body, "signature_b64": "test"}); _write(auth_path, auth)
    validation = _seal({"schema": pilot.ACTIVE_ADAPTER_VALIDATION_SCHEMA,
        "status": "PASS", "task_id": "t0",
        "task_payload_sha256": task["payload_sha256"],
        "stage": "colorpcr_direction", "candidate_path": str(result_path),
        "candidate_sha256": sha256_file(result_path),
        "candidate_payload_sha256": result["payload_sha256"],
        "operational_result_schema": pilot.RESULT_SCHEMA,
        "parent_result_payload_sha256s": [],
        "production_adapter_contract_path": "unused",
        "production_adapter_contract_sha256": "0" * 64,
        "production_adapter_contract_payload_sha256": "0" * 64,
        "production_input_manifest_sha256": "0" * 64,
        "production_input_manifest_payload_sha256": "0" * 64,
        "execution_manifest_path": "unused",
        "execution_manifest_sha256": "0" * 64,
        "execution_manifest_payload_sha256": "0" * 64,
        "production_attempt_path": "unused",
        "production_attempt_sha256": "0" * 64,
        "production_attempt_payload_sha256": "0" * 64,
        "output_artifact_rows": [], "output_artifact_closure_sha256":
            stable_json_sha256([]),
        "validator_source_sha256": "0" * 64,
        "runner_source_sha256": "0" * 64,
        "stage_semantics": {}, "stage_semantics_sha256": stable_json_sha256({}),
        **pilot.POLICY_FALSE_FIELDS})
    validation_path = task_root / "adapter_validation.json"
    _write(validation_path, validation)
    attempt = _seal({"schema": pilot.ACTIVE_STAGE_ATTEMPT_SCHEMA,
        "status": "succeeded", "task_id": "t0",
        "task_payload_sha256": task["payload_sha256"],
        "executed_at": executed_at.isoformat(),
        "authorization_request_path": str(request_path.resolve()),
        "authorization_request_sha256": sha256_file(request_path),
        "authorization_request_payload_sha256": request["payload_sha256"],
        "authorization_path": str(auth_path.resolve()),
        "authorization_sha256": sha256_file(auth_path),
        "authorization_payload_sha256": auth["payload_sha256"],
        "adapter_validation_sha256": sha256_file(validation_path),
        "adapter_validation_payload_sha256": validation["payload_sha256"],
        "result_sha256": sha256_file(result_path),
        "result_payload_sha256": result["payload_sha256"],
        "parent_result_payload_sha256s": [], **pilot.POLICY_FALSE_FIELDS})
    attempt_path = (task_root / "active_attempts" /
                    f"{auth['payload_sha256']}.json")
    _write(attempt_path, attempt)
    commit = _seal({"schema": pilot.ACTIVE_STAGE_COMMIT_SCHEMA,
        "status": "COMMITTED", "task_id": "t0",
        "task_payload_sha256": task["payload_sha256"],
        "attempt_path": str(attempt_path.resolve()),
        "attempt_sha256": sha256_file(attempt_path),
        "attempt_payload_sha256": attempt["payload_sha256"],
        "adapter_validation_path": str(validation_path.resolve()),
        "adapter_validation_sha256": sha256_file(validation_path),
        "adapter_validation_payload_sha256": validation["payload_sha256"],
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "result_payload_sha256": result["payload_sha256"],
        "parent_result_payload_sha256s": [],
        "operational_result_released": True, **pilot.POLICY_FALSE_FIELDS})
    commit_path = task_root / "active_commit.json"
    _write(commit_path, commit)
    return root, task_root, task, result, attempt_path, now, commit_path


def test_completed_parent_uses_execution_time_not_current_auth_time(
        tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    case = _parent_attempt_case(
        tmp_path, executed_at=now - timedelta(hours=2))
    observed = []
    monkeypatch.setattr(
        pilot, "validate_active_stage_authorization",
        lambda *_args, **kwargs: observed.append(kwargs["now"]))
    pilot._validate_active_parent_attempt(
        task=case[2], task_root=case[1], result=case[3],
        upstream_results={}, preflight={}, output_root=case[0])
    assert len(observed) == 1
    assert abs((observed[0] - (now - timedelta(hours=2))).total_seconds()) < 2


def test_parent_attempt_missing_or_outside_ttl_fails_closed(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(pilot, "validate_active_stage_authorization",
                        lambda *_args, **_kwargs: None)
    outside = _parent_attempt_case(
        tmp_path / "outside", executed_at=now - timedelta(minutes=30))
    with pytest.raises(pilot.Fixed4ExecutionPilotError,
                       match="inside authorization TTL"):
        pilot._validate_active_parent_attempt(
            task=outside[2], task_root=outside[1], result=outside[3],
            upstream_results={}, preflight={}, output_root=outside[0])
    missing = _parent_attempt_case(
        tmp_path / "missing", executed_at=now - timedelta(hours=2))
    value = json.loads(missing[4].read_text())
    value.pop("executed_at"); value = _seal(value); _write(missing[4], value)
    with pytest.raises(pilot.Fixed4ExecutionPilotError,
                       match="commit mismatch|keys mismatch"):
        pilot._validate_active_parent_attempt(
            task=missing[2], task_root=missing[1], result=missing[3],
            upstream_results={}, preflight={}, output_root=missing[0])


def test_result_without_final_commit_marker_is_not_consumable(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    case = _parent_attempt_case(
        tmp_path, executed_at=now - timedelta(hours=2))
    case[6].unlink()
    monkeypatch.setattr(pilot, "validate_active_stage_authorization",
                        lambda *_args, **_kwargs: None)
    with pytest.raises(pilot.Fixed4ExecutionPilotError,
                       match="parent active commit missing"):
        pilot._validate_active_parent_attempt(
            task=case[2], task_root=case[1], result=case[3],
            upstream_results={}, preflight={}, output_root=case[0])


def test_signed_control_files_bind_manifest_payload_and_reject_tamper(tmp_path):
    root = tmp_path / "run"; repo = tmp_path / "repo"; repo.mkdir()
    task_path = root / "tasks/t0/task.json"
    preflight_path = root / "execution_preflight.json"
    manifest_path = root / "task_manifest.json"
    task = _seal({"task_id": "t0", "upstream_task_ids": []})
    preflight = _seal({"runner_registry_closure_sha256": "a" * 64,
                       "execution_source_closure_sha256": "b" * 64})
    manifest = _seal({"tasks": []})
    for path, value in ((task_path, task), (preflight_path, preflight),
                        (manifest_path, manifest)):
        _write(path, value)
    authorization = {"repo_root": str(repo.resolve()),
        "output_root": str(root.resolve()), "task_path": str(task_path.resolve()),
        "task_sha256": sha256_file(task_path),
        "task_payload_sha256": task["payload_sha256"],
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_payload_sha256": preflight["payload_sha256"],
        "task_manifest_path": str(manifest_path.resolve()),
        "task_manifest_sha256": sha256_file(manifest_path),
        "task_manifest_payload_sha256": manifest["payload_sha256"],
        "runner_registry_closure_sha256": "a" * 64,
        "execution_source_closure_sha256": "b" * 64,
        "upstream_task_ids": []}
    contract._validate_signed_active_control_files(
        repo=repo, output_root=root, task=task, task_path=task_path,
        preflight=preflight, preflight_path=preflight_path,
        authorization=authorization, task_manifest_path=manifest_path)
    manifest_path.write_text('{"tampered":true}\n')
    with pytest.raises(contract.Fixed4SubprocessContractError,
                       match="SHA drift"):
        contract._validate_signed_active_control_files(
            repo=repo, output_root=root, task=task, task_path=task_path,
            preflight=preflight, preflight_path=preflight_path,
            authorization=authorization, task_manifest_path=manifest_path)
