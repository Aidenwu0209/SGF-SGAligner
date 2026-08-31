#!/usr/bin/python3.12
"""Candidate-only topological active dispatcher; never signs or runs formal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from safety.v16_b716_fixed4_execution_pilot import (  # noqa: E402
    ACTIVE_STAGE_ATTEMPT_SCHEMA, ACTIVE_STAGE_COMMIT_SCHEMA,
    POLICY_FALSE_FIELDS,
    plan_active_stagewise_dispatch, validate_active_adapter_result_release,
    validate_active_dispatch_execution_inputs,
)
from safety.v16_b716_fixed4_subprocess_contract import (  # noqa: E402
    build_topological_parent_receipt, create_only_bytes_beneath,
    read_no_symlink_bytes, sha256_file, stable_json_sha256,
)


def _write_create_only(root: Path, path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    create_only_bytes_beneath(root, path, encoded, create_parents=True)


def _write_create_or_verify(root: Path, path: Path, value: dict) -> None:
    """Create immutable state, or accept an exact crash-recovery replay."""
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or read_no_symlink_bytes(
                path, "staged active dispatch state") != encoded:
            raise ValueError("staged active dispatch state mismatch")
        return
    create_only_bytes_beneath(root, path, encoded, create_parents=True)


def _load_completed_wrapper(task_root: Path, authorization_path: Path) -> dict:
    """Accept a completed child only when its trace receipt binds this auth."""
    output_path = task_root / "active" / "runner_output.bin"
    consumption_path = task_root / "wrapper" / "active_consumption_receipt.json"
    wrapper = _json(output_path, "production wrapper result")
    consumption = _json(consumption_path, "active consumption receipt")
    controls = consumption.get("control_inputs")
    auth_row = next((row for row in controls or []
                     if row.get("path") == str(authorization_path)), None)
    runner_row = consumption.get("runner_output")
    if (consumption.get("failure_type") is not None
            or consumption.get("operational_result_release_allowed") is not True
            or consumption.get("operational_result_emitted") is not True
            or not isinstance(auth_row, dict)
            or auth_row.get("sha256") != sha256_file(authorization_path)
            or not isinstance(runner_row, dict)
            or runner_row.get("path") != str(output_path)
            or runner_row.get("sha256") != sha256_file(output_path)):
        raise ValueError("completed production wrapper/auth binding mismatch")
    return wrapper


def _json(path: Path, role: str) -> dict:
    value = json.loads(read_no_symlink_bytes(path, role))
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be an object")
    return value


def _execute_authorized(args: argparse.Namespace) -> int:
    """Run exactly one signed node, then release RESULT-v5 in the parent."""
    repo = Path(args.repo).resolve(); root = Path(args.output_root).resolve()
    preflight_path = Path(args.preflight).resolve()
    manifest_path = Path(args.task_manifest).resolve()
    task_path = Path(args.task).resolve()
    request_path = Path(args.authorization_request).resolve()
    authorization_path = Path(args.authorization).resolve()
    execution_path = Path(args.execution_manifest).resolve()
    preflight = _json(preflight_path, "active preflight")
    task = _json(task_path, "active task")
    request = _json(request_path, "authorization request")
    authorization = _json(authorization_path, "signed authorization")
    manifest = _json(manifest_path, "active task manifest")
    execution_started_at = datetime.now(timezone.utc)
    parents = validate_active_dispatch_execution_inputs(
        preflight=preflight, preflight_path=preflight_path,
        manifest=manifest, manifest_path=manifest_path,
        task=task, task_path=task_path, request=request,
        request_path=request_path, authorization=authorization,
        authorization_path=authorization_path,
        execution_manifest_path=execution_path, output_root=root,
        now=execution_started_at)
    task_root = root / "tasks" / str(task["task_id"])
    if task_path != task_root / "task.json":
        raise ValueError("task path is not canonical")
    executor = repo / "scripts/v16_b716_fixed4_active_sealed_executor.py"
    command = ["/usr/bin/python3.12", "-I", "-S", str(executor),
        "--repo", str(repo), "--task", str(task_path),
        "--task-sha256", sha256_file(task_path),
        "--preflight", str(preflight_path),
        "--preflight-sha256", sha256_file(preflight_path),
        "--authorization", str(authorization_path),
        "--authorization-sha256", sha256_file(authorization_path),
        "--task-manifest", str(manifest_path),
        "--task-manifest-sha256", sha256_file(manifest_path),
        "--task-root", str(task_root), "--runner-mode", "active",
        "--execution-manifest", str(execution_path),
        "--execution-manifest-sha256", sha256_file(execution_path)]
    runner_output_path = task_root / "active" / "runner_output.bin"
    if runner_output_path.is_file() and not runner_output_path.is_symlink():
        wrapper = _load_completed_wrapper(task_root, authorization_path)
        completed_returncode = 0
    else:
        completed = subprocess.run(command, cwd=task_root,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            capture_output=True, check=False)
        completed_returncode = completed.returncode
        wrapper = (_load_completed_wrapper(task_root, authorization_path)
                   if completed_returncode == 0 else None)
    if completed_returncode != 0:
        receipt = {"schema": "v16-b716-fixed4-active-dispatch-execution-v1",
            "status": "PROCESS_FAILURE", "task_id": task["task_id"],
            "task_payload_sha256": task["payload_sha256"],
            "sealed_executor_returncode": completed_returncode,
            "operational_result_released": False, **POLICY_FALSE_FIELDS}
        receipt["payload_sha256"] = stable_json_sha256(receipt)
        _write_create_or_verify(root, Path(args.receipt_output).resolve(), receipt)
        return 70
    assert wrapper is not None
    candidate_path = Path(wrapper["candidate_path"])
    validation_path = Path(wrapper["validation_path"])
    candidate = _json(candidate_path, "operational result candidate")
    validation = _json(validation_path, "adapter validation")
    result = validate_active_adapter_result_release(
        task=task, candidate=candidate, candidate_path=candidate_path,
        adapter_validation=validation, adapter_validation_path=validation_path,
        output_root=root, upstream_results=parents, repo_root=repo,
        release_result=False)
    canonical_validation_path = task_root / "adapter_validation.json"
    canonical_validation = _json(validation_path, "adapter validation")
    _write_create_or_verify(root, canonical_validation_path,
                            canonical_validation)
    result_path = task_root / "result.json"
    result_bytes = (json.dumps(result, sort_keys=True, indent=2,
                               allow_nan=False) + "\n").encode()
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    parent_payloads = [parents[parent]["payload_sha256"]
                       for parent in task.get("upstream_task_ids", [])]
    # Bind the attempt to authorization validity at dispatch start.  A valid
    # long-running node may finish after its short-lived authorization expires;
    # downstream consumers revalidate this recorded instant rather than the
    # current clock.
    executed_at = execution_started_at.isoformat().replace("+00:00", "Z")
    attempt_path = (task_root / "active_attempts" /
                    f"{authorization['payload_sha256']}.json")
    if attempt_path.is_file() and not attempt_path.is_symlink():
        prior_attempt = _json(attempt_path, "staged active attempt")
        executed_at = prior_attempt.get("executed_at")
    attempt = {"schema": ACTIVE_STAGE_ATTEMPT_SCHEMA,
        "status": result["status"], "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "executed_at": executed_at,
        "authorization_request_path": str(request_path),
        "authorization_request_sha256": sha256_file(request_path),
        "authorization_request_payload_sha256": request["payload_sha256"],
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "authorization_payload_sha256": authorization["payload_sha256"],
        "adapter_validation_sha256": sha256_file(canonical_validation_path),
        "adapter_validation_payload_sha256": validation["payload_sha256"],
        "result_sha256": result_sha256,
        "result_payload_sha256": result["payload_sha256"],
        "parent_result_payload_sha256s": parent_payloads,
        **POLICY_FALSE_FIELDS}
    attempt["payload_sha256"] = stable_json_sha256(attempt)
    _write_create_or_verify(root, attempt_path, attempt)
    _write_create_or_verify(root, result_path, result)
    commit = {"schema": ACTIVE_STAGE_COMMIT_SCHEMA, "status": "COMMITTED",
        "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "attempt_path": str(attempt_path.resolve()),
        "attempt_sha256": sha256_file(attempt_path),
        "attempt_payload_sha256": attempt["payload_sha256"],
        "adapter_validation_path": str(canonical_validation_path.resolve()),
        "adapter_validation_sha256": sha256_file(canonical_validation_path),
        "adapter_validation_payload_sha256": validation["payload_sha256"],
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "result_payload_sha256": result["payload_sha256"],
        "parent_result_payload_sha256s": parent_payloads,
        "operational_result_released": True, **POLICY_FALSE_FIELDS}
    commit["payload_sha256"] = stable_json_sha256(commit)
    _write_create_or_verify(root, task_root / "active_commit.json", commit)
    receipt = {"schema": "v16-b716-fixed4-active-dispatch-execution-v1",
        "status": result["status"], "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "authorization_payload_sha256": authorization["payload_sha256"],
        "execution_manifest_payload_sha256":
            _json(execution_path, "execution manifest")["payload_sha256"],
        "adapter_validation_payload_sha256": validation["payload_sha256"],
        "result_payload_sha256": result["payload_sha256"],
        "sealed_executor_returncode": completed_returncode,
        "operational_result_released": True, **POLICY_FALSE_FIELDS}
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    _write_create_or_verify(root, Path(args.receipt_output).resolve(), receipt)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-plan", allow_abbrev=False)
    validate.add_argument("--task-order-manifest", required=True)
    validate.add_argument("--receipt-output", required=True)
    dispatch = sub.add_parser("dispatch-next", allow_abbrev=False)
    dispatch.add_argument("--preflight", required=True)
    dispatch.add_argument("--task-manifest", required=True)
    dispatch.add_argument("--output-root", required=True)
    dispatch.add_argument("--signing-key-id", required=True)
    dispatch.add_argument("--receipt-output", required=True)
    execute = sub.add_parser("execute-authorized", allow_abbrev=False)
    for name in ("repo", "preflight", "task-manifest", "task",
                 "authorization-request", "authorization",
                 "execution-manifest", "output-root", "receipt-output"):
        execute.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    if args.command == "validate-plan":
        source = Path(args.task_order_manifest)
        value = json.loads(read_no_symlink_bytes(
            source, "active task-order manifest"))
        receipt = build_topological_parent_receipt(value)
        output = Path(args.receipt_output).resolve()
        _write_create_only(output.parent, output, receipt)
        return 0
    if args.command == "execute-authorized":
        return _execute_authorized(args)

    root = Path(args.output_root).resolve()
    plan = plan_active_stagewise_dispatch(
        preflight_path=Path(args.preflight).resolve(),
        manifest_path=Path(args.task_manifest).resolve(), output_root=root,
        signing_key_id=args.signing_key_id)
    request = plan["unsigned_request"]
    if request is not None:
        request_path = Path(plan["receipt"]["request_path"])
        _write_create_only(root, request_path, request)
        plan["receipt"]["request_path"] = str(request_path)
        plan["receipt"].pop("payload_sha256", None)
        from safety.v13_dual_solver_runtime import stable_json_sha256  # noqa: E402
        plan["receipt"]["payload_sha256"] = stable_json_sha256(plan["receipt"])
    _write_create_only(root, Path(args.receipt_output).resolve(), plan["receipt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
