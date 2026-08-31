import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import safety.v16_b716_fixed4_production_manifest_builder as manifest_builder
from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_active_production_wrapper import (
    ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
    load_active_production_execution_manifest,
)
from safety.v16_b716_fixed4_execution_pilot import (
    ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA, POLICY_FALSE_FIELDS,
    PREFLIGHT_SCHEMA, RESULT_SCHEMA, TASK_SCHEMA,
)
from safety.v16_b716_fixed4_production_adapters import (
    INPUT_MANIFEST_SCHEMA, load_bound_input_manifest,
)
from safety.v16_b716_fixed4_production_manifest_builder import (
    PRODUCTION_ASSETS_MANIFEST_SCHEMA, PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
    PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA,
    ProductionManifestBuilderError, build_production_input_manifest,
    load_committed_production_manifest_transaction,
    materialize_production_manifests,
)
from safety.v16_b716_fixed4_subprocess_contract import ACTIVE_PREFLIGHT_V2_SCHEMA


def _sealed(value):
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _file_row(path, role):
    path = Path(path)
    return {"role": role, "path": str(path), "bytes": path.resolve().stat().st_size,
            "sha256": sha256_file(path.resolve())}


def _directory_row(root, role):
    rows = []
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)})
    return {"role": role, "path": str(root), "files": rows,
            "closure_sha256": stable_json_sha256(rows)}


def _preflight(root):
    return _sealed({"schema": PREFLIGHT_SCHEMA, "output_root": str(root.resolve()),
        "runner_registry_closure_sha256": "a" * 64,
        "active_subprocess_contract": {
            "schema": ACTIVE_PREFLIGHT_V2_SCHEMA, "runner_mode": "active",
            "runner_registry_closure_sha256": "a" * 64,
            "production_adapter_protocol_ready": True}})


def _task(root, task_id, stage, *, upstream=(), **extra):
    prepared = _write(root / "prepared" / f"{task_id}.json", b"prepared")
    descriptor = _sealed({
        "schema": ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA,
        "task_id": task_id, "stage": stage,
        "upstream_task_ids": list(upstream),
        "input_source": ("sealed_preregistered_source_closure" if not upstream
                         else "verified_upstream_operational_result_v5_receipts"),
        "derivation_policy": "dispatcher_only_never_trust_task_runtime_paths",
        "production_input_manifest_schema":
            "v16-b716-fixed4-production-input-manifest-v1",
        "production_adapter_contract_schema":
            "v16-b716-fixed4-production-adapter-contract-v1",
        "operational_result_schema": RESULT_SCHEMA,
        "production_adapter_protocol_ready": True})
    return _sealed({"schema": TASK_SCHEMA, "task_id": task_id, "stage": stage,
        "upstream_task_ids": list(upstream), "prepared_input_path": str(prepared),
        "prepared_input_sha256": sha256_file(prepared),
        "preflight_identity": {"runner_registry_closure_sha256": "a" * 64},
        "stage_runner_input_descriptor": descriptor,
        "execution_binding": {"runner_mode": "active",
            "stage_implementation_status": "production_adapter_ready"},
        **extra})


def _assets(files, directories, stage_parameters, stage):
    return _sealed({"schema": PRODUCTION_ASSETS_MANIFEST_SCHEMA,
        "stage": stage,
        "file_assets": files, "directory_assets": directories,
        "stage_parameters": stage_parameters, **POLICY_FALSE_FIELDS})


def _runtime(tmp_path):
    executable = Path(sys.executable)
    real = executable.resolve(strict=True)
    dependency = _write(tmp_path / "runtime" / "dependency.py", b"dependency")
    runner = _write(tmp_path / "runtime" / "runner.sh", b"#!/bin/sh\n")
    wrapper = _write(tmp_path / "runtime" / "wrapper_cli.py", b"# wrapper cli\n")
    validator = _write(tmp_path / "runtime" / "validator.py", b"# validator\n")
    sys_path = tmp_path / "runtime" / "site-packages"
    sys_path.mkdir()
    return _sealed({"schema": PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
        "interpreter": {"path": str(executable), "realpath": str(real),
            "bytes": real.stat().st_size, "sha256": sha256_file(real),
            "version": sys.version},
        "runtime_dependency_files": [{"path": str(dependency),
            "bytes": dependency.stat().st_size, "sha256": sha256_file(dependency)}],
        "controlled_sys_path": [str(sys_path)],
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPYCACHEPREFIX": "/proc/v16-b716-fixed4-no-pyc",
                        "PYTHONHASHSEED": "0", "CUDA_CACHE_DISABLE": "1"},
        "runner_source": _file_row(runner, "runner_source"),
        "production_wrapper_cli": _file_row(wrapper, "production_wrapper_cli"),
        "validator_source": _file_row(validator, "validator_source"),
        **POLICY_FALSE_FIELDS})


def _color_assets(tmp_path, *, symlink_python=False):
    files = []
    for role in ("sgaligner_python", "jojo_python", "sentinel_subprocess",
                 "sentinel_worker", "corr_converter", "weights", "extension"):
        target = _write(tmp_path / "assets" / f"{role}.bin", role.encode())
        source = target
        if symlink_python and role == "sgaligner_python":
            source = tmp_path / "assets" / "sgaligner-link"
            source.symlink_to(target)
        files.append(_file_row(source, role))
    repo = tmp_path / "colorpcr"
    _write(repo / "module.py", b"module")
    identity = {"commit": "1" * 40,
        "repo_closure_sha256": _directory_row(repo, "colorpcr_repo")["closure_sha256"],
        "python_tree_sha256": "2" * 64, "tracked_diff_sha256": "3" * 64}
    return _assets(files, [_directory_row(repo, "colorpcr_repo")],
        {"colorpcr_direction": {"colorpcr_dependency_identity": identity,
                                "arm": "sgf_selected_union", "device": "cpu"}},
        "colorpcr_direction")


def _pilot_assets(tmp_path):
    roles = ("python", "v14_builder", "v14_strict_runner", "v13_preregister",
             "v14_preregister", "preflight_manifest", "pointdsc_checkpoint")
    files = [_file_row(_write(tmp_path / "pilot-assets" / role, role.encode()), role)
             for role in roles]
    repo = tmp_path / "pointdsc"
    _write(repo / "model.py", b"model")
    return _assets(files, [_directory_row(repo, "pointdsc_root")],
                   {"bidirectional_multi_solver_pilot":
                    {"arm": "sgf_selected_union"}},
                   "bidirectional_multi_solver_pilot")


def _store_parent(root, task, result):
    task_root = root / "tasks" / task["task_id"]
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task.json").write_text(json.dumps(task, sort_keys=True))
    (task_root / "result.json").write_text(json.dumps(result, sort_keys=True))


def _write_json(path, value):
    _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())
    return path


def _color_parent(root, pair, direction):
    task = _task(root, f"color-{direction}", "colorpcr_direction", pair_id=pair,
                 direction=direction)
    exact3 = _write(root / "tasks" / task["task_id"] / "production" /
                    "exact_three_cache.npz", direction.encode())
    binding = _sealed({"role": "exact_three_cache", "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "source_path": str(exact3),
        "source_sha256": sha256_file(exact3), **POLICY_FALSE_FIELDS})
    binding_path = root / "tasks" / task["task_id"] / "artifacts" / "exact3.json"
    _write(binding_path, json.dumps(binding, sort_keys=True).encode())
    row = {"path": str(binding_path.relative_to(root)), "bytes": binding_path.stat().st_size,
           "sha256": sha256_file(binding_path)}
    result = _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "stage": "colorpcr_direction", "status": "succeeded",
        "output_artifacts": [row], **POLICY_FALSE_FIELDS})
    _store_parent(root, task, result)
    return task, result


def test_materializes_canonical_color_manifests_and_validates_wrapper_boundary(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.0", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    receipt = materialize_production_manifests(
        task=task, preflight=_preflight(root), output_root=root,
        upstream_results={}, production_assets_manifest=_color_assets(tmp_path),
        runtime_manifest=_runtime(tmp_path))
    transaction_root = (root / "tasks/color.0/control" /
                        "production_manifest_transactions" /
                        receipt["transaction_id"])
    input_path = transaction_root / "production_input_manifest.json"
    execution_path = transaction_root / "production_execution_manifest.json"
    commit_path = transaction_root / "COMMITTED.json"
    assert receipt["production_input_manifest"]["path"] == str(input_path)
    assert receipt["production_execution_manifest"]["path"] == str(execution_path)
    assert receipt["receipt_path"] == str(commit_path)
    assert receipt["receipt_sha256"] == sha256_file(commit_path)
    assert json.loads(commit_path.read_text())["schema"] == \
        PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA
    assert receipt["authorization_binding"]["execution_manifest_sha256"] == \
        sha256_file(execution_path)
    assert receipt["authorization_binding"]["production_wrapper_path"].endswith(
        "wrapper_cli.py")
    input_value = json.loads(input_path.read_text())
    assert input_value["schema"] == INPUT_MANIFEST_SCHEMA
    assert load_bound_input_manifest(input_path, sha256_file(input_path), task, root) \
        == input_value
    execution = json.loads(execution_path.read_text())
    assert execution["schema"] == ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA
    assert load_active_production_execution_manifest(
        execution_path, sha256_file(execution_path), task=task) == execution
    loaded = load_committed_production_manifest_transaction(
        commit_path=commit_path, task=task, preflight=_preflight(root),
        output_root=root)
    assert loaded["production_input_manifest"] == input_value
    assert loaded["production_execution_manifest"] == execution


def test_cli_returns_commit_receipt_path_and_sha(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.cli", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    preflight = _preflight(root)
    assets = _color_assets(tmp_path)
    runtime = _runtime(tmp_path)
    task_path = _write_json(tmp_path / "controls/task.json", task)
    preflight_path = _write_json(tmp_path / "controls/preflight.json", preflight)
    assets_path = _write_json(tmp_path / "controls/assets.json", assets)
    runtime_path = _write_json(tmp_path / "controls/runtime.json", runtime)
    script = (Path(__file__).resolve().parents[1] / "scripts" /
              "v16_b716_fixed4_production_manifest_builder.py")
    completed = subprocess.run([sys.executable, str(script),
        "--task", str(task_path), "--task-sha256", sha256_file(task_path),
        "--preflight", str(preflight_path), "--preflight-sha256",
        sha256_file(preflight_path), "--production-assets-manifest",
        str(assets_path), "--production-assets-manifest-sha256",
        sha256_file(assets_path), "--runtime-manifest", str(runtime_path),
        "--runtime-manifest-sha256", sha256_file(runtime_path),
        "--output-root", str(root)], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["transaction_state"] == "COMMITTED"
    assert receipt["receipt_sha256"] == sha256_file(Path(receipt["receipt_path"]))
    assert receipt["authorization_binding"]["transaction_commit_sha256"] == \
        receipt["receipt_sha256"]


def test_cli_rejects_v1_preflight_with_zero_control_writes(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.cli.reject", "colorpcr_direction",
                 pair_id="a_to_b", direction="forward",
                 neighbor_limits=[38, 36, 36, 38])
    preflight = _preflight(root)
    preflight["active_subprocess_contract"]["schema"] = \
        "v16-b716-fixed4-active-subprocess-preflight-v1"
    preflight["payload_sha256"] = stable_json_sha256({
        key: value for key, value in preflight.items() if key != "payload_sha256"})
    assets = _color_assets(tmp_path)
    runtime = _runtime(tmp_path)
    task_path = _write_json(tmp_path / "controls/task.json", task)
    preflight_path = _write_json(tmp_path / "controls/preflight.json", preflight)
    assets_path = _write_json(tmp_path / "controls/assets.json", assets)
    runtime_path = _write_json(tmp_path / "controls/runtime.json", runtime)
    script = (Path(__file__).resolve().parents[1] / "scripts" /
              "v16_b716_fixed4_production_manifest_builder.py")
    completed = subprocess.run([sys.executable, str(script),
        "--task", str(task_path), "--task-sha256", sha256_file(task_path),
        "--preflight", str(preflight_path), "--preflight-sha256",
        sha256_file(preflight_path), "--production-assets-manifest",
        str(assets_path), "--production-assets-manifest-sha256",
        sha256_file(assets_path), "--runtime-manifest", str(runtime_path),
        "--runtime-manifest-sha256", sha256_file(runtime_path),
        "--output-root", str(root)], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert completed.returncode == 70
    assert not (root / "tasks/color.cli.reject/control").exists()


def test_asset_symlink_is_canonicalized_not_preserved(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.1", "colorpcr_direction", pair_id="a_to_b",
                 direction="reverse", neighbor_limits=[38, 36, 36, 38])
    value = build_production_input_manifest(
        task=task, preflight=_preflight(root), output_root=root,
        upstream_results={},
        production_assets_manifest=_color_assets(tmp_path, symlink_python=True))
    row = next(row for row in value["file_inputs"]
               if row["role"] == "sgaligner_python")
    assert not Path(row["path"]).is_symlink()
    assert row["path"] == str(Path(row["path"]).resolve())


def test_pilot_inputs_are_derived_from_two_canonical_color_results(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    forward_task, forward = _color_parent(root, "a_to_b", "forward")
    reverse_task, reverse = _color_parent(root, "a_to_b", "reverse")
    task = _task(root, "pilot.0", "bidirectional_multi_solver_pilot",
        upstream=(forward_task["task_id"], reverse_task["task_id"]), pair_id="a_to_b")
    value = build_production_input_manifest(
        task=task, preflight=_preflight(root), output_root=root,
        upstream_results={forward_task["task_id"]: forward,
                          reverse_task["task_id"]: reverse},
        production_assets_manifest=_pilot_assets(tmp_path))
    roles = {row["role"] for row in value["file_inputs"]}
    assert {"forward_exact_three_cache", "reverse_exact_three_cache"} <= roles
    assert value["parameters"] == {"pair_id": "a_to_b",
        "arm": "sgf_selected_union", "max_candidate_slots": 8, "device": "cpu"}


@pytest.mark.parametrize("stage,parent_stage", [
    ("v16_pair_hypothesis_cluster", "bidirectional_multi_solver_pilot"),
    ("fixed4_aggregate", "v16_pair_hypothesis_cluster"),
])
def test_gate_manifest_binds_ordered_canonical_parent_results(
        tmp_path, stage, parent_stage):
    root = tmp_path / "out"; root.mkdir()
    parents = []; supplied = {}
    for index in range(2):
        parent_task = _task(root, f"parent.{index}", parent_stage)
        result = _sealed({"schema": RESULT_SCHEMA, "task_id": parent_task["task_id"],
            "task_payload_sha256": parent_task["payload_sha256"],
            "stage": parent_stage, "status": "typed_failure", **POLICY_FALSE_FIELDS})
        _store_parent(root, parent_task, result)
        parents.append(parent_task["task_id"]); supplied[parent_task["task_id"]] = result
    task = _task(root, "gate.0", stage, upstream=parents)
    value = build_production_input_manifest(
        task=task, preflight=_preflight(root), output_root=root,
        upstream_results=supplied,
        production_assets_manifest=_assets([], [], {stage: {}}, stage))
    assert value["parameters"]["parent_task_ids"] == parents
    assert value["parameters"]["parent_result_payload_sha256s"] == \
        [supplied[parent]["payload_sha256"] for parent in parents]


def test_rejects_parent_payload_mismatch(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    parent_task = _task(root, "parent", "bidirectional_multi_solver_pilot")
    result = _sealed({"schema": RESULT_SCHEMA, "task_id": "parent",
        "task_payload_sha256": parent_task["payload_sha256"],
        "stage": "bidirectional_multi_solver_pilot", **POLICY_FALSE_FIELDS})
    _store_parent(root, parent_task, result)
    task = _task(root, "pair", "v16_pair_hypothesis_cluster", upstream=("parent",))
    tampered = dict(result); tampered["status"] = "succeeded"
    with pytest.raises(ProductionManifestBuilderError,
                       match="parent RESULT-v5 binding drift"):
        build_production_input_manifest(
            task=task, preflight=_preflight(root), output_root=root,
            upstream_results={"parent": tampered},
            production_assets_manifest=_assets([], [],
                {"v16_pair_hypothesis_cluster": {}},
                "v16_pair_hypothesis_cluster"))


def test_retry_uses_new_create_only_transaction_without_overwrite(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.2", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    kwargs = dict(task=task, preflight=_preflight(root), output_root=root,
        upstream_results={}, production_assets_manifest=_color_assets(tmp_path),
        runtime_manifest=_runtime(tmp_path))
    first = materialize_production_manifests(**kwargs)
    first_bytes = Path(first["receipt_path"]).read_bytes()
    second = materialize_production_manifests(**kwargs)
    assert first["transaction_id"] != second["transaction_id"]
    assert Path(first["receipt_path"]).read_bytes() == first_bytes
    assert Path(second["receipt_path"]).is_file()


def test_rejects_symlink_control_escape(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.3", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    task_root = root / "tasks/color.3"; task_root.mkdir(parents=True)
    outside = tmp_path / "outside"; outside.mkdir()
    (task_root / "control").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProductionManifestBuilderError,
                       match="create-only failed"):
        materialize_production_manifests(
            task=task, preflight=_preflight(root), output_root=root,
            upstream_results={}, production_assets_manifest=_color_assets(tmp_path),
            runtime_manifest=_runtime(tmp_path))
    assert list(outside.iterdir()) == []


def test_rejects_task_id_path_escape(tmp_path):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "safe", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    task["task_id"] = "../escape"
    task["payload_sha256"] = stable_json_sha256(
        {key: value for key, value in task.items() if key != "payload_sha256"})
    with pytest.raises(ProductionManifestBuilderError, match="path safe"):
        build_production_input_manifest(
            task=task, preflight=_preflight(root), output_root=root,
            upstream_results={}, production_assets_manifest=_color_assets(tmp_path))


@pytest.mark.parametrize("mutation", [
    "preflight_missing_active", "preflight_v1", "preflight_ready_false",
    "task_missing_descriptor", "task_descriptor_v1", "task_descriptor_ready_false",
    "task_binding_not_ready",
])
def test_legacy_or_nonready_controls_fail_before_any_write(tmp_path, mutation):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.reject", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    preflight = _preflight(root)
    if mutation == "preflight_missing_active":
        preflight.pop("active_subprocess_contract")
    elif mutation == "preflight_v1":
        preflight["active_subprocess_contract"]["schema"] = \
            "v16-b716-fixed4-active-subprocess-preflight-v1"
    elif mutation == "preflight_ready_false":
        preflight["active_subprocess_contract"][
            "production_adapter_protocol_ready"] = False
    elif mutation == "task_missing_descriptor":
        task.pop("stage_runner_input_descriptor")
    elif mutation == "task_descriptor_v1":
        task["stage_runner_input_descriptor"]["schema"] = \
            "v16-b716-fixed4-active-stage-input-descriptor-v1"
    elif mutation == "task_descriptor_ready_false":
        task["stage_runner_input_descriptor"][
            "production_adapter_protocol_ready"] = False
    else:
        task["execution_binding"]["stage_implementation_status"] = \
            "contract_fixture_only"
    preflight["payload_sha256"] = stable_json_sha256({
        key: value for key, value in preflight.items() if key != "payload_sha256"})
    if "stage_runner_input_descriptor" in task:
        descriptor = task["stage_runner_input_descriptor"]
        descriptor["payload_sha256"] = stable_json_sha256({
            key: value for key, value in descriptor.items()
            if key != "payload_sha256"})
    task["payload_sha256"] = stable_json_sha256({
        key: value for key, value in task.items() if key != "payload_sha256"})
    with pytest.raises(ProductionManifestBuilderError):
        materialize_production_manifests(
            task=task, preflight=preflight, output_root=root,
            upstream_results={}, production_assets_manifest=_color_assets(tmp_path),
            runtime_manifest=_runtime(tmp_path))
    control = root / "tasks/color.reject/control"
    assert not control.exists()


def test_mid_transaction_failure_is_invisible_and_retry_commits(tmp_path,
                                                               monkeypatch):
    root = tmp_path / "out"; root.mkdir()
    task = _task(root, "color.retry", "colorpcr_direction", pair_id="a_to_b",
                 direction="forward", neighbor_limits=[38, 36, 36, 38])
    preflight = _preflight(root)
    kwargs = dict(task=task, preflight=preflight, output_root=root,
        upstream_results={}, production_assets_manifest=_color_assets(tmp_path),
        runtime_manifest=_runtime(tmp_path))
    original = manifest_builder._write_manifest
    writes = 0

    def fail_second(root_arg, path, value, role):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise ProductionManifestBuilderError("injected second-write failure")
        return original(root_arg, path, value, role)

    monkeypatch.setattr(manifest_builder, "_write_manifest", fail_second)
    with pytest.raises(ProductionManifestBuilderError,
                       match="injected second-write failure"):
        materialize_production_manifests(**kwargs)
    transactions = list((root / "tasks/color.retry/control" /
                         "production_manifest_transactions").iterdir())
    assert len(transactions) == 1
    assert (transactions[0] / "production_input_manifest.json").is_file()
    assert not (transactions[0] / "production_execution_manifest.json").exists()
    assert not (transactions[0] / "COMMITTED.json").exists()
    with pytest.raises(ProductionManifestBuilderError):
        load_committed_production_manifest_transaction(
            commit_path=transactions[0] / "COMMITTED.json", task=task,
            preflight=preflight, output_root=root)

    monkeypatch.setattr(manifest_builder, "_write_manifest", original)
    receipt = materialize_production_manifests(**kwargs)
    assert receipt["transaction_state"] == "COMMITTED"
    assert Path(receipt["receipt_path"]).is_file()
    assert Path(receipt["receipt_path"]).parent != transactions[0]
