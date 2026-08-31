import json
import importlib.util
from pathlib import Path
import sys

import pytest

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_active_production_wrapper import (
    ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
    ActiveProductionWrapperError,
    _run_command,
    load_active_production_execution_manifest,
)
from safety.v16_b716_fixed4_execution_pilot import POLICY_FALSE_FIELDS


def _entrypoint_module():
    path = (Path(__file__).resolve().parents[1] / "scripts" /
            "v16_b716_fixed4_active_production_wrapper.py")
    spec = importlib.util.spec_from_file_location(
        "fixed4_active_production_wrapper_entrypoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed(value):
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _fixture(tmp_path):
    task = _sealed({"task_id": "cpu-contract", "stage":
                    "v16_pair_hypothesis_cluster"})
    executable = Path(sys.executable)
    real = executable.resolve(strict=True)
    dependency = Path(__file__).resolve()
    rows = [{"path": str(dependency), "bytes": dependency.stat().st_size,
             "sha256": sha256_file(dependency)}]
    production_input = tmp_path / "production-input.json"
    production_input.write_text("{}")
    value = _sealed({
        "schema": ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "stage": task["stage"],
        "production_input_manifest_path": str(production_input.resolve()),
        "production_input_manifest_sha256": sha256_file(production_input),
        "production_input_manifest_payload_sha256": "1" * 64,
        "interpreter": {"path": str(executable), "realpath": str(real),
            "bytes": real.stat().st_size, "sha256": sha256_file(real),
            "version": sys.version},
        "runtime_dependency_files": rows,
        "runtime_dependency_closure_sha256": stable_json_sha256(rows),
        "controlled_sys_path": [str(Path(__file__).resolve().parents[1] / "src")],
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPYCACHEPREFIX": "/proc/v16-b716-fixed4-no-pyc",
                        "PYTHONHASHSEED": "0", "CUDA_CACHE_DISABLE": "1"},
        "parent_result_payload_sha256s": [],
        "runner_source_sha256": "2" * 64,
        "wrapper_source_sha256": "3" * 64,
        **POLICY_FALSE_FIELDS})
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return task, value, path


def test_cpu_execution_manifest_uses_hash_bound_interpreter_and_dependency_closure(
        tmp_path):
    task, value, path = _fixture(tmp_path)
    loaded = load_active_production_execution_manifest(
        path, sha256_file(path), task=task)
    assert loaded == value
    assert loaded["interpreter"]["realpath"] == str(Path(sys.executable).resolve())
    assert loaded["environment"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in loaded["environment"]


def test_cli_accepts_the_complete_three_entry_controlled_prefix(tmp_path):
    module = _entrypoint_module()
    repo = tmp_path / "repo"
    controlled = [str(repo), str(repo / "src"), str(repo / "scripts"),
                  str(tmp_path / "runtime")]
    assert module._validated_controlled_sys_path(controlled, repo) == controlled


def test_cli_rejects_truncated_or_reordered_controlled_prefix(tmp_path):
    module = _entrypoint_module()
    repo = tmp_path / "repo"
    valid = [str(repo), str(repo / "src"), str(repo / "scripts")]
    for controlled in (valid[:2], [valid[1], valid[0], valid[2]]):
        with pytest.raises(ValueError, match="controlled sys.path contract mismatch"):
            module._validated_controlled_sys_path(controlled, repo)


def test_cpu_execution_manifest_rejects_dependency_tamper(tmp_path):
    task, _value, path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_dependency_files"][0]["sha256"] = "f" * 64
    value["runtime_dependency_closure_sha256"] = stable_json_sha256(
        value["runtime_dependency_files"])
    value["payload_sha256"] = stable_json_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"})
    path.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ActiveProductionWrapperError,
                       match="runtime dependency SHA drift"):
        load_active_production_execution_manifest(path, sha256_file(path), task=task)


def test_cpu_execution_manifest_rejects_pythonpath_pollution(tmp_path):
    task, _value, path = _fixture(tmp_path)
    value = json.loads(path.read_text())
    value["environment"]["PYTHONPATH"] = "/tmp/untrusted"
    value["payload_sha256"] = stable_json_sha256(
        {key: item for key, item in value.items() if key != "payload_sha256"})
    path.write_text(json.dumps(value, sort_keys=True))
    with pytest.raises(ActiveProductionWrapperError,
                       match="environment is not sealed"):
        load_active_production_execution_manifest(path, sha256_file(path), task=task)


def test_controlled_python_launcher_imports_only_explicit_path(tmp_path):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "controlled_only.py").write_text("VALUE = 7\n")
    script = tmp_path / "run.py"
    script.write_text("import controlled_only\nassert controlled_only.VALUE == 7\n")
    command = {"argv": [sys.executable, str(script)],
        "normal_return_codes": [0], "shell": False,
        "environment_inherited": False}
    row = _run_command(command,
        environment={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/proc/v16-b716-fixed4-no-pyc",
            "CUDA_CACHE_DISABLE": "1"},
        controlled_sys_path=[str(modules), *[
            item for item in sys.path if item and Path(item).is_dir()]],
        python_path=sys.executable,
        cwd=tmp_path)
    assert row["returncode"] == 0
    assert row["classification"] == "stage_succeeded"
    assert row["launched_argv_sha256"] != row["argv_sha256"]
