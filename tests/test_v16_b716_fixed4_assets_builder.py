import json
from pathlib import Path
import subprocess
import sys

import pytest

import safety.v16_b716_fixed4_assets_builder as builder
from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_execution_pilot import (
    ACTIVE_STAGE_INPUT_DESCRIPTOR_V2_SCHEMA,
    POLICY_FALSE_FIELDS,
    PREFLIGHT_SCHEMA,
    RESULT_SCHEMA,
    TASK_SCHEMA,
)
from safety.v16_b716_fixed4_production_manifest_builder import (
    build_production_input_manifest,
    materialize_production_manifests,
)
from safety.v16_b716_fixed4_subprocess_contract import ACTIVE_PREFLIGHT_V2_SCHEMA


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sealed(value):
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _preflight(output_root):
    return _sealed({"schema": PREFLIGHT_SCHEMA,
        "output_root": str(output_root.resolve()),
        "runner_registry_closure_sha256": "a" * 64,
        "active_subprocess_contract": {
            "schema": ACTIVE_PREFLIGHT_V2_SCHEMA, "runner_mode": "active",
            "runner_registry_closure_sha256": "a" * 64,
            "production_adapter_protocol_ready": True}})


def _task(output_root, task_id, stage, *, upstream=(), **extra):
    prepared = _write(output_root / "prepared" / f"{task_id}.json", b"prepared")
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


def _color_parent(output_root, pair_id, direction):
    task = _task(output_root, f"color-{direction}", "colorpcr_direction",
                 pair_id=pair_id, direction=direction)
    task_root = output_root / "tasks" / task["task_id"]
    exact3 = _write(task_root / "production" / "exact_three_cache.npz",
                    direction.encode())
    binding = _sealed({"role": "exact_three_cache", "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "source_path": str(exact3), "source_sha256": sha256_file(exact3),
        **POLICY_FALSE_FIELDS})
    binding_path = task_root / "artifacts" / "exact3.json"
    _write(binding_path, json.dumps(binding, sort_keys=True).encode())
    artifact = {"path": str(binding_path.relative_to(output_root)),
        "bytes": binding_path.stat().st_size, "sha256": sha256_file(binding_path)}
    result = _sealed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "stage": "colorpcr_direction", "status": "succeeded",
        "output_artifacts": [artifact], **POLICY_FALSE_FIELDS})
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task.json").write_text(json.dumps(task, sort_keys=True))
    (task_root / "result.json").write_text(json.dumps(result, sort_keys=True))
    return task, result


def _git(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "t@example"],
                   check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "test"],
                   check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    for relative in builder.DEFAULT_RUNTIME_SOURCE_FILES:
        _write(repo / relative, relative.encode())
    for relative in (
        "scripts/v13_colorpcr_sentinel_subprocess.py",
        "scripts/v13_colorpcr_official_worker.py",
        "scripts/v13_corr_cache_converter.py",
        "scripts/v14_fixed4_input_builder.py",
        "scripts/v14_candidate_strict_runner.py",
        "scripts/v16_b716_fixed4_active_stage_runner.sh",
        "scripts/v16_b716_fixed4_active_production_wrapper.py",
        "manifests/v13_colorpcr_pointdsc_fixed4_preregister.json",
        "manifests/v14_rigid_multihypothesis_preregister.json",
    ):
        _write(repo / relative, relative.encode())
    return repo, _git(repo)


def _source_repo(tmp_path, name, python_tree=False):
    repo = tmp_path / name
    _write(repo / "README.md", name.encode())
    if python_tree:
        _write(repo / "geotransformer/a.py", b"a=1\n")
        _write(repo / "experiments/ColorPCR/b.py", b"b=2\n")
    return repo, _git(repo)


def _documents(tmp_path, monkeypatch):
    repo, repo_commit = _repo(tmp_path)
    color, color_commit = _source_repo(tmp_path, "color", python_tree=True)
    point, point_commit = _source_repo(tmp_path, "point")
    extension = _write(tmp_path / "ext.so", b"extension")
    weights = _write(tmp_path / "weights", b"weights")
    checkpoint = _write(point / "model.pkl", b"checkpoint")
    # model.pkl is deliberately untracked but is covered by the recursive closure.
    preflight = _write(tmp_path / "preflight.json", b"{}")
    monkeypatch.setattr(builder, "COLORPCR_COMMIT", color_commit)
    monkeypatch.setattr(builder, "POINTDSC_COMMIT", point_commit)
    monkeypatch.setattr(builder, "COLORPCR_EXTENSION_SHA256", sha256_file(extension))
    monkeypatch.setattr(builder, "COLORPCR_WEIGHT_SHA256", sha256_file(weights))
    monkeypatch.setattr(builder, "POINTDSC_CHECKPOINT_SHA256", sha256_file(checkpoint))
    monkeypatch.setattr(builder, "COLORPCR_PYTHON_TREE_SHA256",
                        builder.colorpcr_python_tree_sha256(color))
    monkeypatch.setattr(builder, "_probe_python",
        lambda interpreter, modules: (sys.version, [Path(__file__).resolve()],
                                      [str(Path(sys.prefix).resolve())]))
    documents = builder.build_documents(repo=repo,
        expected_repo_commit=repo_commit, colorpcr_repo=color,
        colorpcr_weights=weights, colorpcr_extension=extension,
        pointdsc_root=point, pointdsc_checkpoint=checkpoint,
        sgaligner_python=Path(sys.executable), jojo_python=Path(sys.executable),
        preflight_manifest=preflight, probe_modules=())
    return documents


def test_builds_all_stage_assets_and_runtime_with_recursive_identities(
        tmp_path, monkeypatch):
    documents = _documents(tmp_path, monkeypatch)
    assert set(documents) == {"colorpcr_direction",
        "bidirectional_multi_solver_pilot", "v16_pair_hypothesis_cluster",
        "fixed4_aggregate", "runtime", "receipt"}
    color = documents["colorpcr_direction"]
    assert color["schema"] == builder.PRODUCTION_ASSETS_MANIFEST_SCHEMA
    assert color["directory_assets"][0]["files"]
    assert color["payload_sha256"] == stable_json_sha256(
        {key: value for key, value in color.items() if key != "payload_sha256"})
    pilot = documents["bidirectional_multi_solver_pilot"]
    assert {row["role"] for row in pilot["file_assets"]} >= {
        "pointdsc_checkpoint", "preflight_manifest", "python"}
    assert documents["v16_pair_hypothesis_cluster"]["file_assets"] == []
    runtime = documents["runtime"]
    assert runtime["schema"] == builder.PRODUCTION_RUNTIME_MANIFEST_SCHEMA
    assert Path(runtime["interpreter"]["realpath"]).is_file()
    assert runtime["runner_source"]["role"] == "runner_source"
    assert runtime["production_wrapper_cli"]["role"] == "production_wrapper_cli"
    assert runtime["validator_source"]["role"] == "validator_source"
    assert runtime["runtime_dependency_files"]
    assert all(documents["receipt"][key] is False for key in POLICY_FALSE_FIELDS)


def test_real_assets_documents_feed_pilot_input_and_materialization(
        tmp_path, monkeypatch):
    """Exercise the actual assets builder at the production consumer boundary."""
    documents = _documents(tmp_path, monkeypatch)
    pilot_assets = documents["bidirectional_multi_solver_pilot"]
    roles = [row["role"] for row in pilot_assets["file_assets"]]
    assert roles.count("preflight_manifest") == 1
    assert len(roles) == len(set(roles))

    output_root = tmp_path / "operational"
    output_root.mkdir()
    pair_id = "source_to_reference"
    forward_task, forward = _color_parent(output_root, pair_id, "forward")
    reverse_task, reverse = _color_parent(output_root, pair_id, "reverse")
    upstream = (forward_task["task_id"], reverse_task["task_id"])
    pilot_task = _task(output_root, "pilot.0",
        "bidirectional_multi_solver_pilot", upstream=upstream, pair_id=pair_id)
    upstream_results = {forward_task["task_id"]: forward,
                        reverse_task["task_id"]: reverse}
    preflight = _preflight(output_root)

    input_manifest = build_production_input_manifest(
        task=pilot_task, preflight=preflight, output_root=output_root,
        upstream_results=upstream_results,
        production_assets_manifest=pilot_assets)
    input_roles = [row["role"] for row in input_manifest["file_inputs"]]
    assert input_roles.count("preflight_manifest") == 1
    assert len(input_roles) == len(set(input_roles))

    receipt = materialize_production_manifests(
        task=pilot_task, preflight=preflight, output_root=output_root,
        upstream_results=upstream_results,
        production_assets_manifest=pilot_assets,
        runtime_manifest=documents["runtime"])
    assert receipt["transaction_state"] == "COMMITTED"
    assert Path(receipt["receipt_path"]).is_file()


def test_materialization_is_create_only(tmp_path, monkeypatch):
    documents = _documents(tmp_path, monkeypatch)
    output = tmp_path / "bundle"
    receipt = builder.materialize_documents(output, documents)
    assert len(receipt["files"]) == 6
    for row in receipt["files"].values():
        assert sha256_file(Path(row["path"])) == row["sha256"]
        assert json.loads(Path(row["path"]).read_text())["payload_sha256"] \
            == row["payload_sha256"]
    with pytest.raises(builder.ProductionAssetsBuilderError,
                       match="already exists"):
        builder.materialize_documents(output, documents)


def test_recursive_closure_rejects_symlink(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    target = _write(tmp_path / "target", b"target")
    (root / "link").symlink_to(target)
    with pytest.raises(builder.ProductionAssetsBuilderError, match="symlink"):
        builder.directory_row(root, "source")


def test_git_identity_rejects_tracked_drift(tmp_path):
    repo = tmp_path / "repo"; tracked = _write(repo / "tracked", b"before")
    commit = _git(repo); tracked.write_bytes(b"after")
    with pytest.raises(builder.ProductionAssetsBuilderError, match="dirty"):
        builder.git_identity(repo, commit)


def test_expected_binary_hash_drift_fails_closed(tmp_path, monkeypatch):
    documents = _documents(tmp_path, monkeypatch)
    assert documents["receipt"]["colorpcr_extension_sha256"]
    monkeypatch.setattr(builder, "COLORPCR_EXTENSION_SHA256", "0" * 64)
    # The helper is independently fail-closed even before a bundle is emitted.
    extension = _write(tmp_path / "different.so", b"extension")
    with pytest.raises(builder.ProductionAssetsBuilderError, match="SHA drift"):
        builder._verify_expected(extension, "extension", "0" * 64)


def test_runtime_read_probe_preserves_import_order(tmp_path, monkeypatch):
    observed = {}

    def fake_regular_file(path, _role, **_kwargs):
        return Path(path)

    def fake_run(argv, *, input, **_kwargs):
        observed["modules"] = json.loads(input)["modules"]
        trace_path = Path(argv[argv.index("-o") + 1])
        trace_path.write_text("")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(builder, "_regular_file", fake_regular_file)
    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    builder._probe_python_runtime_reads(
        tmp_path / "python", ("torch", "geotransformer.modules.ops", "torch"),
        (), cuda_probe=False)

    assert observed["modules"] == ["torch", "geotransformer.modules.ops"]
