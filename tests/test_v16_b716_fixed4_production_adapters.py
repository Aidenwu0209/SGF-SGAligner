import json
import math
from pathlib import Path

import numpy as np
import pytest

import safety.v16_b716_fixed4_production_adapters as adapters
from safety.v13_dual_solver_runtime import (
    array_sha256, sha256_file, stable_json_sha256, summarize_workers,
)
from safety.v16_b716_fixed4_stage_runners import classify_finite_consensus


IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _sealed(value):
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _task(stage, task_id="task-1", upstream=()):
    return _sealed({"task_id": task_id, "stage": stage,
                    "pair_id": "src_to_ref",
                    "upstream_task_ids": list(upstream),
                    "preflight_identity": {"git_head": "a" * 40}})


def _file(path: Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _file_row(role, path: Path, text="x"):
    return {"role": role, **_file(path, text)}


def _directory_row(role, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({"path": str(path.relative_to(root)),
                     "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)})
    return {"role": role, "path": str(root.resolve()), "files": rows,
            "closure_sha256": stable_json_sha256(rows)}


def _output(role, path, kind="file"):
    return {"role": role, "path": f"production/{path}", "kind": kind}


def _manifest(task, *, files, directories, outputs, parameters):
    return _sealed({"schema": adapters.INPUT_MANIFEST_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "stage": task["stage"], "file_inputs": files,
        "directory_inputs": directories, "outputs": outputs,
        "parameters": parameters, **adapters.POLICY_FALSE})


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, sort_keys=True))
    return sha256_file(path)


def _color_fixture(tmp_path):
    task = _task("colorpcr_direction")
    files = [_file_row(role, tmp_path / "inputs" / role) for role in (
        "sgaligner_python", "jojo_python", "sentinel_subprocess",
        "sentinel_worker", "corr_converter", "weights", "prepared_input",
        "extension")]
    repo = tmp_path / "repo"; _file(repo / "tracked.py")
    repo_row = _directory_row("colorpcr_repo", repo)
    manifest = _manifest(task, files=files,
        directories=[repo_row], outputs=[
            _output("sentinel_cache", "sentinel.npz"),
            _output("sentinel_evidence_dir", "sentinel_evidence", "directory"),
            _output("exact_three_cache", "exact3.npz"),
            _output("conversion_receipt", "conversion.json")],
        parameters={"colorpcr_dependency_identity": {
            "commit": "b" * 40,
            "repo_closure_sha256": repo_row["closure_sha256"],
            "python_tree_sha256": "b" * 64,
            "tracked_diff_sha256": "c" * 64},
            "arm": "sgf_selected_union", "direction": "forward",
            "neighbor_limits": [38, 36, 36, 38], "sampling": "voxel10",
            "device": "cuda:0"})
    return task, manifest


def _pilot_fixture(tmp_path):
    task = _task("bidirectional_multi_solver_pilot")
    files = [_file_row(role, tmp_path / "inputs" / role) for role in (
        "python", "v14_builder", "v14_strict_runner",
        "forward_exact_three_cache", "reverse_exact_three_cache",
        "v13_preregister", "v14_preregister", "preflight_manifest",
        "pointdsc_checkpoint", "prepared_input")]
    pointdsc = tmp_path / "pointdsc"; _file(pointdsc / "model.py")
    manifest = _manifest(task, files=files,
        directories=[_directory_row("pointdsc_root", pointdsc)], outputs=[
            _output("forward_candidate_dir", "forward", "directory"),
            _output("reverse_candidate_dir", "reverse", "directory"),
            _output("candidate_set", "candidate_set.json"),
            _output("slot_root", "slots", "directory"),
            _output("v15_outcome", "v15_outcome.json")],
        parameters={"pair_id": "src_to_ref", "arm": "sgf_selected_union",
                    "max_candidate_slots": 8, "device": "cpu"})
    return task, manifest


def _pose(rotation_deg=0.0, translation_m=0.0):
    angle = math.radians(rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0.0, translation_m],
                       [s, c, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def _science_worker(solver, direction, repeat, transform):
    cache_sha = "1" * 64 if direction == "forward" else "2" * 64
    corr_sha = "3" * 64 if direction == "forward" else "4" * 64
    value = {"schema": "v13-dual-solver-worker-v1", "solver": solver,
        "direction": direction, "repeat": repeat, "permutation_seed": repeat,
        "selected_original_indices_sha256": "5" * 64,
        "cache_path": f"/{direction}.npz", "cache_sha256": cache_sha,
        "correspondence_sha256": corr_sha, "runtime_sha256": "6" * 64,
        "unit": "metre", "transform_direction": (
            "source_to_reference" if direction == "forward"
            else "reference_to_source"),
        "gt_free": True, "gt_inputs": [], "fallback_used": False,
        "correspondence_count": 100, "dependency": {},
        "known_bad_pair": False, "status": "ok", "failure_type": None,
        "transform": np.asarray(transform).tolist(), "diagnostics": {}}
    value["evidence_sha256"] = stable_json_sha256(value)
    return value


def _write_science_evidence(slot, *, rotation_deg=0.3,
                            translation_m=0.003):
    point_forward = _pose()
    pyg_forward = _pose(rotation_deg, translation_m)
    forward = {"pointdsc": point_forward, "pygcransac": pyg_forward}
    rows = []
    worker_dir = slot / "raw" / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    for solver in ("pointdsc", "pygcransac"):
        for direction in ("forward", "reverse"):
            transform = (forward[solver] if direction == "forward"
                         else np.linalg.inv(forward[solver]))
            for repeat in range(5):
                row = _science_worker(solver, direction, repeat, transform)
                _file(worker_dir / f"{solver}_{direction}_{repeat}.json",
                      json.dumps(row, allow_nan=False))
                rows.append(row)
    raw = summarize_workers(rows)
    raw.update({"cache_sha256": {"forward": "1" * 64,
                                  "reverse": "2" * 64},
                "correspondence_sha256": {"forward": "3" * 64,
                                           "reverse": "4" * 64},
                "runtime_sha256": "6" * 64, "worker_count": 20,
                "worker_evidence_sha256": {
                    f"{row['solver']}/{row['direction']}/{row['repeat']}":
                        row["evidence_sha256"] for row in rows}})
    medoids = {}
    for solver in ("pointdsc", "pygcransac"):
        for direction in ("forward", "reverse"):
            name = f"{solver}/{direction}"
            transform = np.asarray(raw["gates"][name]["medoid_transform"])
            medoids[name] = {"raw_transform": transform.tolist(),
                             "final_transform": transform.tolist()}
    strict = {"safe": True, "medoid_safety": medoids}
    return raw, strict


def _finalization_fixture(tmp_path, monkeypatch, *, rotation_deg=0.3,
                          translation_m=0.003):
    task, manifest = _pilot_fixture(tmp_path)
    task_root = tmp_path / "out" / "tasks" / task["task_id"] / "production"
    (task_root / "forward").mkdir(parents=True)
    (task_root / "reverse").mkdir()
    candidate_set = task_root / "candidate_set.json"
    candidate_set.write_text("candidate-set")
    monkeypatch.setattr(adapters, "verify_candidate_set_contract", lambda _path: {
        "value": {"candidate_count": 2},
        "candidate_set_payload_sha256": "d" * 64})
    expansion = adapters.expand_verified_candidate_slots(
        task, manifest, sha256_file(candidate_set), tmp_path / "out")
    rows = []
    for index in range(8):
        if index >= 2:
            rows.append({"candidate_slot": index,
                "status": "typed_not_generated", "returncode": None,
                "raw_summary_path": None, "raw_summary_sha256": None,
                "strict_summary_path": None, "strict_summary_sha256": None})
            continue
        slot = task_root / "slots" / f"slot_{index:02d}"
        raw, strict = _write_science_evidence(
            slot, rotation_deg=rotation_deg, translation_m=translation_m)
        strict.update({"safe": index == 0, "candidate_index": index,
                       "candidate_sha256": f"{index + 7:x}" * 64})
        raw_path = slot / "raw_summary.json"
        strict_path = slot / "summary.json"
        _file(raw_path, json.dumps(raw, allow_nan=False))
        _file(strict_path, json.dumps(strict, allow_nan=False))
        rows.append({"candidate_slot": index, "status": "stage_succeeded",
            "returncode": 0, "raw_summary_path": str(raw_path.resolve()),
            "raw_summary_sha256": sha256_file(raw_path),
            "strict_summary_path": str(strict_path.resolve()),
            "strict_summary_sha256": sha256_file(strict_path)})
    result_manifest = _sealed({"schema": adapters.SLOT_RESULTS_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "slot_expansion_payload_sha256": expansion["payload_sha256"],
        "candidate_set_sha256": expansion["candidate_set_sha256"],
        "slots": rows, **adapters.POLICY_FALSE})
    path = tmp_path / "slot-results.json"
    _file(path, json.dumps(result_manifest))
    decision = {"accepted": True, "known_bad": False,
        "selected_candidate_index": 0,
        "selected_candidate_sha256": "7" * 64,
        "selected_realization": "pointdsc/forward",
        "selected_transform": IDENTITY,
        "selected_transform_sha256": array_sha256(
            np.asarray(IDENTITY, dtype=np.float64))}
    monkeypatch.setattr(adapters, "load_candidate_contract",
                        lambda _path, index: {"candidate": index})
    monkeypatch.setattr(adapters, "select_unique_safe_pose_cluster",
        lambda evidence, known_bad: dict(decision))
    return (task, manifest, expansion, path, rows, result_manifest, decision,
            task_root)


def _rebind_slot_results(path, result_manifest, rows):
    value = _sealed({**{key: item for key, item in result_manifest.items()
        if key not in {"payload_sha256", "slots"}}, "slots": rows})
    _file(path, json.dumps(value))
    return value


def test_color_contract_uses_reviewed_sentinel_and_converter_argv(tmp_path):
    task, manifest = _color_fixture(tmp_path)
    path = tmp_path / "input-manifest.json"
    loaded = adapters.load_bound_input_manifest(
        path, _write_manifest(path, manifest), task, tmp_path / "out")
    value = adapters.build_stage_adapter_contract(task, loaded, tmp_path / "out")
    assert value["execution_authorized"] is False
    assert len(value["commands"]) == 2
    sentinel, converter = value["commands"]
    assert sentinel["argv"][1].endswith("sentinel_subprocess")
    assert sentinel["argv"][0].endswith("sgaligner_python")
    assert sentinel["argv"][sentinel["argv"].index("--python") + 1].endswith(
        "jojo_python")
    assert sentinel["argv"][sentinel["argv"].index("--sampling") + 1] == "voxel10"
    assert sentinel["argv"][sentinel["argv"].index("--neighbor-limits") + 1] == \
        "38,36,36,38"
    assert sentinel["argv"][sentinel["argv"].index("--expected-commit") + 1] == \
        "b" * 40
    assert sentinel["argv"][sentinel["argv"].index("--expected-commit") + 1] != \
        task["preflight_identity"]["git_head"]
    assert converter["argv"][1].endswith("corr_converter")
    assert converter["argv"][0].endswith("sgaligner_python")
    assert all(row["shell"] is False and row["environment_inherited"] is False
               for row in value["commands"])


def test_color_dependency_identity_is_bound_to_repo_closure(tmp_path):
    task, manifest = _color_fixture(tmp_path)
    manifest["parameters"]["colorpcr_dependency_identity"][
        "repo_closure_sha256"] = "f" * 64
    with pytest.raises(adapters.ProductionAdapterError,
                       match="frozen parameter drift"):
        adapters.build_stage_adapter_contract(task, manifest, tmp_path / "out")


def test_manifest_detects_file_and_directory_tamper(tmp_path):
    task, manifest = _color_fixture(tmp_path)
    path = tmp_path / "input-manifest.json"
    digest = _write_manifest(path, manifest)
    Path(manifest["file_inputs"][0]["path"]).write_text("changed")
    with pytest.raises(adapters.ProductionAdapterError, match="bytes/SHA"):
        adapters.load_bound_input_manifest(path, digest, task, tmp_path / "out")

    task, manifest = _color_fixture(tmp_path / "fresh")
    path = tmp_path / "fresh" / "input-manifest.json"
    digest = _write_manifest(path, manifest)
    Path(manifest["directory_inputs"][0]["path"], "extra.py").write_text("x")
    with pytest.raises(adapters.ProductionAdapterError, match="shallow or changed"):
        adapters.load_bound_input_manifest(path, digest, task, tmp_path / "out2")


def test_forbidden_or_existing_output_fails_closed(tmp_path):
    task, manifest = _color_fixture(tmp_path)
    manifest["outputs"][0]["path"] = "production/fused_map.ply"
    manifest["payload_sha256"] = stable_json_sha256(
        {key: value for key, value in manifest.items() if key != "payload_sha256"})
    path = tmp_path / "input-manifest.json"
    with pytest.raises(adapters.ProductionAdapterError, match="forbidden"):
        adapters.load_bound_input_manifest(
            path, _write_manifest(path, manifest), task, tmp_path / "out")


def test_pilot_contract_and_exact_eight_slot_expansion(tmp_path, monkeypatch):
    task, manifest = _pilot_fixture(tmp_path)
    contract = adapters.build_stage_adapter_contract(task, manifest, tmp_path / "out")
    assert len(contract["commands"]) == 3
    assert [row["argv"][2] for row in contract["commands"]] == [
        "build-direction", "build-direction", "pair-directions"]
    task_root = tmp_path / "out" / "tasks" / task["task_id"] / "production"
    (task_root / "forward").mkdir(parents=True)
    (task_root / "reverse").mkdir()
    candidate_set = task_root / "candidate_set.json"
    candidate_set.write_text("candidate-set")
    monkeypatch.setattr(adapters, "verify_candidate_set_contract", lambda _path: {
        "value": {"candidate_count": 2},
        "candidate_set_payload_sha256": "d" * 64})
    value = adapters.expand_verified_candidate_slots(
        task, manifest, sha256_file(candidate_set), tmp_path / "out")
    assert len(value["slots"]) == 8
    assert [row["status"] for row in value["slots"]] == [
        "generated", "generated", *(["typed_not_generated"] * 6)]
    generated = value["slots"][0]["command"]
    assert generated["normal_return_codes"] == [0, 2]
    assert generated["argv"][generated["argv"].index("--candidate-index") + 1] == "0"


def test_v15_finalization_requires_exact20_and_closes_absent_slots(
        tmp_path, monkeypatch):
    (task, manifest, expansion, path, rows, result_manifest, decision,
     _task_root) = _finalization_fixture(tmp_path, monkeypatch)
    value = adapters.finalize_v15_from_slot_results(
        task, manifest, expansion, path, sha256_file(path), tmp_path / "out")
    assert value["v15_decision"] == decision
    observation = value["gate_observation"]
    assert observation["gate_status"] == "PASS"
    assert observation["measured_rotation_deg"] == pytest.approx(0.3)
    assert observation["measured_translation_m"] == pytest.approx(0.003)
    assert observation["measurement_source_file_sha256"] == \
        rows[0]["raw_summary_sha256"]
    assert observation["measurement_source_payload_sha256"] == \
        stable_json_sha256(json.loads(
            Path(rows[0]["raw_summary_path"]).read_text()))
    assert observation["measurement_candidate_slot"] == 0
    assert observation["measurement_candidate_set_sha256"] == \
        expansion["candidate_set_sha256"]
    assert observation["measurement_slot_results_payload_sha256"] == \
        result_manifest["payload_sha256"]
    assert observation["measurement_v15_decision_sha256"] == \
        stable_json_sha256(decision)
    assert value["downstream_authorized"] is False

    raw_path = Path(rows[0]["raw_summary_path"])
    original_raw = json.loads(raw_path.read_text())
    original_raw["cross_solver_check"]["translation_m"] += 0.01
    _file(raw_path, json.dumps(original_raw))
    with pytest.raises(adapters.ProductionAdapterError, match="SHA mismatch"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path), tmp_path / "out")
    rows[0]["raw_summary_sha256"] = sha256_file(raw_path)
    result_manifest = _rebind_slot_results(path, result_manifest, rows)
    with pytest.raises(adapters.ProductionAdapterError,
                       match="disagrees with worker evidence"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path), tmp_path / "out")


def test_science_verifier_rejects_worker_and_selected_medoid_tamper(
        tmp_path, monkeypatch):
    (task, manifest, expansion, path, rows, _result, decision,
     task_root) = _finalization_fixture(tmp_path, monkeypatch)
    worker = task_root / "slots/slot_00/raw/workers/pointdsc_forward_0.json"
    value = json.loads(worker.read_text())
    value["transform"][0][3] = 0.05
    _file(worker, json.dumps(value))
    with pytest.raises(adapters.ProductionAdapterError,
                       match="payload/SHA mismatch"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path), tmp_path / "out")

    # A caller cannot substitute a different V15 observed transform either.
    (task, manifest, expansion, path, _rows, _result, decision,
     _task_root) = _finalization_fixture(tmp_path / "fresh", monkeypatch)
    decision["selected_transform"] = _pose(1.0).tolist()
    with pytest.raises(adapters.ProductionAdapterError,
                       match="selected V15 medoid transform"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path),
            tmp_path / "fresh/out")


def test_science_verifier_rejects_missing_nan_and_direction_drift(
        tmp_path, monkeypatch):
    (task, manifest, expansion, path, rows, result, _decision,
     task_root) = _finalization_fixture(tmp_path, monkeypatch)
    raw_path = Path(rows[0]["raw_summary_path"])
    raw = json.loads(raw_path.read_text())
    raw["cross_solver_check"]["rotation_deg"] = float("nan")
    _file(raw_path, json.dumps(raw))
    rows[0]["raw_summary_sha256"] = sha256_file(raw_path)
    _rebind_slot_results(path, result, rows)
    with pytest.raises(adapters.ProductionAdapterError,
                       match="finite nonnegative"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path), tmp_path / "out")

    (task, manifest, expansion, path, rows, result, _decision,
     task_root) = _finalization_fixture(tmp_path / "missing", monkeypatch)
    missing = task_root / "slots/slot_00/raw/workers/pointdsc_forward_0.json"
    missing.unlink()
    with pytest.raises(adapters.ProductionAdapterError,
                       match="not exact20"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path),
            tmp_path / "missing/out")

    (task, manifest, expansion, path, rows, result, _decision,
     _task_root) = _finalization_fixture(tmp_path / "direction", monkeypatch)
    raw_path = Path(rows[0]["raw_summary_path"])
    raw = json.loads(raw_path.read_text())
    raw["direction_checks"]["pointdsc"]["translation_m"] = 0.01
    _file(raw_path, json.dumps(raw))
    rows[0]["raw_summary_sha256"] = sha256_file(raw_path)
    _rebind_slot_results(path, result, rows)
    with pytest.raises(adapters.ProductionAdapterError,
                       match="direction translation disagrees"):
        adapters.finalize_v15_from_slot_results(
            task, manifest, expansion, path, sha256_file(path),
            tmp_path / "direction/out")


@pytest.mark.parametrize(("rotation", "translation", "expected"), [
    (5.0, 0.10, "PASS"),
    (5.000001, 0.10, "FAIL"),
    (5.0, 0.100001, "FAIL"),
])
def test_science_verifier_frozen_finite_boundaries(
        tmp_path, monkeypatch, rotation, translation, expected):
    (task, manifest, expansion, path, _rows, _result, _decision,
     _task_root) = _finalization_fixture(
        tmp_path, monkeypatch, rotation_deg=rotation,
        translation_m=translation)
    value = adapters.finalize_v15_from_slot_results(
        task, manifest, expansion, path, sha256_file(path), tmp_path / "out")
    assert value["gate_observation"]["gate_status"] == expected


@pytest.mark.parametrize(("returncode", "summary", "expected"), [
    (0, True, "stage_succeeded"), (2, True, "normal_gate_failed"),
    (2, False, "typed_process_failure"), (1, True, "typed_process_failure")])
def test_returncode_classification(returncode, summary, expected):
    assert adapters.classify_stage_returncode(
        returncode, summary_exists=summary) == expected


def test_gate_contract_hash_binds_semantic_parent_results(tmp_path):
    parent_ids = ["h0", "h1"]
    task = _task("v16_pair_hypothesis_cluster", upstream=parent_ids)
    parents = []
    payload_shas = []
    for index, task_id in enumerate(parent_ids):
        value = _sealed({"schema": "runner", "task_id": task_id,
            "stage": "bidirectional_multi_solver_pilot",
            "status": "succeeded", "gt_consumed": False,
            "official92_run": False, "thresholds_changed": False,
            "result_selection_used": False, "default_checkpoint_replaced": False,
            "refusion_run": False, "reconstruction_authorized": False,
            "hypothesis_outcome": {"hypothesis_task_id": task_id,
                "gate_status": "PASS", "failure_class": None,
                "safe_transform": IDENTITY,
                "measured_rotation_deg": 0.0,
                "measured_translation_m": 0.0,
                "measurement_source_file_sha256": "f" * 64,
                "measurement_source_payload_sha256": "e" * 64,
                "measurement_candidate_slot": index,
                "measurement_candidate_set_sha256": "1" * 64,
                "measurement_slot_results_payload_sha256": "2" * 64,
                "measurement_v15_decision_sha256": "3" * 64,
                "source_result_payload_sha256": f"{index + 1:x}" * 64}})
        payload_shas.append(value["payload_sha256"])
        parents.append(_file_row(f"parent_result_{index}",
            tmp_path / f"parent-{index}.json", json.dumps(value)))
    manifest = _manifest(task, files=parents, directories=[],
        outputs=[_output("core_gate_receipt", "gate.json")],
        parameters={"parent_task_ids": parent_ids,
                    "parent_result_payload_sha256s": payload_shas})
    contract = adapters.build_stage_adapter_contract(task, manifest, tmp_path / "out")
    assert contract["verified_parent_result_payload_sha256s"] == payload_shas
    assert [row["hypothesis_task_id"] for row in
            contract["verified_parent_outcomes"]] == parent_ids
    assert contract["adapter_action"] == "invoke_frozen_pair_stage_runner"

    manifest["parameters"]["parent_result_payload_sha256s"][0] = "f" * 64
    with pytest.raises(adapters.ProductionAdapterError, match="semantic binding"):
        adapters.build_stage_adapter_contract(task, manifest, tmp_path / "out")


def test_frozen_pair_and_aggregate_gates_are_invoked_without_authorization():
    safe = classify_finite_consensus(hypothesis_task_id="h0",
        rotation_deg=0.0, translation_m=0.0, transform=IDENTITY)
    pair_payload = {"task_id": "pair-task", "pair_id": "p0",
        "replayed_hypothesis_task_ids": ["h0"],
        "eligible_hypothesis_task_ids": ["h0"],
        "typed_abstention_hypothesis_task_ids": [],
        "hypothesis_gate_results": [safe],
        "cluster_decision": {"accepted": True,
            "reason": "unique_safe_hypothesis_pose_cluster",
            "selected_transform": IDENTITY}, "known_bad": False}
    verified_hypotheses = [{"hypothesis_task_id": "h0", "gate_status": "PASS",
        "failure_class": None, "safe_transform": IDENTITY,
        "measured_rotation_deg": 0.0, "measured_translation_m": 0.0,
        "measurement_source_file_sha256": "c" * 64,
        "measurement_source_payload_sha256": "b" * 64,
        "measurement_candidate_slot": 0,
        "measurement_candidate_set_sha256": "d" * 64,
        "measurement_slot_results_payload_sha256": "e" * 64,
        "measurement_v15_decision_sha256": "f" * 64,
        "source_result_payload_sha256": "a" * 64}]
    operational = adapters.run_frozen_pair_gate(
        pair_payload, verified_parent_outcomes=verified_hypotheses)
    assert operational["status"] == "succeeded"
    assert operational["safe_vote_hypothesis_task_ids"] == ["h0"]
    with pytest.raises(adapters.ProductionAdapterError,
                       match="not parent-derived"):
        adapters.run_frozen_pair_gate(
            {**pair_payload, "eligible_hypothesis_task_ids": []},
            verified_parent_outcomes=verified_hypotheses)

    normal_rows = []
    for index in range(3):
        row = dict(pair_payload, task_id=f"pair-task-{index}", pair_id=f"p{index}")
        normal_rows.append(adapters.build_pair_gate_result(**row))
    bad = dict(pair_payload, task_id="pair-task-3", pair_id="p3", known_bad=True)
    normal_rows.append(adapters.build_pair_gate_result(**bad))
    aggregate_payload = {"task_id": "aggregate",
        "pair_results": normal_rows, "expected_pair_ids": ["p0", "p1", "p2", "p3"],
        "known_bad_pair_id": "p3"}
    verified_pairs = [{"task_id": row["task_id"],
        "status": "succeeded" if row["gate_status"] == "PASS" else "typed_failure",
        "decision": row["decision"], "safe_cluster_transform": row["transform"],
        "source_result_payload_sha256": row["payload_sha256"]}
        for row in normal_rows]
    aggregate = adapters.run_frozen_aggregate_gate(
        aggregate_payload, verified_parent_outcomes=verified_pairs)
    assert aggregate["status"] == "succeeded"
    assert aggregate["replayed_pair_task_ids"] == [
        "pair-task-0", "pair-task-1", "pair-task-2", "pair-task-3"]
    poisoned = [dict(row) for row in verified_pairs]
    poisoned[0]["decision"] = "CALLER_INVENTED_DECISION"
    with pytest.raises(adapters.ProductionAdapterError,
                       match="not parent-derived"):
        adapters.run_frozen_aggregate_gate(
            aggregate_payload, verified_parent_outcomes=poisoned)


def test_materialize_is_create_only(tmp_path):
    value = _sealed({"schema": "test"})
    out = tmp_path / "contracts" / "contract.json"
    receipt = adapters.materialize_contract_create_only(tmp_path, out, value)
    assert receipt["sha256"] == sha256_file(out)
    with pytest.raises(adapters.ProductionAdapterError, match="already exists"):
        adapters.materialize_contract_create_only(tmp_path, out, value)
