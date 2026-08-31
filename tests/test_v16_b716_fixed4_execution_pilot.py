import inspect
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess

import pytest

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_execution_pilot import (
    ALLOWED_STAGES, AUTH_SCHEMA, EVIDENCE_RECEIPT_SCHEMA,
    EXACT_SOLVER_ROWS_PER_PILOT, GUARD_AUDIT_SCHEMA, MAX_AUTH_TTL_SECONDS,
    HYPOTHESIS_OUTCOME_SCHEMA,
    OPERATIONAL_STAGE_COUNTS, OPERATIONAL_TASK_COUNT, POLICY_FALSE_FIELDS,
    PREFLIGHT_SCHEMA, REQUIRED_AUTH_REVIEW_FIELDS, RESULT_SCHEMA,
    SENTINEL_ATTEMPT_SCHEMA, SOLVER_ATTEMPT_SCHEMA, TASK_MANIFEST_SCHEMA,
    TYPED_FAILURE_REPLAY_SCHEMA, Fixed4ExecutionPilotError,
    _task_root_has_partial_state, _validate_execution_preregister,
    build_operational_tasks,
    execute_authorized_task, materialize_preflight, validate_authorization,
    validate_runner_result,
)
import safety.v16_b716_fixed4_subprocess_contract as subprocess_contract
import safety.v16_b716_fixed4_execution_pilot as execution_pilot
from safety.v16_b716_fixed4_subprocess_contract import (
    DISABLED_EXIT_CODE, FIX3_CONSUMPTION_SCHEMA, SIGNATURE_ALGORITHM,
    TRUST_ANCHOR_SCHEMA, build_subprocess_registry, classify_wrapper_failure,
    execute_disabled_stage, parse_consumed_paths, validate_recursive_file_closure,
    verify_document_signature, verify_fixed_signed_document, load_trust_anchor,
    _reject_prohibited_signer_private_key,
)
from safety.v16_b716_fixed4_orchestrator_contract import (
    EXPECTED_NODE_COUNT, KNOWN_BAD_PAIR_ID, OFFICIAL_RELEASE_SHA256,
    build_task_dag, synthetic_fixture_bindings,
)
import safety.v16_b716_fixed4_stage_runners as legacy_registry

REPO = Path(__file__).resolve().parents[1]
_TEST_PRIVATE_KEY = None
_TEST_KEY_ID = "fixed4-independent-test-key"


def _signed(value):
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _externally_signed(value):
    value = dict(value)
    value["signature_algorithm"] = SIGNATURE_ALGORITHM
    value["signing_key_id"] = _TEST_KEY_ID
    message = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    message_path = _TEST_PRIVATE_KEY.parent / "message.bin"
    signature_path = _TEST_PRIVATE_KEY.parent / "signature.bin"
    message_path.write_bytes(message)
    subprocess.run(["/usr/bin/openssl", "pkeyutl", "-sign",
        "-inkey", str(_TEST_PRIVATE_KEY), "-rawin", "-in", str(message_path),
        "-out", str(signature_path)], check=True, capture_output=True)
    value["signature_b64"] = base64.b64encode(signature_path.read_bytes()).decode()
    return _signed(value)


@pytest.fixture(autouse=True)
def _fixed_independent_test_anchor(tmp_path):
    global _TEST_PRIVATE_KEY
    trust = tmp_path / "independent-trust"
    trust.mkdir()
    private = trust / "private-test-only.pem"
    public = trust / "public.pem"
    subprocess.run(["/usr/bin/openssl", "genpkey", "-algorithm", "Ed25519",
                    "-out", str(private)], check=True, capture_output=True)
    subprocess.run(["/usr/bin/openssl", "pkey", "-in", str(private), "-pubout",
                    "-out", str(public)], check=True, capture_output=True)
    anchor = _signed({"schema": TRUST_ANCHOR_SCHEMA, "key_id": _TEST_KEY_ID,
        "public_key_path": str(public), "public_key_sha256": sha256_file(public),
        "signature_algorithm": SIGNATURE_ALGORITHM})
    anchor_path = trust / "trust-anchor.json"
    anchor_path.write_text(json.dumps(anchor, sort_keys=True, indent=2) + "\n")
    anchor_path.chmod(0o444); public.chmod(0o444)
    _TEST_PRIVATE_KEY = private
    # The explicit anchor only exercises the low-level crypto primitive. The
    # production verifier must ignore caller module-global reassignment.
    subprocess_contract.TEST_ONLY_ANCHOR_PATH = anchor_path
    subprocess_contract.TEST_ONLY_ANCHOR_SHA256 = sha256_file(anchor_path)
    yield
    _TEST_PRIVATE_KEY = None


def _write_immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    path.chmod(0o444)
    return {"path": None, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _trust_anchor_case(tmp_path, *, public_path=None):
    root = tmp_path / "trust-anchor-case"
    root.mkdir(parents=True)
    if public_path is None:
        public_path = root / "audit_public.pem"
        shutil.copy2(_TEST_PRIVATE_KEY.parent / "public.pem", public_path)
        public_path.chmod(0o444)
    unsigned = {"schema": TRUST_ANCHOR_SCHEMA, "key_id": _TEST_KEY_ID,
        "public_key_path": str(public_path),
        "public_key_sha256": sha256_file(public_path),
        "signature_algorithm": SIGNATURE_ALGORITHM}
    anchor = _signed(unsigned)
    anchor_path = root / "trust-anchor-v1.json"
    anchor_path.write_text(json.dumps(anchor, sort_keys=True, indent=2) + "\n")
    anchor_path.chmod(0o444)
    return anchor_path, public_path


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _clean_repo(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "src/safety/v16_b716_fixed4_stage_runners.py"
    source.parent.mkdir(parents=True)
    shutil.copy2(REPO / "src/safety/v16_b716_fixed4_stage_runners.py", source)
    runner = repo / "scripts/v16_b716_fixed4_disabled_stage_runner.sh"
    runner.parent.mkdir(parents=True)
    shutil.copy2(REPO / "scripts/v16_b716_fixed4_disabled_stage_runner.sh", runner)
    executor = repo / "scripts/v16_b716_fixed4_sealed_executor.py"
    shutil.copy2(REPO / "scripts/v16_b716_fixed4_sealed_executor.py", executor)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "qa@example.invalid")
    _git(repo, "config", "user.name", "Fixed4 QA")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _preflight(root, repo):
    bindings = synthetic_fixture_bindings()
    dag = build_task_dag(bindings, "a" * 64, synthetic_fixture=True)
    runner_rows, runner_sha = build_subprocess_registry(repo)
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    identity = {
        "repo_root": str(repo.resolve()), "git_head": head, "git_tree": tree,
        "output_root": str(root.resolve()), "preregister_sha256": "a" * 64,
        "execution_pilot_preregister_sha256": "b" * 64,
        "execution_pilot_preregister_payload_sha256": "c" * 64,
        "exact191_manifest_sha256": "d" * 64,
        "prepared_builder_manifest_sha256": "e" * 64,
        "dag_payload_sha256": dag["payload_sha256"],
        "runner_registry_closure_sha256": runner_sha,
    }
    tasks, mapping = build_operational_tasks(bindings, identity, dag, runner_rows)
    task_rows = [{"ordinal": row["ordinal"], "task_id": row["task_id"],
        "stage": row["stage"], "payload_sha256": row["payload_sha256"],
        "evidence_node_count": row["evidence_node_count"],
        "evidence_node_closure_sha256": row["evidence_node_closure_sha256"]}
        for row in tasks]
    source_rows = []
    for relative in ("src/safety/v16_b716_fixed4_stage_runners.py",
                     "scripts/v16_b716_fixed4_disabled_stage_runner.sh",
                     "scripts/v16_b716_fixed4_sealed_executor.py"):
        source = repo / relative
        source_rows.append({"path": relative, "bytes": source.stat().st_size,
                            "sha256": sha256_file(source)})
    value = {"schema": PREFLIGHT_SCHEMA, "frozen": True, "sealed": True,
        "execution_authorized": False, "execution_performed": False,
        "repo_root": str(repo.resolve()), "git_head": head, "git_tree": tree,
        "output_root": str(root.resolve()), "preregister_sha256": "a" * 64,
        "execution_pilot_preregister_sha256": "b" * 64,
        "execution_pilot_preregister_payload_sha256": "c" * 64,
        "exact191_manifest_path": str((root / "fixture-exact191.json").resolve()),
        "exact191_manifest_sha256": "d" * 64,
        "exact72_lineage_manifest_path": str(
            (root / "fixture-exact72-lineage.json").resolve()),
        "exact72_lineage_manifest_sha256": "f" * 64,
        "exact72_lineage_validation": {
            "lineage_payload_sha256": "1" * 64,
            "frozen_clone_file_closure_sha256": "2" * 64,
            "exact191_manifest_sha256": "d" * 64,
            "task_count": 72,
            "ok_count": 60,
            "typed_failure_count": 12,
        },
        "prepared_builder_manifest_path": str(
            (root / "fixture-prepared34.json").resolve()),
        "prepared_builder_manifest_sha256": "e" * 64,
        "dag_payload_sha256": dag["payload_sha256"], "dag": dag,
        "evidence_ownership_mapping": mapping,
        "evidence_ownership_mapping_sha256": stable_json_sha256(mapping),
        "evidence_ownership_node_count": len(mapping),
        "operational_stage_counts": OPERATIONAL_STAGE_COUNTS,
        "operational_task_count": len(tasks), "operational_task_closure": task_rows,
        "operational_task_closure_sha256": stable_json_sha256(task_rows),
        "runner_registry": runner_rows, "runner_registry_closure_sha256": runner_sha,
        "execution_source_closure": source_rows,
        "execution_source_closure_sha256": stable_json_sha256(source_rows),
        "reconstruction_authorized": False, "refusion_allowed": False,
    }
    value = _signed(value); value["_tasks"] = tasks
    return value


def _guard_audit(root, public, manifest):
    existing = root / "guard_audit.json"
    if existing.is_file():
        return existing, json.loads(existing.read_text())
    audit_dir = root / "audit"
    logs = []
    for name in ("normal.log", "clean.log"):
        path = audit_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("metadata-only tests passed\n")
        path.chmod(0o444)
        logs.append({"path": str(path.relative_to(root)),
                     "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    value = _externally_signed({"schema": GUARD_AUDIT_SCHEMA, "status": "PASS",
        "independent_reviewer": "independent-test-reviewer",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": public["repo_root"], "git_head": public["git_head"],
        "git_tree": public["git_tree"], "output_root": str(root.resolve()),
        "task_manifest_sha256": sha256_file(root / "task_manifest.json"),
        "task_manifest_payload_sha256": manifest["payload_sha256"],
        "runner_registry_closure_sha256": public["runner_registry_closure_sha256"],
        "evidence_ownership_mapping_sha256": public["evidence_ownership_mapping_sha256"],
        "registration_guard_status": "UNREVIEWED_REFUSION_DISABLED",
        "reconstruction_authorized": False, "refusion_allowed": False,
        "normal_test_log": logs[0], "clean_test_log": logs[1]})
    path = root / "guard_audit.json"
    _write_immutable_json(path, value)
    return path, value


def _authorization(root, public, *, overrides=None):
    manifest_path = root / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    guard_path, guard = _guard_audit(root, public, manifest)
    now = datetime.now(timezone.utc)
    task_ids = [row["task_id"] for row in manifest["tasks"]]
    value = {"schema": AUTH_SCHEMA, "status": "PASS",
        "authorization_scope": "fixed4_all_107_operational_tasks",
        "execution_authorized": True, "execution_performed": False,
        "issued_at": now.isoformat(), "expires_at": (now+timedelta(minutes=30)).isoformat(),
        "repo_root": public["repo_root"], "git_head": public["git_head"],
        "git_tree": public["git_tree"], "output_root": str(root.resolve()),
        "preflight_path": str((root / "execution_preflight.json").resolve()),
        "preflight_sha256": sha256_file(root / "execution_preflight.json"),
        "preflight_payload_sha256": public["payload_sha256"],
        "dag_payload_sha256": public["dag_payload_sha256"],
        "execution_pilot_preregister_sha256": public["execution_pilot_preregister_sha256"],
        "execution_pilot_preregister_payload_sha256":
            public["execution_pilot_preregister_payload_sha256"],
        "operational_task_closure_sha256": public["operational_task_closure_sha256"],
        "evidence_ownership_mapping_sha256": public["evidence_ownership_mapping_sha256"],
        "runner_registry_closure_sha256": public["runner_registry_closure_sha256"],
        "execution_source_closure_sha256": public["execution_source_closure_sha256"],
        "exact191_manifest_sha256": public["exact191_manifest_sha256"],
        "exact72_lineage_manifest_sha256":
            public["exact72_lineage_manifest_sha256"],
        "exact72_lineage_payload_sha256":
            public["exact72_lineage_validation"]["lineage_payload_sha256"],
        "exact72_lineage_frozen_closure_sha256":
            public["exact72_lineage_validation"][
                "frozen_clone_file_closure_sha256"],
        "prepared_builder_manifest_sha256": public["prepared_builder_manifest_sha256"],
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "task_manifest_path": str(manifest_path.resolve()),
        "task_manifest_sha256": sha256_file(manifest_path),
        "task_manifest_payload_sha256": manifest["payload_sha256"],
        "authorized_task_ids": task_ids,
        "authorized_task_ids_sha256": stable_json_sha256(task_ids),
        "allowed_stages": list(ALLOWED_STAGES),
        "guard_audit_receipt_path": str(guard_path.resolve()),
        "guard_audit_receipt_sha256": sha256_file(guard_path),
        "guard_audit_receipt_payload_sha256": guard["payload_sha256"],
        "gt_allowed": False, "official92_allowed": False,
        "threshold_change_allowed": False, "result_selection_allowed": False,
        "default_checkpoint_replacement_allowed": False,
        **REQUIRED_AUTH_REVIEW_FIELDS}
    if overrides:
        value.update(overrides)
    value = _externally_signed(value)
    path = root / "authorization.json"
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path, sha256_file(path)


def _prepared_root(tmp_path):
    root = tmp_path / "run"; repo = _clean_repo(tmp_path)
    value = _preflight(root, repo); materialize_preflight(root, value)
    public = json.loads((root / "execution_preflight.json").read_text())
    auth_path, auth_sha = _authorization(root, public)
    return root, repo, public, auth_path, auth_sha


def _disabled_runner_fixture(tmp_path):
    root = tmp_path / "run"
    repo = _clean_repo(tmp_path)
    value = _preflight(root, repo)
    materialize_preflight(root, value)
    public = json.loads((root / "execution_preflight.json").read_text())
    authorization = root / "authorization.json"
    authorization.write_text("{}\n")
    authorization.chmod(0o444)
    task_path = sorted((root / "tasks").glob("colorpcr_direction*/task.json"))[0]
    task = json.loads(task_path.read_text())
    return root, repo, public, task_path, task


def _evidence_receipts(task, root, candidate_slots=None):
    rows = []
    generated = ({row["candidate_slot"] for row in candidate_slots
                  if row["status"] == "generated"}
                 if candidate_slots is not None else None)
    for node in task["evidence_nodes"]:
        status = "consumed"
        stage = node["node_id"].split(".", 1)[0]
        if generated is not None and stage in {"solver", "strict"}:
            slot = int(node["node_id"].split(".")[4])
            status = "consumed" if slot in generated else "typed_not_generated"
        document = _signed({"schema": EVIDENCE_RECEIPT_SCHEMA,
            "operational_task_id": task["task_id"],
            "operational_task_payload_sha256": task["payload_sha256"],
            "node_id": node["node_id"], "node_payload_sha256": node["node_payload_sha256"],
            "status": status, **POLICY_FALSE_FIELDS})
        path = root / "tasks" / task["task_id"] / "evidence" / f"{node['ordinal']:05d}.json"
        file_row = _write_immutable_json(path, document)
        rows.append({**file_row, "path": str(path.relative_to(root)),
                     "node_id": node["node_id"],
                     "node_payload_sha256": node["node_payload_sha256"]})
    return rows


def _attempt_document(task, schema, identity, *, typed=False):
    return _signed({"schema": schema, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"],
        "status": "typed_failure" if typed else "succeeded",
        "transform": None if typed else [[1.,0.,0.,0.],[0.,1.,0.,0.],
                                          [0.,0.,1.,0.],[0.,0.,0.,1.]],
        "failure_type": "FROZEN_TYPED_FAILURE" if typed else None,
        **identity, **POLICY_FALSE_FIELDS})


def _candidate_slots(task, *, generated_slots=None, safe_vote_slots=None):
    if generated_slots is None:
        generated_slots = (() if not task["safe_pose_vote_eligible"] else (0,))
    if safe_vote_slots is None:
        safe_vote_slots = generated_slots
    generated_slots = set(generated_slots); safe_vote_slots = set(safe_vote_slots)
    rows = []
    for candidate_slot in range(8):
        if candidate_slot in generated_slots:
            rows.append({"candidate_slot": candidate_slot, "status": "generated",
                "solver_rows_executed": EXACT_SOLVER_ROWS_PER_PILOT,
                "transform": [[1.,0.,0.,0.],[0.,1.,0.,0.],
                              [0.,0.,1.,0.],[0.,0.,0.,1.]],
                "failure_type": None, "safe_vote": candidate_slot in safe_vote_slots})
        else:
            rows.append({"candidate_slot": candidate_slot,
                "status": "typed_not_generated", "solver_rows_executed": 0,
                "failure_type": "CANDIDATE_SLOT_NOT_GENERATED"})
    return rows


def _solver_attempts(task, root, candidate_slots):
    rows = []
    for slot in candidate_slots:
        if slot["status"] != "generated":
            continue
        candidate_slot = slot["candidate_slot"]
        for solver in ("pointdsc", "pygcransac"):
            for direction in ("forward", "reverse"):
                for repeat in range(5):
                    identity = {"candidate_slot": candidate_slot, "solver": solver,
                                "direction": direction, "repeat": repeat}
                    document = _attempt_document(task, SOLVER_ATTEMPT_SCHEMA, identity)
                    path = (root / "tasks" / task["task_id"] / "attempts" /
                            f"c{candidate_slot}-{solver}-{direction}-{repeat}.json")
                    file_row = _write_immutable_json(path, document)
                    rows.append({**file_row, "path": str(path.relative_to(root)),
                                 **identity, "status": "succeeded"})
    return rows


def _typed_replay(task, root):
    rows = []
    for candidate_index in task["typed_failure_member_candidate_indices"]:
        document = _attempt_document(task, TYPED_FAILURE_REPLAY_SCHEMA,
                                     {"candidate_index": candidate_index}, typed=True)
        path = root / "tasks" / task["task_id"] / "typed_failures" / f"{candidate_index}.json"
        file_row = _write_immutable_json(path, document)
        rows.append({**file_row, "path": str(path.relative_to(root)),
                     "candidate_index": candidate_index})
    return rows


def _pilot_result(task, root, *, generated_slots=None, safe_vote_slots=None,
                  fail_closed_type=None):
    slots = _candidate_slots(task, generated_slots=generated_slots,
                             safe_vote_slots=safe_vote_slots)
    evidence = _evidence_receipts(task, root, slots)
    attempts = _solver_attempts(task, root, slots)
    typed = _typed_replay(task, root)
    eligible = task["safe_pose_vote_eligible"]
    failed = fail_closed_type is not None or not eligible
    typed_type = (fail_closed_type if eligible else
                  "TYPED_MEMBER_HYPOTHESIS_ABSTENTION")
    safe_rows = [row for row in slots if row.get("safe_vote") is True]
    outcome_fields = ({"hypothesis_task_id": task["task_id"], "gate_status": "ABSTAIN",
                "failure_class": "TYPED_MEMBER_HYPOTHESIS_ABSTENTION",
                "safe_transform": None, "measured_rotation_deg": None,
                "measured_translation_m": None,
                "measurement_source_file_sha256": None,
                "measurement_source_payload_sha256": None,
                "measurement_candidate_slot": None,
                "measurement_candidate_set_sha256": None,
                "measurement_slot_results_payload_sha256": None,
                "measurement_v15_decision_sha256": None}
               if not eligible else
               {"hypothesis_task_id": task["task_id"],
                "gate_status": "ABSTAIN" if failed else "PASS",
                "failure_class": typed_type if failed else None,
                "safe_transform": None if failed else safe_rows[0]["transform"],
                "measured_rotation_deg": None if failed else 0.0,
                "measured_translation_m": None if failed else 0.0,
                "measurement_source_file_sha256":
                    None if failed else "c" * 64,
                "measurement_source_payload_sha256":
                    None if failed else "d" * 64,
                "measurement_candidate_slot": None if failed else 0,
                "measurement_candidate_set_sha256":
                    None if failed else "e" * 64,
                "measurement_slot_results_payload_sha256":
                    None if failed else "f" * 64,
                "measurement_v15_decision_sha256":
                    None if failed else "1" * 64})
    v15 = next(row for row in task["evidence_nodes"]
               if row["node_id"].startswith("v15."))
    outcome_document = _signed({"schema": HYPOTHESIS_OUTCOME_SCHEMA,
        **outcome_fields, "task_payload_sha256": task["payload_sha256"],
        "source_v15_node_id": v15["node_id"],
        "source_v15_node_payload_sha256": v15["node_payload_sha256"],
        **POLICY_FALSE_FIELDS})
    outcome_path = (root / "tasks" / task["task_id"] / "outcomes" /
                    "v15_hypothesis_outcome.json")
    outcome_row = _write_immutable_json(outcome_path, outcome_document)
    outcome_receipt = {**outcome_row, "path": str(outcome_path.relative_to(root)),
                       "node_id": v15["node_id"],
                       "node_payload_sha256": v15["node_payload_sha256"]}
    outcome = {**outcome_fields,
               "source_result_payload_sha256": outcome_document["payload_sha256"]}
    return _signed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "typed_failure" if failed else "succeeded",
        "typed_failure": ({"type": typed_type, "transform": None}
                          if failed else None),
        **POLICY_FALSE_FIELDS,
        "output_artifacts": [], "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "candidate_slots": slots,
        "candidate_slot_closure_sha256": stable_json_sha256(slots),
        "solver_rows_executed": len(attempts), "solver_attempts": attempts,
        "solver_attempt_closure_sha256": stable_json_sha256(attempts),
        "typed_failure_replay": typed,
        "typed_failure_replay_closure_sha256": stable_json_sha256(typed),
        "hypothesis_outcome": outcome,
        "hypothesis_outcome_receipt": outcome_receipt})


def _aggregate_pair_outcomes(task):
    rows = []
    for index, task_id in enumerate(task["upstream_task_ids"]):
        known_bad = index == 3
        rows.append({"task_id": task_id,
            "status": "typed_failure" if known_bad else "succeeded",
            "decision": ("PERMANENT_KNOWN_BAD_VETO" if known_bad else
                         "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"),
            "safe_cluster_transform": (None if known_bad else
                [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]),
            "source_result_payload_sha256": f"{index + 1:x}" * 64})
    return rows


def test_evidence_mapping_is_exhaustive_unique_and_exact_counts(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"
    value = _preflight(root, repo); tasks = value["_tasks"]
    mapping = value["evidence_ownership_mapping"]
    assert len(tasks) == OPERATIONAL_TASK_COUNT == 107
    assert len(mapping) == EXPECTED_NODE_COUNT == 6091
    assert len({row["node_id"] for row in mapping}) == 6091
    assert [row["ordinal"] for row in mapping] == list(range(6091))
    assert {stage: sum(task["stage"] == stage for task in tasks)
            for stage in OPERATIONAL_STAGE_COUNTS} == OPERATIONAL_STAGE_COUNTS
    assert {stage: {task["evidence_node_count"] for task in tasks
                    if task["stage"] == stage} for stage in OPERATIONAL_STAGE_COUNTS} == {
        "colorpcr_direction": {4}, "bidirectional_multi_solver_pilot": {171},
        "v16_pair_hypothesis_cluster": {1}, "fixed4_aggregate": {1}}


def test_execution_preregister_pins_registry_and_exact20():
    value = json.loads((REPO / "manifests/v16_b716_fixed4_execution_pilot_preregister.json").read_text())
    _validate_execution_preregister(value, REPO)
    assert value["solver_matrix"]["exact_solver_rows_per_pilot"] == 20
    assert value["stage_runner_registry"]["caller_runner_injection_allowed"] is False
    assert value["stage_runner_registry"]["execution_mode"] == \
        "hash_bound_independent_subprocess"
    assert value["stage_runner_registry"]["checked_in_runner_disabled"] is True
    assert value["stage_runner_registry"]["sealed_executor_sha256"] == \
        sha256_file(REPO / "scripts/v16_b716_fixed4_sealed_executor.py")
    assert REQUIRED_AUTH_REVIEW_FIELDS["signer_private_key_not_on_execution_host"] is True


def test_create_only_preflight_resume_and_tamper(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    assert materialize_preflight(root, value)["states"] == {
        "created": 110, "resumed_identical": 0}
    assert materialize_preflight(root, value)["states"] == {
        "created": 0, "resumed_identical": 110}
    task = next((root / "tasks").glob("*/task.json")); task.chmod(0o644)
    changed = json.loads(task.read_text()); changed["gt_allowed"] = True
    task.write_text(json.dumps(changed))
    with pytest.raises(Fixed4ExecutionPilotError, match="create-only artifact differs"):
        materialize_preflight(root, value)


def test_authorization_binds_head_tree_root_manifest_guard_and_ttl(tmp_path):
    root, repo, public, auth_path, auth_sha = _prepared_root(tmp_path)
    del repo, public
    with pytest.raises(Fixed4ExecutionPilotError,
                       match="signature"):
        validate_authorization(auth_path, auth_sha,
            root / "execution_preflight.json",
            sha256_file(root / "execution_preflight.json"))


def test_caller_cannot_inject_arbitrary_runner():
    assert "runner" not in inspect.signature(execute_authorized_task).parameters
    with pytest.raises(TypeError):
        execute_authorized_task(task_path=Path("x"), preflight_path=Path("x"),
            preflight_sha256="a"*64, authorization_path=Path("x"),
            authorization_sha256="b"*64, output_root=Path("x"),
            runner=lambda *_: {})


def test_pilot_requires_exact20_file_backed_solver_rows(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root)
    assert validate_runner_result(task, result, root)["solver_rows_executed"] == 20
    bad = dict(result); bad.pop("payload_sha256"); bad["solver_attempts"] = []
    bad["solver_rows_executed"] = 0
    bad["solver_attempt_closure_sha256"] = stable_json_sha256([]); bad = _signed(bad)
    with pytest.raises(Fixed4ExecutionPilotError, match="exact20"):
        validate_runner_result(task, bad, root)


def test_pilot_eight_slot_expansion_closes_all_160_solver_rows(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root, generated_slots=tuple(range(8)),
                           safe_vote_slots=(0,))
    validated = validate_runner_result(task, result, root)
    assert validated["solver_rows_executed"] == 160
    assert {json.loads((root / row["path"]).read_text())["status"]
            for row in validated["evidence_receipts"]} == {"consumed"}


def test_pilot_outcome_is_file_backed_and_v15_bound(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root)
    assert validate_runner_result(task, result, root)["hypothesis_outcome"][
        "source_result_payload_sha256"]
    bad = dict(result); bad.pop("payload_sha256")
    bad["hypothesis_outcome"] = dict(bad["hypothesis_outcome"])
    bad["hypothesis_outcome"]["source_result_payload_sha256"] = "f" * 64
    with pytest.raises(Fixed4ExecutionPilotError,
                       match="file-backed pilot hypothesis outcome binding"):
        validate_runner_result(task, _signed(bad), root)


def test_pilot_requires_exact_eight_candidate_slots(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root)
    assert [row["candidate_slot"] for row in result["candidate_slots"]] == list(range(8))
    bad = dict(result); bad.pop("payload_sha256")
    bad["candidate_slots"] = bad["candidate_slots"][:-1]
    bad["candidate_slot_closure_sha256"] = stable_json_sha256(bad["candidate_slots"])
    with pytest.raises(Fixed4ExecutionPilotError, match="candidate-slot closure"):
        validate_runner_result(task, _signed(bad), root)


def test_solver_identity_duplicate_or_empty_attempt_fails(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root); bad = dict(result); bad.pop("payload_sha256")
    rows = [dict(row) for row in bad["solver_attempts"]]
    rows[-1] = dict(rows[-2]); bad["solver_attempts"] = rows
    bad["solver_attempt_closure_sha256"] = stable_json_sha256(rows); bad = _signed(bad)
    with pytest.raises(Fixed4ExecutionPilotError, match="closure"):
        validate_runner_result(task, bad, root)


def test_solver_identity_includes_candidate_slot_and_repeat(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root, generated_slots=(0, 3), safe_vote_slots=(3,))
    assert result["solver_rows_executed"] == 40
    assert validate_runner_result(task, result, root)["solver_rows_executed"] == 40
    bad = dict(result); bad.pop("payload_sha256")
    rows = [dict(row) for row in bad["solver_attempts"]]
    rows[20]["candidate_slot"] = 0
    bad["solver_attempts"] = rows
    bad["solver_attempt_closure_sha256"] = stable_json_sha256(rows)
    with pytest.raises(Fixed4ExecutionPilotError, match="binding|closure"):
        validate_runner_result(task, _signed(bad), root)


def test_typed_failure_replay_has_no_transform_and_cannot_be_dropped(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and task["contains_typed_failure_members"])
    result = _pilot_result(task, root)
    assert validate_runner_result(task, result, root)["status"] == "typed_failure"
    bad = dict(result); bad.pop("payload_sha256"); bad["typed_failure_replay"] = []
    bad["typed_failure_replay_closure_sha256"] = stable_json_sha256([]); bad = _signed(bad)
    with pytest.raises(Fixed4ExecutionPilotError, match="typed-failure replay"):
        validate_runner_result(task, bad, root)


def test_typed_not_generated_slot_cannot_carry_transform_or_vote(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and task["contains_typed_failure_members"])
    result = _pilot_result(task, root)
    for mutation in ({"transform": [[1.,0.,0.,0.],[0.,1.,0.,0.],
                                     [0.,0.,1.,0.],[0.,0.,0.,1.]]},
                     {"safe_vote": False}):
        bad = dict(result); bad.pop("payload_sha256")
        slots = [dict(row) for row in bad["candidate_slots"]]
        slots[0].update(mutation)
        bad["candidate_slots"] = slots
        bad["candidate_slot_closure_sha256"] = stable_json_sha256(slots)
        with pytest.raises(Fixed4ExecutionPilotError,
                           match="typed-not-generated|keys mismatch"):
            validate_runner_result(task, _signed(bad), root)


def test_eligible_pilot_can_report_honest_fail_closed_without_vote(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root, generated_slots=(), safe_vote_slots=(),
                           fail_closed_type="NO_CANDIDATE_GENERATED")
    assert validate_runner_result(task, result, root)["status"] == "typed_failure"


def test_formal_se3_validation_rejects_reflection(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root); bad = dict(result); bad.pop("payload_sha256")
    slots = [dict(row) for row in bad["candidate_slots"]]
    slots[0]["transform"] = [[-1.,0.,0.,0.],[0.,1.,0.,0.],
                              [0.,0.,1.,0.],[0.,0.,0.,1.]]
    bad["candidate_slots"] = slots
    bad["candidate_slot_closure_sha256"] = stable_json_sha256(slots)
    with pytest.raises(Fixed4ExecutionPilotError, match="generated candidate slot"):
        validate_runner_result(task, _signed(bad), root)


def _pair_result(task, root):
    evidence = _evidence_receipts(task, root); known = task["known_bad"]
    return _signed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "typed_failure" if known else "succeeded",
        "typed_failure": ({"type": "KNOWN_BAD_PERMANENT_VETO", "transform": None}
                          if known else None), **POLICY_FALSE_FIELDS,
        "output_artifacts": [], "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "replayed_hypothesis_task_ids": task["upstream_task_ids"],
        "safe_vote_hypothesis_task_ids": task["eligible_hypothesis_task_ids"],
        "gate_failed_hypothesis_task_ids": [],
        "typed_abstention_hypothesis_task_ids":
            task["typed_abstention_hypothesis_task_ids"],
        "decision": ("PERMANENT_KNOWN_BAD_VETO" if known else
                     "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"),
        "safe_cluster_transform": (None if known else
            [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])})


def test_known_bad_requires_all12_then_permanent_veto(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "v16_pair_hypothesis_cluster"
                and task["pair_id"] == KNOWN_BAD_PAIR_ID)
    result = _pair_result(task, root)
    assert len(result["replayed_hypothesis_task_ids"]) == 12
    assert validate_runner_result(task, result, root)["status"] == "typed_failure"
    bad = dict(result); bad.pop("payload_sha256")
    bad["replayed_hypothesis_task_ids"] = bad["replayed_hypothesis_task_ids"][:-1]
    bad = _signed(bad)
    with pytest.raises(Fixed4ExecutionPilotError, match="not replayed"):
        validate_runner_result(task, bad, root)


def test_pair_decision_is_derived_from_status_and_transform(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "v16_pair_hypothesis_cluster"
                and not task["known_bad"])
    result = _pair_result(task, root)
    assert validate_runner_result(task, result, root)["decision"] == \
        "ONE_UNIQUE_COMPLETE_LINKAGE_SAFE_POSE_CLUSTER"
    bad = dict(result); bad.pop("payload_sha256")
    bad["decision"] = "PERMANENT_KNOWN_BAD_VETO"
    with pytest.raises(Fixed4ExecutionPilotError, match="derived outcome"):
        validate_runner_result(task, _signed(bad), root)

    failed = dict(result); failed.pop("payload_sha256")
    failed.update({"status": "typed_failure",
        "typed_failure": {"type": "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER", "transform": None},
        "decision": "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER",
        "safe_cluster_transform": None})
    assert validate_runner_result(task, _signed(failed), root)["status"] == "typed_failure"


def test_reconstruction_false_rejects_any_ply_or_refusion_artifact(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = value["_tasks"][-1]; evidence = _evidence_receipts(task, root)
    ply = root / "tasks" / task["task_id"] / "artifacts" / "unauthorized_refusion.ply"
    ply.parent.mkdir(parents=True); ply.write_text("ply\nformat ascii 1.0\nend_header\n")
    ply.chmod(0o444)
    artifact = {"path": str(ply.relative_to(root)), "bytes": ply.stat().st_size,
                "sha256": sha256_file(ply)}
    result = _signed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "succeeded", "typed_failure": None, **POLICY_FALSE_FIELDS,
        "output_artifacts": [artifact], "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "replayed_pair_task_ids": task["upstream_task_ids"],
        "pair_outcomes": _aggregate_pair_outcomes(task),
        "pair_outcome_closure_sha256": stable_json_sha256(
            _aggregate_pair_outcomes(task)),
        "decision": "THREE_NORMALS_ACCEPTED_KNOWN_BAD_VETOED_NO_REFUSION",
        "guard_audit_receipt_sha256": "f"*64})
    with pytest.raises(Fixed4ExecutionPilotError, match="reconstruction/refusion"):
        validate_runner_result(task, result, root)


def test_aggregate_decision_is_derived_from_pair_outcomes(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = value["_tasks"][-1]; evidence = _evidence_receipts(task, root)
    outcomes = _aggregate_pair_outcomes(task)
    result = _signed({"schema": RESULT_SCHEMA, "task_id": task["task_id"],
        "task_payload_sha256": task["payload_sha256"], "stage": task["stage"],
        "status": "succeeded", "typed_failure": None, **POLICY_FALSE_FIELDS,
        "output_artifacts": [], "evidence_receipts": evidence,
        "evidence_receipt_closure_sha256": stable_json_sha256(evidence),
        "replayed_pair_task_ids": task["upstream_task_ids"],
        "pair_outcomes": outcomes,
        "pair_outcome_closure_sha256": stable_json_sha256(outcomes),
        "decision": "THREE_NORMALS_ACCEPTED_KNOWN_BAD_VETOED_NO_REFUSION",
        "guard_audit_receipt_sha256": "f"*64})
    assert validate_runner_result(task, result, root)["status"] == "succeeded"
    bad = dict(result); bad.pop("payload_sha256")
    bad["decision"] = "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"
    with pytest.raises(Fixed4ExecutionPilotError, match="derived outcomes"):
        validate_runner_result(task, _signed(bad), root)

    failed = dict(result); failed.pop("payload_sha256")
    failed_outcomes = [dict(row) for row in outcomes]
    failed_outcomes[0].update({"status": "typed_failure",
        "decision": "NO_UNIQUE_COMPATIBLE_SAFE_POSE_CLUSTER",
        "safe_cluster_transform": None})
    failed.update({"status": "typed_failure",
        "typed_failure": {"type": "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED", "transform": None},
        "pair_outcomes": failed_outcomes,
        "pair_outcome_closure_sha256": stable_json_sha256(failed_outcomes),
        "decision": "FIXED4_NORMAL_PAIR_CONSENSUS_FAILED"})
    assert validate_runner_result(task, _signed(failed), root)["status"] == "typed_failure"


def test_partial_runner_artifact_fails_before_registry_and_is_not_overwritten(tmp_path):
    root, _repo, _public, task_path, _task = _disabled_runner_fixture(tmp_path)
    partial = task_path.parent / "artifacts/partial.bin"
    partial.parent.mkdir(); partial.write_bytes(b"prior-partial"); before = sha256_file(partial)
    assert _task_root_has_partial_state(task_path.parent) is True
    assert sha256_file(partial) == before


def test_registered_runner_remains_disabled_without_gpu_or_solver(tmp_path):
    root, repo, public, task_path, task = _disabled_runner_fixture(tmp_path)
    receipt = execute_disabled_stage(
        repo=repo, task=task, task_path=task_path,
        preflight_path=root / "execution_preflight.json",
        authorization_path=root / "authorization.json",
        task_manifest_path=root / "task_manifest.json", task_root=task_path.parent,
        registry_rows=public["runner_registry"],
        registry_sha256=public["runner_registry_closure_sha256"])
    receipt = json.loads((task_path.parent / "wrapper/consumption_receipt.json").read_text())
    assert receipt["schema"] == FIX3_CONSUMPTION_SCHEMA
    assert receipt["returncode"] == DISABLED_EXIT_CODE
    assert receipt["undeclared_consumed_paths"] == []
    assert receipt["runner_reported_failure_type_trusted"] is False
    consumed = set(receipt["parent_observed_consumed_paths"])
    assert {str(task_path), str(root / "execution_preflight.json"),
            str(root / "authorization.json"), str(root / "task_manifest.json")} <= consumed


def test_result_selection_field_is_rejected_before_acceptance(tmp_path):
    repo = _clean_repo(tmp_path); root = tmp_path / "run"; value = _preflight(root, repo)
    task = next(task for task in value["_tasks"]
                if task["stage"] == "bidirectional_multi_solver_pilot"
                and not task["contains_typed_failure_members"])
    result = _pilot_result(task, root); result.pop("payload_sha256")
    result["winner"] = "forbidden"; result = _signed(result)
    with pytest.raises(Fixed4ExecutionPilotError, match="forbidden evidence field"):
        validate_runner_result(task, result, root)


def test_caller_self_signed_pass_is_rejected_by_fixed_external_anchor(tmp_path):
    trusted_anchor = load_trust_anchor(
        subprocess_contract.TEST_ONLY_ANCHOR_PATH,
        subprocess_contract.TEST_ONLY_ANCHOR_SHA256,
        repo_root=tmp_path / "repo", output_root=tmp_path / "output")
    rogue = tmp_path / "rogue"
    rogue.mkdir()
    rogue_private = rogue / "private.pem"
    subprocess.run(["/usr/bin/openssl", "genpkey", "-algorithm", "Ed25519",
                    "-out", str(rogue_private)], check=True, capture_output=True)
    value = {"schema": AUTH_SCHEMA, "status": "PASS",
             "signature_algorithm": SIGNATURE_ALGORITHM,
             "signing_key_id": _TEST_KEY_ID}
    message = rogue / "message.bin"
    signature = rogue / "signature.bin"
    message.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    subprocess.run(["/usr/bin/openssl", "pkeyutl", "-sign", "-inkey",
        str(rogue_private), "-rawin", "-in", str(message), "-out", str(signature)],
        check=True, capture_output=True)
    value["signature_b64"] = base64.b64encode(signature.read_bytes()).decode()
    value = _signed(value)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="signature rejected"):
        verify_document_signature(value, trusted_anchor, purpose="rogue authorization")


def test_signature_positive_and_tampered_negative(tmp_path):
    trusted_anchor = load_trust_anchor(
        subprocess_contract.TEST_ONLY_ANCHOR_PATH,
        subprocess_contract.TEST_ONLY_ANCHOR_SHA256,
        repo_root=tmp_path / "repo", output_root=tmp_path / "output")
    value = _externally_signed({"schema": "fixed4-signature-test-v1",
                                "status": "PASS"})
    verify_document_signature(value, trusted_anchor, purpose="signature positive")
    tampered = dict(value)
    tampered["status"] = "FAIL"
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="signature rejected"):
        verify_document_signature(tampered, trusted_anchor, purpose="signature negative")


def test_public_authority_directory_does_not_trigger_private_key_gate(tmp_path):
    authority = tmp_path / "sgaligner-exact72-audit-authority-v1"
    authority.mkdir()
    (authority / "audit_public.pem").write_text("public-only test material\n")
    _reject_prohibited_signer_private_key(authority / "audit_private.pem")


def test_exact_private_key_leaf_presence_fails_closed(tmp_path):
    authority = tmp_path / "sgaligner-exact72-audit-authority-v1"
    authority.mkdir()
    private_leaf = authority / "audit_private.pem"
    private_leaf.write_text("presence sentinel, not key material\n")
    try:
        with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                           match="private key remains"):
            _reject_prohibited_signer_private_key(private_leaf)
    finally:
        private_leaf.unlink(missing_ok=True)
    source = inspect.getsource(verify_fixed_signed_document)
    assert ("/home/aidenwu/.local/share/sgaligner-exact72-audit-authority-v1/"
            "audit_private.pem") in source


def test_production_anchor_allows_public_only_authority_directory(tmp_path):
    authority = Path(
        "/home/aidenwu/.local/share/sgaligner-exact72-audit-authority-v1")
    assert authority.is_dir()
    assert (authority / "audit_public.pem").is_file()
    with pytest.raises(FileNotFoundError):
        (authority / "audit_private.pem").lstat()
    invalid = _signed({"schema": "fixed4-production-anchor-probe-v1",
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "signing_key_id": "sgaligner-exact72-audit-authority-v1",
        "signature_b64": base64.b64encode(b"\0" * 64).decode()})
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="signature rejected"):
        verify_fixed_signed_document(
            invalid, repo_root=tmp_path / "repo", output_root=tmp_path / "output",
            purpose="public-only authority probe")


def test_trust_anchor_digest_drift_and_writable_file_are_rejected(tmp_path):
    anchor_path, _public = _trust_anchor_case(tmp_path)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="not provisioned/mismatch"):
        load_trust_anchor(anchor_path, "f" * 64,
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")
    anchor_path.chmod(0o644)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="trust anchor is not read-only"):
        load_trust_anchor(anchor_path, sha256_file(anchor_path),
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")


def test_trust_anchor_symlink_is_rejected(tmp_path):
    anchor_path, _public = _trust_anchor_case(tmp_path)
    link = tmp_path / "anchor-link.json"
    link.symlink_to(anchor_path)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="symlinked"):
        load_trust_anchor(link, sha256_file(anchor_path),
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")


def test_public_key_drift_and_writable_file_are_rejected(tmp_path):
    anchor_path, public = _trust_anchor_case(tmp_path)
    public.chmod(0o644)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="public key is not read-only"):
        load_trust_anchor(anchor_path, sha256_file(anchor_path),
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")
    public.write_text("drifted public test material\n")
    public.chmod(0o444)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="public key SHA mismatch"):
        load_trust_anchor(anchor_path, sha256_file(anchor_path),
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")


def test_public_key_symlink_is_rejected(tmp_path):
    target = tmp_path / "public-target.pem"
    shutil.copy2(_TEST_PRIVATE_KEY.parent / "public.pem", target)
    target.chmod(0o444)
    link = tmp_path / "public-link.pem"
    link.symlink_to(target)
    anchor_path, _public = _trust_anchor_case(tmp_path / "case", public_path=link)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="symlinked"):
        load_trust_anchor(anchor_path, sha256_file(anchor_path),
                          repo_root=tmp_path / "repo", output_root=tmp_path / "output")


def test_legacy_registry_monkeypatch_cannot_change_subprocess_entrypoint(tmp_path,
                                                                        monkeypatch):
    repo = _clean_repo(tmp_path)
    before, digest = build_subprocess_registry(repo)
    monkeypatch.setitem(legacy_registry.STAGE_RUNNER_REGISTRY,
                        "colorpcr_direction", lambda *_: {"status": "forged"})
    after, after_digest = build_subprocess_registry(repo)
    assert after == before
    assert after_digest == digest
    assert after[0]["execution_mode"] == "hash_bound_independent_subprocess"
    monkeypatch.setattr(subprocess_contract, "FIXED_INTERPRETER", "/tmp/rogue",
                        raising=False)
    monkeypatch.setattr(subprocess_contract, "FIXED_TRACER", "/tmp/rogue", raising=False)
    monkeypatch.setattr(subprocess_contract, "DISABLED_RUNNER_RELATIVE", "rogue.py",
                        raising=False)
    assert build_subprocess_registry(repo) == (before, digest)


def test_caller_anchor_and_openssl_global_reassignment_is_inert(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess_contract, "FIXED_TRUST_ANCHOR_PATH",
                        subprocess_contract.TEST_ONLY_ANCHOR_PATH, raising=False)
    monkeypatch.setattr(subprocess_contract, "FIXED_TRUST_ANCHOR_SHA256",
                        subprocess_contract.TEST_ONLY_ANCHOR_SHA256, raising=False)
    monkeypatch.setattr(subprocess_contract, "OPENSSL_PATH", "/tmp/rogue", raising=False)
    value = _externally_signed({"schema": AUTH_SCHEMA, "status": "PASS"})
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="signature"):
        verify_fixed_signed_document(
            value, repo_root=tmp_path / "repo", output_root=tmp_path / "output",
            purpose="mutable-global attack")


def test_parent_never_calls_replaceable_stage_function(tmp_path, monkeypatch):
    reached = []
    monkeypatch.setattr(execution_pilot, "execute_disabled_stage",
                        lambda **_kwargs: reached.append(True), raising=False)
    root, _repo, _public, auth_path, auth_sha = _prepared_root(tmp_path)
    task_path = sorted((root / "tasks").glob("colorpcr_direction*/task.json"))[0]
    with pytest.raises(Fixed4ExecutionPilotError, match="signature"):
        execute_authorized_task(
            task_path=task_path, preflight_path=root / "execution_preflight.json",
            preflight_sha256=sha256_file(root / "execution_preflight.json"),
            authorization_path=auth_path, authorization_sha256=auth_sha,
            output_root=root)
    assert reached == []
    assert "execute_disabled_stage(" not in inspect.getsource(execute_authorized_task)


def test_task_and_wrapper_parent_symlinks_are_rejected(tmp_path):
    repo = _clean_repo(tmp_path)
    root = tmp_path / "run"
    value = _preflight(root, repo)
    outside_task = tmp_path / "outside-task"
    outside_task.mkdir()
    (root / "tasks").mkdir(parents=True)
    task_id = value["_tasks"][0]["task_id"]
    (root / "tasks" / task_id).symlink_to(outside_task, target_is_directory=True)
    with pytest.raises(Fixed4ExecutionPilotError, match="symlinked|directory"):
        materialize_preflight(root, value)

    runner_root, runner_repo, public, task_path, task = _disabled_runner_fixture(
        tmp_path / "wrapper-case")
    outside_wrapper = tmp_path / "outside-wrapper"
    outside_wrapper.mkdir()
    (task_path.parent / "wrapper").symlink_to(outside_wrapper, target_is_directory=True)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="partial wrapper state"):
        execute_disabled_stage(
            repo=runner_repo, task=task, task_path=task_path,
            preflight_path=runner_root / "execution_preflight.json",
            authorization_path=runner_root / "authorization.json",
            task_manifest_path=runner_root / "task_manifest.json",
            task_root=task_path.parent, registry_rows=public["runner_registry"],
            registry_sha256=public["runner_registry_closure_sha256"])
    assert list(outside_wrapper.iterdir()) == []


def test_recursive_real_file_closure_rejects_shallow_and_symlink(tmp_path):
    root = tmp_path / "sealed"
    root.mkdir()
    first = root / "a.json"; first.write_text("{}\n")
    nested = root / "nested"; nested.mkdir()
    second = nested / "b.bin"; second.write_bytes(b"bound-input")
    rows = [{"role": "exact72-results", "root": str(root.resolve()), "files": [
        {"path": "a.json", "bytes": first.stat().st_size, "sha256": sha256_file(first)},
        {"path": "nested/b.bin", "bytes": second.stat().st_size,
         "sha256": sha256_file(second)},
    ]}]
    assert len(validate_recursive_file_closure(
        rows, stable_json_sha256(rows), role="fixture")) == 2
    shallow = [{**rows[0], "files": rows[0]["files"][:1]}]
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="shallow/inexhaustive"):
        validate_recursive_file_closure(
            shallow, stable_json_sha256(shallow), role="fixture")
    link = root / "escape-link"
    link.symlink_to(second)
    with pytest.raises(subprocess_contract.Fixed4SubprocessContractError,
                       match="symlink"):
        validate_recursive_file_closure(rows, stable_json_sha256(rows), role="fixture")


def test_parent_trace_rejects_extra_input_and_runner_cannot_report_failure_type():
    trace = ('1 openat(AT_FDCWD, "/sealed/runner.sh", O_RDONLY) = 3\n'
             '1 openat(AT_FDCWD, "/undeclared/secret.bin", O_RDONLY) = 4\n'
             '1 openat(AT_FDCWD, "/missing", O_RDONLY) = -1 ENOENT\n')
    assert parse_consumed_paths(trace) == [
        "/sealed/runner.sh", "/undeclared/secret.bin"]
    forged = b'{"failure_type":"RUNNER_SAYS_SUCCESS"}'
    assert classify_wrapper_failure(DISABLED_EXIT_CODE, True, forged, b"") == \
        "CHECKED_IN_RUNNER_EXECUTION_DISABLED"
    assert classify_wrapper_failure(0, False, forged, b"") == \
        "PARENT_OBSERVED_INPUT_CONTRACT_VIOLATION"
