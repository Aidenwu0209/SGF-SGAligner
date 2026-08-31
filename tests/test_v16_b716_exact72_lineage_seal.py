import json
from pathlib import Path

import pytest

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
import safety.v16_b716_exact72_lineage_seal as lineage


SHA = "a" * 64
AUTHORIZATION_SHA = "b" * 64


def _write_payload_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value)
    value["payload_sha256"] = stable_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return sha256_file(path)


def _write_task_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(value)
    value["task_sha256"] = stable_json_sha256(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value["task_sha256"]


def _file_row(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _implementation_closure() -> dict:
    rows = []
    for relative in lineage.IMPLEMENTATION_PATHS:
        path = lineage.ROOT / relative
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "repo_root": str(lineage.ROOT),
        "git_head": "c" * 40,
        "git_tree": "d" * 40,
        "source_closure": rows,
        "source_closure_sha256": stable_json_sha256(rows),
    }


def _make_exact72_source(root: Path) -> tuple[Path, Path]:
    exact_root = root / "historical_exact72"
    tasks_root = exact_root / "tasks"
    tasks_root.mkdir(parents=True)

    inputs = []
    for index, role in enumerate(sorted(lineage.EXPECTED_INPUT_ROLES)):
        path = root / "upstream" / f"{index:02d}-{role}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": role, "index": index}) + "\n")
        inputs.append({"role": role, **_file_row(path)})

    result_rows = []
    for index in range(lineage.EXPECTED_TASKS):
        typed = index >= lineage.EXPECTED_OK
        status = lineage.TYPED_FAILURE if typed else "ok"
        short_id = f"candidate{index:03d}"
        node_pair = [index, index + 1]
        object_pair = [index + 1000, index + 1001]
        task_id = f"{short_id}__{node_pair[0]}_{node_pair[1]}"
        task_root = tasks_root / task_id

        semantic_task_sha = _write_task_json(task_root / "task.json", {
            "short_id": short_id,
            "candidate_index": index,
            "node_pair": node_pair,
            "object_pair": object_pair,
        })
        view_sha = _write_payload_json(
            task_root / "authorized_task_view.json", {
                "planned_task_sha256": semantic_task_sha,
                "short_id": short_id,
                "node_pair": node_pair,
            })
        attempt_sha = _write_payload_json(task_root / "attempt_receipt.json", {
            "task_sha256": semantic_task_sha,
            "authorization_sha256": AUTHORIZATION_SHA,
        })
        result_sha = _write_payload_json(task_root / "result.json", {
            "task_sha256": semantic_task_sha,
            "authorization_sha256": AUTHORIZATION_SHA,
            "short_id": short_id,
            "node_pair": node_pair,
            "object_pair": object_pair,
            "status": status,
        })
        correspondence_sha = None
        if not typed:
            correspondence = task_root / "correspondences.npz"
            correspondence.write_bytes(b"NPZ" + index.to_bytes(2, "big"))
            correspondence_sha = sha256_file(correspondence)
        result_rows.append({
            "short_id": short_id,
            "candidate_index": index,
            "node_pair": node_pair,
            "object_pair": object_pair,
            "status": status,
            "task_sha256": semantic_task_sha,
            "authorized_task_view_sha256": view_sha,
            "attempt_sha256": attempt_sha,
            "result_sha256": result_sha,
            "correspondence_sha256": correspondence_sha,
        })

    manifest = {
        "schema": lineage.EXACT191_SCHEMA,
        "sealed": True,
        "provenance_sealed": True,
        "candidate_count": 191,
        "new_authorized_count": lineage.EXPECTED_TASKS,
        "new_authorized_ok_count": lineage.EXPECTED_OK,
        "new_authorized_typed_failure_count":
            lineage.EXPECTED_TYPED_FAILURES,
        "new_result_closure": result_rows,
        "input_closure": inputs,
        "recursive_input_closure_sha256": stable_json_sha256(inputs),
        "execution_binding": {"authorization_sha256": AUTHORIZATION_SHA},
    }
    manifest_path = root / "exact191_manifest.json"
    _write_payload_json(manifest_path, manifest)
    return manifest_path, exact_root


def _build(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path]:
    manifest_path, exact_root = _make_exact72_source(root)
    manifest_sha = sha256_file(manifest_path)
    monkeypatch.setattr(
        lineage, "_implementation_closure", _implementation_closure)
    monkeypatch.setattr(lineage, "_verify_delivery", lambda **_kwargs: {
        "delivery_root": str(root / "signed-delivery"),
        "manifest_sha256": manifest_sha,
        "delivery_seal_sha256": "e" * 64,
        "delivery_signature_sha256": "f" * 64,
        "public_key_sha256": lineage.PUBLIC_KEY_SHA256,
        "recursive_file_closure_sha256": "1" * 64,
        "covered_file_count": 191,
        "signature_verified": True,
    })
    value = lineage.build_lineage_seal(
        exact191_manifest_path=manifest_path,
        exact191_manifest_sha256=manifest_sha,
        exact191_delivery_seal_path=root / "unused-delivery-seal.json",
        exact191_delivery_seal_sha256="e" * 64,
        exact191_delivery_signature_path=root / "unused-delivery-seal.sig",
        exact191_delivery_signature_sha256="f" * 64,
        exact72_root=exact_root,
    )
    return value, exact_root


@pytest.fixture
def case_root(tmp_path):
    yield tmp_path
    # materialize_lineage_seal intentionally freezes files/directories.  Thaw
    # the disposable fixture so pytest can remove its temporary directory.
    for path in sorted(tmp_path.rglob("*"),
                       key=lambda item: len(item.parts), reverse=True):
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except FileNotFoundError:
            pass


def test_exactly_60_ok_and_12_typed_have_the_required_npz_partition(
        case_root, monkeypatch):
    value, _exact_root = _build(case_root, monkeypatch)

    assert value["task_count"] == 72
    assert value["ok_count"] == 60
    assert value["typed_failure_count"] == 12
    assert sum("correspondences.npz" in row["files"]
               for row in value["task_closure"] if not row["typed_failure"]) == 60
    assert all("correspondences.npz" not in row["files"]
               for row in value["task_closure"] if row["typed_failure"])

    manifest_path, state = lineage.materialize_lineage_seal(
        case_root / "frozen", value)
    assert state == "created"
    verified = lineage.verify_lineage_seal(
        manifest_path, sha256_file(manifest_path))
    assert verified["task_count"] == 72
    assert verified["ok_count"] == 60
    assert verified["typed_failure_count"] == 12


def test_typed_failure_with_npz_is_rejected(case_root, monkeypatch):
    manifest_path, exact_root = _make_exact72_source(case_root)
    typed_root = sorted((exact_root / "tasks").iterdir())[60]
    (typed_root / "correspondences.npz").write_bytes(b"forbidden")
    monkeypatch.setattr(
        lineage, "_implementation_closure", _implementation_closure)
    monkeypatch.setattr(lineage, "_verify_delivery", lambda **_kwargs: {
        "manifest_sha256": sha256_file(manifest_path),
        "signature_verified": True,
    })

    with pytest.raises(lineage.Exact72LineageSealError,
                       match="task file closure mismatch"):
        lineage.build_lineage_seal(
            exact191_manifest_path=manifest_path,
            exact191_manifest_sha256=sha256_file(manifest_path),
            exact191_delivery_seal_path=case_root / "unused.json",
            exact191_delivery_seal_sha256="e" * 64,
            exact191_delivery_signature_path=case_root / "unused.sig",
            exact191_delivery_signature_sha256="f" * 64,
            exact72_root=exact_root,
        )


def test_ok_result_without_npz_is_rejected(case_root, monkeypatch):
    manifest_path, exact_root = _make_exact72_source(case_root)
    ok_root = sorted((exact_root / "tasks").iterdir())[0]
    (ok_root / "correspondences.npz").unlink()
    monkeypatch.setattr(
        lineage, "_implementation_closure", _implementation_closure)
    monkeypatch.setattr(lineage, "_verify_delivery", lambda **_kwargs: {
        "manifest_sha256": sha256_file(manifest_path),
        "signature_verified": True,
    })

    with pytest.raises(lineage.Exact72LineageSealError,
                       match="task file closure mismatch"):
        lineage.build_lineage_seal(
            exact191_manifest_path=manifest_path,
            exact191_manifest_sha256=sha256_file(manifest_path),
            exact191_delivery_seal_path=case_root / "unused.json",
            exact191_delivery_seal_sha256="e" * 64,
            exact191_delivery_signature_path=case_root / "unused.sig",
            exact191_delivery_signature_sha256="f" * 64,
            exact72_root=exact_root,
        )


def _materialized(case_root: Path, monkeypatch: pytest.MonkeyPatch):
    value, _exact_root = _build(case_root, monkeypatch)
    manifest_path, _state = lineage.materialize_lineage_seal(
        case_root / "frozen", value)
    return manifest_path, sha256_file(manifest_path)


def test_frozen_lineage_rejects_undeclared_closure_member(
        case_root, monkeypatch):
    manifest_path, manifest_sha = _materialized(case_root, monkeypatch)
    root = manifest_path.parent
    root.chmod(0o700)
    extra = root / "undeclared.bin"
    extra.write_bytes(b"not in frozen closure")
    extra.chmod(0o400)
    root.chmod(0o500)

    with pytest.raises(lineage.Exact72LineageSealError,
                       match="closure is not exhaustive"):
        lineage.verify_lineage_seal(manifest_path, manifest_sha)


def test_frozen_lineage_rejects_member_mode_drift(case_root, monkeypatch):
    manifest_path, manifest_sha = _materialized(case_root, monkeypatch)
    member = next(path for path in manifest_path.parent.rglob("*")
                  if path.is_file() and path != manifest_path)
    member.chmod(0o600)

    with pytest.raises(lineage.Exact72LineageSealError,
                       match="bytes/mode drift"):
        lineage.verify_lineage_seal(manifest_path, manifest_sha)


def test_frozen_lineage_accepts_relocated_identical_implementation(
        case_root, monkeypatch):
    value, _exact_root = _build(case_root, monkeypatch)
    value["implementation"] = dict(value["implementation"])
    value["implementation"]["repo_root"] = str(
        (case_root / "retired-identical-checkout").resolve())
    value["payload_sha256"] = stable_json_sha256({
        key: item for key, item in value.items() if key != "payload_sha256"})
    manifest_path, _state = lineage.materialize_lineage_seal(
        case_root / "frozen", value)

    verified = lineage.verify_lineage_seal(
        manifest_path, sha256_file(manifest_path))

    assert verified["task_count"] == lineage.EXPECTED_TASKS


def test_frozen_lineage_rejects_noncanonical_historical_repo_root(
        case_root, monkeypatch):
    value, _exact_root = _build(case_root, monkeypatch)
    value["implementation"] = dict(value["implementation"])
    value["implementation"]["repo_root"] = "relative/retired-checkout"
    value["payload_sha256"] = stable_json_sha256({
        key: item for key, item in value.items() if key != "payload_sha256"})
    manifest_path, _state = lineage.materialize_lineage_seal(
        case_root / "frozen", value)

    with pytest.raises(lineage.Exact72LineageSealError,
                       match="lineage upstream binding drift"):
        lineage.verify_lineage_seal(
            manifest_path, sha256_file(manifest_path))
