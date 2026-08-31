"""Create and verify a detached, CPU-only exact72 execution-lineage seal.

The corrected exact191 delivery intentionally seals the merged output tree, not
the historical exact72 task tree.  This module joins those two immutable
closures without rerunning a model: it verifies the signed exact191 delivery,
opens every historical task artifact, and records the precise absence of a
correspondence file for typed failures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256


LINEAGE_SCHEMA = "v16-b716-exact72-execution-lineage-seal-v1"
EXACT191_SCHEMA = "v16-b716-exact191-merged-manifest-v1"
DELIVERY_SEAL_SCHEMA = "v16-b716-exact191-detached-delivery-seal-v2"
PUBLIC_KEY_SHA256 = (
    "172a0387ec4953734fe5c799a31c7d7718be6b350868c06c0bb97fe0a685c022"
)
EXPECTED_TASKS = 72
EXPECTED_OK = 60
EXPECTED_TYPED_FAILURES = 12
EXPECTED_INPUTS = 5
TYPED_FAILURE = "insufficient_post_voxel_points"
ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_PATHS = (
    "src/safety/v16_b716_exact72_lineage_seal.py",
    "scripts/v16_b716_exact72_lineage_seal.py",
)
EXPECTED_INPUT_ROLES = {
    "frozen_candidate_manifest", "authorized_preflight_manifest",
    "authorized_preregistration", "execution_authorization",
    "exact72_batch_result",
}


class Exact72LineageSealError(RuntimeError):
    """A signed delivery, historical file, or semantic binding drifted."""


def _sha(value: Any, role: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise Exact72LineageSealError(f"invalid {role} SHA")
    return value


def _payload_valid(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    return value.get("payload_sha256") == stable_json_sha256(unsigned)


def _regular(path: Path, role: str) -> Path:
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Exact72LineageSealError(f"{role} missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise Exact72LineageSealError(f"{role} is not a regular no-symlink file")
    return path.resolve()


def _directory(path: Path, role: str) -> Path:
    path = Path(path)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise Exact72LineageSealError(f"{role} missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise Exact72LineageSealError(f"{role} is not a no-symlink directory")
    return path.resolve()


def _json(path: Path, expected_sha: str, role: str, *,
          self_hash_field: str = "payload_sha256") -> dict[str, Any]:
    path = _regular(path, role)
    before = sha256_file(path)
    if before != _sha(expected_sha, role):
        raise Exact72LineageSealError(f"{role} SHA mismatch")
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise Exact72LineageSealError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise Exact72LineageSealError(f"{role} changed or is not a JSON object")
    unsigned = {key: item for key, item in value.items()
                if key != self_hash_field}
    if value.get(self_hash_field) != stable_json_sha256(unsigned):
        raise Exact72LineageSealError(
            f"{role} {self_hash_field} is invalid")
    return value


def _file_row(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = _regular(path, "lineage member")
    if root is None:
        label = str(path)
    else:
        try:
            label = path.relative_to(Path(root).resolve()).as_posix()
        except ValueError as exc:
            raise Exact72LineageSealError("lineage member escapes root") from exc
    with path.open("rb") as stream:
        while stream.read(1024 * 1024):
            pass
    return {"path": label, "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _verify_delivery(
    *, manifest_path: Path, manifest_sha256: str,
    delivery_seal_path: Path, delivery_seal_sha256: str,
    delivery_signature_path: Path, delivery_signature_sha256: str,
) -> dict[str, Any]:
    manifest_path = _regular(manifest_path, "exact191 manifest")
    root = _directory(manifest_path.parent, "exact191 delivery root")
    if manifest_path != root / "exact191_manifest.json":
        raise Exact72LineageSealError("exact191 manifest has unexpected delivery path")
    seal_path = _regular(delivery_seal_path, "delivery seal")
    signature_path = _regular(delivery_signature_path, "delivery signature")
    if (seal_path != root / "delivery_seal.json"
            or signature_path != root / "delivery_seal.sig"):
        raise Exact72LineageSealError("delivery seal/signature path drift")
    seal = _json(seal_path, delivery_seal_sha256, "delivery seal")
    if sha256_file(signature_path) != _sha(
            delivery_signature_sha256, "delivery signature"):
        raise Exact72LineageSealError("delivery signature SHA mismatch")
    public_key = _regular(
        root / "exact191_cpu_merge_authority_v1_public.pem",
        "exact191 delivery public key")
    if sha256_file(public_key) != PUBLIC_KEY_SHA256:
        raise Exact72LineageSealError("exact191 delivery trust anchor drift")
    completed = subprocess.run([
        "/usr/bin/openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
        str(public_key), "-rawin", "-in", str(seal_path), "-sigfile",
        str(signature_path),
    ], check=False, capture_output=True)
    if completed.returncode != 0:
        raise Exact72LineageSealError("delivery seal signature rejected")
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise Exact72LineageSealError("delivery root contains symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise Exact72LineageSealError("delivery root contains non-regular entry")
        relative = path.relative_to(root).as_posix()
        if relative not in {"delivery_seal.json", "delivery_seal.sig"}:
            rows.append(_file_row(path, root=root))
    if (seal.get("schema") != DELIVERY_SEAL_SCHEMA
            or Path(str(seal.get("output_root", ""))).resolve() != root
            or seal.get("manifest_sha256") != manifest_sha256
            or seal.get("covered_file_closure") != rows
            or seal.get("recursive_file_closure_sha256")
            != stable_json_sha256(rows)
            or seal.get(
                "seal_and_signature_excluded_from_file_closure_to_avoid_self_reference")
            is not True):
        raise Exact72LineageSealError("signed delivery closure mismatch")
    return {
        "delivery_root": str(root),
        "manifest_sha256": manifest_sha256,
        "delivery_seal_sha256": delivery_seal_sha256,
        "delivery_signature_sha256": delivery_signature_sha256,
        "public_key_sha256": PUBLIC_KEY_SHA256,
        "recursive_file_closure_sha256": seal["recursive_file_closure_sha256"],
        "covered_file_count": len(rows),
        "signature_verified": True,
    }


def _validate_input_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if set(row) != {"role", "path", "bytes", "sha256"}:
        raise Exact72LineageSealError("exact72 input row keys mismatch")
    path = _regular(Path(str(row.get("path", ""))), "exact72 input")
    observed = _file_row(path)
    if (type(row.get("bytes")) is not int
            or observed["bytes"] != row["bytes"]
            or observed["sha256"] != _sha(row.get("sha256"), "exact72 input")):
        raise Exact72LineageSealError("exact72 input bytes/SHA drift")
    return {"role": row["role"], **observed}


def _implementation_closure() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True).stdout.strip()
        tree = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], check=True,
            capture_output=True, text=True).stdout
    except Exception as exc:
        raise Exact72LineageSealError("lineage implementation git identity unavailable") from exc
    if dirty:
        raise Exact72LineageSealError("lineage builder requires a clean committed repository")
    rows = []
    for relative in IMPLEMENTATION_PATHS:
        path = _regular(ROOT / relative, "lineage implementation source")
        rows.append({"path": relative, "bytes": path.stat().st_size,
                     "sha256": sha256_file(path)})
    return {"repo_root": str(ROOT), "git_head": head, "git_tree": tree,
            "source_closure": rows,
            "source_closure_sha256": stable_json_sha256(rows)}


def _verify_implementation_source_closure(
        implementation: Mapping[str, Any]) -> None:
    """Verify the sealed builder sources by bytes, independent of checkout path.

    A detached lineage may be consumed from a sibling checkout after the
    builder commit.  When the current checkout has evolved, reopen the exact
    source blobs from the sealed historical Git commit instead of weakening
    the byte/SHA closure or requiring the retired absolute directory to exist.
    """
    rows = implementation.get("source_closure")
    head = implementation.get("git_head")
    tree = implementation.get("git_tree")
    if (not isinstance(rows, list)
            or [row.get("path") if isinstance(row, Mapping) else None
                for row in rows] != list(IMPLEMENTATION_PATHS)
            or implementation.get("source_closure_sha256")
                != stable_json_sha256(rows)
            or not isinstance(head, str) or len(head) != 40
            or any(ch not in "0123456789abcdef" for ch in head)
            or not isinstance(tree, str) or len(tree) != 40
            or any(ch not in "0123456789abcdef" for ch in tree)):
        raise Exact72LineageSealError("lineage implementation source drift")
    mismatches: list[tuple[Mapping[str, Any], str, str]] = []
    for row, relative in zip(rows, IMPLEMENTATION_PATHS):
        if (set(row) != {"path", "bytes", "sha256"}
                or row.get("path") != relative
                or type(row.get("bytes")) is not int or row["bytes"] < 1):
            raise Exact72LineageSealError("lineage implementation source drift")
        expected_sha = _sha(row.get("sha256"), "lineage implementation source")
        try:
            current = _file_row(ROOT / relative, root=ROOT)
        except Exact72LineageSealError:
            current = None
        if current == dict(row):
            continue
        mismatches.append((row, relative, expected_sha))
    if not mismatches:
        return
    observed_tree = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--verify",
         f"{head}^{{tree}}"],
        check=False, capture_output=True, text=True).stdout.strip()
    if observed_tree != tree:
        raise Exact72LineageSealError("lineage implementation source drift")
    for row, relative, expected_sha in mismatches:
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "blob",
             f"{head}:{relative}"],
            check=False, capture_output=True).stdout
        if (len(blob) != row["bytes"]
                or hashlib.sha256(blob).hexdigest() != expected_sha):
            raise Exact72LineageSealError("lineage implementation source drift")


def build_lineage_seal(
    *, exact191_manifest_path: Path, exact191_manifest_sha256: str,
    exact191_delivery_seal_path: Path, exact191_delivery_seal_sha256: str,
    exact191_delivery_signature_path: Path,
    exact191_delivery_signature_sha256: str,
    exact72_root: Path,
) -> dict[str, Any]:
    """Verify all immutable inputs and return a detached lineage manifest."""
    manifest_path = _regular(exact191_manifest_path, "exact191 manifest")
    implementation = _implementation_closure()
    manifest = _json(manifest_path, exact191_manifest_sha256,
                     "exact191 manifest")
    delivery = _verify_delivery(
        manifest_path=manifest_path, manifest_sha256=exact191_manifest_sha256,
        delivery_seal_path=exact191_delivery_seal_path,
        delivery_seal_sha256=exact191_delivery_seal_sha256,
        delivery_signature_path=exact191_delivery_signature_path,
        delivery_signature_sha256=exact191_delivery_signature_sha256)
    rows = manifest.get("new_result_closure")
    inputs = manifest.get("input_closure")
    binding = manifest.get("execution_binding")
    if (manifest.get("schema") != EXACT191_SCHEMA
            or manifest.get("sealed") is not True
            or manifest.get("provenance_sealed") is not True
            or manifest.get("candidate_count") != 191
            or manifest.get("new_authorized_count") != EXPECTED_TASKS
            or manifest.get("new_authorized_ok_count") != EXPECTED_OK
            or manifest.get("new_authorized_typed_failure_count")
            != EXPECTED_TYPED_FAILURES
            or not isinstance(rows, list) or len(rows) != EXPECTED_TASKS
            or not isinstance(inputs, list) or len(inputs) != EXPECTED_INPUTS
            or not isinstance(binding, Mapping)):
        raise Exact72LineageSealError("exact191 execution-lineage contract mismatch")
    checked_inputs = [_validate_input_row(row) for row in inputs]
    if ({row["role"] for row in checked_inputs} != EXPECTED_INPUT_ROLES
            or stable_json_sha256(inputs)
            != manifest.get("recursive_input_closure_sha256")):
        raise Exact72LineageSealError("exact72 input closure mismatch")
    root = _directory(exact72_root, "exact72 root")
    tasks_root = _directory(root / "tasks", "exact72 tasks root")
    task_directories = sorted(
        (path for path in tasks_root.iterdir() if path.is_dir()),
        key=lambda path: path.name)
    if (len(task_directories) != EXPECTED_TASKS
            or any(path.is_symlink() for path in task_directories)):
        raise Exact72LineageSealError("exact72 task-directory closure mismatch")
    expected_task_ids: set[str] = set()
    task_closure = []
    ok_count = typed_count = 0
    authorization_sha = binding.get("authorization_sha256")
    for source in rows:
        if not isinstance(source, Mapping):
            raise Exact72LineageSealError("exact72 result row invalid")
        node_pair = source.get("node_pair")
        if (not isinstance(node_pair, list) or len(node_pair) != 2
                or any(type(item) is not int for item in node_pair)):
            raise Exact72LineageSealError("exact72 node pair invalid")
        task_id = f"{source.get('short_id')}__{node_pair[0]}_{node_pair[1]}"
        if task_id in expected_task_ids:
            raise Exact72LineageSealError("duplicate exact72 task identity")
        expected_task_ids.add(task_id)
        task_root = _directory(tasks_root / task_id, "exact72 task root")
        status = source.get("status")
        typed = status == TYPED_FAILURE
        if status not in {"ok", TYPED_FAILURE}:
            raise Exact72LineageSealError("unexpected exact72 status")
        expected_names = {"task.json", "authorized_task_view.json",
                          "attempt_receipt.json", "result.json"}
        if not typed:
            expected_names.add("correspondences.npz")
        actual_names = set()
        for path in task_root.iterdir():
            if path.is_symlink() or not path.is_file():
                raise Exact72LineageSealError("exact72 task contains unsafe entry")
            actual_names.add(path.name)
        if actual_names != expected_names:
            raise Exact72LineageSealError(
                f"exact72 task file closure mismatch: {task_id}")
        file_rows = {name: _file_row(task_root / name, root=root)
                     for name in sorted(expected_names)}
        task = _json(task_root / "task.json", file_rows["task.json"]["sha256"],
                     "exact72 task", self_hash_field="task_sha256")
        view = _json(task_root / "authorized_task_view.json",
                     source.get("authorized_task_view_sha256"),
                     "exact72 authorized task view")
        attempt = _json(task_root / "attempt_receipt.json",
                        source.get("attempt_sha256"), "exact72 attempt")
        result = _json(task_root / "result.json", source.get("result_sha256"),
                       "exact72 result")
        identity_ok = (
            task.get("task_sha256") == source.get("task_sha256")
            and task.get("short_id") == source.get("short_id")
            and task.get("candidate_index") == source.get("candidate_index")
            and task.get("node_pair") == node_pair
            and task.get("object_pair") == source.get("object_pair")
            and view.get("planned_task_sha256") == source.get("task_sha256")
            and view.get("short_id") == source.get("short_id")
            and view.get("node_pair") == node_pair
            and attempt.get("task_sha256") == source.get("task_sha256")
            and attempt.get("authorization_sha256") == authorization_sha
            and result.get("task_sha256") == source.get("task_sha256")
            and result.get("authorization_sha256") == authorization_sha
            and result.get("short_id") == source.get("short_id")
            and result.get("node_pair") == node_pair
            and result.get("object_pair") == source.get("object_pair")
            and result.get("status") == status)
        if not identity_ok:
            raise Exact72LineageSealError(
                f"exact72 task semantic binding drift: {task_id}")
        if typed:
            typed_count += 1
            if source.get("correspondence_sha256") is not None:
                raise Exact72LineageSealError(
                    "typed failure claims a correspondence SHA")
        else:
            ok_count += 1
            if file_rows["correspondences.npz"]["sha256"] != _sha(
                    source.get("correspondence_sha256"),
                    "exact72 correspondence"):
                raise Exact72LineageSealError(
                    "exact72 correspondence SHA mismatch")
        task_closure.append({
            "task_id": task_id,
            "short_id": source.get("short_id"),
            "candidate_index": source.get("candidate_index"),
            "node_pair": node_pair,
            "object_pair": source.get("object_pair"),
            "status": status,
            "typed_failure": typed,
            "correspondence_required": not typed,
            "correspondence_absent_by_contract": typed,
            "semantic_task_sha256": source.get("task_sha256"),
            "files": file_rows,
            "file_closure_sha256": stable_json_sha256(file_rows),
        })
    if ({path.name for path in task_directories} != expected_task_ids
            or ok_count != EXPECTED_OK or typed_count != EXPECTED_TYPED_FAILURES):
        raise Exact72LineageSealError("exact72 exhaustive task closure mismatch")
    value = {
        "schema": LINEAGE_SCHEMA,
        "sealed": True,
        "cpu_only": True,
        "model_imported": False,
        "solver_executed": False,
        "gpu_used": False,
        "official92_used": False,
        "gt_used": False,
        "exact191_delivery": delivery,
        "implementation": implementation,
        "exact191_manifest_path": str(manifest_path),
        "exact191_manifest_sha256": exact191_manifest_sha256,
        "exact72_root": str(root),
        "execution_binding": dict(binding),
        "input_count": len(checked_inputs),
        "input_closure": checked_inputs,
        "input_closure_sha256": stable_json_sha256(checked_inputs),
        "task_count": len(task_closure),
        "ok_count": ok_count,
        "typed_failure_count": typed_count,
        "typed_failure_correspondence_absence_verified": True,
        "task_closure": task_closure,
        "task_closure_sha256": stable_json_sha256(task_closure),
    }
    value["payload_sha256"] = stable_json_sha256(value)
    return value


def _copy_create_only(source: Path, target: Path,
                      expected: Mapping[str, Any]) -> dict[str, Any]:
    source = _regular(source, "lineage clone source")
    before = _file_row(source)
    if (before["bytes"] != expected.get("bytes")
            or before["sha256"] != expected.get("sha256")):
        raise Exact72LineageSealError("lineage clone source drift")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
            while True:
                chunk = input_stream.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        target.chmod(0o400)
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    observed = _file_row(target)
    if (observed["bytes"] != before["bytes"]
            or observed["sha256"] != before["sha256"]):
        raise Exact72LineageSealError("lineage clone verification failed")
    return observed


def materialize_lineage_seal(output_root: Path,
                             value: Mapping[str, Any]) -> tuple[Path, str]:
    """Clone all evidence create-only, freeze it, and write the manifest last."""
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise Exact72LineageSealError("lineage output already exists")
    output_root.mkdir(parents=True, mode=0o700)
    source_root = Path(str(value.get("exact72_root", ""))).resolve()
    frozen_rows = []
    try:
        for index, row in enumerate(value.get("input_closure", [])):
            source = Path(str(row["path"]))
            safe_role = "".join(ch if ch.isalnum() else "_"
                                for ch in str(row["role"]))
            relative = Path("inputs") / f"{index:02d}-{safe_role}-{source.name}"
            observed = _copy_create_only(source, output_root / relative, row)
            frozen_rows.append({"path": relative.as_posix(),
                                "bytes": observed["bytes"],
                                "sha256": observed["sha256"]})
        for task in value.get("task_closure", []):
            task_id = str(task["task_id"])
            for name, row in task["files"].items():
                source = source_root / str(row["path"])
                relative = Path("tasks") / task_id / name
                observed = _copy_create_only(source, output_root / relative, row)
                frozen_rows.append({"path": relative.as_posix(),
                                    "bytes": observed["bytes"],
                                    "sha256": observed["sha256"]})
        frozen_rows.sort(key=lambda row: row["path"])
        sealed = dict(value)
        sealed["frozen_clone_root"] = str(output_root)
        sealed["frozen_clone_file_count"] = len(frozen_rows)
        sealed["frozen_clone_file_closure"] = frozen_rows
        sealed["frozen_clone_file_closure_sha256"] = stable_json_sha256(frozen_rows)
        sealed["historical_sources_remain_mutable_but_are_not_consumed_after_clone"] = True
        sealed["payload_sha256"] = stable_json_sha256({
            key: item for key, item in sealed.items() if key != "payload_sha256"})
        manifest_path = output_root / "exact72_lineage_manifest.json"
        encoded = (json.dumps(sealed, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode()
        fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_path.chmod(0o400)
        for directory in sorted(
                [output_root, *[path for path in output_root.rglob("*")
                               if path.is_dir()]],
                key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o500)
        # Freshly traverse every frozen clone after permissions are sealed.
        for row in frozen_rows:
            path = _regular(output_root / row["path"], "frozen lineage file")
            if (_file_row(path)["sha256"] != row["sha256"]
                    or stat.S_IMODE(path.stat().st_mode) != 0o400):
                raise Exact72LineageSealError("frozen lineage clone drift")
        return manifest_path, "created"
    except Exception:
        # Preserve a failed partial tree for audit; never overwrite/resume it.
        raise


def verify_lineage_seal(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Reopen every frozen clone file and return the historical binding."""
    path = _regular(path, "exact72 lineage manifest")
    root = _directory(path.parent, "exact72 lineage root")
    if path != root / "exact72_lineage_manifest.json":
        raise Exact72LineageSealError("lineage manifest path is not canonical")
    value = _json(path, expected_sha256, "exact72 lineage manifest")
    rows = value.get("frozen_clone_file_closure")
    if (value.get("schema") != LINEAGE_SCHEMA
            or value.get("sealed") is not True
            or value.get("cpu_only") is not True
            or value.get("model_imported") is not False
            or value.get("solver_executed") is not False
            or value.get("gpu_used") is not False
            or value.get("official92_used") is not False
            or value.get("gt_used") is not False
            or Path(str(value.get("frozen_clone_root", ""))).resolve() != root
            or value.get("task_count") != EXPECTED_TASKS
            or value.get("ok_count") != EXPECTED_OK
            or value.get("typed_failure_count") != EXPECTED_TYPED_FAILURES
            or value.get("typed_failure_correspondence_absence_verified") is not True
            or not isinstance(rows, list) or len(rows) != 353
            or value.get("frozen_clone_file_count") != len(rows)
            or value.get("frozen_clone_file_closure_sha256")
            != stable_json_sha256(rows)):
        raise Exact72LineageSealError("exact72 frozen lineage contract mismatch")
    actual_paths = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise Exact72LineageSealError("frozen lineage contains symlink")
        if candidate.is_file() and candidate != path:
            actual_paths.add(candidate.relative_to(root).as_posix())
    declared_paths = set()
    rows_by_path: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if (not isinstance(row, Mapping)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row.get("path"), str)
                or Path(row["path"]).is_absolute()
                or ".." in Path(row["path"]).parts
                or row["path"] in declared_paths):
            raise Exact72LineageSealError("frozen lineage file row invalid")
        declared_paths.add(row["path"])
        rows_by_path[row["path"]] = row
        observed = _file_row(root / row["path"], root=root)
        if (observed != dict(row)
                or stat.S_IMODE((root / row["path"]).stat().st_mode) != 0o400):
            raise Exact72LineageSealError("frozen lineage file bytes/mode drift")
    if declared_paths != actual_paths:
        raise Exact72LineageSealError("frozen lineage closure is not exhaustive")
    inputs = value.get("input_closure")
    tasks = value.get("task_closure")
    if (not isinstance(inputs, list) or len(inputs) != EXPECTED_INPUTS
            or not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS
            or value.get("input_closure_sha256") != stable_json_sha256(inputs)
            or value.get("task_closure_sha256") != stable_json_sha256(tasks)):
        raise Exact72LineageSealError("lineage semantic closure mismatch")
    for index, source in enumerate(inputs):
        safe_role = "".join(ch if ch.isalnum() else "_"
                            for ch in str(source["role"]))
        frozen = (Path("inputs") /
                  f"{index:02d}-{safe_role}-{Path(source['path']).name}").as_posix()
        if (frozen not in rows_by_path
                or rows_by_path[frozen]["bytes"] != source["bytes"]
                or rows_by_path[frozen]["sha256"] != source["sha256"]):
            raise Exact72LineageSealError("frozen lineage input binding drift")
    for task in tasks:
        if not isinstance(task, Mapping):
            raise Exact72LineageSealError("lineage task row invalid")
        typed = task.get("typed_failure") is True
        files = task.get("files")
        if (not isinstance(files, Mapping)
                or task.get("correspondence_required") is not (not typed)
                or task.get("correspondence_absent_by_contract") is not typed
                or (("correspondences.npz" in files) is not (not typed))):
            raise Exact72LineageSealError("typed-failure absence contract drift")
        for name, source in files.items():
            frozen = (Path("tasks") / str(task["task_id"]) / name).as_posix()
            if (frozen not in rows_by_path
                    or rows_by_path[frozen]["bytes"] != source["bytes"]
                    or rows_by_path[frozen]["sha256"] != source["sha256"]):
                raise Exact72LineageSealError("frozen task binding drift")
    delivery = value.get("exact191_delivery")
    binding = value.get("execution_binding")
    implementation = value.get("implementation")
    implementation_root = (implementation.get("repo_root")
                           if isinstance(implementation, Mapping) else None)
    if (not isinstance(delivery, Mapping)
            or delivery.get("signature_verified") is not True
            or delivery.get("manifest_sha256")
            != value.get("exact191_manifest_sha256")
            or not isinstance(binding, Mapping)
            or not isinstance(implementation, Mapping)
            or not isinstance(implementation_root, str)
            or not Path(implementation_root).is_absolute()
            or str(Path(implementation_root).resolve()) != implementation_root
            ):
        raise Exact72LineageSealError("lineage upstream binding drift")
    _verify_implementation_source_closure(implementation)
    return {
        "lineage_manifest_sha256": expected_sha256,
        "lineage_payload_sha256": value["payload_sha256"],
        "exact191_manifest_sha256": value["exact191_manifest_sha256"],
        "frozen_clone_file_closure_sha256":
            value["frozen_clone_file_closure_sha256"],
        "task_closure_sha256": value["task_closure_sha256"],
        "input_closure_sha256": value["input_closure_sha256"],
        "task_count": value["task_count"],
        "ok_count": value["ok_count"],
        "typed_failure_count": value["typed_failure_count"],
        "execution_binding": dict(binding),
    }
