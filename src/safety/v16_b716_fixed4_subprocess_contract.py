"""Independent, fail-closed subprocess boundary for fixed4 execution fix5.

Legacy tasks retain the permanently-disabled child.  The separately pinned
active boundary can execute a pure CPU contract fixture; production stages
emit only a typed adapter-unavailable refusal until the unified operational
RESULT-v5 adapter is code-pinned.  The independent trust anchor never creates
or grants authorization by itself.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

SUBPROCESS_REGISTRY_SCHEMA = "v16-b716-fixed4-subprocess-registry-v3"
# Historical imports use this name.  Keep it as an alias while making the
# registry revision and runner mode explicit for the future active boundary.
FIX3_SUBPROCESS_SCHEMA = SUBPROCESS_REGISTRY_SCHEMA
FIX3_CONSUMPTION_SCHEMA = "v16-b716-fixed4-parent-consumption-receipt-v2"
TRUST_ANCHOR_SCHEMA = "v16-b716-fixed4-independent-trust-anchor-v1"
SIGNATURE_ALGORITHM = "ed25519-openssl-pkeyutl-raw"
DISABLED_EXIT_CODE = 78
ACTIVE_ADAPTER_UNAVAILABLE_EXIT_CODE = 69
RUNNER_MODE_DISABLED = "disabled"
RUNNER_MODE_ACTIVE = "active"
RUNNER_MODES = (RUNNER_MODE_DISABLED, RUNNER_MODE_ACTIVE)
TRACE_ACCESS_POLICY_SCHEMA = "v16-b716-fixed4-trace-access-policy-v1"
ACTIVE_PREFLIGHT_SCHEMA = "v16-b716-fixed4-active-subprocess-preflight-v1"
ACTIVE_PREFLIGHT_V2_SCHEMA = "v16-b716-fixed4-active-subprocess-preflight-v2"
ACTIVE_AUTHORIZATION_SCHEMA = (
    "v16-b716-fixed4-active-subprocess-authorization-v1")
ACTIVE_SIGNED_STAGE_AUTHORIZATION_SCHEMA = (
    "v16-b716-fixed4-active-stage-authorization-v1")
ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-active-production-execution-manifest-v1")
PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA = (
    "v16-b716-fixed4-production-manifest-transaction-commit-v2")
ACTIVE_PRODUCTION_WRAPPER_RESULT_SCHEMA = (
    "v16-b716-fixed4-active-production-wrapper-result-v1")
ACTIVE_STAGE_INPUT_SCHEMA = "v16-b716-fixed4-active-stage-input-v1"
ACTIVE_RUNNER_REFUSAL_SCHEMA = "v16-b716-fixed4-active-runner-refusal-v1"
ACTIVE_CONSUMPTION_SCHEMA = (
    "v16-b716-fixed4-active-parent-consumption-receipt-v1")
TOPOLOGICAL_PARENT_SCHEMA = "v16-b716-fixed4-active-topological-parent-v1"
CONTRACT_FIXTURE_STAGE = "contract_fixture"
ACTIVE_POLICY_FALSE_FIELDS = (
    "default_checkpoint_replaced", "gt_consumed", "official92_run",
    "reconstruction_authorized", "refusion_run", "result_selection_used",
    "thresholds_changed",
)


class Fixed4SubprocessContractError(RuntimeError):
    """The independent execution boundary failed closed."""


def sha256_file(path: Path) -> str:
    """Stdlib-only copy: active ``python -I -S`` must not require NumPy."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    """Canonical digest kept byte-compatible with v13's implementation."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("ascii")).hexdigest()


def _absolute_parts(path: Path, role: str) -> tuple[Path, tuple[str, ...]]:
    path = Path(path)
    if not path.is_absolute():
        raise Fixed4SubprocessContractError(f"{role} path must be absolute")
    parts = tuple(path.parts[1:])
    if any(part in {"", ".", ".."} for part in parts):
        raise Fixed4SubprocessContractError(f"{role} path is not canonical")
    return path, parts


def _open_directory_fd(path: Path, role: str, *, create: bool = False) -> int:
    """Open every directory component with ``O_NOFOLLOW`` from the root fd."""
    path, parts = _absolute_parts(path, role)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.anchor, flags)
    try:
        for part in parts:
            try:
                child = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o755, dir_fd=fd)
                child = os.open(part, flags, dir_fd=fd)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise Fixed4SubprocessContractError(f"{role} component is not a directory")
            os.close(fd)
            fd = child
        return fd
    except Exception as exc:
        os.close(fd)
        if isinstance(exc, Fixed4SubprocessContractError):
            raise
        raise Fixed4SubprocessContractError(
            f"{role} missing, symlinked, or not a directory") from exc


def ensure_no_symlink_directory(path: Path, role: str, *, create: bool = False) -> Path:
    fd = _open_directory_fd(path, role, create=create)
    os.close(fd)
    return Path(path)


def _open_regular_fd(path: Path, role: str) -> int:
    path, parts = _absolute_parts(path, role)
    if not parts:
        raise Fixed4SubprocessContractError(f"{role} cannot be filesystem root")
    parent_fd = _open_directory_fd(path.parent, f"{role} parent")
    try:
        fd = os.open(parts[-1], os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                     dir_fd=parent_fd)
    except Exception as exc:
        raise Fixed4SubprocessContractError(
            f"missing, symlinked, or unreadable {role}") from exc
    finally:
        os.close(parent_fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise Fixed4SubprocessContractError(f"{role} is not a regular file")
    return fd


def read_no_symlink_bytes(path: Path, role: str) -> bytes:
    fd = _open_regular_fd(path, role)
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def no_symlink_file_row(path: Path, role: str) -> dict[str, Any]:
    data = read_no_symlink_bytes(path, role)
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def no_symlink_file_mode(path: Path, role: str) -> int:
    fd = _open_regular_fd(path, role)
    try:
        return os.fstat(fd).st_mode
    finally:
        os.close(fd)


def create_only_bytes_beneath(root: Path, path: Path, data: bytes, *,
                              create_parents: bool = False,
                              resume_identical: bool = False) -> tuple[dict[str, Any], str]:
    """Create a leaf through an anchored root dirfd without following symlinks."""
    root, _ = _absolute_parts(root, "authorized output root")
    path, _ = _absolute_parts(path, "authorized output artifact")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Fixed4SubprocessContractError("artifact escapes authorized output root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise Fixed4SubprocessContractError("artifact path is not canonical")
    root_fd = _open_directory_fd(root, "authorized output root", create=create_parents)
    parent_fd = root_fd
    opened_parent = False
    try:
        if len(relative.parts) > 1:
            parent_path = root.joinpath(*relative.parts[:-1])
            parent_fd = _open_directory_fd(
                parent_path, "authorized artifact parent", create=create_parents)
            opened_parent = True
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0))
        try:
            fd = os.open(relative.parts[-1], flags, 0o444, dir_fd=parent_fd)
        except FileExistsError:
            if not resume_identical:
                raise Fixed4SubprocessContractError("artifact already exists")
            existing = read_no_symlink_bytes(path, "existing create-only artifact")
            if existing != data:
                raise Fixed4SubprocessContractError(
                    "existing create-only artifact differs")
            return ({"path": str(path), "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()}, "resumed_identical")
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise Fixed4SubprocessContractError("created artifact is not regular")
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                os.unlink(relative.parts[-1], dir_fd=parent_fd)
            except OSError:
                pass
            raise
    finally:
        if opened_parent:
            os.close(parent_fd)
        os.close(root_fd)
    return ({"path": str(path), "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}, "created")


def reserve_output_fd_beneath(root: Path, path: Path, *,
                              create_parents: bool = False) -> int:
    """Create and return one read/write output fd anchored below ``root``."""
    root, _ = _absolute_parts(root, "authorized output root")
    path, _ = _absolute_parts(path, "authorized output artifact")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Fixed4SubprocessContractError("artifact escapes authorized output root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise Fixed4SubprocessContractError("artifact path is not canonical")
    root_fd = _open_directory_fd(root, "authorized output root")
    parent_fd = root_fd
    opened_parent = False
    try:
        if len(relative.parts) > 1:
            parent_fd = _open_directory_fd(
                root.joinpath(*relative.parts[:-1]),
                "authorized artifact parent", create=create_parents)
            opened_parent = True
        try:
            fd = os.open(relative.parts[-1], os.O_RDWR | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise Fixed4SubprocessContractError("artifact already exists") from exc
    finally:
        if opened_parent:
            os.close(parent_fd)
        os.close(root_fd)
    return fd


def _canonical_signature_message(value: Mapping[str, Any]) -> bytes:
    unsigned = {key: item for key, item in value.items()
                if key not in {"signature_b64", "payload_sha256"}}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _no_symlink_regular(path: Path, role: str) -> Path:
    fd = _open_regular_fd(path, role)
    os.close(fd)
    return path


def _no_symlink_directory(path: Path, role: str) -> Path:
    return ensure_no_symlink_directory(path, role)


def _file_row(path: Path) -> dict[str, Any]:
    path = _no_symlink_regular(Path(path).absolute(), "closure file")
    return no_symlink_file_row(path, "closure file")


def _runtime_files(executable: Path) -> list[dict[str, Any]]:
    executable = _no_symlink_regular(Path(executable), "runtime executable")
    try:
        output = subprocess.run(
            ["/usr/bin/ldd", str(executable)], check=True,
            capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        ).stdout
    except Exception as exc:
        raise Fixed4SubprocessContractError("runtime dependency closure unavailable") from exc
    paths = {str(executable)}
    for token in re.findall(r"(/[A-Za-z0-9_+.,/@:=\-]+)", output):
        candidate = Path(token)
        if candidate.is_file():
            paths.add(str(candidate.resolve()))
    cache = Path("/etc/ld.so.cache")
    if cache.is_file():
        paths.add(str(cache.resolve()))
    rows = [_file_row(Path(path)) for path in sorted(paths)]
    if not rows:
        raise Fixed4SubprocessContractError("runtime closure is empty")
    return rows


def _runtime_device(path: Path, allowed_access: str) -> dict[str, Any]:
    """Bind one host device by exact path and kernel major/minor identity."""
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink()
            or allowed_access not in {"read", "read_write"}):
        raise Fixed4SubprocessContractError("runtime device declaration invalid")
    try:
        observed = path.lstat()
    except OSError as exc:
        raise Fixed4SubprocessContractError("runtime device missing") from exc
    if not stat.S_ISCHR(observed.st_mode):
        raise Fixed4SubprocessContractError("runtime device is not character device")
    return {"path": str(path), "major": os.major(observed.st_rdev),
            "minor": os.minor(observed.st_rdev),
            "allowed_access": allowed_access}


def build_subprocess_registry(
    repo: Path, *, runner_mode: str = RUNNER_MODE_DISABLED,
    include_contract_fixture: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Build the literal, non-callable stage registry from real file hashes.

    Active mode pins a real child entrypoint, but production stage adapters are
    explicitly unavailable until RESULT-v5 and upstream-receipt semantics are
    unified.  The test fixture is absent unless independently requested.
    """
    if runner_mode not in RUNNER_MODES:
        raise Fixed4SubprocessContractError("unknown subprocess runner mode")
    if include_contract_fixture and runner_mode != RUNNER_MODE_ACTIVE:
        raise Fixed4SubprocessContractError(
            "contract fixture is only valid for explicit active registry")
    # These literals are deliberately reconstructed on every call. Security
    # decisions never read caller-mutable module globals or a callable registry.
    if runner_mode == RUNNER_MODE_DISABLED:
        runner_relative = "scripts/v16_b716_fixed4_disabled_stage_runner.sh"
        executor_relative = "scripts/v16_b716_fixed4_sealed_executor.py"
        runner_sha256 = (
            "096ee122db71f08af9612840c42fb15f1135fc645912bd289f8355e42fe13f3e")
        executor_sha256 = (
            "6b6b3b3165e9a521af226b12d4c002f888de3da80c9fef1977b5846025019f19")
        expected_exit_code = DISABLED_EXIT_CODE
    else:
        runner_relative = "scripts/v16_b716_fixed4_active_stage_runner.sh"
        executor_relative = "scripts/v16_b716_fixed4_active_sealed_executor.py"
        runner_sha256 = (
            "ed59bd5e262dd32d4e44b437cc3955f9da6c5670a233cd972f1fcee8641b3a42")
        # The active executor pins this contract before importing it.  Its own
        # hash is therefore bound by the independently signed registry closure
        # rather than a circular executor<->contract literal hash dependency.
        executor_sha256 = None
        expected_exit_code = 0
    interpreter_path = "/usr/bin/dash"
    interpreter_sha256 = "86d31f6fb799e91fa21bad341484564510ca287703a16e9e46c53338776f4f42"
    tracer_path = "/usr/bin/strace"
    tracer_sha256 = "28f957c227012de0b18d1bd7fff2d396cb693ea60ed8013be68de071e84b5001"
    control_hasher_path = "/usr/bin/sha256sum"
    control_hasher_sha256 = "9992e1f1feb6f0f396bc8d6691ebc1adbfc269fd628bce84eda1d4ba5c3995c7"
    fixture_reader_path = "/usr/bin/cat"
    fixture_reader_sha256 = "a63158e6e5bce20616425f5d61e5bd7374bb5bccf15bbb93ae2e40238248f179"
    executor_interpreter_path = "/usr/bin/python3.12"
    executor_interpreter_sha256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
    production_stages = (
        "colorpcr_direction", "bidirectional_multi_solver_pilot",
        "v16_pair_hypothesis_cluster", "fixed4_aggregate")
    allowed_stages = production_stages + (
        (CONTRACT_FIXTURE_STAGE,) if include_contract_fixture else ())
    repo = Path(repo).resolve()
    runner = repo / runner_relative
    executor = repo / executor_relative
    _no_symlink_regular(runner, f"{runner_mode} stage runner")
    _no_symlink_regular(executor, "sealed stage executor")
    interpreter = Path(interpreter_path)
    tracer = Path(tracer_path)
    control_hasher = Path(control_hasher_path)
    fixture_reader = Path(fixture_reader_path)
    executor_interpreter = Path(executor_interpreter_path)
    if (sha256_file(interpreter) != interpreter_sha256
            or sha256_file(tracer) != tracer_sha256
            or sha256_file(runner) != runner_sha256
            or sha256_file(control_hasher) != control_hasher_sha256
            or (runner_mode == RUNNER_MODE_ACTIVE
                and sha256_file(fixture_reader) != fixture_reader_sha256)
            or sha256_file(executor_interpreter) != executor_interpreter_sha256
            or (executor_sha256 is not None
                and sha256_file(executor) != executor_sha256)):
        raise Fixed4SubprocessContractError(
            "code-pinned interpreter/tracer/runner SHA drift")
    runtime_executables = [interpreter, control_hasher]
    if runner_mode == RUNNER_MODE_ACTIVE:
        runtime_executables.append(fixture_reader)
    runtime = sorted({row["path"]: row for executable in runtime_executables
                      for row in _runtime_files(executable)}.values(),
                     key=lambda row: row["path"])
    executor_runtime = (
        _runtime_files(executor_interpreter)
        if runner_mode == RUNNER_MODE_ACTIVE else None)
    runtime_devices = ([
        _runtime_device(Path("/dev/null"), "read_write"),
        _runtime_device(Path("/dev/urandom"), "read"),
        _runtime_device(Path("/dev/nvidiactl"), "read_write"),
        _runtime_device(Path("/dev/nvidia0"), "read_write"),
        _runtime_device(Path("/dev/nvidia-uvm"), "read_write"),
    ] if runner_mode == RUNNER_MODE_ACTIVE else None)
    metadata_candidates = (
        "/proc/cpuinfo", "/proc/devices", "/proc/stat", "/proc/self/fd",
        "/proc/driver/nvidia/capabilities/mig/config",
        "/proc/driver/nvidia/capabilities/mig/monitor",
        "/proc/driver/nvidia/params", "/proc/self/cmdline",
        "/proc/self/environ", "/proc/self/maps", "/proc/self/status",
        "/proc/sys/vm/mmap_min_addr",
        "/sys/bus/pci/devices/0000:01:00.0/numa_node",
        "/sys/devices/system/cpu/kernel_max",
        "/sys/devices/system/cpu/online",
        "/sys/devices/system/cpu/present",
        "/sys/devices/system/cpu/possible",
        "/sys/devices/system/memory/block_size_bytes",
        "/sys/devices/system/node/node0/cpumap",
    )
    runtime_metadata_reads = ([path for path in metadata_candidates
                               if Path(path).exists()]
                              if runner_mode == RUNNER_MODE_ACTIVE else None)
    contract_source = repo / "src/safety/v16_b716_fixed4_subprocess_contract.py"
    common = {
        "schema": FIX3_SUBPROCESS_SCHEMA,
        "execution_mode": "hash_bound_independent_subprocess",
        "runner_mode": runner_mode,
        "disabled": runner_mode == RUNNER_MODE_DISABLED,
        "trace_access_policy": {
            "schema": TRACE_ACCESS_POLICY_SCHEMA,
            "declared_inputs": "read_only_exact_paths",
            "runner_outputs": "create_only_beneath_canonical_task_root",
            "write_requires_create_exclusive": True,
            "overwrite_allowed": False,
            "symlink_follow_allowed": False,
            "undeclared_read_allowed": False,
            "undeclared_write_allowed": False,
            "reserved_parent_output_prefixes": ["wrapper"],
        },
        "interpreter": _file_row(interpreter),
        "tracer": _file_row(tracer),
        "runner": {"path": runner_relative,
                   "bytes": runner.stat().st_size, "sha256": sha256_file(runner)},
        "control_hasher": _file_row(control_hasher),
        "sealed_executor": {
            "interpreter": _file_row(executor_interpreter),
            "source": {"path": executor_relative, "bytes": executor.stat().st_size,
                       "sha256": sha256_file(executor)},
            **({
                "contract_source": {
                    "path": "src/safety/v16_b716_fixed4_subprocess_contract.py",
                    "bytes": contract_source.stat().st_size,
                    "sha256": sha256_file(contract_source),
                },
                "runtime_closure": executor_runtime,
                "runtime_closure_sha256": stable_json_sha256(executor_runtime),
                "isolated_stdlib_only": True,
                "third_party_imports_allowed": False,
            } if runner_mode == RUNNER_MODE_ACTIVE else {}),
            "argv_template": [
                str(executor_interpreter), "-I", "-S", str(executor),
                "--repo", "{repo}", "--task", "{task_path}",
                "--task-sha256", "{task_sha256}",
                "--preflight", "{preflight_path}",
                "--preflight-sha256", "{preflight_sha256}",
                "--authorization", "{authorization_path}",
                "--authorization-sha256", "{authorization_sha256}",
                "--task-manifest", "{task_manifest_path}",
                "--task-manifest-sha256", "{task_manifest_sha256}",
                "--task-root", "{task_root}",
                *(["--runner-mode", RUNNER_MODE_ACTIVE]
                  if runner_mode == RUNNER_MODE_ACTIVE else []),
                *(["--allow-contract-fixture"]
                  if include_contract_fixture else [])],
        },
        "runtime_closure": runtime,
        "runtime_closure_sha256": stable_json_sha256(runtime),
        **({
            "runtime_devices": runtime_devices,
            "runtime_device_closure_sha256": stable_json_sha256(runtime_devices),
            "runtime_metadata_read_paths": runtime_metadata_reads,
            "runtime_metadata_read_closure_sha256":
                stable_json_sha256(runtime_metadata_reads),
        } if runner_mode == RUNNER_MODE_ACTIVE else {}),
        "argv_template": [
            str(interpreter), str(runner), "--stage", "{stage}",
            "--task-id", "{task_id}", "--task", "{task_path}",
            "--task-sha256", "{task_sha256}",
            "--preflight", "{preflight_path}",
            "--preflight-sha256", "{preflight_sha256}",
            "--authorization", "{authorization_path}",
            "--authorization-sha256", "{authorization_sha256}",
            "--task-manifest", "{task_manifest_path}",
            "--task-manifest-sha256", "{task_manifest_sha256}",
            *(["--runner-output", "{runner_output_path}"]
              if runner_mode == RUNNER_MODE_ACTIVE else [])],
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        "expected_exit_code": expected_exit_code,
        "runner_reported_status_trusted": False,
        "in_process_callable_allowed": False,
    }
    rows = [{
        **common, "stage": stage,
        "stage_implementation_status": (
            "permanently_disabled" if runner_mode == RUNNER_MODE_DISABLED
            else ("contract_fixture_only" if stage == CONTRACT_FIXTURE_STAGE
                  else "production_adapter_ready")),
        "expected_exit_code": (
            0 if stage == CONTRACT_FIXTURE_STAGE else expected_exit_code),
        "argv_template": [
            *common["argv_template"],
            *(["--fixture-input", "{fixture_input_path}",
               "--fixture-input-sha256", "{fixture_input_sha256}"]
              if (runner_mode == RUNNER_MODE_ACTIVE
                  and stage == CONTRACT_FIXTURE_STAGE) else ([
                  "--repo", "{repo}", "--output-root", "{output_root}",
                  "--execution-manifest", "{execution_manifest_path}",
                  "--execution-manifest-sha256", "{execution_manifest_sha256}",
                  "--production-manifest-commit",
                  "{production_manifest_commit_path}",
                  "--production-manifest-commit-sha256",
                  "{production_manifest_commit_sha256}",
                  "--production-python", "{production_python_path}",
                  "--production-python-sha256", "{production_python_sha256}",
                  "--production-wrapper", "{production_wrapper_path}",
                  "--production-wrapper-sha256", "{production_wrapper_sha256}",
                  "--runner-source-sha256", "{runner_source_sha256}"]
              if runner_mode == RUNNER_MODE_ACTIVE else [])),
        ],
    } for stage in allowed_stages]
    return rows, stable_json_sha256(rows)


def validate_subprocess_registry(
    repo: Path, rows: Any, expected_sha: Any, *,
    runner_mode: str = RUNNER_MODE_DISABLED,
    include_contract_fixture: bool = False,
) -> None:
    observed, digest = build_subprocess_registry(
        repo, runner_mode=runner_mode,
        include_contract_fixture=include_contract_fixture)
    if rows != observed or expected_sha != digest:
        raise Fixed4SubprocessContractError("subprocess registry/source/runtime drift")


def _validate_file_row(row: Mapping[str, Any], *, root: Path, role: str) -> Path:
    if set(row) != {"path", "bytes", "sha256"}:
        raise Fixed4SubprocessContractError(f"{role} row keys mismatch")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise Fixed4SubprocessContractError(f"{role} relative path invalid")
    root = Path(root).resolve()
    candidate = root / relative
    _no_symlink_regular(candidate, role)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Fixed4SubprocessContractError(f"{role} escapes recursive root") from exc
    if (type(row.get("bytes")) is not int or row["bytes"] < 1
            or resolved.stat().st_size != row["bytes"]
            or sha256_file(resolved) != row.get("sha256")):
        raise Fixed4SubprocessContractError(f"{role} bytes/SHA drift")
    return resolved


def validate_recursive_file_closure(rows: Any, expected_sha: Any,
                                    *, role: str) -> list[Path]:
    """Open every file below every sealed root and reject shallow/symlink rows.

    Each row is ``{role, root, files}``; ``files`` must enumerate every regular
    file recursively below ``root``.  Empty roots and digest-only placeholders
    are rejected.
    """
    if (not isinstance(rows, list) or not rows
            or stable_json_sha256(rows) != expected_sha):
        raise Fixed4SubprocessContractError(f"{role} recursive closure malformed")
    opened: list[Path] = []
    seen_roots: set[str] = set()
    for index, entry in enumerate(rows):
        if (not isinstance(entry, Mapping)
                or set(entry) != {"role", "root", "files"}
                or not isinstance(entry.get("role"), str)
                or not entry["role"]
                or not isinstance(entry.get("root"), str)
                or not Path(entry["root"]).is_absolute()
                or not isinstance(entry.get("files"), list)
                or not entry["files"]):
            raise Fixed4SubprocessContractError(f"{role} root row {index} invalid")
        root = _no_symlink_directory(Path(entry["root"]), f"{role} root").resolve()
        if str(root) in seen_roots:
            raise Fixed4SubprocessContractError(f"{role} duplicate root")
        seen_roots.add(str(root))
        actual: set[str] = set()
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise Fixed4SubprocessContractError(f"{role} symlink rejected")
            if candidate.is_file():
                actual.add(str(candidate.relative_to(root)))
        declared: set[str] = set()
        for file_index, file_row in enumerate(entry["files"]):
            if not isinstance(file_row, Mapping):
                raise Fixed4SubprocessContractError(f"{role} file row invalid")
            path = _validate_file_row(file_row, root=root,
                                      role=f"{role} file {index}:{file_index}")
            relative = str(path.relative_to(root))
            if relative in declared:
                raise Fixed4SubprocessContractError(f"{role} duplicate file")
            declared.add(relative)
            # Force an actual no-follow open/read, not merely validation of a SHA string.
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                with os.fdopen(fd, "rb") as stream:
                    while stream.read(1024 * 1024):
                        pass
            except Exception:
                os.close(fd)
                raise
            opened.append(path)
        if declared != actual:
            raise Fixed4SubprocessContractError(
                f"{role} is shallow/inexhaustive missing={sorted(actual-declared)} "
                f"extra={sorted(declared-actual)}")
    return opened


def load_trust_anchor(anchor_path: Path, anchor_sha256: str, *, repo_root: Path,
                      output_root: Path) -> dict[str, Any]:
    anchor_path = _no_symlink_regular(Path(anchor_path), "trust anchor")
    if stat.S_IMODE(no_symlink_file_mode(anchor_path, "trust anchor")) & 0o222:
        raise Fixed4SubprocessContractError("trust anchor is not read-only")
    anchor_bytes = read_no_symlink_bytes(anchor_path, "trust anchor")
    if (anchor_sha256 == "0" * 64
            or hashlib.sha256(anchor_bytes).hexdigest() != anchor_sha256):
        raise Fixed4SubprocessContractError("fixed independent trust anchor not provisioned/mismatch")
    for boundary, label in ((Path(repo_root).resolve(), "repository"),
                            (Path(output_root).resolve(), "output root")):
        try:
            anchor_path.resolve().relative_to(boundary)
        except ValueError:
            pass
        else:
            raise Fixed4SubprocessContractError(f"trust anchor is inside caller {label}")
    try:
        anchor = json.loads(anchor_bytes)
    except Exception as exc:
        raise Fixed4SubprocessContractError("invalid trust anchor JSON") from exc
    required = {"schema", "key_id", "public_key_path", "public_key_sha256",
                "signature_algorithm", "payload_sha256"}
    if (not isinstance(anchor, dict) or set(anchor) != required
            or anchor.get("schema") != TRUST_ANCHOR_SCHEMA
            or anchor.get("signature_algorithm") != SIGNATURE_ALGORITHM
            or anchor.get("payload_sha256") != stable_json_sha256(
                {key: item for key, item in anchor.items() if key != "payload_sha256"})):
        raise Fixed4SubprocessContractError("trust anchor contract mismatch")
    public_key = _no_symlink_regular(Path(str(anchor.get("public_key_path"))),
                                     "independent public key")
    if stat.S_IMODE(no_symlink_file_mode(
            public_key, "independent public key")) & 0o222:
        raise Fixed4SubprocessContractError("independent public key is not read-only")
    for boundary, label in ((Path(repo_root).resolve(), "repository"),
                            (Path(output_root).resolve(), "output root")):
        try:
            public_key.resolve().relative_to(boundary)
        except ValueError:
            pass
        else:
            raise Fixed4SubprocessContractError(
                f"independent public key is inside caller {label}")
    public_key_bytes = read_no_symlink_bytes(public_key, "independent public key")
    if hashlib.sha256(public_key_bytes).hexdigest() != anchor.get("public_key_sha256"):
        raise Fixed4SubprocessContractError("independent public key SHA mismatch")
    anchor["_public_key"] = str(public_key)
    return anchor


def verify_document_signature(value: Mapping[str, Any], anchor: Mapping[str, Any],
                              *, purpose: str) -> None:
    # Reconstructed locally for every verification: caller reassignment of
    # module attributes cannot redirect the cryptographic executable.
    openssl_path = "/usr/bin/openssl"
    openssl_sha256 = "30cc7c491903d6d8bca54406889c0334a167777397458214b5bd498c51b6fd97"
    if no_symlink_file_row(Path(openssl_path), "OpenSSL executable")["sha256"] \
            != openssl_sha256:
        raise Fixed4SubprocessContractError("code-pinned OpenSSL executable drift")
    if (value.get("signature_algorithm") != SIGNATURE_ALGORITHM
            or value.get("signing_key_id") != anchor.get("key_id")
            or not isinstance(value.get("signature_b64"), str)):
        raise Fixed4SubprocessContractError(f"{purpose} signature binding mismatch")
    try:
        signature = base64.b64decode(value["signature_b64"], validate=True)
    except Exception as exc:
        raise Fixed4SubprocessContractError(f"{purpose} signature encoding invalid") from exc
    with tempfile.TemporaryDirectory(prefix="fixed4-signature-verify-") as tmp:
        message_path = Path(tmp) / "message.bin"
        signature_path = Path(tmp) / "signature.bin"
        message_path.write_bytes(_canonical_signature_message(value))
        signature_path.write_bytes(signature)
        completed = subprocess.run([
            openssl_path, "pkeyutl", "-verify", "-pubin",
            "-inkey", str(anchor["_public_key"]), "-rawin",
            "-in", str(message_path), "-sigfile", str(signature_path),
        ], capture_output=True, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    if completed.returncode != 0:
        raise Fixed4SubprocessContractError(f"{purpose} independent signature rejected")


def _reject_prohibited_signer_private_key(path: Path) -> None:
    """Fail closed only when the exact code-pinned private-key leaf exists."""
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    raise Fixed4SubprocessContractError(
        "signer private key remains on execution host")


def verify_fixed_signed_document(value: Mapping[str, Any], *, repo_root: Path,
                                 output_root: Path, purpose: str) -> dict[str, Any]:
    # These values are code-pinned literals, not a module-level configuration
    # surface.  Public authority material is allowed on the execution host;
    # only the exact private-key leaf below is prohibited.
    anchor_path = Path(
        "/home/aidenwu/Documents/fixed4-independent-trust-anchor/trust-anchor-v1.json")
    anchor_sha256 = "f490dc70fbcfe7887a50de9f8d50c316b2226eb7c8bba81ddb21d4f3f1efca0b"
    prohibited_signer_private_key_path = Path(
        "/home/aidenwu/.local/share/sgaligner-exact72-audit-authority-v1/audit_private.pem")
    _reject_prohibited_signer_private_key(prohibited_signer_private_key_path)
    anchor = load_trust_anchor(anchor_path, anchor_sha256,
                               repo_root=repo_root, output_root=output_root)
    verify_document_signature(value, anchor, purpose=purpose)
    return anchor


_TRACE_OPEN = re.compile(
    r'\b(?:openat2?|open)\([^\"]*\"(?P<path>[^\"]+)\"\s*,\s*'
    r'(?P<flags>[^)]*)\)\s+=\s+(?P<result>-?[0-9]+)'
    r'(?:<(?P<result_path>/[^>\n]*)(?:<[^>\n]*>)?>)?')
_TRACE_PROHIBITED_MUTATION = re.compile(
    r'\b(?P<call>unlinkat|unlink|renameat2|renameat|rename|linkat|link|'
    r'symlinkat|symlink|rmdir|creat|truncate|ftruncate|'
    r'fchmodat|fchmod|chmod|fchownat|fchown|chown|lchown)'
    r'\([^)]*\)\s+=\s+(?P<result>-?[0-9]+)')
_TRACE_MKDIR = re.compile(
    r'\b(?:mkdirat\([^,]+,\s*|mkdir\()"(?P<path>[^"]+)"[^)]*\)'
    r'\s+=\s+(?P<result>-?[0-9]+)')


def _open_access(flags: str) -> str:
    tokens = set(re.findall(r'O_[A-Z0-9_]+', flags))
    if "O_RDWR" in tokens:
        return "read_write"
    if tokens.intersection({"O_WRONLY", "O_CREAT", "O_TRUNC", "O_APPEND",
                            "O_TMPFILE"}):
        return "write"
    return "read"


def parse_open_events(trace_text: str) -> list[dict[str, Any]]:
    """Parse strace ``open*`` calls without discarding their access flags."""
    events: list[dict[str, Any]] = []
    for match in _TRACE_OPEN.finditer(trace_text):
        result = int(match.group("result"))
        flags = match.group("flags").strip()
        requested_path = match.group("path")
        # ``strace -yy`` appends the kernel-resolved pathname to a successful
        # returned descriptor.  Prefer that absolute spelling over the raw
        # relative argument from openat(dirfd, ...).  This both preserves the
        # least-authority audit and avoids treating each legitimate relative
        # path component as an independent path.  Device annotations use a
        # second ``<...>`` suffix (for example /dev/null<char 1:3>); keep only
        # the pathname portion.
        result_path = match.group("result_path")
        path = (result_path.split("<", 1)[0]
                if result >= 0 and result_path else requested_path)
        events.append({
            "path": path,
            "requested_path": requested_path,
            "flags": flags,
            "access": _open_access(flags),
            "result": result,
            "successful": result >= 0,
        })
    return events


def parse_consumed_paths(trace_text: str) -> list[str]:
    """Compatibility view: successful absolute opens in stable order."""
    observed: list[str] = []
    for event in parse_open_events(trace_text):
        if not event["successful"]:
            continue
        path = event["path"]
        if path.startswith("/") and path not in observed:
            observed.append(path)
    return observed


def _canonical_trace_path(raw: str, cwd: Path) -> Path:
    """Return an absolute lexical normal form for one traced path.

    Native loaders legitimately emit absolute spellings such as
    ``lib-dynload/../../libz.so``.  Rejecting the raw ``..`` component before
    normalization made an allow-listed dependency fail closed even though the
    normalized target was already sealed.  Normalize lexically first (without
    following symlinks); the read audit below still resolves existing symlinks
    and requires their final target to be declared exactly.
    """
    candidate = Path(raw)
    if not candidate.is_absolute() or raw.startswith("//"):
        raise Fixed4SubprocessContractError("trace path is not canonical")
    normalized = Path(os.path.normpath(raw))
    if (not normalized.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.parts)):
        raise Fixed4SubprocessContractError("trace path is not canonical")
    return normalized


def _contains_existing_symlink(path: Path) -> bool:
    """Check all currently existing components without following a symlink."""
    path, parts = _absolute_parts(path, "trace path")
    current = Path(path.anchor)
    for part in parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(mode):
            return True
    return False


def _successful_prohibited_mutations(trace_text: str) -> list[dict[str, Any]]:
    rows = []
    for match in _TRACE_PROHIBITED_MUTATION.finditer(trace_text):
        if int(match.group("result")) < 0:
            continue
        rows.append({"call": match.group("call"),
                     "paths": re.findall(r'"([^"]*)"', match.group(0))})
    return rows


def audit_trace_access(
    trace_text: str, *, cwd: Path, declared_read_paths: Sequence[Path],
    canonical_task_root: Path, preexisting_paths: Sequence[Path] = (),
    reserved_write_prefixes: Sequence[str] = ("wrapper",),
    declared_runtime_devices: Sequence[Mapping[str, Any]] = (),
    declared_runtime_metadata_reads: Sequence[str] = (),
    mutable_scratch_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit child file access under the active-runner least-authority policy.

    Reads must name an exact declared input.  Writes must be successful
    ``O_CREAT|O_EXCL`` opens below the canonical task root, may not target a
    pre-existing path, and may not enter parent-owned prefixes.  Rename/link,
    unlink, and symlink operations are rejected because they can otherwise
    bypass create-only enforcement after the open audit.
    """
    cwd = Path(cwd).absolute()
    task_root = Path(canonical_task_root).absolute()
    declared = {str(Path(path).absolute()) for path in declared_read_paths}
    preexisting = {str(Path(path).absolute()) for path in preexisting_paths}
    reads: list[str] = []
    writes: list[str] = []
    runtime_device_paths: list[str] = []
    runtime_metadata_paths: list[str] = []
    violations: list[dict[str, str]] = []
    successful_events: list[dict[str, Any]] = []
    devices: dict[str, str] = {}
    metadata_reads = set()
    scratch_roots: list[Path] = []
    for index, item in enumerate(mutable_scratch_prefixes):
        if (not isinstance(item, str) or not item or Path(item).is_absolute()
                or len(Path(item).parts) != 1 or item in {".", ".."}
                or item in set(reserved_write_prefixes)):
            raise Fixed4SubprocessContractError(
                f"mutable scratch prefix {index} invalid")
        scratch_roots.append(task_root / item)

    def within_scratch(path: Path) -> bool:
        return any(path == root or root in path.parents for root in scratch_roots)
    for index, item in enumerate(declared_runtime_metadata_reads):
        if (not isinstance(item, str) or not Path(item).is_absolute()
                or item in metadata_reads):
            raise Fixed4SubprocessContractError(
                f"runtime metadata row {index} invalid")
        metadata_reads.add(item)
    for index, row in enumerate(declared_runtime_devices):
        if (not isinstance(row, Mapping)
                or set(row) != {"path", "major", "minor", "allowed_access"}):
            raise Fixed4SubprocessContractError(
                f"runtime device row {index} malformed")
        path = Path(str(row.get("path", "")))
        observed = _runtime_device(path, str(row.get("allowed_access", "")))
        if (observed != dict(row) or str(path) in devices):
            raise Fixed4SubprocessContractError(
                f"runtime device row {index} drift")
        devices[str(path)] = str(row["allowed_access"])

    for event in parse_open_events(trace_text):
        if not event["successful"]:
            continue
        try:
            path = _canonical_trace_path(str(event["path"]), cwd)
        except Fixed4SubprocessContractError as exc:
            violations.append({"path": str(event["path"]),
                               "reason": str(exc)})
            continue
        canonical = str(path)
        normalized = {**event, "path": canonical}
        successful_events.append(normalized)
        if canonical in devices:
            allowed_access = devices[canonical]
            if (event["access"] == "read"
                    or (event["access"] in {"write", "read_write"}
                        and allowed_access == "read_write")):
                if canonical not in runtime_device_paths:
                    runtime_device_paths.append(canonical)
                continue
            violations.append({"path": canonical,
                               "reason": "runtime device access widened"})
            continue
        requested = str(event.get("requested_path", ""))
        if (event["access"] == "read"
                and (canonical in metadata_reads or requested in metadata_reads)):
            observed_metadata = (requested if requested in metadata_reads
                                 else canonical)
            if observed_metadata not in runtime_metadata_paths:
                runtime_metadata_paths.append(observed_metadata)
            continue
        # CPython/native runtimes name their own threads through procfs.  This
        # changes process-local kernel metadata only; no persistent file or
        # other process is reachable through this exact spelling.
        if (event["access"] != "read"
                and (re.fullmatch(r"/proc/self/task/[0-9]+/comm", canonical)
                     or re.fullmatch(r"/proc/self/task/[0-9]+/comm", requested))):
            observed_metadata = (requested if requested.startswith("/proc/self/")
                                 else canonical)
            if observed_metadata not in runtime_metadata_paths:
                runtime_metadata_paths.append(observed_metadata)
            continue
        if event["access"] == "read":
            # Dynamic-loader traces may spell an allow-listed file through a
            # system alias (for example /lib -> /usr/lib).  Accept the alias
            # only when its fully-resolved target is itself an exact declared
            # input; arbitrary symlink traversal remains fail-closed.
            try:
                resolved = str(path.resolve(strict=True))
            except (FileNotFoundError, RuntimeError, OSError):
                resolved = canonical
            traverses_symlink = _contains_existing_symlink(path)
            observed = resolved if traverses_symlink else canonical
            flags = set(re.findall(r'O_[A-Z0-9_]+', str(event["flags"])))
            if "O_DIRECTORY" in flags:
                directory = Path(observed)
                descendants = [*declared, *devices, *metadata_reads]
                if any(Path(item) != directory
                       and directory in Path(item).parents
                       for item in descendants):
                    if observed not in reads:
                        reads.append(observed)
                    continue
            if observed not in declared and observed not in writes:
                violations.append({
                    "path": canonical,
                    "reason": ("declared read traverses symlink"
                               if canonical in declared and traverses_symlink
                               else "undeclared read"),
                })
                observed = canonical
            if observed not in reads:
                reads.append(observed)
            continue

        try:
            relative = path.relative_to(task_root)
        except ValueError:
            violations.append({"path": canonical,
                               "reason": "write escapes canonical task root"})
            continue
        flags = set(re.findall(r'O_[A-Z0-9_]+', str(event["flags"])))
        if within_scratch(path):
            if _contains_existing_symlink(path):
                violations.append({"path": canonical,
                                   "reason": "scratch write traverses symlink"})
            if canonical not in writes:
                writes.append(canonical)
            continue
        if (not relative.parts
                or relative.parts[0] in set(reserved_write_prefixes)):
            violations.append({"path": canonical,
                               "reason": "write targets parent-reserved path"})
        if canonical in preexisting or canonical in declared:
            violations.append({"path": canonical,
                               "reason": "write targets existing/declared input"})
        if not {"O_CREAT", "O_EXCL"}.issubset(flags):
            violations.append({"path": canonical,
                               "reason": "write is not create-exclusive"})
        if flags.intersection({"O_TRUNC", "O_APPEND", "O_TMPFILE"}):
            violations.append({"path": canonical,
                               "reason": "write uses overwrite/unlinkable flags"})
        if _contains_existing_symlink(path):
            violations.append({"path": canonical,
                               "reason": "write traverses symlink"})
        if canonical not in writes:
            writes.append(canonical)

    for mutation in _successful_prohibited_mutations(trace_text):
        paths = [Path(raw) for raw in mutation["paths"] if Path(raw).is_absolute()]
        scratch_only = (mutation["call"] in {"unlink", "unlinkat", "rmdir"}
                        and paths and all(within_scratch(path) for path in paths))
        if not scratch_only:
            violations.append({"path": "", "reason":
                f"prohibited mutation syscall: {mutation['call']}"})
    for match in _TRACE_MKDIR.finditer(trace_text):
        if int(match.group("result")) < 0:
            continue
        try:
            directory = _canonical_trace_path(match.group("path"), cwd)
            relative = directory.relative_to(task_root)
        except (Fixed4SubprocessContractError, ValueError):
            violations.append({"path": match.group("path"),
                               "reason": "mkdir escapes canonical task root"})
            continue
        if (not relative.parts
                or relative.parts[0] in set(reserved_write_prefixes)
                or _contains_existing_symlink(directory)):
            violations.append({"path": str(directory),
                               "reason": "mkdir targets reserved/symlink path"})
    return {
        "schema": TRACE_ACCESS_POLICY_SCHEMA,
        "successful_open_events": successful_events,
        "declared_read_paths": sorted(declared),
        "observed_read_paths": reads,
        "observed_write_paths": writes,
        "declared_runtime_devices": list(declared_runtime_devices),
        "declared_runtime_metadata_read_paths": sorted(metadata_reads),
        "observed_runtime_device_paths": runtime_device_paths,
        "observed_runtime_metadata_paths": runtime_metadata_paths,
        "violations": violations,
        "valid": not violations,
    }


def validate_trace_access(**kwargs: Any) -> dict[str, Any]:
    """Return a valid access report or fail closed on the first violation."""
    report = audit_trace_access(**kwargs)
    if report["violations"]:
        first = report["violations"][0]
        raise Fixed4SubprocessContractError(
            f"runner filesystem access rejected: {first['reason']} {first['path']}")
    return report


def _payload_valid_mapping(value: Mapping[str, Any]) -> bool:
    expected = value.get("payload_sha256")
    return (isinstance(expected, str)
            and expected == stable_json_sha256({
                key: item for key, item in value.items()
                if key != "payload_sha256"}))


def _validate_active_stage_input(
    value: Any, *, task: Mapping[str, Any], contract_fixture: bool,
) -> list[Path]:
    required = {"schema", "task_id", "stage", "declared_read_files",
                "declared_read_closure_sha256", "implementation_status",
                "payload_sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise Fixed4SubprocessContractError(
            "active stage_runner_input is absent/unbound")
    rows = value.get("declared_read_files")
    expected_status = (
        "contract_fixture" if contract_fixture
        else "production_adapter_unavailable")
    if (not _payload_valid_mapping(value)
            or value.get("schema") != ACTIVE_STAGE_INPUT_SCHEMA
            or value.get("task_id") != task.get("task_id")
            or value.get("stage") != task.get("stage")
            or value.get("implementation_status") != expected_status
            or not isinstance(rows, list)
            or stable_json_sha256(rows)
            != value.get("declared_read_closure_sha256")
            or (contract_fixture and len(rows) != 1)
            or (not contract_fixture and rows)):
        raise Fixed4SubprocessContractError(
            "active stage_runner_input binding/status invalid")
    paths: list[Path] = []; seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise Fixed4SubprocessContractError(
                f"active declared read row {index} malformed")
        path = Path(str(row.get("path", "")))
        if not path.is_absolute() or str(path) in seen:
            raise Fixed4SubprocessContractError(
                f"active declared read row {index} path invalid")
        observed = no_symlink_file_row(path, f"active declared read {index}")
        if (type(row.get("bytes")) is not int or row["bytes"] < 1
                or observed["bytes"] != row["bytes"]
                or observed["sha256"] != row.get("sha256")):
            raise Fixed4SubprocessContractError(
                f"active declared read row {index} bytes/SHA drift")
        seen.add(str(path)); paths.append(path)
    return paths


def _json_mapping_file(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    row = no_symlink_file_row(path, role)
    if row["sha256"] != expected_sha256:
        raise Fixed4SubprocessContractError(f"{role} SHA drift")
    try:
        value = json.loads(read_no_symlink_bytes(path, role))
    except Exception as exc:
        raise Fixed4SubprocessContractError(f"{role} JSON invalid") from exc
    if not isinstance(value, Mapping):
        raise Fixed4SubprocessContractError(f"{role} must be an object")
    return dict(value)


def _validate_active_production_descriptor(
        task: Mapping[str, Any]) -> Mapping[str, Any]:
    descriptor = task.get("stage_runner_input_descriptor")
    required = {"schema", "task_id", "stage", "upstream_task_ids",
        "input_source", "derivation_policy", "production_input_manifest_schema",
        "production_adapter_contract_schema", "operational_result_schema",
        "production_adapter_protocol_ready", "payload_sha256"}
    if (not isinstance(descriptor, Mapping) or set(descriptor) != required
            or not _payload_valid_mapping(descriptor)
            or descriptor.get("schema")
                != "v16-b716-fixed4-active-stage-input-descriptor-v2"
            or descriptor.get("task_id") != task.get("task_id")
            or descriptor.get("stage") != task.get("stage")
            or descriptor.get("upstream_task_ids") != task.get("upstream_task_ids")
            or descriptor.get("derivation_policy")
                != "dispatcher_only_never_trust_task_runtime_paths"
            or descriptor.get("production_adapter_protocol_ready") is not True):
        raise Fixed4SubprocessContractError(
            "active production descriptor is not protocol-ready")
    return descriptor


def _validate_execution_manifest_binding(
        *, task: Mapping[str, Any], authorization: Mapping[str, Any],
        execution_manifest: Mapping[str, Any], execution_manifest_path: Path,
        repo: Path) -> tuple[list[Path], dict[str, Any]]:
    """Bind the signed node authorization to both production manifests.

    The executor intentionally validates only stdlib-readable identity data.
    The production wrapper performs the stage-specific semantic validation.
    """
    if (not _payload_valid_mapping(execution_manifest)
            or execution_manifest.get("schema")
                != ACTIVE_PRODUCTION_EXECUTION_MANIFEST_SCHEMA
            or execution_manifest.get("task_id") != task.get("task_id")
            or execution_manifest.get("task_payload_sha256")
                != task.get("payload_sha256")
            or execution_manifest.get("stage") != task.get("stage")):
        raise Fixed4SubprocessContractError(
            "active production execution manifest binding invalid")
    expected_execution_path = (Path(authorization.get(
        "execution_manifest_path", "")))
    if (not expected_execution_path.is_absolute()
            or execution_manifest_path != expected_execution_path
            or no_symlink_file_row(execution_manifest_path,
                "production execution manifest")["sha256"]
                != authorization.get("execution_manifest_sha256")
            or execution_manifest.get("payload_sha256")
                != authorization.get("execution_manifest_payload_sha256")):
        raise Fixed4SubprocessContractError(
            "signed execution manifest path/file/payload binding mismatch")
    commit_path = Path(str(authorization.get(
        "production_manifest_commit_path", "")))
    expected_commit_path = execution_manifest_path.with_name("COMMITTED.json")
    if (not commit_path.is_absolute() or commit_path != expected_commit_path):
        raise Fixed4SubprocessContractError(
            "signed production transaction commit path mismatch")
    commit = _json_mapping_file(
        commit_path, str(authorization.get(
            "production_manifest_commit_sha256", "")),
        "production manifest transaction commit")
    execution_commit_row = commit.get("production_execution_manifest")
    input_commit_row = commit.get("production_input_manifest")
    if (not _payload_valid_mapping(commit)
            or commit.get("schema")
                != PRODUCTION_MANIFEST_TRANSACTION_COMMIT_SCHEMA
            or commit.get("transaction_state") != "COMMITTED"
            or commit.get("task_id") != task.get("task_id")
            or commit.get("task_payload_sha256") != task.get("payload_sha256")
            or commit.get("stage") != task.get("stage")
            or commit.get("payload_sha256")
                != authorization.get(
                    "production_manifest_commit_payload_sha256")
            or not isinstance(execution_commit_row, Mapping)
            or execution_commit_row.get("path") != str(execution_manifest_path)
            or execution_commit_row.get("sha256")
                != authorization.get("execution_manifest_sha256")
            or execution_commit_row.get("payload_sha256")
                != execution_manifest.get("payload_sha256")
            or not isinstance(input_commit_row, Mapping)):
        raise Fixed4SubprocessContractError(
            "signed production transaction commit binding mismatch")
    input_path = Path(str(execution_manifest.get(
        "production_input_manifest_path", "")))
    if (not input_path.is_absolute()
            or str(input_path) != authorization.get(
                "production_input_manifest_path")
            or execution_manifest.get("production_input_manifest_sha256")
                != authorization.get("production_input_manifest_sha256")
            or execution_manifest.get("production_input_manifest_payload_sha256")
                != authorization.get("production_input_manifest_payload_sha256")):
        raise Fixed4SubprocessContractError(
            "signed production input manifest binding mismatch")
    if (input_commit_row.get("path") != str(input_path)
            or input_commit_row.get("sha256")
                != authorization.get("production_input_manifest_sha256")
            or input_commit_row.get("payload_sha256")
                != authorization.get(
                    "production_input_manifest_payload_sha256")):
        raise Fixed4SubprocessContractError(
            "production transaction input row mismatch")
    input_manifest = _json_mapping_file(
        input_path, str(execution_manifest["production_input_manifest_sha256"]),
        "production input manifest")
    if (not _payload_valid_mapping(input_manifest)
            or input_manifest.get("payload_sha256")
                != execution_manifest.get(
                    "production_input_manifest_payload_sha256")
            or input_manifest.get("task_id") != task.get("task_id")
            or input_manifest.get("task_payload_sha256")
                != task.get("payload_sha256")
            or input_manifest.get("stage") != task.get("stage")):
        raise Fixed4SubprocessContractError(
            "production input manifest semantic binding invalid")
    runtime_rows = execution_manifest.get("runtime_dependency_files")
    if (not isinstance(runtime_rows, list) or not runtime_rows
            or stable_json_sha256(runtime_rows)
                != execution_manifest.get("runtime_dependency_closure_sha256")
            or execution_manifest.get("runtime_dependency_closure_sha256")
                != authorization.get("runtime_dependency_closure_sha256")):
        raise Fixed4SubprocessContractError(
            "signed runtime dependency closure binding mismatch")
    reads: list[Path] = [commit_path, execution_manifest_path, input_path]
    for index, row in enumerate(runtime_rows):
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise Fixed4SubprocessContractError(
                f"runtime dependency row {index} malformed")
        path = Path(str(row.get("path", "")))
        observed = no_symlink_file_row(path, f"runtime dependency {index}")
        if (observed["bytes"] != row.get("bytes")
                or observed["sha256"] != row.get("sha256")):
            raise Fixed4SubprocessContractError(
                f"runtime dependency row {index} drift")
        reads.append(path)
    interpreter = execution_manifest.get("interpreter")
    if not isinstance(interpreter, Mapping) or set(interpreter) != {
            "path", "realpath", "bytes", "sha256", "version"}:
        raise Fixed4SubprocessContractError("production interpreter row malformed")
    interpreter_path = Path(str(interpreter.get("path", "")))
    try:
        interpreter_real = interpreter_path.resolve(strict=True)
    except OSError as exc:
        raise Fixed4SubprocessContractError("production interpreter missing") from exc
    observed_interpreter = no_symlink_file_row(
        interpreter_real, "production interpreter")
    if (str(interpreter_real) != interpreter.get("realpath")
            or observed_interpreter["bytes"] != interpreter.get("bytes")
            or observed_interpreter["sha256"] != interpreter.get("sha256")
            or str(interpreter_path) != authorization.get(
                "production_interpreter_path")
            or interpreter.get("sha256")
                != authorization.get("production_interpreter_sha256")):
        raise Fixed4SubprocessContractError(
            "signed production interpreter binding mismatch")
    wrapper_path = Path(repo).resolve() / \
        "scripts/v16_b716_fixed4_active_production_wrapper.py"
    wrapper_row = no_symlink_file_row(wrapper_path, "production wrapper")
    validator_path = Path(repo).resolve() / \
        "src/safety/v16_b716_fixed4_active_production_wrapper.py"
    validator_row = no_symlink_file_row(validator_path, "production validator")
    runner_path = Path(repo).resolve() / \
        "scripts/v16_b716_fixed4_active_stage_runner.sh"
    runner_row = no_symlink_file_row(runner_path, "active runner")
    if (str(wrapper_path) != authorization.get("production_wrapper_path")
            or wrapper_row["sha256"]
                != authorization.get("production_wrapper_sha256")
            or str(validator_path) != authorization.get("validator_source_path")
            or validator_row["sha256"]
                != authorization.get("validator_source_sha256")
            or validator_row["sha256"]
                != execution_manifest.get("wrapper_source_sha256")
            or runner_row["sha256"]
                != authorization.get("runner_source_sha256")
            or runner_row["sha256"]
                != execution_manifest.get("runner_source_sha256")):
        raise Fixed4SubprocessContractError(
            "signed production wrapper/runner source binding mismatch")
    parents = execution_manifest.get("parent_result_payload_sha256s")
    if (not isinstance(parents, list)
            or parents != authorization.get("parent_result_payload_sha256s")):
        raise Fixed4SubprocessContractError(
            "signed parent result payload binding mismatch")
    for key in ACTIVE_POLICY_FALSE_FIELDS:
        if execution_manifest.get(key) is not False or authorization.get(key) is not False:
            raise Fixed4SubprocessContractError("active production policy widened")
    reads.extend([interpreter_real, wrapper_path, validator_path, runner_path])
    return reads, input_manifest


def _production_input_declared_reads(value: Mapping[str, Any]) -> list[Path]:
    """Expand the already sealed production input closure into exact files."""
    result: list[Path] = []
    for index, row in enumerate(value.get("file_inputs", ())):
        if not isinstance(row, Mapping) or set(row) != {
                "role", "path", "bytes", "sha256"}:
            raise Fixed4SubprocessContractError(
                f"production input file row {index} malformed")
        path = Path(str(row.get("path", "")))
        observed = no_symlink_file_row(path, f"production input file {index}")
        if (observed["bytes"] != row.get("bytes")
                or observed["sha256"] != row.get("sha256")):
            raise Fixed4SubprocessContractError(
                f"production input file row {index} drift")
        result.append(path)
    for index, row in enumerate(value.get("directory_inputs", ())):
        if not isinstance(row, Mapping) or set(row) != {
                "role", "path", "files", "closure_sha256"}:
            raise Fixed4SubprocessContractError(
                f"production input directory row {index} malformed")
        root = Path(str(row.get("path", "")))
        ensure_no_symlink_directory(root, f"production input directory {index}")
        files = row.get("files")
        if not isinstance(files, list) or stable_json_sha256(files) \
                != row.get("closure_sha256"):
            raise Fixed4SubprocessContractError(
                f"production input directory row {index} closure invalid")
        declared: set[str] = set()
        for child_index, child in enumerate(files):
            if not isinstance(child, Mapping) or set(child) != {
                    "path", "bytes", "sha256"}:
                raise Fixed4SubprocessContractError(
                    f"production directory child {index}:{child_index} malformed")
            relative = Path(str(child.get("path", "")))
            if relative.is_absolute() or not relative.parts \
                    or any(part in {"", ".", ".."} for part in relative.parts):
                raise Fixed4SubprocessContractError(
                    f"production directory child {index}:{child_index} path invalid")
            path = root / relative
            observed = no_symlink_file_row(
                path, f"production directory child {index}:{child_index}")
            if (observed["bytes"] != child.get("bytes")
                    or observed["sha256"] != child.get("sha256")):
                raise Fixed4SubprocessContractError(
                    f"production directory child {index}:{child_index} drift")
            declared.add(str(relative)); result.append(path)
        actual: set[str] = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise Fixed4SubprocessContractError(
                    "production input directory contains symlink")
            if path.is_file():
                actual.add(str(path.relative_to(root)))
        if actual != declared:
            raise Fixed4SubprocessContractError(
                "production input directory closure is inexhaustive")
    return result


def validate_active_control_bindings(
    *, task: Mapping[str, Any], preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
    registry_rows: Sequence[Mapping[str, Any]], registry_sha256: str,
    include_contract_fixture: bool = False,
) -> list[Path]:
    """Require new active schemas; legacy disabled documents cannot activate."""
    if not _payload_valid_mapping(task):
        raise Fixed4SubprocessContractError("active task payload SHA invalid")
    stage = task.get("stage"); task_id = task.get("task_id")
    contract_fixture = stage == CONTRACT_FIXTURE_STAGE
    if (not isinstance(stage, str) or not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", task_id)
            or contract_fixture is not bool(include_contract_fixture)):
        raise Fixed4SubprocessContractError("active task identity/mode invalid")
    expected_binding = task_execution_binding(
        registry_rows, registry_sha256, stage, task_id)
    if (expected_binding.get("runner_mode") != RUNNER_MODE_ACTIVE
            or task.get("execution_binding") != expected_binding):
        raise Fixed4SubprocessContractError(
            "active task executable/registry binding drift")
    declared = _validate_active_stage_input(
        task.get("stage_runner_input"), task=task,
        contract_fixture=contract_fixture)
    row = next((item for item in registry_rows if item.get("stage") == stage), None)
    if row is None:
        raise Fixed4SubprocessContractError("active task stage not registered")
    expected_preflight = {
        "schema": ACTIVE_PREFLIGHT_SCHEMA,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "runner_registry_closure_sha256": registry_sha256,
        "sealed_executor_sha256": row["sealed_executor"]["source"]["sha256"],
        "legacy_disabled_preflight_accepted": False,
        "contract_fixture_allowed": contract_fixture,
        "operational_result_release_allowed": False,
    }
    if preflight.get("active_subprocess_contract") != expected_preflight:
        raise Fixed4SubprocessContractError(
            "legacy/unbound preflight cannot activate runner")
    stage_input = task["stage_runner_input"]
    expected_authorization = {
        "schema": ACTIVE_AUTHORIZATION_SCHEMA,
        "runner_mode": RUNNER_MODE_ACTIVE,
        "runner_registry_closure_sha256": registry_sha256,
        "task_id": task_id, "task_payload_sha256": task["payload_sha256"],
        "stage": stage,
        "execution_binding_sha256": stable_json_sha256(expected_binding),
        "stage_input_payload_sha256": stage_input["payload_sha256"],
        "execution_authorized": True,
        "contract_fixture_allowed": contract_fixture,
        "operational_result_release_allowed": False,
    }
    if authorization.get("active_subprocess_authorization") \
            != expected_authorization:
        raise Fixed4SubprocessContractError(
            "active task authorization binding mismatch")
    return declared


def validate_active_production_control_bindings(
    *, task: Mapping[str, Any], preflight: Mapping[str, Any],
    authorization: Mapping[str, Any], execution_manifest: Mapping[str, Any],
    execution_manifest_path: Path, repo: Path,
    registry_rows: Sequence[Mapping[str, Any]], registry_sha256: str,
) -> tuple[list[Path], dict[str, Any]]:
    """Validate a signed, manifest-bound production node without upgrading v1."""
    if not _payload_valid_mapping(task):
        raise Fixed4SubprocessContractError("active task payload SHA invalid")
    stage = task.get("stage"); task_id = task.get("task_id")
    if (not isinstance(stage, str) or stage == CONTRACT_FIXTURE_STAGE
            or not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+", task_id)):
        raise Fixed4SubprocessContractError("active production task identity invalid")
    descriptor = _validate_active_production_descriptor(task)
    expected_binding = task_execution_binding(
        registry_rows, registry_sha256, stage, task_id)
    if (expected_binding.get("runner_mode") != RUNNER_MODE_ACTIVE
            or expected_binding.get("stage_implementation_status")
                != "production_adapter_ready"
            or task.get("execution_binding") != expected_binding):
        raise Fixed4SubprocessContractError(
            "active production executable/registry binding drift")
    row = next((item for item in registry_rows if item.get("stage") == stage), None)
    active = preflight.get("active_subprocess_contract")
    if (row is None or not isinstance(active, Mapping)
            or active.get("schema") != ACTIVE_PREFLIGHT_V2_SCHEMA
            or active.get("runner_mode") != RUNNER_MODE_ACTIVE
            or active.get("runner_registry_closure_sha256") != registry_sha256
            or active.get("sealed_executor_sha256")
                != row["sealed_executor"]["source"]["sha256"]
            or active.get("legacy_disabled_preflight_accepted") is not False
            or active.get("contract_fixture_allowed") is not False
            or active.get("production_adapter_protocol_ready") is not True
            or active.get("operational_result_release_allowed") is not False):
        raise Fixed4SubprocessContractError(
            "production-ready v2 preflight binding missing")
    if (not _payload_valid_mapping(authorization)
            or authorization.get("schema")
                != ACTIVE_SIGNED_STAGE_AUTHORIZATION_SCHEMA
            or authorization.get("status") != "PASS"
            or authorization.get("authorization_scope") != "one_topological_task"
            or authorization.get("execution_authorized") is not True
            or authorization.get("execution_performed") is not False
            or authorization.get("task_id") != task_id
            or authorization.get("task_payload_sha256")
                != task.get("payload_sha256")
            or authorization.get("stage") != stage
            or authorization.get("stage_input_descriptor_payload_sha256")
                != descriptor.get("payload_sha256")
            or authorization.get("runner_registry_closure_sha256")
                != registry_sha256
            or authorization.get("production_adapter_protocol_ready") is not True
            or authorization.get("signer_private_key_not_on_execution_host") is not True):
        raise Fixed4SubprocessContractError(
            "signed active production authorization binding mismatch")
    return _validate_execution_manifest_binding(
        task=task, authorization=authorization,
        execution_manifest=execution_manifest,
        execution_manifest_path=Path(execution_manifest_path), repo=Path(repo))


def _validate_signed_active_control_files(
        *, repo: Path, output_root: Path, task: Mapping[str, Any],
        task_path: Path, preflight: Mapping[str, Any], preflight_path: Path,
        authorization: Mapping[str, Any], task_manifest_path: Path) -> None:
    """Bind every CLI control file to the signed one-node authorization."""
    task_manifest = _json_mapping_file(
        task_manifest_path, str(authorization.get("task_manifest_sha256", "")),
        "active task manifest")
    if (not _payload_valid_mapping(preflight)
            or not _payload_valid_mapping(task_manifest)
            or authorization.get("repo_root") != str(Path(repo).resolve())
            or authorization.get("output_root") != str(Path(output_root).resolve())
            or authorization.get("task_path") != str(Path(task_path).resolve())
            or authorization.get("task_sha256") != sha256_file(task_path)
            or authorization.get("task_payload_sha256")
                != task.get("payload_sha256")
            or authorization.get("preflight_path")
                != str(Path(preflight_path).resolve())
            or authorization.get("preflight_sha256")
                != sha256_file(preflight_path)
            or authorization.get("preflight_payload_sha256")
                != preflight.get("payload_sha256")
            or authorization.get("task_manifest_path")
                != str(Path(task_manifest_path).resolve())
            or authorization.get("task_manifest_payload_sha256")
                != task_manifest.get("payload_sha256")
            or authorization.get("runner_registry_closure_sha256")
                != preflight.get("runner_registry_closure_sha256")
            or authorization.get("execution_source_closure_sha256")
                != preflight.get("execution_source_closure_sha256")
            or authorization.get("upstream_task_ids")
                != task.get("upstream_task_ids")):
        raise Fixed4SubprocessContractError(
            "signed active control file binding mismatch")


def _production_refusal_bytes(task: Mapping[str, Any], task_sha256: str) -> bytes:
    return (
        '{"schema":"' + ACTIVE_RUNNER_REFUSAL_SCHEMA
        + '","runner_mode":"active","stage":"' + str(task["stage"])
        + '","status":"adapter_unavailable","failure_type":'
          '"PRODUCTION_STAGE_ADAPTER_UNAVAILABLE","task_id":"'
        + str(task["task_id"]) + '","task_sha256":"' + task_sha256
        + '","operational_result_emitted":false}\n').encode("utf-8")


def build_topological_parent_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deterministic upstream-before-child order without executing."""
    rows = value.get("tasks") if isinstance(value, Mapping) else None
    if (not isinstance(value, Mapping) or set(value) != {
            "schema", "tasks", "task_closure_sha256", "payload_sha256"}
            or value.get("schema") != TOPOLOGICAL_PARENT_SCHEMA
            or not _payload_valid_mapping(value) or not isinstance(rows, list)
            or stable_json_sha256(rows) != value.get("task_closure_sha256")):
        raise Fixed4SubprocessContractError("active topological plan malformed")
    seen: set[str] = set(); order: list[dict[str, Any]] = []
    production = {
        "colorpcr_direction", "bidirectional_multi_solver_pilot",
        "v16_pair_hypothesis_cluster", "fixed4_aggregate"}
    for index, row in enumerate(rows):
        if (not isinstance(row, Mapping) or set(row) != {
                "task_id", "stage", "upstream_task_ids", "task_payload_sha256"}
                or not isinstance(row.get("task_id"), str)
                or row["task_id"] in seen or row.get("stage") not in production
                or not isinstance(row.get("upstream_task_ids"), list)
                or any(not isinstance(item, str) or item not in seen
                       for item in row["upstream_task_ids"])
                or not isinstance(row.get("task_payload_sha256"), str)
                or len(row["task_payload_sha256"]) != 64):
            raise Fixed4SubprocessContractError(
                f"active topological task row {index} invalid/non-topological")
        seen.add(row["task_id"]); order.append(dict(row))
    receipt = {
        "schema": "v16-b716-fixed4-active-topological-parent-receipt-v1",
        "task_count": len(order), "task_order": order,
        "task_order_sha256": stable_json_sha256(order),
        "execution_performed": False,
        "production_dispatch_available": False,
        "failure_type": "PRODUCTION_STAGE_ADAPTER_UNAVAILABLE",
    }
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    return receipt


def execute_disabled_stage(*, repo: Path, task: Mapping[str, Any], task_path: Path,
                           preflight_path: Path, authorization_path: Path,
                           task_manifest_path: Path, task_root: Path,
                           registry_rows: Sequence[Mapping[str, Any]],
                           registry_sha256: str) -> dict[str, Any]:
    """Run the only checked-in stage entrypoint and seal parent observations.

    This routine cannot accept a runner path or callable.  It derives the sole
    executable from the hash-bound registry, revalidates it before and after,
    and only accepts the permanently-disabled exit code.
    """
    repo = Path(repo).resolve()
    task_root = Path(task_root)
    output_root = task_root.parents[1]
    ensure_no_symlink_directory(output_root, "authorized output root")
    ensure_no_symlink_directory(task_root, "authorized task root")
    if task_root.relative_to(output_root) != Path("tasks") / str(task.get("task_id")):
        raise Fixed4SubprocessContractError("task root escapes authorized layout")
    validate_subprocess_registry(repo, registry_rows, registry_sha256)
    expected_binding = task_execution_binding(
        registry_rows, registry_sha256, str(task.get("stage")), str(task.get("task_id")))
    if task.get("execution_binding") != expected_binding:
        raise Fixed4SubprocessContractError("task executable/argv/env/cwd binding drift")
    row = next(item for item in registry_rows if item["stage"] == task["stage"])
    if (row.get("schema") != SUBPROCESS_REGISTRY_SCHEMA
            or row.get("runner_mode") != RUNNER_MODE_DISABLED
            or row.get("disabled") is not True
            or expected_binding.get("runner_mode") != RUNNER_MODE_DISABLED):
        raise Fixed4SubprocessContractError(
            "disabled executor received non-disabled registry mode")
    runner = repo / row["runner"]["path"]
    cwd = task_root
    if expected_binding["cwd_contract"] != str(cwd.relative_to(output_root)):
        # The task root is always <output>/tasks/<task-id>.
        raise Fixed4SubprocessContractError("task cwd binding mismatch")
    control_paths = [Path(task_path), Path(preflight_path), Path(authorization_path),
                     Path(task_manifest_path)]
    expected_controls = [task_root / "task.json", output_root / "execution_preflight.json",
                         output_root / "authorization.json",
                         output_root / "task_manifest.json"]
    if control_paths != expected_controls:
        raise Fixed4SubprocessContractError("control input path/layout mismatch")
    control_rows = [_file_row(path) for path in control_paths]
    substitutions = {
        "task_path": str(control_paths[0]), "task_sha256": control_rows[0]["sha256"],
        "preflight_path": str(control_paths[1]),
        "preflight_sha256": control_rows[1]["sha256"],
        "authorization_path": str(control_paths[2]),
        "authorization_sha256": control_rows[2]["sha256"],
        "task_manifest_path": str(control_paths[3]),
        "task_manifest_sha256": control_rows[3]["sha256"],
    }
    argv = [token.format(**substitutions) for token in expected_binding["argv"]]
    wrapper_root = task_root / "wrapper"
    try:
        wrapper_root.lstat()
    except FileNotFoundError:
        pass
    else:
        raise Fixed4SubprocessContractError("partial wrapper state; overwrite forbidden")
    preexisting_task_paths = []
    for path in task_root.rglob("*"):
        if path.is_symlink():
            raise Fixed4SubprocessContractError(
                "preexisting task-root symlink rejected")
        if path.is_file():
            preexisting_task_paths.append(path.absolute())
    ensure_no_symlink_directory(wrapper_root, "authorized wrapper root", create=True)
    env = dict(row["environment"])
    trace_path = wrapper_root / "strace.log"
    trace_fd = reserve_output_fd_beneath(output_root, trace_path)
    try:
        completed = subprocess.run([
          row["tracer"]["path"], "-f", "-qq", "-yy", "-s", "4096",
            "-e", ("trace=open,openat,openat2,execve,creat,truncate,ftruncate,"
                   "mkdir,mkdirat,rmdir,unlink,unlinkat,rename,renameat,renameat2,"
                   "link,linkat,symlink,symlinkat,chmod,fchmod,fchmodat,chown,"
                   "fchown,lchown,fchownat"),
            "-o", f"/proc/self/fd/{trace_fd}", *argv,
        ], cwd=cwd, env=env, capture_output=True, check=False, pass_fds=(trace_fd,))
        os.fsync(trace_fd)
        os.fchmod(trace_fd, 0o400)
        os.lseek(trace_fd, 0, os.SEEK_SET)
        trace_bytes = b""
        while True:
            chunk = os.read(trace_fd, 1024 * 1024)
            if not chunk:
                break
            trace_bytes += chunk
    finally:
        os.close(trace_fd)
    validate_subprocess_registry(repo, registry_rows, registry_sha256)
    observed = parse_consumed_paths(trace_bytes.decode("utf-8", errors="replace"))
    allowed = {str(runner.resolve()), str(Path(row["interpreter"]["path"]).resolve())}
    allowed.update(str(Path(item["path"]).resolve()) for item in row["runtime_closure"])
    allowed.update(str(path) for path in control_paths)
    access_report = audit_trace_access(
        trace_bytes.decode("utf-8", errors="replace"), cwd=cwd,
        declared_read_paths=[Path(path) for path in sorted(allowed)],
        canonical_task_root=task_root,
        preexisting_paths=preexisting_task_paths,
        reserved_write_prefixes=row["trace_access_policy"][
            "reserved_parent_output_prefixes"],
    )
    post_symlinks = [str(path.absolute()) for path in task_root.rglob("*")
                     if path.is_symlink()]
    for path in post_symlinks:
        access_report["violations"].append(
            {"path": path, "reason": "post-run task-root symlink rejected"})
    access_report["valid"] = not access_report["violations"]
    extras = sorted({item["path"] for item in access_report["violations"]
                     if item["path"]})
    normalized_observed = set(access_report["observed_read_paths"])
    trace_valid = (access_report["valid"]
                   and str(runner.resolve()) in normalized_observed
                   and all(str(path) in normalized_observed for path in control_paths))
    failure_type = classify_wrapper_failure(
        completed.returncode, trace_valid, completed.stdout, completed.stderr)
    stdout_row, _ = create_only_bytes_beneath(
        output_root, wrapper_root / "stdout.bin", completed.stdout)
    stderr_row, _ = create_only_bytes_beneath(
        output_root, wrapper_root / "stderr.bin", completed.stderr)
    trace_row = no_symlink_file_row(trace_path, "parent trace")
    receipt = {
        "schema": FIX3_CONSUMPTION_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "subprocess_registry_closure_sha256": registry_sha256,
        "executable": row["interpreter"], "runner": row["runner"],
        "runtime_closure_sha256": row["runtime_closure_sha256"],
        "argv": argv, "environment": env, "cwd": str(cwd),
        "control_inputs": control_rows,
        "control_input_closure_sha256": stable_json_sha256(control_rows),
        "parent_observed_consumed_paths": observed,
        "parent_observed_consumed_paths_sha256": stable_json_sha256(observed),
        "parent_observed_accesses": access_report,
        "parent_observed_accesses_sha256": stable_json_sha256(access_report),
        "undeclared_consumed_paths": extras,
        "returncode": completed.returncode,
        "stdout": stdout_row, "stderr": stderr_row, "strace": trace_row,
        "failure_type": failure_type,
        "runner_reported_failure_type_trusted": False,
    }
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    encoded = (json.dumps(receipt, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    create_only_bytes_beneath(
        output_root, wrapper_root / "consumption_receipt.json", encoded)
    return receipt


def _bound_output_row_path(row: Mapping[str, Any], output_root: Path,
                           role: str) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(row):
        raise Fixed4SubprocessContractError(f"{role} artifact row malformed")
    raw = Path(str(row.get("path", "")))
    path = raw if raw.is_absolute() else Path(output_root) / raw
    path = path.absolute()
    try:
        path.relative_to(Path(output_root).absolute())
    except ValueError as exc:
        raise Fixed4SubprocessContractError(f"{role} artifact escapes output root") from exc
    observed = no_symlink_file_row(path, role)
    if (observed["bytes"] != row.get("bytes")
            or observed["sha256"] != row.get("sha256")):
        raise Fixed4SubprocessContractError(f"{role} artifact binding drift")
    return path


def _collect_result_artifact_paths(value: Any, output_root: Path,
                                   result: set[str], role: str) -> None:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            result.add(str(_bound_output_row_path(value, output_root, role)))
        for key, child in value.items():
            _collect_result_artifact_paths(child, output_root, result,
                                           f"{role}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_result_artifact_paths(child, output_root, result,
                                           f"{role}[{index}]")


def _validate_production_wrapper_output(
        *, output_bytes: bytes, task: Mapping[str, Any], task_root: Path,
        output_root: Path, observed_write_paths: set[str]) -> bool:
    """Require every child-created file to be transitively hash referenced."""
    try:
        wrapper = json.loads(output_bytes)
        if (not isinstance(wrapper, Mapping) or not _payload_valid_mapping(wrapper)
                or wrapper.get("schema") != ACTIVE_PRODUCTION_WRAPPER_RESULT_SCHEMA
                or wrapper.get("task_id") != task.get("task_id")
                or wrapper.get("stage") != task.get("stage")
                or wrapper.get("operational_result_released") is not False
                or wrapper.get("process_failure_observed") is not False):
            return False
        expected: set[str] = {str(Path(task_root) / "active" / "runner_output.bin")}
        documents: dict[str, Mapping[str, Any]] = {}
        for prefix in ("candidate", "validation", "attempt"):
            path = Path(str(wrapper.get(f"{prefix}_path", ""))).absolute()
            if path.parent not in {Path(task_root) / "active",
                                   Path(task_root) / "production"}:
                return False
            value = _json_mapping_file(
                path, str(wrapper.get(f"{prefix}_sha256", "")),
                f"production {prefix}")
            if value.get("payload_sha256") != wrapper.get(
                    f"{prefix}_payload_sha256"):
                return False
            documents[prefix] = value; expected.add(str(path))
        candidate = documents["candidate"]
        validation = documents["validation"]
        attempt = documents["attempt"]
        if (candidate.get("task_id") != task.get("task_id")
                or validation.get("task_id") != task.get("task_id")
                or attempt.get("task_id") != task.get("task_id")
                or validation.get("status") != "PASS"
                or validation.get("candidate_payload_sha256")
                    != candidate.get("payload_sha256")
                or attempt.get("candidate_payload_sha256")
                    != candidate.get("payload_sha256")):
            return False
        _collect_result_artifact_paths(candidate, output_root, expected,
                                       "production candidate")
        rows = validation.get("output_artifact_rows")
        if (not isinstance(rows, list)
                or stable_json_sha256(rows)
                    != validation.get("output_artifact_closure_sha256")):
            return False
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                return False
            expected.add(str(_bound_output_row_path(
                row, output_root, f"production output {index}")))
        contract_path = Path(str(validation.get(
            "production_adapter_contract_path", ""))).absolute()
        if contract_path != Path(task_root) / "active" / \
                "production_adapter_contract.json":
            return False
        contract = _json_mapping_file(
            contract_path, str(validation.get(
                "production_adapter_contract_sha256", "")),
            "production adapter contract")
        if contract.get("payload_sha256") != validation.get(
                "production_adapter_contract_payload_sha256"):
            return False
        expected.add(str(contract_path))
        if task.get("stage") == "bidirectional_multi_solver_pilot":
            for name in ("slot_expansion.json", "slot_results.json"):
                path = Path(task_root) / "active" / name
                value = _json_mapping_file(path, sha256_file(path),
                                           f"pilot {name}")
                if not _payload_valid_mapping(value):
                    return False
                expected.add(str(path))
        return expected == observed_write_paths
    except (Fixed4SubprocessContractError, KeyError, TypeError, ValueError,
            json.JSONDecodeError):
        return False


def execute_active_stage(*, repo: Path, task: Mapping[str, Any], task_path: Path,
                         preflight_path: Path, authorization_path: Path,
                         task_manifest_path: Path, task_root: Path,
                         registry_rows: Sequence[Mapping[str, Any]],
                         registry_sha256: str,
                         execution_manifest_path: Path | None = None,
                         execution_manifest_sha256: str | None = None,
                         include_contract_fixture: bool = False) -> dict[str, Any]:
    """Run the active child while preventing any operational-result release.

    The pure CPU fixture proves process and filesystem enforcement. Production
    stages execute the same hash-bound child but can only emit a typed adapter
    refusal until the unified RESULT-v5 adapter is separately code-pinned.
    """
    repo = Path(repo).resolve(); task_root = Path(task_root)
    output_root = task_root.parents[1]
    ensure_no_symlink_directory(output_root, "authorized output root")
    ensure_no_symlink_directory(task_root, "authorized task root")
    if task_root.relative_to(output_root) != Path("tasks") / str(task.get("task_id")):
        raise Fixed4SubprocessContractError("task root escapes authorized layout")
    validate_subprocess_registry(
        repo, registry_rows, registry_sha256, runner_mode=RUNNER_MODE_ACTIVE,
        include_contract_fixture=include_contract_fixture)
    try:
        preflight = json.loads(read_no_symlink_bytes(
            preflight_path, "active preflight"))
        authorization = json.loads(read_no_symlink_bytes(
            authorization_path, "active authorization"))
    except Exception as exc:
        raise Fixed4SubprocessContractError(
            "active control JSON malformed") from exc
    fixture = task.get("stage") == CONTRACT_FIXTURE_STAGE
    input_manifest: dict[str, Any] | None = None
    if fixture:
        declared = validate_active_control_bindings(
            task=task, preflight=preflight, authorization=authorization,
            registry_rows=registry_rows, registry_sha256=registry_sha256,
            include_contract_fixture=include_contract_fixture)
        execution_manifest = None
    else:
        if execution_manifest_path is None or execution_manifest_sha256 is None:
            raise Fixed4SubprocessContractError(
                "active production execution manifest absent")
        execution_manifest_path = Path(execution_manifest_path)
        execution_manifest = _json_mapping_file(
            execution_manifest_path, execution_manifest_sha256,
            "production execution manifest")
        declared, input_manifest = validate_active_production_control_bindings(
            task=task, preflight=preflight, authorization=authorization,
            execution_manifest=execution_manifest,
            execution_manifest_path=execution_manifest_path, repo=repo,
            registry_rows=registry_rows, registry_sha256=registry_sha256)
        declared.extend(_production_input_declared_reads(input_manifest))
    expected_binding = task_execution_binding(
        registry_rows, registry_sha256, str(task["stage"]), str(task["task_id"]))
    row = next(item for item in registry_rows if item["stage"] == task["stage"])
    expected_status = (
        "contract_fixture_only" if fixture else "production_adapter_ready")
    if (row.get("runner_mode") != RUNNER_MODE_ACTIVE
            or row.get("disabled") is not False
            or row.get("stage_implementation_status") != expected_status):
        raise Fixed4SubprocessContractError("active registry status drift")
    control_paths = [Path(task_path), Path(preflight_path), Path(authorization_path),
                     Path(task_manifest_path)]
    authorization_body_sha256 = stable_json_sha256({
        key: item for key, item in authorization.items()
        if key not in {"signature_b64", "payload_sha256"}
    })
    expected_controls = [
        task_root / "task.json", output_root / "execution_preflight.json",
        output_root / "authorizations" / str(task["task_id"]) /
            f"{authorization_body_sha256}.json",
        output_root / "task_manifest.json",
    ]
    if control_paths != expected_controls:
        raise Fixed4SubprocessContractError("active control input path/layout mismatch")
    if not fixture:
        _validate_signed_active_control_files(
            repo=repo, output_root=output_root, task=task,
            task_path=Path(task_path), preflight=preflight,
            preflight_path=Path(preflight_path), authorization=authorization,
            task_manifest_path=Path(task_manifest_path))
    control_rows = [_file_row(path) for path in control_paths]
    wrapper_root = task_root / "wrapper"; active_root = task_root / "active"
    for path in (wrapper_root, active_root):
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise Fixed4SubprocessContractError(
                "partial active state; overwrite forbidden")
    preexisting: list[Path] = []
    for path in task_root.rglob("*"):
        if path.is_symlink():
            raise Fixed4SubprocessContractError(
                "preexisting task-root symlink rejected")
        if path.is_file():
            preexisting.append(path.absolute())
    expected_preexisting = {str(Path(task_path).absolute())}
    if not fixture:
        expected_preexisting.update({
            str(Path(execution_manifest_path).with_name(
                "COMMITTED.json").absolute()),
            str(Path(execution_manifest_path).absolute()),
            str(Path(str(execution_manifest[
                "production_input_manifest_path"])).absolute()),
        })
    if {str(path) for path in preexisting} != expected_preexisting:
        raise Fixed4SubprocessContractError(
            "active task inventory contains extra/unsealed file")
    ensure_no_symlink_directory(wrapper_root, "authorized wrapper root", create=True)
    ensure_no_symlink_directory(active_root, "authorized active root", create=True)
    scratch_root = task_root / "scratch"
    ensure_no_symlink_directory(scratch_root, "authorized task scratch", create=True)
    output_path = active_root / "runner_output.bin"
    substitutions = {
        "task_path": str(control_paths[0]), "task_sha256": control_rows[0]["sha256"],
        "preflight_path": str(control_paths[1]),
        "preflight_sha256": control_rows[1]["sha256"],
        "authorization_path": str(control_paths[2]),
        "authorization_sha256": control_rows[2]["sha256"],
        "task_manifest_path": str(control_paths[3]),
        "task_manifest_sha256": control_rows[3]["sha256"],
        "runner_output_path": str(output_path),
        "fixture_input_path": str(declared[0]) if fixture else "",
        "fixture_input_sha256": (
            no_symlink_file_row(declared[0], "fixture input")["sha256"]
            if fixture else ""),
        "repo": str(repo), "output_root": str(output_root),
        "execution_manifest_path": (
            "" if fixture else str(execution_manifest_path)),
        "execution_manifest_sha256": (
            "" if fixture else str(execution_manifest_sha256)),
        "production_manifest_commit_path": (
            "" if fixture else str(Path(execution_manifest_path).with_name(
                "COMMITTED.json"))),
        "production_manifest_commit_sha256": (
            "" if fixture else str(authorization[
                "production_manifest_commit_sha256"])),
        "production_python_path": (
            "" if fixture else str(execution_manifest["interpreter"]["path"])),
        "production_python_sha256": (
            "" if fixture else str(execution_manifest["interpreter"]["sha256"])),
        "production_wrapper_path": (
            "" if fixture else str(repo /
                "scripts/v16_b716_fixed4_active_production_wrapper.py")),
        "production_wrapper_sha256": (
            "" if fixture else str(authorization[
                "production_wrapper_sha256"])),
        "runner_source_sha256": (
            "" if fixture else str(execution_manifest["runner_source_sha256"])),
    }
    argv = [token.format(**substitutions) for token in expected_binding["argv"]]
    runner = repo / row["runner"]["path"]; env = dict(row["environment"])
    # All non-persistent runtime state is derived from the signed task root.
    # This keeps compiler/matplotlib/tempfile activity out of shared /tmp and
    # gives the parent a single scratch subtree that can be audited and erased.
    env.update({"TMPDIR": str(scratch_root), "TMP": str(scratch_root),
        "TEMP": str(scratch_root), "MPLCONFIGDIR": str(scratch_root / "matplotlib"),
        "JOBLIB_TEMP_FOLDER": str(scratch_root / "joblib"),
        "JOBLIB_MULTIPROCESSING": "0", "LOKY_MAX_CPU_COUNT": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    trace_path = wrapper_root / "strace.log"
    trace_fd = reserve_output_fd_beneath(output_root, trace_path)
    try:
        completed = subprocess.run([
            row["tracer"]["path"], "-f", "-qq", "-yy", "-s", "4096",
            "-e", ("trace=open,openat,openat2,execve,creat,truncate,ftruncate,"
                   "mkdir,mkdirat,rmdir,unlink,unlinkat,rename,renameat,renameat2,"
                   "link,linkat,symlink,symlinkat,chmod,fchmod,fchmodat,chown,"
                   "fchown,lchown,fchownat"),
            "-o", f"/proc/self/fd/{trace_fd}", *argv,
        ], cwd=task_root, env=env, capture_output=True, check=False,
           pass_fds=(trace_fd,))
        os.fsync(trace_fd); os.fchmod(trace_fd, 0o400); os.lseek(trace_fd, 0, os.SEEK_SET)
        trace_bytes = b""
        while True:
            chunk = os.read(trace_fd, 1024 * 1024)
            if not chunk:
                break
            trace_bytes += chunk
    finally:
        os.close(trace_fd)
    scratch_symlinks = [str(path.absolute()) for path in scratch_root.rglob("*")
                        if path.is_symlink()]
    shutil.rmtree(scratch_root)
    validate_subprocess_registry(
        repo, registry_rows, registry_sha256, runner_mode=RUNNER_MODE_ACTIVE,
        include_contract_fixture=include_contract_fixture)
    trace_text = trace_bytes.decode("utf-8", errors="replace")
    allowed = {str(runner.resolve()), str(Path(row["interpreter"]["path"]).resolve())}
    allowed.update(str(Path(item["path"]).resolve()) for item in row["runtime_closure"])
    allowed.update(str(path) for path in control_paths)
    allowed.update(str(path) for path in declared)
    access = audit_trace_access(
        trace_text, cwd=task_root,
        declared_read_paths=[Path(path) for path in sorted(allowed)],
        canonical_task_root=task_root, preexisting_paths=preexisting,
        declared_runtime_devices=row.get("runtime_devices", ()),
        declared_runtime_metadata_reads=row.get(
            "runtime_metadata_read_paths", ()),
        mutable_scratch_prefixes=("scratch",),
        reserved_write_prefixes=row["trace_access_policy"][
            "reserved_parent_output_prefixes"])
    for path in scratch_symlinks:
        access["violations"].append(
            {"path": path, "reason": "task scratch symlink rejected"})
    for path in [path for path in task_root.rglob("*") if path.is_symlink()]:
        access["violations"].append(
            {"path": str(path.absolute()), "reason": "post-run task-root symlink rejected"})
    access["valid"] = not access["violations"]
    reads = set(access["observed_read_paths"]); writes = set(access["observed_write_paths"])
    scratch_prefix = str(scratch_root) + "/"
    persistent_writes = {path for path in writes
                         if not path.startswith(scratch_prefix)}
    allowed_write_prefixes = tuple(str(task_root / name) + "/" for name in (
        "active", "production", "attempts", "evidence", "artifacts",
        "typed_failures", "outcomes"))
    trace_valid = (access["valid"] and str(runner.resolve()) in reads
                   and all(str(path) in reads for path in control_paths)
                   and all(str(path) in reads for path in declared)
                   and (persistent_writes == {str(output_path)} if fixture else
                        bool(persistent_writes)
                        and all(path.startswith(allowed_write_prefixes)
                                for path in persistent_writes)))
    try:
        output_bytes = read_no_symlink_bytes(output_path, "active runner output")
        output_row = no_symlink_file_row(output_path, "active runner output")
    except Fixed4SubprocessContractError:
        output_bytes = b""; output_row = None
    expected_bytes = (read_no_symlink_bytes(declared[0], "fixture input")
                      if fixture else None)
    expected_returncode = 0
    if not trace_valid:
        failure_type = "PARENT_OBSERVED_INPUT_CONTRACT_VIOLATION"
    elif completed.returncode != expected_returncode:
        failure_type = "ACTIVE_RUNNER_UNEXPECTED_EXIT"
    elif fixture and output_bytes != expected_bytes:
        failure_type = "ACTIVE_RUNNER_OUTPUT_CONTRACT_VIOLATION"
    elif not fixture and not _validate_production_wrapper_output(
            output_bytes=output_bytes, task=task, task_root=task_root,
            output_root=output_root, observed_write_paths=writes):
        failure_type = "ACTIVE_RUNNER_OUTPUT_CONTRACT_VIOLATION"
    else:
        failure_type = None
    stdout_row, _ = create_only_bytes_beneath(
        output_root, wrapper_root / "stdout.bin", completed.stdout)
    stderr_row, _ = create_only_bytes_beneath(
        output_root, wrapper_root / "stderr.bin", completed.stderr)
    trace_row = no_symlink_file_row(trace_path, "parent trace")
    declared_rows = [no_symlink_file_row(path, "declared stage input")
                     for path in declared]
    receipt = {
        "schema": ACTIVE_CONSUMPTION_SCHEMA,
        "task_id": task["task_id"], "task_payload_sha256": task["payload_sha256"],
        "subprocess_registry_closure_sha256": registry_sha256,
        "runner_mode": RUNNER_MODE_ACTIVE, "contract_fixture": fixture,
        "executable": row["interpreter"], "runner": row["runner"],
        "runtime_closure_sha256": row["runtime_closure_sha256"],
        "argv": argv, "environment": env, "cwd": str(task_root),
        "control_inputs": control_rows,
        "control_input_closure_sha256": stable_json_sha256(control_rows),
        "declared_stage_inputs": declared_rows,
        "declared_stage_input_closure_sha256": stable_json_sha256(declared_rows),
        "parent_observed_accesses": access,
        "parent_observed_accesses_sha256": stable_json_sha256(access),
        "returncode": completed.returncode, "runner_output": output_row,
        "stdout": stdout_row, "stderr": stderr_row, "strace": trace_row,
        "failure_type": failure_type,
        "operational_result_emitted": (not fixture and failure_type is None),
        "operational_result_release_allowed": (not fixture and failure_type is None),
        "runner_reported_failure_type_trusted": False,
    }
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    encoded = (json.dumps(receipt, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode("utf-8")
    create_only_bytes_beneath(
        output_root, wrapper_root / "active_consumption_receipt.json", encoded)
    return receipt


def classify_wrapper_failure(returncode: int, trace_valid: bool,
                             runner_stdout: bytes, runner_stderr: bytes) -> str | None:
    """Trusted classification; runner text is intentionally never interpreted."""
    del runner_stdout, runner_stderr
    if not trace_valid:
        return "PARENT_OBSERVED_INPUT_CONTRACT_VIOLATION"
    if returncode == DISABLED_EXIT_CODE:
        return "CHECKED_IN_RUNNER_EXECUTION_DISABLED"
    if returncode < 0:
        return "TRUSTED_WRAPPER_CHILD_SIGNAL"
    if returncode != 0:
        return "TRUSTED_WRAPPER_NONZERO_EXIT"
    return None


def task_execution_binding(registry_rows: Sequence[Mapping[str, Any]],
                           registry_sha256: str, stage: str,
                           task_id: str) -> dict[str, Any]:
    matches = [row for row in registry_rows if row.get("stage") == stage]
    if len(matches) != 1:
        raise Fixed4SubprocessContractError("task stage subprocess descriptor missing")
    row = matches[0]
    argv = [token.replace("{stage}", stage).replace("{task_id}", task_id)
            for token in row["argv_template"]]
    return {
        "subprocess_registry_closure_sha256": registry_sha256,
        "subprocess_registry_schema": row["schema"],
        "runner_mode": row["runner_mode"],
        "stage_implementation_status": row["stage_implementation_status"],
        "executable": row["interpreter"], "runner": row["runner"],
        "sealed_executor": row["sealed_executor"],
        "argv": argv, "environment": row["environment"],
        "cwd_contract": f"tasks/{task_id}",
        "trace_access_policy": row["trace_access_policy"],
        "input_contract": (["sealed_task_json", "sealed_authorization",
                            "sealed_preflight", "sealed_task_manifest",
                            "signed_production_execution_manifest",
                            "signed_production_input_manifest",
                            "signed_parent_result_payloads"]
                           if (row["runner_mode"] == RUNNER_MODE_ACTIVE
                               and stage != CONTRACT_FIXTURE_STAGE) else
                           ["sealed_task_json", "sealed_authorization",
                            "sealed_preflight", "sealed_task_manifest"]),
        "output_contract": (
            (["active/runner_output.bin", "wrapper/stdout.bin",
              "wrapper/stderr.bin", "wrapper/strace.log",
              "wrapper/active_consumption_receipt.json"]
             if stage == CONTRACT_FIXTURE_STAGE else
            ["active/runner_output.bin", "active/adapter_validation.json",
             "active/operational_result_candidate.json",
             "active/production_attempt.json", "production/**",
             "attempts/**", "evidence/**", "artifacts/**", "outcomes/**",
             "typed_failures/**", "wrapper/stdout.bin",
             "wrapper/stderr.bin", "wrapper/strace.log",
             "wrapper/active_consumption_receipt.json"])
            if row["runner_mode"] == RUNNER_MODE_ACTIVE else
            ["wrapper/stdout.bin", "wrapper/stderr.bin",
             "wrapper/strace.log", "wrapper/consumption_receipt.json"]),
        "expected_exit_code": row["expected_exit_code"],
        "parent_observed_consumption_required": True,
        "runner_reported_failure_type_trusted": False,
    }
