from pathlib import Path
import os
import stat

import pytest

from safety.v16_b716_fixed4_subprocess_contract import (
    RUNNER_MODE_ACTIVE,
    RUNNER_MODE_DISABLED,
    SUBPROCESS_REGISTRY_SCHEMA,
    Fixed4SubprocessContractError,
    audit_trace_access,
    build_subprocess_registry,
    parse_consumed_paths,
    parse_open_events,
    validate_trace_access,
)


def _open(path: Path, flags: str, fd: int = 3) -> str:
    return f'11 openat(AT_FDCWD, "{path}", {flags}) = {fd}\n'


def test_open_trace_keeps_read_write_flags_and_compatibility_view(tmp_path):
    declared = tmp_path / "declared.json"
    output = tmp_path / "tasks/t1/artifacts/result.json"
    missing = tmp_path / "missing.json"
    trace = (_open(declared, "O_RDONLY|O_CLOEXEC")
             + _open(output, "O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC", 4)
             + _open(missing, "O_RDONLY", -1))
    events = parse_open_events(trace)
    assert [event["access"] for event in events] == ["read", "write", "read"]
    assert [event["successful"] for event in events] == [True, True, False]
    assert parse_consumed_paths(trace) == [str(declared), str(output)]


def test_open_trace_prefers_strace_yy_resolved_descriptor_path(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    declared = task_root / "declared.json"
    declared.write_text("{}\n")
    trace = (
        f'11 openat(AT_FDCWD<{task_root}>, "declared.json", '
        f'O_RDONLY|O_CLOEXEC) = 3<{declared}>\n'
        '11 openat(AT_FDCWD</>, "/dev/null", O_WRONLY|O_CLOEXEC) '
        '= 4</dev/null<char 1:3>>\n'
    )
    events = parse_open_events(trace)
    assert [event["requested_path"] for event in events] == [
        "declared.json", "/dev/null"]
    assert [event["path"] for event in events] == [
        str(declared), "/dev/null"]
    assert parse_consumed_paths(trace) == [str(declared), "/dev/null"]


def test_declared_read_and_create_exclusive_task_write_pass(tmp_path):
    task_root = tmp_path / "tasks/t1"
    artifacts = task_root / "artifacts"
    artifacts.mkdir(parents=True)
    declared = tmp_path / "sealed-input.json"
    declared.write_text("{}\n")
    output = artifacts / "result.json"
    trace = (_open(declared, "O_RDONLY|O_CLOEXEC")
             + _open(output, "O_WRONLY|O_CREAT|O_EXCL", 4))
    report = validate_trace_access(
        trace_text=trace, cwd=task_root, declared_read_paths=[declared],
        canonical_task_root=task_root, preexisting_paths=[declared])
    assert report["valid"] is True
    assert report["observed_read_paths"] == [str(declared)]
    assert report["observed_write_paths"] == [str(output)]


def test_declared_loader_parent_components_normalize_before_allowlist(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    dynload = runtime / "lib/python3.11/lib-dynload"
    dynload.mkdir(parents=True)
    declared = runtime / "lib/libz.so.1"
    declared.write_bytes(b"sealed-runtime-dependency\n")
    traced = dynload / "../../libz.so.1"
    report = validate_trace_access(
        trace_text=_open(traced, "O_RDONLY|O_CLOEXEC"), cwd=task_root,
        declared_read_paths=[declared], canonical_task_root=task_root)
    assert report["valid"] is True
    assert report["observed_read_paths"] == [str(declared)]


def test_normalized_loader_parent_components_cannot_escape_allowlist(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    declared = tmp_path / "runtime/lib/libz.so.1"
    declared.parent.mkdir(parents=True)
    declared.write_bytes(b"sealed-runtime-dependency\n")
    escaped = tmp_path / "runtime/lib/python3.11/lib-dynload/../../../../secret"
    report = audit_trace_access(
        _open(escaped, "O_RDONLY|O_CLOEXEC"), cwd=task_root,
        declared_read_paths=[declared], canonical_task_root=task_root)
    assert report["valid"] is False
    assert report["violations"] == [{
        "path": str(tmp_path / "secret"), "reason": "undeclared read"}]


@pytest.mark.parametrize("raw", ["relative.json", "../escape.json", "./local.json"])
def test_relative_trace_paths_remain_fail_closed(tmp_path, raw):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    report = audit_trace_access(
        _open(Path(raw), "O_RDONLY"), cwd=task_root,
        declared_read_paths=[], canonical_task_root=task_root)
    assert report["valid"] is False
    assert report["violations"][0]["reason"] == "trace path is not canonical"


@pytest.mark.parametrize("flags", [
    "O_RDONLY",
    "O_WRONLY|O_TRUNC",
    "O_RDWR|O_CREAT",
    "O_WRONLY|O_CREAT|O_EXCL|O_APPEND",
])
def test_undeclared_or_non_create_only_access_fails_closed(tmp_path, flags):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    path = ((tmp_path / "undeclared.json") if flags == "O_RDONLY"
            else (task_root / "result.json"))
    with pytest.raises(Fixed4SubprocessContractError,
                       match="runner filesystem access rejected"):
        validate_trace_access(
            trace_text=_open(path, flags), cwd=task_root,
            declared_read_paths=[], canonical_task_root=task_root)


def test_write_escape_existing_leaf_and_reserved_wrapper_are_rejected(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    existing = task_root / "existing.json"
    existing.write_text("do-not-overwrite\n")
    outside = tmp_path / "outside.json"
    reserved = task_root / "wrapper/forged.json"
    for path, preexisting in ((outside, []), (existing, [existing]), (reserved, [])):
        report = audit_trace_access(
            _open(path, "O_WRONLY|O_CREAT|O_EXCL"), cwd=task_root,
            declared_read_paths=[], canonical_task_root=task_root,
            preexisting_paths=preexisting)
        assert report["valid"] is False
    assert existing.read_text() == "do-not-overwrite\n"


def test_symlink_escape_and_rename_bypass_are_rejected(tmp_path):
    task_root = tmp_path / "tasks/t1"
    outside = tmp_path / "outside"
    task_root.mkdir(parents=True)
    outside.mkdir()
    (task_root / "escape").symlink_to(outside, target_is_directory=True)
    escaped = task_root / "escape/result.json"
    trace = (_open(escaped, "O_WRONLY|O_CREAT|O_EXCL")
             + f'11 rename("{task_root}/new.tmp", "{task_root}/result.json") = 0\n')
    report = audit_trace_access(
        trace, cwd=task_root, declared_read_paths=[],
        canonical_task_root=task_root)
    reasons = {item["reason"] for item in report["violations"]}
    assert "write traverses symlink" in reasons
    assert "prohibited mutation syscall: rename" in reasons


def test_declared_read_symlink_cannot_expand_authority(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": true}\n')
    alias = tmp_path / "declared-alias.json"
    alias.symlink_to(secret)
    report = audit_trace_access(
        _open(alias, "O_RDONLY"), cwd=task_root,
        declared_read_paths=[alias], canonical_task_root=task_root)
    assert report["valid"] is False
    assert report["violations"] == [{
        "path": str(alias), "reason": "declared read traverses symlink"}]


def test_directory_metadata_open_is_limited_to_declared_descendant(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    package = tmp_path / "runtime/pkg"
    declared = package / "module.py"
    declared.parent.mkdir(parents=True)
    declared.write_text("x = 1\n")
    accepted = audit_trace_access(
        _open(package, "O_RDONLY|O_DIRECTORY|O_CLOEXEC"), cwd=task_root,
        declared_read_paths=[declared], canonical_task_root=task_root)
    assert accepted["valid"] is True
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    rejected = audit_trace_access(
        _open(unrelated, "O_RDONLY|O_DIRECTORY|O_CLOEXEC"), cwd=task_root,
        declared_read_paths=[declared], canonical_task_root=task_root)
    assert rejected["valid"] is False


def test_exact_runtime_device_and_own_thread_metadata_are_allowed(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    device = Path("/dev/urandom")
    observed = device.lstat()
    assert stat.S_ISCHR(observed.st_mode)
    row = {"path": str(device), "major": os.major(observed.st_rdev),
           "minor": os.minor(observed.st_rdev), "allowed_access": "read"}
    trace = (_open(device, "O_RDONLY|O_CLOEXEC")
             + _open(Path("/proc/self/task/123/comm"),
                     "O_WRONLY|O_CREAT|O_TRUNC", 4))
    report = audit_trace_access(
        trace, cwd=task_root, declared_read_paths=[],
        canonical_task_root=task_root, declared_runtime_devices=[row])
    assert report["valid"] is True
    assert report["observed_runtime_device_paths"] == [str(device)]
    assert report["observed_runtime_metadata_paths"] == [
        "/proc/self/task/123/comm"]


def test_declared_read_write_runtime_device_accepts_write_only_open(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    device = Path("/dev/null")
    observed = device.lstat()
    assert stat.S_ISCHR(observed.st_mode)
    row = {"path": str(device), "major": os.major(observed.st_rdev),
           "minor": os.minor(observed.st_rdev),
           "allowed_access": "read_write"}
    report = audit_trace_access(
        _open(device, "O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC"), cwd=task_root,
        declared_read_paths=[], canonical_task_root=task_root,
        declared_runtime_devices=[row])
    assert report["valid"] is True
    assert report["observed_runtime_device_paths"] == [str(device)]


def test_resolved_proc_alias_uses_exact_requested_metadata_authority(tmp_path):
    task_root = tmp_path / "tasks/t1"
    task_root.mkdir(parents=True)
    trace = (
        '11 openat(AT_FDCWD</>, "/proc/self/fd", '
        'O_RDONLY|O_DIRECTORY|O_CLOEXEC) = 3</proc/456/fd>\n'
        '11 openat(AT_FDCWD</>, "/proc/self/task/123/comm", '
        'O_WRONLY|O_TRUNC|O_CLOEXEC) = 4</proc/456/task/123/comm>\n'
    )
    report = audit_trace_access(
        trace, cwd=task_root, declared_read_paths=[],
        canonical_task_root=task_root,
        declared_runtime_metadata_reads=["/proc/self/fd"])
    assert report["valid"] is True
    assert report["observed_runtime_metadata_paths"] == [
        "/proc/self/fd", "/proc/self/task/123/comm"]


def test_task_local_scratch_is_mutable_but_links_and_outside_mutation_fail(tmp_path):
    task_root = tmp_path / "tasks/t1"
    scratch = task_root / "scratch"
    scratch.mkdir(parents=True)
    temporary = scratch / "compiler.tmp"
    accepted = (_open(temporary, "O_WRONLY|O_CREAT|O_EXCL", 4)
                + _open(temporary, "O_RDWR", 4)
                + f'11 unlink("{temporary}") = 0\n')
    report = audit_trace_access(
        accepted, cwd=task_root, declared_read_paths=[],
        canonical_task_root=task_root,
        mutable_scratch_prefixes=("scratch",))
    assert report["valid"] is True
    outside = tmp_path / "outside"
    rejected = (f'11 link("{temporary}", "{scratch}/linked") = 0\n'
                + f'11 unlink("{outside}") = 0\n')
    report = audit_trace_access(
        rejected, cwd=task_root, declared_read_paths=[],
        canonical_task_root=task_root,
        mutable_scratch_prefixes=("scratch",))
    reasons = [row["reason"] for row in report["violations"]]
    assert "prohibited mutation syscall: link" in reasons
    assert "prohibited mutation syscall: unlink" in reasons


def test_runner_modes_are_explicit_and_unknown_mode_is_rejected(tmp_path):
    assert SUBPROCESS_REGISTRY_SCHEMA.endswith("registry-v3")
    assert RUNNER_MODE_DISABLED == "disabled"
    with pytest.raises(Fixed4SubprocessContractError,
                       match="unknown subprocess runner mode"):
        build_subprocess_registry(tmp_path, runner_mode="auto")
