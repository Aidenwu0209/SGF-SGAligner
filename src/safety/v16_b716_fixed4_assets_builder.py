"""Reproducible production asset/runtime inventories for active fixed4.

This module only inventories already reviewed code and dependencies.  It does
not authorize, dispatch, execute, select, reconstruct, refusion, or evaluate
official92.  Every emitted document is create-only and hash sealed.

The stage manifests intentionally retain the existing production-adapter
schema.  Additional repository commit/diff evidence lives in the build
receipt, while the recursively verified directory closure remains the binding
consumed by the production adapter.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from safety.v13_dual_solver_runtime import sha256_file, stable_json_sha256
from safety.v16_b716_fixed4_execution_pilot import POLICY_FALSE_FIELDS
from safety.v16_b716_fixed4_subprocess_contract import parse_open_events
PRODUCTION_ASSETS_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-production-assets-manifest-v1")
PRODUCTION_RUNTIME_MANIFEST_SCHEMA = (
    "v16-b716-fixed4-production-runtime-manifest-v1")
ASSETS_BUILD_RECEIPT_SCHEMA = (
    "v16-b716-fixed4-production-assets-build-receipt-v1")
SOURCE_IDENTITY_SCHEMA = "v16-b716-fixed4-source-identity-v1"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

COLORPCR_COMMIT = "d579a80d71c3d6ae37ee58ba5a3943fe81e8427d"
COLORPCR_EXTENSION_SHA256 = (
    "33160b284931483c570b32fd73513ceb5603cce3d30d4e0903937ce30c8b594f")
COLORPCR_WEIGHT_SHA256 = (
    "b4900863c86629c24386189094691f159c1ff437b5623510a11c9468bc8cb814")
COLORPCR_PYTHON_TREE_SHA256 = (
    "26f732740d70433324f7e3a2368b9f7bf1670fb3e7a95945f80d7af6ae50958d")
POINTDSC_COMMIT = "b009d536ac10b570853833f2178397c154745da9"
POINTDSC_CHECKPOINT_SHA256 = (
    "20662778fca1a7d2c4e2f79f381d4be6cb891834d7bb4bd91ade9d89b0d13bd4")

DEFAULT_RUNTIME_SOURCE_FILES = (
    "src/safety/__init__.py",
    "src/safety/decision_features.py",
    "src/safety/v13_dual_solver_runtime.py",
    "src/safety/v13_fixed4_aggregate.py",
    "src/safety/v13_strict_pair_gate.py",
    "src/safety/v14_rigid_multihypothesis.py",
    "src/safety/v15_safe_pose_cluster.py",
    "src/safety/v16_b716_fixed4_active_production_wrapper.py",
    "src/safety/v16_b716_exact72_lineage_seal.py",
    "src/safety/v16_b716_fixed4_execution_pilot.py",
    "src/safety/v16_b716_fixed4_orchestrator_contract.py",
    "src/safety/v16_b716_fixed4_production_adapters.py",
    "src/safety/v16_b716_fixed4_stage_runners.py",
    "src/safety/v16_b716_fixed4_subprocess_contract.py",
    "src/safety/v16_safe_hypothesis_cluster.py",
)

DEFAULT_PROBE_MODULES = (
    "numpy", "scipy", "sklearn", "torch", "pygcransac",
)
ACTIVE_RUNTIME_PROBE_MODULES = (
    "safety.v16_b716_fixed4_active_production_wrapper",
    "scripts.v13_colorpcr_sentinel_subprocess",
    "scripts.v13_corr_cache_converter",
    "scripts.v14_fixed4_input_builder",
    "scripts.v14_candidate_strict_runner",
)
JOJO_RUNTIME_PROBE_MODULES = (
    "numpy", "torch", "skimage.color", "geotransformer.utils.common",
    "config", "model", "geotransformer.modules.ops",
    "scripts.v13_colorpcr_official_worker",
)
NO_PYCACHE_PREFIX = "/proc/v16-b716-fixed4-no-pyc"


class ProductionAssetsBuilderError(RuntimeError):
    """An input identity or create-only output failed closed."""


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = stable_json_sha256(result)
    return result


def _regular_file(path: Path, role: str, *, allow_symlink_name: bool = False) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ProductionAssetsBuilderError(f"{role} path must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProductionAssetsBuilderError(f"{role} missing") from exc
    if stat.S_ISLNK(mode):
        if not allow_symlink_name:
            raise ProductionAssetsBuilderError(f"{role} must not be a symlink")
        try:
            path = path.resolve(strict=True)
        except OSError as exc:
            raise ProductionAssetsBuilderError(f"{role} symlink is broken") from exc
        mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ProductionAssetsBuilderError(f"{role} must be a regular file")
    return path


def _real_directory(path: Path, role: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise ProductionAssetsBuilderError(f"{role} path must be absolute")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ProductionAssetsBuilderError(f"{role} missing") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ProductionAssetsBuilderError(
            f"{role} must be a non-symlink directory")
    return path


def file_row(path: Path, role: str, *, allow_symlink_name: bool = False
             ) -> dict[str, Any]:
    path = _regular_file(path, role, allow_symlink_name=allow_symlink_name)
    return {"role": role, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def dependency_row(path: Path, role: str) -> dict[str, Any]:
    path = _regular_file(path, role)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def directory_row(root: Path, role: str) -> dict[str, Any]:
    """Inventory the complete directory recursively; reject every symlink."""
    root = _real_directory(root, role)
    rows: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        dirnames.sort(); filenames.sort()
        for name in dirnames:
            path = base / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ProductionAssetsBuilderError(
                    f"{role} contains non-directory/symlink: {path}")
        for name in filenames:
            path = base / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ProductionAssetsBuilderError(
                    f"{role} contains non-regular/symlink: {path}")
            rows.append({"path": path.relative_to(root).as_posix(),
                         "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)})
    rows.sort(key=lambda row: row["path"])
    return {"role": role, "path": str(root), "files": rows,
            "closure_sha256": stable_json_sha256(rows)}


def _run(argv: Sequence[str], *, cwd: Path | None = None,
         input_bytes: bytes | None = None) -> bytes:
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin",
                   "LANG": "C", "LC_ALL": "C"}
    completed = subprocess.run(list(argv), cwd=cwd, env=environment,
        input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)
    if completed.returncode:
        raise ProductionAssetsBuilderError(
            f"identity command failed rc={completed.returncode}: {argv[0]}")
    return completed.stdout


def git_identity(root: Path, expected_commit: str) -> dict[str, Any]:
    """Bind a clean tracked Git checkout to commit/tree/diff/file inventory."""
    root = _real_directory(root, "Git source root")
    top = Path(_run(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
               .decode().strip())
    if top.resolve() != root.resolve():
        raise ProductionAssetsBuilderError("Git source root is not repository root")
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    if commit != expected_commit:
        raise ProductionAssetsBuilderError("Git commit drift")
    tree = _run(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"] \
                ).decode().strip()
    diff = _run(["git", "-C", str(root), "diff", "--binary", "HEAD", "--"])
    if diff:
        raise ProductionAssetsBuilderError("tracked Git worktree is dirty")
    tracked = _run(["git", "-C", str(root), "ls-files", "-s", "--"])
    return _sealed({"schema": SOURCE_IDENTITY_SCHEMA, "path": str(root),
        "commit": commit, "tree": tree,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_index_sha256": hashlib.sha256(tracked).hexdigest(),
        "tracked_source_modified": False})


def colorpcr_python_tree_sha256(repo: Path) -> str:
    repo = _real_directory(repo, "ColorPCR repository")
    digest = hashlib.sha256(); files: list[Path] = []
    for root in (repo / "geotransformer", repo / "experiments/ColorPCR"):
        _real_directory(root, "ColorPCR Python tree")
        files.extend(path for path in root.rglob("*.py")
                     if "__pycache__" not in path.parts)
    for path in sorted(files, key=lambda item: item.relative_to(repo).as_posix()):
        path = _regular_file(path, "ColorPCR Python source")
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _verify_expected(path: Path, role: str, expected_sha256: str,
                     *, allow_symlink_name: bool = False) -> dict[str, Any]:
    row = file_row(path, role, allow_symlink_name=allow_symlink_name)
    if row["sha256"] != expected_sha256:
        raise ProductionAssetsBuilderError(f"{role} SHA drift")
    return row


def _probe_python(interpreter: Path, modules: Iterable[str]) -> tuple[str, list[Path], list[str]]:
    interpreter = _regular_file(interpreter, "probe interpreter",
                                allow_symlink_name=True)
    # Import order is part of the real runtime contract.  In particular torch
    # must load libc10 before the ColorPCR native extension imports it.
    requested = list(dict.fromkeys(modules))
    program = r'''
import importlib, json, pathlib, sys
requested = json.loads(sys.stdin.read())
for name in requested:
    importlib.import_module(name)
rows = []
for module in tuple(sys.modules.values()):
    path = getattr(module, "__file__", None)
    if not isinstance(path, str):
        continue
    candidate = pathlib.Path(path)
    if candidate.suffix in {".pyc", ".pyo"} and "__pycache__" in candidate.parts:
        stem = candidate.name.split(".", 1)[0] + ".py"
        candidate = candidate.parent.parent / stem
    try:
        candidate = candidate.resolve(strict=True)
    except OSError:
        continue
    if candidate.is_file():
        rows.append(str(candidate))
print(json.dumps({"version": sys.version, "sys_path": sys.path,
                  "files": sorted(set(rows))}, sort_keys=True))
'''
    completed = subprocess.run([str(interpreter), "-I", "-s", "-B", "-c", program],
        input=json.dumps(requested).encode(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env={"PATH": "/usr/bin:/bin", "LANG": "C",
        "LC_ALL": "C", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
        check=False)
    if completed.returncode:
        raise ProductionAssetsBuilderError("Python dependency probe failed")
    try:
        value = json.loads(completed.stdout)
    except Exception as exc:
        raise ProductionAssetsBuilderError("Python dependency probe malformed") from exc
    paths = [_regular_file(Path(path), "probed Python dependency")
             for path in value["files"]]
    sys_path = []
    for item in value["sys_path"]:
        if not item:
            continue
        candidate = Path(item)
        # CPython commonly advertises a non-existent pythonXY.zip.  It is not
        # a consumable runtime path and must not enter the controlled closure.
        if not candidate.exists():
            continue
        path = _real_directory(candidate, "probed sys.path")
        sys_path.append(str(path))
    return str(value["version"]), paths, sys_path


def _probe_python_runtime_reads(
        interpreter: Path, modules: Iterable[str],
        controlled_sys_path: Sequence[str], *, cuda_probe: bool,
) -> tuple[list[Path], dict[str, Any]]:
    """Trace exact successful runtime-file reads under the production flags.

    The probe imports the same top-level modules used by the reviewed wrappers
    and performs a one-element CUDA allocation.  It never runs a model or
    consumes a dataset.  Character devices and proc/sys runtime metadata are
    bound by the independently signed subprocess registry rather than being
    misrepresented as immutable regular files.
    """
    interpreter = _regular_file(interpreter, "runtime trace interpreter",
                                allow_symlink_name=True)
    tracer = _regular_file(Path("/usr/bin/strace"), "runtime trace tool")
    # Keep the reviewed production import order.  Alphabetic sorting can load
    # the ColorPCR native extension before torch has made libc10 available,
    # even though the real worker imports torch first.
    requested = list(dict.fromkeys(modules))
    if not requested:
        return [], dependency_row(tracer, "runtime trace tool")
    program = r'''
import importlib, json, sys
config = json.loads(sys.stdin.read())
sys.path[:] = config["sys_path"]
for name in config["modules"]:
    if name == "config":
        # Official ColorPCR creates training output directories while importing
        # config.py.  Production inference suppresses only that unrelated side
        # effect, so the dependency probe must exercise the same import path.
        common = importlib.import_module("geotransformer.utils.common")
        original = common.ensure_dir
        common.ensure_dir = lambda _path: None
        try:
            importlib.import_module(name)
        finally:
            common.ensure_dir = original
    else:
        importlib.import_module(name)
if config["cuda_probe"]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("sealed CUDA runtime unavailable")
    tensor = torch.empty(1, device="cuda")
    tensor.cpu()
'''
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": NO_PYCACHE_PREFIX, "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "0", "CUDA_CACHE_DISABLE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    with tempfile.TemporaryDirectory(prefix="fixed4-runtime-trace-") as directory:
        trace_path = Path(directory) / "open.trace"
        completed = subprocess.run([
            str(tracer), "-f", "-qq", "-yy", "-s", "4096",
            "-e", "trace=open,openat,openat2,execve",
            "-o", str(trace_path), str(interpreter), "-I", "-S", "-B",
            "-X", f"pycache_prefix={NO_PYCACHE_PREFIX}", "-c", program,
        ], input=json.dumps({"modules": requested,
            "sys_path": list(controlled_sys_path),
            "cuda_probe": cuda_probe}, sort_keys=True).encode(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, check=False)
        if completed.returncode:
            detail = completed.stderr.decode(errors="replace").strip()
            if len(detail) > 1200:
                detail = detail[-1200:]
            raise ProductionAssetsBuilderError(
                "Python runtime read probe failed"
                + (f": {detail}" if detail else ""))
        trace_text = trace_path.read_text(errors="replace")
    paths: dict[str, Path] = {}
    for event in parse_open_events(trace_text):
        if not event["successful"] or event["access"] != "read":
            continue
        candidate = Path(str(event["path"]))
        if not candidate.is_absolute() or str(candidate).startswith((
                "/proc/", "/sys/", "/dev/")):
            continue
        try:
            candidate = candidate.resolve(strict=True)
            mode = candidate.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode):
            paths[str(candidate)] = candidate
    return [paths[key] for key in sorted(paths)], dependency_row(
        tracer, "runtime trace tool")


def _assets(stage: str, files: Sequence[Mapping[str, Any]],
            directories: Sequence[Mapping[str, Any]],
            parameters: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed({"schema": PRODUCTION_ASSETS_MANIFEST_SCHEMA,
        "stage": stage, "file_assets": list(files),
        "directory_assets": list(directories),
        "stage_parameters": dict(parameters), **POLICY_FALSE_FIELDS})


def build_documents(*, repo: Path, expected_repo_commit: str,
        colorpcr_repo: Path, colorpcr_weights: Path, colorpcr_extension: Path,
        pointdsc_root: Path, pointdsc_checkpoint: Path,
        sgaligner_python: Path, jojo_python: Path,
        preflight_manifest: Path,
        runtime_source_files: Sequence[str] = DEFAULT_RUNTIME_SOURCE_FILES,
        probe_modules: Sequence[str] = DEFAULT_PROBE_MODULES,
        color_device: str = "cuda:0") -> dict[str, dict[str, Any]]:
    """Build all stage asset manifests and the runtime manifest in memory."""
    repo = _real_directory(Path(repo), "active candidate repository")
    colorpcr_repo = _real_directory(Path(colorpcr_repo), "ColorPCR repository")
    pointdsc_root = _real_directory(Path(pointdsc_root), "PointDSC repository")
    source_identity = git_identity(repo, expected_repo_commit)
    color_identity = git_identity(colorpcr_repo, COLORPCR_COMMIT)
    point_identity = git_identity(pointdsc_root, POINTDSC_COMMIT)
    color_directory = directory_row(colorpcr_repo, "colorpcr_repo")
    point_directory = directory_row(pointdsc_root, "pointdsc_root")
    python_tree = colorpcr_python_tree_sha256(colorpcr_repo)
    if python_tree != COLORPCR_PYTHON_TREE_SHA256:
        raise ProductionAssetsBuilderError("ColorPCR Python tree drift")

    sgaligner_row = file_row(Path(sgaligner_python), "sgaligner_python",
                             allow_symlink_name=True)
    jojo_row = file_row(Path(jojo_python), "jojo_python",
                        allow_symlink_name=True)
    color_files = [sgaligner_row, jojo_row,
        file_row(repo / "scripts/v13_colorpcr_sentinel_subprocess.py",
                 "sentinel_subprocess"),
        file_row(repo / "scripts/v13_colorpcr_official_worker.py", "sentinel_worker"),
        file_row(repo / "scripts/v13_corr_cache_converter.py", "corr_converter"),
        _verify_expected(Path(colorpcr_weights), "weights", COLORPCR_WEIGHT_SHA256),
        _verify_expected(Path(colorpcr_extension), "extension",
                         COLORPCR_EXTENSION_SHA256)]
    color_parameters = {"colorpcr_direction": {
        "colorpcr_dependency_identity": {
            "commit": color_identity["commit"],
            "repo_closure_sha256": color_directory["closure_sha256"],
            "python_tree_sha256": python_tree,
            "tracked_diff_sha256": color_identity["tracked_diff_sha256"]},
        "arm": "sgf_selected_union", "device": color_device}}

    pilot_files = [
        {**sgaligner_row, "role": "python"},
        file_row(repo / "scripts/v14_fixed4_input_builder.py", "v14_builder"),
        file_row(repo / "scripts/v14_candidate_strict_runner.py",
                 "v14_strict_runner"),
        file_row(repo / "manifests/v13_colorpcr_pointdsc_fixed4_preregister.json",
                 "v13_preregister"),
        file_row(repo / "manifests/v14_rigid_multihypothesis_preregister.json",
                 "v14_preregister"),
        file_row(Path(preflight_manifest), "preflight_manifest"),
        _verify_expected(Path(pointdsc_checkpoint), "pointdsc_checkpoint",
                         POINTDSC_CHECKPOINT_SHA256)]

    runner = file_row(repo / "scripts/v16_b716_fixed4_active_stage_runner.sh",
                      "runner_source")
    wrapper_cli = file_row(
        repo / "scripts/v16_b716_fixed4_active_production_wrapper.py",
        "production_wrapper_cli")
    validator = file_row(
        repo / "src/safety/v16_b716_fixed4_active_production_wrapper.py",
        "validator_source")
    probe_modules = tuple(probe_modules)
    main_version, main_probe, main_sys_path = _probe_python(
        Path(sgaligner_python), probe_modules)
    _jojo_version, jojo_probe, jojo_sys_path = _probe_python(
        Path(jojo_python), ("numpy", "torch"))
    real_interpreter = _regular_file(Path(sgaligner_python), "runtime interpreter",
                                     allow_symlink_name=True)
    interpreter_prefix = real_interpreter.parent.parent
    # The top-level wrapper runs under ``python -I -S``.  The repository root
    # is needed by reviewed ``scripts.*`` imports and ``repo/src`` is needed
    # by ``safety.*``.  Both precede the interpreter-owned standard library and
    # site-packages; no editable .pth from an ambient checkout is processed.
    controlled_sys_path = [str(repo), str(repo / "src"), str(repo / "scripts")]
    for item in main_sys_path:
        path = Path(item)
        try:
            path.relative_to(interpreter_prefix)
        except ValueError:
            continue
        if item not in controlled_sys_path:
            controlled_sys_path.append(item)
    jojo_real = _regular_file(Path(jojo_python), "jojo interpreter",
                              allow_symlink_name=True)
    jojo_prefix = jojo_real.parent.parent
    jojo_controlled_sys_path = [str(repo), str(repo / "src"),
                                str(repo / "scripts"), str(colorpcr_repo),
                                str(colorpcr_repo / "experiments/ColorPCR")]
    for item in jojo_sys_path:
        try:
            Path(item).relative_to(jojo_prefix)
        except ValueError:
            continue
        if item not in jojo_controlled_sys_path:
            jojo_controlled_sys_path.append(item)
    traced: list[Path] = []
    trace_tool: dict[str, Any] | None = None
    if probe_modules:
        main_traced, trace_tool = _probe_python_runtime_reads(
            Path(sgaligner_python), (*probe_modules,
                *ACTIVE_RUNTIME_PROBE_MODULES), controlled_sys_path,
            cuda_probe=True)
        jojo_traced, jojo_trace_tool = _probe_python_runtime_reads(
            Path(jojo_python), JOJO_RUNTIME_PROBE_MODULES,
            jojo_controlled_sys_path, cuda_probe=True)
        if trace_tool != jojo_trace_tool:
            raise ProductionAssetsBuilderError("runtime trace tool identity drift")
        traced = [*main_traced, *jojo_traced]
    dependencies: dict[str, dict[str, Any]] = {}
    for relative in runtime_source_files:
        path = _regular_file(repo / relative, f"runtime source {relative}")
        dependencies[str(path)] = dependency_row(path, f"runtime source {relative}")
    for path in [*main_probe, *jojo_probe, *traced]:
        dependencies[str(path)] = dependency_row(path, "probed runtime dependency")
    ordered_dependencies = [dependencies[key] for key in sorted(dependencies)]
    runtime = _sealed({"schema": PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
        "interpreter": {"path": str(Path(sgaligner_python)),
            "realpath": str(real_interpreter), "bytes": real_interpreter.stat().st_size,
            "sha256": sha256_file(real_interpreter), "version": main_version},
        "runtime_dependency_files": ordered_dependencies,
        "controlled_sys_path": controlled_sys_path,
        "environment": {"PATH": "/home/aidenwu/miniconda3/envs/sgaligner/bin:"
                                "/home/aidenwu/miniconda3/envs/jojo2026/bin:"
                                "/usr/bin:/bin",
            "LANG": "C", "LC_ALL": "C", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/proc/v16-b716-fixed4-no-pyc",
            "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_CACHE_DISABLE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "runner_source": runner, "production_wrapper_cli": wrapper_cli,
        "validator_source": validator, **POLICY_FALSE_FIELDS})

    documents = {
        "colorpcr_direction": _assets("colorpcr_direction", color_files,
            [color_directory], color_parameters),
        "bidirectional_multi_solver_pilot": _assets(
            "bidirectional_multi_solver_pilot", pilot_files, [point_directory],
            {"bidirectional_multi_solver_pilot": {"arm": "sgf_selected_union"}}),
        "v16_pair_hypothesis_cluster": _assets(
            "v16_pair_hypothesis_cluster", [], [],
            {"v16_pair_hypothesis_cluster": {}}),
        "fixed4_aggregate": _assets("fixed4_aggregate", [], [],
            {"fixed4_aggregate": {}}),
        "runtime": runtime,
    }
    documents["receipt"] = _sealed({"schema": ASSETS_BUILD_RECEIPT_SCHEMA,
        "source_identity": source_identity,
        "colorpcr_identity": color_identity,
        "pointdsc_identity": point_identity,
        "colorpcr_directory_closure_sha256": color_directory["closure_sha256"],
        "pointdsc_directory_closure_sha256": point_directory["closure_sha256"],
        "colorpcr_extension_sha256": COLORPCR_EXTENSION_SHA256,
        "colorpcr_weight_sha256": COLORPCR_WEIGHT_SHA256,
        "pointdsc_checkpoint_sha256": POINTDSC_CHECKPOINT_SHA256,
        "sgaligner_interpreter_realpath": str(real_interpreter),
        "sgaligner_interpreter_sha256": sha256_file(real_interpreter),
        "jojo_interpreter_realpath": jojo_row["path"],
        "jojo_interpreter_sha256": jojo_row["sha256"],
        "runner_source_sha256": runner["sha256"],
        "production_wrapper_cli_sha256": wrapper_cli["sha256"],
        "validator_source_sha256": validator["sha256"],
        "runtime_import_trace_performed": bool(probe_modules),
        "runtime_import_trace_tool": trace_tool,
        "runtime_import_trace_dependency_closure_sha256":
            stable_json_sha256([dependency_row(path, "runtime trace dependency")
                                for path in sorted(set(traced))]),
        "manifest_payload_sha256s": {
            key: value["payload_sha256"] for key, value in documents.items()},
        "create_only": True, "recursive_closures": True,
        "symlinks_in_directory_closures_allowed": False,
        **POLICY_FALSE_FIELDS})
    return documents


def _write_create_only(root: Path, name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    root = _real_directory(root, "assets output root")
    if Path(name).name != name or name in {".", ".."}:
        raise ProductionAssetsBuilderError("output filename invalid")
    path = root / name
    if path.exists() or path.is_symlink():
        raise ProductionAssetsBuilderError("create-only output exists")
    encoded = (json.dumps(value, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise ProductionAssetsBuilderError("short create-only write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    observed = _regular_file(path, "created manifest")
    if observed.read_bytes() != encoded:
        raise ProductionAssetsBuilderError("created manifest post-write mismatch")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "payload_sha256": value["payload_sha256"]}


def materialize_documents(output_dir: Path,
                          documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Write a complete bundle once; never overwrite or resume."""
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise ProductionAssetsBuilderError("assets output directory already exists")
    parent = _real_directory(output_dir.parent.resolve(), "assets output parent")
    os.mkdir(parent / output_dir.name, 0o700)
    output_dir = parent / output_dir.name
    filenames = {
        "colorpcr_direction": "colorpcr_direction_assets.json",
        "bidirectional_multi_solver_pilot": "pilot_assets.json",
        "v16_pair_hypothesis_cluster": "pair_gate_assets.json",
        "fixed4_aggregate": "aggregate_gate_assets.json",
        "runtime": "production_runtime.json",
        "receipt": "build_receipt.json",
    }
    if set(documents) != set(filenames):
        raise ProductionAssetsBuilderError("document inventory mismatch")
    rows = {key: _write_create_only(output_dir, filenames[key], documents[key])
            for key in filenames}
    return {"output_dir": str(output_dir), "files": rows}
