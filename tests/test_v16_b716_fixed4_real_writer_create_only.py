"""Real fixed4 stage writers must match the parent trace contract.

The Linux trace test is intentionally a smoke test over the production writer
functions, not a mock filesystem.  It can run in the remote execution checkout
without GPU/model assets; heavyweight inference is outside the writer boundary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

from safety.v13_dual_solver_runtime import atomic_json
from safety.v13_colorpcr_pointdsc_shadow import build_color_pair
from safety.v14_rigid_multihypothesis import (
    RigidMultiHypothesisError,
    build_direction_candidates,
)
from scripts.v13_colorpcr_official_worker import (
    write_npz_create_only as write_worker_npz,
)
from scripts.v13_colorpcr_sentinel_subprocess import (
    write_npz_create_only as write_sentinel_npz,
)
from scripts.v13_corr_cache_converter import convert, sha256_file
from scripts.v13_colorpcr_pointdsc_preflight import atomic_json as preflight_atomic_json


def _conversion_fixture(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / "prepared.npz"
    manifest = {
        "schema": "v13-color-preserving-pair-v2",
        "pair_id": "src_to_ref",
        "payload_sha256": "a" * 64,
    }
    np.savez(prepared, manifest_json=np.asarray(json.dumps(manifest)))
    rng = np.random.default_rng(1616)
    src = rng.normal(size=(80, 3)).astype(np.float32)
    ref = src + np.asarray([0.1, -0.03, 0.02], np.float32)
    scores = np.linspace(1.0, 0.1, len(src), dtype=np.float32)
    evidence_paths = {}
    evidence_hashes = {}
    for name in ("identity", "proper_nonzero"):
        path = root / f"{name}.npz"
        np.savez(path, marker=np.asarray(name))
        evidence_paths[name] = str(path)
        evidence_hashes[name] = sha256_file(path)
    meta = {
        "schema": "v13-colorpcr-corr-cache-v2",
        "sentinel_invariant": True,
        "gt_consumed": False,
        "identity_fallback": False,
        "input_sha256": sha256_file(prepared),
        "sentinel_artifact_path": evidence_paths,
        "sentinel_artifact_sha256": evidence_hashes,
        "worker_contract": {
            "arm": "sgf_selected_union",
            "direction": "forward",
            "neighbor_limits": [38, 36, 36, 38],
            "sampling": "voxel10",
            "coarsest_cap": 512,
        },
    }
    source = root / "sentinel.npz"
    np.savez(
        source,
        src_corr_points=src,
        ref_corr_points=ref,
        corr_scores=scores,
        estimated_transform=np.eye(4),
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)),
    )
    return prepared, source


def _pointdsc_fixture(root: Path) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    labels = np.arange(4, dtype=np.int32)
    colors = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], np.uint8)
    source = root / "source.npz"
    reference = root / "reference.npz"
    np.savez(source, xyz=xyz, labels=labels, colors=colors)
    np.savez(reference, xyz=xyz + [0.1, 0.0, 0.0], labels=labels, colors=colors)
    shadow = root / "shadow.npz"
    source_points = xyz[:3]
    reference_points = (xyz + [0.1, 0.0, 0.0])[:3]
    np.savez(
        shadow,
        source_points=source_points,
        reference_points=reference_points,
        forward_src_corr=source_points[:2],
        forward_ref_corr=reference_points[:2],
        forward_scores=np.ones(2, np.float32),
        reverse_src_corr=reference_points[:2],
        reverse_ref_corr=source_points[:2],
        reverse_scores=np.ones(2, np.float32),
    )
    return source, reference, shadow


def test_real_writers_refuse_existing_files_without_changing_bytes(tmp_path):
    existing = tmp_path / "sealed.json"
    existing.write_bytes(b"do-not-overwrite\n")
    before = existing.read_bytes()
    with pytest.raises(FileExistsError):
        atomic_json(existing, {"changed": True})
    assert existing.read_bytes() == before

    for writer in (write_worker_npz, write_sentinel_npz):
        output = tmp_path / f"{writer.__module__.split('.')[-1]}.npz"
        output.write_bytes(b"do-not-overwrite\n")
        with pytest.raises(FileExistsError):
            if writer is write_worker_npz:
                writer(output, value=np.arange(3))
            else:
                writer(output, {"value": np.arange(3)})
        assert output.read_bytes() == b"do-not-overwrite\n"

    prepared, source = _conversion_fixture(tmp_path / "converter-input")
    output = tmp_path / "converted.npz"
    output.write_bytes(b"do-not-overwrite\n")
    with pytest.raises(FileExistsError):
        convert(
            source, prepared, output, tmp_path / "receipt.json",
            pair_id="src_to_ref", arm="sgf_selected_union",
            direction="forward",
        )
    assert output.read_bytes() == b"do-not-overwrite\n"

    source_raw, reference_raw, shadow = _pointdsc_fixture(tmp_path / "pointdsc-input")
    pointdsc = tmp_path / "pointdsc.npz"
    pointdsc.write_bytes(b"do-not-overwrite\n")
    with pytest.raises(FileExistsError):
        build_color_pair(
            "source_to_reference", source_raw, reference_raw, shadow, pointdsc,
        )
    assert pointdsc.read_bytes() == b"do-not-overwrite\n"

    preflight = tmp_path / "preflight.json"
    preflight.write_bytes(b"do-not-overwrite\n")
    with pytest.raises(FileExistsError):
        preflight_atomic_json(preflight, {"changed": True})
    assert preflight.read_bytes() == b"do-not-overwrite\n"


def test_v14_manifest_is_bound_before_its_single_create(tmp_path):
    rng = np.random.default_rng(1414)
    src = rng.normal(size=(80, 3))
    ref = src + [0.1, -0.03, 0.02]
    cache = tmp_path / "cache.npz"
    np.savez(cache, src_corr=src, ref_corr=ref, scores=np.ones(80))
    preregister = (Path(__file__).resolve().parents[1]
                   / "manifests/v14_rigid_multihypothesis_preregister.json")
    output = tmp_path / "direction"
    value = build_direction_candidates(
        cache, output, pair_id="pair", arm="sgf_selected_union",
        direction="forward", preregister_path=preregister,
    )
    assert value["preregister_sha256"] == sha256_file(preregister)
    sealed = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_direction_candidates(
            cache, output, pair_id="pair", arm="sgf_selected_union",
            direction="forward", preregister_path=preregister,
        )
    assert (output / "manifest.json").read_bytes() == sealed


@pytest.mark.skipif(shutil.which("strace") is None,
                    reason="Linux strace is required for syscall evidence")
def test_real_writer_trace_has_only_authorized_create_only_writes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    prepared, source = _conversion_fixture(fixture)
    source_raw, reference_raw, shadow = _pointdsc_fixture(fixture / "pointdsc")
    production = tmp_path / "task-root" / "production"
    production.mkdir(parents=True)
    trace = tmp_path / "real-writers.strace"
    child = tmp_path / "trace_child.py"
    child.write_text(
        """
import json
import sys
from pathlib import Path
import numpy as np
from safety.v13_dual_solver_runtime import atomic_json
from safety.v13_colorpcr_pointdsc_shadow import build_color_pair
from safety.v14_rigid_multihypothesis import build_direction_candidates
from scripts.v13_colorpcr_official_worker import write_npz_create_only as worker_write
from scripts.v13_colorpcr_sentinel_subprocess import write_npz_create_only as sentinel_write
from scripts.v13_corr_cache_converter import convert
from scripts.v13_colorpcr_pointdsc_preflight import atomic_json as preflight_atomic_json
production, prepared, source, prereg, source_raw, reference_raw, shadow = map(Path, sys.argv[1:])
atomic_json(production / "dual_solver" / "worker.json", {"safe": False})
worker_write(production / "worker" / "worker.npz", value=np.arange(3))
sentinel_write(production / "sentinel" / "cache.npz", {"value": np.arange(3)})
convert(source, prepared, production / "converter" / "exact.npz",
        production / "converter" / "receipt.json", pair_id="src_to_ref",
        arm="sgf_selected_union", direction="forward")
build_direction_candidates(production / "converter" / "exact.npz",
        production / "v14", pair_id="pair", arm="sgf_selected_union",
        direction="forward", preregister_path=prereg)
build_color_pair("source_to_reference", source_raw, reference_raw, shadow,
        production / "pointdsc-shadow" / "prepared.npz")
preflight_atomic_json(production / "pointdsc-shadow" / "preflight.json",
        {"ready": False})
""",
        encoding="utf-8",
    )
    preregister = repo / "manifests/v14_rigid_multihypothesis_preregister.json"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "src"), str(repo / "scripts")])
    subprocess.run(
        [
            shutil.which("strace"), "-f", "-qq", "-s", "4096",
            "-e", "trace=open,openat,openat2,creat,rename,renameat,renameat2,"
                  "unlink,unlinkat,truncate,ftruncate",
            "-o", str(trace), sys.executable, str(child), str(production),
            str(prepared), str(source), str(preregister), str(source_raw),
            str(reference_raw), str(shadow),
        ],
        cwd=repo, env=environment, check=True,
    )
    lines = trace.read_text(errors="replace").splitlines()
    successful = [line for line in lines
                  if re.search(r"=\s*(?:0|[1-9][0-9]*)\s*$", line)]
    forbidden = [line for line in successful if re.search(
        r"\b(?:rename(?:at2?|)|unlink(?:at|)|truncate|ftruncate)\(", line)]
    assert forbidden == []

    writes = [line for line in successful
              if re.search(r"\b(?:open|openat|openat2|creat)\(", line)
              and re.search(r"O_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND)", line)]
    assert writes
    for line in writes:
        assert "O_CREAT" in line and "O_EXCL" in line, line
        quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', line)
        assert quoted, line
        raw = bytes(quoted[-1], "utf-8").decode("unicode_escape")
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        assert path.resolve().is_relative_to(production.resolve()), line
