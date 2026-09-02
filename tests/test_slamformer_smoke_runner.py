from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_slamformer_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_slamformer_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _plan() -> dict:
    return {
        "input": {
            "frame_ids": [0, 1, 2],
            "timestamp_us": [0, 33_333, 66_666],
        },
    }


def test_trajectory_contract_accepts_sparse_manifest_keyframes(tmp_path: Path) -> None:
    trajectory = tmp_path / "final_traj.txt"
    trajectory.write_text(
        "0 0 0 0 0 0 0 1\n2 0.1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    result = RUNNER._trajectory_contract(trajectory, _plan(), "frame_id")

    assert result["row_count"] == 2
    assert result["first_identifier"] == 0.0
    assert result["last_identifier"] == 2.0


def test_trajectory_contract_rejects_unmanifested_identifier(tmp_path: Path) -> None:
    trajectory = tmp_path / "final_traj.txt"
    trajectory.write_text(
        "0 0 0 0 0 0 0 1\n9 0.1 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    with pytest.raises(RUNNER.SmokeContractError, match="outside the admitted"):
        RUNNER._trajectory_contract(trajectory, _plan(), "frame_id")


def test_staging_preserves_nonzero_manifest_frame_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"rgb")
    input_root = tmp_path / "staged"
    input_root.mkdir()

    rows = RUNNER._stage_rgb_create_only(input_root, [{
        "frame_id": 2334,
        "timestamp_us": 78_799_922,
        "color_path": str(source),
        "intrinsics": [574.5, 577.6, 322.5, 238.6],
        "rotate_ccw": False,
    }])

    staged = input_root / "rgb" / "002334.jpg"
    assert staged.is_symlink()
    assert staged.resolve() == source.resolve()
    assert rows[0]["frame_id"] == 2334
    assert rows[0]["staged_color_path"] == str(staged)


def test_ply_contract_requires_nonempty_xyz_payload(tmp_path: Path) -> None:
    cloud = tmp_path / "final.ply"
    cloud.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "0 0 0\n",
        encoding="ascii",
    )

    result = RUNNER._ply_contract(cloud)

    assert result["vertex_count"] == 1
    assert result["format"] == "ascii"
