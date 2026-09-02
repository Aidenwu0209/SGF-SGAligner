from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_mapanything_rgbd_window.py"
SPEC = importlib.util.spec_from_file_location("run_mapanything_rgbd_window", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_prepare_raw_views_uses_official_scalar_metric_default(tmp_path: Path) -> None:
    color_root = tmp_path / "color"
    depth_root = tmp_path / "depth"
    color_root.mkdir()
    depth_root.mkdir()
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(color_root / "0.jpg")
    Image.fromarray(np.full((1, 2), 1250, dtype=np.uint16)).save(depth_root / "0.png")

    intrinsic = np.asarray(
        [[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    views = RUNNER.prepare_raw_views(tmp_path, [0], intrinsic, 1000.0)

    assert len(views) == 1
    assert set(views[0]) == {"img", "intrinsics", "depth_z"}
    assert "is_metric_scale" not in views[0]
    assert views[0]["depth_z"].shape == (2, 3)
    np.testing.assert_allclose(views[0]["depth_z"], 1.25)
    np.testing.assert_array_equal(views[0]["intrinsics"], intrinsic)


def test_prepare_raw_views_rejects_invalid_depth_scale(tmp_path: Path) -> None:
    intrinsic = np.eye(3, dtype=np.float32)
    for depth_scale in (0.0, -1.0, float("nan"), float("inf")):
        try:
            RUNNER.prepare_raw_views(tmp_path, [0], intrinsic, depth_scale)
        except ValueError as exc:
            assert "depth scale" in str(exc)
        else:
            raise AssertionError(f"invalid depth scale accepted: {depth_scale}")


def test_prepare_raw_views_adds_pose_conditioning(tmp_path: Path) -> None:
    color_root = tmp_path / "color"
    depth_root = tmp_path / "depth"
    color_root.mkdir()
    depth_root.mkdir()
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(color_root / "7.jpg")
    Image.fromarray(np.full((2, 3), 1000, dtype=np.uint16)).save(depth_root / "7.png")
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 0.25

    views = RUNNER.prepare_raw_views(
        tmp_path, [7], np.eye(3, dtype=np.float32), 1000.0, {7: pose},
    )

    assert set(views[0]) == {"img", "intrinsics", "depth_z", "camera_poses"}
    np.testing.assert_array_equal(views[0]["camera_poses"], pose)
