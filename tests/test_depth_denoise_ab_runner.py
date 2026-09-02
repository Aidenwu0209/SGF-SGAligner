from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord,
    PoseRecord,
    SequenceManifest,
    write_manifest,
    write_trajectory,
)


REQUIRED = ("cv2", "open3d", "matplotlib", "plyfile", "sklearn")


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_depth_denoise_ab.py"
    spec = importlib.util.spec_from_file_location("run_depth_denoise_ab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in REQUIRED),
    "full geometry runtime required",
)
class DepthDenoiseABRunnerTests(unittest.TestCase):
    def test_synthetic_fixed_trajectory_ab_is_create_only(self):
        import cv2

        runner = _load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color_path = root / "color.png"
            depth_path = root / "depth.png"
            color = np.zeros((96, 96, 3), dtype=np.uint8)
            color[..., 1] = 180
            depth = np.full((96, 96), 1000, dtype=np.uint16)
            depth[:, 48:] = 1100
            depth[20, 20] = 1020
            depth[30, 30] = 0
            self.assertTrue(cv2.imwrite(str(color_path), color))
            self.assertTrue(cv2.imwrite(str(depth_path), depth))
            frame = FrameRecord(
                0, 0, color_path, depth_path,
                (100.0, 100.0, 47.5, 47.5),
            )
            manifest_path = root / "manifest.json"
            trajectory_path = root / "trajectory.json"
            write_manifest(manifest_path, SequenceManifest(
                "scannet", "synthetic_dual_plane", root, 1000.0,
                (frame,), "unit_test",
            ))
            write_trajectory(
                trajectory_path, [PoseRecord(0, 0, np.eye(4))],
                sequence_id="synthetic_dual_plane", arm="baseline",
            )
            output = root / "ab"
            result = runner.run_ab(
                manifest_path=manifest_path,
                trajectory_path=trajectory_path,
                profiles=(
                    "off", "range_v1", "bilateral_light_v1",
                    "bilateral_medium_v1",
                ),
                output_dir=output,
                include_sor_diagnostic=False,
            )
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["production_default_changed"])
            self.assertEqual(result["production_default"], "off")
            self.assertTrue((output / "source_sha256.json").is_file())
            self.assertTrue((output / "MANIFEST.sha256").is_file())
            for profile in result["profiles"]:
                report = json.loads((
                    output / profile / "refusion" / "refusion_result.json"
                ).read_text())
                self.assertEqual(report["integrated_frame_count"], 1)
                self.assertFalse(report["identity_fallback_used"])
                self.assertFalse(report["gt_consumed"])
            with self.assertRaises(FileExistsError):
                runner.run_ab(
                    manifest_path=manifest_path,
                    trajectory_path=trajectory_path,
                    profiles=("off", "range_v1"),
                    output_dir=output,
                    include_sor_diagnostic=False,
                )


if __name__ == "__main__":
    unittest.main()
