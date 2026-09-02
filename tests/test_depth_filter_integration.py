from __future__ import annotations

from pathlib import Path
import importlib.util
import tempfile
import unittest

import numpy as np

from pose_pipeline.contracts import FrameRecord
from pose_pipeline.depth_filter import DepthFilterConfig
from pose_pipeline.replay import _read_frame
from pose_pipeline.submaps import _read_depth
from reconstruction.rgbd_refusion import _read_rgbd


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV runtime required")
class DepthFilterIntegrationTests(unittest.TestCase):
    def test_replay_submap_refusion_share_filtered_hash(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color_path = root / "color.png"
            depth_path = root / "depth.png"
            color = np.full((16, 16, 3), 127, dtype=np.uint8)
            depth = np.full((16, 16), 1000, dtype=np.uint16)
            depth[:, 8:] = 1120
            depth[3, 3] = 0
            depth[5, 5] = 1025
            self.assertTrue(cv2.imwrite(str(color_path), color))
            self.assertTrue(cv2.imwrite(str(depth_path), depth))
            frame = FrameRecord(
                frame_id=7,
                timestamp_us=11,
                color_path=color_path,
                depth_path=depth_path,
                intrinsics=(500.0, 500.0, 8.0, 8.0),
            )
            config = DepthFilterConfig.from_profile("bilateral_light_v1")
            replay = _read_frame(
                frame, 1000.0, config, return_filter_stats=True,
            )
            replay_depth, replay_stats = replay[1], replay[4]
            submap_depth, submap_stats = _read_depth(
                frame, 1000.0, config, return_filter_stats=True,
            )
            refusion = _read_rgbd(
                frame, 1000.0, config, return_filter_stats=True,
            )
            refusion_depth, refusion_stats = refusion[1], refusion[3]
            np.testing.assert_array_equal(replay_depth, submap_depth)
            np.testing.assert_array_equal(submap_depth, refusion_depth)
            self.assertEqual(
                replay_stats.filtered_sha256,
                submap_stats.filtered_sha256,
            )
            self.assertEqual(
                submap_stats.filtered_sha256,
                refusion_stats.filtered_sha256,
            )
            self.assertEqual(replay_depth[3, 3], 0)
            self.assertLess(abs(int(replay_depth[5, 5]) - 1000), 25)


if __name__ == "__main__":
    unittest.main()
