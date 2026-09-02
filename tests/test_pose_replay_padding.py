from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

import numpy as np

from pose_pipeline.replay import _load_finalized_poses, _pad_rgbd_to_multiple


class PoseReplayPaddingTests(unittest.TestCase):
    def test_bottom_right_padding_preserves_original_pixels(self):
        color = np.full((224, 172, 3), 7, dtype=np.uint8)
        depth = np.full((224, 172), 1000, dtype=np.uint16)
        padded_color, padded_depth, audit = _pad_rgbd_to_multiple(color, depth)
        self.assertEqual(padded_depth.shape, (224, 176))
        self.assertEqual(padded_color.shape, (224, 176, 3))
        np.testing.assert_array_equal(padded_color[:, :172], color)
        np.testing.assert_array_equal(padded_depth[:, :172], depth)
        self.assertTrue(np.all(padded_color[:, 172:] == 0))
        self.assertTrue(np.all(padded_depth[:, 172:] == 0))
        self.assertEqual(audit["pad_right_px"], 4)
        self.assertEqual(audit["pad_bottom_px"], 0)

    def test_divisible_input_is_unchanged(self):
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.uint16)
        padded_color, padded_depth, audit = _pad_rgbd_to_multiple(color, depth)
        self.assertIs(padded_color, color)
        self.assertIs(padded_depth, depth)
        self.assertEqual(audit["pad_right_px"], 0)
        self.assertEqual(audit["pad_bottom_px"], 0)

    def test_invalid_multiple_fails(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _pad_rgbd_to_multiple(
                np.zeros((1, 1, 3), dtype=np.uint8),
                np.zeros((1, 1), dtype=np.uint16),
                0,
            )

    def test_finalized_pose_sidecar_is_gt_free_and_inverted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finalized.jsonl"
            t_camera_world = np.eye(4)
            t_camera_world[0, 3] = 2.0
            row = {
                "schema": "dpv_finalized_pose.v1",
                "frame_id": 7,
                "timestamp_us": 99,
                "T_camera_world_m": t_camera_world.reshape(-1).tolist(),
                "source": "DPV-SLAM:warmup_backfill",
                "finalized_at_frame": 21,
                "identity_fallback_used": False,
                "gt_consumed": False,
            }
            path.write_text(json.dumps(row) + "\n")
            poses = _load_finalized_poses(path)
            self.assertAlmostEqual(poses[7].t_world_camera[0, 3], -2.0)
            row["gt_consumed"] = True
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "GT-free"):
                _load_finalized_poses(path)


if __name__ == "__main__":
    unittest.main()
