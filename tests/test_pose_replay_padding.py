from __future__ import annotations

import unittest

import numpy as np

from pose_pipeline.replay import _pad_rgbd_to_multiple


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


if __name__ == "__main__":
    unittest.main()
