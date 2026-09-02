from __future__ import annotations

import unittest

import numpy as np

from pose_pipeline.depth_filter import DepthFilterConfig, apply_depth_filter


class DepthFilterTests(unittest.TestCase):
    def test_off_is_byte_exact_and_preserves_contract(self):
        depth = np.arange(63, dtype=np.uint16).reshape(7, 9)
        filtered, stats = apply_depth_filter(
            depth, 1000.0, DepthFilterConfig.from_profile("off"),
        )
        self.assertEqual(filtered.dtype, np.uint16)
        self.assertEqual(filtered.shape, depth.shape)
        self.assertEqual(filtered.tobytes(), depth.tobytes())
        self.assertEqual(stats.changed_pixels, 0)
        self.assertEqual(stats.input_sha256, stats.filtered_sha256)

    def test_range_profile_clips_only_outside_inclusive_bounds(self):
        depth = np.asarray([[0, 299, 300, 4500, 4501]], dtype=np.uint16)
        filtered, stats = apply_depth_filter(
            depth, 1000.0, DepthFilterConfig.from_profile("range_v1"),
        )
        np.testing.assert_array_equal(
            filtered, np.asarray([[0, 0, 300, 4500, 0]], dtype=np.uint16),
        )
        self.assertEqual(stats.clipped_below_pixels, 1)
        self.assertEqual(stats.clipped_above_pixels, 1)

    def test_bilateral_never_fills_invalid_pixels(self):
        depth = np.full((9, 9), 1000, dtype=np.uint16)
        depth[4, 4] = 0
        filtered, _stats = apply_depth_filter(
            depth, 1000.0,
            DepthFilterConfig.from_profile("bilateral_light_v1"),
        )
        self.assertEqual(filtered[4, 4], 0)
        np.testing.assert_array_equal(filtered == 0, depth == 0)

    def test_bilateral_reduces_planar_impulse(self):
        depth = np.full((11, 11), 1000, dtype=np.uint16)
        depth[5, 5] = 1020
        filtered, _stats = apply_depth_filter(
            depth, 1000.0,
            DepthFilterConfig.from_profile("bilateral_medium_v1"),
        )
        self.assertLess(abs(int(filtered[5, 5]) - 1000), 20)

    def test_fifty_millimetre_edge_is_retained(self):
        depth = np.full((24, 24), 1000, dtype=np.uint16)
        depth[:, 12:] = 1100
        _filtered, stats = apply_depth_filter(
            depth, 1000.0,
            DepthFilterConfig.from_profile("bilateral_medium_v1"),
        )
        self.assertGreater(stats.strong_edge_pairs, 0)
        self.assertGreaterEqual(
            stats.retained_strong_edge_pairs / stats.strong_edge_pairs,
            0.995,
        )

    def test_depth_scale_and_hash_are_deterministic(self):
        depth = np.full((13, 17), 5000, dtype=np.uint16)
        depth[6, 8] = 5100
        config = DepthFilterConfig.from_profile("bilateral_light_v1")
        first, first_stats = apply_depth_filter(depth, 5000.0, config)
        second, second_stats = apply_depth_filter(depth, 5000.0, config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            first_stats.filtered_sha256, second_stats.filtered_sha256,
        )


if __name__ == "__main__":
    unittest.main()
