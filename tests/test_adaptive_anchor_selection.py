from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.contracts import FrameRecord, PoseRecord
from pose_pipeline.submaps import (
    AdaptiveAnchorConfig,
    audit_anchor_schedule,
    geometric_reprojection_stats,
    select_adaptive_anchor_ordinals,
)


def _bound(translations: list[float]) -> list[tuple[FrameRecord, PoseRecord]]:
    rows = []
    for index, translation in enumerate(translations):
        frame = FrameRecord(
            frame_id=index,
            timestamp_us=index,
            color_path=f"color-{index}",
            depth_path=f"depth-{index}",
            intrinsics=(100.0, 100.0, 31.5, 23.5),
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = translation
        rows.append((
            frame,
            PoseRecord(index, index, transform, source="test"),
        ))
    return rows


class AdaptiveAnchorSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = np.full((48, 64), 1000, dtype=np.uint16)

    def test_metric_translation_produces_expected_pixel_flow(self):
        rows = _bound([0.0, 0.30])
        stats = geometric_reprojection_stats(
            rows[0], rows[1], 1000.0,
            AdaptiveAnchorConfig(
                minimum_gap=1,
                maximum_gap=10,
                pixel_stride=2,
                minimum_valid_depth_samples=100,
            ),
            anchor_depth=self.depth,
            current_depth=self.depth,
        )
        self.assertAlmostEqual(stats["relative_translation_m"], 0.30)
        self.assertAlmostEqual(stats["median_flow_px"], 30.0, places=6)
        self.assertLess(stats["in_bounds_fraction"], 1.0)

    def test_quiet_sequence_uses_bounded_gap_and_endpoint(self):
        rows = _bound([0.0] * 6)
        config = AdaptiveAnchorConfig(
            minimum_gap=2,
            maximum_gap=3,
            pixel_stride=2,
            flow_threshold_px=100.0,
            minimum_overlap_fraction=0.0,
            translation_threshold_m=10.0,
            rotation_threshold_deg=90.0,
            minimum_valid_depth_samples=100,
        )
        with patch("pose_pipeline.submaps._read_depth", return_value=self.depth):
            anchors, evidence = select_adaptive_anchor_ordinals(
                rows, 1000.0, config,
            )
        self.assertEqual(anchors, [0, 3, 5])
        self.assertIn("maximum_gap", evidence[3]["triggers"])
        self.assertIn("endpoint", evidence[-1]["triggers"])

    def test_flow_trigger_selects_anchor_after_minimum_gap(self):
        rows = _bound([0.0, 0.10, 0.30, 0.31])
        config = AdaptiveAnchorConfig(
            minimum_gap=2,
            maximum_gap=10,
            pixel_stride=2,
            flow_threshold_px=20.0,
            minimum_overlap_fraction=0.0,
            translation_threshold_m=10.0,
            rotation_threshold_deg=90.0,
            minimum_valid_depth_samples=100,
        )
        with patch("pose_pipeline.submaps._read_depth", return_value=self.depth):
            anchors, evidence = select_adaptive_anchor_ordinals(
                rows, 1000.0, config,
            )
        self.assertEqual(anchors, [0, 2, 3])
        self.assertIn("geometric_flow", evidence[2]["triggers"])
        self.assertNotIn("geometric_flow", evidence[1]["triggers"])

    def test_invalid_depth_fails_closed(self):
        rows = _bound([0.0, 0.10])
        invalid = np.zeros_like(self.depth)
        with self.assertRaisesRegex(ValueError, "valid depth samples"):
            geometric_reprojection_stats(
                rows[0], rows[1], 1000.0,
                AdaptiveAnchorConfig(minimum_gap=1),
                anchor_depth=invalid,
                current_depth=invalid,
            )

    def test_schedule_audit_covers_each_noninitial_frame_once(self):
        rows = _bound([0.0] * 6)
        config = AdaptiveAnchorConfig(
            minimum_gap=2,
            maximum_gap=3,
            pixel_stride=2,
            minimum_valid_depth_samples=100,
        )
        with patch("pose_pipeline.submaps._read_depth", return_value=self.depth):
            audit = audit_anchor_schedule(
                rows, [0, 3, 5], 1000.0, config,
            )
        self.assertEqual([row["ordinal"] for row in audit], [1, 2, 3, 4, 5])
        self.assertEqual(
            [row["anchor_ordinal"] for row in audit], [0, 0, 0, 3, 3],
        )

    def test_schedule_audit_rejects_missing_endpoint(self):
        with self.assertRaisesRegex(ValueError, "sorted unique endpoints"):
            audit_anchor_schedule(_bound([0.0] * 3), [0, 1], 1000.0)


if __name__ == "__main__":
    unittest.main()
