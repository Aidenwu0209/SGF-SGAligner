from __future__ import annotations

import unittest

import numpy as np

from pose_pipeline.contracts import PoseRecord
from pose_pipeline.evaluation import trajectory_metrics


class PoseEvaluationTests(unittest.TestCase):
    def test_non_finite_gt_frames_can_be_excluded_without_hiding_coverage(self):
        estimate = [
            PoseRecord(index, index, np.eye(4), source="estimate")
            for index in range(3)
        ]
        reference = [
            PoseRecord(index, index, np.eye(4), source="reference")
            for index in (0, 2)
        ]
        result = trajectory_metrics(estimate, reference)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["excluded_estimate_frame_count"], 1)
        self.assertAlmostEqual(result["evaluation_coverage"], 2 / 3)

    def test_reference_may_not_contain_unknown_estimate_frame(self):
        estimate = [PoseRecord(0, 0, np.eye(4))]
        reference = [PoseRecord(1, 1, np.eye(4))]
        with self.assertRaisesRegex(ValueError, "missing from estimate"):
            trajectory_metrics(estimate, reference)

    def test_single_pose_reports_unavailable_relative_metrics(self):
        estimate = [PoseRecord(0, 0, np.eye(4), source="estimate")]
        reference = [PoseRecord(0, 0, np.eye(4), source="reference")]
        result = trajectory_metrics(estimate, reference)
        self.assertEqual(result["absolute_translation_m"]["count"], 1)
        self.assertTrue(result["absolute_translation_m"]["available"])
        self.assertEqual(result["relative_translation_m"]["count"], 0)
        self.assertFalse(result["relative_translation_m"]["available"])
        self.assertIsNone(result["relative_translation_m"]["rmse"])


if __name__ == "__main__":
    unittest.main()
