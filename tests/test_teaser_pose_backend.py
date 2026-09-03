from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from pose_pipeline.robust_backend import RobustPoseConfig, teaser_hypotheses


@unittest.skipUnless(
    importlib.util.find_spec("teaserpp_python"), "TEASER++ binding required",
)
class TeaserPoseBackendTests(unittest.TestCase):
    def test_gnc_tls_fixed_scale_hypothesis(self):
        rng = np.random.default_rng(20260901)
        source = rng.normal(size=(30, 3))
        angle = np.deg2rad(8.0)
        rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.asarray([0.20, -0.10, 0.30])
        reference = source @ rotation.T + translation
        reference[-4:] = rng.normal(size=(4, 3)) * 3.0
        values, reason = teaser_hypotheses(
            source, reference, RobustPoseConfig(minimum_support=6),
        )
        self.assertIsNone(reason)
        self.assertTrue(values)
        self.assertTrue(all(row["estimate_scaling"] is False for row in values))
        self.assertTrue(all(row["solver_family"] == "teaserpp" for row in values))
        self.assertTrue(all(
            row["certificate"]["rotation_algorithm"] == "GNC_TLS"
            for row in values
        ))


if __name__ == "__main__":
    unittest.main()
