from __future__ import annotations

import copy
import unittest

import numpy as np

from pose_pipeline.robust_backend import (
    RobustPoseConfig,
    compatibility_hypotheses,
    deterministic_ransac_hypothesis,
    generate_hypotheses,
    decide_registration_v2,
    repeated_solver_consensus,
    select_cross_solver_consensus,
)


def rigid_points():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(30, 3))
    angle = np.deg2rad(8.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    reference = source @ rotation.T + np.array([0.2, -0.1, 0.05])
    return source, reference


class RobustPoseBackendTests(unittest.TestCase):
    def test_compatibility_pyramid_recovers_rigid_pose(self):
        source, reference = rigid_points()
        values = compatibility_hypotheses(source, reference)
        self.assertEqual(len(values), 3)
        self.assertTrue(all(row["support_count"] == 30 for row in values))
        moved = source @ np.asarray(values[0]["transform"])[:3, :3].T \
            + np.asarray(values[0]["transform"])[:3, 3]
        self.assertLess(float(np.max(np.linalg.norm(moved - reference, axis=1))), 1e-8)
        deterministic = deterministic_ransac_hypothesis(source, reference)
        self.assertIsNotNone(deterministic)
        self.assertEqual(deterministic["solver_family"], "deterministic_ransac")

    def test_consensus_requires_two_solver_families(self):
        source, reference = rigid_points()
        values = compatibility_hypotheses(source, reference)
        rejected = select_cross_solver_consensus(values)
        self.assertFalse(rejected["accepted"])
        witness = copy.deepcopy(values[0])
        witness["solver_family"] = "pygcransac"
        witness["solver"] = "fixture_pygcransac"
        accepted = select_cross_solver_consensus(values + [witness])
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["solver_families"], [
            "compatibility_graph", "pygcransac",
        ])
        deterministic = deterministic_ransac_hypothesis(source, reference)
        stable = select_cross_solver_consensus(values + [deterministic])
        self.assertTrue(stable["accepted"])
        selected = (values + [deterministic])[stable["selected_index"]]
        self.assertEqual(selected["solver_family"], "deterministic_ransac")

    def test_decision_is_fail_closed_and_rejects_gt(self):
        source, reference = rigid_points()
        values = compatibility_hypotheses(source, reference)
        witness = copy.deepcopy(values[0])
        witness["solver_family"] = "pygcransac"
        consensus = select_cross_solver_consensus(values + [witness])
        metrics = {
            "spatial_extent_m": 5.0,
            "spatial_second_axis_m": 2.0,
            "icp_update_translation_m": 0.01,
            "icp_update_rotation_deg": 0.1,
            "bidirectional_translation_m": 0.01,
            "bidirectional_rotation_deg": 0.1,
            "cycle_translation_m": 0.01,
            "cycle_rotation_deg": 0.1,
            "overlap_ratio": 0.8,
        }
        self.assertTrue(decide_registration_v2(
            consensus, metrics,
        )["usable_for_reconstruction"])
        bad = dict(metrics, overlap_ratio=0.01)
        self.assertFalse(decide_registration_v2(
            consensus, bad,
        )["usable_for_reconstruction"])
        with self.assertRaisesRegex(ValueError, "GT fields"):
            decide_registration_v2(consensus, dict(metrics, gt_rte=0.0))

    def test_repeat_consensus_rejects_rival_and_selects_observed_medoid(self):
        _source, reference = rigid_points()
        values = compatibility_hypotheses(_source, reference)
        repeated = []
        for index, tx in enumerate((0.0, 0.005, 0.010, 0.015, 0.30)):
            row = copy.deepcopy(values[0])
            transform = np.asarray(row["transform"], dtype=float)
            transform[0, 3] += tx
            row["transform"] = transform.tolist()
            row["transform_sha256"] = f"{index + 1:064x}"
            row["hypothesis_sha256"] = f"{index + 10:064x}"
            repeated.append(row)
        selected = repeated_solver_consensus(repeated)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["certificate"]["repeat_quorum"], 4)
        rival = copy.deepcopy(repeated)
        rival[-2]["transform"][0][3] += 0.30
        self.assertIsNone(repeated_solver_consensus(rival))

    def test_pygcransac_is_never_an_acceptance_voter(self):
        source, reference = rigid_points()
        result = generate_hypotheses(
            source, reference, include_pygcransac=False, include_teaser=False,
        )
        self.assertEqual(
            {row["solver_family"] for row in result["hypotheses"]},
            {"compatibility_graph", "deterministic_ransac"},
        )
        self.assertEqual(result["witness_hypotheses"], [])


if __name__ == "__main__":
    unittest.main()
