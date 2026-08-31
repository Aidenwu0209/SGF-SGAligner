import unittest

import numpy as np

from safety.registration_consensus import (
    ConsensusConfig, cross_direction_agreement, evaluate_direction,
    transform_distance,
)


def pose(tx=0.0, yaw_deg=0.0):
    angle = np.radians(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    out = np.eye(4)
    out[:3, :3] = [
        [cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    out[0, 3] = tx
    return out


def row(index, transform, accepted=True):
    return {
        "status": "ok", "transform": transform,
        "rule_b_accepted": accepted, "stable_signature": f"{index:02d}",
    }


class RegistrationConsensusTests(unittest.TestCase):
    def test_transform_distance_uses_geodesic_and_world_translation(self):
        rotation, translation = transform_distance(pose(), pose(0.03, 2.0))
        self.assertAlmostEqual(2.0, rotation, places=7)
        self.assertAlmostEqual(0.03, translation, places=7)

    def test_single_spurious_rule_b_pass_is_vetoed(self):
        rows = [
            row(0, pose(), True),
            row(1, pose(0.7, 25), False),
            row(2, pose(0.8, 28), False),
            row(3, pose(0.9, 30), False),
            row(4, pose(1.0, 35), False),
        ]
        result = evaluate_direction(rows, ConsensusConfig())
        self.assertFalse(result["usable"])
        self.assertIn(
            "rule_b_quorum_not_met", result["rejection_reasons"])

    def test_four_consistent_rule_b_passes_are_accepted(self):
        rows = [
            row(0, pose(0.000, 0.0)),
            row(1, pose(0.010, 0.4)),
            row(2, pose(0.015, 0.8)),
            row(3, pose(0.020, 1.0)),
            row(4, pose(0.8, 40), False),
        ]
        result = evaluate_direction(rows, ConsensusConfig())
        self.assertTrue(result["usable"])
        self.assertEqual(4, result["clique_sizes"][0])
        self.assertIn(result["medoid_original_index"], {0, 1, 2, 3})

    def test_rival_cluster_is_fail_closed(self):
        config = ConsensusConfig(quorum=2)
        rows = [
            row(0, pose(0.00, 0.0)),
            row(1, pose(0.01, 0.5)),
            row(2, pose(0.80, 30.0)),
            row(3, pose(0.81, 30.5)),
            row(4, pose(1.5, 80), False),
        ]
        result = evaluate_direction(rows, config)
        self.assertFalse(result["usable"])
        self.assertIn(
            "largest_clique_not_unique", result["rejection_reasons"])

    def test_cross_direction_requires_distinct_matches(self):
        config = ConsensusConfig(quorum=4)
        forward = [row(i, pose(i * 0.003, i * 0.2)) for i in range(5)]
        reverse = [row(i, pose(i * 0.004, i * 0.2)) for i in range(5)]
        result = cross_direction_agreement(forward, reverse, config)
        self.assertTrue(result["usable"])
        self.assertEqual(5, result["agreement_count"])

    def test_nonfinite_transform_is_rejected(self):
        rows = [row(i, pose()) for i in range(5)]
        rows[0]["transform"][0, 0] = np.nan
        result = evaluate_direction(rows, ConsensusConfig())
        self.assertFalse(result["usable"])
        self.assertEqual(4, result["valid"])


if __name__ == "__main__":
    unittest.main()
