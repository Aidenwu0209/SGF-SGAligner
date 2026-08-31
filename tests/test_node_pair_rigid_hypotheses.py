import copy
import unittest

import numpy as np

from safety.node_pair_rigid_hypotheses import (
    NodePairHypothesisConfig,
    cross_direction_mode_matches,
    disjoint_complete_linkage_modes,
    estimate_node_pair,
    rigid_kabsch,
)


def transform(tx=0.0, yaw=0.0):
    theta = np.radians(yaw)
    value = np.eye(4)
    value[:3, :3] = [[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    value[0, 3] = tx
    return value


def apply(points, value):
    return points @ value[:3, :3].T + value[:3, 3]


class NodePairRigidHypothesisTests(unittest.TestCase):
    def test_kabsch_recovers_rigid_transform(self):
        source = np.random.default_rng(4).normal(size=(20, 3))
        expected = transform(0.13, 7.0)
        observed = rigid_kabsch(source, apply(source, expected))
        np.testing.assert_allclose(observed, expected, atol=1e-10)

    def test_deterministic_ransac_rejects_outliers(self):
        rng = np.random.default_rng(5)
        source = rng.normal(size=(80, 3))
        expected = transform(0.2, 4.0)
        reference = apply(source, expected)
        reference[50:] = rng.normal(size=(30, 3)) + 8
        scores = np.linspace(1, 0.1, len(source))
        first = estimate_node_pair(
            source, reference, scores, seed_context="same")
        second = estimate_node_pair(
            source, reference, scores, seed_context="same")
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["inliers_5cm"], 50)
        np.testing.assert_array_equal(first["transform"], second["transform"])
        np.testing.assert_allclose(first["transform"], expected, atol=1e-10)

    def test_two_modes_are_disjoint_and_complete_linkage(self):
        rows = []
        for index, tx in enumerate((0.0, 0.01, 0.02, 0.5, 0.51, 0.52)):
            rows.append({
                "node_pair_original": [index, index + 10],
                "transform": transform(tx),
                "inliers_5cm": 20 - index,
                "weighted_inlier_support": 3.0,
            })
        result = disjoint_complete_linkage_modes(rows)
        self.assertTrue(result["assigned_once"])
        self.assertEqual(
            sorted(row["member_count"] for row in result["eligible_modes"]),
            [3, 3])
        members = [tuple(pair) for mode in result["modes"]
                   for pair in mode["members"]]
        self.assertEqual(len(members), len(set(members)))

    def test_chain_cannot_enter_one_complete_linkage_mode(self):
        rows = [{
            "node_pair_original": [index, index],
            "transform": transform(tx),
            "inliers_5cm": 10,
            "weighted_inlier_support": 1.0,
        } for index, tx in enumerate((0.00, 0.09, 0.18))]
        result = disjoint_complete_linkage_modes(rows)
        self.assertEqual(max(row["member_count"] for row in result["modes"]), 2)

    def test_order_does_not_change_partition(self):
        rows = [{
            "node_pair_original": [index, index + 20],
            "transform": transform(tx),
            "inliers_5cm": 10,
            "weighted_inlier_support": 1.0,
        } for index, tx in enumerate((0.0, 0.01, 0.02, 0.5, 0.51, 0.52))]
        expected = disjoint_complete_linkage_modes(rows)
        shuffled = copy.deepcopy(rows)[::-1]
        observed = disjoint_complete_linkage_modes(shuffled)
        normal = lambda result: sorted(
            sorted(tuple(pair) for pair in mode["members"])
            for mode in result["modes"])
        self.assertEqual(normal(expected), normal(observed))

    def test_forward_reverse_modes_match_only_after_inverse(self):
        forward = [{"medoid_transform": transform(0.2), "eligible": True}]
        reverse = [{"medoid_transform": transform(-0.2), "eligible": True}]
        matches = cross_direction_mode_matches(forward, reverse)
        self.assertEqual(len(matches), 1)
        self.assertLess(matches[0]["translation_m"], 1e-9)

    def test_insufficient_correspondences_fail_closed(self):
        points = np.zeros((5, 3))
        result = estimate_node_pair(
            points, points, np.ones(5), seed_context="few")
        self.assertEqual(result["status"], "insufficient_correspondences")


if __name__ == "__main__":
    unittest.main()
