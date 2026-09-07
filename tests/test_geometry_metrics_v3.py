from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_pipeline.geometry_metrics_v3 import (
    GeometryDiagnosticConfig, candidate_layer_pairs, compare_no_gt_geometry_v3,
    compare_plane_support,
)


def plane_cloud(*, separation=0.0, origin=(0.0, 0.0, 0.0)):
    x, z = np.meshgrid(np.linspace(0.0, 1.0, 21), np.linspace(0.0, 1.0, 21))
    points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    if separation:
        points = np.vstack([points, points + [0.0, separation, 0.0]])
    points = points + origin
    return points, np.tile([0.0, 1.0, 0.0], (len(points), 1))


def plane_hypothesis(identifier="floor", origin=(0.5, 0.0, 0.5)):
    return {"plane_id": identifier, "normal": [0.0, 1.0, 0.0],
            "centroid_m": list(origin)}


class CandidateLayerPairsTests(unittest.TestCase):
    def test_perfect_wall_and_floor_do_not_count_as_multiple_surfaces(self):
        points, normals = plane_cloud()
        floor = candidate_layer_pairs(points, normals)
        rotation = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        wall = candidate_layer_pairs(points @ rotation.T, normals @ rotation.T)
        self.assertEqual(floor["point_fraction_with_candidate_layer_pair"], 0.0)
        self.assertEqual(wall["point_fraction_with_candidate_layer_pair"], 0.0)
        self.assertFalse(wall["error_truth_established"])

    def test_parallel_pair_is_detected_without_grid_phase(self):
        points, normals = plane_cloud(separation=0.045)
        original = candidate_layer_pairs(points, normals)
        shifted = candidate_layer_pairs(points + [0.017, 0.01, -0.019], normals)
        self.assertEqual(original["point_fraction_with_candidate_layer_pair"], 1.0)
        self.assertEqual(original["candidate_pair_count"], shifted["candidate_pair_count"])
        self.assertEqual(original["points_with_candidate_layer_pair"],
                         shifted["points_with_candidate_layer_pair"])

    def test_pair_predicate_is_invariant_to_rigid_rotation_and_normal_sign(self):
        points, normals = plane_cloud(separation=0.045)
        rotation = Rotation.from_rotvec([0.4, -0.7, 0.8]).as_matrix()
        transformed = candidate_layer_pairs(
            points @ rotation.T + [2.12, -1.04, 3.31], -normals @ rotation.T,
        )
        original = candidate_layer_pairs(points, normals)
        self.assertEqual(original["candidate_pair_count"], transformed["candidate_pair_count"])
        self.assertEqual(transformed["point_fraction_with_candidate_layer_pair"], 1.0)

    def test_missing_normals_are_inconclusive_and_not_implicit_pass(self):
        points, normals = plane_cloud(separation=0.04)
        result = candidate_layer_pairs(points, normals * 0)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["point_fraction_with_candidate_layer_pair"])

    def test_nonparallel_near_surfaces_are_not_parallel_layer_evidence(self):
        points, normals = plane_cloud()
        rotation = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        wall = points @ rotation.T + [0.0, 0.04, 0.0]
        wall_normals = normals @ rotation.T
        result = candidate_layer_pairs(np.vstack([points, wall]),
                                       np.vstack([normals, wall_normals]))
        self.assertEqual(result["candidate_pair_count"], 0)


class PlaneSupportTests(unittest.TestCase):
    def test_far_coplanar_surfaces_fail_pose_compatibility(self):
        bp, bn = plane_cloud(separation=0.03)
        cp, cn = plane_cloud(separation=0.02, origin=(20.0, 0.0, 0.0))
        result = compare_plane_support(bp, bn, cp, cn, plane_hypothesis(),
                                       plane_hypothesis("other", (20.5, 0.0, 0.5)))
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["reason"], "plane_pose_incompatible")

    def test_near_but_disjoint_coplanar_support_is_rejected(self):
        bp, bn = plane_cloud(separation=0.03)
        cp, cn = plane_cloud(separation=0.02, origin=(1.5, 0.0, 0.0))
        result = compare_plane_support(bp, bn, cp, cn, plane_hypothesis(),
                                       plane_hypothesis("other", (2.0, 0.0, 0.5)))
        self.assertEqual(result["reason"], "projected_support_does_not_overlap_sufficiently")
        self.assertEqual(result["baseline_projected_overlap_fraction"], 0.0)

    def test_thickness_uses_all_common_support_not_only_ransac_inliers(self):
        bp, bn = plane_cloud(separation=0.06)
        cp, cn = plane_cloud(separation=0.02)
        left = {**plane_hypothesis(), "thickness_p90_p10_m": 0.0}
        result = compare_plane_support(bp, bn, cp, cn, left, plane_hypothesis("candidate"))
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["baseline_thickness_p90_p10_m"], 0.06)
        self.assertAlmostEqual(result["candidate_thickness_p90_p10_m"], 0.02)
        self.assertAlmostEqual(result["thickness_candidate_over_baseline"], 1 / 3)
        self.assertFalse(result["common_rgbd_visibility_established"])

    def test_support_and_thickness_invariant_with_transformed_hypotheses(self):
        bp, bn = plane_cloud(separation=0.06)
        cp, cn = plane_cloud(separation=0.02)
        rot = Rotation.from_rotvec([0.4, -0.7, 0.8]).as_matrix()
        shift = np.array([0.73, -0.22, 0.91])
        def transform(hypothesis):
            return {**hypothesis,
                    "normal": (rot @ hypothesis["normal"]).tolist(),
                    "centroid_m": (rot @ hypothesis["centroid_m"] + shift).tolist()}
        result = compare_plane_support(
            bp @ rot.T + shift, bn @ rot.T, cp @ rot.T + shift, cn @ rot.T,
            transform(plane_hypothesis()), transform(plane_hypothesis("candidate")),
        )
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["thickness_candidate_over_baseline"], 1 / 3)

    def test_zero_baseline_thickness_makes_relative_improvement_inconclusive(self):
        bp, bn = plane_cloud()
        cp, cn = plane_cloud()
        result = compare_plane_support(bp, bn, cp, cn, plane_hypothesis(),
                                       plane_hypothesis("candidate"))
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["thickness_candidate_over_baseline"])

    def test_diagnostic_never_promotes_even_when_both_measures_improve(self):
        bp, bn = plane_cloud(separation=0.06)
        cp, cn = plane_cloud(separation=0.02)
        result = compare_no_gt_geometry_v3(
            bp, bn, cp, cn, baseline_planes=[plane_hypothesis()],
            candidate_planes=[plane_hypothesis("candidate")],
        )
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["candidate_layer_pairs"]["candidate_over_baseline"], 0.0)
        self.assertFalse(result["usable_for_promotion"])
        self.assertTrue(result["diagnostic_only"])

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            GeometryDiagnosticConfig(query_batch_size=0)
        with self.assertRaises(ValueError):
            GeometryDiagnosticConfig(minimum_projected_overlap_fraction=1.1)
        with self.assertRaises(ValueError):
            GeometryDiagnosticConfig(minimum_normal_separation_m=0.13)


if __name__ == "__main__":
    unittest.main()
