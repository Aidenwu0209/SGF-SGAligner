from __future__ import annotations

import unittest

import numpy as np

from matching.sgpgm_enhancements import (
    SGPGMEnhancementConfig,
    enhance_node_matching,
    object_geometry_descriptors,
    rescore_point_correspondences,
)


def _clouds(seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.normal(size=(512, 3)) * np.asarray([0.1, 0.2, 0.3])
    large = rng.normal(size=(512, 3)) * np.asarray([0.5, 0.8, 1.2])
    return np.stack([small, large, large.copy(), small.copy()])


class SGPGMEnhancementTests(unittest.TestCase):
    def test_production_default_remains_official_and_disabled(self):
        config = SGPGMEnhancementConfig()
        self.assertEqual(config.matching_policy, "official_top3")
        self.assertEqual(config.geometry_fusion_alpha, 0.0)
        self.assertEqual(config.graph_rescore_beta, 0.0)
        self.assertFalse(config.provenance()["production_default_enabled"])

    def test_partial_assignment_is_deterministic_one_to_one_and_gt_free(self):
        embedding = np.asarray([
            [1.0, 0.0], [0.0, 1.0], [0.95, 0.05], [0.05, 0.95],
        ])
        config = SGPGMEnhancementConfig(matching_policy="sinkhorn_partial")
        first = enhance_node_matching(embedding, _clouds(), 2, config)
        second = enhance_node_matching(embedding, _clouds(), 2, config)
        self.assertEqual(first, second)
        self.assertEqual(
            len({a for a, _ in first.node_corrs}), len(first.node_corrs)
        )
        self.assertEqual(
            len({b for _, b in first.node_corrs}), len(first.node_corrs)
        )
        self.assertFalse(config.provenance()["gt_at_inference"])

    def test_p2sg_lite_descriptor_is_rigid_transform_invariant(self):
        rng = np.random.default_rng(9)
        cloud = rng.normal(size=(512, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        moved = cloud @ rotation.T + np.asarray([4.0, -2.0, 0.3])
        observed = object_geometry_descriptors(np.stack([cloud, moved]))
        np.testing.assert_allclose(observed[0], observed[1], atol=1e-10)

    def test_geometry_fusion_can_disambiguate_equal_graph_embeddings(self):
        embedding = np.ones((4, 3), dtype=np.float64)
        config = SGPGMEnhancementConfig(
            matching_policy="sinkhorn_partial",
            geometry_fusion_alpha=1.0,
            min_matches=2,
        )
        result = enhance_node_matching(embedding, _clouds(), 2, config)
        self.assertEqual(set(result.node_corrs), {(0, 3), (1, 2)})

    def test_graph_prior_rescore_promotes_trusted_node_pair(self):
        raw = [np.asarray([0.8, 0.4]), np.asarray([0.8, 0.4])]
        pairs = [(0, 2), (1, 3)]
        rescored = rescore_point_correspondences(
            raw, pairs, {(0, 2): 0.1, (1, 3): 0.9}, beta=1.0
        )
        self.assertGreater(max(rescored[1]), max(rescored[0]))
        unchanged = rescore_point_correspondences(
            raw, pairs, {(0, 2): 0.1, (1, 3): 0.9}, beta=0.0
        )
        np.testing.assert_allclose(unchanged[0], unchanged[1])

    def test_invalid_or_implicit_geometry_configuration_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires sinkhorn_partial"):
            SGPGMEnhancementConfig(geometry_fusion_alpha=0.35).validate()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SGPGMEnhancementConfig(graph_rescore_beta=-1.0).validate()


if __name__ == "__main__":
    unittest.main()
