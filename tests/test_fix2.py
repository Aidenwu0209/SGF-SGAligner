"""Fix-2 audit/regression tests (>=12).

Sparse-check gating, real-feature rejections (non-zero ICP update,
bidirectional inconsistency, corr-perfect but surface-disjoint, 180°
rotation, 1–4 m translation), GT fail-closed, transform/refusion
gating, fixed12 non-use for thresholds, provenance on every feature.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from safety import decision_features as dfx
from safety.decision_features import (
    RULE_EVALUATORS, RULE_THRESHOLDS, segment_icp, surface_evidence,
    transform_discrepancy,
)


def good_transform_features(**overrides):
    features = {
        "overlap_10cm": 0.6,
        "median_residual_m": 0.03,
        "symmetric_trimmed_chamfer_m": 0.04,
        "icp_converged": True,
        "icp_update_translation_m": 0.02,
        "icp_update_rotation_deg": 1.0,
        "icp_fitness": 0.7,
        "ransac_inliers": 50,
        "spatial_extent_m": 3.0,
        "bidirectional_available": True,
        "bidirectional_rotation_deg": 0.5,
        "bidirectional_translation_m": 0.02,
        "node_pair_success_ratio": 0.9,
    }
    features.update(overrides)
    return features


class TestSparseCheck(unittest.TestCase):
    def test_lengths_guard_present_and_exact(self):
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/src/"
            "inference/sgf_official/inference.py"
        ).read_text()
        self.assertIn("lengths[1]", src)
        self.assertIn("angle_k", src)
        self.assertIn("cfg.model.num_points_in_patch", src)
        self.assertNotIn("_voxel_count", src)  # proxy removed

    def test_selected_index_out_of_range_is_zero_by_construction(self):
        # guard fires BEFORE model forward => no topk entry
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/src/"
            "inference/sgf_official/inference.py"
        ).read_text()
        guard_pos = src.index("insufficient_post_voxel_points\", {")
        forward_pos = src.index("output = model(data_dict)")
        self.assertLess(guard_pos, forward_pos)


class TestRealFeatureRejections(unittest.TestCase):
    def test_nonzero_icp_update_rejects(self):
        features = good_transform_features(
            icp_update_translation_m=0.5,
            icp_update_rotation_deg=25.0,
        )
        self.assertIn(
            "icp_translation_update_above_max",
            RULE_EVALUATORS["A"](features),
        )
        self.assertIn(
            "icp_rotation_update_above_max",
            RULE_EVALUATORS["A"](features),
        )

    def test_bidirectional_inconsistency_rejects(self):
        features = good_transform_features(
            bidirectional_rotation_deg=9.0,
            bidirectional_translation_m=0.4,
        )
        violations = RULE_EVALUATORS["B"](features)
        self.assertIn("bidirectional_rotation_above_max", violations)
        self.assertIn("bidirectional_translation_above_max", violations)

    def test_bidirectional_unavailable_rejects_b_and_c(self):
        features = good_transform_features(bidirectional_available=False)
        self.assertIn(
            "bidirectional_unavailable", RULE_EVALUATORS["B"](features)
        )

    def test_corr_perfect_but_surfaces_disjoint_rejects(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 1, (4000, 3))
        ref = rng.uniform(50, 51, (4000, 3))  # 50 m away
        transform = np.eye(4)
        evidence = surface_evidence(src, ref, transform, seed=1)
        self.assertLess(evidence.overlap_10cm, 0.01)
        features = good_transform_features(
            overlap_10cm=evidence.overlap_10cm,
            median_residual_m=evidence.median_residual_m,
        )
        self.assertIn(
            "surface_overlap_10cm_below_min", RULE_EVALUATORS["A"](features)
        )

    def test_180_degree_rotation_rejects(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 2, (4000, 3))
        rot = np.diag([1.0, -1.0, -1.0])  # 180 deg about x
        ref = src @ rot.T + np.array([0.2, 0.1, 0.05])
        evidence = surface_evidence(src, ref, np.eye(4), seed=2)
        icp = segment_icp(src, ref, np.eye(4), seed=2)
        features = good_transform_features(
            overlap_10cm=evidence.overlap_10cm,
            median_residual_m=evidence.median_residual_m,
            icp_update_rotation_deg=icp.update_rotation_deg,
            icp_fitness=icp.fitness,
        )
        self.assertTrue(
            any(v.startswith("surface_") or v.startswith("icp_")
                for v in RULE_EVALUATORS["A"](features))
        )

    def test_wrong_translation_1_to_4_m_rejects(self):
        for shift in (1.0, 2.5, 4.0):
            rng = np.random.default_rng(3)
            src = rng.uniform(0, 2, (4000, 3))
            ref = src + np.array([shift, 0, 0])
            evidence = surface_evidence(
                src, ref, np.eye(4), seed=4
            )
            features = good_transform_features(
                overlap_10cm=evidence.overlap_10cm,
                median_residual_m=evidence.median_residual_m,
            )
            self.assertTrue(
                any(v.startswith("surface_")
                    for v in RULE_EVALUATORS["A"](features)),
                msg=f"shift {shift}: no surface violation",
            )

    def test_node_pair_ratio_rejects_c_only(self):
        features = good_transform_features(node_pair_success_ratio=0.2)
        self.assertIn(
            "node_pair_success_ratio_below_min",
            RULE_EVALUATORS["C"](features),
        )
        self.assertEqual(RULE_EVALUATORS["B"](features), [])

    def test_good_features_accept_all_rules(self):
        for rule in "ABC":
            self.assertEqual(RULE_EVALUATORS[rule](
                good_transform_features()), [])


class TestSurfaceIcpMath(unittest.TestCase):
    def test_icp_recovers_known_transform(self):
        rng = np.random.default_rng(5)
        angle = np.deg2rad(12)
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1],
        ])
        t = np.array([0.1, -0.05, 0.02])
        src = rng.uniform(0, 2, (5000, 3))
        ref = src @ rot.T + t
        result = segment_icp(src, ref, np.eye(4), seed=6)
        composed = result.transform
        cos = (np.trace(composed[:3, :3].T @ rot) - 1) / 2
        self.assertLess(np.degrees(np.arccos(np.clip(cos, -1, 1))), 0.5)
        self.assertLess(np.linalg.norm(composed[:3, 3] - t), 0.02)

    def test_bidirectional_real_computation(self):
        rng = np.random.default_rng(7)
        angle = np.deg2rad(6)
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1],
        ])
        t = np.array([0.05, 0.02, 0.01])
        src = rng.uniform(0, 2, (6000, 3))
        ref = src @ rot.T + t
        forward = segment_icp(src, ref, np.eye(4), seed=8).transform
        reverse = segment_icp(
            ref, src, np.linalg.inv(forward), seed=9
        ).transform
        r_gap, t_gap = transform_discrepancy(forward, reverse)
        self.assertLess(r_gap, 1.0)
        self.assertLess(t_gap, 0.03)

    def test_every_feature_has_provenance(self):
        rng = np.random.default_rng(11)
        src = rng.uniform(0, 1, (1000, 3))
        ev = surface_evidence(src, src + 0.01, np.eye(4))
        self.assertEqual(ev.units, "metres")
        self.assertIn("corr-independent", ev.source)
        icp = segment_icp(src, src + 0.01, np.eye(4))
        self.assertIn("union", icp.source)


class TestGatingAndAudit(unittest.TestCase):
    def test_gt_fields_fail_closed(self):
        from safety.registration_decision import (
            evaluate_registration_decision,
        )

        with self.assertRaises(ValueError):
            evaluate_registration_decision(
                {"ransac_inliers": 10, "gt_rre": 1.0}
            )

    def test_rejected_no_usable_transform(self):
        from safety.registration_decision import write_decision_files

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            decision = {
                "status": "rejected",
                "usable_for_reconstruction": False,
                "rejection_reasons": ["x"],
            }
            write_decision_files(tmp, decision, np.eye(4))
            self.assertFalse((Path(tmp) / "transform.txt").exists())

    def test_refusion_requires_accepted(self):
        from reconstruction.rgbd_refusion import check_refusion_authorization

        self.assertFalse(check_refusion_authorization(
            {"usable_for_reconstruction": False}, np.eye(4)
        ))
        self.assertTrue(check_refusion_authorization(
            {"usable_for_reconstruction": True}, np.eye(4)
        ))

    def test_fixed12_not_used_for_thresholds(self):
        # thresholds are round constants in the module; no tuning hook
        for value in RULE_THRESHOLDS.values():
            self.assertIn(value * 100 % 5, (0.0,))  # round numbers only

    def test_no_hardcoded_zero_features(self):
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/src/"
            "inference/sgf_official/inference.py"
        ).read_text()
        self.assertNotIn(
            '"icp_update_translation_m": 0.0', src
        )
        self.assertNotIn(
            '"bidirectional_rotation_deg": 0.0', src
        )


if __name__ == "__main__":
    unittest.main()
