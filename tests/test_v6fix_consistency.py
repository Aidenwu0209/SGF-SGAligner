import ast
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from spatial_consistency import (  # noqa: E402
    cluster_candidates_corrected,
    rank_hypotheses_corrected,
)


class CorrectedClusteringTests(unittest.TestCase):
    def setUp(self):
        self.src = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
        }
        self.ref = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0]),
            2: np.array([0.0, 1.0, 0.0]),
        }

    def test_permutation_invariant(self):
        candidates = [(2, 2), (0, 0), (1, 1)]
        a = cluster_candidates_corrected(candidates, self.src, self.ref)
        b = cluster_candidates_corrected(
            list(reversed(candidates)), self.src, self.ref)
        self.assertEqual(a, b)

    def test_incompatible_tail_is_not_residual_merged(self):
        ref = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([4.0, 0.0, 0.0]),
            2: np.array([0.0, 7.0, 0.0]),
        }
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1), (2, 2)], self.src, ref)
        self.assertEqual(3, len(clusters))
        self.assertTrue(all(len(cluster) == 1 for cluster in clusters))

    def test_adjacency_contradiction_splits(self):
        src_adj = {0: {1}, 1: {0}, 2: set()}
        ref_adj = {0: set(), 1: set(), 2: set()}
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1)], self.src, self.ref,
            adjacency_src=src_adj, adjacency_ref=ref_adj)
        self.assertEqual([[(0, 0)], [(1, 1)]], clusters)

    def test_unknown_adjacency_is_neutral(self):
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1)], self.src, self.ref,
            adjacency_src=None, adjacency_ref=None)
        self.assertEqual([[(0, 0), (1, 1)]], clusters)

    def test_extent_contradiction_splits(self):
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1)], self.src, self.ref,
            extents_src={0: 1.0, 1: 1.0},
            extents_ref={0: 1.0, 1: 9.0})
        self.assertEqual([[(0, 0)], [(1, 1)]], clusters)

    def test_common_extent_scale_error_does_not_pass(self):
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1)], self.src, self.ref,
            extents_src={0: 100.0, 1: 100.0},
            extents_ref={0: 1.0, 1: 1.0})
        self.assertEqual([[(0, 0)], [(1, 1)]], clusters)

    def test_semantic_unknown_is_neutral(self):
        clusters = cluster_candidates_corrected(
            [(0, 0), (1, 1)], self.src, self.ref,
            semantic_src={0: None, 1: None},
            semantic_ref={0: None, 1: None})
        self.assertEqual([[(0, 0), (1, 1)]], clusters)


def record(name, quality, *, kind="cluster_corrected", size=1):
    return {
        "stable_signature": name,
        "kind": kind,
        "cluster_size": size,
        "registration_valid": True,
        "bidirectional_available": True,
        "ransac_support": quality,
        "icp_fitness": quality,
        "surface_overlap": quality,
        "bidir_rotation_deg": 1.0 / max(quality, 1e-6),
        "bidir_translation_m": 1.0 / max(quality, 1e-6),
        "spatial_extent_m": quality,
    }


class CorrectedRankingTests(unittest.TestCase):
    def test_cluster_size_cannot_buy_quality(self):
        good = record("good", 0.9, size=1)
        bad = record("bad", 0.1, size=1000)
        winner, _ = rank_hypotheses_corrected([bad, good])
        self.assertEqual("good", winner["stable_signature"])

    def test_monotonic_scale_does_not_change_winner(self):
        records = [record("a", 0.2), record("b", 0.7), record("c", 0.5)]
        winner_a, _ = rank_hypotheses_corrected(records)
        scaled = []
        for item in records:
            row = dict(item)
            for key in ("ransac_support", "icp_fitness",
                        "surface_overlap", "spatial_extent_m"):
                row[key] = row[key] * 100.0 + 17.0
            for key in ("bidir_rotation_deg", "bidir_translation_m"):
                row[key] = row[key] * 3.0 + 2.0
            scaled.append(row)
        winner_b, _ = rank_hypotheses_corrected(scaled)
        self.assertEqual(winner_a["stable_signature"],
                         winner_b["stable_signature"])

    def test_cluster_size_is_only_after_quality_keys(self):
        small = record("small", 0.9, size=1)
        large = record("large", 0.1, size=100)
        winner, _ = rank_hypotheses_corrected([large, small])
        self.assertEqual("small", winner["stable_signature"])

    def test_ransac_rank_precedes_later_quality_keys(self):
        first = record("first", 0.5)
        second = record("second", 0.5)
        first.update({"ransac_support": 0.9, "icp_fitness": 0.1,
                      "surface_overlap": 0.1})
        second.update({"ransac_support": 0.8, "icp_fitness": 0.99,
                       "surface_overlap": 0.99})
        winner, _ = rank_hypotheses_corrected([second, first])
        self.assertEqual("first", winner["stable_signature"])

    def test_missing_metric_is_worst_not_best(self):
        missing = record("missing", 0.9)
        missing["ransac_support"] = None
        finite = record("finite", 0.1)
        winner, _ = rank_hypotheses_corrected([missing, finite])
        self.assertEqual("finite", winner["stable_signature"])


class ProtocolBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_path = SCRIPTS / "v6fix_consistency_audit.py"
        cls.tree = ast.parse(source_path.read_text())
        cls.functions = {
            node.name: node for node in cls.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    @staticmethod
    def calls_named(function, name):
        return any(
            isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == name)
                 or (isinstance(node.func, ast.Attribute)
                     and node.func.attr == name))
            for node in ast.walk(function)
        )

    def test_inference_has_no_gt_loader(self):
        self.assertFalse(self.calls_named(
            self.functions["infer_paths"], "load_gt_transform"))

    def test_gt_is_only_loaded_posthoc(self):
        self.assertTrue(self.calls_named(
            self.functions["evaluate_posthoc"], "load_gt_transform"))

    def test_no_historical_root_literal_in_runner_output(self):
        constants = {
            node.value for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertNotIn(
            "/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
            "official_sgaligner_v6_fix_consistency_audit_20260829",
            constants)

    def test_exact_flat_preserves_official_candidate_order(self):
        import v6fix_consistency_audit as audit

        captured = []

        def fake_registration(_geot, members):
            captured.append(list(members))
            return ({"transform": np.eye(4), "inlier_ratio": 0.5,
                     "inliers": 5, "corrs": 10,
                     "node_pairs_used": list(members)}, list(members), [])

        features = {
            "icp_fitness": 0.5, "overlap_10cm": 0.5,
            "bidirectional_available": True,
            "bidirectional_rotation_deg": 0.0,
            "bidirectional_translation_m": 0.0,
            "spatial_extent_m": 1.0,
        }
        with mock.patch.object(audit, "combo_registration",
                               side_effect=fake_registration), \
                mock.patch.object(
                    audit, "combo_decision",
                    return_value=(features, {
                        "usable_for_reconstruction": False,
                        "status": "rejected", "rejection_reasons": []},
                                  None)):
            audit.hypothesis_record(
                {}, {}, [(2, 7), (0, 5), (1, 6)], "flat", {})
        self.assertEqual([[(2, 7), (0, 5), (1, 6)]], captured)

    def test_runtime_gt_loader_cannot_be_reached_from_infer(self):
        import adapters.sgf.data_sources as data_sources
        import v6fix_consistency_audit as audit

        fake = record("fake", 0.5, kind="flat")
        fake.update({"members": [(0, 1)], "registration_valid": True})
        geometry = {
            "centres_src": {0: np.zeros(3)},
            "centres_ref": {0: np.zeros(3)},
            "extents_src": {0: 1.0}, "extents_ref": {0: 1.0},
            "adjacency_src": {0: set()}, "adjacency_ref": {0: set()},
        }
        with mock.patch.object(
                data_sources, "load_gt_transform",
                side_effect=AssertionError("GT reached during inference")), \
                mock.patch.object(audit, "object_geometry",
                                  return_value=geometry), \
                mock.patch.object(audit, "cluster_candidates",
                                  return_value=[[(0, 0)]]), \
                mock.patch.object(audit, "cluster_candidates_corrected",
                                  return_value=[[(0, 0)]]), \
                mock.patch.object(audit, "hypothesis_record",
                                  return_value=dict(fake)), \
                mock.patch.object(audit, "hypothesis_rank",
                                  return_value=1.0):
            winners, _ = audit.infer_paths(
                {"src_count": 1}, {"node_corrs": [(0, 1)], "geot": {}})
        self.assertEqual({"F", "C0", "C1"}, set(winners))

    def test_zero_candidate_is_typed_not_failed(self):
        import v6fix_consistency_audit as audit

        rows = [{
            "paths": {"F": {"valid": False, "strict": False,
                             "accepted": False}},
            "audit": {"zero_candidate": True},
        }]
        counts = audit.aggregate(rows, "F")
        self.assertEqual(1, counts["zero_candidate"])
        self.assertEqual(0, counts["failed"])

    def test_gate_order_rejects_fixed12_before_freeze(self):
        import v6fix_consistency_audit as audit

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "missing prerequisite"):
                audit.enforce_gate_order(
                    Path(directory), "fixed12", "A", 0, None)

    def test_formal_gate_rejects_partial_limit(self):
        import v6fix_consistency_audit as audit

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "partial"):
                audit.enforce_gate_order(
                    Path(directory), "selection", "A", 0, 1)


if __name__ == "__main__":
    unittest.main()
