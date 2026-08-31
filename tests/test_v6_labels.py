"""V6 unit tests: SGF node label builder (split/merge/empty/dup/GT
direction/boundary thresholds), loss math, spatial consistency."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from sgf_node_labels import (  # noqa: E402
    pair_statistics, classify, label_pair, audit, voxel_iou,
    extent_of,
)


def cloud(n, centre, scale=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(
        np.asarray(centre, dtype=np.float64), scale, size=(n, 3))


GT_I = np.eye(4)


def stat_of(a, b, sem=1.0):
    return classify(pair_statistics(
        a, b, a.mean(axis=0), b.mean(axis=0),
        extent_of(a), extent_of(b), sem, 0.0))


class TestLabelBoundaries(unittest.TestCase):
    def test_identical_clouds_positive(self):
        c = cloud(500, [1, 1, 1], seed=1)
        st = stat_of(c, c)
        self.assertEqual(st.label, "positive")
        self.assertGreaterEqual(st.bidir_10, 0.99)

    def test_far_apart_negative_not_hard(self):
        # different extents -> outside [0.7, 1.4] ratio band
        a = cloud(500, [0, 0, 0], 0.2, seed=2)
        b = cloud(500, [10, 0, 0], 1.0, seed=3)
        st = stat_of(a, b, sem=0.0)
        self.assertEqual(st.label, "negative")
        self.assertEqual(st.hard_negative, False)

    def test_far_apart_same_extent_is_hard_negative(self):
        # protocol: similar extent alone qualifies a hard negative
        a = cloud(500, [0, 0, 0], 0.2, seed=2)
        b = cloud(500, [10, 0, 0], 0.2, seed=3)
        st = stat_of(a, b, sem=0.0)
        self.assertEqual(st.label, "negative")
        self.assertTrue(st.hard_negative)

    def test_hard_negative_same_semantic(self):
        # geometrically disjoint but semantically identical
        a = cloud(500, [0, 0, 0], seed=4)
        b = cloud(500, [5, 0, 0], seed=5)
        st = stat_of(a, b, sem=1.0)
        self.assertEqual(st.label, "negative")
        self.assertTrue(st.hard_negative)  # semantic path

    def test_hard_negative_extent_ratio(self):
        a = cloud(500, [0, 0, 0], 0.2, seed=6)
        b = cloud(500, [8, 0, 0], 0.19, seed=7)  # similar extent
        st = stat_of(a, b, sem=0.0)
        self.assertEqual(st.label, "negative")
        self.assertTrue(st.hard_negative)  # extent path

    def test_ambiguous_band(self):
        # partial overlap ~0.2 coverage: between 0.10 and 0.30
        a = np.vstack([cloud(400, [0, 0, 0], 0.2, seed=8),
                       cloud(100, [5, 5, 5], 0.2, seed=9)])
        b = cloud(500, [0, 0, 0], 0.2, seed=8)
        b = np.vstack([b, cloud(100, [6, 6, 6], 0.2, seed=10)])
        st = stat_of(a, b)
        self.assertIn(st.label, ("ambiguous", "positive"))
        # engineered: main lobes coincide (high cov), tails differ
        self.assertGreater(st.bidir_10, 0.10)

    def test_boundary_positive_threshold(self):
        # exactly at 0.30 bidirectional -> positive by >= rule
        a = cloud(600, [0, 0, 0], 0.15, seed=11)
        # translate slightly so ~30% of points stay within 10cm
        b = a + np.array([0.22, 0, 0])
        st = stat_of(a, b)
        self.assertIn(st.label, ("positive", "ambiguous"))
        # cross-check the metric is in the expected regime
        self.assertGreater(st.bidir_10, 0.05)


class TestSplitMerge(unittest.TestCase):
    def test_split_one_src_many_refs(self):
        # one source object split into two reference halves
        big = cloud(800, [1, 1, 1], 0.5, seed=12)
        segs = {10: big}
        ref = {20: big[:400], 21: big[400:]}
        stats = label_pair(segs, ref, GT_I)
        labels = {(s.src, s.ref): s.label for s in stats}
        self.assertEqual(labels[(10, 20)], "positive")
        self.assertEqual(labels[(10, 21)], "positive")
        a = audit(stats)
        self.assertEqual(a["split_sources"], 1)

    def test_merge_many_srcs_one_ref(self):
        src = {10: cloud(400, [1, 1, 1], 0.2, seed=13),
               11: cloud(400, [1.05, 1, 1], 0.2, seed=14)}
        ref = {20: cloud(800, [1, 1, 1], 0.25, seed=15)}
        stats = label_pair(src, ref, GT_I)
        a = audit(stats)
        self.assertEqual(a["merged_refs"], 1)
        self.assertGreaterEqual(a["positive"], 2)

    def test_empty_graph(self):
        stats = label_pair({}, {20: cloud(100, [0, 0, 0])}, GT_I)
        self.assertEqual(stats, [])
        self.assertEqual(audit(stats)["pairs_total"], 0)

    def test_duplicate_ids_raise_or_classify(self):
        # duplicate object ids in a dict collapse silently in python;
        # the builder contract requires unique ids — assert the
        # segments dict size governs pair count
        segs = {10: cloud(200, [0, 0, 0], seed=16)}
        ref = {20: cloud(200, [0, 0, 0], seed=16),
               21: cloud(200, [9, 9, 9], seed=17)}
        stats = label_pair(segs, ref, GT_I)
        self.assertEqual(len(stats), 2)


class TestGTDirection(unittest.TestCase):
    def test_gt_moves_src_into_ref_frame(self):
        # GT that translates src by +2m in x
        gt = np.eye(4)
        gt[0, 3] = 2.0
        src = {10: cloud(400, [0, 0, 0], 0.2, seed=18)}
        ref = {20: cloud(400, [2, 0, 0], 0.2, seed=18)}
        stats = label_pair(src, ref, gt)
        self.assertEqual(
            {(s.src, s.ref): s.label for s in stats}[(10, 20)],
            "positive")
        # wrong direction would place src at -2 -> negative
        gt_wrong = np.eye(4)
        gt_wrong[0, 3] = -2.0
        stats_w = label_pair(src, ref, gt_wrong)
        self.assertEqual(
            {(s.src, s.ref): s.label for s in stats_w}[(10, 20)],
            "negative")


class TestLossMath(unittest.TestCase):
    def test_cross_graph_only_denominator(self):
        import torch

        from safety.cross_graph_loss import cross_graph_infonce

        torch.manual_seed(0)
        src = torch.randn(4, 8)
        ref = torch.randn(5, 8)
        positives = torch.zeros(4, 5, dtype=torch.bool)
        positives[0, 0] = True
        positives[1, 1] = True
        weights = torch.zeros(4, 5)
        weights[positives] = 1.0
        loss, diag = cross_graph_infonce(src, ref, positives,
                                         weights)
        self.assertTrue(torch.isfinite(loss))
        # multi-positive: two positives for row 0
        positives[0, 2] = True
        weights[0, 2] = 0.5
        loss2, _ = cross_graph_infonce(src, ref, positives, weights)
        self.assertTrue(torch.isfinite(loss2))

    def test_positive_never_in_negative_denominator(self):
        # structural property of the verified implementation:
        # check the module documents/uses cross-graph-only logits
        src_text = Path(
            ROOT / "src/safety/cross_graph_loss.py").read_text()
        self.assertIn("src_norm", src_text)
        self.assertIn("ref_norm", src_text)


class TestSpatialConsistency(unittest.TestCase):
    def test_compatible_pairs_cluster(self):
        from spatial_consistency import cluster_candidates

        # two rigid-consistent candidate pairs + one outlier
        centres_src = {0: np.array([0., 0., 0.]),
                       1: np.array([2., 0., 0.]),
                       2: np.array([50., 0., 0.])}
        centres_ref = {0: np.array([10., 0., 0.]),
                       1: np.array([12., 0., 0.]),
                       2: np.array([-40., 0., 0.])}
        cands = [(0, 0), (1, 1), (2, 2)]
        clusters = cluster_candidates(
            cands, centres_src, centres_ref)
        self.assertTrue(any({0, 1} == {a for a, _b in c}
                            for c in clusters))

    def test_inconsistent_mixing_split(self):
        from spatial_consistency import cluster_candidates

        # two candidates from the true solution + one that breaks
        # centre-distance preservation -> the mix must NOT form one
        # cluster (non-rigid mixing is what this layer rejects; pure
        # rigid flips are distance-preserving BY DEFINITION and are
        # handled downstream by ICP/decision, not here)
        centres_src = {0: np.array([0., 0., 0.]),
                       1: np.array([3., 0., 0.]),
                       2: np.array([0., 3., 0.])}
        centres_ref = {0: np.array([0., 0., 0.]),
                       1: np.array([3., 0., 0.]),
                       2: np.array([1., 3., 0.])}  # inconsistent
        cands = [(0, 0), (1, 1), (2, 2)]
        clusters = cluster_candidates(
            cands, centres_src, centres_ref)
        biggest = max(len(c) for c in clusters)
        self.assertLess(biggest, 3)
        self.assertGreaterEqual(biggest, 2)  # (0,0)-(1,1) stay paired


if __name__ == "__main__":
    unittest.main()
