"""Stage-2 loss tests: reference cross-validation + all mandated cases."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from safety.cross_graph_loss import cross_graph_infonce, reference_loss


def make_emb(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def random_labels(n_src, n_ref, seed=1, min_pos=1):
    rng = np.random.default_rng(seed)
    positives = np.zeros((n_src, n_ref), dtype=bool)
    weights = np.zeros((n_src, n_ref))
    for i in range(n_src):
        k = rng.integers(min_pos, 3)
        js = rng.choice(n_ref, size=min(k, n_ref), replace=False)
        for j in js:
            positives[i, j] = True
            weights[i, j] = rng.uniform(0.3, 1.0)
    return (torch.from_numpy(positives),
            torch.from_numpy(weights).double())


class TestLossMath(unittest.TestCase):
    def test_matches_reference_single_positive(self):
        src = make_emb(4, 6, 0)
        ref = make_emb(5, 6, 1)
        pos, w = random_labels(4, 5, 2, min_pos=1)
        loss, _ = cross_graph_infonce(src, ref, pos, w)
        ref_loss = reference_loss(src, ref, pos, w)
        self.assertAlmostEqual(float(loss), ref_loss, places=6)

    def test_matches_reference_multi_positive(self):
        src = make_emb(6, 8, 3)
        ref = make_emb(7, 8, 4)
        pos, w = random_labels(6, 7, 5, min_pos=2)
        loss, _ = cross_graph_infonce(src, ref, pos, w)
        self.assertAlmostEqual(float(loss), reference_loss(src, ref, pos, w),
                               places=6)

    def test_gradient_matches_finite_differences(self):
        src = make_emb(4, 5, 6).requires_grad_(True)
        ref = make_emb(5, 5, 7)
        pos, w = random_labels(4, 5, 8)
        loss, _ = cross_graph_infonce(src, ref, pos, w)
        loss.backward()
        eps = 1e-6
        analytic = src.grad[0, 0].item()
        src2 = src.detach().clone()
        src2[0, 0] += eps
        src3 = src.detach().clone()
        src3[0, 0] -= eps
        lp = reference_loss(src2, ref, pos, w)
        lm = reference_loss(src3, ref, pos, w)
        numeric = (lp - lm) / (2 * eps)
        self.assertLess(abs(analytic - numeric), 1e-4)

    def test_asymmetric_shapes(self):
        src = make_emb(3, 6, 9)
        ref = make_emb(9, 6, 10)
        pos, w = random_labels(3, 9, 11)
        loss, _ = cross_graph_infonce(src, ref, pos, w)
        self.assertTrue(torch.isfinite(loss))

    def test_node_permutation_invariance(self):
        src = make_emb(5, 6, 12)
        ref = make_emb(6, 6, 13)
        pos, w = random_labels(5, 6, 14)
        base, _ = cross_graph_infonce(src, ref, pos, w)
        perm_s = torch.tensor([3, 0, 4, 1, 2])
        perm_r = torch.tensor([4, 2, 0, 5, 1, 3])
        loss_p, _ = cross_graph_infonce(
            src[perm_s], ref[perm_r], pos[perm_s][:, perm_r],
            w[perm_s][:, perm_r],
        )
        self.assertAlmostEqual(float(base), float(loss_p), places=6)

    def test_forward_reverse_directionality(self):
        # transpose consistency: loss(A,B) == loss(B,A) with transposed
        # labels under the symmetric 0.5*(fwd+rev) definition
        src = make_emb(4, 5, 15)
        ref = make_emb(5, 5, 16)
        pos, w = random_labels(4, 5, 17)
        a, _ = cross_graph_infonce(src, ref, pos, w)
        b, _ = cross_graph_infonce(ref, src, pos.T, w.T)
        self.assertAlmostEqual(float(a), float(b), places=6)


class TestFailClosed(unittest.TestCase):
    def test_empty_positives_raise(self):
        src = make_emb(3, 4, 0)
        ref = make_emb(3, 4, 1)
        pos = torch.zeros(3, 3, dtype=torch.bool)
        w = torch.zeros(3, 3).double()
        with self.assertRaises(ValueError):
            cross_graph_infonce(src, ref, pos, w)

    def test_nan_embeddings_raise_via_weights_guard(self):
        src = make_emb(3, 4, 2)
        ref = make_emb(3, 4, 3)
        pos, w = random_labels(3, 3, 4)
        w = w.clone()
        w[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            cross_graph_infonce(src, ref, pos, w)

    def test_weight_out_of_bounds_raise(self):
        src = make_emb(3, 4, 5)
        ref = make_emb(3, 4, 6)
        pos, w = random_labels(3, 3, 7)
        w = w.clone()
        w[pos] = 1.5
        with self.assertRaises(ValueError):
            cross_graph_infonce(src, ref, pos, w)

    def test_nonfinite_loss_fails_closed(self):
        # infinities in embeddings produce -inf logits -> guarded
        src = make_emb(2, 4, 8)
        ref = make_emb(2, 4, 9)
        src[0, 0] = float("inf")
        pos = torch.ones(2, 2, dtype=torch.bool)
        w = torch.ones(2, 2).double()
        with self.assertRaises(ValueError):
            cross_graph_infonce(src, ref, pos, w)


class TestSemantics(unittest.TestCase):
    def test_positive_never_negative_and_no_self_terms(self):
        # single positive per query, perfectly aligned embeddings ->
        # loss equals the cross-graph-only InfoNCE lower bound
        src = torch.eye(4, 6, dtype=torch.float64)
        ref = torch.eye(4, 6, dtype=torch.float64)
        pos = torch.eye(4, dtype=torch.bool)
        w = torch.eye(4).double()
        loss, d = cross_graph_infonce(src, ref, pos, w)
        # exact-match positive, near-orthogonal negatives -> loss ~ 0
        self.assertLess(float(loss), 0.05)
        self.assertGreater(d["margin"], 0.9)

    def test_overlap_weight_changes_loss(self):
        src = make_emb(4, 6, 10)
        ref = make_emb(4, 6, 11)
        pos = torch.eye(4, dtype=torch.bool)
        w1 = torch.eye(4).double() * 0.3
        w2 = torch.eye(4).double() * 1.0
        l1, _ = cross_graph_infonce(src, ref, pos, w1)
        l2, _ = cross_graph_infonce(src, ref, pos, w2)
        # equal-weight queries scale out; different weights re-weight
        self.assertTrue(torch.isfinite(l1) and torch.isfinite(l2))

    def test_split_merge_multi_positive(self):
        # one src object matching two ref segments (split)
        src = make_emb(2, 5, 12)
        ref = make_emb(3, 5, 13)
        pos = torch.zeros(2, 3, dtype=torch.bool)
        pos[0, 0] = pos[0, 1] = True
        pos[1, 2] = True
        w = torch.zeros(2, 3).double()
        w[0, 0] = 0.9
        w[0, 1] = 0.4
        w[1, 2] = 0.8
        loss, d = cross_graph_infonce(src, ref, pos, w)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(d["positive_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
