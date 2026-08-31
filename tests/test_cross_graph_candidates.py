import unittest

from safety.cross_graph_candidates import (
    CrossGraphCandidateConfig,
    CrossGraphCandidateError,
    candidate_fingerprint,
    cross_graph_candidates,
)


class CrossGraphCandidateTests(unittest.TestCase):
    def test_filters_before_top_k_and_keeps_cross_graph_candidate(self):
        # Node 0 has three same-graph neighbours before any ref. Old global
        # top-3 filtering yields zero, but cross-graph top-k retains refs.
        ranks = [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 2, 4, 3, 5],
            [2, 1, 0, 5, 4, 3],
            [3, 0, 1, 2, 4, 5],
            [4, 1, 0, 2, 3, 5],
            [5, 2, 0, 1, 3, 4],
        ]
        rows = cross_graph_candidates(ranks, 3)
        self.assertIn((0, 3), {
            (row["source_index"], row["reference_index"]) for row in rows})

    def test_mutual_constraint_removes_one_way_candidate(self):
        ranks = [
            [0, 2, 1, 3], [1, 3, 0, 2],
            [2, 1, 0, 3], [3, 0, 1, 2],
        ]
        config = CrossGraphCandidateConfig(cross_graph_k=1)
        rows = cross_graph_candidates(ranks, 2, config)
        self.assertEqual(rows, [])

    def test_resource_cap_uses_stable_reciprocal_rank_order(self):
        ranks = [
            [0, 3, 4, 5, 1, 2], [1, 4, 3, 5, 0, 2],
            [2, 5, 3, 4, 0, 1], [3, 0, 1, 2, 4, 5],
            [4, 1, 0, 2, 3, 5], [5, 2, 0, 1, 3, 4],
        ]
        config = CrossGraphCandidateConfig(
            cross_graph_k=3, max_candidates_per_pair=2)
        first = cross_graph_candidates(ranks, 3, config)
        second = cross_graph_candidates(ranks, 3, config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["worst_cross_rank"], 1)

    def test_no_duplicates_and_indices_are_cross_graph(self):
        ranks = [
            [0, 2, 3, 1], [1, 3, 2, 0],
            [2, 0, 1, 3], [3, 1, 0, 2],
        ]
        rows = cross_graph_candidates(ranks, 2)
        pairs = [(row["source_index"], row["reference_index"])
                 for row in rows]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(a < 2 <= b for a, b in pairs))

    def test_fingerprint_changes_with_policy(self):
        ranks = [[0, 1], [1, 0]]
        rows = cross_graph_candidates(ranks, 1)
        a = candidate_fingerprint(rows, CrossGraphCandidateConfig())
        b = candidate_fingerprint(rows, CrossGraphCandidateConfig(
            max_candidates_per_pair=47))
        self.assertNotEqual(a, b)

    def test_malformed_rank_list_fails_closed(self):
        with self.assertRaises(CrossGraphCandidateError):
            cross_graph_candidates([[0, 1], [1, 1]], 1)


if __name__ == "__main__":
    unittest.main()
