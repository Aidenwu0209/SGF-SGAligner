import copy
import unittest

import numpy as np

from safety.matched_region_hypotheses import (
    MatchedRegionConfig,
    MatchedRegionError,
    full_scene_union,
    generate_matched_region_hypotheses,
    run_independent_bidirectional_geot,
    union_hypothesis_surfaces,
    unique_safe_hypothesis,
)


def cloud(centre, count=20):
    offsets = np.linspace(-0.02, 0.02, count)[:, None] * np.array([[1, 2, 3]])
    return np.asarray(centre, dtype=np.float32) + offsets.astype(np.float32)


class MatchedRegionHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.surfaces = {
            0: cloud((0, 0, 0)), 1: cloud((1, 0, 0)),
            2: cloud((2, 0, 0)), 3: cloud((3, 0, 0)),
            4: cloud((0.03, 0, 0)), 5: cloud((1.03, 0, 0)),
            6: cloud((2.03, 0, 0)), 7: cloud((3.03, 0, 0)),
        }
        self.candidates = [
            {"source_index": i, "reference_index": i + 4,
             "forward_cross_rank": 1, "reverse_cross_rank": 1,
             "worst_cross_rank": 1, "rank_sum": 2}
            for i in range(4)
        ]

    def test_builds_three_plus_member_hypotheses(self):
        result = generate_matched_region_hypotheses(
            self.candidates, self.surfaces, 4)
        self.assertTrue(result)
        self.assertTrue(all(3 <= row["member_count"] <= 6 for row in result))
        self.assertEqual(result[0]["member_count"], 4)

    def test_order_invariant_and_deterministic(self):
        first = generate_matched_region_hypotheses(
            self.candidates, self.surfaces, 4)
        second = generate_matched_region_hypotheses(
            copy.deepcopy(self.candidates)[::-1], self.surfaces, 4)
        self.assertEqual(first, second)

    def test_repeated_reference_cannot_share_hypothesis(self):
        candidates = self.candidates + [{
            "source_index": 3, "reference_index": 4,
            "forward_cross_rank": 2, "reverse_cross_rank": 2,
            "worst_cross_rank": 2, "rank_sum": 4,
        }]
        result = generate_matched_region_hypotheses(
            candidates, self.surfaces, 4)
        for hypothesis in result:
            references = [pair[1] for pair in hypothesis["members"]]
            self.assertEqual(len(references), len(set(references)))

    def test_inconsistent_shape_is_not_combined(self):
        surfaces = dict(self.surfaces)
        surfaces[7] = cloud((30, 0, 0))
        result = generate_matched_region_hypotheses(
            self.candidates, surfaces, 4,
            config=MatchedRegionConfig(local_neighbors=3))
        self.assertTrue(all([3, 7] not in row["members"] for row in result))

    def test_union_is_canonical_under_member_order(self):
        hypothesis = {"members": [[2, 6], [0, 4], [1, 5]]}
        first = union_hypothesis_surfaces(hypothesis, self.surfaces)
        hypothesis["members"].reverse()
        second = union_hypothesis_surfaces(hypothesis, self.surfaces)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_bidirectional_calls_are_independent_and_swapped(self):
        calls = []
        def runner(source, reference):
            calls.append((source.copy(), reference.copy()))
            return "ok", {"call": len(calls)}
        source, reference = full_scene_union(self.surfaces, 4)
        result = run_independent_bidirectional_geot(source, reference, runner)
        self.assertTrue(result["both_ok"])
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0][0], calls[1][1])
        np.testing.assert_array_equal(calls[0][1], calls[1][0])

    def test_bidirectional_failure_is_fail_closed(self):
        calls = []
        def runner(source, reference):
            calls.append(1)
            return ("ok", {}) if len(calls) == 1 else ("runtime_error", {})
        source, reference = full_scene_union(self.surfaces, 4)
        result = run_independent_bidirectional_geot(source, reference, runner)
        self.assertFalse(result["both_ok"])

    def test_unique_safe_gate_rejects_zero_multiple_and_known_bad(self):
        safe = {"forward_status": "ok", "reverse_status": "ok",
                "cross_direction_consistent": True, "rule_b_safe": True,
                "q4_stable": True, "hypothesis_sha256": "abc"}
        self.assertFalse(unique_safe_hypothesis([])["accepted"])
        self.assertFalse(unique_safe_hypothesis([safe, safe])["accepted"])
        self.assertFalse(unique_safe_hypothesis([safe], known_bad=True)["accepted"])
        self.assertTrue(unique_safe_hypothesis([safe])["accepted"])

    def test_malformed_surface_fails_closed(self):
        broken = dict(self.surfaces)
        broken[0] = np.zeros((3, 2))
        with self.assertRaises(MatchedRegionError):
            generate_matched_region_hypotheses(
                self.candidates, broken, 4)

    def test_missing_candidate_surface_fails_closed(self):
        broken = dict(self.surfaces)
        del broken[7]
        with self.assertRaises(MatchedRegionError):
            generate_matched_region_hypotheses(
                self.candidates, broken, 4)


if __name__ == "__main__":
    unittest.main()
