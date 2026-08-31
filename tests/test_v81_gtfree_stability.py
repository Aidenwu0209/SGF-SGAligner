import copy
import random
import unittest

import numpy as np

from safety.v81_gtfree_stability import (
    compare_gtfree_policies,
    evaluate_v81_stability,
)


def rule_b(features):
    return ["synthetic_rule_b_reject"] if features.get("reject") else []


def transform(x=0.0, yaw_deg=0.0):
    theta = np.radians(yaw_deg)
    value = np.eye(4)
    value[:3, :3] = [[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0, 0.0, 1.0]]
    value[0, 3] = x
    return value.tolist()


def worker(outer, direction, replicate, *, x=0.0, yaw=0.0,
           reject=False, signature=None):
    reasons = ["synthetic_rule_b_reject"] if reject else []
    final = transform(x, yaw)
    return {
        "status": "ok", "direction": direction, "replicate": replicate,
        "raw_transform": final, "final_transform": final,
        "evidence_sha256": signature or f"{outer}-{direction}-{replicate}",
        "permutation_provenance_sha256": f"p-{outer}-{direction}-{replicate}",
        "rule_b_features": {"reject": reject},
        "rule_b_accepted": not reject,
        "decision": {"rejection_reasons": reasons},
        "icp": {"trace": [{
            "fixed_correspondence_rmse_before_m": 0.02,
            "fixed_correspondence_rmse_after_m": 0.01,
            "update_rotation_deg": 0.01,
            "update_translation_m": 0.0001,
        }]},
    }


def stable_outers():
    return [[worker(outer, direction, replicate,
                    x=0.0001 * (outer * 5 + replicate))
             for direction in ("forward", "reverse")
             for replicate in range(5)] for outer in range(2)]


class V81GTFreeStabilityTest(unittest.TestCase):
    def test_stable_pool_passes_every_jackknife(self):
        result = evaluate_v81_stability(stable_outers(), rule_b)
        self.assertTrue(result["usable_for_reconstruction"])
        self.assertEqual(result["jackknife"]["scenario_count"], 20)
        self.assertEqual(result["jackknife"]["passed"], 20)
        self.assertEqual(result["pool"]["member_safety"]["forward_pass"], 10)
        self.assertEqual(result["pool"]["member_safety"]["reverse_pass"], 10)

    def test_two_outer_medoid_flip_is_recovered_by_pool(self):
        rows = stable_outers()
        # Outer 0's lexicographic medoid is the rejected row; the pooled
        # medoid is a safe outer-1 observation.  Nine independent members
        # remain safe and all leave-one-out replays retain q=8.
        for row in rows[0]:
            row["evidence_sha256"] = "z" + row["evidence_sha256"]
            row["permutation_provenance_sha256"] = (
                "z" + row["permutation_provenance_sha256"])
        bad_rows = [row for row in rows[0] if row["replicate"] == 2]
        for bad in bad_rows:
            bad["evidence_sha256"] = "m-bad-" + bad["direction"]
            bad["permutation_provenance_sha256"] = (
                "000-bad-" + bad["direction"])
            bad["rule_b_features"] = {"reject": True}
            bad["rule_b_accepted"] = False
            bad["decision"] = {
                "rejection_reasons": ["synthetic_rule_b_reject"]}
        for row in rows[1]:
            row["evidence_sha256"] = "a" + row["evidence_sha256"]
        comparison = compare_gtfree_policies(rows, rule_b)
        self.assertEqual(comparison["outer_v8"], [False, True])
        self.assertFalse(comparison["dual_outer_unanimity"])
        self.assertTrue(comparison["v81_recommended"])

    def test_known_bad_style_bimodal_geometry_is_vetoed(self):
        rows = stable_outers()
        # Six observations at identity and four at a distant alternative do
        # not meet the pre-registered q=9 component gate.
        for direction in ("forward", "reverse"):
            changed = 0
            for outer in rows:
                for row in outer:
                    if row["direction"] == direction and changed < 4:
                        row["final_transform"] = transform(0.4, 15.0)
                        row["raw_transform"] = row["final_transform"]
                        changed += 1
        result = evaluate_v81_stability(rows, rule_b)
        self.assertFalse(result["usable_for_reconstruction"])
        self.assertIn("pooled_geometry_unusable", result["rejection_reasons"])

    def test_isolated_lucky_medoid_fails_member_vote(self):
        rows = stable_outers()
        for outer in rows:
            for row in outer:
                row["rule_b_features"] = {"reject": True}
                row["rule_b_accepted"] = False
                row["decision"] = {
                    "rejection_reasons": ["synthetic_rule_b_reject"]}
        for direction in ("forward", "reverse"):
            candidates = [row for outer in rows for row in outer
                          if row["direction"] == direction]
            lucky = candidates[0]
            lucky["evidence_sha256"] = "000-" + direction
            lucky["rule_b_features"] = {"reject": False}
            lucky["rule_b_accepted"] = True
            lucky["decision"] = {"rejection_reasons": []}
        result = evaluate_v81_stability(rows, rule_b)
        self.assertFalse(result["usable_for_reconstruction"])
        self.assertIn("forward_member_safety_vote_not_met",
                      result["rejection_reasons"])
        self.assertIn("reverse_member_safety_vote_not_met",
                      result["rejection_reasons"])

    def test_invalid_transform_fails_closed(self):
        rows = stable_outers()
        rows[0][0]["final_transform"] = [[float("nan")]]
        result = evaluate_v81_stability(rows, rule_b)
        self.assertFalse(result["usable_for_reconstruction"])

    def test_input_order_is_irrelevant(self):
        rows = stable_outers()
        expected = evaluate_v81_stability(rows, rule_b)
        shuffled = copy.deepcopy(rows)
        random.Random(42).shuffle(shuffled[0])
        random.Random(43).shuffle(shuffled[1])
        observed = evaluate_v81_stability(shuffled, rule_b)
        self.assertEqual(expected["usable_for_reconstruction"],
                         observed["usable_for_reconstruction"])
        self.assertEqual(expected["selected_observed_forward_medoid"],
                         observed["selected_observed_forward_medoid"])
        self.assertEqual(expected["jackknife"]["passed"],
                         observed["jackknife"]["passed"])

    def test_missing_worker_evidence_identity_fails_closed(self):
        rows = stable_outers()
        rows[1][0]["evidence_sha256"] = ""
        result = evaluate_v81_stability(rows, rule_b)
        self.assertFalse(result["usable_for_reconstruction"])
        self.assertIn("worker_evidence_identity_missing",
                      result["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
