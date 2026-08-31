import unittest

from safety.v13_fixed4_aggregate import AUTHORITY, aggregate_fixed4


NORMAL = ("n0_to_r0", "n1_to_r1", "n2_to_r2")
KNOWN = "bad_to_ref"
SPEC = {"normal_pair_ids": list(NORMAL), "known_bad_pair_id": KNOWN,
        "primary_arm": "sgf_selected_union", "control_arm": "fullscan",
        "control_can_rescue": False,
        "strict_gate_runtime_pins": {"v7_registration_pilot.py":"a"*64,
                                     "decision_features.py":"b"*64,
                                     "v8_stage_order_consensus.py":"c"*64}}


def medoid_evidence():
    evidence = {
        "usable": True,
        "rule_b_features": {
            "overlap_10cm": .8, "median_residual_m": .01,
            "symmetric_trimmed_chamfer_m": .01, "icp_converged": True,
            "icp_update_translation_m": 0., "icp_update_rotation_deg": 0.,
            "icp_fitness": .9, "ransac_inliers": 40,
            "spatial_extent_m": 2., "bidirectional_available": True,
            "bidirectional_rotation_deg": 0.,
            "bidirectional_translation_m": 0.,
        },
        "recorded_rule_b_decision": {
            "usable_for_reconstruction": True, "rejection_reasons": [],
            "rule": "fix2-B-unchanged", "thresholds": {}},
        "icp": {"transform": [[1., 0., 0., 0.], [0., 1., 0., 0.],
                                [0., 0., 1., 0.], [0., 0., 0., 1.]],
                "converged": True, "fitness": 1., "rmse_m": 0.,
                "update_rotation_deg": 0., "update_translation_m": 0.,
                "trace": [{"fixed_correspondence_rmse_before_m": .1,
                           "fixed_correspondence_rmse_after_m": .05}]},
        "surface_source_point_count": 60,
        "surface_reference_point_count": 61,
        "surface_source_sha256": "d" * 64,
        "surface_reference_sha256": "e" * 64,
    }
    return {f"{solver}/{direction}": dict(evidence)
            for solver in ("pointdsc", "pygcransac")
            for direction in ("forward", "reverse")}


def row(pair_id, arm, safe, *, veto=False):
    return {"schema": "v13-strict-pair-gate-v1", "pair_id": pair_id,
            "arm": arm, "safe": safe, "gate_authority": AUTHORITY,
            "known_bad_veto": veto,
            "runtime_receipt":{"mode":"SEALED_FORMAL_RUNTIME",
                               "source_sha256":dict(SPEC["strict_gate_runtime_pins"])},
            "medoid_safety": medoid_evidence(),
            "rule_b_evaluator":"evaluate_rule_b","rule_c_claimed":False,
            "reason": "known_bad_veto" if veto else ("pass" if safe else "failed"),
            "bound_known_bad_pair_id": KNOWN if veto else None}


def matrix():
    rows = []
    for pair_id in NORMAL:
        rows += [row(pair_id, "sgf_selected_union", True),
                 row(pair_id, "fullscan", True)]
    rows += [row(KNOWN, "sgf_selected_union", False, veto=True),
             row(KNOWN, "fullscan", False, veto=True)]
    return rows


class Tests(unittest.TestCase):
    def test_global_pass_requires_all_primary_and_knownbad_veto(self):
        result = aggregate_fixed4(matrix(), SPEC)
        self.assertTrue(result["safe"])
        self.assertEqual(result["normal_primary_failures"], [])
        self.assertTrue(result["known_bad_veto"])

    def test_normal_primary_failure_cannot_be_rescued_by_control(self):
        rows = matrix()
        rows[0] = row(NORMAL[0], "sgf_selected_union", False)
        result = aggregate_fixed4(rows, SPEC)
        self.assertFalse(result["safe"])
        self.assertEqual(result["normal_primary_failures"], [NORMAL[0]])
        self.assertEqual(result["control_rescue_candidates_not_used"], [NORMAL[0]])

    def test_control_failure_does_not_veto_a_safe_primary(self):
        rows = matrix()
        rows[1] = row(NORMAL[0], "fullscan", False)
        result = aggregate_fixed4(rows, SPEC)
        self.assertTrue(result["safe"])
        self.assertFalse(result["control_safe_diagnostic"][NORMAL[0]])

    def test_knownbad_must_be_bound_and_vetoed(self):
        rows = matrix()
        rows[-2] = row(KNOWN, "sgf_selected_union", True)
        result = aggregate_fixed4(rows, SPEC)
        self.assertFalse(result["safe"])
        self.assertEqual(result["reason"], "known_bad_veto_failed")

    def test_knownbad_control_acceptance_is_a_global_safety_failure(self):
        rows = matrix()
        rows[-1] = row(KNOWN, "fullscan", True)
        result = aggregate_fixed4(rows, SPEC)
        self.assertFalse(result["safe"])
        self.assertTrue(result["known_bad_veto_by_arm"]["sgf_selected_union"])
        self.assertFalse(result["known_bad_veto_by_arm"]["fullscan"])

    def test_test_only_injected_pair_evidence_cannot_enter_aggregate(self):
        rows=matrix();rows[0]["runtime_receipt"]={"mode":"TEST_ONLY_INJECTION_UNSEALED"}
        with self.assertRaisesRegex(Exception,"test-only"):
            aggregate_fixed4(rows,SPEC)

    def test_missing_rule_b_or_icp_evidence_cannot_enter_aggregate(self):
        rows = matrix()
        del rows[0]["medoid_safety"]["pointdsc/forward"]["rule_b_features"]
        with self.assertRaisesRegex(Exception, "auditable strict-gate fields"):
            aggregate_fixed4(rows, SPEC)

        rows = matrix()
        rows[0]["medoid_safety"]["pointdsc/forward"]["icp"] = {"trace": []}
        with self.assertRaisesRegex(Exception, "lacks Rule-B/ICP evidence"):
            aggregate_fixed4(rows, SPEC)


if __name__ == "__main__":
    unittest.main()
