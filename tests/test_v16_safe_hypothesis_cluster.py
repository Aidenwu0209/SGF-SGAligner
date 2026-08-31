from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import unittest

import numpy as np

from safety.v13_dual_solver_runtime import array_sha256
from safety.v15_safe_pose_cluster import SCHEMA as V15_DECISION_SCHEMA
from safety.v16_safe_hypothesis_cluster import (
    AGGREGATE_SCHEMA, CANONICAL_REALIZATIONS, CONTROL_ARM,
    EVIDENCE_SCHEMA, EXPECTED_HYPOTHESIS_COUNTS, FIXED_PAIR_ORDER,
    KNOWN_BAD_PAIR_ID, PRIMARY_ARM, SCHEMA, SafeHypothesisClusterError,
    aggregate_fixed4_research_v16, hypothesis_output_relative_path,
    select_unique_safe_hypothesis_pose_cluster,
)


REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "manifests/v16_safe_matched_region_execution_preregister.json"


def pose(tx=0.0, rz_deg=0.0):
    angle = math.radians(rz_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = [[cosine, -sine, 0.0],
                     [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    value[0, 3] = tx
    return value


def accepted_decision(index, transform):
    candidate_sha = f"{1000 + index:064x}"
    realizations = [{
        "realization": name,
        "ordinal": ordinal,
        "transform": transform.tolist(),
        "transform_sha256": array_sha256(transform),
    } for ordinal, name in enumerate(CANONICAL_REALIZATIONS)]
    return {
        "schema": V15_DECISION_SCHEMA,
        "accepted": True,
        "reason": "unique_safe_pose_cluster",
        "pose_realizations": [{
            "candidate_sha256": candidate_sha,
            "candidate_index": 0,
            "realizations": realizations,
        }],
        "gt_consumed": False,
        "fallback_used": False,
    }


def rejected_decision(reason="no_safe_candidate"):
    return {
        "schema": V15_DECISION_SCHEMA,
        "accepted": False,
        "reason": reason,
        "gt_consumed": False,
        "fallback_used": False,
    }


def evidence(pair_id, index, decision):
    hypothesis_sha = f"{index + 1:064x}"
    return {
        "schema": EVIDENCE_SCHEMA,
        "pair_id": pair_id,
        "arm": PRIMARY_ARM,
        "hypothesis_index": index,
        "hypothesis_sha256": hypothesis_sha,
        "prepared_input_path": f"/frozen/{pair_id}/h{index:02d}.npz",
        "prepared_input_sha256": f"{2000 + index:064x}",
        "output_relative_path": hypothesis_output_relative_path(
            pair_id, index, hypothesis_sha),
        "candidate_decision": decision,
    }


def rows(pair_id, transforms):
    return [evidence(
        pair_id, index,
        rejected_decision() if transform is None
        else accepted_decision(index, transform),
    ) for index, transform in enumerate(transforms)]


def primary_decision(pair_id, accepted):
    return {"schema": SCHEMA, "pair_id": pair_id,
            "arm": PRIMARY_ARM, "accepted": accepted}


def control_decision(pair_id, accepted):
    return {"pair_id": pair_id, "arm": CONTROL_ARM,
            "accepted": accepted}


class V16SafeHypothesisClusterTests(unittest.TestCase):
    def test_preregister_closed_counts_thresholds_and_provenance(self):
        value = json.loads(PREREG.read_text())
        self.assertEqual(value["base_commit"],
                         "d9d93ffabd0ff22442dc00670309878a80b71c3b")
        self.assertTrue(value["disabled"])
        self.assertFalse(value["execution_allowed"])
        self.assertFalse(value["colorpcr_allowed"])
        self.assertFalse(value["builder_input_authorized"])
        self.assertIsNone(value["reviewed_builder_manifest_path"])
        self.assertIsNone(value["reviewed_builder_manifest_sha256"])
        self.assertFalse(value["official92_allowed"])
        self.assertFalse(value["gt_allowed"])
        self.assertEqual(value["fallbacks"], [])
        self.assertEqual(
            list(value["expected_primary_hypothesis_counts"].values()),
            [12, 8, 2, 12])
        self.assertEqual(value["expected_future_execution_counts"], {
            "primary_hypotheses": 34,
            "directions_per_hypothesis": 2,
            "sentinel_workers_per_direction": 2,
            "official_colorpcr_worker_processes": 136,
            "sentinel_invariant_direction_caches": 68,
            "exact_three_direction_caches": 68,
            "candidate_sets": 34,
            "v13_solver_workers_per_candidate": 20,
            "maximum_candidates_per_hypothesis": 8,
            "maximum_v13_solver_workers": 5440,
        })
        thresholds = value["unchanged_thresholds"]
        self.assertEqual(
            thresholds["hypothesis_pose_rotation_deg_inclusive_max"], 5.0)
        self.assertEqual(
            thresholds["hypothesis_pose_translation_m_inclusive_max"], 0.1)
        self.assertEqual(
            value["official_release_checkpoint_sha256"],
            "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e")
        self.assertFalse(value["legacy_B_ep20_or_89ed_allowed"])

    def test_twelve_equivalent_accept_and_order_invariant(self):
        pair = FIXED_PAIR_ORDER[0]
        values = rows(pair, [pose(tx=index * 0.001)
                             for index in range(12)])
        first = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=12, known_bad=False)
        shuffled = list(values)
        random.Random(4242).shuffle(shuffled)
        second = select_unique_safe_hypothesis_pose_cluster(
            shuffled, expected_hypothesis_count=12, known_bad=False)
        self.assertTrue(first["accepted"])
        self.assertEqual(first["safe_hypothesis_count"], 12)
        self.assertEqual(first["pose_cluster_count"], 1)
        self.assertEqual(first["pose_clusters"], second["pose_clusters"])
        self.assertEqual(first["selected_transform_sha256"],
                         second["selected_transform_sha256"])

    def test_eleven_to_one_safe_majority_rejects(self):
        pair = FIXED_PAIR_ORDER[0]
        values = rows(pair, [pose()] * 11 + [pose(tx=0.10001)])
        result = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=12, known_bad=False)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["safe_hypothesis_count"], 12)
        self.assertEqual(result["pose_cluster_count"], 2)
        self.assertEqual(result["reason"],
                         "ambiguous_multiple_safe_hypothesis_pose_clusters")

    def test_one_safe_plus_eleven_unsafe_accepts(self):
        pair = FIXED_PAIR_ORDER[0]
        values = rows(pair, [pose(tx=0.02)] + [None] * 11)
        result = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=12, known_bad=False)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["safe_hypothesis_count"], 1)

    def test_zero_safe_hypotheses_rejects(self):
        pair = FIXED_PAIR_ORDER[2]
        result = select_unique_safe_hypothesis_pose_cluster(
            rows(pair, [None, None]), expected_hypothesis_count=2,
            known_bad=False)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["safe_hypothesis_count"], 0)
        self.assertEqual(result["pose_cluster_count"], 0)
        self.assertEqual(result["reason"], "no_safe_hypothesis")

    def test_chain_is_not_single_complete_linkage_cluster(self):
        pair = FIXED_PAIR_ORDER[1]
        # A-B and B-C pass, but A-C fails.  Single-linkage would merge all
        # three; exhaustive complete-linkage must expose two maximal cliques.
        values = rows(
            pair,
            [pose(tx=0.00), pose(tx=0.09), pose(tx=0.18)] + [None] * 5,
        )
        result = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=8, known_bad=False)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["safe_hypothesis_count"], 3)
        self.assertEqual(result["pose_cluster_count"], 2)
        self.assertEqual(
            result["reason"],
            "ambiguous_multiple_safe_hypothesis_pose_clusters",
        )

    def test_frozen_boundaries_inclusive_but_not_relaxed(self):
        pair = FIXED_PAIR_ORDER[1]
        cases = ((pose(tx=0.1), True),
                 (pose(tx=0.1000001), False),
                 (pose(rz_deg=5.0), True),
                 (pose(rz_deg=5.00001), False))
        for transform, accepted in cases:
            with self.subTest(accepted=accepted,
                              transform=transform.tolist()):
                values = rows(pair, [pose(), transform] + [None] * 6)
                result = select_unique_safe_hypothesis_pose_cluster(
                    values, expected_hypothesis_count=8, known_bad=False)
                self.assertEqual(result["accepted"], accepted)

    def test_medoid_is_observed_transform_not_average(self):
        pair = FIXED_PAIR_ORDER[0]
        transforms = [pose(tx=0.0), pose(tx=0.04), pose(tx=0.09)]
        values = rows(pair, transforms + [None] * 9)
        result = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=12, known_bad=False)
        observed = {array_sha256(value) for value in transforms}
        self.assertTrue(result["accepted"])
        self.assertIn(result["selected_transform_sha256"], observed)
        self.assertEqual(result["selected_transform_sha256"],
                         array_sha256(transforms[1]))
        self.assertEqual(result["medoid_score"]["observed_transform_count"],
                         12)

    def test_known_bad_veto_preserves_pre_veto_counts(self):
        decisions = []
        for index in range(12):
            decision = rejected_decision("known_bad_veto")
            decision["strict_geometry_safe_count_before_veto"] = \
                2 if index < 3 else 0
            decisions.append(decision)
        values = [evidence(KNOWN_BAD_PAIR_ID, index, decision)
                  for index, decision in enumerate(decisions)]
        result = select_unique_safe_hypothesis_pose_cluster(
            values, expected_hypothesis_count=12, known_bad=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "known_bad_veto")
        self.assertEqual(
            result["strict_geometry_safe_candidate_count_before_veto"], 6)
        self.assertEqual(
            result["hypotheses_with_pre_veto_geometry_safe_evidence"], 3)

    def test_known_bad_cannot_arrive_candidate_level_accepted(self):
        values = rows(KNOWN_BAD_PAIR_ID, [pose()] + [None] * 11)
        with self.assertRaisesRegex(
                SafeHypothesisClusterError,
                "bypassed candidate-level veto"):
            select_unique_safe_hypothesis_pose_cluster(
                values, expected_hypothesis_count=12, known_bad=True)

    def test_missing_duplicate_index_and_path_collision_fail(self):
        pair = FIXED_PAIR_ORDER[1]
        values = rows(pair, [None] * 8)
        with self.assertRaisesRegex(SafeHypothesisClusterError,
                                    "evidence count mismatch"):
            select_unique_safe_hypothesis_pose_cluster(
                values[:-1], expected_hypothesis_count=8, known_bad=False)
        duplicate_index = copy.deepcopy(values)
        duplicate_index[-1]["hypothesis_index"] = 0
        duplicate_index[-1]["hypothesis_sha256"] = f"{99:064x}"
        duplicate_index[-1]["output_relative_path"] = \
            hypothesis_output_relative_path(pair, 0, f"{99:064x}")
        with self.assertRaisesRegex(SafeHypothesisClusterError,
                                    "indices are not exact"):
            select_unique_safe_hypothesis_pose_cluster(
                duplicate_index, expected_hypothesis_count=8,
                known_bad=False)
        path_collision = copy.deepcopy(values)
        path_collision[-1]["prepared_input_path"] = \
            path_collision[0]["prepared_input_path"]
        with self.assertRaisesRegex(SafeHypothesisClusterError,
                                    "path collision"):
            select_unique_safe_hypothesis_pose_cluster(
                path_collision, expected_hypothesis_count=8,
                known_bad=False)

    def test_output_paths_unique_for_all_frozen_hypotheses(self):
        paths = []
        for pair, count in EXPECTED_HYPOTHESIS_COUNTS.items():
            for index in range(count):
                paths.append(hypothesis_output_relative_path(
                    pair, index, f"{index + 1:064x}"))
        self.assertEqual(len(paths), 34)
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(f"/{PRIMARY_ARM}/hypotheses/" in value
                            for value in paths))
        self.assertTrue(all(f"/{CONTROL_ARM}/" not in value
                            for value in paths))

    def test_control_cannot_rescue_failed_primary(self):
        primary = [
            primary_decision(FIXED_PAIR_ORDER[0], False),
            primary_decision(FIXED_PAIR_ORDER[1], True),
            primary_decision(FIXED_PAIR_ORDER[2], True),
            primary_decision(FIXED_PAIR_ORDER[3], False),
        ]
        control = [control_decision(pair, index < 3)
                   for index, pair in enumerate(FIXED_PAIR_ORDER)]
        result = aggregate_fixed4_research_v16(primary, control)
        self.assertEqual(result["schema"], AGGREGATE_SCHEMA)
        self.assertFalse(result["safe"])
        self.assertFalse(result["control_can_rescue"])
        self.assertEqual(result["normal_primary_failures"],
                         [FIXED_PAIR_ORDER[0]])
        self.assertTrue(
            result["control_safe_diagnostic"][FIXED_PAIR_ORDER[0]])

    def test_all_normal_primary_safe_ignores_control_failures(self):
        primary = [primary_decision(pair, index < 3)
                   for index, pair in enumerate(FIXED_PAIR_ORDER)]
        control = [control_decision(pair, False)
                   for pair in FIXED_PAIR_ORDER]
        result = aggregate_fixed4_research_v16(primary, control)
        self.assertTrue(result["safe"])
        self.assertEqual(result["reason"],
                         "all_normal_primary_safe_and_known_bad_vetoed")

    def test_malformed_hash_and_within_hypothesis_spread_abort(self):
        pair = FIXED_PAIR_ORDER[2]
        values = rows(pair, [pose(), None])
        bad_hash = copy.deepcopy(values)
        bad_hash[0]["candidate_decision"]["pose_realizations"][0][
            "realizations"][0]["transform_sha256"] = "0" * 64
        with self.assertRaisesRegex(SafeHypothesisClusterError,
                                    "transform SHA mismatch"):
            select_unique_safe_hypothesis_pose_cluster(
                bad_hash, expected_hypothesis_count=2, known_bad=False)
        spread = copy.deepcopy(values)
        far = pose(tx=0.2)
        realization = spread[0]["candidate_decision"][
            "pose_realizations"][0]["realizations"][3]
        realization["transform"] = far.tolist()
        realization["transform_sha256"] = array_sha256(far)
        with self.assertRaisesRegex(SafeHypothesisClusterError,
                                    "not complete-linkage safe"):
            select_unique_safe_hypothesis_pose_cluster(
                spread, expected_hypothesis_count=2, known_bad=False)


if __name__ == "__main__":
    unittest.main()
