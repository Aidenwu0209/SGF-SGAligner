import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "manifests/v16_matched_region_colorpcr_preregister.json"


class V16PreregisterTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(PATH.read_text())

    def test_stage_is_frozen_disabled_and_non_independent(self):
        self.assertTrue(self.p["frozen"])
        self.assertTrue(self.p["disabled"])
        self.assertFalse(self.p["independent_evidence"])
        self.assertFalse(self.p["real_pilot_allowed"])
        self.assertFalse(self.p["colorpcr_consumption_allowed"])
        self.assertFalse(self.p["rank_hypotheses_authoritative_for_release"])

    def test_forbidden_authorities_are_closed(self):
        for key in ("official92_allowed", "gt_allowed",
                    "semantic_gt_selection_posthoc_labels_allowed",
                    "posthoc_allowed"):
            self.assertFalse(self.p[key])
        self.assertEqual(self.p["fallbacks"], [])
        self.assertTrue(self.p["raw_inseg_instance_membership_key_allowed"])
        self.assertIn("never cross-scan identity",
                      self.p["raw_instance_membership_scope"])

    def test_thresholds_are_frozen(self):
        g = self.p["unchanged_downstream_gates"]
        self.assertEqual(g["rule_b"], "unchanged")
        self.assertEqual(g["icp"], "unchanged")
        self.assertEqual(g["residual_threshold_m"], 0.1)
        self.assertEqual(g["min_support"], 40)
        self.assertEqual(g["pose_cluster_rotation_deg"], 5.0)
        self.assertEqual(g["pose_cluster_translation_m"], 0.1)

    def test_repository_source_hashes_are_bound(self):
        f = self.p["frozen_inputs"]
        local = {
            f["downstream_official_release_checkpoint"]["path"]:
                f["downstream_official_release_checkpoint"]["sha256"],
            f["canonical_builder"]["path"]: f["canonical_builder"]["sha256"],
            **f["v13_preprocessing_sources"],
        }
        for rel, expected in local.items():
            got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(got, expected, rel)

    def test_checkpoint_domains_are_explicitly_separate_and_blocked(self):
        f = self.p["frozen_inputs"]
        rank = f["rank_source_checkpoint"]
        release = f["downstream_official_release_checkpoint"]
        self.assertEqual(rank["checkpoint_id"], "B")
        self.assertEqual(rank["sha256"],
                         "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200")
        self.assertEqual(release["sha256"],
                         "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e")
        self.assertNotEqual(rank["sha256"], release["sha256"])
        self.assertFalse(self.p["checkpoint_domains_match"])
        self.assertIn("rerank", self.p["checkpoint_domain_blocker"])

    def test_builder_does_not_claim_direct_voxel10_fps512(self):
        contract = self.p["deterministic_contract"]
        self.assertIn("raw and voxel10 prepared arrays only",
                      contract["preprocessing"])
        self.assertIn("final coarsest stage", contract["coarsest_cap512"])
        evidence = self.p["required_hypothesis_evidence"]
        self.assertFalse(any("FPS512 arrays" in row for row in evidence))

    def test_frozen_worker_applies_cap_only_at_final_multilevel_stage(self):
        worker = (ROOT / "scripts/v13_colorpcr_official_worker.py").read_text()
        start = worker.index("def precompute_capped(")
        end = worker.index("def registration_collate_capped(", start)
        body = worker[start:end]
        subsample = body.index("grid_subsample_dps(")
        final_stage = body.index("if stage==num_stages-1:")
        cap = body.index("cap_coarsest_points_fps(")
        self.assertLess(subsample, final_stage)
        self.assertLess(final_stage, cap)
        builder = (ROOT / "src/safety/v16_matched_region_colorpcr.py").read_text()
        self.assertNotIn('arrays[f"{side}_fps512_', builder)


if __name__ == "__main__":
    unittest.main()
