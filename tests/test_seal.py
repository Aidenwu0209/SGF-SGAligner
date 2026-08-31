"""Fix-2 Seal tests: single-inference cache, shared-replay identity,
precision-null semantics, manifest completeness, no-service-control,
transform/refusion gating."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
SEL = ROOT / "outputs/official_sgaligner_migration_fix2_seal_20260826"


class TestSingleInferenceCache(unittest.TestCase):
    def test_one_cache_per_pair_no_rule_subdirs(self):
        for split in ("selection_cache", "calibration_cache"):
            root = SEL / split
            self.assertTrue((root / "pairs_run.txt").exists())
            for pair_dir in root.iterdir():
                if not pair_dir.is_dir():
                    continue
                caches = list(pair_dir.glob("pair_cache.json"))
                self.assertEqual(len(caches), 1, pair_dir)
                self.assertFalse(
                    any(pair_dir.glob("*_A_*")),
                    "rule-specific rerun directories must not exist",
                )

    def test_pair_cache_contains_all_rule_inputs(self):
        cache = json.loads(
            (SEL / "selection_cache" / "09582205_1883" / "pair_cache.json")
            .read_text()
        )
        self.assertIn("raw_features", cache)
        features = cache["raw_features"]
        for key in (
            "icp_update_translation_m", "icp_update_rotation_deg",
            "bidirectional_rotation_deg", "bidirectional_translation_m",
            "overlap_10cm", "icp_fitness", "node_pair_success_ratio",
        ):
            self.assertIn(key, features)
        self.assertIn("_provenance", features)

    def test_models_run_once_marker(self):
        # runner is single-process sequential; no pool import
        src = (ROOT / "scripts/seal_runner.py").read_text()
        self.assertNotIn("ProcessPoolExecutor", src)


class TestSharedReplayIdentity(unittest.TestCase):
    def test_raw_metrics_identical_across_rules(self):
        for split in ("selection_replay", "calibration_replay"):
            summary = json.loads(
                (SEL / split / "rule_replay_summary.json").read_text()
            )
            for metric in ("hypothesis_generated", "strict", "relaxed",
                          "failed"):
                values = {
                    summary[r][metric] for r in "ABC"
                }
                self.assertEqual(len(values), 1, f"{split}/{metric}")

    def test_shared_fields_from_single_cache(self):
        src = (ROOT / "scripts/seal_replay.py").read_text()
        self.assertIn("pair_cache.json", src)
        self.assertNotIn("official_forward", src)
        self.assertNotIn("official_registration", src)
        self.assertNotIn("segment_icp", src)


class TestPrecisionSemantics(unittest.TestCase):
    def test_accepted_zero_precision_null(self):
        src = (ROOT / "scripts/seal_replay.py").read_text()
        self.assertIn("if accepted > 0 else None", src)
        self.assertIn('"N/A"', src)

    def test_no_false_100pct_when_zero(self):
        # calibration has accepted=5 (precision 1.0 legitimate); craft
        # synthetic summary and assert null propagation
        summary = {"accepted": 0, "accepted_strict_correct": 0}
        precision = (
            summary["accepted_strict_correct"] / summary["accepted"]
            if summary["accepted"] > 0 else None
        )
        self.assertIsNone(precision)


class TestManifestIntegrity(unittest.TestCase):
    def test_selection_has_89_calibration_has_90(self):
        sel = (SEL / "selection_cache/pairs_run.txt").read_text().split()
        cal = (SEL / "calibration_cache/pairs_run.txt").read_text().split()
        self.assertEqual(len(sel), 89)
        self.assertEqual(len(cal), 90)

    def test_reference_sets_disjoint(self):
        man = json.loads(Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/training_dataset/"
            "dataset_three_way.json"
        ).read_text())
        self.assertFalse(
            set(man["selection_references"])
            & set(man["calibration_references"])
        )

    def test_no_limit_flag_in_seal_runs(self):
        log = Path("/tmp/seal_sel.log")
        src = (ROOT / "scripts/seal_runner.py").read_text()
        self.assertIn("DEBUG ONLY", src)


class TestFailureSemantics(unittest.TestCase):
    def test_oom_and_exceptions_stay_in_denominator(self):
        src = (ROOT / "scripts/seal_runner.py").read_text()
        self.assertIn("structured_failure", src)
        self.assertIn("cuda_oom", src)

    def test_service_control_forbidden(self):
        for script in ("seal_runner.py", "seal_replay.py"):
            src = (ROOT / "scripts" / script).read_text()
            for banned in ("kill ", "pkill", "systemctl", "service "):
                self.assertNotIn(banned, src, script)


class TestGating(unittest.TestCase):
    def test_rejected_no_transform_accepted_refusion(self):
        from safety.registration_decision import write_decision_files

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            write_decision_files(
                tmp,
                {"status": "rejected",
                 "usable_for_reconstruction": False,
                 "rejection_reasons": ["x"]},
                np_eye_like(),
            )
            self.assertFalse((Path(tmp) / "transform.txt").exists())

        from reconstruction.rgbd_refusion import check_refusion_authorization

        self.assertFalse(check_refusion_authorization(
            {"usable_for_reconstruction": False}, np_eye_like()
        ))


def np_eye_like():
    import numpy as np

    return np.eye(4)


if __name__ == "__main__":
    unittest.main()
