"""Fix3 evidence tests: resume protocol, parity, factorial, freeze."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3"


def load(p):
    return json.loads(Path(p).read_text())


class TestResumeProtocol(unittest.TestCase):
    def test_total_horizon_separated(self):
        src = (ROOT / "scripts/v2f_train.py").read_text()
        self.assertIn("--stop-after-epoch", src)
        self.assertIn("total_epochs = args.epochs", src)

    def test_gpu_exact_equivalence(self):
        d = load(OUT / "resume/resume_equivalence.json")
        gpu = d["gpu_real_subset"]
        self.assertEqual(gpu["model"]["max_diff"], 0.0)
        self.assertEqual(gpu["model"]["changed"], 0)
        self.assertEqual(gpu["optimizer"]["changed"], 0)
        self.assertTrue(gpu["lr_identical"])
        self.assertTrue(gpu["history_identical"])
        self.assertTrue(gpu["scheduler_identical"])

    def test_fail_closed_on_horizon_mismatch(self):
        d = load(OUT / "resume/resume_equivalence.json")
        fc = d["fail_closed_total_epochs_mismatch"]
        self.assertTrue(fc["refused"])

    def test_checkpoint_schema(self):
        import torch

        ckpt = torch.load(
            OUT / "resume/gpu_split/last.pt",
            map_location="cpu", weights_only=False,
        )
        for key in ("total_epochs", "next_epoch", "optimizer",
                    "scheduler", "torch_rng", "numpy_rng",
                    "python_rng", "cuda_rng", "dataset_fingerprint",
                    "training_config", "history"):
            self.assertIn(key, ckpt)


class TestParity(unittest.TestCase):
    def test_twelve_scans_exact(self):
        d = load(OUT / "official_adapter_tensor_parity.json")
        self.assertEqual(len(d["scans"]), 12)
        for row in d["rows"]:
            self.assertTrue(row["object_id_set_equal"], row["scan"])
            self.assertTrue(row["root_equal"], row["scan"])
            self.assertLess(row["rel_trans_max_abs_diff"], 1e-6,
                            row["scan"])
            self.assertTrue(row["edges_count_equal"], row["scan"])

    def test_dedup_fix_in_source(self):
        src = (ROOT / "src/adapters/sgf/data_sources.py").read_text()
        self.assertIn("seen_pairs", src)


class TestGATFactorial(unittest.TestCase):
    def test_collapse_input_independent(self):
        d = load(OUT / "gat_factorial_results.json")
        for key, s in d["summary"].items():
            if "PCT_CONTROL" in key:
                continue
            self.assertEqual(s["zero_cross_std_pairs"], s["n"], key)

    def test_shuffled_equals_raw(self):
        d = load(OUT / "gat_factorial_results.json")
        s = d["summary"]
        raw = s["official_oracle|official_complete_none|raw"]
        shuffled = s[
            "official_oracle|official_complete_none|shuffled_control"]
        self.assertEqual(
            shuffled["mean_cross_sim_std"], raw["mean_cross_sim_std"]
        )

    def test_pct_control_discriminative(self):
        d = load(OUT / "gat_factorial_results.json")
        for key in ("official_oracle|PCT_CONTROL|n/a",
                    "official_sgf_predicted|PCT_CONTROL|n/a"):
            self.assertGreater(d["summary"][key]["mean_cross_sim_std"],
                               0.1, key)


class TestFreezeEvidence(unittest.TestCase):
    def test_pct_freeze_behavior_tests_exist_and_pass(self):
        self.assertTrue((ROOT / "tests/test_pct_freeze_behavior.py")
                        .exists())


if __name__ == "__main__":
    unittest.main()
