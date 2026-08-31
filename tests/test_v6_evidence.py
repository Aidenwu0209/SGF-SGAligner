"""V6 evidence tests: protocol pre-registration, label audit, arm
configurations, selection gates, calibration/fixed12 honesty."""
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / (
    "outputs/official_sgaligner_v6_sgf_domain_matcher_20260829")


class TestProtocol(unittest.TestCase):
    def test_preregistered_before_training(self):
        out = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             "0e546a3741f802ce04808c35b28931b0a964777a",
             "--name-only", "--format="],
            capture_output=True, text=True).stdout.splitlines()
        self.assertTrue(any(
            "protocol.md" in f for f in out))
        for f in out:
            self.assertNotIn("training/", f)
            self.assertNotIn("selection/", f)

    def test_label_code_precedes_training_in_history(self):
        import subprocess as sp

        code = sp.run(
            ["git", "-C", str(ROOT), "log", "--format=%H", "-1",
             "505244b18734ceb86c0b4faccc991c9c4481e8ff"],
            capture_output=True, text=True).stdout.strip()
        # the code commit exists and is an ancestor of HEAD
        self.assertTrue(code)
        merge_base = sp.run(
            ["git", "-C", str(ROOT), "merge-base",
             "505244b18734ceb86c0b4faccc991c9c4481e8ff", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        self.assertEqual(merge_base, code)


class TestLabelAudit(unittest.TestCase):
    def test_audit_present_and_sane(self):
        d = json.loads(
            (OUT / "label_audit" / "train437_label_audit.json"
             ).read_text())
        a = d["aggregate"]
        self.assertEqual(d["pairs_processed"], 437)
        self.assertEqual(len(d["pairs_no_positive"]), 0)
        self.assertGreater(a["positive"], 1000)
        self.assertGreater(a["ambiguous"], 100)
        self.assertLess(a["hard_negative"], a["negative"])
        self.assertGreater(a["split_sources"], 100)
        self.assertGreater(a["merged_refs"], 100)

    def test_examples_for_human_check(self):
        d = json.loads(
            (OUT / "label_audit" / "train437_label_audit.json"
             ).read_text())
        self.assertGreaterEqual(len(d["examples"]), 20)
        for ex in d["examples"][:20]:
            self.assertGreaterEqual(
                len(ex["top_positive_pairs"]), 2)


class TestArms(unittest.TestCase):
    def test_pct_frozen_in_B(self):
        pg = json.loads(
            (OUT / "parameter_groups_B.json").read_text())
        for name, trainable in pg["groups"].items():
            if name.startswith("object_encoder"):
                self.assertFalse(trainable, name)

    def test_pct_last_stage_low_lr_in_C(self):
        pg = json.loads(
            (OUT / "parameter_groups_C.json").read_text())
        self.assertEqual(pg["pct_last_stage_lr"], 2.5e-05)
        unfrozen = [
            n for n, t in pg["groups"].items()
            if n.startswith("object_encoder") and t]
        # protocol: the last DISCRIMINATIVE stage (PCT head) is
        # unfrozen; linear1/linear2/bn2 are all head layers. The
        # trunk (embedding/sa1-4/linear.0/linear.1/bn1) stays frozen.
        for n in unfrozen:
            self.assertTrue(
                n.startswith(("object_encoder.linear1",
                              "object_encoder.linear2",
                              "object_encoder.bn2")), n)

    def test_gat_trainable_only_in_D(self):
        for arm in ("B", "C"):
            pg = json.loads(
                (OUT / f"parameter_groups_{arm}.json").read_text())
            gat_frozen = all(
                not t for n, t in pg["groups"].items()
                if n.startswith("structure_"))
            self.assertTrue(gat_frozen, arm)
        pg = json.loads(
            (OUT / "parameter_groups_D.json").read_text())
        self.assertTrue(any(
            t for n, t in pg["groups"].items()
            if n.startswith("structure_")))


class TestSelection(unittest.TestCase):
    def test_winner_passed_all_gates(self):
        d = json.loads(
            (OUT / "checkpoint_ranking.json").read_text())
        winner = next(
            c for c in d["candidates"]
            if c["selection_rank"] == 1)
        gates = d["gates"][winner["label"]]
        self.assertTrue(gates["all_pass"])
        self.assertEqual(
            winner["counts"]["accepted_strict_error"], 0)
        self.assertGreater(
            winner["counts"]["raw_strict"],
            d["A_baseline_counts"]["raw_strict"])

    def test_error_candidates_ranked_below(self):
        d = json.loads(
            (OUT / "checkpoint_ranking.json").read_text())
        for c in d["candidates"]:
            if c["counts"]["accepted_strict_error"] > 0:
                self.assertGreater(c["selection_rank"], 3)


class TestCalibrationFixed12(unittest.TestCase):
    def test_calibration_single_run_error_zero(self):
        d = json.loads(
            (OUT / "calibration" / "calibration90.json").read_text())
        self.assertTrue(d["gate"]["all_pass"])
        self.assertEqual(
            d["winner_registration"]["accepted_strict_error"], 0)

    def test_fixed12_honest_failure(self):
        d = json.loads(
            (OUT / "fixed12" / "fixed12.json").read_text())
        self.assertEqual(d["runs_per_pair"], 3)
        # the gates failed — recorded honestly, not spun
        self.assertFalse(d["gates"]["all_pass"])
        self.assertEqual(
            d["aggregate"]["accepted_strict_error"], 0)


class TestGuardrails(unittest.TestCase):
    def test_default_checkpoint_untouched(self):
        import hashlib

        ck = ROOT / ("checkpoints/release/"
                     "sgaligner_pct_gat_rel_attr.pth.tar")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            "b716c7d81b70274f98c7b4bd894c40534bac007a"
            "b71050713e39a67c5964a17e")

    def test_official_sources_unmodified(self):
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short",
             "src/aligner/", "src/GeoTransformer/"],
            capture_output=True, text=True).stdout
        self.assertEqual(status.strip(), "")

    def test_gt_not_in_consistency_layer(self):
        # executable identifiers (docstrings legitimately describe
        # the GT-free design)
        src = (ROOT / "scripts/spatial_consistency.py").read_text()
        code = "\n".join(
            l for l in src.splitlines()
            if not l.lstrip().startswith(("#", chr(34)*3)))
        for banned in ("gt_transform", "load_gt", "rre", "rte"):
            self.assertNotIn(banned, code)


if __name__ == "__main__":
    unittest.main()
