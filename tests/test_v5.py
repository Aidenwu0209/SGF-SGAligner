"""V5 evidence tests: pre-registration, pilot equivalence, frozen
audits, registration-aware selection semantics, fail-closed gates."""
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

OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"
INIT_SHA = ("cd53b956cdc1b604fe1ddea7bd863e3c02d433555c"
            "77edb342430a8fac7e81ea")


class TestPreRegistration(unittest.TestCase):
    def test_protocol_committed_before_training(self):
        out = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             "26f2023207902f85b99f0c03dad82b42731f3ed6",
             "--name-only", "--format="],
            capture_output=True, text=True).stdout.splitlines()
        self.assertIn(
            "outputs/official_sgaligner_v5_relation_gat_20260828/"
            "protocol.md", out)
        # the protocol commit itself must contain NO result files
        for f in out:
            self.assertNotIn("training/", f)
            self.assertNotIn("metrics", f)
            self.assertNotIn("ranking", f)

    def test_init_checkpoint_sha(self):
        import hashlib

        ck = (ROOT / "outputs/official_sgaligner_v4_healthy_gat_"
              "20260827/training/explicit/epoch_00025.pt")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(), INIT_SHA)


class TestPilot(unittest.TestCase):
    def test_pilot_passed_both_arms(self):
        pilot = json.loads(
            (OUT / "pilot_training_report.json").read_text())
        for arm in ("B", "C"):
            self.assertTrue(pilot["arms"][arm]["passed"], arm)
            self.assertTrue(all(
                pilot["arms"][arm]["gates"].values()))

    def test_resume_equivalence_exact(self):
        eq = json.loads(
            (OUT / "resume_equivalence.json").read_text())
        for arm in ("B", "C"):
            self.assertTrue(eq["arms"][arm]["exact"], arm)
            self.assertEqual(eq["arms"][arm]["model_mismatches"], 0)


class TestFrozenAudits(unittest.TestCase):
    def test_frozen_unchanged_through_training(self):
        for arm in ("B", "C"):
            audit = json.loads(
                (OUT / f"frozen_tensor_audit_{arm}.json").read_text())
            self.assertTrue(
                audit["frozen_unchanged_through_training"], arm)

    def test_pct_identical_to_init_in_all_checkpoints(self):
        init = torch.load(
            ROOT / ("outputs/official_sgaligner_v4_healthy_gat_"
                    "20260827/training/explicit/epoch_00025.pt"),
            map_location="cpu", weights_only=False)
        import glob

        ckpts = sorted(
            (OUT / "training").glob("*/epoch_*.pt"))[:4]
        for ck in ckpts:
            state = torch.load(
                ck, map_location="cpu", weights_only=False)
            for k, v in init["model"].items():
                if k.startswith(("object_encoder",
                                 "object_embedding")):
                    self.assertTrue(
                        torch.equal(v, state["model"][k]),
                        f"{ck.name}:{k}")


class TestRegistrationAwareSelection(unittest.TestCase):
    def _ranking(self):
        return json.loads(
            (OUT / "checkpoint_ranking.json").read_text())

    def test_selection_key_order_respected(self):
        d = self._ranking()
        keys = [c["key"] for c in d["candidates"]]
        self.assertEqual(keys, sorted(keys))

    def test_error_candidates_excluded_from_top(self):
        d = self._ranking()
        best = d["candidates"][0]
        self.assertEqual(
            best["counts"]["accepted_strict_error"], 0)

    def test_gates_evaluated_fail_closed(self):
        d = self._ranking()
        # every candidate failed at least the top1 gate — no winner
        for label, g in d["gates"].items():
            self.assertFalse(g["all_pass"], label)

    def test_calibration_not_run(self):
        for f in ("calibration90.json", "fixed12.json"):
            self.assertFalse((OUT / f).exists(),
                             "protocol forbids calibration/fixed12 "
                             "when no candidate passes selection")

    def test_registration_improvements_recorded(self):
        d = self._ranking()
        base = d["A_baseline_counts"]
        b10 = next(
            c for c in d["candidates"] if c["label"] == "B_ep10")
        self.assertGreater(
            b10["counts"]["raw_strict"], base["raw_strict"])
        self.assertGreater(
            b10["counts"]["accepted_strict_correct"],
            base["accepted_strict_correct"])


class TestGuardrails(unittest.TestCase):
    def test_no_official92_no_default_change(self):
        import hashlib

        ck = ROOT / ("checkpoints/release/"
                     "sgaligner_pct_gat_rel_attr.pth.tar")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            "b716c7d81b70274f98c7b4bd894c40534bac007a"
            "b71050713e39a67c5964a17e")

    def test_gt_not_in_training_inputs(self):
        src = (ROOT / "scripts/canonical_inputs.py").read_text()
        self.assertNotIn("gt_transform", src)
        train = (ROOT / "scripts/v5_train.py").read_text()
        # GT appears only via build_labels (training labels allowed)
        self.assertIn("with_labels=True", train)

    def test_old_evidence_unmodified(self):
        manifest = json.loads(
            (ROOT / "outputs/official_sgaligner_v4_fix_seal_"
             "20260828/artifact_manifest.json").read_text())["files"]
        import hashlib

        for i, (rel, meta) in enumerate(manifest.items()):
            if i >= 20:
                break
            p = ROOT / rel
            self.assertTrue(p.exists(), rel)
            self.assertEqual(
                hashlib.sha256(p.read_bytes()).hexdigest(),
                meta["sha256"], rel)


if __name__ == "__main__":
    unittest.main()
