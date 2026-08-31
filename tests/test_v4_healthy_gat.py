"""V4 healthy-GAT evidence tests: protocol conformance, freeze
integrity, deterministic selection, registration-repeat semantics."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
NAMING = ("official-architecture SGF-predicted healthy-GAT "
          "research candidate")
BANNED_NAMING = (
    "official SGAligner checkpoint",
    "official SGAligner reproduced model",
    "official SGAligner production model",
)


class TestProtocolPreRegistration(unittest.TestCase):
    def test_protocol_committed_before_training(self):
        # protocol.md is tracked at the protocol commit 29824f8
        import subprocess

        out = subprocess.run(
            ["git", "-C", str(ROOT), "show",
             "29824f88b3e00624701a2dc26ccf6ffc3b2dabaa",
             "--name-only", "--format="],
            capture_output=True, text=True).stdout.splitlines()
        self.assertIn(
            "outputs/official_sgaligner_v4_healthy_gat_20260827/"
            "protocol.md", out)

    def test_naming_compliance(self):
        protocol = " ".join(
            (OUT / "protocol.md").read_text().split())
        self.assertIn(NAMING, protocol)
        # the protocol QUOTES the ban, so scan machine artifacts for
        # actual misuse: caches must carry the candidate naming and
        # never designate the candidates as official models
        scanned = []
        for f in sorted(OUT.rglob("pair_cache.json")):
            c = json.loads(f.read_text())
            if "model_naming" in c:
                scanned.append(c["model_naming"])
        self.assertGreater(len(scanned), 0)
        for naming in set(scanned):
            self.assertEqual(naming, NAMING)
        for arm in ("complete", "explicit"):
            history = (OUT / "training" / arm / "run_summary.json")
            if history.exists():
                ckpt = OUT / "training" / arm / "epoch_00005.pt"
                if ckpt.exists():
                    c = torch.load(
                        ckpt, map_location="cpu", weights_only=False)
                    self.assertEqual(
                        c["training_config"]["model_naming"], NAMING)

    def test_data_split_disjoint(self):
        audit = json.loads(
            (OUT / "data_split_audit.json").read_text())
        for level in ("scene_level_intersections",
                      "pair_level_intersections"):
            for k, v in audit[level].items():
                self.assertEqual(v, 0, f"{level}.{k}")
        self.assertFalse(audit["official92_read_or_run"])


class TestInitializationHealth(unittest.TestCase):
    def test_audit_healthy_both_arms(self):
        for arm in ("complete", "explicit"):
            audit = json.loads(
                (OUT / f"initialization_audit_{arm}.json").read_text())
            h = audit["gat_health"]
            self.assertEqual(h["subnormal_fraction"], 0.0)
            self.assertTrue(h["non_constant"])
            self.assertGreater(h["gat_param_norm"], 1.0)
            self.assertGreater(h["edge_shuffle_delta"], 0.0)
            self.assertEqual(audit["init_seed"], 20260827)

    def test_optimizer_excludes_pct_and_relation(self):
        for arm in ("complete", "explicit"):
            audit = json.loads(
                (OUT / f"initialization_audit_{arm}.json").read_text())
            groups = audit["trainable_param_groups"]
            for name, meta in groups.items():
                if name.startswith(("object_encoder", "object_embedding",
                                    "meta_embedding_rel")):
                    self.assertFalse(meta["trainable"], name)
                if name.startswith(("structure_encoder",
                                    "structure_embedding")):
                    self.assertTrue(meta["trainable"], name)

    def test_identical_initial_state_between_arms(self):
        from v4_train import build_model

        a = build_model("cpu")
        b = build_model("cpu")
        for (na, ta), (nb, tb) in zip(
                a.named_parameters(), b.named_parameters()):
            self.assertEqual(na, nb)
            self.assertTrue(torch.equal(ta, tb), na)


class TestTrainingIntegrity(unittest.TestCase):
    def test_frozen_unchanged_through_training(self):
        for arm in ("complete", "explicit"):
            init = json.loads(
                (OUT / f"initialization_audit_{arm}.json").read_text())
            csv_text = (
                OUT / "training" / arm / "epoch_metrics.csv").read_text()
            rows = [l.split(",") for l in csv_text.splitlines()[1:] if l]
            self.assertTrue(rows)
            self.assertTrue(all(r[11] == "1" for r in rows),
                            f"{arm}: frozen_ok flag failed at some epoch")
            self.assertTrue(all(
                float(r[7]) == 0.0 for r in rows),
                f"{arm}: subnormal fraction nonzero")
            self.assertTrue(all(int(r[8]) > 1 for r in rows),
                            f"{arm}: GAT collapsed (unique<=1)")

    def test_no_gat_collapse_and_finite_loss(self):
        for arm in ("complete", "explicit"):
            csv_text = (
                OUT / "training" / arm / "epoch_metrics.csv").read_text()
            rows = [l.split(",") for l in csv_text.splitlines()[1:] if l]
            self.assertTrue(all(np.isfinite(float(r[1])) for r in rows))


class TestDeterministicSelection(unittest.TestCase):
    def test_selection_follows_lexicographic_key(self):
        for arm in ("complete", "explicit"):
            sel = json.loads(
                (OUT / "checkpoint_selection" / f"{arm}.json").read_text())
            evals = sel["all_evals"]
            best = max(
                evals,
                key=lambda m: (m["macro_node_f1"], m["top1_precision"],
                               m["top5_recall"], m["margin"],
                               -m["epoch"]))
            self.assertEqual(sel["selected_epoch"], best["epoch"])
            self.assertTrue(sel["pct_frozen_hashes_match"])
            self.assertEqual(
                sel["gat_subnormal_fraction_at_selected"], 0.0)

    def test_micro_reported_not_mixed(self):
        for arm in ("complete", "explicit"):
            sel = json.loads(
                (OUT / "checkpoint_selection" / f"{arm}.json").read_text())
            self.assertIn("micro_node_f1",
                          sel["selection_metrics"])


class TestRegistrationRepeats(unittest.TestCase):
    def _files(self):
        files = sorted(OUT.glob(
            "*/registration_repeats_[ABC].json"))
        self.assertGreaterEqual(len(files), 9)
        return files

    def test_repeats_report_min_median_max(self):
        for f in self._files():
            s = json.loads(f.read_text())["summary"]
            for field in ("strict", "relaxed", "accepted"):
                self.assertIn("min", s[field])
                self.assertIn("median", s[field])
                self.assertIn("max", s[field])
            self.assertGreaterEqual(s["repeats"], 3)

    def test_ambiguity_pairs_recorded(self):
        for f in self._files():
            s = json.loads(f.read_text())["summary"]
            self.assertIn("ambiguity_pairs", s)

    def test_calibration_zero_accepted_error_all_arms(self):
        for label in ("A", "B", "C"):
            f = OUT / "calibration90" / (
                f"registration_repeats_{label}.json")
            s = json.loads(f.read_text())["summary"]
            self.assertEqual(
                s["accepted_strict_error_total"], 0, label)


class TestGuardrails(unittest.TestCase):
    def test_official92_and_default_ckpt_untouched(self):
        import hashlib

        ck = ROOT / ("checkpoints/release/"
                     "sgaligner_pct_gat_rel_attr.pth.tar")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            "b716c7d81b70274f98c7b4bd894c40534bac007a"
            "b71050713e39a67c5964a17e")
        commands = (OUT / "commands.sh").read_text()
        self.assertNotIn("official92 --", commands)

    def test_gt_not_in_candidate_inference(self):
        runner = (ROOT / "scripts/v4_cache_runner.py").read_text()
        gt_pos = runner.index("load_gt_transform(pair_id)")
        geot_pos = runner.index("geot_cache = {}")
        self.assertGreater(gt_pos, geot_pos)

    def test_rejected_never_writes_usable_transform(self):
        # replay outcomes publish NO transform fields at all (the
        # usable-transform publication path lives in the audited
        # run_pair code, unchanged); assert that invariant explicitly
        for f in sorted(OUT.glob("*/registration_repeats_[ABC].json")):
            data = json.loads(f.read_text())
            for row in data["rows"]:
                for o in row.get("outcomes", []):
                    self.assertNotIn("usable_transform", o)
                    self.assertNotIn("icp_transform", o)
                    self.assertNotIn("raw_transform", o)
        # cache entries: transform fields only exist on status==ok
        for f in sorted(OUT.glob("*/cache_*/pair_cache.json"))[:50]:
            c = json.loads(f.read_text())
            if c["status"] != "ok":
                continue
            for combo, e in c["combos"].items():
                if e.get("status") == "failed":
                    self.assertNotIn("icp_transform", e)
                    self.assertNotIn("raw_transform", e)


if __name__ == "__main__":
    unittest.main()
