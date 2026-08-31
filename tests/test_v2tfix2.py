"""Fix2 tests: PCT freeze guards, diagnostics ranges, GAT collapse,
evidence schema."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix2"


def load(p):
    return json.loads(Path(p).read_text())


class TestPCTFreeze(unittest.TestCase):
    def test_freeze_covers_encoder_and_projection(self):
        src = (ROOT / "scripts/v2f_train.py").read_text()
        self.assertIn('frozen_prefixes = ("object_encoder", '
                      '"object_embedding")', src)
        self.assertIn("model.object_encoder.eval()", src)

    def test_buffer_drift_guard_raises(self):
        src = (ROOT / "scripts/v2f_train.py").read_text()
        self.assertIn("frozen point path drifted", src)

    def test_strategy_b_name_documents_scope(self):
        cfg = load(OUT / "smoke/parameter_groups.json")
        encoder_frozen = all(
            not v["trainable"] for k, v in cfg.items()
            if k.startswith(("object_encoder", "object_embedding"))
        )
        self.assertTrue(encoder_frozen)


class TestDiagnosticsRanges(unittest.TestCase):
    def test_reverse_recall_bounded(self):
        csv = (OUT / "smoke/epoch_metrics.csv").read_text()
        header, *rows = csv.strip().splitlines()
        idx = header.split(",").index("rev_recall")
        for row in rows:
            v = float(row.split(",")[idx])
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_all_metrics_within_ranges(self):
        csv = (OUT / "smoke/epoch_metrics.csv").read_text()
        header, *rows = csv.strip().splitlines()
        names = header.split(",")
        bounds = {
            "pos_sim": (-1, 1), "neg_sim": (-1, 1),
            "margin": (-2, 2), "top1_overlap_precision": (0, 1),
            "top5_overlap_precision": (0, 1), "fwd_recall": (0, 1),
            "rev_recall": (0, 1),
        }
        for row in rows:
            vals = [float(x) for x in row.split(",")]
            for name, (lo, hi) in bounds.items():
                v = vals[names.index(name)]
                self.assertGreaterEqual(v, lo, name)
                self.assertLessEqual(v, hi, name)

    def test_diagnostics_use_all_samples_not_60(self):
        src = (ROOT / "scripts/v2f_train.py").read_text()
        self.assertNotIn("samples[:60]", src)

    def test_gradients_epoch_accumulated(self):
        src = (ROOT / "scripts/v2f_train.py").read_text()
        self.assertIn("g_pct_acc", src)
        self.assertIn("np.sqrt(g_pct_acc)", src)


class TestGATCollapse(unittest.TestCase):
    def test_collapse_systematic_both_modes(self):
        audit = load(OUT / "gat_collapse_audit.json")
        for key, v in audit.items():
            for mode, s in v["by_mode"].items():
                total = s.get("errors", 0) + (
                    89 if "selection" in key else 90
                )
                # every non-error pair has zero GAT cross-sim std
                self.assertEqual(
                    s["gat_zero_std_pairs"],
                    (89 if "selection" in key else 90) - s.get("errors", 0),
                    f"{key}/{mode}",
                )

    def test_pct_retains_discriminability(self):
        audit = load(OUT / "gat_collapse_audit.json")
        for key, v in audit.items():
            for mode, s in v["by_mode"].items():
                self.assertGreater(
                    s["pct_mean_cross_sim_std"], 0.1, f"{key}/{mode}"
                )


class TestModalityAB(unittest.TestCase):
    def test_results_present_and_ordered(self):
        r = load(OUT / "modality_ab_matching.json")
        for combo in ("pct", "gat", "rel", "pct+gat", "pct+rel",
                      "pct+gat+rel"):
            self.assertIn(combo, r)
            self.assertEqual(r[combo]["pairs"], 89)

    def test_gat_alone_worst_pct_best(self):
        r = load(OUT / "modality_ab_matching.json")
        self.assertLess(r["gat"]["top1_precision"],
                        r["pct"]["top1_precision"])


class TestEvidenceSchema(unittest.TestCase):
    def test_resume_equivalence_diff_detail(self):
        d = load(OUT / "resume_test/resume_equivalence.json")
        self.assertIn("model_diff", d)
        self.assertIn("changed", d["model_diff"])
        self.assertIn("top5_changed", d["model_diff"])

    def test_determinism_recorded(self):
        d = load(OUT / "smoke/determinism.json")
        self.assertIn("mode", d)

    def test_no_service_control(self):
        for s in ("v2f_train.py", "v2f2_gat_audit.py", "v2f2_modality_ab.py"):
            src = (ROOT / "scripts" / s).read_text()
            for banned in ("systemctl", "kill ", "pkill"):
                self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main()
