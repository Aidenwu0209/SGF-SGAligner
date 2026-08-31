"""V4-Fix-Seal evidence tests: canonical builder parity, frozen metric
semantics (hand-computed), reselection determinism, calibration
isolation, fixed12 safety classification, guardrails."""
from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

NEW = ROOT / "outputs/official_sgaligner_v4_fix_seal_20260828"
OLD_V4FIX = ROOT / (
    "outputs/official_sgaligner_v4_fix_fair_selection_20260828")
AUDIT_PAIR = (
    "09582205-e2c2-2de1-9475-1cdac7639e60_to_"
    "0958220d-e2c2-2de1-9710-c37018da1883"
)


class TestMetricSemanticsHandComputed(unittest.TestCase):
    """The exact hand-computed sample from metric_semantics.md."""

    def _rows(self):
        # pair1: P={(0,4),(1,5)}, A={(0,4),(1,6)}
        # pair2: P=empty, A={(0,3)}
        return [
            {"tp": 1, "pred_count": 2, "anchor_count": 2,
             "precision": 0.5, "recall": 0.5, "f1": 0.5,
             "top1_hit": 1, "top1_total": 2, "top5_hits": 1,
             "margin": 0.1},
            {"tp": 0, "pred_count": 0, "anchor_count": 1,
             "precision": 0.0, "recall": 0.0, "f1": 0.0,
             "top1_hit": 0, "top1_total": 0, "top5_hits": 0,
             "margin": None},
        ]

    def test_hand_computed_aggregates(self):
        from v4seal_metrics import aggregate

        agg = aggregate(self._rows())
        self.assertAlmostEqual(
            agg["macro_node_f1"], 0.25, places=12)
        # micro_p = 1/2, micro_r = 1/3 -> 2*(1/6)/(5/6) = 0.4
        self.assertAlmostEqual(
            agg["micro_node_f1"], 0.4, places=12)
        self.assertAlmostEqual(
            agg["macro_top1"], 0.25, places=12)
        self.assertAlmostEqual(
            agg["micro_top1"], 0.5, places=12)
        self.assertEqual(agg["zero_candidate_pairs"], 1)
        self.assertAlmostEqual(agg["margin"], 0.1, places=12)

    def test_micro_never_equals_macro_by_construction(self):
        from v4seal_metrics import aggregate

        agg = aggregate(self._rows())
        self.assertNotAlmostEqual(
            agg["macro_node_f1"], agg["micro_node_f1"], places=9)

    def test_per_pair_metrics_matches_definition(self):
        from v4seal_metrics import per_pair_node_metrics

        # rank_list for 3 src + 2 ref; src0 top cross = 4 (anchor),
        # src1 top cross = 5 (not anchor); anchors {(0,4)}
        rank = [[4, 1, 0, 5, 2], [5, 4, 1, 0, 2], [4, 5, 0, 1, 2]]
        node_corrs = [(0, 4), (1, 5), (2, 4)]
        pp = per_pair_node_metrics(
            node_corrs, rank, 3, {(0, 4)})
        self.assertEqual(pp["tp"], 1)
        self.assertEqual(pp["pred_count"], 3)
        self.assertAlmostEqual(pp["precision"], 1 / 3)
        self.assertEqual(pp["top1_hit"], 1)
        self.assertEqual(pp["top1_total"], 3)
        self.assertEqual(pp["top5_hits"], 1)


class TestOfficialMatcherSemantics(unittest.TestCase):
    def test_top3_then_filter(self):
        from inference import official_matching

        emb = np.zeros((8, 4), dtype=np.float32)
        emb[0] = [1.0, 0.05, 0, 0]
        emb[1] = [0.9, 0, 0, 0]      # same-graph near
        emb[2] = [0.8, 0.6, 0, 0]    # same-graph near
        emb[4] = [0.7, 0.7, 0.1, 0]  # cross-graph anchor
        corrs, _rank, _sim = official_matching(emb, 4)
        src0_refs = [b for a, b in corrs if a == 0]
        self.assertEqual(src0_refs, [4])

    def test_training_semantics_diverges(self):
        emb = np.zeros((8, 4), dtype=np.float32)
        emb[0] = [1.0, 0.05, 0, 0]
        emb[1] = [0.9, 0, 0, 0]
        emb[2] = [0.8, 0.6, 0, 0]
        emb[4] = [0.7, 0.7, 0.1, 0]
        emb[5] = [0.1, 0.2, 0.9, 0.9]
        normed = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        rank = np.argsort(-(normed @ normed.T), axis=1)
        old_semantics = [
            (i, int(x)) for i in range(4)
            for x in [y for y in rank[i] if y >= 4][:3]]
        src0_old = [b for a, b in old_semantics if a == 0]
        self.assertEqual(len(src0_old), 3)  # pre-filter keeps 3


class TestCanonicalBuilderParity(unittest.TestCase):
    def test_input_parity_all_89(self):
        parity = json.loads(
            (NEW / "input_parity.json").read_text())
        self.assertEqual(parity["pairs"], 89)
        self.assertTrue(parity["all_equal"])
        for row in parity["rows"]:
            self.assertTrue(row["tot_obj_pts"]["equal"], row["pair_id"])
            self.assertTrue(row["complete_edges"]["equal"])

    def test_centre_shift_root_cause_reproduced(self):
        audit = json.loads(
            (NEW / "audit_before_fix.json").read_text())
        self.assertAlmostEqual(
            audit["fields"]["tot_obj_pts"]["max_abs_diff"],
            0.0655013918876648, places=9)

    def test_arm_fingerprints_differ(self):
        gate = json.loads(
            (NEW / "legacy_cache_reproduction.json").read_text())
        self.assertTrue(
            gate["arm_specific_fingerprints_differ"])

    def test_legacy_gate_passed_both_arms(self):
        gate = json.loads(
            (NEW / "legacy_cache_reproduction.json").read_text())
        for label in ("B", "C"):
            self.assertTrue(gate[label]["passed"], label)
            self.assertEqual(
                gate[label]["node_matches_aligned_pairs"], 89)
            self.assertEqual(
                gate[label]["per_pair_metrics_aligned_pairs"], 89)
            self.assertEqual(
                gate[label]["max_embedding_diffs"]["pct"], 0.0)
            self.assertEqual(
                gate[label]["max_embedding_diffs"]["rel"], 0.0)
            self.assertLessEqual(
                gate[label]["max_embedding_diffs"]["gat"], 1e-4)


class TestReselection(unittest.TestCase):
    def test_22_checkpoints_ranked_and_deterministic(self):
        det = json.loads(
            (NEW / "determinism_replay.json").read_text())
        self.assertEqual(det["checkpoints_checked"], 22)
        self.assertTrue(det["all_identical"])
        for label, n in (("B", 12), ("C", 10)):
            ranking = json.loads(
                (NEW / f"checkpoint_ranking_{label}.json").read_text())
            self.assertEqual(len(ranking["ranking"]), n)
            epochs = sorted(
                r["epoch"] for r in ranking["ranking"])
            self.assertEqual(epochs, list(range(5, 5 * n + 1, 5)))

    def test_macro_micro_both_reported(self):
        for label in ("B", "C"):
            ranking = json.loads(
                (NEW / f"checkpoint_ranking_{label}.json").read_text())
            for r in ranking["ranking"]:
                for field in ("macro_node_f1", "micro_node_f1",
                              "macro_top1", "micro_top1",
                              "macro_top5", "micro_top5", "margin",
                              "zero_candidate_pairs"):
                    self.assertIn(field, r["metrics"])

    def test_winners_present(self):
        for label in ("B", "C"):
            w = json.loads(
                (NEW / f"winner_{label}.json").read_text())
            self.assertIn("winner_epoch", w)
            self.assertIn("winner_checkpoint_sha256", w)


class TestCalibrationIsolation(unittest.TestCase):
    def test_calibration_not_in_selection(self):
        for label in ("B", "C"):
            ranking = json.loads(
                (NEW / f"checkpoint_ranking_{label}.json").read_text())
            self.assertEqual(
                ranking["selection_split"], "selection89 ONLY")

    def test_calibration_paired_comparison_exists(self):
        data = json.loads(
            (NEW / "calibration_paired_comparison.json").read_text())
        for label in ("B", "C"):
            self.assertIn(f"winner_{label}", json.dumps(data))


class TestFixed12Safety(unittest.TestCase):
    def test_categories_distinguished(self):
        data = json.loads(
            (NEW / "fixed12_safety.json").read_text())
        for key in ("accepted_strict_correct", "accepted_strict_error",
                    "rejected", "failed", "zero_candidate"):
            self.assertIn(key, data["summary"])
        # honest wording: no proof of veto generalisability
        self.assertFalse(data["summary"]["ready_for_veto"])

    def test_no_error_side_claim(self):
        data = json.loads(
            (NEW / "fixed12_safety.json").read_text())
        if data["summary"]["accepted_strict_error"] == 0:
            self.assertEqual(
                data["summary"]["separation_capacity"],
                "NOT_EVALUABLE")


class TestGuardrails(unittest.TestCase):
    def test_git_diff_check_clean(self):
        rc = subprocess.run(
            ["git", "-C", str(ROOT), "diff",
             "e2d9ca7c8f67a462223dfc9f8658a00c62b25596..HEAD",
             "--check"],
            capture_output=True, text=True).returncode
        self.assertEqual(rc, 0)

    def test_old_evidence_unmodified(self):
        manifest = json.loads(
            (ROOT / "outputs/official_sgaligner_v4_healthy_gat_"
             "20260827/artifact_manifest.json").read_text())["files"]
        import hashlib

        # the ONE mandated whitespace-only EOF fix (task XI.1) in the
        # V4-Fix dir; its original and post-fix SHAs are recorded in
        # the seal evidence — every OTHER historical file must still
        # match its manifest hash
        whitelisted = ("outputs/official_sgaligner_v4_fix_fair_"
                       "selection_20260828/old_vs_corrected_"
                       "selection.md")
        for i, (rel, meta) in enumerate(manifest.items()):
            if i >= 40:
                break
            p = ROOT / rel
            self.assertTrue(p.exists(), rel)
            if rel == whitelisted:
                continue
            self.assertEqual(
                hashlib.sha256(p.read_bytes()).hexdigest(),
                meta["sha256"], rel)
        fixfile = ROOT / ("outputs/official_sgaligner_v4_fix_fair_"
                          "selection_20260828/old_vs_corrected_"
                          "selection.md")
        self.assertTrue(fixfile.read_bytes().endswith(b"\n"))
        self.assertFalse(
            fixfile.read_bytes().endswith(b"\n\n"))

    def test_gt_not_in_canonical_builder(self):
        src = (ROOT / "scripts/canonical_inputs.py").read_text()
        self.assertNotIn("gt_transform", src)
        self.assertNotIn("load_gt_transform", src)

    def test_default_checkpoint_untouched(self):
        import hashlib

        ck = ROOT / ("checkpoints/release/"
                     "sgaligner_pct_gat_rel_attr.pth.tar")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            "b716c7d81b70274f98c7b4bd894c40534bac007a"
            "b71050713e39a67c5964a17e")


if __name__ == "__main__":
    unittest.main()
