"""V4-Fix evidence tests: official matcher semantics, fair selection,
errata integrity, safety diagnosis, old-evidence immutability."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

NEW = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"
OLD = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
ERROR_PAIR = (
    "0ad2d384-79e2-2212-9b18-72b44eb5463f_to_"
    "0ad2d399-79e2-2212-99cf-7a3512734bd7"
)


def _embedding_fixture():
    """4 src + 4 ref nodes.  src0's globally-nearest neighbours (excl.
    self) are TWO same-graph nodes, then one ref anchor.  Old training
    semantics (pre-filter cross-graph, take 3) and the official
    matcher (top-3 overall, then drop same-graph) then DISAGREE on
    the candidate set of src0.
    """
    emb = np.zeros((8, 4), dtype=np.float32)
    # same-graph neighbours of src0 at cos-sim 0.9 / 0.8
    emb[1] = [0.9, 0, 0, 0]
    emb[2] = [0.8, 0.6, 0, 0]
    # cross-graph anchor at 0.7, distractor at 0.1
    emb[4] = [0.7, 0.7, 0.1, 0]
    emb[5] = [0.1, 0.2, 0.9, 0.9]
    # src0 dominates dim0
    emb[0] = [1.0, 0.05, 0, 0]
    # remaining nodes far away / orthogonal
    emb[3] = [0, 0, 1.0, 0]
    emb[6] = [0, 0, 0, 1.0]
    emb[7] = [0, 1.0, 0, 0]
    return emb


class TestOfficialMatcherSemantics(unittest.TestCase):
    def test_official_top3_then_filter(self):
        from inference import official_matching

        emb = _embedding_fixture()
        corrs, _rank, _sim = official_matching(emb, 4)
        src0_refs = [b for a, b in corrs if a == 0]
        # official: src0's top-3 overall (excl. self) = nodes 1, 2, 4
        # -> same-graph 1,2 dropped, ONLY ref 4 kept
        self.assertEqual(src0_refs, [4])

    def test_old_training_semantics_differs(self):
        """The v4_train evaluate() pre-filtered cross-graph THEN took
        3 — on this fixture it must produce MORE candidates for src0
        than the official matcher (constructive divergence proof)."""
        from inference import official_matching

        emb = _embedding_fixture()
        normed = emb / np.linalg.norm(
            emb, axis=1, keepdims=True)
        sim = normed @ normed.T
        rank = np.argsort(-sim, axis=1)
        old_semantics = []
        for i in range(4):
            refs = [x for x in rank[i] if x >= 4][:3]
            for r in refs:
                old_semantics.append((i, int(r)))
        official, _r, _s = official_matching(emb, 4)
        src0_old = [b for a, b in old_semantics if a == 0]
        src0_official = [b for a, b in official if a == 0]
        self.assertEqual(len(src0_old), 3)
        self.assertEqual(src0_official, [4])
        self.assertNotEqual(src0_old, src0_official)

    def test_matcher_source_is_the_real_one(self):
        # fair selection must import and call inference.official_matching
        sel = (ROOT / "scripts/v4fix_fair_selection.py").read_text()
        self.assertIn("from inference import official_matching", sel)
        self.assertIn("official_matching(", sel)
        ranking = json.loads(
            (NEW / "official_semantic_checkpoint_ranking_B.json"
             ).read_text())
        self.assertEqual(
            ranking["matcher"]["implementation"],
            "inference.official_matching "
            "(imported and called verbatim)")


class TestFairSelection(unittest.TestCase):
    def test_all_checkpoints_ranked(self):
        for label, n in (("B", 12), ("C", 10)):
            ranking = json.loads(
                (NEW / f"official_semantic_checkpoint_ranking_"
                       f"{label}.json").read_text())
            self.assertEqual(len(ranking["ranking"]), n)
            epochs = [r["epoch"] for r in ranking["ranking"]]
            self.assertEqual(sorted(epochs), list(range(5, 5 * n + 1, 5)))

    def test_lexicographic_order_holds(self):
        for label in ("B", "C"):
            ranking = json.loads(
                (NEW / f"official_semantic_checkpoint_ranking_"
                       f"{label}.json").read_text())["ranking"]
            for a, b in zip(ranking, ranking[1:]):
                ka = (a["metrics"]["macro_node_f1"],
                      a["metrics"]["top1_precision"],
                      a["metrics"]["top5_recall"],
                      a["metrics"]["margin"], -a["epoch"])
                kb = (b["metrics"]["macro_node_f1"],
                      b["metrics"]["top1_precision"],
                      b["metrics"]["top5_recall"],
                      b["metrics"]["margin"], -b["epoch"])
                self.assertGreaterEqual(ka, kb)

    def test_determinism_replay_all_identical(self):
        det = json.loads(
            (NEW / "determinism_replay.json").read_text())
        self.assertTrue(det["all_identical"])
        self.assertEqual(det["checkpoints_checked"], 22)

    def test_calibration_not_used_for_selection(self):
        for label in ("B", "C"):
            sel_text = (
                NEW / f"checkpoint_selection_corrected_{label}.json"
            ).read_text()
            self.assertNotIn("calibration", sel_text)
            ranking = json.loads(
                (NEW / f"official_semantic_checkpoint_ranking_"
                       f"{label}.json").read_text())
            self.assertEqual(
                ranking["selection_split"], "selection89 ONLY")

    def test_winners(self):
        b = json.loads(
            (NEW / "checkpoint_selection_corrected_B.json").read_text())
        c = json.loads(
            (NEW / "checkpoint_selection_corrected_C.json").read_text())
        self.assertEqual(b["winner_epoch"], 15)
        self.assertTrue(b["changed"])
        self.assertEqual(c["winner_epoch"], 20)
        self.assertFalse(c["changed"])


class TestErrata(unittest.TestCase):
    def test_error_pair_from_raw_json(self):
        from v4fix_errata import locate_error_accepts

        found = locate_error_accepts(OLD)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["pair_id"], ERROR_PAIR)
        self.assertAlmostEqual(found[0]["rre"], 2.966, places=2)
        self.assertAlmostEqual(found[0]["rte"], 0.2172, places=3)

    def test_errata_files_consistent_with_raw(self):
        errata = json.loads(
            (NEW / "evidence_errata.json").read_text())
        self.assertEqual(
            errata["e1_error_accept"]["extracted_from_raw"][
                "pair_id"], ERROR_PAIR)
        self.assertIn("10b1792c", json.dumps(errata))

    def test_top5_provenance_typo(self):
        errata = json.loads(
            (NEW / "evidence_errata.json").read_text())
        e2 = errata["e2_top5_provenance_typo"]
        self.assertEqual(e2["typo_value"], 0.4353)
        self.assertEqual(e2["correct_value"], 0.3498433974919869)
        dm = json.loads((OLD / "deterministic_metrics.json").read_text())
        self.assertEqual(
            dm["arms"]["A_incumbent"]["calibration90"]["top5_recall"],
            0.3498433974919869)

    def test_source_evidence_hashes_present(self):
        src = json.loads(
            (NEW / "source_evidence_sha256.json").read_text())
        self.assertGreaterEqual(len(src["files"]), 11)


class TestSafetyDiagnosis(unittest.TestCase):
    def test_three_splits_three_repeats_full_fields(self):
        for split in ("selection89", "calibration90", "fixed12"):
            rows = json.loads(
                (NEW / "safety_diagnosis" /
                 f"{split}_repeats.json").read_text())
            ok_rows = [r for r in rows if r.get("cache_status") == "ok"]
            self.assertGreater(len(ok_rows), 0)
            for row in ok_rows:
                self.assertGreaterEqual(len(row["outcomes"]), 3)
                for o in row["outcomes"]:
                    if o.get("status") != "ok":
                        continue
                    for field in (
                            "raw_transform", "icp_transform",
                            "ransac_matches", "ransac_inliers",
                            "ransac_inlier_ratio", "icp_converged",
                            "icp_fitness", "icp_update_translation_m",
                            "surface_overlap_10cm",
                            "bidirectional_rotation_deg",
                            "spatial_extent_m", "decision_status",
                            "rre_post_hoc", "rte_post_hoc"):
                        self.assertIn(field, o)

    def test_rejected_never_promotes_transform(self):
        for split in ("selection89", "calibration90", "fixed12"):
            rows = json.loads(
                (NEW / "safety_diagnosis" /
                 f"{split}_repeats.json").read_text())
            for row in rows:
                for o in row.get("outcomes", []):
                    if o.get("status") != "ok":
                        continue
                    if not o.get("accepted"):
                        self.assertNotEqual(o["decision_status"],
                                            "accepted")
                        self.assertTrue(o["rejection_reasons"])

    def test_error_pair_analyzed(self):
        deep = json.loads(
            (NEW / "safety_diagnosis/error_pair_analysis.json"
            ).read_text())
        self.assertEqual(deep["pair_id"], ERROR_PAIR)
        self.assertIn("NEAR-MISS", deep["recharacterisation"])
        self.assertNotIn("180", deep["recharacterisation"].split(
            "NOT a")[0])  # the flip claim must appear only negated

    def test_gt_free_separation_reported(self):
        sep = json.loads(
            (NEW / "safety_diagnosis/gt_free_separation.json"
            ).read_text())
        self.assertGreaterEqual(sep["accepted_total"], 1)
        for field, meta in sep["fields"].items():
            self.assertIn("separable_by_interval", meta)


class TestGuardrails(unittest.TestCase):
    def test_gt_not_in_inference_path(self):
        for script in ("v4fix_fair_selection.py",
                       "v4fix_calibration.py"):
            src = (ROOT / "scripts" / script).read_text()
            self.assertNotIn("load_gt_transform", src)
        diag = (ROOT / "scripts/v4fix_safety_diagnosis.py").read_text()
        # GT loads AFTER the matcher ran and is never an argument of
        # the registration/decision calls (exact signatures checked)
        gt_pos = diag.index("load_gt_transform(pair_id)")
        match_pos = diag.index("official_matching(")
        self.assertGreater(gt_pos, match_pos)
        self.assertIn(
            "combo_registration(\n                    geot, node_corrs)",
            diag)
        self.assertIn(
            "combo_decision(\n                data_dict, "
            "registration, pair_id)",
            diag)

    def test_old_evidence_unmodified(self):
        # the git-tracked V4 manifest must still verify against the
        # current tree (proves the old dirs were never touched)
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            capture_output=True, text=True).stdout
        for line in status.splitlines():
            self.assertFalse(
                line.split()[-1].startswith(
                    "outputs/official_sgaligner_v4_healthy_gat_"
                    "20260827")
                or line.split()[-1].startswith(
                    "outputs/official_sgaligner_v3_pct_parity_"
                    "baseline_20260827"),
                f"old evidence modified: {line}")
        manifest = json.loads(
            (OLD / "artifact_manifest.json").read_text())["files"]
        import hashlib

        checked = 0
        for rel, meta in list(manifest.items()):
            p = ROOT / rel
            if not p.exists():
                self.fail(f"old evidence file removed: {rel}")
            digest = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(
                        lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            self.assertEqual(
                digest.hexdigest(), meta["sha256"], rel)
            checked += 1
            if checked >= 60:
                break

    def test_official92_and_default_ckpt_untouched(self):
        import hashlib

        ck = ROOT / ("checkpoints/release/"
                     "sgaligner_pct_gat_rel_attr.pth.tar")
        self.assertEqual(
            hashlib.sha256(ck.read_bytes()).hexdigest(),
            "b716c7d81b70274f98c7b4bd894c40534bac007a"
            "b71050713e39a67c5964a17e")
        commands = (NEW / "commands.sh").read_text()
        self.assertNotIn("official92 --", commands)


if __name__ == "__main__":
    unittest.main()
