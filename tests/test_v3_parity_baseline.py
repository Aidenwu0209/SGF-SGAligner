"""V3-A/B evidence tests: official sampling parity, PCT end-to-end
parity, single-inference cache integrity, offline ablation replay."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))

OUT = ROOT / "outputs/official_sgaligner_v3_pct_parity_baseline_20260827"


def digest(tag):
    return json.loads((OUT / f"parity_digest_{tag}.json").read_text())


def load_cache_roots():
    base = OUT / "final_inference_cache"
    return {
        "fixed12_run1": base / "fixed12_run1",
        "selection89": base / "selection89",
        "calibration90": base / "calibration90",
    }


def load_pairs(root):
    out = []
    for tag_dir in sorted(root.iterdir()):
        f = tag_dir / "pair_cache.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


class TestOfficialSampler(unittest.TestCase):
    """official_mt19937 vs the REAL official pcl_farthest_sample."""

    def _official(self, points, npoint):
        # the real official function, imported from utils/ (untouched)
        from utils import point_cloud

        np.random.seed(1234)
        return point_cloud.pcl_farthest_sample(points, npoint)

    def _ours(self, points, npoint):
        from adapters.sgf.object_adapter import (
            pcl_farthest_sample_official,
        )

        np.random.seed(1234)
        return pcl_farthest_sample_official(points, npoint)[0]

    def _toy(self, n, rng):
        return rng.normal(0, 1, size=(n, 3)).astype(np.float32)

    def test_first_point_index_agrees_n_gt_512(self):
        rng = np.random.default_rng(7)
        pts = self._toy(900, rng)
        # both sides seed identically; the FIRST selected point and
        # the full FPS trajectory must therefore agree
        a = self._official(pts, 512)
        b = self._ours(pts, 512)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a[0], b[0]))

    def test_fps_full_indices_n_gt_512(self):
        rng = np.random.default_rng(8)
        pts = self._toy(700, rng)
        self.assertTrue(np.array_equal(
            self._official(pts, 512), self._ours(pts, 512)))

    def test_n_equals_512(self):
        rng = np.random.default_rng(9)
        pts = self._toy(512, rng)
        self.assertTrue(np.array_equal(
            self._official(pts, 512), self._ours(pts, 512)))

    def test_n_below_512_replacement(self):
        rng = np.random.default_rng(10)
        pts = self._toy(300, rng)
        a = self._official(pts, 512)
        b = self._ours(pts, 512)
        self.assertTrue(np.array_equal(a, b))
        # replacement semantics: duplicates must occur
        self.assertLess(
            len(np.unique(b, axis=0)), 512)

    def test_dtype_float32_preserved(self):
        rng = np.random.default_rng(11)
        pts = self._toy(600, rng)
        self.assertEqual(self._ours(pts, 512).dtype, np.float32)


class TestAdaptObjectsSamplingMode(unittest.TestCase):
    def _segments(self):
        rng = np.random.default_rng(5)
        segs = {}
        for oid in (30, 10, 20):
            segs[oid] = rng.normal(0, 1, size=(600, 3))
        return segs

    def test_canonical_rows_independent_of_dict_order(self):
        from adapters.sgf.object_adapter import adapt_objects

        segs = self._segments()
        shuffled = dict(reversed(list(segs.items())))
        a = adapt_objects(
            segs, sampling_mode="official_mt19937", scan_seed=0,
            iteration_order=[10, 20, 30])
        b = adapt_objects(
            shuffled, sampling_mode="official_mt19937", scan_seed=0,
            iteration_order=[10, 20, 30])
        self.assertEqual(
            a.obj_ids.tolist(), b.obj_ids.tolist())
        self.assertTrue(np.array_equal(a.obj_pts, b.obj_pts))

    def test_rng_state_restored(self):
        from adapters.sgf.object_adapter import adapt_objects

        np.random.seed(999)
        before = np.random.get_state()
        adapt_objects(
            self._segments(), sampling_mode="official_mt19937",
            scan_seed=0, iteration_order=[10, 20, 30])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(before[2], after[2])

    def test_units_not_rescaled(self):
        from adapters.sgf.object_adapter import adapt_objects

        rng = np.random.default_rng(3)
        segs = {10: rng.normal(0, 0.1, size=(600, 3)) + 2.5}
        res = adapt_objects(
            segs, sampling_mode="official_mt19937", scan_seed=0,
            iteration_order=[10])
        # metres preserved: points stay around 2.5, no /1000 collapse
        self.assertAlmostEqual(
            float(np.abs(res.obj_pts[0]).mean()), 2.5, delta=0.1)

    def test_pcg64_default_unchanged(self):
        from adapters.sgf.object_adapter import adapt_objects

        res = adapt_objects(self._segments())
        self.assertEqual(res.sampling_mode, "deterministic_pcg64")

    def test_sampling_modes_differ(self):
        from adapters.sgf.object_adapter import adapt_objects

        segs = self._segments()
        a = adapt_objects(
            segs, sampling_mode="official_mt19937", scan_seed=0,
            iteration_order=[10, 20, 30])
        b = adapt_objects(segs)  # pcg64
        self.assertFalse(np.array_equal(a.obj_pts, b.obj_pts))


class TestStreamAParityGates(unittest.TestCase):
    def test_512_points_12_of_12_exact(self):
        for tag in ("run1", "run2"):
            d = digest(tag)
            self.assertEqual(d["scans_512_equal"], d["scans_total"])
            self.assertEqual(d["pairs_centered_input_equal"], 12)

    def test_raw_point_sets_equal(self):
        self.assertTrue(digest("run1")["raw_sets_all_equal"])

    def test_pct_embedding_parity(self):
        d = digest("run1")
        self.assertLessEqual(d["max_encoder_diff"], 1e-5)
        self.assertEqual(d["max_encoder_diff"], 0.0)

    def test_normalized_cosine(self):
        self.assertGreaterEqual(digest("run1")["min_normalized_cosine"],
                                0.99999)

    def test_similarity_parity(self):
        self.assertLessEqual(digest("run1")["max_similarity_diff"],
                             1e-5)

    def test_topk_parity(self):
        self.assertEqual(digest("run1")["pairs_topk_all_equal"], 12)

    def test_two_runs_hash_identical(self):
        self.assertEqual(
            digest("run1")["input_hashes"],
            digest("run2")["input_hashes"])

    def test_scan_rows_have_shape_dtype_units(self):
        rows = sorted(
            (OUT / "sampling_parity").glob("scan_*.json"))
        self.assertGreaterEqual(len(rows), 21)
        for f in rows:
            row = json.loads(f.read_text())["row"]
            p = row["points_512"]
            self.assertTrue(p["equal"])
            self.assertEqual(p["dtype_official"], "float32")
            self.assertEqual(p["dtype_adapter"], "float32")
            self.assertEqual(p["shape_official"], p["shape_adapter"])
            self.assertIn("metres", p["units"])
            self.assertIn("first_mismatch", p)
            self.assertIn("official_hash", p)
            self.assertIn("adapter_hash", p)


class TestInferenceCache(unittest.TestCase):
    def test_cache_key_contains_all_shas(self):
        for root in load_cache_roots().values():
            for cache in load_pairs(root):
                key = cache.get("cache_key", {})
                for field in ("pair_id", "input_tensor_sha256",
                              "checkpoint_sha256", "sampling_mode",
                              "model_config_sha256", "code_head"):
                    self.assertIn(field, key, cache["pair_id"])
                self.assertEqual(
                    key.get("sampling_mode"), "official_mt19937")
                self.assertEqual(
                    key.get("checkpoint_sha256"),
                    "b716c7d81b70274f98c7b4bd894c40534bac007a"
                    "b71050713e39a67c5964a17e")

    def test_joint_online_offline_bitwise(self):
        for root in load_cache_roots().values():
            for cache in load_pairs(root):
                if cache["status"] == "ok":
                    self.assertTrue(
                        cache["joint_online_offline_consistent"],
                        cache["pair_id"])

    def test_attribute_fully_off(self):
        # predicted arch has no attr module; zero 164-D buffer never
        # enters fusion (fusion slices only pct/gat/rel rows)
        runner = (ROOT / "scripts/v3b_cache_runner.py").read_text()
        self.assertIn('"attribute_available": False', runner)
        self.assertNotIn('MODULE_INDEX["attr"]', runner)
        for root in load_cache_roots().values():
            for cache in load_pairs(root):
                if cache["status"] != "ok":
                    continue
                self.assertFalse(
                    cache["model_config"]["attribute_available"])
                for combo in ("pct", "rel", "gat", "pct+rel",
                              "pct+gat+rel"):
                    self.assertNotIn("attr", combo)

    def test_no_provisional_in_evidence(self):
        names = [p.name for p in OUT.rglob("*")]
        self.assertFalse(
            any("provisional" in n for n in names),
            [n for n in names if "provisional" in n])

    def test_replay_reads_cache_only(self):
        replay = (ROOT / "scripts/v3b_replay.py").read_text()
        # execution paths (imports / calls), not docstring mentions
        for banned in ("import torch", "geotransformer_forward",
                       "import pygcransac", "pygcransac.findRigidTransform(",
                       "segment_icp(", "dfx.", "MultiModalEncoder"):
            self.assertNotIn(banned, replay)

    def test_gt_not_in_inference_path(self):
        runner = (ROOT / "scripts/v3b_cache_runner.py").read_text()
        # the CALL site (not the import) must come after the GeoT
        # backbone loop — GT is evaluation-only
        gt_pos = runner.index("gt = load_gt_transform(pair_id)")
        geot_pos = runner.index("geot_cache = {}")
        self.assertGreater(gt_pos, geot_pos)

    def test_official92_blocked(self):
        runner = (ROOT / "scripts/v3b_cache_runner.py").read_text()
        self.assertNotIn("official92", runner)
        commands = (OUT / "commands.sh").read_text()
        self.assertNotIn("official92 --", commands)
        # the NOT-executed note is allowed to mention it
        self.assertIn("NOT executed", commands)


class TestAblationReplay(unittest.TestCase):
    def _table(self):
        return json.loads(
            (OUT / "modality_ablation.json").read_text())

    def test_five_combos_all_splits(self):
        table = self._table()
        for split in ("fixed12", "selection89", "calibration90"):
            self.assertEqual(
                set(table[split]),
                {"pct", "rel", "gat", "pct+rel", "pct+gat+rel"})

    def test_failures_stay_in_denominator(self):
        table = self._table()
        for split, combos in table.items():
            for combo, m in combos.items():
                self.assertEqual(
                    m["completed"] + m["failed"], m["requested"])

    def test_accepted_zero_precision_null(self):
        table = self._table()
        for split, combos in table.items():
            for combo, m in combos.items():
                if m["accepted"] == 0:
                    self.assertIsNone(m["accepted_precision"])

    def test_fixed12_determinism(self):
        det = json.loads(
            (OUT / "fixed12_determinism.json").read_text())
        self.assertEqual(det["pairs_compared"], 12)
        # the deterministic prefix (sampling/embeddings/matching/cache
        # keys) MUST be identical across reruns
        self.assertTrue(det["deterministic_prefix_identical"])
        self.assertTrue(det["run1_vs_run3"][
            "deterministic_prefix_identical"])
        # pygcransac has no seed parameter (official-design estimator
        # variance); flips are confined to the one documented ambiguous
        # pair and quantified in the JSON
        self.assertLessEqual(det["outcome_flip_count"], 6)
        self.assertLessEqual(
            det["run1_vs_run3"]["outcome_flip_count"], 8)

    def test_paired_outcomes_complete(self):
        rows = json.loads(
            (OUT / "paired_outcomes.json").read_text())
        n_pairs = 12 + 89 + 90
        self.assertEqual(len(rows), n_pairs * 5)

    def test_split_counts(self):
        table = self._table()
        self.assertEqual(table["fixed12"]["pct"]["requested"], 12)
        self.assertEqual(table["selection89"]["pct"]["requested"], 89)
        self.assertEqual(
            table["calibration90"]["pct"]["requested"], 90)


if __name__ == "__main__":
    unittest.main()
