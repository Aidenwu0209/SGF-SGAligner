import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety.v16_b716_candidate_plan import (
    B716PlanError, OFFICIAL_CHECKPOINT_EPOCH, OFFICIAL_CODE_HEAD,
    OFFICIAL_MODEL_CONFIG_SHA256, OFFICIAL_RELEASE_SHA256,
    canonical_boundary, freeze_existing_geot, safe_pair_metadata,
    sha256_file, validate_pair_metadata, write_deterministic_npz,
)


class TestV16B716CandidatePlan(unittest.TestCase):
    def test_pair_combo_node_metrics_are_not_decoded(self):
        value = {
            "pair_id": "a_to_b", "mode": "official_sgf_predicted",
            "sampling_mode": "official_mt19937", "scan_seed": 0,
            "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "checkpoint_epoch": OFFICIAL_CHECKPOINT_EPOCH,
            "code_head": OFFICIAL_CODE_HEAD, "model_config": {},
            "cache_key": {
                "pair_id": "a_to_b", "input_tensor_sha256": "a" * 64,
                "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
                "sampling_mode": "official_mt19937",
                "model_config_sha256": OFFICIAL_MODEL_CONFIG_SHA256,
                "code_head": OFFICIAL_CODE_HEAD,
            },
            "status": "ok", "joint_online_offline_consistent": True,
            "geot_node_pairs": {},
            "combos": {"pct": {"node_metrics": {"gt_secret": [1, 2, 3]}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(json.dumps(value))
            observed = safe_pair_metadata(path)
        self.assertNotIn("combos", observed)
        self.assertNotIn("node_metrics", json.dumps(observed))
        validate_pair_metadata(observed)

    def test_checkpoint_domain_mismatch_fails_closed(self):
        meta = {
            "pair_id": "a_to_b", "mode": "official_sgf_predicted",
            "sampling_mode": "official_mt19937", "scan_seed": 0,
            "checkpoint_sha256": "89ed",
            "checkpoint_epoch": OFFICIAL_CHECKPOINT_EPOCH,
            "code_head": OFFICIAL_CODE_HEAD,
            "model_config": {}, "cache_key": {}, "status": "ok",
            "joint_online_offline_consistent": True, "geot_node_pairs": {},
        }
        with self.assertRaisesRegex(B716PlanError, "not official release"):
            validate_pair_metadata(meta)

    def test_src_count_requires_three_agreeing_boundaries(self):
        arrays = {
            "tot_obj_pts": np.zeros((4, 512, 3), np.float32),
            "tot_rel_pose": np.zeros((4, 3), np.float32),
            "tot_bow_vec_object_edge_feats": np.zeros((4, 41), np.float32),
            "edges": np.zeros((0, 2), np.int64),
            "obj_ids": np.array([10, 20, 30, 40], np.int64),
        }
        data = dict(arrays)
        data.update({
            "src_count": 2, "graph_per_obj_count": np.array([2, 2]),
            "src_object_id2idx": {10: 0, 20: 1},
            "ref_object_id2idx": {30: 0, 40: 1},
            "registration_pts": {i: np.ones((2, 3)) * i for i in range(4)},
            "registration_id2oid": {0: 10, 1: 20, 2: 30, 3: 40},
        })
        self.assertEqual(canonical_boundary(data, arrays)["src_count"], 2)
        data["src_object_id2idx"] = {10: 0}
        with self.assertRaisesRegex(B716PlanError, "object-table proof"):
            canonical_boundary(data, arrays)

    def test_existing_and_missing_geot_are_accounted_without_execution(self):
        candidates = [
            {"source_index": 0, "reference_index": 2},
            {"source_index": 1, "reference_index": 3},
        ]
        data = {"obj_ids": np.array([10, 20, 30, 40])}
        meta = {"0_2": {"status": "ok", "cache_row": 7,
                         "src_object_id": 10, "ref_object_id": 30}}
        arrays = {
            "src_corr_7": np.ones((3, 3), np.float32),
            "ref_corr_7": np.ones((3, 3), np.float32) * 2,
            "scores_7": np.ones(3, np.float32),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geot.npz"
            np.savez(path, **arrays)
            rows, copied, counts = freeze_existing_geot(
                candidates, meta, path, data)
        self.assertEqual(counts["existing_reused"], 1)
        self.assertEqual(counts["missing_disabled"], 1)
        self.assertEqual(counts["new_geot_executed"], 0)
        self.assertTrue(rows[0]["immutable"])
        self.assertFalse(rows[1]["immutable"])
        self.assertEqual(rows[1]["status"], "disabled_missing_geotransformer")
        self.assertEqual(set(copied), {"src_corr_0", "ref_corr_0", "scores_0"})

    def test_deterministic_npz_bytes(self):
        arrays = {"b": np.arange(5), "a": np.eye(3, dtype=np.float32)}
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.npz", Path(directory) / "b.npz"
            write_deterministic_npz(first, arrays)
            write_deterministic_npz(second, dict(reversed(list(arrays.items()))))
            self.assertEqual(sha256_file(first), sha256_file(second))

    def test_deterministic_npz_is_create_only_and_accepts_identical_replay(self):
        arrays = {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.npz"
            write_deterministic_npz(path, arrays)
            before = path.read_bytes()
            inode = path.stat().st_ino
            write_deterministic_npz(path, arrays)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_ino, inode)
            with self.assertRaisesRegex(B716PlanError, "immutable NPZ"):
                write_deterministic_npz(
                    path, {"a": np.ones((2, 3), dtype=np.float32)})
            self.assertEqual(path.read_bytes(), before)

    def test_deterministic_npz_refuses_stale_or_tampered_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immutable.npz"
            path.write_bytes(b"stale-or-tampered")
            before = path.read_bytes()
            with self.assertRaisesRegex(B716PlanError, "immutable NPZ"):
                write_deterministic_npz(
                    path, {"a": np.arange(3, dtype=np.float32)})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
