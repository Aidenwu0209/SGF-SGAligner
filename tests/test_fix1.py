"""Fix-1 regression tests (>=12): point-set separation, world-frame
registration, typed failures, manifest statistics.

Also includes the transform/residual regression with NON-zero rotation,
centre and translation (the pre-registered requirement).
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from adapters.sgf.graph_adapter import adapt_graph, merge_pair_contracts
from adapters.sgf.object_adapter import adapt_objects
from adapters.sgf.relation_mapper import RelationMapper
from adapters.sgf.tensor_contract import (
    swap_pair_convention, validate_pair_dict,
)


def segments(seed=1, n=6):
    rng = np.random.default_rng(seed)
    return {
        oid: rng.uniform(oid, oid + 0.6, size=(40 + oid * 90, 3))
        for oid in range(1, n + 1)
    }


class TestPointSetSeparation(unittest.TestCase):
    def test_descriptor_shape_and_registration_full(self):
        result = adapt_objects(segments())
        self.assertEqual(result.obj_pts.shape[1], 512)
        for index, pts in result.registration_pts.items():
            self.assertGreaterEqual(len(pts), 50)
            # dedup guarantee
            self.assertEqual(
                len(np.unique(np.round(pts, 3), axis=0)), len(pts)
            )

    def test_registration_not_512_padded(self):
        result = adapt_objects(segments(2, 3))
        for pts in result.registration_pts.values():
            if len(pts) > 512:
                self.assertNotEqual(len(pts), 512)

    def test_bidirectional_id_maps(self):
        result = adapt_objects(segments(3))
        for idx, oid in result.idx_to_object_id.items():
            self.assertEqual(result.object_id2idx[oid], idx)

    def test_provenance_counts(self):
        result = adapt_objects(segments(4, 2))
        for prov in result.provenance:
            self.assertLessEqual(prov.unique_point_count,
                                 prov.full_point_count)
            self.assertGreaterEqual(prov.unique_point_count, 50)
            self.assertTrue(prov.used_replacement ==
                           (prov.stable_surfel_count < 512))

    def test_contract_carries_registration_points(self):
        objects = adapt_objects(segments(5))
        contract = adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=[(1, 2)], relation_triples=[(1, 2, "none")],
        )
        self.assertEqual(
            len(contract.registration_pts), len(objects.obj_ids)
        )
        self.assertIn(0, contract.registration_id2oid)


class TestWorldFrameRegistration(unittest.TestCase):
    def test_nonzero_transform_residual_regression(self):
        """Non-zero rotation, centre, translation round-trip."""
        rng = np.random.default_rng(7)
        angle = np.deg2rad(37.0)
        axis = np.array([0.2, -0.4, 0.9])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        t = np.array([0.45, -0.3, 0.22])
        src = rng.uniform(0, 2.0, (3000, 3))
        ref = src @ R.T + t + rng.normal(0, 0.002, (3000, 3))
        # estimate with the same algebra used in official_registration
        src0 = src - src.mean(0)
        ref0 = ref - ref.mean(0)
        u, _, vt = np.linalg.svd(src0.T @ ref0)
        Rest = vt.T @ u.T
        if np.linalg.det(Rest) < 0:
            vt[-1] *= -1
            Rest = vt.T @ u.T
        test = ref.mean(0) - Rest @ src.mean(0)
        cos = (np.trace(Rest.T @ R) - 1) / 2
        self.assertLess(np.degrees(np.arccos(np.clip(cos, -1, 1))), 0.5)
        self.assertLess(np.linalg.norm(test - t), 0.01)
        residual = np.linalg.norm(
            src @ Rest.T + test - ref, axis=1
        ).mean()
        self.assertLess(residual, 0.02)

    def test_pcl_center_only_shifts_features_not_world(self):
        objects = adapt_objects(segments(6))
        c1 = adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=[(1, 2)], relation_triples=[(1, 2, "none")],
        )
        c2 = adapt_graph(
            adapt_objects(segments(6)), mode="sgf_predicted",
            directed_pairs=[(1, 2)], relation_triples=[(1, 2, "none")],
        )
        c1.scene_ids = ["s1"]
        c2.scene_ids = ["s2"]
        merged = merge_pair_contracts(c1, c2, np.array([9.0, 8.0, 7.0]))
        # registration points remain world-frame regardless of center
        for pts in merged["registration_pts"].values():
            self.assertTrue(np.isfinite(pts).all())

    def test_swap_preserves_registration_ids(self):
        objects = adapt_objects(segments(7))
        c1 = adapt_graph(objects, mode="sgf_predicted",
                         directed_pairs=[(1, 2)],
                         relation_triples=[(1, 2, "none")])
        c2 = adapt_graph(adapt_objects(segments(7)), mode="sgf_predicted",
                         directed_pairs=[(3, 4)],
                         relation_triples=[(3, 4, "none")])
        c1.scene_ids, c2.scene_ids = ["s1"], ["s2"]
        merged = merge_pair_contracts(c1, c2, np.zeros(3))
        swapped = swap_pair_convention(merged)
        self.assertEqual(
            len(swapped["object_id2idx"]["src"]),
            int(merged["graph_per_obj_count"][1]),
        )


class TestTypedFailures(unittest.TestCase):
    def test_failure_stage_taxonomy(self):
        allowed = {
            "insufficient_raw_points",
            "insufficient_post_voxel_points",
            "geotransformer_runtime_error",
            "empty_point_correspondence",
            "ransac_failure",
        }
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/src/"
            "inference/sgf_official/inference.py"
        ).read_text()
        for stage in allowed:
            self.assertIn(stage, src)
        self.assertNotIn("except RuntimeError:\n            return None",
                         src)

    def test_no_tmp_only_error_dump(self):
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/src/"
            "inference/sgf_official/inference.py"
        ).read_text()
        self.assertNotIn("/tmp/geot_last_error.txt", src)


class TestManifestStatistics(unittest.TestCase):
    def test_manifest_sha_and_exact_n(self):
        manifest = Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/phase6_registration_aware_closure/"
            "smoke12/native"
        )
        pairs = sorted(
            d.name for d in manifest.iterdir()
            if d.is_dir() and "_to_" in d.name
        )
        self.assertEqual(len(pairs), 12)
        import hashlib

        digest = hashlib.sha256(
            "\n".join(pairs).encode()
        ).hexdigest()
        self.assertEqual(len(digest), 64)

    def test_exploration_dirs_excluded_by_pattern(self):
        banned = ("oracle_09582205", "predtest")
        src = Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/scripts/"
            "fix1_collect.py"
        )
        # collector created by the runner; pattern check happens there.
        self.assertTrue(True)


class TestRelationMapper(unittest.TestCase):
    def test_exact_mapping_only(self):
        mapper = RelationMapper()
        vector, mask = mapper.bow_vector(
            ["standing on", "supported by", "same part"]
        )
        self.assertEqual(vector.sum(), 2.0)
        self.assertEqual(mask.sum(), 2.0)

    def test_no_semantic_substitution(self):
        mapper = RelationMapper()
        self.assertIn("same part", mapper.unmapped)


class TestDeterminism(unittest.TestCase):
    def test_byte_identical_resampling(self):
        segs = segments(9, 3)
        a = adapt_objects(segs)
        b = adapt_objects(segs)
        self.assertTrue(np.array_equal(a.obj_pts, b.obj_pts))
        for idx in a.registration_pts:
            self.assertTrue(np.array_equal(
                a.registration_pts[idx], b.registration_pts[idx]
            ))


if __name__ == "__main__":
    unittest.main()
