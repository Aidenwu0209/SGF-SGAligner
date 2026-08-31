"""Phase D adapter contract tests (pre-registered list, 12 items).

Runner-agnostic unittest; skip-with-reason when the local datasets are
unavailable so the suite is runnable in any environment.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from adapters.sgf.data_sources import (
    DATA_ROOT, OracleGraphSource, PredictedGraphSource,
    attribute_vocab_164, load_gt_transform,
)
from adapters.sgf.graph_adapter import (
    adapt_graph, merge_pair_contracts, select_root_official,
)
from adapters.sgf.object_adapter import (
    adapt_objects, farthest_point_sample_deterministic,
)
from adapters.sgf.relation_mapper import RelationMapper
from adapters.sgf.tensor_contract import (
    swap_pair_convention, validate_contract, validate_pair_dict,
)

HAVE_DATA = (DATA_ROOT / "0958220d-e2c2-2de1-9710-c37018da1883").exists()


def synthetic_segments(seed=0, n_objects=6):
    rng = np.random.default_rng(seed)
    segments = {}
    for oid in range(1, n_objects + 1):
        count = 60 + oid * 30  # spans the 50-128 band and above
        segments[oid] = rng.uniform(
            oid, oid + 0.5, size=(count, 3)
        )
    return segments


class TestObjectAdapter(unittest.TestCase):
    def test_50_to_128_point_objects_reach_512(self):
        rng = np.random.default_rng(1)
        small = rng.uniform(0, 0.5, size=(77, 3))
        sampled, replaced = farthest_point_sample_deterministic(
            small, 512, seed=(42, 3)
        )
        self.assertEqual(sampled.shape, (512, 3))
        self.assertTrue(replaced)
        unique = np.unique(sampled, axis=0).shape[0]
        self.assertEqual(unique, 77)

    def test_byte_identical_determinism(self):
        pts = np.random.default_rng(2).uniform(0, 1, (2000, 3))
        a, _ = farthest_point_sample_deterministic(pts, 512, seed=7)
        b, _ = farthest_point_sample_deterministic(pts, 512, seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_object_ids_continuous_and_sorted(self):
        result = adapt_objects(synthetic_segments())
        self.assertEqual(
            result.object_id2idx,
            {int(o): i for i, o in enumerate(result.obj_ids)},
        )
        self.assertEqual(list(result.obj_ids), sorted(result.obj_ids))


class TestGraphAdapter(unittest.TestCase):
    def _contract(self, seed=3):
        segments = synthetic_segments(seed)
        objects = adapt_objects(segments)
        pairs = [(1, 2), (2, 3), (2, 4)]
        triples = [(1, 2, "standing on"), (2, 3, "attached to"),
                   (2, 4, "part of")]
        return adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=pairs, relation_triples=triples,
        ), objects, pairs

    def test_edges_in_range(self):
        contract, objects, _pairs = self._contract()
        n = len(objects.obj_ids)
        self.assertTrue((contract.edges >= 0).all())
        self.assertTrue((contract.edges < n).all())

    def test_root_matches_official_algorithm(self):
        # object 2 has the highest total degree (3 appearances)
        pairs = [(1, 2), (2, 3), (2, 4)]
        root = select_root_official(pairs, [1, 2, 3, 4])
        self.assertEqual(root, 2)
        contract, objects, _ = self._contract()
        self.assertEqual(contract.provenance["root_obj_id"], 2)
        root_idx = objects.object_id2idx[2]
        self.assertTrue(
            np.allclose(contract.tot_rel_pose[root_idx], 0.0, atol=1e-12)
        )

    def test_relation_bow_shape_41(self):
        contract, _, _ = self._contract()
        self.assertEqual(
            contract.tot_bow_vec_object_edge_feats.shape[1], 41
        )

    def test_predicted_mode_no_attribute_fabrication(self):
        contract, _, _ = self._contract()
        self.assertIsNone(contract.tot_bow_vec_object_attr_feats)
        self.assertFalse(contract.modality_mask["attr"])

    def test_missing_modality_excluded(self):
        mapper = RelationMapper()
        vector, mask = mapper.bow_vector(["same part", "standing on"])
        # 'same part' unmapped -> contributes nothing anywhere
        self.assertEqual(vector.sum(), 1.0)
        self.assertEqual(mask.sum(), 1.0)

    def test_none_supplementation(self):
        contract, objects, pairs = self._contract()
        n = len(objects.obj_ids)
        # all ordered pairs incl. none supplementation: n*(n-1)
        self.assertEqual(len(contract.edges), n * (n - 1))


@unittest.skipUnless(HAVE_DATA, "3RScan data unavailable")
class TestRealDataContract(unittest.TestCase):
    SCAN = "0958220d-e2c2-2de1-9710-c37018da1883"

    def test_oracle_contract_valid_with_164_attributes(self):
        source = OracleGraphSource()
        result = source.load(self.SCAN)
        objects = adapt_objects(result.segments)
        contract = adapt_graph(
            objects, mode="oracle",
            directed_pairs=result.directed_pairs,
            relation_triples=result.relation_triples,
            attributes_per_object=result.attributes_per_object,
            attribute_vocab=attribute_vocab_164(),
        )
        self.assertEqual(
            contract.tot_bow_vec_object_attr_feats.shape[1], 164
        )
        self.assertEqual(validate_contract(contract), [])

    def test_predicted_contract_valid_no_attributes(self):
        source = PredictedGraphSource()
        result = source.load(self.SCAN)
        objects = adapt_objects(result.segments)
        contract = adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=result.directed_pairs,
            relation_triples=result.relation_triples,
        )
        self.assertIsNone(contract.tot_bow_vec_object_attr_feats)
        self.assertEqual(validate_contract(contract), [])


class TestPairConvention(unittest.TestCase):
    def test_swap_order(self):
        segments = synthetic_segments(5)
        objects = adapt_objects(segments)
        c1 = adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=[(1, 2)],
            relation_triples=[(1, 2, "attached to")],
        )
        c2 = adapt_graph(
            objects, mode="sgf_predicted",
            directed_pairs=[(3, 4)],
            relation_triples=[(3, 4, "standing on")],
        )
        c1.scene_ids = ["s1"]
        c2.scene_ids = ["s2"]
        merged = merge_pair_contracts(c1, c2, np.zeros(3))
        self.assertEqual(validate_pair_dict(
            {**merged, "object_id2idx": merged["src_object_id2idx"]}
        ) or validate_pair_dict(merged), [])
        swapped = swap_pair_convention(merged)
        # counts swap; total preserved; transform convention documented
        self.assertEqual(
            int(swapped["graph_per_obj_count"][0]),
            int(merged["graph_per_obj_count"][1]),
        )
        self.assertEqual(
            swapped["tot_obj_pts"].shape, merged["tot_obj_pts"].shape
        )

    def test_units_metres(self):
        rng = np.random.default_rng(9)
        segments = {7: rng.uniform(0, 0.4, (800, 3))}  # metres-scale object
        objects = adapt_objects(segments)
        extent = float(np.abs(objects.obj_pts[0]).max())
        self.assertLess(extent, 5.0, "metre-scale object must stay small")


class TestFailClosed(unittest.TestCase):
    def test_empty_graph_rejected(self):
        with self.assertRaises(ValueError):
            adapt_objects({})

    def test_single_node_graph_is_legal(self):
        rng = np.random.default_rng(4)
        objects = adapt_objects({1: rng.uniform(0, 1, (600, 3))})
        contract = adapt_graph(objects, mode="sgf_predicted")
        self.assertEqual(len(contract.edges), 0)  # n*(n-1) = 0

    def test_relation_mapping_dump(self):
        import tempfile

        mapper = RelationMapper()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "relation_mapping.json"
            mapper.dump_mapping(path)
            payload = json.loads(path.read_text())
            self.assertIn("same part", payload["unmapped_sgf_relations"])
            self.assertIn(
                "standing on", payload["sgf_to_official"]
            )


if __name__ == "__main__":
    unittest.main()
