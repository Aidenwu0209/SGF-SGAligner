import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety.v16_matched_region_colorpcr import (
    V16ContractError, array_sha256, build_hypothesis_artifact,
    build_side_union, canonical_provenance_binding,
    canonical_surface_from_rows, load_raw_inseg, node_object_id,
    raw_float32_sha256, reject_forbidden_fields, resolve_unique_inseg_path,
    sha256_file, stable_json_sha256, validate_hypothesis, validate_metres,
    verify_canonical_surface, verify_file, write_deterministic_npz,
)


class V16BuilderTests(unittest.TestCase):
    def raw(self, path: Path, *, scan_shift=0.0, conflicting=False, scale=1.0):
        a = np.stack([np.linspace(0, .059, 60), np.zeros(60), np.zeros(60)], axis=1)
        b = np.stack([np.linspace(0, .059, 60), np.ones(60), np.zeros(60)], axis=1)
        xyz = (np.concatenate([a, b]) + np.array([scan_shift, 0, 0])) * scale
        labels = np.r_[np.full(60, 7), np.full(60, 8)].astype(np.int64)
        if conflicting:
            xyz[60] = xyz[0]
        colors = np.tile(np.array([[10, 20, 30, 255]], np.uint8), (120, 1))
        np.savez(path, xyz=xyz.astype(np.float32), labels=labels, colors=colors)

    def canonical(self, source, reference):
        _, s7 = canonical_surface_from_rows(source, 7)
        _, s8 = canonical_surface_from_rows(source, 8)
        _, r7 = canonical_surface_from_rows(reference, 7)
        _, r8 = canonical_surface_from_rows(reference, 8)
        return {"src_count": 2, "obj_ids": np.array([7, 8, 7, 8]),
                "registration_pts": {0: s7, 1: s8, 2: r7, 3: r8},
                "edges_explicit": np.array([[0, 1], [2, 3]], np.int64)}

    def chain(self, data):
        surfaces = [{"index": int(index), "points": len(points),
                     "sha256": raw_float32_sha256(points)}
                    for index, points in sorted(data["registration_pts"].items())]
        provenance = {
            "cache_schema": "v6fix-inference-cache-v2",
            "cache_key": "i" * 64,
            "pair_id": "s_to_r",
            "checkpoint_id": "B",
            "checkpoint_sha256": "c" * 64,
            "object_ids_order": [7, 8, 7, 8],
            "src_count": 2,
            "unit": "metres",
            "registration_surfaces": surfaces,
            "source_hashes": {"scripts/canonical_inputs.py": "d" * 64},
        }
        source = {
            "cache_schema": "v6fix-inference-cache-v2", "pair_id": "s_to_r",
            "checkpoint_id": "B", "checkpoint_sha256": "c" * 64,
            "input_sha256": "i" * 64, "embedding_sha256": "e" * 64,
            "rank_list": [[0, 1, 2, 3]] * 4, "provenance": provenance,
        }
        candidate_provenance = copy.deepcopy(provenance)
        candidate_provenance["v10_candidate_contract"] = {"adapter_only": True}
        cache = {
            **source, "candidate_fingerprint": "f" * 64,
            "source_cache_path": "/tmp/source.pt",
            "source_cache_sha256": "s" * 64,
            "provenance": candidate_provenance,
        }
        canonical_payload = {
            "src_count": 2,
            "surface_hashes": {str(row["index"]): row["sha256"]
                               for row in surfaces},
            "explicit_edges": data["edges_explicit"].tolist(),
        }
        plan = {
            "v10_cache_sha256": "v" * 64,
            "candidate_fingerprint": "f" * 64,
            "checkpoint_sha256": "c" * 64,
            "canonical_input_sha256": stable_json_sha256(canonical_payload),
        }
        return cache, source, plan

    def records(self):
        return [
            {"source_index": 0, "reference_index": 2,
             "forward_cross_rank": 1, "reverse_cross_rank": 1,
             "worst_cross_rank": 1, "rank_sum": 2},
            {"source_index": 1, "reference_index": 3,
             "forward_cross_rank": 2, "reverse_cross_rank": 2,
             "worst_cross_rank": 2, "rank_sum": 4},
            {"source_index": 0, "reference_index": 3,
             "forward_cross_rank": 3, "reverse_cross_rank": 3,
             "worst_cross_rank": 3, "rank_sum": 6},
        ]

    def hypothesis(self):
        records = self.records()
        payload = {"members": [[r["source_index"], r["reference_index"]] for r in records],
                   "member_rank_records": records, "member_count": 3}
        return {"hypothesis_index": 0, **payload,
                "hypothesis_sha256": stable_json_sha256(payload)}

    def test_cache_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            path.write_bytes(b"sealed")
            expected = sha256_file(path)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(V16ContractError, "SHA mismatch"):
                verify_file(path, expected)

    def test_rank_record_order_tamper_fails(self):
        hypothesis = self.hypothesis()
        hypothesis["member_rank_records"] = list(reversed(hypothesis["member_rank_records"]))
        with self.assertRaisesRegex(V16ContractError, "rank/member order"):
            validate_hypothesis(hypothesis, self.records())

    def test_node_object_missing_and_duplicate_fail(self):
        with self.assertRaisesRegex(V16ContractError, "missing"):
            node_object_id({"src_count": 1, "obj_ids": np.array([7, 8])}, 3, side="reference")
        with self.assertRaisesRegex(V16ContractError, "not unique"):
            node_object_id({"src_count": 2, "obj_ids": np.array([7, 7, 8])}, 0, side="source")

    def test_surface_hash_mismatch_fails(self):
        expected = np.zeros((60, 3), np.float64)
        with self.assertRaisesRegex(V16ContractError, "surface hash mismatch"):
            verify_canonical_surface({"registration_pts": {0: expected}}, 0,
                                     np.ones((60, 3), np.float64))

    def test_millimetre_scale_fails(self):
        xyz = np.array([[0, 0, 0], [1000, 0, 0]], np.float32)
        with self.assertRaisesRegex(V16ContractError, "metre scale"):
            validate_metres(xyz)

    def test_instance_mapping_conflict_and_duplicate_path_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bad.npz"
            self.raw(path, conflicting=True)
            with self.assertRaisesRegex(V16ContractError, "conflicting"):
                load_raw_inseg(path, scan_id="s", side="source")
            for name in ("a", "b"):
                (root / name / "scan").mkdir(parents=True)
                self.raw(root / name / "scan/inseg_cloud.npz")
            with self.assertRaisesRegex(V16ContractError, "not unique: 2"):
                resolve_unique_inseg_path("scan", [root / "a", root / "b"])

    def test_forbidden_gt_and_fallback_fields_fail(self):
        for payload in ({"gt_transform": np.eye(4).tolist()},
                        {"nested": {"fallback_used": False}},
                        {"selection_label": 1}):
            with self.assertRaisesRegex(V16ContractError, "forbidden field"):
                reject_forbidden_fields(payload)

    def test_input_order_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.raw(root / "s.npz"); self.raw(root / "r.npz", scan_shift=.2)
            source = load_raw_inseg(root / "s.npz", scan_id="s", side="source")
            reference = load_raw_inseg(root / "r.npz", scan_id="r", side="reference")
            data = self.canonical(source, reference)
            records = self.records()[:2]
            first, evidence1 = build_side_union(records, data, source, side="source")
            second, evidence2 = build_side_union(list(reversed(records)), data, source, side="source")
            self.assertEqual(evidence1, evidence2)
            for key in first:
                self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_same_object_id_across_scans_is_side_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.raw(root / "s.npz"); self.raw(root / "r.npz", scan_shift=.2)
            source = load_raw_inseg(root / "s.npz", scan_id="s", side="source")
            reference = load_raw_inseg(root / "r.npz", scan_id="r", side="reference")
            data = self.canonical(source, reference)
            src, src_e = build_side_union(self.records()[:1], data, source, side="source")
            ref, ref_e = build_side_union(self.records()[:1], data, reference, side="reference")
            self.assertEqual(src_e[0]["object_id"], ref_e[0]["object_id"])
            self.assertNotEqual(src_e[0]["scan_id"], ref_e[0]["scan_id"])
            self.assertNotEqual(array_sha256(src["xyz"]), array_sha256(ref["xyz"]))

    def test_missing_scan_or_side_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.npz"; self.raw(path)
            with self.assertRaisesRegex(V16ContractError, "scan_id and side"):
                load_raw_inseg(path, scan_id="", side="source")
            with self.assertRaisesRegex(V16ContractError, "scan_id and side"):
                load_raw_inseg(path, scan_id="s", side="")

    def test_deterministic_npz_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = {"b": np.arange(5, dtype=np.int64),
                      "a": np.eye(3, dtype=np.float32)}
            write_deterministic_npz(root / "one.npz", arrays)
            write_deterministic_npz(root / "two.npz", dict(reversed(list(arrays.items()))))
            self.assertEqual(sha256_file(root / "one.npz"), sha256_file(root / "two.npz"))

    def test_recursive_canonical_provenance_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.raw(root / "s.npz"); self.raw(root / "r.npz", scan_shift=.2)
            source = load_raw_inseg(root / "s.npz", scan_id="s", side="source")
            reference = load_raw_inseg(root / "r.npz", scan_id="r", side="reference")
            data = self.canonical(source, reference)
            cache, source_cache, plan = self.chain(data)
            binding = canonical_provenance_binding(
                data, cache, source_cache, plan,
                v10_cache_sha256="v" * 64, source_cache_sha256="s" * 64)
            self.assertEqual(binding["canonical_src_count"], 2)
            self.assertEqual(binding["canonical_obj_ids"], [7, 8, 7, 8])
            self.assertEqual(binding["canonical_input_sha256"],
                             plan["canonical_input_sha256"])

    def test_recursive_chain_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.raw(root / "s.npz"); self.raw(root / "r.npz", scan_shift=.2)
            source = load_raw_inseg(root / "s.npz", scan_id="s", side="source")
            reference = load_raw_inseg(root / "r.npz", scan_id="r", side="reference")
            data = self.canonical(source, reference)
            cache, source_cache, plan = self.chain(data)
            cases = []
            bad_plan = copy.deepcopy(plan); bad_plan["canonical_input_sha256"] = "0" * 64
            cases.append((data, cache, source_cache, bad_plan, "canonical_input"))
            bad_data = copy.deepcopy(data); bad_data["obj_ids"] = np.array([8, 7, 7, 8])
            cases.append((bad_data, cache, source_cache, plan, "obj_ids/src_count"))
            bad_surface = copy.deepcopy(data)
            bad_surface["registration_pts"][0] = bad_surface["registration_pts"][0] + .001
            cases.append((bad_surface, cache, source_cache, plan, "registration surfaces"))
            bad_source = copy.deepcopy(source_cache); bad_source["checkpoint_sha256"] = "x" * 64
            cases.append((data, cache, bad_source, plan, "checkpoint chain"))
            bad_provenance_cache = copy.deepcopy(cache)
            bad_provenance_cache["provenance"]["src_count"] = 1
            cases.append((data, bad_provenance_cache, source_cache, plan,
                          "not a copy"))
            for current, candidate, frozen_source, frozen_plan, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                        V16ContractError, message):
                    canonical_provenance_binding(
                        current, candidate, frozen_source, frozen_plan,
                        v10_cache_sha256="v" * 64,
                        source_cache_sha256="s" * 64)
            with self.assertRaisesRegex(V16ContractError, "source-cache SHA"):
                canonical_provenance_binding(
                    data, cache, source_cache, plan,
                    v10_cache_sha256="v" * 64,
                    source_cache_sha256="z" * 64)

    def test_builder_writes_prepared_voxel10_without_direct_fps512(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.raw(root / "s.npz"); self.raw(root / "r.npz", scan_shift=.2)
            source = load_raw_inseg(root / "s.npz", scan_id="s", side="source")
            reference = load_raw_inseg(root / "r.npz", scan_id="r", side="reference")
            artifact = build_hypothesis_artifact(
                "s_to_r", self.hypothesis(), self.records(),
                self.canonical(source, reference), source, reference,
                root / "out", {"sealed": True})
            with np.load(artifact["npz_path"], allow_pickle=False) as value:
                self.assertIn("source_voxel10_xyz", value.files)
                self.assertIn("reference_voxel10_xyz", value.files)
                self.assertFalse(any("fps512" in key for key in value.files))
            evidence = json.loads(Path(artifact["evidence_path"]).read_text())
            self.assertFalse(evidence["preprocessing"]["builder_cap512_applied"])
            self.assertTrue(evidence["preprocessing"]["official_worker_owns_cap512"])
            self.assertFalse(evidence["colorpcr_consumption_allowed"])


if __name__ == "__main__":
    unittest.main()
