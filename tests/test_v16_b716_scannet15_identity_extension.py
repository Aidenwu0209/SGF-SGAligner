from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety.v13_dual_solver_runtime import sha256_file
from safety.v16_b716_scannet15_v13_gate_bridge import load_verified_surfaces
from safety.v16_b716_scannet15_identity import (
    POLICY_FALSE_FIELDS, PREPARED_SCHEMA, PREREGISTER_SCHEMA,
    ScanNet15IdentityError, stable_json_sha256, validate_prepared_npz,
    validate_preregister,
)
from scripts.v16_b716_scannet15_v14_identity import validate_pair
from scripts.v16_b716_scannet15_corr_cache_converter import convert


H64 = "a" * 64


def _sealed(value):
    value = dict(value); value["payload_sha256"] = stable_json_sha256(value)
    return value


def _manifest(scene="scene0000_00"):
    return _sealed({
        "schema": PREPARED_SCHEMA, "scene_id": scene,
        "pair_role": "same-terminal-surface-spatial-partition",
        "arm": "sgf_selected_union", "unit": "metre",
        "attribute_available": False,
        "source_raw_ply_sha256": "1" * 64,
        "reference_raw_ply_sha256": "2" * 64,
        "raw_pair_inventory_sha256": "3" * 64,
        "raw_pair_receipt_sha256": "4" * 64,
        "sgf_prediction_sha256": "5" * 64,
        "official_repo_head": "6" * 40,
        "official_checkpoint_sha256": "7" * 64,
        "geotransformer_checkpoint_sha256": "8" * 64,
        "bridge_source_sha256": "9" * 64,
        "colorpcr_schema_source_sha256": "a" * 64,
        "official_sgaligner_tensor_contract_compatible": True,
        "v13_colorpcr_worker_schema_compatible": True,
        "source_pretransformed": False, "transform_present": False,
        "registration_executed": False, "gt_consumed": False,
        "worker_execution_authorized": False,
        "formal_execution_authorized": False,
    })


def _npz(path: Path, manifest):
    obj = np.arange(4, dtype=np.int64)
    offsets = np.arange(0, 41, 10, dtype=np.int64)
    values = {
        "manifest_json": np.asarray(json.dumps(manifest, sort_keys=True)),
        "official_edges": np.asarray([[0, 1]], np.int64),
        "official_graph_per_edge_count": np.asarray([1, 0], np.int64),
        "official_graph_per_obj_count": np.asarray([2, 2], np.int64),
        "official_obj_ids": obj, "official_pcl_center": np.zeros(3, np.float64),
        "official_registration_node_indices": obj,
        "official_registration_object_ids": obj,
        "official_registration_offsets": offsets,
        "official_registration_xyz": np.zeros((40, 3), np.float32),
        "official_src_count": np.asarray(2, np.int64),
        "official_tot_bow_vec_object_edge_feats": np.zeros((4, 41), np.float32),
        "official_tot_obj_pts": np.zeros((4, 512, 3), np.float32),
        "official_tot_rel_pose": np.zeros((4, 3), np.float32),
    }
    xyz = np.stack([np.arange(40) * .1 + .05,
                    np.zeros(40) + .05, np.zeros(40) + .05], axis=1).astype(np.float32)
    labels = np.repeat(np.asarray([1, 2], np.int64), 20)
    for side in ("source", "reference"):
        p = f"sgf_selected_union_{side}_"
        values.update({
            p + "xyz": xyz, p + "colors": np.zeros((40, 3), np.uint8),
            p + "labels": labels, p + "membership_object_ids": labels,
            p + "source_row_indices": np.arange(40, dtype=np.int64),
            p + "member_offsets": np.asarray([0, 20, 40], np.int64),
            p + "voxel10_xyz": xyz,
            p + "voxel10_colors_mean_0_255": np.zeros((40, 3), np.float32),
            p + "voxel10_keys": np.floor(xyz / np.float32(.1)).astype(np.int64),
            p + "voxel10_source_offsets": np.arange(41, dtype=np.int64),
            p + "voxel10_source_row_indices_flat": np.arange(40, dtype=np.int64),
        })
    np.savez(path, **values)


def _preregister(root: Path, prepared: Path, manifest):
    rows = []
    for index in range(15):
        scene = f"scene{index:04d}_00"; pair = f"{scene}_source_to_reference"
        row = {
            "pair_id": pair, "scene_id": scene,
            "prepared_npz_path": str(prepared.resolve() if index == 0 else root / f"p{index}"),
            "prepared_npz_sha256": sha256_file(prepared) if index == 0 else H64,
            "prepared_manifest_path": str(root / f"m{index}.json"),
            "prepared_manifest_sha256": H64,
            "prepared_manifest_payload_sha256": manifest["payload_sha256"] if index == 0 else H64,
            "raw_inventory_path": str(root / "raw.json"),
            "raw_inventory_sha256": manifest["raw_pair_inventory_sha256"] if index == 0 else H64,
            "raw_pair_receipt_path": str(root / f"r{index}.json"),
            "raw_pair_receipt_sha256": manifest["raw_pair_receipt_sha256"] if index == 0 else H64,
            "source_raw_ply_path": str(root / f"s{index}.ply"),
            "source_raw_ply_sha256": manifest["source_raw_ply_sha256"] if index == 0 else H64,
            "reference_raw_ply_path": str(root / f"t{index}.ply"),
            "reference_raw_ply_sha256": manifest["reference_raw_ply_sha256"] if index == 0 else H64,
            "sgf_prediction_path": str(root / f"g{index}.json"),
            "sgf_prediction_sha256": manifest["sgf_prediction_sha256"] if index == 0 else H64,
        }
        row["identity_payload_sha256"] = stable_json_sha256(row); rows.append(row)
    source = root / "source.py"; source.write_text("# fixture\n")
    value = {
        "schema": PREREGISTER_SCHEMA, "pair_count": 15,
        "pair_ids": [row["pair_id"] for row in rows], "pairs": rows,
        "raw_inventory_sha256": "3" * 64,
        "prepared_bridge_summary_sha256": H64,
        "official_repo_head": "6" * 40,
        "official_checkpoint_sha256": "7" * 64,
        "geotransformer_checkpoint_sha256": "8" * 64,
        "sgf_model_closure_sha256": H64,
        "bridge_source_sha256": "9" * 64,
        "colorpcr_schema_source_sha256": "a" * 64,
        "source_paths": {"fixture": str(source.resolve())},
        "source_sha256": {"fixture": sha256_file(source)},
        "dependency_closure_sha256": H64,
        "primary_arm": "sgf_selected_union", "selection_rule": "frozen",
        "allow_real_pilot": False, "allow_gpu_pilot": False,
        "posthoc_allowed": False, "algorithm_or_threshold_change": False,
        **{field: False for field in POLICY_FALSE_FIELDS},
    }
    return _sealed(value)


class Tests(unittest.TestCase):
    def test_exact15_npz_identity_and_v14_identity_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _manifest(); prepared = root / "prepared.npz"
            _npz(prepared, manifest)
            prereg = _preregister(root, prepared, manifest)
            validate_preregister(prereg)
            row, receipt = validate_prepared_npz(
                prepared, pair_id="scene0000_00_source_to_reference",
                preregister=prereg)
            self.assertEqual(row["prepared_npz_sha256"], sha256_file(prepared))
            self.assertEqual(
                receipt["sides"]["source"]["raw_point_count"], 40)
            prereg_path = root / "preregister.json"
            prereg_path.write_text(json.dumps(prereg, sort_keys=True) + "\n")
            self.assertEqual(validate_pair(
                prereg_path, "scene0000_00_source_to_reference")["scene_id"],
                "scene0000_00")
            self.assertFalse(validate_pair(
                prereg_path,
                "scene0000_00_source_to_reference")["execution_authorized"])
            surfaces = load_verified_surfaces(
                prepared, pair_id="scene0000_00_source_to_reference",
                arm="sgf_selected_union", preregister=prereg)
            self.assertEqual(len(surfaces["source"]), 40)

    def test_scannet15_converter_is_create_only_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _manifest(); prepared = root / "prepared.npz"
            _npz(prepared, manifest)
            prereg = _preregister(root, prepared, manifest)
            prereg_path = root / "preregister.json"
            prereg_path.write_text(json.dumps(prereg, sort_keys=True) + "\n")
            sentinel_a = root / "identity.json"
            sentinel_b = root / "proper.json"
            sentinel_a.write_text("{}\n"); sentinel_b.write_text("{}\n")
            meta = {
                "schema": "v13-colorpcr-corr-cache-v2",
                "sentinel_invariant": True, "gt_consumed": False,
                "identity_fallback": False,
                "input_sha256": sha256_file(prepared),
                "worker_contract": {
                    "arm": "sgf_selected_union", "direction": "forward",
                    "neighbor_limits": [38, 36, 36, 38],
                    "sampling": "voxel10", "coarsest_cap": 512},
                "sentinel_artifact_path": {
                    "identity": str(sentinel_a),
                    "proper_nonzero": str(sentinel_b)},
                "sentinel_artifact_sha256": {
                    "identity": sha256_file(sentinel_a),
                    "proper_nonzero": sha256_file(sentinel_b)}}
            source = root / "sentinel.npz"
            points = np.arange(120, dtype=np.float32).reshape(40, 3)
            np.savez(
                source, src_corr_points=points, ref_corr_points=points,
                corr_scores=np.ones(40, np.float32),
                estimated_transform=np.eye(4, dtype=np.float32),
                meta_json=np.asarray(json.dumps(meta, sort_keys=True)))
            output, receipt = root / "out.npz", root / "receipt.json"
            result = convert(
                source, prepared, output, receipt,
                pair_id="scene0000_00_source_to_reference",
                arm="sgf_selected_union", direction="forward",
                identity_preregister_path=prereg_path)
            self.assertTrue(result["schema"].startswith(
                "v16-b716-scannet15"))
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(set(data.files),
                                 {"src_corr", "ref_corr", "scores"})
            before = output.read_bytes()
            with self.assertRaisesRegex(Exception, "create-only"):
                convert(
                    source, prepared, output, receipt,
                    pair_id="scene0000_00_source_to_reference",
                    arm="sgf_selected_union", direction="forward",
                    identity_preregister_path=prereg_path)
            self.assertEqual(output.read_bytes(), before)

    def test_manifest_or_npz_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = _manifest(); prepared = root / "prepared.npz"
            _npz(prepared, manifest)
            prereg = _preregister(root, prepared, manifest)
            bad = dict(manifest); bad["gt_consumed"] = True
            from safety.v16_b716_scannet15_identity import \
                validate_prepared_manifest
            with self.assertRaises(ScanNet15IdentityError):
                validate_prepared_manifest(
                    bad, pair_id="scene0000_00_source_to_reference",
                    prepared_sha256=sha256_file(prepared),
                    preregister=prereg)
            with np.load(prepared, allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            arrays["unexpected_transform"] = np.eye(4)
            tampered = root / "tampered.npz"
            np.savez(tampered, **arrays)
            row = prereg["pairs"][0]
            row["prepared_npz_path"] = str(tampered.resolve())
            row["prepared_npz_sha256"] = sha256_file(tampered)
            row["identity_payload_sha256"] = stable_json_sha256(
                {k: v for k, v in row.items()
                 if k != "identity_payload_sha256"})
            prereg["payload_sha256"] = stable_json_sha256(
                {k: v for k, v in prereg.items()
                 if k != "payload_sha256"})
            with self.assertRaisesRegex(ScanNet15IdentityError, "exact36"):
                validate_prepared_npz(
                    tampered, pair_id="scene0000_00_source_to_reference",
                    preregister=prereg)


if __name__ == "__main__":
    unittest.main()
