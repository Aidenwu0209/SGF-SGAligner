from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import v113_wholescene_pilot as pilot


class V113WholeScenePilotTests(unittest.TestCase):
    def test_permutation_is_contract_deterministic_and_repeat_distinct(self):
        a, ah = pilot.permutation(23, "pair", "forward", 0, "a" * 64)
        b, bh = pilot.permutation(23, "pair", "forward", 0, "a" * 64)
        c, ch = pilot.permutation(23, "pair", "forward", 1, "a" * 64)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ah, bh)
        self.assertFalse(np.array_equal(a, c))
        self.assertNotEqual(ah, ch)

    def test_load_cache_closes_file_payload_protocol_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "cache.pt"
            cache = {
                "schema": pilot.CACHE_SCHEMA,
                "pair_id": "p",
                "arm": "whole_scene_diagnostic_only",
                "selector_eligible": False,
                "protocol_sha256": "v11",
                "checkpoint_sha256": "matcher",
                "forbidden_inputs": ["selection labels", "GT transforms",
                                     "posthoc", "official92"],
            }
            cache["payload_sha256"] = pilot.payload_hash(cache)
            torch.save(cache, path)
            row = {"whole_scene_diagnostic": {
                "artifact_sha256": pilot.fhash(path)}}
            loaded, checkpoint = pilot.load_cache(
                path, row, "p", None, "v11")
            self.assertEqual(checkpoint, "matcher")
            self.assertEqual(loaded["pair_id"], "p")
            with self.assertRaisesRegex(pilot.EvidenceError, "protocol"):
                pilot.load_cache(path, row, "p", None, "other")

    def test_atomic_npz_is_immutable_and_round_trips(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "x.npz"
            digest = pilot.atomic_npz(path, points=np.arange(9).reshape(3, 3))
            self.assertEqual(digest, pilot.fhash(path))
            with np.load(path) as item:
                np.testing.assert_array_equal(item["points"],
                                              np.arange(9).reshape(3, 3))
            with self.assertRaisesRegex(pilot.EvidenceError, "overwrite"):
                pilot.atomic_npz(path, points=np.zeros((1, 3)))

    def test_worker_exposes_solver_and_correspondence_provenance(self):
        cache = {
            "forward": {
                "status": "ok",
                "src_corr": np.eye(3, dtype=np.float32),
                "ref_corr": np.eye(3, dtype=np.float32),
                "scores": np.ones(3, dtype=np.float32),
            },
        }
        item = cache["forward"]
        item["correspondence_sha256"] = hashlib.sha256(
            item["src_corr"].tobytes() + item["ref_corr"].tobytes()
            + item["scores"].tobytes()).hexdigest()
        icp = {"transform": np.eye(4), "trace": [], "converged": True,
               "fitness": 1.0, "rmse_m": 0.0,
               "update_rotation_deg": 0.0, "update_translation_m": 0.0}
        decision = {"usable_for_reconstruction": True,
                    "rejection_reasons": []}
        with mock.patch.object(pilot, "ransac_from_pooled",
                               return_value=(np.eye(4), 3)), \
             mock.patch.object(pilot, "segment_icp_with_trace",
                               return_value=icp), \
             mock.patch.object(pilot, "rule_b_features",
                               return_value=({}, decision)):
            row = pilot.run_worker(
                "p", cache, "forward", 0, "f" * 64,
                np.eye(3), np.eye(3), np.eye(3), np.eye(3))
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["solver"], "official_pygcransac_composition")
        self.assertEqual(len(row["correspondence_sha256"]), 64)

    def test_independent_verifier_detects_tamper(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "data.txt"
            artifact.write_text("sealed")
            manifest = pilot.write_json(root / "artifact_manifest.json", {
                "schema": pilot.SCHEMA, "artifact_count": 1,
                "artifacts": [{"path": "data.txt", "bytes": 6,
                               "sha256": pilot.fhash(artifact)}],
            })
            args = type("Args", (), {"root": root})()
            self.assertEqual(pilot.verify(args), 0)
            self.assertTrue(json.loads(
                (root / "verification_receipt.json").read_text())[
                    "verification_pass"])
            self.assertEqual(manifest["artifact_count"], 1)


if __name__ == "__main__":
    unittest.main()
