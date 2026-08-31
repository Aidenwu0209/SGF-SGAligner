import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v13_corr_cache_converter import ConversionContractError, convert, sha256_file


class Tests(unittest.TestCase):
    def make_source(self, root: Path, *, direction="forward", extra=False):
        prepared = root / "prepared.npz"
        prepared_manifest = {"schema": "v13-color-preserving-pair-v2",
                             "pair_id": "src_to_ref",
                             "payload_sha256": "a" * 64}
        np.savez(prepared, input=np.arange(3),
                 manifest_json=np.asarray(json.dumps(prepared_manifest)))
        points = np.arange(180, dtype=np.float32).reshape(60, 3)
        arrays = {
            "src_corr_points": points, "ref_corr_points": points + 1,
            "corr_scores": np.linspace(0, 1, 60, dtype=np.float32),
            "estimated_transform": np.eye(4),
        }
        sentinel_paths = {}
        sentinel_hashes = {}
        for name in ("identity", "proper_nonzero"):
            path = root / f"{name}.npz"
            np.savez(path, marker=np.asarray([name]))
            sentinel_paths[name] = str(path)
            sentinel_hashes[name] = sha256_file(path)
        meta = {"schema": "v13-colorpcr-corr-cache-v2",
                "sentinel_invariant": True, "gt_consumed": False,
                "identity_fallback": False, "input_sha256": sha256_file(prepared),
                "sentinel_artifact_path": sentinel_paths,
                "sentinel_artifact_sha256": sentinel_hashes,
                "worker_contract": {"arm": "sgf_selected_union", "direction": direction,
                    "neighbor_limits": [38,36,36,38], "sampling": "voxel10",
                    "coarsest_cap": 512}}
        arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
        if extra:
            arrays["gt_transform"] = np.eye(4)
        source = root / "sentinel.npz"
        np.savez(source, **arrays)
        return prepared, source

    def test_exact_three_output_and_bound_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, source = self.make_source(root)
            output, receipt_path = root / "corr.npz", root / "receipt.json"
            receipt = convert(source, prepared, output, receipt_path,
                              pair_id="src_to_ref", arm="sgf_selected_union",
                              direction="forward")
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"src_corr", "ref_corr", "scores"})
                self.assertNotIn("estimated_transform", data.files)
            self.assertEqual(receipt["source_sentinel_cache_sha256"], sha256_file(source))
            self.assertEqual(receipt["prepared_input_sha256"], sha256_file(prepared))
            self.assertTrue(receipt["estimated_transform_discarded"])
            self.assertEqual(receipt["neighbor_limits"], [38,36,36,38])
            self.assertEqual(json.loads(receipt_path.read_text()), receipt)

    def test_wrong_direction_or_extra_gt_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, source = self.make_source(root, direction="reverse")
            with self.assertRaisesRegex(ConversionContractError, "metadata mismatch"):
                convert(source, prepared, root / "o.npz", root / "r.json",
                        pair_id="src_to_ref", arm="sgf_selected_union", direction="forward")
            prepared, source = self.make_source(root, extra=True)
            with self.assertRaisesRegex(ConversionContractError, "schema"):
                convert(source, prepared, root / "o.npz", root / "r.json",
                        pair_id="src_to_ref", arm="sgf_selected_union", direction="forward")


if __name__ == "__main__":
    unittest.main()
