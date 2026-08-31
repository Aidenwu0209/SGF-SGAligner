from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v8_calibration90_locked as locked
import v8_calibration90_posthoc_gate as posthoc


class Calibration90PosthocContractTests(unittest.TestCase):
    def test_label_opening_claim_is_durable_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "batch.json"
            batch_path.write_text("{}")
            manifest = {
                "_file_sha256": "a" * 64,
                "_path": "/frozen/calibration-manifest.json",
                "single_use": {"claim_root": str(root)},
            }
            batch = {"evidence_sha256": "b" * 64}
            claim = posthoc.claim_posthoc(manifest, batch_path, batch)
            self.assertTrue(claim.is_file())
            with self.assertRaises(locked.Calibration90Error):
                posthoc.claim_posthoc(manifest, batch_path, batch)

    def test_posthoc_has_no_threshold_cli_and_never_authorizes_official92(self):
        source = Path(posthoc.__file__).read_text()
        self.assertNotIn("--strict-min", source)
        self.assertNotIn("--accepted-correct-min", source)
        self.assertIn('"official92_authorized": False', source)
        self.assertIn('"rerun_authorized": False', source)


if __name__ == "__main__":
    unittest.main()
