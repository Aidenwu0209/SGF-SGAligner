import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
MODULE_PATH = ROOT / "scripts/v8_selection89_gtfree_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("v8_gtfree_diag", MODULE_PATH)
diag = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diag)


class DiagnosticsTests(unittest.TestCase):
    def test_loo_uses_eight_of_nine_quorum(self):
        records = [{"status": "ok", "transform": np.eye(4),
                    "stable_signature": str(index)} for index in range(10)]
        result = diag._loo(records)
        self.assertTrue(result["all_usable"])
        self.assertEqual(result["largest_clique_min"], 9)
        self.assertEqual(len(result["runs"]), 10)

    def test_source_has_no_gt_loader(self):
        source = MODULE_PATH.read_text()
        self.assertNotIn("load_gt_transform", source)
        self.assertNotIn("adapters.sgf.data_sources", source)


if __name__ == "__main__":
    unittest.main()
