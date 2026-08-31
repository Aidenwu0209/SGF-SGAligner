import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
SPEC = json.loads((ROOT / "docs/V8_SELECTION89_CALIBRATION_GATE.json").read_text())
MODULE_PATH = ROOT / "scripts/v8_selection89_prelabel_gate.py"
SPEC_LOADER = importlib.util.spec_from_file_location("v8_prelabel", MODULE_PATH)
prelabel = importlib.util.module_from_spec(SPEC_LOADER)
assert SPEC_LOADER.loader is not None
SPEC_LOADER.loader.exec_module(prelabel)


KNOWN_BAD = SPEC["known_bad_pair_id"]
AUDIT = {"worker_count": 1780, "worker_exceptions": 0,
         "worker_nonfinite": 0, "worker_cache_or_hash_mismatch": 0,
         "pair_count": 89, "outer_repeats": 2}


def fixture():
    pairs = [{"pair_id": f"pair-{index}", "repeatable": True,
              "outer_outcomes": [False, False]} for index in range(88)]
    pairs.append({"pair_id": KNOWN_BAD, "repeatable": True,
                  "outer_outcomes": [False, False]})
    return {"pairs": pairs}


class PrelabelTests(unittest.TestCase):
    def test_exact_gate_passes(self):
        result = prelabel.evaluate_prelabel(fixture(), AUDIT, SPEC)
        self.assertTrue(result["label_loading_authorized"])

    def test_outer_mismatch_stops_before_labels(self):
        replay = fixture()
        replay["pairs"][0]["repeatable"] = False
        result = prelabel.evaluate_prelabel(replay, AUDIT, SPEC)
        self.assertFalse(result["label_loading_authorized"])
        self.assertIn("all_pair_verdicts_repeatable", result["failed_checks"])

    def test_known_bad_accept_stops_before_labels(self):
        replay = fixture()
        replay["pairs"][-1]["outer_outcomes"] = [False, True]
        result = prelabel.evaluate_prelabel(replay, AUDIT, SPEC)
        self.assertFalse(result["label_loading_authorized"])
        self.assertIn("known_bad_veto_each_outer", result["failed_checks"])

    def test_source_has_no_gt_loader(self):
        source = MODULE_PATH.read_text()
        self.assertNotIn("load_gt_transform", source)
        self.assertNotIn("adapters.sgf.data_sources", source)


if __name__ == "__main__":
    unittest.main()
