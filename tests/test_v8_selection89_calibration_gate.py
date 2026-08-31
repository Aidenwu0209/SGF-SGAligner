import copy
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
MODULE_PATH = ROOT / "scripts/v8_selection89_calibration_gate.py"
SPEC_LOADER = importlib.util.spec_from_file_location("v8_gate", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC_LOADER)
assert SPEC_LOADER.loader is not None
SPEC_LOADER.loader.exec_module(gate)


KNOWN_BAD = SPEC["known_bad_pair_id"]


def replay_fixture():
    pairs = [{"pair_id": f"pair-{index}", "repeatable": True,
              "outer_outcomes": [False, False]}
             for index in range(88)]
    pairs.append({"pair_id": KNOWN_BAD, "repeatable": True,
                  "outer_outcomes": [False, False]})
    return {"pairs": pairs}


def posthoc_fixture():
    return {"per_outer": [
        {"outer_repeat": 0, "accepted_correct": 10,
         "accepted_error": 0, "raw_strict": 12},
        {"outer_repeat": 1, "accepted_correct": 9,
         "accepted_error": 0, "raw_strict": 11},
    ]}


AUDIT = {"worker_count": 1780, "worker_exceptions": 0,
         "worker_nonfinite": 0, "worker_cache_or_hash_mismatch": 0,
         "pair_count": 89, "outer_repeats": 2}


class GateTests(unittest.TestCase):
    def test_exact_preregistered_floor_passes(self):
        result = gate.evaluate_gate(
            replay_fixture(), posthoc_fixture(), AUDIT, SPEC)
        self.assertEqual(result["status"], "CALIBRATION_MECHANICAL_GATE_PASS")
        self.assertTrue(result["may_propose_later_calibration"])
        self.assertFalse(result["automatic_calibration_authorized"])

    def test_no_threshold_may_fail_open(self):
        mutations = []
        for key, value in AUDIT.items():
            audit = dict(AUDIT)
            audit[key] = value + 1
            mutations.append((replay_fixture(), posthoc_fixture(), audit))
        replay = replay_fixture()
        replay[0 if False else "pairs"][0]["repeatable"] = False
        mutations.append((replay, posthoc_fixture(), dict(AUDIT)))
        replay = replay_fixture()
        replay["pairs"][-1]["outer_outcomes"] = [False, True]
        mutations.append((replay, posthoc_fixture(), dict(AUDIT)))
        for key, value in (("accepted_error", 1),
                           ("accepted_correct", 8),
                           ("raw_strict", 10)):
            posthoc = posthoc_fixture()
            posthoc["per_outer"][1][key] = value
            mutations.append((replay_fixture(), posthoc, dict(AUDIT)))
        for replay, posthoc, audit in mutations:
            with self.subTest(audit=audit, posthoc=posthoc):
                result = gate.evaluate_gate(replay, posthoc, audit, SPEC)
                self.assertEqual(
                    result["status"], "CALIBRATION_MECHANICAL_GATE_FAIL")

    def test_gate_source_has_no_gt_loader(self):
        source = MODULE_PATH.read_text()
        self.assertNotIn("load_gt_transform", source)
        self.assertNotIn("adapters.sgf.data_sources", source)


if __name__ == "__main__":
    unittest.main()
