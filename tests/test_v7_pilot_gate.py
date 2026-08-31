import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v7_pilot_gate as gate  # noqa: E402
import v7_registration_batch as batch  # noqa: E402


POLICIES = [f"policy_{index}" for index in range(8)]
PAIR_IDS = [gate.KNOWN_PAIR] + [
    (f"00000000-0000-0000-0000-{index:012x}_to_"
     f"10000000-0000-0000-0000-{index:012x}")
    for index in range(1, 12)
]


def baseline():
    return {
        "majority_existing_pair_ids": PAIR_IDS[1:8],
        "majority_existing_count": 7,
        "median_counts": {
            "raw_strict": 8.0, "accepted_correct": 8.0,
            "accepted_error": 0.0,
        },
    }


def posthoc_index(*, introduce_error=False):
    output = {}
    for pair_index, pair_id in enumerate(PAIR_IDS):
        runs = []
        for outer in range(2):
            policies = {}
            for name in POLICIES:
                strict = 1 <= pair_index <= 7
                usable = strict
                error = bool(introduce_error and pair_index == 8
                             and name == POLICIES[0] and outer == 0)
                policies[name] = {
                    "selected": strict or error,
                    "usable_for_reconstruction": usable or error,
                    "official_raw": {"strict": strict} if strict or error
                    else None,
                    "accepted_strict_error": error,
                }
            runs.append({"pair_id": pair_id, "outer_repeat": outer,
                         "policies": policies})
        output[pair_id] = {"pair_id": pair_id, "runs": runs}
    return output


def batch_receipt(mode=batch.FORMAL_EVIDENCE_MODE):
    summaries = {}
    for name in POLICIES:
        rows = []
        for pair_id in PAIR_IDS:
            veto = pair_id == gate.KNOWN_PAIR
            rows.append({
                "pair_id": pair_id,
                "outer_usable": [not veto, not veto],
                "repeatable": True,
                "outcome": "veto" if veto else "usable",
            })
        summaries[name] = {
            "usable_pairs": 11, "vetoed_pairs": 1, "mixed_pairs": 0,
            "all_pair_outcomes_repeatable": True, "pairs": rows,
        }
    return {
        "evidence_mode": mode,
        "pair_count": 12,
        "outer_repeats_per_pair": 2,
        "replicates_per_outer": {"forward": 5, "reverse": 5},
        "pair_receipts": [{"pair_id": value} for value in PAIR_IDS],
        "global_fail_closed_counts": {
            "exceptions": 0, "nonfinite_transforms": 0,
            "cache_mismatches": 0,
        },
        "policy_pair_summary": summaries,
    }


class PolicyGateTests(unittest.TestCase):
    def test_conservative_majority_and_median_gate_passes(self):
        result = gate.evaluate_policy(
            POLICIES[0], pair_ids=PAIR_IDS,
            posthoc=posthoc_index(), batch_receipt=batch_receipt(),
            baseline=baseline())
        self.assertTrue(result["pass"])
        self.assertEqual(7, result["outer_metrics"][0]["raw_strict"])
        self.assertEqual(
            7, result["outer_metrics"][0][
                "majority_existing_accepted_correct_retained"])

    def test_accepted_error_fails_policy(self):
        result = gate.evaluate_policy(
            POLICIES[0], pair_ids=PAIR_IDS,
            posthoc=posthoc_index(introduce_error=True),
            batch_receipt=batch_receipt(), baseline=baseline())
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["zero_accepted_error"])


class FormalStatusTests(unittest.TestCase):
    def _manifest(self, mode):
        return {
            "pairs": [{"pair_id": value} for value in PAIR_IDS],
            "_evidence_mode": mode,
            "_formal_preregistered": mode == batch.FORMAL_EVIDENCE_MODE,
        }

    def test_all_eight_must_pass_before_selection_authorized(self):
        receipt = batch_receipt()
        posthoc_batch = {"evidence_mode": batch.FORMAL_EVIDENCE_MODE}
        with mock.patch.object(gate, "_index_posthoc",
                               return_value=posthoc_index()), \
                mock.patch.object(gate, "load_v6_controls",
                                  return_value=baseline()):
            result = gate.evaluate(
                self._manifest(batch.FORMAL_EVIDENCE_MODE),
                batch_receipt=receipt, posthoc_batch=posthoc_batch)
        self.assertEqual("PASS", result["status"])
        self.assertTrue(result["selection89_authorized"])
        self.assertTrue(result["all_8_policies_pass"])

    def test_non_preregistered_never_yields_formal_pass(self):
        receipt = batch_receipt(batch.RESEARCH_EVIDENCE_MODE)
        posthoc_batch = {"evidence_mode": batch.RESEARCH_EVIDENCE_MODE}
        with mock.patch.object(gate, "_index_posthoc",
                               return_value=posthoc_index()), \
                mock.patch.object(gate, "load_v6_controls",
                                  return_value=baseline()):
            result = gate.evaluate(
                self._manifest(batch.RESEARCH_EVIDENCE_MODE),
                batch_receipt=receipt, posthoc_batch=posthoc_batch)
        self.assertEqual("INDETERMINATE", result["status"])
        self.assertFalse(result["selection89_authorized"])


if __name__ == "__main__":
    unittest.main()
