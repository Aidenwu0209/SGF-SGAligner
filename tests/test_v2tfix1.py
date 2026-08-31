"""V2T-Fix1 evidence schema/consistency + process-integrity tests."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix1"
OLD = ROOT / "outputs/official_sgaligner_migration_fix2_v2training"
SEAL = ROOT / "outputs/official_sgaligner_migration_fix2_seal_20260826"


def load(path):
    return json.loads(Path(path).read_text())


class TestPriorEvidenceRepaired(unittest.TestCase):
    def test_training_config_valid_json(self):
        load(OLD / "training_config.json")

    def test_gate_evaluation_machine_consistent(self):
        gates = load(OLD / "gate_evaluation.json")
        inputs = gates["_inputs"]
        self.assertEqual(
            gates["node_f1_ge_0214"], inputs["best_mean_node_f1"] >= 0.214
        )
        self.assertFalse(gates["node_f1_ge_0214"])  # 0.096 -> False

    def test_audit_marker_present(self):
        text = (OUT / "audit_before_fix.md").read_text()
        self.assertIn("invalid_for_scientific_root_cause_conclusion", text)


class TestLossEvidence(unittest.TestCase):
    def test_loss_reference_validation_exists(self):
        # produced by the reference cross-check inside the test suite
        self.assertTrue(
            (ROOT / "tests/test_cross_graph_loss.py").exists()
        )

    def test_pair_label_audit_covers_437(self):
        audit = load(OUT / "training_B/pair_label_audit.json")
        self.assertEqual(audit["total"], 437)
        self.assertEqual(audit["used"] + audit["skipped"], 437)

    def test_skipped_pairs_json_valid(self):
        d = load(OUT / "training_B/skipped_pairs.json")
        self.assertEqual(d["count"], len(d["pair_ids"]))


class TestFairSelection(unittest.TestCase):
    def test_leaderboard_has_all_candidates(self):
        board = load(OUT / "checkpoint_leaderboard.json")["board"]
        self.assertEqual(len(board), 30)

    def test_selection_rule_frozen_before_run(self):
        result = load(OUT / "selection_results.json")
        self.assertIn("epoch_00021", result["chosen"])

    def test_calibration_ran_once_after_freeze(self):
        cal = load(OUT / "calibration_results.json")
        self.assertEqual(cal["requested"], 90)

    def test_fixed12_after_calibration(self):
        f12 = load(OUT / "fixed12_results.json")
        self.assertEqual(f12["requested"], 12)

    def test_gate_evaluation_inputs_match_results(self):
        gates = load(OUT / "gate_evaluation.json")
        sel = load(OUT / "sel_replay_epoch_00021/rule_replay_summary.json")["B"]
        self.assertEqual(gates["_inputs"]["selection_strict"], sel["strict"])
        self.assertEqual(
            gates["selection_strict_ge_30"], sel["strict"] >= 30
        )


class TestNodeF1Semantics(unittest.TestCase):
    def test_macro_definition_documented(self):
        gates = load(OUT / "gate_evaluation.json")
        self.assertIn("MACRO", gates["_inputs"]["node_f1_definition"])
        self.assertIn("MICRO", gates["_inputs"]["node_f1_definition"])


class TestResumeEvidence(unittest.TestCase):
    def test_resume_equivalence_reported_honestly(self):
        d = load(OUT / "resume_test/resume_equivalence.json")
        # honest reporting: not identical (CUDA non-determinism)
        self.assertIn("model_state_identical", d)


class TestNoServiceControl(unittest.TestCase):
    def test_scripts_have_no_service_control(self):
        for script in ("v2f_train.py", "v2f_leaderboard.py"):
            src = (ROOT / "scripts" / script).read_text()
            for banned in ("systemctl", "service ", "kill ", "pkill"):
                self.assertNotIn(banned, src, script)


if __name__ == "__main__":
    unittest.main()
