import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import v13_fixed4_matrix_driver as matrix
from v13_formal_source_manifest import FORMAL_SOURCE_PATHS


NORMALS = ["n0_to_r0", "n1_to_r1", "n2_to_r2"]
KNOWN = "bad_to_ref"
AUTHORITY = "fixed_trace_icp_plus_unchanged_rule_b_plus_dual_solver_q4"
PINS = {"v7_registration_pilot.py": "a" * 64,
        "decision_features.py": "b" * 64,
        "v8_stage_order_consensus.py": "c" * 64}


def medoid_evidence():
    row = {
        "usable": True,
        "rule_b_features": {
            "overlap_10cm": .8, "median_residual_m": .01,
            "symmetric_trimmed_chamfer_m": .01, "icp_converged": True,
            "icp_update_translation_m": 0., "icp_update_rotation_deg": 0.,
            "icp_fitness": .9, "ransac_inliers": 40,
            "spatial_extent_m": 2., "bidirectional_available": True,
            "bidirectional_rotation_deg": 0.,
            "bidirectional_translation_m": 0.,
        },
        "recorded_rule_b_decision": {
            "usable_for_reconstruction": True, "rejection_reasons": [],
            "rule": "fix2-B-unchanged", "thresholds": {}},
        "icp": {"transform": np.eye(4).tolist(), "converged": True,
                "fitness": 1., "rmse_m": 0., "update_rotation_deg": 0.,
                "update_translation_m": 0.,
                "trace": [{"fixed_correspondence_rmse_before_m": .1,
                           "fixed_correspondence_rmse_after_m": .05}]},
        "surface_source_point_count": 60,
        "surface_reference_point_count": 61,
        "surface_source_sha256": "d" * 64,
        "surface_reference_sha256": "e" * 64,
    }
    return {f"{solver}/{direction}": dict(row)
            for solver in ("pointdsc", "pygcransac")
            for direction in ("forward", "reverse")}


class Tests(unittest.TestCase):
    def setup_contract(self, root: Path):
        repo = root / "repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "src/safety").mkdir(parents=True)
        for name, relative in FORMAL_SOURCE_PATHS.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# sealed formal source: {name}\n")
        prepared = []
        for pair_id in NORMALS + [KNOWN]:
            path = root / f"{pair_id}.npz"
            np.savez(path, pair=np.asarray(pair_id))
            prepared.append({"pair_id": pair_id,
                             "prepared_npz_path": str(path.resolve()),
                             "prepared_npz_sha256": matrix.sha256_file(path)})
        spec = {"normal_pair_ids": NORMALS, "known_bad_pair_id": KNOWN,
                "primary_arm": "sgf_selected_union", "control_arm": "fullscan",
                "control_can_rescue": False, "strict_gate_runtime_pins": PINS}
        preflight = {"pair_ids": NORMALS + [KNOWN], "pairs": prepared}
        spec_path, flight_path = root / "spec.json", root / "flight.json"
        spec_path.write_text(json.dumps(spec)); flight_path.write_text(json.dumps(preflight))
        return repo, spec_path, flight_path

    @staticmethod
    def fake_pair_run(command, **_kwargs):
        pair_id = command[command.index("--pair-id") + 1]
        arm = command[command.index("--arm") + 1]
        output = Path(command[command.index("--output") + 1])
        known = pair_id == KNOWN
        repo = Path(command[command.index("--repo") + 1])
        row = {"schema": "v13-strict-pair-gate-v1", "pair_id": pair_id,
               "arm": arm, "safe": not known,
               "reason": "known_bad_veto" if known else "strict_pair_gate_pass",
               "known_bad_veto": known, "bound_known_bad_pair_id": KNOWN,
               "gate_authority": AUTHORITY,
               "medoid_safety": medoid_evidence(),
               "formal_source_sha256": matrix._formal_source_sha256(repo),
               "runtime_receipt": {"mode": "SEALED_FORMAL_RUNTIME",
                                    "source_sha256": PINS},
               "rule_b_evaluator": "evaluate_rule_b", "rule_c_claimed": False}
        output.joinpath("dual_solver").mkdir(parents=True, exist_ok=True)
        output.joinpath("dual_solver/summary.json").write_text(json.dumps(row))
        return mock.Mock(returncode=2 if known else 0)

    def test_exact_eight_and_hash_verified_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo, spec, flight = self.setup_contract(root)
            out = root / "out"
            with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                            side_effect=self.fake_pair_run) as called:
                result = matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")
            self.assertTrue(result["safe"])
            self.assertEqual(called.call_count, 8)
            self.assertEqual(len(list((out / "pair_receipts").glob("*.json"))), 8)
            self.assertTrue((out / "artifact_manifest.json").is_file())
            self.assertTrue((out / "closure.json").is_file())
            with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                            side_effect=AssertionError("resume must not rerun")):
                resumed = matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")
            self.assertTrue(resumed["safe"])
            commands = json.loads((out / "commands.json").read_text())["commands"]
            self.assertTrue(all(row["status"] == "resumed_hash_verified"
                                for row in commands))

    def test_tampered_pair_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo, spec, flight = self.setup_contract(root)
            out = root / "out"
            with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                            side_effect=self.fake_pair_run):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")
            target = out / "pairs" / NORMALS[0] / "sgf_selected_union" \
                / "dual_solver/summary.json"
            target.write_text("{}")
            with self.assertRaisesRegex(matrix.Fixed4DriverError, "summary hash"):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")

    def test_formal_source_change_invalidates_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo, spec, flight = self.setup_contract(root)
            out = root / "out"
            with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                            side_effect=self.fake_pair_run):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")
            (repo / "src/safety/v13_strict_pair_gate.py").write_text(
                "# changed strict gate\n")
            with self.assertRaisesRegex(matrix.Fixed4DriverError,
                                        "stale pair receipt dependency"):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")

    def test_each_formal_source_change_invalidates_resume(self):
        required = {
            "sentinel_subprocess", "official_worker", "fixed4_aggregate"}
        self.assertTrue(required.issubset(FORMAL_SOURCE_PATHS))
        for source_name, relative in FORMAL_SOURCE_PATHS.items():
            with self.subTest(source=source_name), \
                    tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo, spec, flight = self.setup_contract(root)
                out = root / "out"
                with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                                side_effect=self.fake_pair_run):
                    matrix.run_fixed4(repo=repo, preregister_path=spec,
                        preflight_path=flight, output_root=out,
                        python=Path("/usr/bin/python3"), device="cuda:0")
                path = repo / relative
                path.write_text(path.read_text() + "# tampered\n")
                with self.assertRaisesRegex(
                        matrix.Fixed4DriverError,
                        "stale pair receipt dependency"):
                    matrix.run_fixed4(repo=repo, preregister_path=spec,
                        preflight_path=flight, output_root=out,
                        python=Path("/usr/bin/python3"), device="cuda:0")

    def test_summary_formal_source_map_is_revalidated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); repo, spec, flight = self.setup_contract(root)
            out = root / "out"
            with mock.patch("v13_fixed4_matrix_driver.subprocess.run",
                            side_effect=self.fake_pair_run):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")
            pair_id, arm = NORMALS[0], "sgf_selected_union"
            summary_path = out / "pairs" / pair_id / arm \
                / "dual_solver/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["formal_source_sha256"]["cli"] = "f" * 64
            summary_path.write_text(json.dumps(summary))
            receipt_path = out / "pair_receipts" / f"{pair_id}.{arm}.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["summary_sha256"] = matrix.sha256_file(summary_path)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(matrix.Fixed4DriverError,
                                        "formal source mismatch"):
                matrix.run_fixed4(repo=repo, preregister_path=spec,
                    preflight_path=flight, output_root=out,
                    python=Path("/usr/bin/python3"), device="cuda:0")


if __name__ == "__main__":
    unittest.main()
