import json
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import v13_colorpcr_sentinel_subprocess as sentinel
import v13_dual_solver_cli as solver_cli


class Tests(unittest.TestCase):
    def test_cli_passes_two_receipts_and_reaches_run_matrix(self):
        argv = ["v13_dual_solver_cli.py",
                "--forward-cache", "f.npz", "--reverse-cache", "r.npz",
                "--forward-receipt", "f.receipt.json",
                "--reverse-receipt", "r.receipt.json",
                "--output-dir", "out", "--pointdsc-root", "pd",
                "--pointdsc-checkpoint", "pd.pkl", "--prepared-input", "p.npz",
                "--arm", "sgf_selected_union", "--pair-id", "src_to_ref",
                "--preregister", "pre.json", "--preflight-manifest", "flight.json"]
        argv += ["--driver-source", str(Path(solver_cli.__file__).resolve().parent /
                                         "v13_fixed4_driver.py")]
        prereg={"normal_pair_ids":["src_to_ref","n1_to_r1","n2_to_r2"],
                "known_bad_pair_id":"bad_to_ref"}
        flight={"pair_ids":["src_to_ref","n1_to_r1","n2_to_r2","bad_to_ref"]}
        receipt={"source_sentinel_cache_sha256":"a"*64}
        strict={"schema":"v13-strict-pair-gate-v1","safe":False,
                "reason":"synthetic"}
        with mock.patch.object(sys, "argv", argv), \
             mock.patch("v13_dual_solver_cli._load_json",
                        side_effect=[prereg,flight]), \
             mock.patch("v13_dual_solver_cli._verify_safety_authority",
                        return_value={}), \
             mock.patch("v13_dual_solver_cli._verify_preflight_closure",
                        return_value={}), \
             mock.patch("v13_dual_solver_cli._verify_conversion_receipt",
                        side_effect=[receipt,receipt]), \
             mock.patch("v13_dual_solver_cli.run_matrix",
                        return_value={"schema":"v13-dual-solver-summary-v1",
                                      "safe":False}) as called, \
             mock.patch("v13_dual_solver_cli.strict_pair_gate",
                        return_value=strict) as gated, \
             mock.patch("v13_dual_solver_cli.sha256_file",
                        return_value="b"*64), \
             mock.patch("v13_dual_solver_cli.atomic_json"):
            self.assertEqual(solver_cli.main(), 2)
        args = called.call_args.args
        self.assertEqual(str(args[0]), "f.npz")
        self.assertEqual(str(args[1]), "r.npz")
        self.assertEqual(called.call_args.kwargs["device"], "cpu")
        self.assertEqual(gated.call_args.kwargs["pair_id"], "src_to_ref")
        self.assertEqual(gated.call_args.kwargs["arm"], "sgf_selected_union")

    def _json_sha(self, value):
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True).encode()).hexdigest()

    def _closure_fixture(self, root: Path):
        pair_id = "src_to_ref"
        semantic = {"schema": "v13-color-preserving-pair-v2",
                    "pair_id": pair_id, "unit_contract": "metres"}
        pair_payload = self._json_sha(semantic)
        embedded = dict(semantic, payload_sha256=pair_payload)
        prepared = root / "prepared" / f"{pair_id}.npz"
        prepared.parent.mkdir()
        np.savez(prepared, manifest_json=np.asarray(json.dumps(
            embedded, sort_keys=True)))
        prereg = root / "preregister.json"
        prereg.write_text(json.dumps({"schema": "preregister"}))
        row = dict(embedded, prepared_npz_path=str(prepared.resolve()),
                   prepared_npz_sha256=solver_cli.sha256_file(prepared))
        other_ids = ["n1_to_r1", "n2_to_r2", "bad_to_ref"]
        other_rows = []
        for value in other_ids:
            other_semantic = {"schema": "v13-color-preserving-pair-v2",
                              "pair_id": value, "unit_contract": "metres"}
            other_rows.append(dict(
                other_semantic,
                payload_sha256=self._json_sha(other_semantic),
                prepared_npz_path=str(root / "prepared" / f"{value}.npz"),
                prepared_npz_sha256="f" * 64))
        preflight = {"schema": "preflight", "pair_ids": [pair_id, *other_ids],
                     "pairs": [row, *other_rows],
                     "preregister_sha256": solver_cli.sha256_file(prereg)}
        preflight["payload_sha256"] = self._json_sha(preflight)
        flight = root / "preflight_manifest.json"
        flight.write_text(json.dumps(preflight, sort_keys=True))
        artifact = {"schema": "artifact",
                    "payload_sha256": preflight["payload_sha256"],
                    "preregister_sha256": solver_cli.sha256_file(prereg),
                    "files": [
                        {"path": flight.name, "bytes": flight.stat().st_size,
                         "sha256": solver_cli.sha256_file(flight)},
                        {"path": str(prepared.relative_to(root)),
                         "bytes": prepared.stat().st_size,
                         "sha256": solver_cli.sha256_file(prepared)}]}
        (root / "artifact_manifest.json").write_text(json.dumps(artifact))
        return pair_id, prepared, prereg, flight, preflight

    def test_preflight_closure_rejects_same_manifest_npz_at_arbitrary_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair, prepared, prereg, flight, preflight = self._closure_fixture(root)
            copied = root / "arbitrary_same_manifest.npz"
            shutil.copyfile(prepared, copied)
            with self.assertRaisesRegex(
                    solver_cli.StrictCliContractError, "path differs"):
                solver_cli._verify_preflight_closure(
                    preregister_path=prereg, preflight_path=flight,
                    prepared_path=copied, pair_id=pair,
                    preregister=json.loads(prereg.read_text()),
                    preflight=preflight)

    def test_preflight_closure_rejects_wrong_hash_and_modified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair, prepared, prereg, flight, preflight = self._closure_fixture(root)
            closure = solver_cli._verify_preflight_closure(
                preregister_path=prereg, preflight_path=flight,
                prepared_path=prepared, pair_id=pair,
                preregister=json.loads(prereg.read_text()), preflight=preflight)
            self.assertEqual(closure["prepared_input_sha256"],
                             solver_cli.sha256_file(prepared))
            wrong = json.loads(json.dumps(preflight))
            wrong["pairs"][0]["prepared_npz_sha256"] = "0" * 64
            wrong["payload_sha256"] = self._json_sha(
                {k: v for k, v in wrong.items() if k != "payload_sha256"})
            flight.write_text(json.dumps(wrong, sort_keys=True))
            with self.assertRaisesRegex(
                    solver_cli.StrictCliContractError, "SHA differs"):
                solver_cli._verify_preflight_closure(
                    preregister_path=prereg, preflight_path=flight,
                    prepared_path=prepared, pair_id=pair,
                    preregister=json.loads(prereg.read_text()), preflight=wrong)
            flight.write_text(json.dumps(preflight, sort_keys=True))
            prereg.write_text("{\"schema\":\"modified\"}")
            with self.assertRaisesRegex(
                    solver_cli.StrictCliContractError, "preregister SHA"):
                solver_cli._verify_preflight_closure(
                    preregister_path=prereg, preflight_path=flight,
                    prepared_path=prepared, pair_id=pair,
                    preregister=json.loads(prereg.read_text()), preflight=preflight)

    def test_preflight_payload_modification_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair, prepared, prereg, flight, preflight = self._closure_fixture(root)
            preflight["dependency_status"] = "tampered"
            flight.write_text(json.dumps(preflight, sort_keys=True))
            with self.assertRaisesRegex(
                    solver_cli.StrictCliContractError, "payload SHA"):
                solver_cli._verify_preflight_closure(
                    preregister_path=prereg, preflight_path=flight,
                    prepared_path=prepared, pair_id=pair,
                    preregister=json.loads(prereg.read_text()),
                    preflight=preflight)

    def test_forward_reverse_persist_four_sentinels_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inp = root / "prepared.npz"
            np.savez(inp, input=np.arange(3))
            evidence = root / "sentinels"
            worker = root / "worker.py"
            worker.write_text("# sealed test worker\n")

            def fake_worker(command, **_kwargs):
                self.assertEqual(
                    command[1:6],
                    ["-I", "-S", "-B", "-X",
                     "pycache_prefix=/proc/v16-b716-fixed4-no-pyc"],
                )
                target = Path(command[command.index("--output") + 1])
                direction = command[command.index("--direction") + 1]
                arm = command[command.index("--arm") + 1]
                points = np.arange(120, dtype=np.float32).reshape(40, 3)
                meta = {"input_sha256": sentinel.fhash(inp), "repo_commit": "c",
                        "weight_sha256": "w", "neighbor_limits": [38,36,36,38],
                        "sampling": "voxel10", "python_tree_sha256": "p",
                        "tracked_diff_sha256": "d", "extension_sha256": "e",
                        "input_voxel_m": 0.1, "coarsest_cap": 512,
                        "coarsest_cap_applied": True, "stage_lengths": [[20,20]],
                        "direction": direction, "arm": arm}
                np.savez(target, ref_corr_points=points + 1,
                         src_corr_points=points, corr_scores=np.ones(40,np.float32),
                         estimated_transform=np.eye(4),
                         meta_json=np.asarray(json.dumps(meta,sort_keys=True)))
                return mock.Mock(returncode=0)

            base = ["sentinel.py", "--python", "/py", "--worker", str(worker),
                    "--repo", "/repo", "--expected-commit", "c",
                    "--weights", "/weights", "--expected-weight-sha256", "w",
                    "--input", str(inp), "--expected-python-tree-sha256", "p",
                    "--expected-tracked-diff-sha256", "d", "--extension", "/ext",
                    "--expected-extension-sha256", "e", "--arm", "fullscan",
                    "--neighbor-limits", "38,36,36,38", "--evidence-dir", str(evidence)]
            with mock.patch("v13_colorpcr_sentinel_subprocess.subprocess.run",
                            side_effect=fake_worker):
                with mock.patch.object(sys,"argv",base+["--direction","forward",
                                      "--output",str(root/"forward.npz")]):
                    sentinel.main()
                before = {path.name: sentinel.fhash(path) for path in evidence.glob("*.npz")}
                with mock.patch.object(sys,"argv",base+["--direction","reverse",
                                      "--output",str(root/"reverse.npz")]):
                    sentinel.main()
            after = {path.name: sentinel.fhash(path) for path in evidence.glob("*.npz")}
            self.assertEqual(len(after), 4)
            self.assertTrue(all(after[name] == digest for name,digest in before.items()))
            for cache in (root/"forward.npz", root/"reverse.npz"):
                with np.load(cache,allow_pickle=False) as data:
                    meta=json.loads(str(data["meta_json"].item()))
                for name,path in meta["sentinel_artifact_path"].items():
                    self.assertEqual(sentinel.fhash(Path(path)),
                                     meta["sentinel_artifact_sha256"][name])


if __name__ == "__main__":
    unittest.main()
