import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "v6fix_gate_freezer.py"
SPEC = importlib.util.spec_from_file_location("v6fix_gate_freezer", MODULE_PATH)
freezer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freezer)


def manifest_sha(pair_ids):
    return hashlib.sha256(
        ("\n".join(pair_ids) + "\n").encode()).hexdigest()


class SyntheticEvidence:
    def __init__(self, root):
        self.root = Path(root)
        self.formal = self.root / "formal_v2"
        self.sidecars = self.formal / "node_evidence"
        self.asset = self.root / "assets"
        self.protocol = self.root / "protocol.json"
        self.protocol_md = self.root / "protocol.md"
        self.pairs = [f"src_{index:03d}_to_ref_{index:03d}"
                      for index in range(89)]
        checkpoints = {}
        for checkpoint, byte in zip(("A", "B", "D"), (b"a", b"b", b"d")):
            path = self.asset / f"{checkpoint}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(byte)
            checkpoints[checkpoint] = {
                "role": "diagnostic only" if checkpoint == "D" else "candidate",
                "path": f"{checkpoint}.pt",
                "sha256": hashlib.sha256(byte).hexdigest(),
            }
        self.protocol_doc = {
            "phase": "V6-Fix consistency audit",
            "checkpoints": checkpoints,
            "gates": {
                "flat_recovery": {
                    "median_raw_strict_min": 9,
                    "median_correct_accepted_min": 7,
                    "accepted_error_max": 0,
                    "failed_unknown_max": 0,
                },
                "selection": {
                    "accepted_error_max": 0,
                    "macro_f1_min": 0.0844,
                    "macro_top1_min": 0.0675,
                    "macro_top5_min": 0.433,
                    "raw_strict_drop_vs_recovered_A_max": 1,
                    "zero_candidate_must_not_regress": True,
                    "three_run_direction_stable": True,
                },
            },
        }
        self.protocol.write_text(json.dumps(self.protocol_doc))
        self.protocol_md.write_text("# frozen protocol\n")

    @staticmethod
    def node_evidence(checkpoint):
        # B beats A; D is diagnostic.  All denominators and arithmetic are
        # integers so macro/micro can be independently recomputed.
        if checkpoint == "A":
            return {"tp": 5, "predicted": 10, "anchors": 10,
                    "top1_hits": 1, "top1_total": 10,
                    "top5_hits": 5, "top5_total": 10}
        if checkpoint == "B":
            return {"tp": 6, "predicted": 10, "anchors": 10,
                    "top1_hits": 1, "top1_total": 10,
                    "top5_hits": 6, "top5_total": 10}
        return {"tp": 7, "predicted": 10, "anchors": 10,
                "top1_hits": 2, "top1_total": 10,
                "top5_hits": 7, "top5_total": 10}

    def make_document(self, checkpoint, repeat):
        rows = []
        for index, pair_id in enumerate(self.pairs):
            paths = {}
            for path in freezer.PATHS:
                if checkpoint == "A" and path == "F":
                    strict = index < 10
                    accepted = index < 7
                elif checkpoint == "B" and path == "C1":
                    strict = index < 10
                    accepted = index < 8
                else:
                    strict = index < 9
                    accepted = index < 7
                paths[path] = {
                    "valid": True, "strict": strict, "relaxed": strict,
                    "accepted": accepted,
                    "accepted_correct": accepted and strict,
                    "accepted_error": accepted and not strict,
                    "decision": {
                        "usable_for_reconstruction": accepted,
                        "status": "accepted" if accepted else "rejected",
                    },
                }
            rows.append({
                "pair_id": pair_id, "paths": paths,
                "audit": {"zero_candidate": False},
            })
        counts = {path: freezer._recompute_counts(
            rows, path, f"{checkpoint}/{repeat}") for path in freezer.PATHS}
        return {
            "schema": freezer.RESULT_SCHEMA,
            "code_root": str(ROOT.resolve()),
            "asset_root": str(self.asset),
            "repository": {"head": "a" * 40,
                           "branch": "wu/test",
                           "tracked_dirty": False},
            "split": "selection", "checkpoint": checkpoint,
            "checkpoint_sha256": self.protocol_doc["checkpoints"]
                                                    [checkpoint]["sha256"],
            "repeat": repeat,
            "split_manifest": {
                "name": "selection", "expected": 89, "actual": 89,
                "unique": 89, "sha256": manifest_sha(self.pairs),
            },
            "counts": counts, "rows": rows,
        }

    def make_sidecar(self, checkpoint):
        pair_rows = []
        for index, pair_id in enumerate(self.pairs):
            raw = [[anchor, anchor + 100] for anchor in range(10)]
            mapped = [[anchor, anchor + 10] for anchor in range(10)]
            source = self.root / f"pair_{index:03d}.json"
            source.write_text(json.dumps({"anchor_pairs": raw}))
            pair_rows.append({
                "pair_id": pair_id,
                "cache": {
                    "path": str(self.root / f"{checkpoint}_{index}.pt"),
                    "bytes": 1, "sha256": f"{index + 1:064x}",
                    "cache_key": f"{index + 101:064x}",
                    "input_sha256": f"{index + 201:064x}",
                    "embedding_sha256": f"{index + 301:064x}",
                    "similarity_sha256": f"{index + 401:064x}",
                },
                "anchors": {
                    "source_pair_json": {
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    },
                    "raw_object_ids": raw, "mapped_indices": mapped,
                    "unmapped_object_ids": [],
                },
                "zero_candidate": False,
                "node_evidence": self.node_evidence(checkpoint),
            })
        cache_entries = [{
            "pair_id": row["pair_id"],
            "sha256": row["cache"]["sha256"],
            "cache_key": row["cache"]["cache_key"],
        } for row in pair_rows]
        anchor_entries = [{
            "pair_id": row["pair_id"],
            "source_pair_json": row["anchors"]["source_pair_json"],
            "raw_object_ids": row["anchors"]["raw_object_ids"],
            "mapped_indices": row["anchors"]["mapped_indices"],
            "unmapped_object_ids": row["anchors"]["unmapped_object_ids"],
        } for row in pair_rows]
        return {
            "schema": freezer.NODE_SIDECAR_SCHEMA,
            "checkpoint": checkpoint,
            "checkpoint_sha256": self.protocol_doc["checkpoints"]
                                                    [checkpoint]["sha256"],
            "split": "selection",
            "pair_manifest": {
                "name": "selection", "expected": 89, "actual": 89,
                "unique": 89, "sha256": manifest_sha(self.pairs),
            },
            "cache_manifest": {
                "count": 89, "unique": 89,
                "sha256": freezer._canonical_json_sha(cache_entries),
            },
            "gt_anchor_manifest": {
                "loader": "adapters.sgf.data_sources.load_anchor_ids",
                "loader_source_sha256": "f" * 64,
                "pair_count": 89,
                "sha256": freezer._canonical_json_sha(anchor_entries),
            },
            "provenance": {
                "gt_posthoc_only": True,
                "source_sha256": {"scripts/v4seal_metrics.py": "e" * 64},
            },
            "pairs": pair_rows,
        }

    def write_all(self):
        for checkpoint in freezer.CHECKPOINTS:
            directory = self.formal / "selection" / checkpoint
            directory.mkdir(parents=True, exist_ok=True)
            for repeat in freezer.REPEATS:
                document = self.make_document(checkpoint, repeat)
                (directory / f"repeat_{repeat:02d}.json").write_text(
                    json.dumps(document))
        self.sidecars.mkdir(parents=True, exist_ok=True)
        for checkpoint in freezer.CHECKPOINTS:
            (self.sidecars / f"{checkpoint}.json").write_text(
                json.dumps(self.make_sidecar(checkpoint)))


class MetricTests(unittest.TestCase):
    def test_macro_micro_and_node_precision_recall(self):
        rows = []
        for index in range(89):
            evidence = ({"tp": 1, "predicted": 2, "anchors": 4,
                         "top1_hits": 1, "top1_total": 2,
                         "top5_hits": 1, "top5_total": 4}
                        if index == 0 else
                        {"tp": 0, "predicted": 0, "anchors": 0,
                         "top1_hits": 0, "top1_total": 0,
                         "top5_hits": 0, "top5_total": 0})
            rows.append({"node_evidence": evidence})
        metrics = freezer.aggregate_node_metrics(rows, "test")
        self.assertAlmostEqual(metrics["macro_node_precision"], 0.5 / 89)
        self.assertAlmostEqual(metrics["macro_node_recall"], 0.25 / 89)
        self.assertAlmostEqual(metrics["macro_node_f1"], (1 / 3) / 89)
        self.assertAlmostEqual(metrics["micro_node_precision"], 0.5)
        self.assertAlmostEqual(metrics["micro_node_recall"], 0.25)
        self.assertAlmostEqual(metrics["micro_node_f1"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_top1"], 0.5 / 89)
        self.assertAlmostEqual(metrics["micro_top1"], 0.5)
        self.assertAlmostEqual(metrics["macro_top5"], 0.25 / 89)
        self.assertAlmostEqual(metrics["micro_top5"], 0.25)

    def test_missing_evidence_is_blocked_not_guessed(self):
        rows = [{"node_evidence": {}} for _ in range(89)]
        with self.assertRaisesRegex(freezer.Blocked, "missing"):
            freezer.aggregate_node_metrics(rows, "test")

    def test_bad_top5_denominator_is_blocked(self):
        value = {"tp": 1, "predicted": 1, "anchors": 2,
                 "top1_hits": 1, "top1_total": 1,
                 "top5_hits": 1, "top5_total": 1}
        rows = [{"node_evidence": value} for _ in range(89)]
        with self.assertRaisesRegex(freezer.Blocked, "must equal anchors"):
            freezer.aggregate_node_metrics(rows, "test")


class GateAndFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.evidence = SyntheticEvidence(self.temporary.name)
        self.evidence.write_all()

    def tearDown(self):
        self.temporary.cleanup()

    def load(self):
        return freezer.load_and_validate(
            self.evidence.formal, self.evidence.protocol_doc)

    def load_sidecars(self):
        return freezer.load_node_sidecars(
            self.evidence.sidecars, self.evidence.protocol_doc,
            self.evidence.pairs)

    def test_successful_gate_recomputation_selects_only_B_C1(self):
        _results, computed, _inputs = self.load()
        node_metrics, _sidecar_inputs = self.load_sidecars()
        decision = freezer.evaluate_gates(
            self.evidence.protocol_doc, computed, node_metrics)
        self.assertTrue(decision["gate1"]["passed"])
        self.assertTrue(decision["gate2"]["passed"])
        self.assertEqual(decision["winner"]["checkpoint"], "B")
        self.assertEqual(decision["winner"]["path"], "C1")
        self.assertEqual(decision["diagnostic_only"], ["D"])
        safety = decision["candidate_path_safety"]
        self.assertTrue(safety["passed"])
        self.assertEqual(set(safety["paths"]), set(freezer.PATHS))
        self.assertTrue(all(
            path["passed"] for path in safety["paths"].values()))

    def test_non_winner_B_path_accepted_error_blocks_global_safety(self):
        path = self.evidence.formal / "selection/B/repeat_01.json"
        document = json.loads(path.read_text())
        row = document["rows"][10]
        self.assertFalse(row["paths"]["F"]["strict"])
        item = row["paths"]["F"]
        item.update({
            "accepted": True,
            "accepted_correct": False,
            "accepted_error": True,
            "decision": {
                "usable_for_reconstruction": True,
                "status": "accepted",
            },
        })
        document["counts"]["F"] = freezer._recompute_counts(
            document["rows"], "F", "B/repeat_01")
        path.write_text(json.dumps(document))
        _results, computed, _inputs = self.load()
        node_metrics, _sidecar_inputs = self.load_sidecars()
        with self.assertRaisesRegex(
                freezer.Blocked, r"B/F/repeat_01: accepted_error=1"):
            freezer.evaluate_gates(
                self.evidence.protocol_doc, computed, node_metrics)

    def test_non_winner_B_path_failed_blocks_global_safety(self):
        path = self.evidence.formal / "selection/B/repeat_02.json"
        document = json.loads(path.read_text())
        row = document["rows"][20]
        item = row["paths"]["C0"]
        item.update({
            "valid": False, "strict": False, "relaxed": False,
            "accepted": False, "accepted_correct": False,
            "accepted_error": False,
        })
        item.pop("decision", None)
        document["counts"]["C0"] = freezer._recompute_counts(
            document["rows"], "C0", "B/repeat_02")
        path.write_text(json.dumps(document))
        _results, computed, _inputs = self.load()
        node_metrics, _sidecar_inputs = self.load_sidecars()
        with self.assertRaisesRegex(
                freezer.Blocked,
                r"B/C0/repeat_02: accepted_error=0, failed=1, unknown=0"):
            freezer.evaluate_gates(
                self.evidence.protocol_doc, computed, node_metrics)

    def test_B_unknown_outcome_blocks_global_safety(self):
        path = self.evidence.formal / "selection/B/repeat_02.json"
        document = json.loads(path.read_text())
        row = document["rows"][20]
        row["audit"]["zero_candidate"] = True
        for registration_path in freezer.PATHS:
            item = row["paths"][registration_path]
            item.update({
                "valid": False, "strict": False, "relaxed": False,
                "accepted": False, "accepted_correct": False,
                "accepted_error": False,
            })
            item.pop("decision", None)
            document["counts"][registration_path] = \
                freezer._recompute_counts(
                    document["rows"], registration_path, "B/repeat_02")
        path.write_text(json.dumps(document))
        _results, computed, _inputs = self.load()
        node_metrics, _sidecar_inputs = self.load_sidecars()
        with self.assertRaisesRegex(
                freezer.Blocked,
                r"B/F/repeat_02: accepted_error=0, failed=0, unknown=1"):
            freezer.evaluate_gates(
                self.evidence.protocol_doc, computed, node_metrics)

    def test_counts_must_be_recomputable(self):
        path = self.evidence.formal / "selection/A/repeat_00.json"
        document = json.loads(path.read_text())
        document["counts"]["F"]["raw_strict"] += 1
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(freezer.Blocked, "not recomputable"):
            self.load()

    def test_duplicate_pair_or_wrong_manifest_is_blocked(self):
        path = self.evidence.formal / "selection/A/repeat_00.json"
        document = json.loads(path.read_text())
        document["rows"][1]["pair_id"] = document["rows"][0]["pair_id"]
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(freezer.Blocked, "unique"):
            self.load()

    def test_exactly_three_repeats_required(self):
        extra = self.evidence.formal / "selection/A/repeat_03.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(freezer.Blocked, "exactly"):
            self.load()

    def test_checkpoint_sha_is_strict(self):
        path = self.evidence.formal / "selection/B/repeat_00.json"
        document = json.loads(path.read_text())
        document["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(freezer.Blocked, "checkpoint SHA"):
            self.load()

    def test_node_sidecar_checkpoint_sha_is_strict(self):
        path = self.evidence.sidecars / "B.json"
        document = json.loads(path.read_text())
        document["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(freezer.Blocked, "checkpoint SHA"):
            self.load_sidecars()

    def test_formal_repository_head_must_be_40_hex(self):
        path = self.evidence.formal / "selection/A/repeat_00.json"
        document = json.loads(path.read_text())
        document["repository"]["head"] = "not-a-git-head"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(
                freezer.Blocked,
                "repository HEAD must be 40 lowercase hex"):
            self.load()

    def test_repo_root_must_match_formal_code_root(self):
        with self.assertRaisesRegex(
                freezer.Blocked, "repo_root/code_root mismatch"):
            freezer.build_freeze(
                self.evidence.formal, self.evidence.protocol,
                self.evidence.root, self.evidence.sidecars)

    def test_zero_candidate_must_match_predicted_denominator(self):
        path = self.evidence.sidecars / "B.json"
        document = json.loads(path.read_text())
        document["pairs"][0]["zero_candidate"] = True
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(freezer.Blocked, "zero-candidate"):
            self.load_sidecars()

    def test_missing_node_evidence_blocks_and_writes_no_freeze(self):
        (self.evidence.sidecars / "B.json").unlink()
        output = self.evidence.formal / "frozen_selection.json"
        rc = freezer.main([
            "--formal-root", str(self.evidence.formal),
            "--protocol-json", str(self.evidence.protocol),
            "--repo-root", str(ROOT), "--output", str(output),
        ])
        self.assertEqual(rc, 2)
        self.assertFalse(output.exists())

    def test_blocked_receipt_matches_stdout_and_has_no_winner(self):
        (self.evidence.sidecars / "B.json").unlink()
        output = self.evidence.formal / "frozen_selection.json"
        receipt = self.evidence.root / "blocked_receipt.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = freezer.main([
                "--formal-root", str(self.evidence.formal),
                "--protocol-json", str(self.evidence.protocol),
                "--repo-root", str(ROOT), "--output", str(output),
                "--receipt", str(receipt),
            ])
        report = json.loads(stdout.getvalue())
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(receipt.read_text()), report)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertNotIn("winner", report)
        self.assertFalse(output.exists())

    def test_existing_receipt_is_not_overwritten_or_run(self):
        output = self.evidence.formal / "frozen_selection.json"
        receipt = self.evidence.root / "existing_receipt.json"
        receipt.write_text("sentinel\n")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = freezer.main([
                "--formal-root", str(self.evidence.formal),
                "--protocol-json", str(self.evidence.protocol),
                "--repo-root", str(ROOT), "--output", str(output),
                "--receipt", str(receipt),
            ])
        self.assertEqual(rc, 2)
        self.assertEqual(receipt.read_text(), "sentinel\n")
        self.assertFalse(output.exists())
        self.assertIn("refusing to overwrite", stdout.getvalue())

    def test_success_receipt_matches_stdout(self):
        output = self.evidence.formal / "frozen_selection.json"
        receipt = self.evidence.root / "success_receipt.json"
        document = {
            "schema": freezer.FREEZE_SCHEMA, "status": "FROZEN",
            "decision": {"winner": {
                "checkpoint": "B", "path": "C1",
            }},
        }
        stdout = io.StringIO()
        with mock.patch.object(freezer, "build_freeze",
                               return_value=document):
            with contextlib.redirect_stdout(stdout):
                rc = freezer.main([
                    "--formal-root", str(self.evidence.formal),
                    "--protocol-json", str(self.evidence.protocol),
                    "--repo-root", str(ROOT), "--output", str(output),
                    "--receipt", str(receipt),
                ])
        report = json.loads(stdout.getvalue())
        self.assertEqual(rc, 0)
        self.assertTrue(output.is_file())
        self.assertEqual(json.loads(receipt.read_text()), report)
        self.assertEqual(report["winner"], document["decision"]["winner"])

    def test_atomic_freeze_contains_all_nine_input_hashes(self):
        output = self.evidence.formal / "frozen_selection.json"
        provenance = {
            "repository_root": str(ROOT), "git_head": "f" * 40,
            "git_branch": "wu/test", "tracked_dirty": False,
            "source_sha256": {"scripts/v6fix_gate_freezer.py": "a" * 64},
            "protocol_json": {"sha256": "b" * 64},
            "protocol_md": {"sha256": "c" * 64},
            "checkpoint_files": {},
        }
        with mock.patch.object(freezer, "collect_provenance",
                               return_value=provenance):
            document = freezer.build_freeze(
                self.evidence.formal, self.evidence.protocol, ROOT,
                self.evidence.sidecars)
        freezer.atomic_write_new(output, document)
        saved = json.loads(output.read_text())
        self.assertEqual(saved["status"], "FROZEN")
        self.assertEqual(saved["decision"]["winner"]["checkpoint"], "B")
        self.assertEqual(len(saved["inputs"]), 12)
        self.assertTrue(all(len(item["sha256"]) == 64
                            for item in saved["inputs"]))
        with self.assertRaisesRegex(freezer.Blocked, "overwrite"):
            freezer.atomic_write_new(output, document)


if __name__ == "__main__":
    unittest.main()
