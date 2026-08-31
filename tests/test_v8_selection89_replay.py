import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v8_selection89_development as dev  # noqa: E402
import v8_selection89_replay as replay  # noqa: E402
import v8_stage_order_posthoc as posthoc  # noqa: E402


class IsolationTests(unittest.TestCase):
    def test_gt_loader_is_absent_and_posthoc_only(self):
        tree = ast.parse(Path(replay.__file__).read_text())
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("load_gt_transform", imports)
        self.assertIn("load_gt_transform", Path(posthoc.__file__).read_text())

    def test_batch_with_policy_already_applied_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "batch.json"
            manifest_path = root / "manifest.json"
            manifest_path.write_text("{}")
            snapshot = {"head": "a" * 40}
            snapshot["snapshot_sha256"] = dev.stable_json_hash(snapshot)
            batch = {
                "schema": dev.BATCH_SCHEMA,
                "status": "GT_FREE_WORKERS_COMPLETE",
                "evidence_class": dev.EVIDENCE_CLASS,
                "posthoc_not_run": True,
                "policy_not_applied": False,
                "pair_count": dev.PAIR_COUNT,
                "outer_repeats_per_pair": dev.OUTER_REPEATS,
                "workers_per_outer": dev.WORKERS_PER_OUTER,
                "total_workers": dev.TOTAL_WORKERS,
                "manifest": {"sha256": "b" * 64},
                "source_snapshot": snapshot,
                "pair_receipts": [],
            }
            batch["evidence_sha256"] = dev.stable_json_hash(batch)
            batch_path.write_text(json.dumps(batch))
            with mock.patch.object(
                    replay.dev, "validate_manifest",
                    return_value={"_file_sha256": "b" * 64,
                                  "pairs": []}), self.assertRaisesRegex(
                    replay.Selection89ReplayError, "identity/shape"):
                replay._validate_batch(
                    batch_path, manifest_path, "c" * 64)


class ReplayTests(unittest.TestCase):
    def test_replay_requires_fresh_trace_and_is_posthoc_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "batch.json"
            manifest_path = root / "manifest.json"
            batch_path.write_text("frozen batch")
            manifest_path.write_text("frozen manifest")
            manifest_sha = dev.sha256_file(manifest_path)
            manifest = {
                "_file_sha256": manifest_sha,
                "pair_ids_sha256": "d" * 64,
            }
            batch = {
                "evidence_sha256": "e" * 64,
                "source_snapshot": {"snapshot_sha256": "f" * 64},
            }
            loaded = [{
                "pair": {"pair_id": "pair", "cache_sha256": "a" * 64},
                "outers": [
                    {"outer_repeat": outer,
                     "receipt_path": str(root / f"outer-{outer}.json"),
                     "receipt_sha256": f"{outer + 1:064x}",
                     "receipt_evidence_sha256": f"{outer + 11:064x}",
                     "workers": [], "bindings": []}
                    for outer in range(2)],
            }]
            result = {
                "usable_for_reconstruction": True,
                "fresh_v8_qualified": True,
                "selected_observed_forward_medoid": None,
            }
            with mock.patch.object(
                    replay, "_validate_batch",
                    return_value=(manifest, batch, loaded)), \
                    mock.patch.object(
                        replay, "evaluate_stage_order",
                        return_value=result) as evaluate:
                output = replay.replay(
                    batch_path, manifest_path, manifest_sha)
            self.assertEqual([1, 1], output["usable_pairs_per_outer"])
            self.assertEqual([1, 1],
                             output["fresh_v8_qualified_pairs_per_outer"])
            self.assertFalse(output["qualifies_as_blind_gate"])
            self.assertTrue(output["posthoc_not_run"])
            self.assertEqual(2, evaluate.call_count)
            self.assertTrue(all(
                call.kwargs["require_fixed_trace"]
                for call in evaluate.call_args_list))

            receipt = root / "replay.json"
            receipt.write_text(json.dumps(output))
            validated = posthoc.validate_replay(receipt)
            self.assertEqual(output["evidence_sha256"],
                             validated["evidence_sha256"])


if __name__ == "__main__":
    unittest.main()
