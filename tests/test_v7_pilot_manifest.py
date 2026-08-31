import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v7_pilot_manifest.py"
PROJECTION = ROOT / "outputs/v7_pilot_manifest_seal_20260830/whitelist_projection.json"
MANIFEST = ROOT / "outputs/v7_pilot_manifest_seal_20260830/v7_pilot_manifest.json"

spec = importlib.util.spec_from_file_location("v7_pilot_manifest", SCRIPT)
manifest = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = manifest
spec.loader.exec_module(manifest)


class FrozenManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projection_bytes = PROJECTION.read_bytes()
        cls.projection = json.loads(cls.projection_bytes)
        cls.frozen = json.loads(MANIFEST.read_bytes())

    def test_batch_runner_contract_is_exact(self):
        frozen = self.frozen
        self.assertEqual("v7-registration-veto-batch-manifest-v1", frozen["schema"])
        self.assertEqual("FROZEN", frozen["status"])
        self.assertEqual(12, frozen["pair_count"])
        self.assertEqual("B", frozen["checkpoint_id"])
        self.assertEqual(manifest.CHECKPOINT_SHA256, frozen["checkpoint_sha256"])
        self.assertEqual(manifest.PROTOCOL_SHA256, frozen["protocol_sha256"])
        self.assertEqual(12, len(frozen["pairs"]))
        self.assertEqual(
            {"pair_id", "cache_sha256", "role"}, set(frozen["pairs"][0])
        )
        ids = [row["pair_id"] for row in frozen["pairs"]]
        self.assertEqual(12, len(set(ids)))
        self.assertEqual(list(manifest.EXPECTED_PAIR_IDS), ids)
        self.assertEqual(list(manifest.EXPECTED_ROLES), [row["role"] for row in frozen["pairs"]])
        expected = hashlib.sha256("".join(f"{value}\n" for value in ids).encode()).hexdigest()
        self.assertEqual(expected, frozen["pair_ids_sha256"])

    def test_manifest_recomputes_from_projection(self):
        pairs = manifest.validate_projection(self.projection)
        selected = manifest.select_pairs(pairs)
        rebuilt = manifest.build_manifest(
            self.projection,
            self.projection_bytes,
            selected,
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        self.assertEqual(self.frozen, rebuilt)

    def test_projection_contains_no_forbidden_key(self):
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key.casefold(), manifest.FORBIDDEN_PROJECTION_KEYS)
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(self.projection)

    def test_projection_population_and_exception(self):
        pairs = manifest.validate_projection(self.projection)
        patterns = {}
        for pair in pairs:
            pattern = manifest.pair_metrics(pair)["pattern"]
            patterns[pattern] = patterns.get(pattern, 0) + 1
        self.assertEqual(
            {"PPP": 8, "PRP": 1, "PRR": 1, "RPR": 2, "RRP": 1, "RRR": 76},
            patterns,
        )
        self.assertEqual(manifest.KNOWN_ERROR_PAIR, manifest.EXPECTED_PAIR_IDS[0])


class FailClosedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projection = json.loads(PROJECTION.read_bytes())

    def test_unknown_top_level_field_rejected(self):
        altered = copy.deepcopy(self.projection)
        altered["unexpected"] = True
        with self.assertRaises(manifest.ManifestSealError):
            manifest.validate_projection(altered)

    def test_unknown_nested_field_rejected(self):
        altered = copy.deepcopy(self.projection)
        altered["pairs"][0]["repeats"][0]["rule_b_features"]["unexpected"] = 1
        with self.assertRaises(manifest.ManifestSealError):
            manifest.validate_projection(altered)

    def test_forbidden_label_key_rejected(self):
        altered = copy.deepcopy(self.projection)
        repeat = altered["pairs"][0]["repeats"][0]
        repeat["strict"] = True
        with self.assertRaises(manifest.ManifestSealError):
            manifest.validate_projection(altered)

    def test_forbidden_runtime_paths_rejected(self):
        components = list(sorted(manifest.FORBIDDEN_PATH_COMPONENTS)) + [
            "node_evidence_v2", "calibration90", "fixed12_results", "official92_eval",
        ]
        for component in components:
            path = Path(tempfile.gettempdir()) / component / "evidence.json"
            with self.assertRaises(manifest.ManifestSealError):
                manifest.guard_path(path, "test")

    def test_frozen_manifest_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "manifest.json"
            manifest.atomic_write_json(output, {"first": True})
            with self.assertRaises(manifest.ManifestSealError):
                manifest.atomic_write_json(output, {"second": True})
            self.assertEqual({"first": True}, json.loads(output.read_bytes()))

    def test_pair_id_is_never_parsed(self):
        tree = ast.parse(SCRIPT.read_text())
        suspicious = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"split", "rsplit", "partition", "rpartition"}:
                    if any(isinstance(arg, ast.Constant) and arg.value == "_to_" for arg in node.args):
                        suspicious.append(node.lineno)
        self.assertEqual([], suspicious)

    def test_gt_loaders_are_absent_from_imports_and_calls(self):
        tree = ast.parse(SCRIPT.read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    names.add(node.func.attr)
        self.assertFalse(manifest.FORBIDDEN_SYMBOLS & names)
        manifest.assert_gt_free_ast(SCRIPT)


class RepeatabilityMeaningTests(unittest.TestCase):
    def test_addendum_freezes_decision_level_repeatability(self):
        text = (ROOT / "docs/V7_PILOT_MANIFEST_ADDENDUM.md").read_text()
        self.assertIn("final per-policy `usable_for_reconstruction`/veto", text)
        self.assertIn("does not require byte-identical raw/final transform hashes", text)
        self.assertIn("audit-semantic misclassification", text)
        self.assertIn("No radius, quorum, Rule-B threshold", text)


if __name__ == "__main__":
    unittest.main()
