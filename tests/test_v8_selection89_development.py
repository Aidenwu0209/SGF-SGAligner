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


def pair_id(index):
    return (f"00000000-0000-0000-0000-{index:012x}_to_"
            f"10000000-0000-0000-0000-{index:012x}")


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.pairlist = self.root / "selection.txt"
        self.cache_root = self.root / "cache"
        self.protocol = self.root / "V8.md"
        self.cache_root.mkdir()
        self.ids = [pair_id(index) for index in range(dev.PAIR_COUNT)]
        self.pairlist.write_text("".join(f"{value}\n" for value in self.ids))
        self.protocol.write_text("frozen v8 protocol\n")
        self.cache_by_path = {}
        for index, value in enumerate(self.ids):
            path = self.cache_root / f"{value}.pt"
            path.write_bytes(f"cache-{index}".encode())
            self.cache_by_path[path] = {
                "pair_id": value,
                "input_sha256": f"{index + 1:064x}",
                "_file_sha256": dev.sha256_file(path),
                "_members": [(0, 0), (1, 1)],
                "geot": {
                    (0, 0): {"status": "ok"},
                    (1, 1): {"status": "insufficient_post_voxel_points"},
                },
            }

    def loader(self, path, pair, expected):
        row = dict(self.cache_by_path[path])
        if row["pair_id"] != pair or row["_file_sha256"] != expected:
            raise AssertionError("bad fixture binding")
        return row

    def patches(self):
        return (
            mock.patch.object(dev, "DEFAULT_PAIRLIST", self.pairlist),
            mock.patch.object(dev, "DEFAULT_CACHE_ROOT", self.cache_root),
            mock.patch.object(dev.pilot, "DEFAULT_CACHE_ROOT", self.cache_root),
            mock.patch.object(dev, "DEFAULT_DECISION_PROTOCOL", self.protocol),
        )

    def build(self):
        first, second, third, fourth = self.patches()
        with first, second, third, fourth:
            return dev.build_manifest(
                pairlist=self.pairlist, cache_root=self.cache_root,
                decision_protocol=self.protocol, cache_loader=self.loader)


class ManifestTests(unittest.TestCase):
    def test_freezes_exact_label_free_selection89_cache_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            manifest = fixture.build()
        self.assertEqual(dev.PAIR_COUNT, len(manifest["pairs"]))
        self.assertEqual(dev.TOTAL_WORKERS,
                         manifest["worker_contract"]["total_workers"])
        self.assertEqual(dev.EVIDENCE_CLASS, manifest["evidence_class"])
        self.assertFalse(manifest["gt_separation"]["labels_in_manifest"])
        serialized = json.dumps(manifest).lower()
        self.assertNotIn('"rre"', serialized)
        self.assertNotIn('"rte"', serialized)
        self.assertNotIn('"accepted_correct"', serialized)

    def test_duplicate_pairlist_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            fixture.ids[-1] = fixture.ids[0]
            fixture.pairlist.write_text(
                "".join(f"{value}\n" for value in fixture.ids))
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth, self.assertRaises(
                    dev.Selection89EvidenceError):
                dev.build_manifest(
                    pairlist=fixture.pairlist, cache_root=fixture.cache_root,
                    decision_protocol=fixture.protocol,
                    cache_loader=fixture.loader)

    def test_extra_or_missing_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            (fixture.cache_root / "extra.pt").write_bytes(b"extra")
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth, self.assertRaisesRegex(
                    dev.Selection89EvidenceError, "exactly match"):
                dev.build_manifest(
                    pairlist=fixture.pairlist, cache_root=fixture.cache_root,
                    decision_protocol=fixture.protocol,
                    cache_loader=fixture.loader)

    def test_manifest_file_payload_and_cache_drift_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            manifest = fixture.build()
            path = Path(directory) / "manifest.json"
            dev._atomic_create_json(path, manifest)
            digest = dev.sha256_file(path)
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth:
                loaded = dev.validate_manifest(path, digest)
                self.assertEqual(dev.PAIR_COUNT, loaded["pair_count"])
                (fixture.cache_root / f"{fixture.ids[0]}.pt").write_bytes(
                    b"drift")
                with self.assertRaisesRegex(
                        dev.Selection89EvidenceError, "cache drift"):
                    dev.validate_manifest(path, digest)

    def test_manifest_unknown_or_forbidden_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            manifest = fixture.build()
            manifest["gt_transform"] = [[1, 0, 0, 0]]
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(manifest))
            digest = dev.sha256_file(path)
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth, self.assertRaises(
                    dev.Selection89EvidenceError):
                dev.validate_manifest(path, digest, verify_caches=False)

    def test_nested_manifest_injection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            manifest = fixture.build()
            manifest["decision_protocol"]["threshold_override"] = 99
            payload = dict(manifest)
            payload.pop("payload_sha256")
            manifest["payload_sha256"] = dev.stable_json_hash(payload)
            path = Path(directory) / "nested.json"
            path.write_text(json.dumps(manifest))
            digest = dev.sha256_file(path)
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth, self.assertRaisesRegex(
                    dev.Selection89EvidenceError, "schema mismatch"):
                dev.validate_manifest(path, digest, verify_caches=False)

    def test_atomic_freeze_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            dev._atomic_create_json(path, {"a": 1})
            with self.assertRaisesRegex(
                    dev.Selection89EvidenceError, "overwrite"):
                dev._atomic_create_json(path, {"a": 2})
            self.assertEqual({"a": 1}, json.loads(path.read_text()))


class PlanAndExecutionContractTests(unittest.TestCase):
    def test_dry_run_is_development_only_and_excludes_later_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(directory)
            manifest = fixture.build()
            manifest["_path"] = str(Path(directory) / "manifest.json")
            manifest["_file_sha256"] = "a" * 64
            first, second, third, fourth = fixture.patches()
            with first, second, third, fourth, mock.patch.object(
                    dev, "excluded_prior_evidence", return_value=[]):
                plan = dev.build_dry_run_plan(manifest)
        self.assertEqual(dev.TOTAL_WORKERS,
                         plan["execution_shape"]["total_workers"])
        self.assertEqual("development only",
                         plan["authorization"]["selection89"])
        self.assertFalse(plan["authorization"]["calibration90"])
        self.assertFalse(plan["authorization"]["fixed12"])
        self.assertFalse(plan["authorization"]["official92"])
        self.assertFalse(plan["authorization"]["confirmatory_claim_allowed"])

    def test_worker_shape_is_two_by_two_by_five(self):
        names = dev._outer_worker_names()
        self.assertEqual(10, len(names))
        self.assertIn("forward_00.json", names)
        self.assertIn("reverse_04.json", names)
        self.assertEqual(1780, dev.TOTAL_WORKERS)

    def test_output_must_be_named_repository_outputs_child(self):
        for path in (dev.CODE_ROOT, dev.CODE_ROOT / "outputs",
                     dev.CODE_ROOT / "scripts" / "batch"):
            with self.subTest(path=path), self.assertRaises(
                    dev.Selection89EvidenceError):
                dev._validate_output_root(path)
        accepted = dev.CODE_ROOT / "outputs" / "v8_selection89_workers"
        self.assertEqual(accepted.resolve(), dev._validate_output_root(accepted))

    def test_prior_v6_repeats_are_hash_only_and_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                (root / f"repeat_{index:02d}.json").write_text(
                    "this is intentionally not parsed")
            with mock.patch.object(dev, "EXCLUDED_V6_REPEAT_ROOT", root):
                rows = dev.excluded_prior_evidence()
        self.assertEqual(3, len(rows))
        self.assertTrue(all(not row["eligible_for_v8_worker_replay"]
                            for row in rows))
        self.assertTrue(all(dev.SHA256_RE.fullmatch(row["sha256"])
                            for row in rows))


if __name__ == "__main__":
    unittest.main()
