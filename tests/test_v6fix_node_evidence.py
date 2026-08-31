import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "v6fix_node_evidence.py"
SPEC = importlib.util.spec_from_file_location("v6fix_node_evidence", MODULE_PATH)
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


class CacheFixtures:
    def __init__(self, root):
        self.root = Path(root)
        self.formal = self.root / "formal_v2"
        self.code_root = ROOT
        self.asset_root = self.root / "assets"
        self.source_relpath = "scripts/v6fix_node_evidence.py"
        self.source_sha = extractor.sha256_file(
            self.code_root / self.source_relpath)
        self.pairs = [f"source_{index:03d}_to_ref_{index:03d}"
                      for index in range(89)]
        self.protocol = {"checkpoints": {}}
        for checkpoint, payload in zip(extractor.CHECKPOINTS,
                                       (b"a", b"b", b"d")):
            ckpt = self.asset_root / f"{checkpoint}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(payload)
            self.protocol["checkpoints"][checkpoint] = {
                "path": f"{checkpoint}.pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self._write_formal(checkpoint)
            self._write_caches(checkpoint)

    def _manifest(self):
        return {
            "name": "selection", "expected": 89, "actual": 89,
            "unique": 89,
            "sha256": hashlib.sha256(
                ("\n".join(self.pairs) + "\n").encode()).hexdigest(),
        }

    def _write_formal(self, checkpoint):
        directory = self.formal / "selection" / checkpoint
        directory.mkdir(parents=True, exist_ok=True)
        for repeat in extractor.REPEATS:
            document = {
                "schema": extractor.RESULT_SCHEMA,
                "code_root": str(self.code_root.resolve()),
                "asset_root": str(self.asset_root.resolve()),
                "repository": {
                    "head": "a" * 40, "tracked_dirty": False,
                },
                "split": "selection", "checkpoint": checkpoint,
                "checkpoint_sha256": self.protocol["checkpoints"]
                                                     [checkpoint]["sha256"],
                "repeat": repeat, "split_manifest": self._manifest(),
                "rows": [{"pair_id": pair_id} for pair_id in self.pairs],
            }
            (directory / f"repeat_{repeat:02d}.json").write_text(
                json.dumps(document))

    def _write_caches(self, checkpoint):
        directory = self.formal / "cache_v2" / checkpoint / "selection"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = self.protocol["checkpoints"][checkpoint]["sha256"]
        for index, pair_id in enumerate(self.pairs):
            cache_key = f"{index + 1:064x}"
            cache = {
                "cache_schema": extractor.CACHE_SCHEMA,
                "pair_id": pair_id, "checkpoint_id": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "input_sha256": f"{index + 101:064x}",
                "embedding_sha256": f"{index + 201:064x}",
                "similarity_sha256": f"{index + 301:064x}",
                "node_corrs": [(0, 1)],
                "rank_list": [[0, 1], [1, 0]],
                "provenance": {
                    "cache_key": cache_key, "pair_id": pair_id,
                    "checkpoint_id": checkpoint,
                    "checkpoint_sha256": checkpoint_sha,
                    "object_ids_order": [1000 + index, 2000 + index],
                    "src_count": 1,
                    "source_hashes": {
                        self.source_relpath: self.source_sha,
                    },
                },
                "geot": {"0:1": {"corrs": 17}},
            }
            torch.save(cache, directory / f"{pair_id}.pt")

    def anchor_loader(self, pair_id):
        index = self.pairs.index(pair_id)
        source = self.root / "gt" / pair_id / "pair.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        anchors = [(1000 + index, 2000 + index), (999999, 888888)]
        source.write_text(json.dumps({"anchor_pairs": anchors}))
        return anchors, source


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CacheFixtures(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_extracts_exact_v4seal_counts_and_manifests(self):
        document = extractor.build_sidecar(
            self.fixture.formal, self.fixture.protocol, ROOT, "A",
            self.fixture.anchor_loader)
        self.assertEqual(document["schema"], extractor.SIDECAR_SCHEMA)
        self.assertEqual(len(document["pairs"]), 89)
        first = document["pairs"][0]
        self.assertEqual(first["node_evidence"], {
            "tp": 1, "predicted": 1, "anchors": 1,
            "top1_hits": 1, "top1_total": 1,
            "top5_hits": 1, "top5_total": 1,
        })
        self.assertEqual(first["anchors"]["unmapped_object_ids"],
                         [[999999, 888888]])
        self.assertEqual(document["cache_manifest"]["count"], 89)
        self.assertEqual(document["gt_anchor_manifest"]["pair_count"], 89)
        self.assertTrue(document["provenance"]["gt_posthoc_only"])

    def test_missing_cache_field_fails_closed_with_field_name(self):
        path = (self.fixture.formal / "cache_v2/A/selection"
                / f"{self.fixture.pairs[0]}.pt")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        cache.pop("rank_list")
        torch.save(cache, path)
        with self.assertRaisesRegex(extractor.ExtractionBlocked, "rank_list"):
            extractor.build_sidecar(
                self.fixture.formal, self.fixture.protocol, ROOT, "A",
                self.fixture.anchor_loader)

    def test_cache_checkpoint_sha_mismatch_fails_closed(self):
        path = (self.fixture.formal / "cache_v2/B/selection"
                / f"{self.fixture.pairs[0]}.pt")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        cache["checkpoint_sha256"] = "0" * 64
        torch.save(cache, path)
        with self.assertRaisesRegex(extractor.ExtractionBlocked,
                                    "checkpoint identity"):
            extractor.build_sidecar(
                self.fixture.formal, self.fixture.protocol, ROOT, "B",
                self.fixture.anchor_loader)

    def test_repo_root_must_match_formal_code_root(self):
        with self.assertRaisesRegex(
                extractor.ExtractionBlocked,
                "repo_root/code_root mismatch"):
            extractor.build_sidecar(
                self.fixture.formal, self.fixture.protocol,
                self.fixture.root, "A", self.fixture.anchor_loader)

    def test_formal_repository_head_must_be_40_hex(self):
        path = self.fixture.formal / "selection/A/repeat_00.json"
        document = json.loads(path.read_text())
        document["repository"]["head"] = "not-a-git-head"
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(
                extractor.ExtractionBlocked,
                "repository HEAD must be 40 lowercase hex"):
            extractor.build_sidecar(
                self.fixture.formal, self.fixture.protocol, ROOT, "A",
                self.fixture.anchor_loader)

    def test_rank_list_must_be_full_permutation(self):
        path = (self.fixture.formal / "cache_v2/D/selection"
                / f"{self.fixture.pairs[0]}.pt")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        cache["rank_list"][0] = [0, 0]
        torch.save(cache, path)
        with self.assertRaisesRegex(extractor.ExtractionBlocked,
                                    "full permutation"):
            extractor.build_sidecar(
                self.fixture.formal, self.fixture.protocol, ROOT, "D",
                self.fixture.anchor_loader)

    def test_atomic_sidecar_refuses_overwrite(self):
        output = self.fixture.root / "A.json"
        extractor.atomic_write_new(output, {"schema": "test"})
        with self.assertRaisesRegex(extractor.ExtractionBlocked, "overwrite"):
            extractor.atomic_write_new(output, {"schema": "changed"})


if __name__ == "__main__":
    unittest.main()
