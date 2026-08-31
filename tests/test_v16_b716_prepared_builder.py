from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from safety.v16_b716_prepared_builder import (
    ARM,
    CANDIDATE_MANIFEST_SCHEMA,
    CANDIDATE_PLAN_SCHEMA,
    EXACT191_PAIR_SCHEMA,
    EXACT191_SCHEMA,
    EXPECTED_EXISTING_TYPED_FAILURE_COUNTS,
    FIXED_PAIR_ORDER,
    OFFICIAL_RELEASE_SHA256,
    V16ContractError,
    build_fixed4_prepared_inputs,
    validate_candidate_and_exact191,
)
from safety.v13_strict_pair_gate import _load_surfaces
from safety.v16_matched_region_colorpcr import (
    array_sha256,
    canonical_surface_from_rows,
    load_raw_inseg,
    resolve_unique_inseg_path,
    sha256_file,
    stable_json_sha256,
)
from scripts.v13_corr_cache_converter import FROZEN_NEIGHBOR_LIMITS
from scripts.v14_fixed4_input_builder import verify_conversion_lineage, v13_sources
from safety.v13_dual_solver_runtime import array_sha256 as v14_array_sha256


CANDIDATE_COUNTS = (48, 48, 48, 47)
EXISTING_COUNTS = (46, 27, 27, 19)
NEW_COUNTS = (2, 21, 21, 28)
HYPOTHESIS_COUNTS = (12, 8, 2, 12)
ROOT = Path(__file__).resolve().parents[1]


def seal_json(path: Path, value: dict) -> str:
    value = copy.deepcopy(value)
    value.pop("payload_sha256", None)
    value["payload_sha256"] = stable_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def raw_cloud(path: Path, object_ids: list[int], shift: float) -> None:
    xyz, labels, colors = [], [], []
    for ordinal, object_id in enumerate(object_ids):
        points = np.stack([
            np.linspace(0.0, 0.059, 60) + shift,
            np.full(60, ordinal * 0.12),
            np.full(60, (ordinal % 3) * 0.02),
        ], axis=1)
        xyz.append(points)
        labels.append(np.full(60, object_id, np.int64))
        colors.append(np.tile(
            np.array([[10 + ordinal % 50, 20, 30, 255]], np.uint8),
            (60, 1)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, xyz=np.concatenate(xyz).astype(np.float32),
             labels=np.concatenate(labels), colors=np.concatenate(colors))


class SyntheticClosure:
    def __init__(self, root: Path):
        self.root = root
        self.raw_root = root / "raw"
        self.candidate_root = root / "candidate"
        self.exact_root = root / "exact"
        self.data: dict[str, dict] = {}
        self.plan_paths: list[Path] = []
        self._build()

    def _build(self) -> None:
        candidate_pairs, exact_pairs = [], []
        for ordinal, pair_id in enumerate(FIXED_PAIR_ORDER):
            src_scan, ref_scan = pair_id.split("_to_")
            candidate_count = CANDIDATE_COUNTS[ordinal]
            hypothesis_count = HYPOTHESIS_COUNTS[ordinal]
            source_ids = [10 + ordinal]
            reference_ids = [1000 + ordinal * 100 + index
                             for index in range(candidate_count)]
            src_path = self.raw_root / src_scan / "inseg_cloud.npz"
            ref_path = self.raw_root / ref_scan / "inseg_cloud.npz"
            raw_cloud(src_path, source_ids, ordinal * 0.01)
            raw_cloud(ref_path, reference_ids, 0.25 + ordinal * 0.01)
            source = load_raw_inseg(src_path, scan_id=src_scan, side="source")
            reference = load_raw_inseg(ref_path, scan_id=ref_scan, side="reference")
            registration = {}
            _, registration[0] = canonical_surface_from_rows(source, source_ids[0])
            for index, object_id in enumerate(reference_ids):
                _, registration[index + 1] = canonical_surface_from_rows(
                    reference, object_id)
            obj_ids = np.asarray(source_ids + reference_ids, np.int64)
            data = {
                "src_count": 1,
                "obj_ids": obj_ids,
                "registration_pts": registration,
                "edges_explicit": np.empty((0, 2), np.int64),
            }
            self.data[pair_id] = data
            records = [{
                "source_index": 0,
                "reference_index": index + 1,
                "forward_cross_rank": index + 1,
                "reverse_cross_rank": index + 1,
                "worst_cross_rank": index + 1,
                "rank_sum": 2 * (index + 1),
            } for index in range(candidate_count)]
            # Mirror the current sealed exact191 closure rather than the old
            # artificial four-failures-per-pair fixture.  Eight frozen
            # hypotheses contain historical typed failures and exactly two
            # additional hypotheses contain authorized-backfill typed
            # failures, yielding the production 8/10 split.
            existing_typed_by_pair = (
                {0, 1, 2, 3, 12, 13, 14, 15, 16},
                {0, 1},
                {0, 2, 3, 4},
                {0},
            )
            hypothesis_candidate_indices = (
                list(range(hypothesis_count)),
                [EXISTING_COUNTS[ordinal], 0, 1, 2, 3, 4, 5, 6],
                [EXISTING_COUNTS[ordinal], 0],
                list(range(hypothesis_count)),
            )[ordinal]
            hypotheses = []
            for index, candidate_index in enumerate(
                    hypothesis_candidate_indices):
                member = records[candidate_index]
                payload = {
                    "members": [[0, candidate_index + 1]],
                    "member_rank_records": [member],
                    "member_count": 1,
                }
                hypotheses.append({
                    "hypothesis_index": index,
                    "hypothesis_sha256": stable_json_sha256(payload),
                    **payload,
                })
            bindings = []
            for node in range(len(obj_ids)):
                side = "source" if node == 0 else "reference"
                raw = source if side == "source" else reference
                object_id = int(obj_ids[node])
                indices, reconstructed = canonical_surface_from_rows(raw, object_id)
                bindings.append({
                    "node_index": node,
                    "side": side,
                    "scan_id": raw.scan_id,
                    "object_id": object_id,
                    "raw_inseg_path": str(raw.path),
                    "raw_inseg_sha256": raw.file_sha256,
                    "raw_row_count": len(indices),
                    "raw_row_indices_sha256": array_sha256(indices),
                    "canonical_registration_surface_sha256":
                        array_sha256(np.asarray(reconstructed, np.float64)),
                    "canonical_registration_points": len(reconstructed),
                })
            short_id = f"pair{ordinal}"
            plan_path = self.candidate_root / "pairs" / short_id / "plan.json"
            plan = {
                "schema": CANDIDATE_PLAN_SCHEMA,
                "short_id": short_id,
                "pair_id": pair_id,
                "candidate_count": candidate_count,
                "hypothesis_count": hypothesis_count,
                "candidate_rank_records": records,
                "hypotheses": hypotheses,
                "canonical_surface_bindings": bindings,
                "geot_entries": [],
                "domain": {
                    "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
                    "matched": True,
                    "legacy_B_ep20_or_89ed_consumed": False,
                },
            }
            typed_indices = existing_typed_by_pair[ordinal]
            if (len(typed_indices)
                    != EXPECTED_EXISTING_TYPED_FAILURE_COUNTS[ordinal]):
                raise AssertionError("synthetic existing typed count mismatch")
            for index, record in enumerate(records):
                existing = index < EXISTING_COUNTS[ordinal]
                entry = {
                    "candidate_index": index,
                    "node_pair": [0, index + 1],
                    "object_pair": [source_ids[0], reference_ids[index]],
                    "origin": ("official_pair_cache" if existing
                               else "missing_execution_disabled"),
                    "immutable": existing,
                    "status": ("typed_failure" if existing and index in typed_indices
                               else ("ok" if existing else "missing_execution_disabled")),
                    "source_cache_row": index if existing else None,
                }
                payload = dict(entry)
                entry["entry_sha256"] = stable_json_sha256(payload)
                plan["geot_entries"].append(entry)
            plan_sha = seal_json(plan_path, plan)
            self.plan_paths.append(plan_path)
            candidate_pairs.append({
                "short_id": short_id,
                "pair_id": pair_id,
                "candidate_count": candidate_count,
                "hypothesis_count": hypothesis_count,
                "existing_reused": EXISTING_COUNTS[ordinal],
                "existing_failed":
                    EXPECTED_EXISTING_TYPED_FAILURE_COUNTS[ordinal],
                "missing_disabled": NEW_COUNTS[ordinal],
                "plan_path": plan_path.relative_to(self.candidate_root).as_posix(),
                "plan_bytes": plan_path.stat().st_size,
                "plan_sha256": plan_sha,
            })
            entries_path = self.exact_root / "pairs" / short_id / "entries.json"
            entries = {
                "schema": EXACT191_PAIR_SCHEMA,
                "pair_id": pair_id,
                "candidate_count": candidate_count,
                "entries": [{
                    "candidate_index": index,
                    "node_pair": [0, index + 1],
                    "object_pair": [source_ids[0], reference_ids[index]],
                    "status": "ok" if index % 5 else "typed_failure",
                    "origin": "existing" if index < EXISTING_COUNTS[ordinal]
                    else "new_authorized",
                } for index in range(candidate_count)],
            }
            entries_sha = seal_json(entries_path, entries)
            corr_path = self.exact_root / "pairs" / short_id / "corrs.npz"
            np.savez(corr_path, synthetic=np.arange(candidate_count, dtype=np.int64))
            allowlist_path = self.exact_root / "pairs" / short_id / "allowlist.json"
            allowlist = {
                "schema": "v16-b716-frozen-hypothesis-allowlist-v1",
                "pair_id": pair_id,
                "all_hypotheses_must_be_replayed": True,
                "hypotheses": [{
                    "hypothesis_index": row["hypothesis_index"],
                    "hypothesis_sha256": row["hypothesis_sha256"],
                    "member_candidate_indices": [
                        hypothesis_candidate_indices[
                            row["hypothesis_index"]]],
                } for row in hypotheses],
            }
            allowlist_sha = seal_json(allowlist_path, allowlist)
            exact_pairs.append({
                "short_id": short_id,
                "pair_id": pair_id,
                "candidate_count": candidate_count,
                "existing_count": EXISTING_COUNTS[ordinal],
                "new_count": NEW_COUNTS[ordinal],
                "existing_failed_count": 0,
                "hypothesis_count": hypothesis_count,
                "entries_path": entries_path.relative_to(self.exact_root).as_posix(),
                "entries_bytes": entries_path.stat().st_size,
                "entries_sha256": entries_sha,
                "correspondences_path": corr_path.relative_to(self.exact_root).as_posix(),
                "correspondences_bytes": corr_path.stat().st_size,
                "correspondences_sha256": sha256_file(corr_path),
                "allowlist_path": allowlist_path.relative_to(self.exact_root).as_posix(),
                "allowlist_bytes": allowlist_path.stat().st_size,
                "allowlist_sha256": allowlist_sha,
            })
        candidate = {
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "official_release_domain_matched": True,
            "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "legacy_B_ep20_or_89ed_consumed": False,
            "candidate_count": 191,
            "hypothesis_count": 34,
            "pair_count": 4,
            "official92_executed": False,
            "geot_existing_ok": 103,
            "geot_existing_failed": 16,
            "geot_missing_disabled": 72,
            "pairs": candidate_pairs,
        }
        self.candidate_path = self.candidate_root / "fixed4_manifest.json"
        self.candidate_sha = seal_json(self.candidate_path, candidate)
        shutil.rmtree(self.exact_root)
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/v16_b716_synthetic_exact191_fixture.py"),
            "--candidate-manifest", str(self.candidate_path),
            "--candidate-manifest-sha256", self.candidate_sha,
            "--output-root", str(self.exact_root),
        ], check=True, cwd=ROOT)
        self.exact_path = self.exact_root / "exact191_manifest.json"
        self.exact_sha = sha256_file(self.exact_path)
        return
        exact = {
            "schema": EXACT191_SCHEMA,
            "sealed": True,
            "synthetic_test_fixture": True,
            "candidate_count": 191,
            "existing_count": 119,
            "new_authorized_count": 72,
            "existing_ok_count": 119,
            "existing_failed_count": 0,
            "hypothesis_count": 34,
            "consumer_scope": "only_the_34_frozen_hypotheses_across_fixed4",
            "candidate_selection_allowed": False,
            "result_based_selection_allowed": False,
            "hypothesis_selection_allowed": False,
            "gt_allowed": False,
            "official92_allowed": False,
            "new_geot_execution_performed_by_merger": False,
            "b716_domain_only": True,
            "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "fixed_hypothesis_distribution": list(HYPOTHESIS_COUNTS),
            "legacy_B_ep20_or_89ed_consumed": False,
            "candidate_manifest_sha256": self.candidate_sha,
            "preflight_manifest_sha256": "1" * 64,
            "preregister_sha256": "2" * 64,
            "authorization_sha256": "3" * 64,
            "batch_result_sha256": "4" * 64,
            "pairs": exact_pairs,
            "artifact_closure": [],
            "recursive_artifact_closure_sha256": "5" * 64,
        }
        self.exact_path = self.exact_root / "exact191_manifest.json"
        self.exact_sha = seal_json(self.exact_path, exact)

    def reseal_candidate_chain(self) -> None:
        candidate = json.loads(self.candidate_path.read_text())
        for row, plan_path in zip(candidate["pairs"], self.plan_paths):
            row["plan_sha256"] = sha256_file(plan_path)
        self.candidate_sha = seal_json(self.candidate_path, candidate)
        shutil.rmtree(self.exact_root)
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/v16_b716_synthetic_exact191_fixture.py"),
            "--candidate-manifest", str(self.candidate_path),
            "--candidate-manifest-sha256", self.candidate_sha,
            "--output-root", str(self.exact_root),
        ], check=True, cwd=ROOT)
        self.exact_path = self.exact_root / "exact191_manifest.json"
        self.exact_sha = sha256_file(self.exact_path)

    def build(self, output: Path, *, artifact_event_hook=None):
        return build_fixed4_prepared_inputs(
            candidate_manifest_path=self.candidate_path,
            candidate_manifest_sha256=self.candidate_sha,
            exact191_manifest_path=self.exact_path,
            exact191_manifest_sha256=self.exact_sha,
            output_root=output,
            raw_roots=[self.raw_root],
            canonical_builder=lambda pair_id: (self.data[pair_id], {}),
            raw_loader=lambda path, scan_id, side: load_raw_inseg(
                path, scan_id=scan_id, side=side),
            raw_resolver=resolve_unique_inseg_path,
            source_hashes={"synthetic.py": "6" * 64},
            allow_test_fixture=True,
            artifact_event_hook=artifact_event_hook,
        )


class V16B716PreparedBuilderTests(unittest.TestCase):
    def test_builds_exact34_v13_schema_and_replays_byte_identically(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            first = closure.build(Path(tmp) / "out1")
            second = closure.build(Path(tmp) / "out2")
            self.assertEqual(first["hypothesis_distribution"], [12, 8, 2, 12])
            self.assertEqual(first["hypothesis_count"], 34)
            self.assertEqual(first["existing_typed_failure_count"], 16)
            self.assertEqual(first["new_typed_failure_count"], 12)
            self.assertEqual(first["typed_failure_total_count"], 28)
            self.assertEqual(
                first["hypotheses_with_existing_typed_failure_members"], 8)
            self.assertEqual(first["hypotheses_with_typed_failure_members"], 10)
            self.assertEqual(first["artifact_manifest_sha256"],
                             second["artifact_manifest_sha256"])
            files = list((Path(tmp) / "out1").rglob("*.npz"))
            self.assertEqual(len(files), 34)
            typed_hypotheses = 0
            for pair in first["pairs"]:
                pair_manifest = json.loads((Path(tmp) / "out1" /
                    pair["pair_manifest_path"]).read_text())
                self.assertTrue(pair_manifest[
                    "typed_failure_members_visible_and_never_filtered"])
                typed_hypotheses += pair_manifest[
                    "hypotheses_with_typed_failure_members"]
            self.assertEqual(typed_hypotheses, 10)
            required = {
                f"{ARM}_source_xyz", f"{ARM}_source_labels",
                f"{ARM}_reference_xyz", f"{ARM}_reference_labels",
                f"{ARM}_source_voxel10_xyz",
                f"{ARM}_source_voxel10_colors_mean_0_255",
                f"{ARM}_reference_voxel10_xyz",
                f"{ARM}_reference_voxel10_colors_mean_0_255",
                "manifest_json",
            }
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    self.assertTrue(required.issubset(data.files))
                    self.assertTrue(np.array_equal(
                        data[f"{ARM}_source_labels"],
                        data[f"{ARM}_source_membership_object_ids"]))
                    manifest = json.loads(str(data["manifest_json"].item()))
                    self.assertEqual(manifest["schema"],
                                     "v13-color-preserving-pair-v2")
                    unsigned = dict(manifest)
                    payload = unsigned.pop("payload_sha256")
                    self.assertEqual(payload, stable_json_sha256(unsigned))
                    self.assertEqual(len(manifest["exact191_manifest_sha256"]), 64)
                    self.assertFalse(manifest["gt_consumed"])
                    self.assertFalse(manifest["selector_eligible"])
                    if manifest["contains_typed_failure_members"]:
                        self.assertFalse(manifest["safe_pose_vote_eligible"])
                surfaces = _load_surfaces(
                    path, manifest["pair_id"], ARM)
                self.assertGreater(len(surfaces["source"]), 0)
                self.assertGreater(len(surfaces["reference"]), 0)

    def test_synthetic_fixture_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp))
            with self.assertRaisesRegex(V16ContractError, "explicit test opt-in"):
                validate_candidate_and_exact191(
                    candidate_manifest_path=closure.candidate_path,
                    candidate_manifest_sha256=closure.candidate_sha,
                    exact191_manifest_path=closure.exact_path,
                    exact191_manifest_sha256=closure.exact_sha)

    def test_old_checkpoint_and_schema_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp))
            exact = json.loads(closure.exact_path.read_text())
            exact["official_release_checkpoint_sha256"] = \
                "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200"
            closure.exact_sha = seal_json(closure.exact_path, exact)
            with self.assertRaisesRegex(V16ContractError, "legacy B/89ed"):
                closure.build(Path(tmp) / "old")
            exact["official_release_checkpoint_sha256"] = OFFICIAL_RELEASE_SHA256
            exact["schema"] = "wrong-schema"
            closure.exact_sha = seal_json(closure.exact_path, exact)
            with self.assertRaisesRegex(V16ContractError, "contract mismatch"):
                closure.build(Path(tmp) / "schema")

    def test_missing_and_duplicate_hypotheses_fail(self):
        for mode in ("missing", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                closure = SyntheticClosure(Path(tmp))
                path = closure.plan_paths[0]
                plan = json.loads(path.read_text())
                if mode == "missing":
                    plan["hypotheses"].pop()
                else:
                    plan["hypotheses"][-1] = copy.deepcopy(plan["hypotheses"][0])
                seal_json(path, plan)
                closure.reseal_candidate_chain()
                with self.assertRaisesRegex(
                        V16ContractError, "hypothesis|sealed manifest"):
                    closure.build(Path(tmp) / "out")

    def test_path_collision_tampered_plan_exact_and_raw_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            output = Path(tmp) / "collision"
            output.mkdir()
            (output / "occupied").write_text("x")
            with self.assertRaisesRegex(V16ContractError, "output root must be empty"):
                closure.build(output)

            closure.plan_paths[0].write_text(
                closure.plan_paths[0].read_text() + "tamper")
            with self.assertRaisesRegex(V16ContractError, "source SHA mismatch"):
                closure.build(Path(tmp) / "plan-tamper")

        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            exact = json.loads(closure.exact_path.read_text())
            entries = closure.exact_root / exact["pairs"][0]["entries_path"]
            entries.write_text(entries.read_text() + "tamper")
            with self.assertRaisesRegex(V16ContractError, "bytes/SHA mismatch"):
                closure.build(Path(tmp) / "exact-tamper")

        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            src_scan = FIXED_PAIR_ORDER[0].split("_to_")[0]
            raw = closure.raw_root / src_scan / "inseg_cloud.npz"
            with np.load(raw, allow_pickle=False) as data:
                xyz, labels, colors = (np.asarray(data[key])
                                       for key in ("xyz", "labels", "colors"))
            xyz = xyz.copy(); xyz[0, 0] += 0.001
            np.savez(raw, xyz=xyz, labels=labels, colors=colors)
            with self.assertRaisesRegex(V16ContractError,
                                        "raw surface binding mismatch"):
                closure.build(Path(tmp) / "raw-tamper")

    def test_create_only_rejects_midrun_collision_and_tamper(self):
        for mode in ("collision", "tamper"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                closure = SyntheticClosure(Path(tmp) / "fixture")
                seen = {"count": 0, "foreign": None}
                def hook(stage, path):
                    if stage == "before_publish":
                        seen["count"] += 1
                        if mode == "collision" and seen["count"] == 3:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(b"foreign-create-only")
                            seen["foreign"] = path
                    elif stage == "after_publish" and mode == "tamper" \
                            and seen["count"] == 3:
                        path.write_bytes(b"foreign-tamper")
                with self.assertRaisesRegex(V16ContractError,
                                             "create-only artifact"):
                    closure.build(Path(tmp) / "out", artifact_event_hook=hook)
                self.assertTrue(seen["foreign"] is not None or mode == "tamper")

    def test_hardened_closure_missing_and_wrong_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            exact = json.loads(closure.exact_path.read_text())
            preflight_path = next(Path(row["path"]) for row in exact["input_closure"]
                                  if row["role"] == "authorized_preflight_manifest")
            preflight = json.loads(preflight_path.read_text())
            source_row = next(row for row in preflight["source_closure"]
                              if row["role"].startswith("immutable_runtime_source"))
            Path(source_row["path"]).unlink()
            with self.assertRaisesRegex(V16ContractError, "source closure"):
                closure.build(Path(tmp) / "missing-source")
        with tempfile.TemporaryDirectory() as tmp:
            closure = SyntheticClosure(Path(tmp) / "fixture")
            exact = json.loads(closure.exact_path.read_text())
            exact["execution_binding"]["task_closure_sha256"] = "f" * 64
            closure.exact_sha = seal_json(closure.exact_path, exact)
            with self.assertRaisesRegex(V16ContractError, "binding mismatch"):
                closure.build(Path(tmp) / "wrong-binding")

    def test_real_v14_conversion_lineage_accepts_prepared_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            closure = SyntheticClosure(root / "fixture")
            result = closure.build(root / "prepared")
            pair_manifest = json.loads((root / "prepared" /
                result["pairs"][0]["pair_manifest_path"]).read_text())
            record = pair_manifest["hypotheses"][0]
            prepared = (root / "prepared" / record["prepared_input_path"]).resolve()
            prepared_record = {
                "prepared_npz_path": str(prepared),
                "prepared_npz_sha256": record["prepared_input_sha256"],
                "payload_sha256": record["prepared_manifest_payload_sha256"],
            }
            pair_id = result["pairs"][0]["pair_id"]
            src = np.asarray([[0., 0., 0.], [1., 0., 0.]], np.float32)
            ref = src + np.asarray([0.1, 0., 0.], np.float32)
            scores = np.asarray([0.9, 0.8], np.float32)
            transform = np.eye(4, dtype=np.float64); transform[0, 3] = 0.1
            meta = {
                "schema": "v13-colorpcr-corr-cache-v2",
                "sentinel_invariant": True, "gt_consumed": False,
                "identity_fallback": False, "input_sha256": sha256_file(prepared),
                "worker_contract": {"arm": ARM, "direction": "forward",
                    "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
                    "sampling": "voxel10", "coarsest_cap": 512},
            }
            sentinel = root / "sentinel.npz"
            np.savez(sentinel, src_corr_points=src, ref_corr_points=ref,
                     corr_scores=scores, estimated_transform=transform,
                     meta_json=np.asarray(json.dumps(meta)))
            cache = root / "cache.npz"
            np.savez(cache, src_corr=src, ref_corr=ref, scores=scores)
            with np.load(cache, allow_pickle=False) as loaded_cache:
                cache_hashes = {key: v14_array_sha256(np.asarray(loaded_cache[key]))
                                for key in loaded_cache.files}
            sentinels = {}
            for name in ("identity", "proper_nonzero"):
                path = root / f"{name}.bin"; path.write_bytes(name.encode())
                sentinels[name] = path
            receipt = {
                "schema": "v13-colorpcr-corr-conversion-receipt-v1",
                "pair_id": pair_id, "arm": ARM, "direction": "forward",
                "prepared_input": str(prepared),
                "prepared_input_sha256": sha256_file(prepared),
                "prepared_manifest_payload_sha256":
                    record["prepared_manifest_payload_sha256"],
                "output_cache": str(cache),
                "output_cache_sha256": sha256_file(cache),
                "estimated_transform_discarded": True,
                "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
                "sampling": "voxel10", "coarsest_cap": 512,
                "gt_consumed": False, "fallback_used": False,
                "converter_sha256": v13_sources(ROOT)["converter"],
                "output_keys": ["src_corr", "ref_corr", "scores"],
                "output_array_sha256": cache_hashes,
                "source_sentinel_cache": str(sentinel),
                "source_sentinel_cache_sha256": sha256_file(sentinel),
                "source_array_sha256": {"src_corr_points": v14_array_sha256(src),
                    "ref_corr_points": v14_array_sha256(ref),
                    "corr_scores": v14_array_sha256(scores),
                    "estimated_transform": v14_array_sha256(transform)},
                "sentinel_artifact_path": {k: str(v) for k, v in sentinels.items()},
                "sentinel_artifact_sha256": {k: sha256_file(v)
                                              for k, v in sentinels.items()},
            }
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            verified = verify_conversion_lineage(
                repo=ROOT, cache_path=cache, receipt_path=receipt_path,
                prepared_path=prepared, prepared_record=prepared_record,
                pair_id=pair_id, arm=ARM, direction="forward")
            self.assertEqual(verified["cache_sha256"], sha256_file(cache))


if __name__ == "__main__":
    unittest.main()
