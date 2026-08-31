import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from safety.v13_dual_solver_runtime import (
    apply_transform, array_sha256, load_frozen_correspondences, sha256_file,
    stable_json_sha256,
)
from safety.v14_rigid_multihypothesis import (
    RigidMultiHypothesisConfig,
    RigidMultiHypothesisError,
    aggregate_fixed4_research,
    build_direction_candidates,
    generate_direction_hypotheses,
    load_candidate_contract,
    pair_bidirectional_hypotheses,
    seal_bidirectional_candidate_set,
    select_unique_safe_candidate,
    verify_candidate_set_contract,
)
from scripts.v13_corr_cache_converter import FROZEN_NEIGHBOR_LIMITS
from scripts.v13_formal_source_manifest import formal_source_sha256 as v13_sources
from scripts.v14_candidate_strict_runner import verify_v14_authorization
from scripts.v14_fixed4_input_builder import (
    Fixed4InputError, build_fixed4_inputs, verify_conversion_lineage,
)
from scripts.v14_fixed4_research_orchestrator import load_input_manifest
from scripts.v14_formal_source_manifest import (
    FORMAL_SOURCE_PATHS, formal_source_sha256,
)
from scripts.v14_rigid_multihypothesis_builder import (
    preregister as verify_builder_authorization,
)


def pose(angle_deg, translation):
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[:3, 3] = translation
    return value


def two_modes():
    rng = np.random.default_rng(1401)
    source_a = rng.uniform([-1.0, -0.8, -0.5], [0.8, 0.9, 0.7], (60, 3))
    source_b = rng.uniform([2.0, -0.7, -0.4], [3.8, 0.8, 0.9], (60, 3))
    transform_a = pose(17, [0.31, -0.14, 0.08])
    transform_b = pose(-31, [-0.42, 0.27, -0.11])
    reference_a = apply_transform(source_a, transform_a)
    reference_b = apply_transform(source_b, transform_b)
    source = np.empty((120, 3), np.float64)
    reference = np.empty((120, 3), np.float64)
    source[0::2], source[1::2] = source_a, source_b
    reference[0::2], reference[1::2] = reference_a, reference_b
    scores = np.linspace(1.0, 0.5, len(source), dtype=np.float64)
    return source, reference, scores, transform_a, transform_b


def three_modes():
    rng = np.random.default_rng(1402)
    transforms = (pose(11, [0.2, -0.1, 0.05]),
                  pose(-26, [-0.4, 0.3, 0.11]),
                  pose(43, [0.7, -0.25, -0.08]))
    source_parts, reference_parts = [], []
    for index, transform in enumerate(transforms):
        source = rng.uniform(-0.9, 0.9, (45, 3)) + [index * 3.0, 0, 0]
        source_parts.append(source)
        reference_parts.append(apply_transform(source, transform))
    source = np.stack(source_parts, axis=1).reshape(-1, 3)
    reference = np.stack(reference_parts, axis=1).reshape(-1, 3)
    return source, reference, np.ones(len(source), np.float64)


def sealed_pair(root: Path, preregister: Path) -> Path:
    source, reference, scores, _a, _b = two_modes()
    for direction, src, ref in (("forward", source, reference),
                                ("reverse", reference, source)):
        cache = root / f"{direction}.npz"
        np.savez(cache, src_corr=src, ref_corr=ref, scores=scores)
        build_direction_candidates(
            cache, root / direction, pair_id="pair",
            arm="sgf_selected_union", direction=direction,
            preregister_path=preregister)
    path = root / "paired.json"
    seal_bidirectional_candidate_set(
        root / "forward/manifest.json", root / "reverse/manifest.json",
        path, preregister)
    return path


def resign_coordinated_tamper(root: Path, paired: Path, mode: str) -> Path:
    manifest_path = root / "forward/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["hypotheses"][0]
    source = load_frozen_correspondences(Path(manifest["source_cache_path"]))
    support = np.asarray(row["support_original_indices"], dtype=np.int64)
    if mode == "wrong_order":
        support = support[::-1].copy()
    elif mode == "legal_other_subset":
        support = np.asarray([value for value in source.selected_original_indices
                              if int(value) not in set(support)][:len(support)],
                             dtype=np.int64)
    elif mode == "out_of_range":
        support = np.arange(10000, 10000 + len(support), dtype=np.int64)
    elif mode != "mutated_arrays":
        raise AssertionError(mode)
    if mode != "mutated_arrays":
        row["support_original_indices"] = [int(value) for value in support]
        row["support_original_indices_sha256"] = array_sha256(support)
        core = {key: item for key, item in row.items()
                if key not in {"hypothesis_sha256", "candidate_cache_path",
                               "candidate_cache_sha256", "support_indices_path",
                               "support_indices_sha256", "candidate_receipt_path",
                               "candidate_receipt_sha256"}}
        row["hypothesis_sha256"] = stable_json_sha256(core)
    indices_path = Path(row["support_indices_path"])
    np.save(indices_path, support, allow_pickle=False)
    cache_path = Path(row["candidate_cache_path"])
    if mode in ("wrong_order", "legal_other_subset"):
        mapping = {int(value): index for index, value in enumerate(
            source.selected_original_indices)}
        local = np.asarray([mapping[int(value)] for value in support])
        arrays = (source.src[local], source.ref[local], source.scores[local])
    else:
        arrays = (np.full((len(support), 3), 123.0),
                  np.full((len(support), 3), -77.0),
                  np.ones(len(support), dtype=np.float64))
    np.savez(cache_path, src_corr=arrays[0], ref_corr=arrays[1], scores=arrays[2])
    receipt_path = Path(row["candidate_receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt.update({
        "hypothesis_sha256": row["hypothesis_sha256"],
        "candidate_cache_sha256": sha256_file(cache_path),
        "support_indices_sha256": sha256_file(indices_path),
        "support_original_indices_sha256": array_sha256(support),
        "correspondence_count": len(support),
    })
    receipt.pop("payload_sha256", None)
    receipt["payload_sha256"] = stable_json_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    row.update({
        "candidate_cache_sha256": sha256_file(cache_path),
        "support_indices_sha256": sha256_file(indices_path),
        "candidate_receipt_sha256": sha256_file(receipt_path),
    })
    manifest.pop("payload_sha256", None)
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    preregister = Path(json.loads(paired.read_text())["preregister_path"])
    tampered_paired = root / "tampered-paired.json"
    seal_bidirectional_candidate_set(
        manifest_path, root / "reverse/manifest.json", tampered_paired,
        preregister)
    return tampered_paired


class Tests(unittest.TestCase):
    def test_two_modes_are_deterministic_and_keep_at_least_40(self):
        source, reference, scores, _a, _b = two_modes()
        first = generate_direction_hypotheses(
            source, reference, scores, source_cache_sha256="a" * 64,
            pair_id="pair", arm="sgf_selected_union", direction="forward")
        second = generate_direction_hypotheses(
            source, reference, scores, source_cache_sha256="a" * 64,
            pair_id="pair", arm="sgf_selected_union", direction="forward")
        self.assertGreaterEqual(len(first["hypotheses"]), 2)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertEqual(
            [row["support_original_indices"] for row in first["hypotheses"]],
            [row["support_original_indices"] for row in second["hypotheses"]])
        self.assertTrue(all(row["correspondence_count"] >= 40
                            for row in first["hypotheses"]))
        supports = [set(row["support_original_indices"])
                    for row in first["hypotheses"][:2]]
        self.assertLess(len(supports[0] & supports[1]), 10)
        self.assertEqual(first["gt_consumed"], False)
        self.assertEqual(first["fallback_used"], False)

    def test_direction_is_independent_and_bidirectional_pairing_is_geometric(self):
        source, reference, scores, _a, _b = two_modes()
        forward = generate_direction_hypotheses(
            source, reference, scores, source_cache_sha256="a" * 64,
            pair_id="pair", arm="sgf_selected_union", direction="forward")
        reverse = generate_direction_hypotheses(
            reference, source, scores[::-1].copy(),
            source_cache_sha256="b" * 64, pair_id="pair",
            arm="sgf_selected_union", direction="reverse")
        paired = pair_bidirectional_hypotheses(forward, reverse)
        self.assertGreaterEqual(len(paired["candidates"]), 2)
        self.assertTrue(all(row["forward_cache_sha256"] == "a" * 64
                            and row["reverse_cache_sha256"] == "b" * 64
                            for row in paired["candidates"]))
        self.assertTrue(all(row["rotation_deg"] <= 5.0
                            and row["translation_m"] <= 0.10
                            for row in paired["candidates"]))

    def test_hash_budget_recalls_three_modes_with_ties_and_row_permutation(self):
        source, reference, scores = three_modes()
        original = np.arange(len(source), dtype=np.int64)
        first = generate_direction_hypotheses(
            source, reference, scores, source_cache_sha256="3" * 64,
            pair_id="pair", arm="sgf_selected_union", direction="forward",
            original_indices=original)
        permutation = np.random.default_rng(44).permutation(len(source))
        shuffled = generate_direction_hypotheses(
            source[permutation], reference[permutation], scores[permutation],
            source_cache_sha256="4" * 64, pair_id="pair",
            arm="sgf_selected_union", direction="forward",
            original_indices=original[permutation])
        self.assertGreaterEqual(len(first["hypotheses"]), 3)
        self.assertGreaterEqual(len(shuffled["hypotheses"]), 3)
        for result in (first, shuffled):
            mode_ids = []
            for row in result["hypotheses"]:
                values = np.asarray(row["support_original_indices"])
                counts = [int(((values % 3) == mode).sum()) for mode in range(3)]
                mode_ids.append(int(np.argmax(counts)))
            self.assertEqual(set(mode_ids), {0, 1, 2})

    def test_fewer_than_40_and_non_exact_cache_fail_closed(self):
        points = np.arange(117, dtype=np.float64).reshape(39, 3) / 100
        with self.assertRaisesRegex(RigidMultiHypothesisError, "at least 40"):
            generate_direction_hypotheses(
                points, points, np.ones(39), source_cache_sha256="a" * 64,
                pair_id="pair", arm="sgf_selected_union", direction="forward")
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, src_corr=np.vstack([points, points[:1]]),
                     ref_corr=np.vstack([points, points[:1]]),
                     scores=np.ones(40), gt_transform=np.eye(4))
            with self.assertRaisesRegex(Exception, "exactly"):
                build_direction_candidates(
                    bad, Path(tmp) / "out", pair_id="pair",
                    arm="sgf_selected_union", direction="forward")

    def test_builder_writes_exact_solver_caches_and_index_receipts(self):
        source, reference, scores, _a, _b = two_modes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "source.npz"
            np.savez(cache, src_corr=source, ref_corr=reference, scores=scores)
            manifest = build_direction_candidates(
                cache, root / "out", pair_id="pair",
                arm="sgf_selected_union", direction="forward")
            self.assertEqual(manifest["source_cache_sha256"], sha256_file(cache))
            self.assertGreaterEqual(manifest["candidate_count"], 2)
            for row in manifest["hypotheses"]:
                candidate = Path(row["candidate_cache_path"])
                indices = Path(row["support_indices_path"])
                with np.load(candidate, allow_pickle=False) as data:
                    self.assertEqual(set(data.files),
                                     {"src_corr", "ref_corr", "scores"})
                    self.assertGreaterEqual(len(data["scores"]), 40)
                self.assertEqual(row["candidate_cache_sha256"],
                                 sha256_file(candidate))
                self.assertEqual(row["support_indices_sha256"],
                                 sha256_file(indices))
            sealed = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(sealed, manifest)

    def test_bidirectional_contract_rehashes_cache_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preregister = (Path(__file__).resolve().parents[1]
                           / "manifests/v14_rigid_multihypothesis_preregister.json")
            path = sealed_pair(root, preregister)
            contract = load_candidate_contract(path, 0)
            self.assertEqual(contract["candidate_index"], 0)
            self.assertEqual(set(contract["cache_sha256"]),
                             {"forward", "reverse"})
            candidate_cache = Path(
                contract["candidate"]["forward_candidate_cache_path"])
            with candidate_cache.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RigidMultiHypothesisError,
                                        "artifact closure"):
                load_candidate_contract(path, 0)

    def test_contract_rehashes_source_indices_and_direction_manifest(self):
        preregister = (Path(__file__).resolve().parents[1]
                       / "manifests/v14_rigid_multihypothesis_preregister.json")
        for artifact_name in ("source_cache_path", "support_indices_path"):
            with self.subTest(artifact_name=artifact_name), \
                    tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = sealed_pair(root, preregister)
                contract = load_candidate_contract(path, 0)
                receipt_path = Path(contract["candidate"][
                    "forward_candidate_receipt_path"])
                receipt = json.loads(receipt_path.read_text())
                with Path(receipt[artifact_name]).open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(RigidMultiHypothesisError,
                                            "mismatch|closure"):
                    load_candidate_contract(path, 0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = sealed_pair(root, preregister)
            manifest = root / "forward/manifest.json"
            with manifest.open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(RigidMultiHypothesisError,
                                        "manifest file closure"):
                load_candidate_contract(path, 0)

    def test_coordinated_resigning_cannot_replace_source_subset(self):
        preregister = (Path(__file__).resolve().parents[1]
                       / "manifests/v14_rigid_multihypothesis_preregister.json")
        for mode in ("out_of_range", "wrong_order", "legal_other_subset",
                     "mutated_arrays"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paired = sealed_pair(root, preregister)
                paired = resign_coordinated_tamper(root, paired, mode)
                with self.assertRaisesRegex(
                        RigidMultiHypothesisError,
                        "source-cache subset|deterministic hypothesis|support"):
                    load_candidate_contract(paired, 0)

    def test_zero_candidate_set_is_recursively_verified(self):
        preregister = (Path(__file__).resolve().parents[1]
                       / "manifests/v14_rigid_multihypothesis_preregister.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(19)
            source = rng.normal(size=(50, 3))
            reference = rng.normal(size=(50, 3)) * 5 + 20
            for direction, src, ref in (("forward", source, reference),
                                        ("reverse", reference, source)):
                cache = root / f"{direction}.npz"
                np.savez(cache, src_corr=src, ref_corr=ref,
                         scores=np.linspace(1, 0, 50, dtype=np.float64))
                build_direction_candidates(
                    cache, root / direction, pair_id="pair",
                    arm="sgf_selected_union", direction=direction,
                    preregister_path=preregister)
            paired = root / "paired.json"
            value = seal_bidirectional_candidate_set(
                root / "forward/manifest.json", root / "reverse/manifest.json",
                paired, preregister)
            self.assertEqual(value["candidate_count"], 0)
            verified = verify_candidate_set_contract(paired)
            self.assertEqual(verified["value"]["candidate_count"], 0)
            with (root / "forward.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RigidMultiHypothesisError,
                                        "source cache selection"):
                verify_candidate_set_contract(paired)

    def test_forward_reverse_cannot_share_one_source_cache(self):
        preregister = (Path(__file__).resolve().parents[1]
                       / "manifests/v14_rigid_multihypothesis_preregister.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, reference, scores, _a, _b = two_modes()
            cache = root / "shared.npz"
            np.savez(cache, src_corr=source, ref_corr=reference, scores=scores)
            for direction in ("forward", "reverse"):
                build_direction_candidates(
                    cache, root / direction, pair_id="pair",
                    arm="sgf_selected_union", direction=direction,
                    preregister_path=preregister)
            paired = root / "paired.json"
            seal_bidirectional_candidate_set(
                root / "forward/manifest.json", root / "reverse/manifest.json",
                paired, preregister)
            with self.assertRaisesRegex(RigidMultiHypothesisError,
                                        "must be distinct"):
                verify_candidate_set_contract(paired)

    def test_v13_converter_lineage_rejects_resigned_nonprojection(self):
        repo = Path(__file__).resolve().parents[1]
        pair_id, arm, direction = "a_to_b", "sgf_selected_union", "forward"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = root / "prepared.npz"
            pair_unsigned = {"schema": "v13-color-preserving-pair-v2",
                             "pair_id": pair_id}
            pair_manifest = {**pair_unsigned,
                             "payload_sha256": stable_json_sha256(pair_unsigned)}
            np.savez(prepared, manifest_json=np.asarray(
                json.dumps(pair_manifest)))
            record = {"pair_id": pair_id,
                      "prepared_npz_path": str(prepared.resolve()),
                      "prepared_npz_sha256": sha256_file(prepared),
                      "payload_sha256": pair_manifest["payload_sha256"]}
            identity, proper = root / "identity.npz", root / "proper.npz"
            identity.write_bytes(b"identity")
            proper.write_bytes(b"proper")
            rng = np.random.default_rng(8)
            src, ref = rng.normal(size=(45, 3)), rng.normal(size=(45, 3))
            scores = np.linspace(1, 0, 45, dtype=np.float64)
            estimated = np.eye(4)
            meta = {
                "schema": "v13-colorpcr-corr-cache-v2",
                "sentinel_invariant": True, "gt_consumed": False,
                "identity_fallback": False,
                "input_sha256": sha256_file(prepared),
                "worker_contract": {
                    "arm": arm, "direction": direction,
                    "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
                    "sampling": "voxel10", "coarsest_cap": 512,
                },
            }
            sentinel = root / "sentinel.npz"
            np.savez(sentinel, src_corr_points=src, ref_corr_points=ref,
                     corr_scores=scores, estimated_transform=estimated,
                     meta_json=np.asarray(json.dumps(meta)))
            cache = root / "forward.three_key.npz"
            np.savez(cache, src_corr=src, ref_corr=ref, scores=scores)
            receipt_path = root / "forward.three_key.receipt.json"
            receipt = {
                "schema": "v13-colorpcr-corr-conversion-receipt-v1",
                "pair_id": pair_id, "arm": arm, "direction": direction,
                "prepared_input": str(prepared.resolve()),
                "prepared_input_sha256": sha256_file(prepared),
                "prepared_manifest_payload_sha256": pair_manifest["payload_sha256"],
                "output_cache": str(cache.resolve()),
                "output_cache_sha256": sha256_file(cache),
                "estimated_transform_discarded": True,
                "neighbor_limits": FROZEN_NEIGHBOR_LIMITS,
                "sampling": "voxel10", "coarsest_cap": 512,
                "gt_consumed": False, "fallback_used": False,
                "converter_sha256": v13_sources(repo)["converter"],
                "output_keys": ["src_corr", "ref_corr", "scores"],
                "output_array_sha256": {
                    "src_corr": array_sha256(src), "ref_corr": array_sha256(ref),
                    "scores": array_sha256(scores)},
                "source_sentinel_cache": str(sentinel.resolve()),
                "source_sentinel_cache_sha256": sha256_file(sentinel),
                "source_array_sha256": {
                    "src_corr_points": array_sha256(src),
                    "ref_corr_points": array_sha256(ref),
                    "corr_scores": array_sha256(scores),
                    "estimated_transform": array_sha256(estimated)},
                "sentinel_artifact_path": {
                    "identity": str(identity.resolve()),
                    "proper_nonzero": str(proper.resolve())},
                "sentinel_artifact_sha256": {
                    "identity": sha256_file(identity),
                    "proper_nonzero": sha256_file(proper)},
            }
            receipt_path.write_text(json.dumps(receipt) + "\n")
            verified = verify_conversion_lineage(
                repo=repo, cache_path=cache, receipt_path=receipt_path,
                prepared_path=prepared, prepared_record=record,
                pair_id=pair_id, arm=arm, direction=direction)
            self.assertEqual(verified["cache_sha256"], sha256_file(cache))
            fake = np.full_like(src, 4.0)
            np.savez(cache, src_corr=fake, ref_corr=ref, scores=scores)
            receipt["output_cache_sha256"] = sha256_file(cache)
            receipt["output_array_sha256"]["src_corr"] = array_sha256(fake)
            receipt_path.write_text(json.dumps(receipt) + "\n")
            with self.assertRaisesRegex(Fixed4InputError,
                                        "not the sentinel projection"):
                verify_conversion_lineage(
                    repo=repo, cache_path=cache, receipt_path=receipt_path,
                    prepared_path=prepared, prepared_record=record,
                    pair_id=pair_id, arm=arm, direction=direction)

    def test_direction_manifest_payload_tamper_fails_closed(self):
        source, reference, scores, _a, _b = two_modes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preregister = (Path(__file__).resolve().parents[1]
                           / "manifests/v14_rigid_multihypothesis_preregister.json")
            for direction, src, ref in (("forward", source, reference),
                                        ("reverse", reference, source)):
                cache = root / f"{direction}.npz"
                np.savez(cache, src_corr=src, ref_corr=ref, scores=scores)
                build_direction_candidates(
                    cache, root / direction, pair_id="pair",
                    arm="sgf_selected_union", direction=direction,
                    preregister_path=preregister)
            forward_path = root / "forward/manifest.json"
            value = json.loads(forward_path.read_text())
            value["hypothesis_count"] += 1
            forward_path.write_text(json.dumps(value, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RigidMultiHypothesisError,
                                        "payload SHA"):
                seal_bidirectional_candidate_set(
                    forward_path, root / "reverse/manifest.json",
                    root / "paired.json", preregister)

    def test_unique_safe_candidate_and_ambiguity_fail_closed(self):
        base = {
            "schema": "v14-bidirectional-candidate-v1",
            "candidate_sha256": "c" * 64,
            "forward_candidate_cache_sha256": "f" * 64,
            "reverse_candidate_cache_sha256": "e" * 64,
            "forward_candidate_receipt_sha256": "1" * 64,
            "reverse_candidate_receipt_sha256": "2" * 64,
            "forward_candidate_cache_path": "/sealed/forward.npz",
            "reverse_candidate_cache_path": "/sealed/reverse.npz",
            "forward_candidate_receipt_path": "/sealed/forward.receipt.json",
            "reverse_candidate_receipt_path": "/sealed/reverse.receipt.json",
            "pair_id": "pair", "arm": "sgf_selected_union",
        }
        contract = {
            "candidate": base, "candidate_index": 0,
            "candidate_set_path": "/sealed/candidate_set.json",
            "candidate_set_sha256": "a" * 64,
        }
        strict = {
            "schema": "v13-strict-pair-gate-v1", "safe": True,
            "gate_authority": (
                "fixed_trace_icp_plus_unchanged_rule_b_plus_dual_solver_q4"),
            "cache_sha256": {"forward": "f" * 64, "reverse": "e" * 64},
            "candidate_receipt_sha256": {"forward": "1" * 64,
                                            "reverse": "2" * 64},
            "candidate_receipt_path": {
                "forward": "/sealed/forward.receipt.json",
                "reverse": "/sealed/reverse.receipt.json"},
            "candidate_cache_path": {
                "forward": "/sealed/forward.npz",
                "reverse": "/sealed/reverse.npz"},
            "candidate_sha256": "c" * 64, "candidate_index": 0,
            "candidate_set_path": "/sealed/candidate_set.json",
            "candidate_set_sha256": "a" * 64,
            "pair_id": "pair", "arm": "sgf_selected_union",
            "gt_consumed": False, "fallback_used": False,
        }
        one = select_unique_safe_candidate([(contract, strict)], known_bad=False)
        self.assertTrue(one["accepted"])
        second = dict(base, candidate_sha256="d" * 64)
        second_contract = dict(contract, candidate=second, candidate_index=1)
        second_strict = dict(strict, candidate_sha256="d" * 64,
                             candidate_index=1)
        ambiguous = select_unique_safe_candidate(
            [(contract, strict), (second_contract, second_strict)], known_bad=False)
        self.assertFalse(ambiguous["accepted"])
        self.assertEqual(ambiguous["reason"],
                         "ambiguous_multiple_safe_candidates")
        veto = select_unique_safe_candidate([(contract, strict)], known_bad=True)
        self.assertFalse(veto["accepted"])
        self.assertEqual(veto["reason"], "known_bad_veto")
        broken = dict(strict, cache_sha256={"forward": "0" * 64,
                                            "reverse": "e" * 64})
        self.assertEqual(select_unique_safe_candidate(
            [(contract, broken)], known_bad=False)["reason"], "no_safe_candidate")
        for field in ("candidate_sha256", "candidate_index",
                      "candidate_set_path", "candidate_set_sha256",
                      "pair_id", "arm", "candidate_receipt_path",
                      "candidate_cache_path"):
            with self.subTest(missing_strict_identity=field):
                incomplete = dict(strict)
                incomplete.pop(field)
                self.assertEqual(select_unique_safe_candidate(
                    [(contract, incomplete)], known_bad=False)["reason"],
                    "no_safe_candidate")

    def test_config_rejects_threshold_or_resource_mutation(self):
        with self.assertRaisesRegex(RigidMultiHypothesisError, "frozen"):
            generate_direction_hypotheses(
                *two_modes()[:3], source_cache_sha256="a" * 64,
                pair_id="pair", arm="sgf_selected_union", direction="forward",
                config=RigidMultiHypothesisConfig(residual_threshold_m=0.11))

    def test_current_preregister_mechanically_blocks_real_pilot(self):
        preregister = (Path(__file__).resolve().parents[1]
                       / "manifests/v14_rigid_multihypothesis_preregister.json")
        value = json.loads(preregister.read_text())
        contract = {
            "candidate": {"pair_id": value["fixed_pair_order"][0],
                          "arm": value["primary_arm"]},
            "preregister_path": str(preregister.resolve()),
            "preregister_sha256": sha256_file(preregister),
        }
        with self.assertRaisesRegex(
                RuntimeError, "not explicitly authorized|missing or stale"):
            verify_v14_authorization(preregister, contract)
        with self.assertRaisesRegex(
                RuntimeError, "not explicitly authorized|missing or stale"):
            verify_builder_authorization(preregister)

    def test_fixed4_input_builder_rejects_candidate_lineage_misbinding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = ["a", "b", "c", "bad"]
            preregister = root / "v14.json"
            preregister.write_text(json.dumps({
                "schema": "v14-rigid-multihypothesis-preregister-v1",
                "allow_real_pilot": True,
                "fixed_pair_order": pairs,
                "primary_arm": "sgf_selected_union",
                "control_arm": "fullscan",
            }, sort_keys=True) + "\n")
            v13_preregister = root / "v13.json"
            v13_preregister.write_text("{}\n")
            prepared = root / "prepared.npz"
            np.savez(prepared, dummy=np.ones(1))
            preflight_unsigned = {
                "schema": "v13-colorpcr-pointdsc-shadow-v2",
                "pair_ids": pairs,
                "pairs": [{"pair_id": pair_id,
                           "prepared_npz_path": str(prepared)}
                          for pair_id in pairs],
            }
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({
                **preflight_unsigned,
                "payload_sha256": stable_json_sha256(preflight_unsigned),
            }, sort_keys=True) + "\n")
            candidate = (root / "candidates" / "pairs" / "a"
                         / "sgf_selected_union" / "candidate_set.json")
            candidate.parent.mkdir(parents=True)
            candidate.write_text("{}\n")

            def fake_lineage(**kwargs):
                direction = kwargs["direction"]
                return {
                    "cache_path": f"/sealed/{direction}.npz",
                    "cache_sha256": ("f" if direction == "forward" else "e") * 64,
                    "source_sentinel_cache_sha256": (
                        "1" if direction == "forward" else "2") * 64,
                }

            verified = {
                "value": {"pair_id": "a", "arm": "sgf_selected_union"},
                "direction_manifests": {
                    "forward": {"source_cache_path": "/wrong/forward.npz",
                                "source_cache_sha256": "f" * 64},
                    "reverse": {"source_cache_path": "/sealed/reverse.npz",
                                "source_cache_sha256": "e" * 64},
                },
            }
            with mock.patch(
                    "scripts.v14_fixed4_input_builder.verify_reviewed_source_authorization"), \
                    mock.patch(
                        "scripts.v14_fixed4_input_builder.verify_v13_fixed4_root",
                        return_value={"pair_receipts": {}}), \
                    mock.patch(
                        "scripts.v14_fixed4_input_builder.verify_conversion_lineage",
                        side_effect=fake_lineage), \
                    mock.patch(
                        "scripts.v14_fixed4_input_builder.verify_candidate_set_contract",
                        return_value=verified):
                with self.assertRaisesRegex(
                        Fixed4InputError, "differs from V13 conversion lineage"):
                    build_fixed4_inputs(
                        repo=root, v13_root=root / "v13-root",
                        candidate_root=root / "candidates",
                        v13_preregister=v13_preregister,
                        v14_preregister=preregister, preflight=preflight,
                        output=root / "inputs.json")

    def test_fixed4_aggregate_primary_only_and_ambiguity_fail_closed(self):
        preregister = json.loads((Path(__file__).resolve().parents[1]
                                  / "manifests/v14_rigid_multihypothesis_preregister.json").read_text())
        rows = []
        for pair_id in preregister["fixed_pair_order"]:
            for arm in (preregister["primary_arm"], preregister["control_arm"]):
                if pair_id == preregister["known_bad_pair_id"]:
                    decision = {"accepted": False, "reason": "known_bad_veto"}
                elif arm == preregister["primary_arm"]:
                    decision = {"accepted": True, "reason": "unique_safe_candidate"}
                else:
                    decision = {"accepted": False, "reason": "no_safe_candidate"}
                rows.append({"pair_id": pair_id, "arm": arm,
                             "decision": decision})
        passed = aggregate_fixed4_research(rows, preregister)
        self.assertTrue(passed["safe"])
        self.assertFalse(passed["control_can_rescue"])
        ambiguous = json.loads(json.dumps(rows))
        ambiguous[0]["decision"] = {
            "accepted": False, "reason": "ambiguous_multiple_safe_candidates"}
        failed = aggregate_fixed4_research(ambiguous, preregister)
        self.assertFalse(failed["safe"])
        self.assertEqual(failed["reason"], "ambiguous_safe_candidates")
        with self.assertRaisesRegex(RigidMultiHypothesisError, "ordered 4x2"):
            aggregate_fixed4_research(list(reversed(rows)), preregister)

    def test_fixed4_input_manifest_binds_all_formal_sources_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            for relative in FORMAL_SOURCE_PATHS.values():
                source = repo / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"source\n")
            v13, preflight = root / "v13.json", root / "preflight.json"
            v13.write_text("{}\n")
            preflight.write_text(json.dumps({
                "pairs": [{"pair_id": pair_id}
                          for pair_id in ("a", "b", "c", "bad")],
            }, sort_keys=True) + "\n")
            v14 = root / "v14.json"
            preregister = {
                "schema": "v14-rigid-multihypothesis-preregister-v1",
                "fixed_pair_order": ["a", "b", "c", "bad"],
                "known_bad_pair_id": "bad",
                "primary_arm": "sgf_selected_union", "control_arm": "fullscan",
                "v13_preregister_sha256": sha256_file(v13),
                "preflight_manifest_sha256": sha256_file(preflight),
                "reviewed_formal_source_sha256": formal_source_sha256(repo),
            }
            v14.write_text(json.dumps(preregister, sort_keys=True) + "\n")
            rows = []
            for pair_id in preregister["fixed_pair_order"]:
                for arm in (preregister["primary_arm"],
                            preregister["control_arm"]):
                    prepared = root / f"{pair_id}-{arm}.npz"
                    prepared.write_bytes(b"prepared")
                    candidate_set = root / f"{pair_id}-{arm}.json"
                    candidate_set.write_text(json.dumps({
                        "pair_id": pair_id, "arm": arm, "candidate_count": 0,
                        "candidates": [], "preregister_sha256": sha256_file(v14),
                    }, sort_keys=True) + "\n")
                    rows.append({
                        "pair_id": pair_id, "arm": arm,
                        "candidate_set_path": str(candidate_set),
                        "candidate_set_sha256": sha256_file(candidate_set),
                        "prepared_input_path": str(prepared),
                        "prepared_input_sha256": sha256_file(prepared),
                        "direction_lineage": {
                            direction: {
                                "cache_path": str(root / f"{pair_id}-{arm}-{direction}.npz"),
                                "cache_sha256": f"{direction}-cache-sha",
                                "conversion_receipt_path": str(
                                    root / f"{pair_id}-{arm}-{direction}.receipt.json"),
                            }
                            for direction in ("forward", "reverse")
                        },
                        "v13_pair_receipt": {
                            "pair_id": pair_id, "arm": arm,
                            "receipt_sha256": f"{pair_id}-{arm}-receipt",
                        },
                    })
            v13_root_binding = {
                "root": str((root / "v13-root").resolve()),
                "closure_sha256": "closure-sha",
                "artifact_manifest_sha256": "artifact-sha",
            }
            unsigned = {
                "schema": "v14-fixed4-candidate-inputs-v1",
                "v14_preregister_path": str(v14.resolve()),
                "v14_preregister_sha256": sha256_file(v14),
                "v13_preregister_path": str(v13.resolve()),
                "v13_preregister_sha256": sha256_file(v13),
                "preflight_manifest_path": str(preflight.resolve()),
                "preflight_manifest_sha256": sha256_file(preflight),
                "formal_source_sha256": formal_source_sha256(repo),
                "v13_fixed4_binding": v13_root_binding,
                "rows": rows,
            }
            manifest = root / "inputs.json"
            manifest.write_text(json.dumps({
                **unsigned, "payload_sha256": stable_json_sha256(unsigned),
            }, sort_keys=True) + "\n")
            def fake_set(path):
                row = next(item for item in rows
                           if Path(item["candidate_set_path"]) == Path(path))
                value = json.loads(Path(path).read_text())
                return {
                    "value": value,
                    "direction_manifests": {
                        direction: {
                            "source_cache_path": row["direction_lineage"][direction]["cache_path"],
                            "source_cache_sha256": row["direction_lineage"][direction]["cache_sha256"],
                        }
                        for direction in ("forward", "reverse")
                    },
                }

            def fake_lineage(**kwargs):
                row = next(item for item in rows
                           if item["pair_id"] == kwargs["pair_id"]
                           and item["arm"] == kwargs["arm"])
                return row["direction_lineage"][kwargs["direction"]]

            v13_verified = {
                **v13_root_binding,
                "pair_receipts": {
                    (row["pair_id"], row["arm"]): row["v13_pair_receipt"]
                    for row in rows
                },
            }
            patches = (
                mock.patch(
                    "scripts.v14_fixed4_research_orchestrator.verify_v13_fixed4_root",
                    return_value=v13_verified),
                mock.patch(
                    "scripts.v14_fixed4_research_orchestrator.verify_candidate_set_contract",
                    side_effect=fake_set),
                mock.patch(
                    "scripts.v14_fixed4_research_orchestrator.verify_conversion_lineage",
                    side_effect=fake_lineage),
            )
            with patches[0], patches[1], patches[2]:
                loaded = load_input_manifest(
                    manifest, preregister, preregister_path=v14,
                    v13_preregister_path=v13, preflight_manifest_path=preflight,
                    repo=repo)
            self.assertEqual(len(loaded["rows"]), 8)
            for name, relative in FORMAL_SOURCE_PATHS.items():
                with self.subTest(formal_source=name):
                    source = repo / relative
                    source.write_bytes(b"tamper\n")
                    with patches[0], patches[1], patches[2]:
                        with self.assertRaisesRegex(RuntimeError,
                                                    "formal source mismatch"):
                            load_input_manifest(
                                manifest, preregister, preregister_path=v14,
                                v13_preregister_path=v13,
                                preflight_manifest_path=preflight, repo=repo)
                    source.write_bytes(b"source\n")
            with Path(rows[0]["prepared_input_path"]).open("ab") as stream:
                stream.write(b"tamper")
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(RuntimeError, "artifact closure"):
                    load_input_manifest(
                        manifest, preregister, preregister_path=v14,
                        v13_preregister_path=v13,
                        preflight_manifest_path=preflight, repo=repo)


if __name__ == "__main__":
    unittest.main()
