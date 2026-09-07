from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, load_manifest, load_trajectory,
    sha256_file, write_manifest, write_trajectory,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_unified_pose_replay.py"
SPEC = importlib.util.spec_from_file_location("validate_unified_pose_replay", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def pose(frame: int, x: float = 0.0) -> PoseRecord:
    matrix = np.eye(4)
    matrix[0, 3] = x
    return PoseRecord(frame, frame * 1000, matrix, source="DPV")


class ReplayContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {}
        frames = []
        for frame in (0, 1):
            color = self.root / "color" / f"{frame}.png"
            depth = self.root / "depth" / f"{frame}.png"
            for path in (color, depth):
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"tiny fixture, not decoded by contract test")
            frames.append(FrameRecord(frame, frame * 1000, color, depth, (1, 1, 0, 0)))
        self.manifest = SequenceManifest("scannet", "fixture", self.root, 1000,
                                         tuple(frames), "recorded_rgbd")
        self.baseline = [pose(0), pose(1)]
        self.paths["manifest"] = self.root / "tracked_manifest.json"
        self.paths["trajectory"] = self.root / "baseline_trajectory.json"
        write_manifest(self.paths["manifest"], self.manifest)
        write_trajectory(self.paths["trajectory"], self.baseline,
                         sequence_id="fixture", arm="dpv")
        self.registration = {
            "sequence_id": "fixture", "gt_consumed": False,
            "rows": [{"source_anchor_index": 0, "target_anchor_index": 1,
                      "source_frame_id": 0, "target_frame_id": 1,
                      "arms": {replay.REGISTRATION_ARM: {
                          "accepted": True, "transform": np.eye(4).tolist(),
                          "forward": {"verification": {"minimum_overlap": .5}}}}}],
        }
        self.paths["registration_inference"] = self.root / "registration.json"
        replay.write_json(self.paths["registration_inference"], self.registration)
        self.visual = {
            "sequence_id": "fixture", "gt_consumed": False,
            "registration_arm": replay.REGISTRATION_ARM,
            "registration_inference_sha256": sha256_file(self.paths["registration_inference"]),
            "rows": [{"registration_row_index": 0, "accepted": {"recall": True},
                      "source_frame_id": 0, "target_frame_id": 1,
                      "registration_transform": np.eye(4).tolist(),
                      "rgbd_inputs_sha256": {
                          "source_color": sha256_file(frames[0].color_path),
                          "source_depth": sha256_file(frames[0].depth_path),
                          "target_color": sha256_file(frames[1].color_path),
                          "target_depth": sha256_file(frames[1].depth_path),
                      }}],
        }
        self.evidence = {"sequence_id": "fixture", "gt_consumed": False,
                         "anchors": [{"anchor_ordinal": 0, "anchor_frame_id": 0},
                                     {"anchor_ordinal": 1, "anchor_frame_id": 1}]}
        for name, value in (("visual_inference", self.visual), ("loop_evidence", self.evidence)):
            self.paths[name] = self.root / f"{name}.json"
            replay.write_json(self.paths[name], value)
        self.paths["baseline_cloud"] = self.root / "baseline.ply"
        self.paths["baseline_cloud"].write_bytes(b"sealed baseline cloud")
        self.receipt = {
            "status": "completed", "gt_consumed": False, "identity_fallback_used": False,
            "trajectory_sha256": sha256_file(self.paths["trajectory"]),
            "manifest_sha256": sha256_file(self.paths["manifest"]),
            "cloud_sha256": sha256_file(self.paths["baseline_cloud"]),
            "integrated_frame_count": 2, "requested_frame_count": 2,
            "trajectory_pose_count": 2, "voxel_length_m": .02,
            "sdf_trunc_m": .08, "depth_trunc_m": 4.5,
        }
        self.paths["baseline_refusion_receipt"] = self.root / "receipt.json"
        replay.write_json(self.paths["baseline_refusion_receipt"], self.receipt)
        self.scene = {"scene_id": "fixture", **{
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in self.paths.items()}}

    def edges(self):
        return replay.frozen_pnp_edges(
            self.registration, self.visual, self.evidence, self.baseline,
            sha256_file(self.paths["registration_inference"]), "fixture")

    def test_rejects_manifest_superset_and_reordering(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            replay.validate_exact_coverage(self.manifest, self.baseline[:1])
        with self.assertRaisesRegex(ValueError, "exactly"):
            replay.validate_exact_coverage(self.manifest, self.baseline[::-1])
        replay.validate_exact_coverage(self.manifest, self.baseline)

    def test_frozen_hash_catches_mutation(self):
        replay.resolve_bound_inputs(self.scene, self.root)
        self.paths["baseline_cloud"].write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            replay.resolve_bound_inputs(self.scene, self.root)

    def test_receipt_requires_exact_input_and_complete_fusion(self):
        replay.validate_baseline_receipt(self.receipt, self.paths, 2)
        for key, value in (("manifest_sha256", "0" * 64),
                           ("integrated_frame_count", 1),
                           ("identity_fallback_used", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                replay.validate_baseline_receipt({**self.receipt, key: value}, self.paths, 2)

    def test_rejects_nested_gt_and_foreign_visual_registration(self):
        self.registration["rows"][0]["arms"][replay.REGISTRATION_ARM]["gt_consumed"] = True
        with self.assertRaisesRegex(ValueError, "GT consumption"):
            self.edges()
        del self.registration["rows"][0]["arms"][replay.REGISTRATION_ARM]["gt_consumed"]
        self.visual["registration_inference_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "visual-to-registration"):
            self.edges()

    def test_rejects_visual_selection_outside_accepted_registration(self):
        self.registration["rows"][0]["arms"][replay.REGISTRATION_ARM]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "outside fallback accepts"):
            self.edges()

    def test_rejects_duplicate_selection_and_anchor_mismatch(self):
        self.visual["rows"].append(self.visual["rows"][0])
        with self.assertRaisesRegex(ValueError, "duplicate visual"):
            self.edges()
        self.visual["rows"].pop()
        self.evidence["anchors"][1]["anchor_frame_id"] = 99
        with self.assertRaisesRegex(ValueError, "anchor-to-baseline"):
            self.edges()

    def test_arms_keep_causal_factors_and_strict_defaults(self):
        original, original_correction, _ = replay.arm_configuration("original_pnp", {})
        weight, weight_correction, _ = replay.arm_configuration("weight_only", {})
        correction, correction_audit, _ = replay.arm_configuration("correction_only", {})
        combined, combined_audit, _ = replay.arm_configuration("combined_bounded", {})
        loo, _, _ = replay.arm_configuration("combined_loo", {})
        self.assertFalse(original.enabled)
        self.assertIsNone(original_correction.maximum_absolute_correction_translation_m)
        self.assertEqual(weight.correction_backtracking_scales, (1.0,))
        self.assertIsNone(weight_correction.maximum_absolute_correction_translation_m)
        self.assertIsNone(correction.maximum_loop_weight)
        self.assertIsNone(correction.high_leverage_min_span_fraction)
        self.assertEqual(correction_audit.maximum_absolute_correction_translation_m, .25)
        self.assertEqual(combined_audit.maximum_absolute_correction_rotation_deg, 5)
        self.assertFalse(combined.enforce_leave_one_out)
        self.assertTrue(loo.enforce_leave_one_out)

    def test_json_settings_preserve_declared_local_policy_and_scale_grid(self):
        settings = json.loads('{"bounded": {"correction_scaling_policy": '
                              '"smooth_local", "correction_backtracking_scales": '
                              '[1.0, 0.5, 0.25]}}')
        bounded, _, _ = replay.arm_configuration("combined_bounded", settings)
        self.assertEqual(bounded.correction_scaling_policy, "smooth_local")
        self.assertEqual(bounded.correction_backtracking_scales, (1.0, .5, .25))

    def test_create_only_copy_and_json(self):
        destination = self.root / "copy.ply"
        replay.copy_exact(self.paths["baseline_cloud"], destination)
        with self.assertRaises(FileExistsError):
            replay.copy_exact(self.paths["baseline_cloud"], destination)
        with self.assertRaises(FileExistsError):
            replay.write_json(self.paths["baseline_refusion_receipt"], {})

    def run_mocked_scene(self, *, improvement: bool, large_correction: bool = False,
                         no_op: bool = False):
        corrected = [pose(0), pose(1, .5 if large_correction else .01)]
        geometry = {"passes_scene_safety": True,
                    "passes_scene_improvement": improvement}
        calls = []

        def refusion(request):
            calls.append(request)
            admitted = load_manifest(request.manifest)
            poses, _ = load_trajectory(request.trajectory)
            self.assertEqual([p.frame_id for p in poses], [f.frame_id for f in admitted.frames])
            request.output_dir.mkdir()
            cloud = request.output_dir / "refused.ply"
            cloud.write_bytes(b"new candidate geometry")
            return {"integrated_frame_count": len(poses), "cloud": str(cloud),
                    "cloud_sha256": sha256_file(cloud)}

        with patch("pose_pipeline.bounded_backend.optimize_bounded_trajectory",
                   return_value=(corrected, {"success": True, "selected_correction_scale": 1.0,
                                             "no_op": no_op, "applied_loop_count": 0 if no_op else 1})), \
             patch("pose_pipeline.geometry_metrics.ply_geometry_metrics", return_value={}), \
             patch("pose_pipeline.geometry_metrics.compare_no_gt_geometry_v2", return_value=geometry), \
             patch("reconstruction.rgbd_refusion.run_full_rgbd_refusion", side_effect=refusion):
            result = replay.infer_scene(self.scene, self.root, self.root / "out",
                                        ["combined_bounded"], {})
        self.assertEqual(len(calls), 1)
        return result["arms"]["combined_bounded"], self.root / "out/fixture/combined_bounded"

    def test_safe_but_not_improved_fuses_then_rolls_back_exact_bytes(self):
        result, directory = self.run_mocked_scene(improvement=False)
        self.assertFalse(result["committed_candidate"])
        self.assertTrue(result["dpv_rollback_byte_identical"])
        self.assertEqual((directory / "committed_trajectory.json").read_bytes(),
                         self.paths["trajectory"].read_bytes())
        self.assertNotEqual((directory / "candidate_trajectory.json").read_bytes(),
                            self.paths["trajectory"].read_bytes())
        self.assertTrue((directory / "candidate_refusion/refused.ply").is_file())

    def test_strict_correction_guard_rejects_even_if_geometry_improves(self):
        result, _ = self.run_mocked_scene(improvement=True, large_correction=True)
        self.assertFalse(result["strict_correction_guard"]["passes"])
        self.assertTrue(result["dpv_rollback_byte_identical"])

    def test_no_op_does_not_claim_applied_correction(self):
        result, _ = self.run_mocked_scene(improvement=True, no_op=True)
        self.assertFalse(result["correction_applied"])
        self.assertFalse(result["committed_candidate"])
        self.assertTrue(result["dpv_rollback_byte_identical"])

    def test_visual_transform_and_rgbd_content_are_bound(self):
        self.visual["rows"][0]["registration_transform"][0][3] = 1.0
        with self.assertRaisesRegex(ValueError, "transform mismatch"):
            self.edges()
        self.visual["rows"][0]["registration_transform"][0][3] = 0.0
        self.manifest.frames[0].color_path.write_bytes(b"changed color input")
        with self.assertRaisesRegex(ValueError, "RGB-D input digest"):
            replay.frozen_pnp_edges(
                self.registration, self.visual, self.evidence, self.baseline,
                sha256_file(self.paths["registration_inference"]), "fixture", self.manifest)

    def test_all_gates_pass_commits_candidate_exact_bytes(self):
        result, directory = self.run_mocked_scene(improvement=True)
        self.assertTrue(result["committed_candidate"])
        self.assertEqual((directory / "committed_trajectory.json").read_bytes(),
                         (directory / "candidate_trajectory.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
