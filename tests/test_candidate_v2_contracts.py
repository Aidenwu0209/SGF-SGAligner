from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_pipeline.adapters import scan3r_manifest
from pose_pipeline.contracts import FrameRecord, PoseRecord, write_manifest
from pose_pipeline.evaluation import scan3r_reference_trajectory
from pose_pipeline.geometry_backend import _point_information
from pose_pipeline.geometry_metrics import compare_no_gt_geometry_v2
from pose_pipeline.runner import (
    PrecommitGeometryConfig, precommit_geometry_decision,
)
from pose_pipeline.robust_backend import (
    decide_registration_v3,
)
from pose_pipeline.submaps import LoopProposalConfig, propose_loop_pairs


def _pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


class HybridProposalTests(unittest.TestCase):
    def test_clip_can_recover_a_pair_outside_drift_radius_deterministically(self):
        bound = []
        for index in range(6):
            frame = FrameRecord(
                index, index, Path(f"/{index}.jpg"), Path(f"/{index}.png"),
                (500.0, 500.0, 10.0, 10.0),
            )
            bound.append((frame, PoseRecord(index, index, _pose(index * 10.0))))
        descriptors = np.eye(6, dtype=float)
        descriptors[5] = descriptors[0]
        config = LoopProposalConfig(
            policy="hybrid36", minimum_anchor_gap=2,
            maximum_initial_distance_m=1.0, maximum_pairs=4,
            appearance_mutual_top_k=1, maximum_pairs_per_anchor=2,
        )
        first = propose_loop_pairs(
            bound, list(range(6)), config,
            appearance_descriptors=descriptors,
        )
        second = propose_loop_pairs(
            bound, list(range(6)), config,
            appearance_descriptors=descriptors,
        )
        self.assertEqual(first, second)
        pair = next(row for row in first if (
            row["source_anchor_index"], row["target_anchor_index"]
        ) == (0, 5))
        self.assertGreater(pair["initial_centre_distance_m"], 1.0)
        self.assertEqual(pair["proposal_sources"], ["clip_mutual_topk"])
        self.assertLessEqual(len(first), 4)


class RegistrationDecisionV3Tests(unittest.TestCase):
    def _inputs(self):
        consensus = {
            "accepted": True,
            "reason": "fixture",
            "selected_transform": np.eye(4).tolist(),
        }
        metrics = {
            "spatial_extent_m": 5.0,
            "spatial_second_axis_m": 2.0,
            "spatial_third_axis_m": 1.0,
            "icp_update_translation_m": 0.01,
            "icp_update_rotation_deg": 0.1,
            "bidirectional_translation_m": 0.01,
            "bidirectional_rotation_deg": 0.1,
            "cycle_translation_m": 0.01,
            "cycle_rotation_deg": 0.1,
            "overlap_ratio": 0.8,
            "forward_overlap": 0.8,
            "reverse_overlap": 0.7,
            "trimmed_rmse_m": 0.02,
            "correspondence_confidence": 0.5,
            "information_condition_number": 1.0,
            "information_matrix": np.eye(6).tolist(),
        }
        return consensus, metrics

    def test_v3_accepts_complete_non_degenerate_evidence(self):
        consensus, metrics = self._inputs()
        result = decide_registration_v3(consensus, metrics)
        self.assertTrue(result["usable_for_reconstruction"])
        self.assertEqual(result["schema"], "registration_decision.v3")

    def test_v3_rejects_trimmed_residual_and_degeneracy(self):
        consensus, metrics = self._inputs()
        metrics.update(trimmed_rmse_m=0.20, spatial_third_axis_m=0.0)
        result = decide_registration_v3(consensus, metrics)
        self.assertFalse(result["usable_for_reconstruction"])
        self.assertIn("trimmed_rmse_too_large", result["rejection_reasons"])
        self.assertIn("spatial_third_axis_too_small", result["rejection_reasons"])


class GeometryComparisonV2Tests(unittest.TestCase):
    def _metrics(self, *, voxels=1000, extent=(2, 2, 2), tilt=1.0,
                 thickness=0.02, conflict=0.20):
        return {
            "vertices": 1000,
            "occupied_voxels_2cm": voxels,
            "bbox_extent_m": list(extent),
            "robust_extent_p99_p01_m": list(extent),
            "near_parallel_layer_conflict_ratio": conflict,
            "horizontal_planes": [{
                "plane_id": "floor",
                "points": 900,
                "normal": [0.0, 1.0, 0.0],
                "centroid_m": [0.0, 0.0, 0.0],
                "tilt_from_gravity_deg": tilt,
                "thickness_p90_p10_m": thickness,
            }],
        }

    def test_all_precommit_geometry_gates_bind_matched_plane(self):
        result = compare_no_gt_geometry_v2(
            self._metrics(), self._metrics(voxels=900, extent=(1.8, 1.8, 1.8),
                                           tilt=2.5, thickness=0.021),
            admitted_frame_sha256="a" * 64,
            common_visibility_mask_sha256="b" * 64,
        )
        self.assertTrue(result["passes_scene_safety"])
        self.assertTrue(result["matched_plane"]["matched_plane_id"].startswith(
            "matched_plane_"
        ))
        failed = compare_no_gt_geometry_v2(
            self._metrics(), self._metrics(voxels=790, extent=(1.6, 2, 2)),
        )
        self.assertFalse(failed["passes_scene_safety"])

    def test_thresholds_are_runtime_bound_and_frame_hash_is_not_a_mask_hash(self):
        result = compare_no_gt_geometry_v2(
            self._metrics(), self._metrics(voxels=890),
            admitted_frame_sha256="a" * 64,
            minimum_occupied_voxel_ratio=0.90,
        )
        self.assertFalse(result["passes_scene_safety"])
        self.assertIsNone(result["common_visibility_mask_sha256"])
        self.assertEqual(
            result["safety_thresholds"]["minimum_occupied_voxel_ratio"], 0.90,
        )

    def test_precommit_refusion_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            PrecommitGeometryConfig(voxel_length_m=0.0)

    def test_improvement_can_be_required_in_addition_to_geometry_safety(self):
        geometry = {
            "passes_scene_safety": True,
            "passes_scene_improvement": False,
        }
        accepted, reason = precommit_geometry_decision(
            geometry, PrecommitGeometryConfig(require_scene_improvement=False),
        )
        self.assertTrue(accepted)
        self.assertIsNone(reason)
        accepted, reason = precommit_geometry_decision(
            geometry, PrecommitGeometryConfig(require_scene_improvement=True),
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "precommit_geometry_improvement_gate_failed")


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy runtime required")
class InformationMatrixTests(unittest.TestCase):
    def test_insufficient_support_is_not_reported_as_well_conditioned(self):
        points = np.asarray([[0.0, 0.0, 0.0]] * 5)
        information, condition, support = _point_information(
            points, points, np.eye(4), 0.1,
        )
        self.assertEqual(support, 5)
        self.assertGreater(condition, 1.0e12)
        self.assertTrue(np.all(np.linalg.eigvalsh(information) > 0.0))

    def test_reported_condition_precedes_spd_regularization(self):
        x = np.linspace(-1.0, 1.0, 20)
        points = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        information, condition, support = _point_information(
            points, points, np.eye(4), 0.1,
        )
        self.assertEqual(support, len(points))
        self.assertGreater(condition, 1.0e12)
        self.assertLessEqual(np.linalg.cond(information), 1.0e6 + 1.0)


class Scan3RCameraBasisTests(unittest.TestCase):
    def test_manifest_serializes_native_and_rotated_basis_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / "scan" / "sequence"
            sequence.mkdir(parents=True)
            intrinsic = " ".join(str(value) for value in [
                500, 0, 10, 0, 0, 510, 11, 0, 0, 0, 1, 0, 0, 0, 0, 1,
            ])
            (sequence / "_info.txt").write_text(
                f"m_calibrationDepthIntrinsic = {intrinsic}\n"
                "m_depthShift = 1000\n"
            )
            (sequence / "frame-000000.depth.pgm").write_bytes(b"depth")
            (sequence / "frame-000000.color.jpg").write_bytes(b"color")
            for mode, expected in (
                ("native", "native_sensor_camera"),
                ("rotated_ccw", "image_rotated_ccw_from_native"),
            ):
                manifest = scan3r_manifest(sequence, preprocessing=mode)
                path = Path(directory) / f"{mode}.json"
                write_manifest(path, manifest)
                payload = __import__("json").loads(path.read_text())
                self.assertEqual(payload["camera_basis"], expected)
                self.assertEqual(payload["frames"][0]["rotate_ccw"], mode != "native")

    def test_rotated_input_applies_explicit_camera_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory)
            np.savetxt(sequence / "frame-000000.pose.txt", np.eye(4))
            np.savetxt(sequence / "frame-000001.pose.txt", _pose(1.0))
            timestamps = {0: 0, 1: 1}
            native = scan3r_reference_trajectory(
                sequence, [0, 1], timestamps, input_rotated_ccw=False,
            )
            rotated = scan3r_reference_trajectory(
                sequence, [0, 1], timestamps, input_rotated_ccw=True,
            )
            np.testing.assert_allclose(native[1].t_world_camera[:3, 3], [1, 0, 0])
            np.testing.assert_allclose(rotated[1].t_world_camera[:3, 3], [0, -1, 0])


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy runtime required")
class PoseGraphV2Tests(unittest.TestCase):
    def test_absolute_gate_rejects_large_smooth_correction_hidden_from_adjacent_gate(self):
        from pose_pipeline.pose_graph import (
            CorrectionAuditConfig, audit_corrected_trajectory,
        )

        original = [PoseRecord(i, i, _pose(float(i))) for i in range(6)]
        corrected = [
            PoseRecord(i, i, _pose(float(i) + 0.60)) for i in range(6)
        ]
        audit = audit_corrected_trajectory(
            original, corrected, CorrectionAuditConfig(
                maximum_adjacent_correction_translation_m=0.05,
                maximum_adjacent_correction_rotation_deg=2.0,
                maximum_absolute_correction_translation_m=0.25,
                maximum_absolute_correction_rotation_deg=5.0,
            ),
        )
        self.assertLess(audit["maximum_adjacent_correction_translation_m"], 1e-9)
        self.assertAlmostEqual(
            audit["maximum_absolute_correction_translation_m"], 0.60,
        )
        self.assertTrue(
            audit["gates"]["adjacent_correction_translation_within_limit"],
        )
        self.assertFalse(
            audit["gates"]["absolute_correction_translation_within_limit"],
        )
        self.assertFalse(audit["passes"])

    def test_gnc_downweights_wrong_high_leverage_edge_and_audits_correction(self):
        from pose_pipeline.pose_graph import (
            CorrectionAuditConfig, PoseGraphEdge, PoseGraphOptimizationConfig,
            audit_corrected_trajectory, optimize_pose_graph,
            propagate_anchor_corrections,
        )

        initial = [_pose(index * 1.1) for index in range(6)]
        loops = [
            PoseGraphEdge(0, 3, _pose(-3.0), "loop", information=np.eye(6)),
            PoseGraphEdge(1, 4, _pose(-3.0), "loop", information=np.eye(6)),
            PoseGraphEdge(2, 5, _pose(-3.0), "loop", information=np.eye(6)),
            PoseGraphEdge(0, 5, _pose(5.0), "loop", information=np.eye(6)),
        ]
        optimized, report = optimize_pose_graph(
            initial, loops, maximum_loop_degree=3,
            optimization_config=PoseGraphOptimizationConfig(
                robustifier="adaptive_gnc", calculate_leave_one_out=True,
            ),
        )
        self.assertTrue(report["success"])
        wrong = next(row for row in report["edges"] if (
            row["kind"] == "loop" and row["source"] == 0 and row["target"] == 5
        ))
        self.assertLess(wrong["final_robust_weight"], 0.5)
        self.assertGreater(wrong["leave_one_edge_out_max_translation_m"], 0.0)
        rows = [PoseRecord(i, i, value) for i, value in enumerate(initial)]
        corrected = propagate_anchor_corrections(
            rows, list(range(6)), optimized, propagation="se3_correction_field",
        )
        audit = audit_corrected_trajectory(
            rows, corrected, CorrectionAuditConfig(
                propagation="se3_correction_field",
                maximum_adjacent_correction_translation_m=1.0,
                maximum_adjacent_correction_rotation_deg=20.0,
            ),
        )
        self.assertTrue(audit["passes"])


if __name__ == "__main__":
    unittest.main()
