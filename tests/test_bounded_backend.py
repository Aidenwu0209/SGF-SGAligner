from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.bounded_backend import (
    BoundedBackendConfig,
    optimize_bounded_trajectory,
)
from pose_pipeline.contracts import PoseRecord
from pose_pipeline.pose_graph import (
    CorrectionAuditConfig,
    PoseGraphEdge,
    PoseGraphOptimizationConfig,
    interpolate_transform,
    optimize_pose_graph,
    propagate_anchor_corrections,
)


def pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


def trajectory(count: int = 13, step: float = 0.275) -> list[PoseRecord]:
    return [PoseRecord(10 + index, 1000 * index, pose(step * index), source="frontend")
            for index in range(count)]


LOOSE_AUDIT = CorrectionAuditConfig(
    maximum_adjacent_correction_translation_m=2.0,
    maximum_adjacent_correction_rotation_deg=90.0,
    maximum_absolute_correction_translation_m=2.0,
    maximum_absolute_correction_rotation_deg=90.0,
)


class BoundedConfigTests(unittest.TestCase):
    def test_config_is_frozen_and_opt_in(self):
        config = BoundedBackendConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.correction_scaling_policy, "global")
        with self.assertRaises(FrozenInstanceError):
            config.enabled = True

    def test_nonfinite_and_invalid_config_is_rejected(self):
        for name in (
            "maximum_loop_weight", "high_leverage_min_span_fraction",
            "high_leverage_weight_cap", "maximum_leave_one_out_translation_m",
            "maximum_leave_one_out_rotation_deg", "local_scaling_smoothness",
        ):
            for value in (float("nan"), float("inf"), -1.0, 0.0):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    BoundedBackendConfig(**{name: value})
        for value in (0, 1.5, True, float("nan")):
            with self.subTest(degree=value), self.assertRaises(ValueError):
                BoundedBackendConfig(maximum_loop_degree=value)
        for value in ((), (0.5,), (1.0, 1.0), (1.0, 0.0), (1.0, float("nan")), [1.0, 0.5]):
            with self.subTest(scales=value), self.assertRaises(ValueError):
                BoundedBackendConfig(correction_backtracking_scales=value)
        with self.assertRaises(ValueError):
            BoundedBackendConfig(high_leverage_min_span_fraction=1.01)
        for value in ("clip", "", None, []):
            with self.subTest(policy=value), self.assertRaises(ValueError):
                BoundedBackendConfig(correction_scaling_policy=value)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy runtime required")
class BoundedBackendTests(unittest.TestCase):
    def test_smooth_local_preserves_distant_corrections_around_large_anchor(self):
        rows = trajectory(61, 0.01)
        anchors = list(range(0, 61, 10))
        changes = [0, 0.10, 0.10, 2.0, 0.10, 0.10, 0.10]
        optimized = [pose(change) @ rows[index].t_world_camera
                     for change, index in zip(changes, anchors)]
        bounds = CorrectionAuditConfig(
            maximum_absolute_correction_translation_m=0.25,
            maximum_absolute_correction_rotation_deg=5.0,
            maximum_adjacent_correction_translation_m=0.03,
        )
        with patch("pose_pipeline.bounded_backend.optimize_pose_graph", return_value=(optimized, {"success": True})):
            global_output, global_report = optimize_bounded_trajectory(
                rows, anchors, [PoseGraphEdge(0, 6, pose(-0.6), "loop")],
                config=BoundedBackendConfig(enabled=True), correction_config=bounds,
            )
            local_output, local_report = optimize_bounded_trajectory(
                rows, anchors, [PoseGraphEdge(0, 6, pose(-0.6), "loop")],
                config=BoundedBackendConfig(enabled=True, correction_scaling_policy="smooth_local"),
                correction_config=bounds,
            )
        self.assertTrue(global_report["success"])
        self.assertEqual(global_report["selected_correction_scale"], 0.125)
        self.assertTrue(local_report["success"])
        self.assertTrue(local_report["local_scaling"]["success"])
        self.assertEqual(local_report["selected_correction_scale"], 1.0)
        self.assertEqual(local_report["correction_audit"]["pose_count"], len(rows))
        for ordinal in (10, 50, 60):
            retained = local_output[ordinal].t_world_camera[0, 3] - rows[ordinal].t_world_camera[0, 3]
            global_retained = global_output[ordinal].t_world_camera[0, 3] - rows[ordinal].t_world_camera[0, 3]
            self.assertGreater(retained, 0.09)
            self.assertGreater(retained, 7 * global_retained)
        self.assertLessEqual(local_report["correction_audit"]["maximum_absolute_correction_translation_m"], 0.25)
        self.assertLessEqual(local_report["correction_audit"]["maximum_adjacent_correction_translation_m"], 0.03)
        self.assertFalse(local_report["gt_consumed"])
        json.dumps(local_report, allow_nan=False)

    def test_smooth_local_constrains_dense_opposite_rotations_and_all_frames(self):
        from scipy.spatial.transform import Rotation

        rows = trajectory(11, 0.02)
        anchors = [0, 1, 3, 4, 8, 10]
        optimized = []
        for ordinal, angle, translation_x in zip(anchors, (0, 30, -30, 20, 4, 4), (0, 1, -1, 1, 0.1, 0.1)):
            delta = pose(translation_x)
            delta[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            optimized.append(delta @ rows[ordinal].t_world_camera)
        bounds = CorrectionAuditConfig(
            maximum_absolute_correction_translation_m=0.25,
            maximum_absolute_correction_rotation_deg=5.0,
            maximum_adjacent_correction_translation_m=0.05,
            maximum_adjacent_correction_rotation_deg=2.0,
        )
        with patch("pose_pipeline.bounded_backend.optimize_pose_graph", return_value=(optimized, {"success": True})):
            output, report = optimize_bounded_trajectory(
                rows, anchors, [PoseGraphEdge(0, 5, pose(-0.2), "loop")],
                config=BoundedBackendConfig(enabled=True, correction_scaling_policy="smooth_local"),
                correction_config=bounds,
            )
        self.assertTrue(report["success"], report)
        self.assertEqual(report["selected_correction_scale"], 1.0)
        self.assertTrue(all(report["correction_audit"]["gates"].values()))
        self.assertEqual([(r.frame_id, r.timestamp_us) for r in output],
                         [(r.frame_id, r.timestamp_us) for r in rows])
        scales = report["selected_anchor_correction_scales"]
        # Intermediate frames use the original SLERP/linear propagation of
        # jointly fitted anchor corrections, with no individual frame clipping.
        fitted = []
        for original_index, optimized_pose, scale in zip(anchors, optimized, scales):
            delta = optimized_pose @ np.linalg.inv(rows[original_index].t_world_camera)
            fitted.append(interpolate_transform(np.eye(4), delta, scale) @ rows[original_index].t_world_camera)
        expected = propagate_anchor_corrections(rows, anchors, fitted)
        for actual, wanted in zip(output, expected):
            np.testing.assert_allclose(actual.t_world_camera, wanted.t_world_camera, atol=1e-12)

    def test_smooth_local_solver_failure_returns_original(self):
        from types import SimpleNamespace

        rows = trajectory(3, 0.1)
        with patch("scipy.optimize.minimize", return_value=SimpleNamespace(
            x=np.zeros(3), success=False, message="iteration limit", nit=200,
        )):
            output, report = optimize_bounded_trajectory(
                rows, [0, 1, 2], [PoseGraphEdge(0, 2, pose(-0.18), "loop")],
                config=BoundedBackendConfig(enabled=True, correction_scaling_policy="smooth_local"),
                correction_config=LOOSE_AUDIT,
            )
        self.assertFalse(report["success"])
        self.assertEqual(report["failure_reason"], "smooth_local_scale_optimization_failed")
        self.assertTrue(report["requires_byte_rollback"])
        self.assertTrue(all(left is right for left, right in zip(rows, output)))

    def test_smooth_local_real_pgo_preserves_more_of_known_drift_correction(self):
        rows = trajectory()
        anchors = [0, 4, 8, 12]
        loops = [PoseGraphEdge(0, 3, pose(-3.0), "loop", 1.5)]
        bounds = CorrectionAuditConfig(
            maximum_adjacent_correction_translation_m=0.015,
            maximum_adjacent_correction_rotation_deg=2.0,
            maximum_absolute_correction_translation_m=0.15,
            maximum_absolute_correction_rotation_deg=3.0,
        )
        global_output, _ = optimize_bounded_trajectory(
            rows, anchors, loops, config=BoundedBackendConfig(enabled=True),
            correction_config=bounds,
        )
        output, report = optimize_bounded_trajectory(
            rows, anchors, loops,
            config=BoundedBackendConfig(enabled=True, correction_scaling_policy="smooth_local"),
            correction_config=bounds,
        )
        self.assertTrue(report["success"])
        self.assertTrue(report["pose_graph"]["optimizer_success"])
        self.assertEqual(report["selected_correction_scale"], 1.0)
        self.assertTrue(report["correction_audit"]["passes"])
        # The fixture drifts to 3.3 m despite a verified 3.0 m loop endpoint.
        # Both methods retain safety; the local fit spends more of its allowed
        # correction budget on that physical error without consulting GT.
        self.assertLess(abs(output[-1].t_world_camera[0, 3] - 3.0),
                        abs(global_output[-1].t_world_camera[0, 3] - 3.0) - 0.03)

    def test_smooth_local_never_bypasses_complete_trajectory_audit(self):
        rows = trajectory(3, 0.1)
        with patch("pose_pipeline.bounded_backend.audit_corrected_trajectory", return_value={"passes": False}):
            output, report = optimize_bounded_trajectory(
                rows, [0, 1, 2], [PoseGraphEdge(0, 2, pose(-0.18), "loop")],
                config=BoundedBackendConfig(enabled=True, correction_scaling_policy="smooth_local"),
                correction_config=LOOSE_AUDIT,
            )
        self.assertTrue(report["local_scaling"]["success"])
        self.assertFalse(report["success"])
        self.assertEqual(len(report["scale_trials"]), 5)
        self.assertEqual(report["failure_reason"], "no_declared_correction_scale_passed")
        self.assertTrue(report["requires_byte_rollback"])
        self.assertTrue(all(left is right for left, right in zip(rows, output)))

    def test_backtracking_reduces_drift_and_scales_whole_trajectory(self):
        rows = trajectory()
        anchors = [0, 4, 8, 12]
        loops = [PoseGraphEdge(0, 3, pose(-3.0), "loop", 1.5, "verified_pnp")]
        bounds = CorrectionAuditConfig(
            maximum_adjacent_correction_translation_m=0.015,
            maximum_adjacent_correction_rotation_deg=2.0,
            maximum_absolute_correction_translation_m=0.15,
            maximum_absolute_correction_rotation_deg=3.0,
        )
        output, report = optimize_bounded_trajectory(
            rows, anchors, loops, config=BoundedBackendConfig(enabled=True),
            correction_config=bounds,
        )
        self.assertTrue(report["success"])
        self.assertEqual(report["selected_correction_scale"], 0.5)
        self.assertEqual(len(output), len(rows))
        self.assertEqual([(r.frame_id, r.timestamp_us) for r in output],
                         [(r.frame_id, r.timestamp_us) for r in rows])
        self.assertLess(abs(output[-1].t_world_camera[0, 3] - 3.0), 0.3)
        self.assertFalse(report["scale_trials"][0]["audit"]["passes"])
        self.assertTrue(report["correction_audit"]["passes"])
        self.assertEqual(report["correction_audit"]["pose_count"], 13)
        self.assertLessEqual(report["correction_audit"]["maximum_absolute_correction_translation_m"], 0.15)
        capped_loop = replace(loops[0], weight=1.0)
        optimized, _ = optimize_pose_graph([rows[i].t_world_camera for i in anchors], [capped_loop])
        raw = propagate_anchor_corrections(rows, anchors, optimized)
        # One factor preserves the shape of every intermediate correction.
        for original, unscaled, corrected in zip(rows, raw, output):
            np.testing.assert_allclose(
                corrected.t_world_camera[:3, 3] - original.t_world_camera[:3, 3],
                0.5 * (unscaled.t_world_camera[:3, 3] - original.t_world_camera[:3, 3]),
                atol=1e-7,
            )
        self.assertFalse(report["gt_consumed"])
        json.dumps(report, allow_nan=False)

    def test_full_frame_rotation_and_translation_limits_drive_same_scale(self):
        from scipy.spatial.transform import Rotation

        rows = trajectory(7, 0.1)
        anchors = [0, 2, 6]
        corrections = []
        for angle, translation_x in ((0, 0), (4, 0.04), (8, 0.08)):
            delta = pose(translation_x)
            delta[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            corrections.append(delta)
        optimized = [delta @ rows[index].t_world_camera for delta, index in zip(corrections, anchors)]
        # Prescribe a rotational result to isolate all-frame propagation from
        # optimizer conditioning; 2-degree jumps require the global 1/4 scale.
        with patch("pose_pipeline.bounded_backend.optimize_pose_graph", return_value=(optimized, {"success": True})):
            corrected, report = optimize_bounded_trajectory(
                rows, anchors, [PoseGraphEdge(0, 2, pose(-0.6), "loop")],
                config=BoundedBackendConfig(enabled=True),
                correction_config=CorrectionAuditConfig(
                    maximum_adjacent_correction_translation_m=0.006,
                    maximum_adjacent_correction_rotation_deg=0.51,
                    maximum_absolute_correction_translation_m=0.03,
                    maximum_absolute_correction_rotation_deg=3.0,
                ),
            )
        self.assertTrue(report["success"])
        self.assertEqual(report["selected_correction_scale"], 0.25)
        self.assertEqual(report["correction_audit"]["pose_count"], 7)
        for index, expected_angle in enumerate((0, 0.5, 1, 1.25, 1.5, 1.75, 2)):
            delta = corrected[index].t_world_camera @ np.linalg.inv(rows[index].t_world_camera)
            actual = np.degrees(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()))
            self.assertAlmostEqual(actual, expected_angle, places=6)

    def test_no_declared_scale_passes_returns_original_records(self):
        rows = trajectory()
        output, report = optimize_bounded_trajectory(
            rows, [0, 4, 8, 12], [PoseGraphEdge(0, 3, pose(-3.0), "loop")],
            config=BoundedBackendConfig(enabled=True),
            correction_config=CorrectionAuditConfig(maximum_absolute_correction_translation_m=1e-6),
        )
        self.assertFalse(report["success"])
        self.assertTrue(report["requires_byte_rollback"])
        self.assertIsNone(report["selected_correction_scale"])
        self.assertEqual(len(report["scale_trials"]), 5)
        self.assertTrue(all(left is right for left, right in zip(rows, output)))

    def test_disabled_backend_matches_base_huber_without_weight_caps(self):
        rows = trajectory()
        anchors = [0, 4, 8, 12]
        loops = [PoseGraphEdge(0, 3, pose(-3.0), "loop", 1.5)]
        anchors_expected, _ = optimize_pose_graph([rows[i].t_world_camera for i in anchors], loops)
        expected = propagate_anchor_corrections(rows, anchors, anchors_expected)
        for policy in ("global", "smooth_local"):
            with self.subTest(disabled_policy=policy):
                output, report = optimize_bounded_trajectory(
                    rows, anchors, loops, correction_config=LOOSE_AUDIT,
                    config=BoundedBackendConfig(correction_scaling_policy=policy),
                )
                self.assertTrue(report["success"])
                self.assertEqual(report["selected_correction_scale"], 1.0)
                self.assertNotIn("local_scaling", report)
                self.assertEqual(report["loop_selection"]["weights"][0]["effective_weight"], 1.5)
                for actual, wanted in zip(output, expected):
                    np.testing.assert_allclose(actual.t_world_camera, wanted.t_world_camera, atol=1e-10)

    def test_leave_one_out_removes_bad_edge_and_rechecks_good_edges(self):
        rows = trajectory(5, 1.0)
        loops = [
            PoseGraphEdge(0, 2, pose(-2), "loop", 0.9, "good_a"),
            PoseGraphEdge(2, 4, pose(-2), "loop", 0.9, "good_b"),
            PoseGraphEdge(0, 4, pose(-3.5), "loop", 1.0, "bad"),
        ]
        output, report = optimize_bounded_trajectory(
            rows, list(range(5)), loops,
            config=BoundedBackendConfig(enabled=True, enforce_leave_one_out=True),
            correction_config=LOOSE_AUDIT,
        )
        self.assertTrue(report["success"])
        self.assertEqual([edge["provenance"] for edge in report["influence"]["rejected_edges"]], ["bad"])
        self.assertEqual(report["loop_selection"]["retained_count"], 2)
        self.assertEqual(len(report["influence"]["rounds"]), 2)
        for check in report["influence"]["rounds"][-1]["checks"]:
            self.assertTrue(check["passes"])
            self.assertTrue(check["pose_graph"]["optimizer_success"])
            self.assertTrue(check["correction_audit"]["passes"])
            self.assertEqual(check["pose_graph"]["robustifier"], "huber")
            self.assertEqual(check["raw_influence"]["pose_count"], 5)
        for original, corrected in zip(rows, output):
            np.testing.assert_allclose(corrected.t_world_camera, original.t_world_camera, atol=1e-8)

    def test_rejected_degree_edges_are_never_refilled_after_influence(self):
        rows = trajectory(5, 1.0)
        loops = [
            PoseGraphEdge(0, 4, pose(-3.5), "loop", 1.0, "selected_bad"),
            PoseGraphEdge(0, 2, pose(-2.0), "loop", 0.7, "frozen_out"),
        ]
        output, report = optimize_bounded_trajectory(
            rows, list(range(5)), loops,
            config=BoundedBackendConfig(enabled=True, maximum_loop_degree=1, enforce_leave_one_out=True),
            correction_config=LOOSE_AUDIT,
        )
        self.assertTrue(report["success"])
        self.assertTrue(report["no_op"])
        self.assertEqual(report["loop_selection"]["selected_count"], 1)
        self.assertEqual(report["loop_selection"]["retained_count"], 0)
        self.assertEqual(report["loop_selection"]["rejected"][0]["reason"], "loop_degree_cap")
        self.assertTrue(all(left is right for left, right in zip(rows, output)))

    def test_optimizer_failure_never_commits_candidate(self):
        rows = trajectory(3, 1.0)
        with patch("pose_pipeline.bounded_backend.optimize_pose_graph", return_value=(
            [pose(0), pose(1.1), pose(2.2)], {"success": False, "optimizer_success": False},
        )):
            output, report = optimize_bounded_trajectory(
                rows, [0, 1, 2], [PoseGraphEdge(0, 2, pose(-2), "loop")],
                config=BoundedBackendConfig(enabled=True),
            )
        self.assertFalse(report["success"])
        self.assertEqual(report["failure_reason"], "pose_graph_optimization_failed")
        self.assertTrue(all(left is right for left, right in zip(rows, output)))

    def test_nonfinite_nested_config_and_invalid_inputs_fail_closed(self):
        rows = trajectory(3, 1.0)
        for make_kwargs in (
            lambda: {"optimization_config": PoseGraphOptimizationConfig(gnc_initial_mu=float("nan"))},
            lambda: {"correction_config": CorrectionAuditConfig(maximum_adjacent_correction_translation_m=float("nan"))},
            lambda: {"optimization_config": PoseGraphOptimizationConfig(robustifier="adaptive_gnc")},
            lambda: {"correction_config": CorrectionAuditConfig(propagation="se3_correction_field")},
        ):
            with self.subTest(make_kwargs=make_kwargs), self.assertRaises(ValueError):
                optimize_bounded_trajectory(rows, [0, 1, 2], [], **make_kwargs())
        for anchors, loops in (
            ([0, 0, 2], []), ([0, 3], []),
            ([0, 1, 2], [PoseGraphEdge(0, 3, pose(-3), "loop")]),
            ([0, 1, 2], [PoseGraphEdge(0, 2, pose(-2), "loop", float("nan"))]),
        ):
            with self.subTest(anchors=anchors, loops=loops), self.assertRaises(ValueError):
                optimize_bounded_trajectory(rows, anchors, loops)


if __name__ == "__main__":
    unittest.main()
