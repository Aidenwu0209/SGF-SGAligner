from __future__ import annotations

from dataclasses import asdict, fields, replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.contracts import FrameRecord
from pose_pipeline.visual_verification import (
    VisualVerificationConfig, _depth_correspondences, _gate, _read_frame,
    _solve_pnp, compare_visual_registration, estimate_rgbd_loop, verify_rgbd_loop,
)


def _frame(frame_id=1, *, rotate_ccw=False):
    return FrameRecord(
        frame_id, frame_id, Path(f"/not-present/color/{frame_id}.jpg"),
        Path(f"/not-present/depth/{frame_id}.png"), (100., 200., 2., 1.),
        rotate_ccw=rotate_ccw,
    )


class VisualVerificationContractTests(unittest.TestCase):
    def test_all_numeric_configuration_fields_reject_nan_and_infinity(self):
        for field in fields(VisualVerificationConfig):
            if field.name == "enabled":
                continue
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field.name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        VisualVerificationConfig(**{field.name: invalid})
        for kwargs in (
            {"enabled": "true"}, {"minimum_inlier_ratio": 1.01},
            {"minimum_depth_m": 3., "maximum_depth_m": 2.},
            {"ransac_confidence": 1.}, {"ratio_threshold": 0.},
            {"minimum_depth_correspondences": 3}, {"seed": 2**31 - 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VisualVerificationConfig(**kwargs)
        json.dumps(asdict(VisualVerificationConfig(
            seed=np.int64(7), minimum_inlier_ratio=np.float32(.2),
        )), allow_nan=False)

    def test_disabled_verifier_does_not_load_runtime_or_files(self):
        with patch("pose_pipeline.visual_verification.importlib.import_module") as load:
            result = verify_rgbd_loop(_frame(1), _frame(2), 1000., np.eye(4))
            estimate = estimate_rgbd_loop(_frame(1), _frame(2), 1000.)
        load.assert_not_called()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "visual_verification_disabled")
        self.assertFalse(result["gt_consumed"])
        self.assertEqual(estimate["schema"], "rgbd_visual_loop_estimate.v1")
        self.assertFalse(estimate["usable_for_reconstruction"])

    def test_pure_comparison_cannot_promote_failed_or_mislabeled_estimate(self):
        config = VisualVerificationConfig(enabled=True)
        item = {
            "depth_correspondences": 30, "inliers": 28, "inlier_ratio": 28 / 30,
            "transform": np.eye(4).tolist(),
        }
        estimate = {
            "schema": "rgbd_visual_loop_estimate.v1", "accepted": True,
            "matrix_convention": "T_target_source_m", "gt_consumed": False,
            "forward": dict(item), "reverse": dict(item),
            "cycle_rotation_deg": 0., "cycle_translation_m": 0.,
        }
        original = json.dumps(estimate, sort_keys=True)
        with patch("pose_pipeline.visual_verification.importlib.import_module", side_effect=AssertionError("comparison must be pure")):
            result = compare_visual_registration(estimate, np.eye(4), config)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["schema"], "rgbd_visual_loop_verification.v1")
            for mutation in (
                {"accepted": False, "reason": "estimator_failed"},
                {"gt_consumed": True}, {"matrix_convention": "T_source_target_m"},
            ):
                with self.subTest(mutation=mutation):
                    self.assertFalse(compare_visual_registration({**estimate, **mutation}, np.eye(4), config)["accepted"])
            wrong = np.eye(4)
            wrong[0, 3] = 1.
            self.assertFalse(compare_visual_registration(estimate, wrong, config)["accepted"])
            # Inconsistent matrices cannot hide behind a forged zero-cycle scalar.
            conflicting = {**estimate, "reverse": {**item, "transform": wrong.tolist()}}
            self.assertFalse(compare_visual_registration(conflicting, np.eye(4), config)["accepted"])
        self.assertEqual(json.dumps(estimate, sort_keys=True), original)

    def test_legacy_verifier_estimates_only_once(self):
        estimate = {
            "schema": "rgbd_visual_loop_estimate.v1", "accepted": False,
            "matrix_convention": "T_target_source_m", "gt_consumed": False,
            "reason": "forward_pnp_unavailable", "forward": None, "reverse": None,
        }
        with patch("pose_pipeline.visual_verification.estimate_rgbd_loop", return_value=estimate) as solver:
            result = verify_rgbd_loop(_frame(1), _frame(2), 1000., np.eye(4), VisualVerificationConfig(enabled=True))
        solver.assert_called_once()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "forward_pnp_unavailable")

    def test_missing_runtime_and_invalid_input_fail_closed(self):
        config = VisualVerificationConfig(enabled=True)
        with patch("pose_pipeline.visual_verification.importlib.import_module", side_effect=ImportError("unavailable")):
            result = verify_rgbd_loop(_frame(1), _frame(2), 1000., np.eye(4), config)
        self.assertFalse(result["accepted"])
        self.assertIn("ImportError", result["error"])
        for source, target, scale, transform in (
            (_frame(1), _frame(1), 1000., np.eye(4)),
            (_frame(1), _frame(2, rotate_ccw=True), 1000., np.eye(4)),
            (_frame(1), _frame(2), float("nan"), np.eye(4)),
            (_frame(1), _frame(2), None, np.eye(4)),
            (_frame(1), _frame(2), 1000., np.zeros((4, 4))),
        ):
            with self.subTest(scale=scale, target=target):
                result = verify_rgbd_loop(source, target, scale, transform, config)
                self.assertFalse(result["accepted"])
                self.assertFalse(result["gt_consumed"])
                json.dumps(result, allow_nan=False)

    def test_metric_backprojection_uses_depth_scale_and_subpixel_intrinsics(self):
        config = VisualVerificationConfig(enabled=True)
        points, pixels = _depth_correspondences(
            [SimpleNamespace(pt=(3.2, 2.))], [SimpleNamespace(pt=(8.5, 9.5))],
            [SimpleNamespace(queryIdx=0, trainIdx=0)],
            np.full((5, 6), 5000, dtype=np.uint16),
            np.array([[100., 0., 2.], [0., 200., 1.], [0., 0., 1.]]),
            5000., config,
        )
        np.testing.assert_allclose(points, [[.012, .005, 1.]])
        np.testing.assert_allclose(pixels, [[8.5, 9.5]])

    def test_bidirectional_gate_rejects_single_witness_or_disagreement(self):
        config = VisualVerificationConfig(enabled=True)
        witness = {
            "depth_correspondences": 30, "inliers": 28, "inlier_ratio": 28 / 30,
            "registration_rotation_deg": .1, "registration_translation_m": .001,
        }
        evidence = {
            "forward": dict(witness), "reverse": dict(witness),
            "cycle_rotation_deg": .1, "cycle_translation_m": .001,
        }
        self.assertTrue(_gate(evidence, config)[0])
        for mutation in (
            {"reverse": None}, {"cycle_rotation_deg": 11.},
            {"cycle_translation_m": float("nan")},
            {"forward": {**witness, "registration_translation_m": .36}},
            {"reverse": {**witness, "inliers": 6}},
        ):
            with self.subTest(mutation=mutation):
                self.assertFalse(_gate({**evidence, **mutation}, config)[0])


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV runtime required")
class VisualVerificationOpenCVTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2

    def test_rotated_camera_intrinsics_match_submap_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color_path, depth_path = root / "color.png", root / "depth.png"
            depth = np.arange(15, dtype=np.uint16).reshape(3, 5) + 1000
            self.cv2.imwrite(str(color_path), np.zeros((6, 10), dtype=np.uint8))
            self.cv2.imwrite(str(depth_path), depth)
            frame = FrameRecord(1, 1, color_path, depth_path, (100., 200., 1.5, .5), True)
            color, rotated, intrinsic = _read_frame(self.cv2, frame)
            self.assertEqual(color.shape, (5, 3))
            np.testing.assert_array_equal(rotated, np.rot90(depth))
            np.testing.assert_allclose(intrinsic, [[200., 0., .5], [0., 100., 2.5], [0., 0., 1.]])

    def test_missing_corrupt_and_forbidden_inputs_are_rejected(self):
        config = VisualVerificationConfig(enabled=True)
        result = verify_rgbd_loop(_frame(1), _frame(2), 1000., np.eye(4), config)
        self.assertFalse(result["accepted"])
        forbidden = replace(_frame(1), color_path=Path("/gt/color.png"))
        result = verify_rgbd_loop(forbidden, _frame(2), 1000., np.eye(4), config)
        self.assertIn("forbidden GT", result["error"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            color_path, depth_path = root / "color.png", root / "depth.png"
            self.cv2.imwrite(str(color_path), np.zeros((4, 4), dtype=np.uint8))
            self.cv2.imwrite(str(depth_path), np.ones((4, 4), dtype=np.uint8))
            corrupt = replace(_frame(1), color_path=color_path, depth_path=depth_path)
            result = verify_rgbd_loop(corrupt, _frame(2), 1000., np.eye(4), config)
            self.assertFalse(result["accepted"])
            self.assertIn("uint16", result["error"])
        with patch.object(self.cv2, "imread", side_effect=self.cv2.error("decoder failed")):
            result = verify_rgbd_loop(_frame(1), _frame(2), 1000., np.eye(4), config)
        self.assertFalse(result["accepted"])
        self.assertIn("decoder failed", result["error"])

    def test_pnp_recovers_direction_with_outliers_and_rejects_behind_camera(self):
        rng = np.random.default_rng(19)
        points = rng.uniform([-.7, -.5, 1.3], [.7, .5, 3.5], size=(100, 3))
        rotation = self.cv2.Rodrigues(np.array([.08, -.10, .04]))[0]
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [.14, -.04, .07]
        intrinsic = np.array([[430., 0., 310.], [0., 390., 245.], [0., 0., 1.]])
        target = points @ rotation.T + transform[:3, 3]
        pixels = target @ intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        pixels[:20] = rng.uniform([0, 0], [640, 480], size=(20, 2))
        config = VisualVerificationConfig(enabled=True)
        result = _solve_pnp(self.cv2, points, pixels, intrinsic, config, config.seed)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["transform"], transform, atol=1e-5)
        self.assertGreaterEqual(result["inliers"], 80)
        behind_translation = np.array([[0.], [0.], [-10.]])
        with patch.object(self.cv2, "solvePnPRansac", return_value=(
            True, np.zeros((3, 1)), behind_translation, np.arange(100).reshape(-1, 1),
        )):
            self.assertIsNone(_solve_pnp(self.cv2, points, pixels, intrinsic, config, config.seed))

    def test_manifest_pair_uses_independent_reverse_matches_and_each_camera_k(self):
        cv2 = self.cv2
        rng = np.random.default_rng(7)
        source_k = np.array([[100., 0., 100.], [0., 130., 80.], [0., 0., 1.]])
        target_k = np.array([[180., 0., 120.], [0., 160., 90.], [0., 0., 1.]])
        source_pixels = np.array([(u, v) for u in range(65, 141, 15) for v in range(50, 111, 15)], dtype=float)
        source_z = rng.integers(7000, 14000, len(source_pixels)) / 5000.
        source_points = np.column_stack((source_pixels, np.ones(len(source_pixels)))) @ np.linalg.inv(source_k).T * source_z[:, None]
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(np.array([.02, .05, -.03]))[0]
        transform[:3, 3] = [.08, -.03, .02]
        target_points = source_points @ transform[:3, :3].T + transform[:3, 3]
        target_pixels_h = target_points @ target_k.T
        target_pixels = target_pixels_h[:, :2] / target_pixels_h[:, 2:]
        keypoints = [
            [cv2.KeyPoint(float(u), float(v), 1.) for u, v in pixels]
            for pixels in (source_pixels, target_pixels)
        ]
        # Feature detection alone is injected; real BF matching, RGB-D pixel
        # loading/backprojection and both real OpenCV RANSAC solves run below.
        descriptors = rng.normal(size=(len(source_pixels), 128)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            for index, (pixels, z, intrinsic, shape) in enumerate((
                (source_pixels, source_z, source_k, (160, 200)),
                (target_pixels, target_points[:, 2], target_k, (180, 240)),
            )):
                depth = np.zeros(shape, dtype=np.uint16)
                rounded = np.round(pixels).astype(int)
                depth[rounded[:, 1], rounded[:, 0]] = np.round(z * 5000.).astype(np.uint16)
                color_path, depth_path = root / f"color{index}.png", root / f"depth{index}.png"
                cv2.imwrite(str(color_path), np.zeros(shape, dtype=np.uint8))
                cv2.imwrite(str(depth_path), depth)
                frames.append(FrameRecord(index, index, color_path, depth_path, (
                    intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2],
                )))
            sift = SimpleNamespace(detectAndCompute=unittest.mock.Mock(side_effect=[
                (keypoints[0], descriptors), (keypoints[1], descriptors.copy()),
            ]))
            from pose_pipeline.visual_verification import _ratio_matches
            with patch.object(cv2, "SIFT_create", return_value=sift), patch(
                "pose_pipeline.visual_verification._ratio_matches", wraps=_ratio_matches,
            ) as matcher:
                config = VisualVerificationConfig(enabled=True)
                estimate = estimate_rgbd_loop(*frames, 5000., config)
                result = compare_visual_registration(estimate, transform, config)
                wrong = transform.copy()
                wrong[0, 3] += 1.
                rejected = compare_visual_registration(estimate, wrong, config)
            self.assertTrue(estimate["accepted"], estimate)
            self.assertEqual(estimate["schema"], "rgbd_visual_loop_estimate.v1")
            self.assertEqual(estimate["solver_family"], "rgbd_pnp")
            self.assertFalse(estimate["usable_for_reconstruction"])
            self.assertNotIn("registration_rotation_deg", estimate["forward"])
            self.assertFalse(rejected["accepted"])
            self.assertEqual(sift.detectAndCompute.call_count, 2)
            self.assertTrue(result["accepted"], result)
            self.assertEqual(matcher.call_count, 2)
            self.assertIs(matcher.call_args_list[0].args[1], descriptors)
            self.assertIs(matcher.call_args_list[1].args[2], descriptors)
            np.testing.assert_allclose(result["forward"]["transform"], transform, atol=1e-5)
            np.testing.assert_allclose(result["reverse"]["transform"], transform, atol=2e-4)
            self.assertEqual(len(result["rgbd_inputs_sha256"]), 4)
            self.assertFalse(result["gt_consumed"])
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
