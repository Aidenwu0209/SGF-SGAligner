from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.contracts import (
    FrameRecord, PoseRecord, SequenceManifest, load_trajectory,
    write_manifest, write_trajectory,
)
from pose_pipeline.runner import run_sequence
from pose_pipeline.submaps import AdaptiveAnchorConfig


class PoseRunnerFailClosedTests(unittest.TestCase):
    def _two_pose_fixture(self, root: Path):
        (root / "color").mkdir()
        (root / "depth").mkdir()
        frames, poses = [], []
        for index in range(2):
            color = root / "color" / f"{index}.jpg"
            depth = root / "depth" / f"{index}.png"
            color.write_bytes(b"rgb")
            depth.write_bytes(b"depth")
            frames.append(FrameRecord(
                index, index, color, depth, (500.0, 500.0, 1.0, 1.0),
            ))
            transform = np.eye(4)
            transform[0, 3] = float(index)
            poses.append(PoseRecord(index, index, transform, source="DPV-SLAM"))
        manifest_path = root / "manifest.json"
        trajectory_path = root / "trajectory.json"
        write_manifest(manifest_path, SequenceManifest(
            "scannet", "scene", root, 1000.0, tuple(frames), "test",
        ))
        write_trajectory(
            trajectory_path, poses, sequence_id="scene", arm="baseline",
        )
        return manifest_path, trajectory_path, poses

    def test_sparse_submap_failure_is_an_explicit_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trajectory_path, poses = self._two_pose_fixture(root)
            with patch(
                "pose_pipeline.runner.build_submap",
                side_effect=ValueError("anchor produced only 176 points"),
            ):
                result = run_sequence(
                    arm="candidate", manifest_path=manifest_path,
                    trajectory_path=trajectory_path, output_dir=root / "candidate",
                )
            self.assertEqual(result["reason"], "submap_construction_failed")
            self.assertFalse(result["backend_correction_applied"])
            output, _ = load_trajectory(root / "candidate" / "trajectory.json")
            self.assertEqual(len(output), len(poses))
            evidence = (root / "candidate" / "loop_evidence.json").read_text()
            self.assertIn("anchor produced only 176 points", evidence)

    def test_single_valid_pose_is_an_explicit_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            color, depth = root / "color" / "0.jpg", root / "depth" / "0.png"
            color.write_bytes(b"rgb")
            depth.write_bytes(b"depth")
            frame = FrameRecord(
                0, 1000, color, depth, (500.0, 500.0, 1.0, 1.0),
            )
            pose = PoseRecord(0, 1000, np.eye(4), source="DPV-SLAM")
            manifest_path = root / "manifest.json"
            trajectory_path = root / "trajectory.json"
            write_manifest(manifest_path, SequenceManifest(
                "3rscan", "scan", root, 1000.0, (frame,), "test",
            ))
            write_trajectory(
                trajectory_path, [pose], sequence_id="scan", arm="baseline",
            )
            result = run_sequence(
                arm="candidate", manifest_path=manifest_path,
                trajectory_path=trajectory_path, output_dir=root / "candidate",
            )
            self.assertEqual(
                result["reason"], "insufficient_valid_poses_for_sparse_backend",
            )
            self.assertFalse(result["backend_correction_applied"])
            self.assertFalse(result["identity_fallback_used"])
            output, payload = load_trajectory(root / "candidate" / "trajectory.json")
            self.assertEqual(len(output), 1)
            np.testing.assert_allclose(output[0].t_world_camera, pose.t_world_camera)
            self.assertEqual(
                payload["metadata"]["fail_closed_action"],
                "retain_original_dpv_trajectory",
            )

    def test_no_verified_loop_retains_complete_dpv_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trajectory_path, poses = self._two_pose_fixture(root)
            fake_submap = SimpleNamespace(
                anchor_frame_id=0, source_frame_ids=(0,),
                points=np.zeros((500, 3)), points_sha256="0" * 64,
            )
            with (
                patch("pose_pipeline.runner.build_submap", return_value=fake_submap),
                patch("pose_pipeline.runner.save_submap"),
                patch("pose_pipeline.runner.propose_loop_pairs", return_value=[]),
            ):
                result = run_sequence(
                    arm="candidate", manifest_path=manifest_path,
                    trajectory_path=trajectory_path, output_dir=root / "candidate",
                )
            self.assertFalse(result["accepted"])
            self.assertTrue(result["corrected_trajectory_written"])
            self.assertFalse(result["backend_correction_applied"])
            self.assertFalse(result["identity_fallback_used"])
            output, payload = load_trajectory(root / "candidate" / "trajectory.json")
            self.assertEqual(len(output), len(poses))
            for before, after in zip(poses, output):
                np.testing.assert_allclose(before.t_world_camera, after.t_world_camera)
            self.assertEqual(
                payload["metadata"]["fail_closed_action"],
                "retain_original_dpv_trajectory",
            )

    def test_adaptive_anchor_failure_retains_complete_dpv_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, trajectory_path, poses = self._two_pose_fixture(root)
            with patch(
                "pose_pipeline.runner.select_adaptive_anchor_ordinals",
                side_effect=ValueError("depth evidence unavailable"),
            ):
                result = run_sequence(
                    arm="candidate",
                    manifest_path=manifest_path,
                    trajectory_path=trajectory_path,
                    output_dir=root / "candidate",
                    adaptive_anchor_config=AdaptiveAnchorConfig(),
                )
            self.assertEqual(result["reason"], "adaptive_anchor_selection_failed")
            self.assertFalse(result["backend_correction_applied"])
            self.assertFalse(result["identity_fallback_used"])
            output, payload = load_trajectory(root / "candidate" / "trajectory.json")
            self.assertEqual(len(output), len(poses))
            for before, after in zip(poses, output):
                np.testing.assert_allclose(before.t_world_camera, after.t_world_camera)
            evidence = json.loads(
                (root / "candidate" / "loop_evidence.json").read_text()
            )
            self.assertEqual(
                evidence["rejection_reason"], "adaptive_anchor_selection_failed",
            )
            self.assertEqual(
                payload["metadata"]["fail_closed_action"],
                "retain_original_dpv_trajectory",
            )


if __name__ == "__main__":
    unittest.main()
