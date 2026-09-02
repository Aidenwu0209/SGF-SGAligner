from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from pose_pipeline.adapters import orbbec_capture_manifest


class OrbbecCaptureAdapterTests(unittest.TestCase):
    def test_rgbd_and_imu_capture_becomes_one_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            (root / "color" / "000000.png").write_bytes(b"rgb")
            (root / "depth" / "000000.png").write_bytes(b"depth")
            (root / "manifest.json").write_text(json.dumps({
                "state": "complete",
                "calibration": "calibration.json",
                "frames_index": "frames.csv",
                "imu_index": "imu.csv",
            }))
            (root / "calibration.json").write_text(json.dumps({
                "camera_parameters": {"rgb_intrinsic": {
                    "fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0,
                }},
            }))
            with (root / "frames.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "frame_index", "color_file", "depth_file", "color_timestamp_us",
                ])
                writer.writeheader()
                writer.writerow({
                    "frame_index": 0,
                    "color_file": "color/000000.png",
                    "depth_file": "depth/000000.png",
                    "color_timestamp_us": 100,
                })
            with (root / "imu.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "kind", "timestamp_us", "x", "y", "z",
                ])
                writer.writeheader()
                writer.writerow({
                    "kind": "gyro", "timestamp_us": 99,
                    "x": 0.1, "y": 0.2, "z": 0.3,
                })
                writer.writerow({
                    "kind": "accel", "timestamp_us": 98,
                    "x": 0.0, "y": 9.8, "z": 0.0,
                })
            manifest = orbbec_capture_manifest(root)
            self.assertEqual(len(manifest.frames), 1)
            self.assertEqual([sample.kind for sample in manifest.imu_samples], [0, 1])
            self.assertEqual(manifest.imu_samples[0].timestamp_us, 98)


if __name__ == "__main__":
    unittest.main()
