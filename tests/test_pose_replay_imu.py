from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pose_pipeline.contracts import FrameRecord, ImuSample, SequenceManifest, write_manifest
from pose_pipeline.replay import REQUEST, RESPONSE, RESPONSE_MAGIC, replay_manifest


class _FakeConnection:
    def __init__(self, response: bytes):
        self.responses = bytearray(response * 2)
        self.sent = []

    def settimeout(self, _timeout):
        pass

    def connect(self, _path):
        pass

    def sendall(self, value):
        self.sent.append(bytes(value))

    def recv(self, count):
        value = bytes(self.responses[:count])
        del self.responses[:count]
        return value

    def close(self):
        pass


class PoseReplayImuTests(unittest.TestCase):
    def test_replay_partitions_and_sends_imu_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "color").mkdir()
            (root / "depth").mkdir()
            frames = []
            for frame_id, timestamp_us in ((0, 10), (1, 20)):
                color = root / "color" / f"{frame_id}.png"
                depth = root / "depth" / f"{frame_id}.png"
                color.write_bytes(b"rgb")
                depth.write_bytes(b"depth")
                frames.append(FrameRecord(
                    frame_id, timestamp_us, color, depth,
                    (100.0, 100.0, 1.0, 1.0),
                ))
            manifest_path = root / "manifest.json"
            write_manifest(manifest_path, SequenceManifest(
                "orbbec", "scene", root, 1000.0, tuple(frames), "test",
                (
                    ImuSample(5, 0, 0.0, 9.8, 0.0),
                    ImuSample(10, 1, 0.1, 0.0, 0.0),
                    ImuSample(11, 0, 0.0, 9.8, 0.0),
                    ImuSample(20, 1, 0.1, 0.0, 0.0),
                ),
            ))
            identity = np.eye(4).reshape(-1).tolist()
            response = RESPONSE.pack(
                RESPONSE_MAGIC, 1, 1, 1, 1, 0, 1,
                *identity, 0.0, 0.0, 1.0, 0.0, 1.0, 0,
            )
            connection = _FakeConnection(response)
            rgb = np.zeros((16, 16, 3), dtype=np.uint8)
            depth = np.ones((16, 16), dtype=np.uint16)
            with (
                patch("pose_pipeline.replay.socket.socket", return_value=connection),
                patch(
                    "pose_pipeline.replay._read_frame",
                    return_value=(
                        rgb, depth, (100.0, 100.0, 1.0, 1.0),
                        {"pad_right_px": 0, "pad_bottom_px": 0},
                    ),
                ),
            ):
                summary = replay_manifest(
                    manifest_path=manifest_path,
                    socket_path=root / "worker.sock",
                    output_dir=root / "output",
                )
            self.assertEqual(summary["imu_samples_sent"], 4)
            headers = [value for value in connection.sent if len(value) == REQUEST.size]
            self.assertEqual([REQUEST.unpack(value)[9] for value in headers], [2, 2])


if __name__ == "__main__":
    unittest.main()
