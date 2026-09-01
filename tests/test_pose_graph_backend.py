from __future__ import annotations

import unittest
import importlib.util

import numpy as np

from pose_pipeline.contracts import PoseRecord
from pose_pipeline.pose_graph import (
    PoseGraphEdge,
    optimize_pose_graph,
    propagate_anchor_corrections,
    sparsify_loop_edges,
)


def pose(x: float) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy runtime required")
class PoseGraphBackendTests(unittest.TestCase):
    def test_loop_reduces_terminal_drift_and_propagates_all_frames(self):
        initial = [pose(0.0), pose(1.1), pose(2.2), pose(3.3)]
        loop = PoseGraphEdge(
            source=0, target=3, source_to_target=pose(-3.0),
            kind="loop", weight=1.5, provenance="test",
        )
        optimized, report = optimize_pose_graph(initial, [loop])
        self.assertTrue(report["success"])
        self.assertLess(abs(optimized[-1][0, 3] - 3.0), 0.3)
        rows = [PoseRecord(index, index, value) for index, value in enumerate(initial)]
        corrected = propagate_anchor_corrections(rows, [0, 3], [optimized[0], optimized[3]])
        self.assertEqual([row.frame_id for row in corrected], [0, 1, 2, 3])
        self.assertTrue(all(np.isfinite(row.t_world_camera).all() for row in corrected))

    def test_duplicate_and_degree_edges_are_rejected(self):
        edges = [
            PoseGraphEdge(0, 2, pose(-2.0), "loop", 1.0, "a"),
            PoseGraphEdge(2, 0, pose(2.0), "loop", 0.5, "b"),
            PoseGraphEdge(0, 3, pose(-3.0), "loop", 0.9, "c"),
            PoseGraphEdge(0, 4, pose(-4.0), "loop", 0.8, "d"),
        ]
        accepted, rejected = sparsify_loop_edges(edges, maximum_loop_degree=2)
        self.assertEqual(len(accepted), 2)
        self.assertEqual({row["reason"] for row in rejected}, {
            "duplicate_pair", "loop_degree_cap",
        })


if __name__ == "__main__":
    unittest.main()
