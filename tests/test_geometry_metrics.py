from __future__ import annotations

import unittest

from pose_pipeline.geometry_metrics import compare_no_gt_geometry


def geometry(conflict, vertices=1000, bbox=(2.0, 2.0, 2.0), tilt=1.0, thickness=0.02):
    return {
        "vertices": vertices,
        "bbox_extent_m": list(bbox),
        "near_parallel_layer_conflict_ratio": conflict,
        "horizontal_planes": [{
            "points": 900,
            "tilt_from_gravity_deg": tilt,
            "thickness_p90_p10_m": thickness,
        }],
    }


class GeometryMetricTests(unittest.TestCase):
    def test_safety_and_improvement_are_separate(self):
        result = compare_no_gt_geometry(
            geometry(0.30), geometry(0.28, vertices=900, tilt=2.0),
        )
        self.assertTrue(result["passes_scene_safety"])
        self.assertFalse(result["passes_scene_improvement"])
        improved = compare_no_gt_geometry(
            geometry(0.30), geometry(0.25, vertices=900, tilt=2.0),
        )
        self.assertTrue(improved["passes_scene_improvement"])


if __name__ == "__main__":
    unittest.main()
