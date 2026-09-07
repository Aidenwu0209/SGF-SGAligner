from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from pose_pipeline.geometry_backend import GeometryBootstrapConfig, _one_direction, register_submaps_bidirectional
from pose_pipeline.robust_backend import RobustPoseConfig, _hypothesis, select_cross_solver_consensus


def hypothesis(transform, family):
    return _hypothesis(family=family, solver=family, transform=transform,
                       support_count=30, correspondence_count=30,
                       threshold_m=.1, certificate={})


def geometry_fixture(stack, hypotheses):
    points = np.random.default_rng(9).normal(size=(30, 3))
    stack.enter_context(patch("pose_pipeline.geometry_backend.fpfh_correspondences",
                             return_value=(points, points, points, points)))
    stack.enter_context(patch("pose_pipeline.geometry_backend.generate_hypotheses",
                             return_value={"hypotheses": hypotheses}))
    icp = stack.enter_context(patch("pose_pipeline.geometry_backend._icp", return_value=np.eye(4)))
    return points, icp


def test_independent_pnp_adds_a_family_without_relaxing_consensus():
    geometry = hypothesis(np.eye(4), "compatibility_graph")
    pnp = hypothesis(np.eye(4), "rgbd_pnp")
    with ExitStack() as stack:
        points, icp = geometry_fixture(stack, [geometry])
        original = _one_direction(points, points, RobustPoseConfig(), GeometryBootstrapConfig())
        assert original["accepted"] is False
        icp.assert_not_called()
        combined = _one_direction(points, points, RobustPoseConfig(), GeometryBootstrapConfig(), pnp)
    assert combined["accepted"] is True
    assert combined["consensus"]["solver_families"] == ["compatibility_graph", "rgbd_pnp"]
    assert combined["verification"]["minimum_overlap"] == 1.0


def test_pnp_alone_or_two_pnp_directions_cannot_supply_geometry_evidence():
    pnp = hypothesis(np.eye(4), "rgbd_pnp")
    assert not select_cross_solver_consensus([pnp, pnp])["accepted"]
    with ExitStack() as stack:
        points, icp = geometry_fixture(stack, [])
        result = _one_direction(points, points, RobustPoseConfig(), GeometryBootstrapConfig(), pnp)
        icp.assert_not_called()
    assert result["accepted"] is False


def test_visual_witness_cannot_rescue_disagreeing_geometric_solution():
    offset = np.eye(4)
    offset[0, 3] = .10  # Within historical PnP recall tolerance but outside 5 cm consensus.
    with ExitStack() as stack:
        points, icp = geometry_fixture(stack, [hypothesis(np.eye(4), "compatibility_graph")])
        result = _one_direction(points, points, RobustPoseConfig(), GeometryBootstrapConfig(),
                                hypothesis(offset, "rgbd_pnp"))
        icp.assert_not_called()
    assert result["reason"] == "no_cross_solver_cluster"


def test_visual_witness_requires_real_geometry_in_both_directions():
    witness = {"schema": "rgbd_visual_loop_estimate.v1", "accepted": True, "gt_consumed": False,
               "forward": {"transform": np.eye(4).tolist(), "inliers": 30, "depth_correspondences": 30},
               "reverse": {"transform": np.eye(4).tolist(), "inliers": 30, "depth_correspondences": 30}}
    with ExitStack() as stack:
        points, _ = geometry_fixture(stack, [])
        stack.enter_context(patch("pose_pipeline.geometry_backend.generate_hypotheses", side_effect=[
            {"hypotheses": [hypothesis(np.eye(4), "compatibility_graph")]}, {"hypotheses": []},
        ]))
        result = register_submaps_bidirectional(points, points, visual_evidence=witness)
    assert result["forward"]["accepted"]
    assert not result["reverse"]["accepted"]
    assert result["reason"] == "direction_failed"
    with pytest.raises(ValueError, match="two solver families"):
        register_submaps_bidirectional(points, points, RobustPoseConfig(minimum_solver_families=1),
                                       visual_evidence=witness)
