"""Independent geometric refinement must preserve the visual evidence gate."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from pose_pipeline.geometry_backend import (
    GeometryBootstrapConfig,
    _one_direction,
    register_submaps_bidirectional,
)
from pose_pipeline.robust_backend import RobustPoseConfig, _hypothesis, transform_points


CONFIG = GeometryBootstrapConfig(preconsensus_geometric_icp=True)


def hypothesis(transform, family="compatibility_graph"):
    return _hypothesis(
        family=family, solver=family, transform=transform, support_count=30,
        correspondence_count=30, threshold_m=.1, certificate={},
    )


def translate(x=0.0, y=0.0, z=0.0):
    value = np.eye(4)
    value[:3, 3] = [x, y, z]
    return value


def three_planes():
    rng = np.random.default_rng(23)
    parts = []
    for axis in range(3):
        cloud = rng.uniform(-.8, .8, (240, 3))
        cloud[:, axis] = -.55 + .03 * cloud[:, (axis + 1) % 3] ** 2
        parts.append(cloud)
    return np.concatenate(parts)


def fixture(stack, source, reference, hypotheses):
    stack.enter_context(patch(
        "pose_pipeline.geometry_backend.fpfh_correspondences",
        return_value=(source, reference, source, reference),
    ))
    stack.enter_context(patch(
        "pose_pipeline.geometry_backend.generate_hypotheses",
        side_effect=lambda *args, **kwargs: {"hypotheses": deepcopy(hypotheses)},
    ))


def test_real_icp_recovers_seven_cm_raw_disagreement_without_changing_visual():
    pytest.importorskip("open3d")
    source = three_planes()
    truth = translate(.4, -.2, .15)
    reference = transform_points(source, truth)
    raw = translate(.47, -.2, .15)
    geometric = hypothesis(raw)
    visual = hypothesis(truth, "rgbd_pnp")
    original_visual = deepcopy(visual)
    with ExitStack() as stack:
        fixture(stack, source, reference, [geometric])
        original = _one_direction(
            source, reference, RobustPoseConfig(), GeometryBootstrapConfig(), visual,
        )
        assert not original["accepted"]
        assert original["reason"] == "no_cross_solver_cluster"
        result = _one_direction(source, reference, RobustPoseConfig(), CONFIG, visual)
    assert result["accepted"]
    np.testing.assert_allclose(result["transform"], truth, atol=1e-5)
    records = result["hypothesis_set"]["preconsensus_refinement"]["records"]
    assert records[0]["accepted"]
    assert records[0]["update_translation_m"] == pytest.approx(.07, abs=1e-5)
    assert result["consensus"]["solver_families"] == ["compatibility_graph", "rgbd_pnp"]
    assert visual == original_visual
    assert result["hypothesis_set"]["hypotheses"][-1] == visual


def test_real_icp_cannot_rescue_wrong_visual_or_missing_geometric_family():
    pytest.importorskip("open3d")
    points = three_planes()
    wrong_visual = hypothesis(translate(.30), "rgbd_pnp")
    with ExitStack() as stack:
        fixture(stack, points, points, [hypothesis(translate(.07))])
        result = _one_direction(points, points, RobustPoseConfig(), CONFIG, wrong_visual)
    assert not result["accepted"]
    assert result["reason"] == "no_cross_solver_cluster"
    with ExitStack() as stack:
        fixture(stack, points, points, [])
        result = _one_direction(
            points, points, RobustPoseConfig(), CONFIG, hypothesis(np.eye(4), "rgbd_pnp"),
        )
    assert not result["accepted"]


def test_converged_three_d_families_cannot_self_certify_without_visual_in_clique():
    points = three_planes()
    with ExitStack() as stack:
        fixture(stack, points, points, [
            hypothesis(translate(.07)), hypothesis(translate(-.04), "pygcransac"),
        ])
        icp = stack.enter_context(patch(
            "pose_pipeline.geometry_backend._icp", return_value=np.eye(4),
        ))
        result = _one_direction(
            points, points, RobustPoseConfig(), CONFIG,
            hypothesis(translate(.30), "rgbd_pnp"),
        )
    assert not result["accepted"]
    assert result["reason"] == "preconsensus_icp_requires_visual_family_in_winning_cluster"
    assert icp.call_count == 2  # No final ICP after rejection.


def test_large_preconsensus_update_is_removed_before_consensus():
    points = three_planes()
    with ExitStack() as stack:
        fixture(stack, points, points, [hypothesis(translate(.11))])
        stack.enter_context(patch("pose_pipeline.geometry_backend._icp", return_value=np.eye(4)))
        result = _one_direction(
            points, points, RobustPoseConfig(), CONFIG, hypothesis(np.eye(4), "rgbd_pnp"),
        )
    assert not result["accepted"]
    record = result["hypothesis_set"]["preconsensus_refinement"]["records"][0]
    assert record["reason"] == "preconsensus_icp_update_limit"
    assert result["hypothesis_set"]["preconsensus_refinement"]["retained_hypothesis_count"] == 0


def test_staging_icp_does_not_expand_original_final_update_cap():
    source = three_planes()
    reference = transform_points(source, translate(.06))
    with ExitStack() as stack:
        fixture(stack, source, reference, [hypothesis(np.eye(4))])
        stack.enter_context(patch(
            "pose_pipeline.geometry_backend._icp",
            side_effect=[translate(.06), translate(.24)],
        ))
        result = _one_direction(
            source, reference, RobustPoseConfig(), CONFIG,
            hypothesis(translate(.06), "rgbd_pnp"),
        )
    assert not result["accepted"]
    assert result["reason"] == "preconsensus_cumulative_icp_update_limit"
    assert result["icp_update_translation_m"] == pytest.approx(.24)


def test_original_accepted_path_does_not_run_recovery_icp():
    points = three_planes()
    with ExitStack() as stack:
        fixture(stack, points, points, [hypothesis(np.eye(4))])
        icp = stack.enter_context(patch(
            "pose_pipeline.geometry_backend._icp", return_value=np.eye(4),
        ))
        result = _one_direction(
            points, points, RobustPoseConfig(), CONFIG,
            hypothesis(np.eye(4), "rgbd_pnp"),
        )
    assert result["accepted"]
    assert icp.call_count == 1  # The original final ICP only.
    assert "preconsensus_refinement" not in result["hypothesis_set"]


def test_refinement_is_opt_in_and_requires_visual_and_finite_limits():
    assert not GeometryBootstrapConfig().preconsensus_geometric_icp
    with pytest.raises(ValueError, match="independent visual witness"):
        register_submaps_bidirectional(three_planes(), three_planes(), geometry_config=CONFIG)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            replace(CONFIG, preconsensus_maximum_translation_m=value)
        with pytest.raises(ValueError, match="finite and positive"):
            replace(CONFIG, preconsensus_maximum_rotation_deg=value)
