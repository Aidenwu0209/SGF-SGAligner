from dataclasses import replace
import numpy as np
import pytest

from pose_pipeline.lightweight_rgbd import (
    LightweightConfig, frame_indices, source_information, overlap_roi, refine_pair,
)
from pose_pipeline.pose_graph import (
    PoseGraphEdge, PoseGraphOptimizationConfig, normalized_local_information_root,
    optimize_pose_graph, _exp_se3,
)


def test_check_frames_disjoint_at_both_boundaries():
    for anchor in (0,80,159):
        fit,check=frame_indices(anchor,160,LightweightConfig())
        assert check not in fit
        assert min(fit+[check])>=0 and max(fit+[check])<160
    with pytest.raises(ValueError):LightweightConfig(check_offset=8)


def test_information_matches_right_source_perturbation_with_rotation():
    rng=np.random.default_rng(10)
    points=rng.normal(size=(50,3));normals=rng.normal(size=(50,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
    transform=_exp_se3(np.array([0.2,-0.1,0.3,0.4,0.1,-0.2]))
    moved=points@transform[:3,:3].T+transform[:3,3]
    jac=[]
    for j in range(6):
        delta=np.zeros(6);delta[j]=1e-6
        perturbed=transform@_exp_se3(delta)
        p=points@perturbed[:3,:3].T+perturbed[:3,3]
        jac.append(np.sum((p-moved)*normals,axis=1)/1e-6)
    numeric=np.asarray(jac).T
    h=source_information(points,normals,transform[:3,:3])
    np.testing.assert_allclose(h,numeric.T@numeric/50,atol=3e-6)


def test_information_is_scale_invariant_and_never_amplifies():
    h=np.diag([1.,2.,3.,0.0001,4.,5.])
    root=normalized_local_information_root(h,0.03,0.04)
    np.testing.assert_allclose(root,normalized_local_information_root(1000*h,0.03,0.04))
    assert np.linalg.eigvalsh(root).max()<=1+1e-12
    with pytest.raises(ValueError):normalized_local_information_root(-np.eye(6),0.03,0.04)


def test_directional_weight_reduces_unobserved_axis_pull():
    t=np.eye(4);t[:2,3]=[0.08,0.08]
    edge=PoseGraphEdge(0,1,t,'local_rgbd',information=np.diag([1.,1.,1.,0.0001,1.,1.]))
    scalar,_=optimize_pose_graph([np.eye(4),np.eye(4)],[edge])
    weighted,_=optimize_pose_graph([np.eye(4),np.eye(4)],[edge],optimization_config=PoseGraphOptimizationConfig(huber_information_policy='local_normalized'))
    assert abs(weighted[1][0,3])<abs(scalar[1][0,3])*0.5
    assert abs(weighted[1][1,3])>abs(weighted[1][0,3])*3
    ordinary=replace(edge,kind='loop')
    unchanged,_=optimize_pose_graph([np.eye(4),np.eye(4)],[ordinary],optimization_config=PoseGraphOptimizationConfig(huber_information_policy='local_normalized'))
    np.testing.assert_allclose(unchanged,scalar,atol=1e-10)


def cloud(points):
    import open3d as o3d
    c=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    c.colors=o3d.utility.Vector3dVector(np.clip((points+1)/2,0,1))
    return c


def test_overlap_roi_preserves_both_sides_and_colors():
    a=cloud(np.array([[0.,0,0],[1.,0,0],[8.,0,0]]));b=cloud(np.array([[0.01,0,0],[1.01,0,0],[-8.,0,0]]))
    x,y=overlap_roi(a,b,np.eye(4),0.1)
    assert len(x.points)==len(y.points)==2
    assert len(x.colors)==2
    np.testing.assert_allclose(np.asarray(x.points),np.asarray(a.points)[:2])


def make_corner(seed,translation):
    rng=np.random.default_rng(seed);p=rng.uniform(-0.7,0.7,size=(3000,3))
    for i in range(3):p[i*1000:(i+1)*1000,i]=0
    return cloud(p+translation)


def test_real_icp_has_independent_depth_support_and_bounded_update():
    shift=np.array([0.025,-0.02,0.015])
    result=refine_pair(make_corner(1,0),make_corner(1,shift),make_corner(2,0),make_corner(2,shift),np.eye(4))
    assert result['accepted'],result
    np.testing.assert_allclose(np.asarray(result['transform'])[:3,3],shift,atol=0.005)
    assert result['check_forward']['after_plane_rmse_m']<result['check_forward']['before_plane_rmse_m']
    # A model that fits the training clouds cannot pass unrelated checking clouds.
    rejected=refine_pair(make_corner(1,0),make_corner(1,shift),make_corner(2,0),make_corner(2,0),np.eye(4))
    assert not rejected['accepted']


def test_excessive_updates_rejected_without_identity_measurement():
    shift=np.array([0.025,-0.02,0.015])
    result=refine_pair(make_corner(1,0),make_corner(1,shift),make_corner(2,0),make_corner(2,shift),np.eye(4),
                       config=LightweightConfig(maximum_update_translation_m=0.005))
    assert not result['accepted'] and result['reason']=='update_limit'


def test_empty_roi_is_explicit_failure():
    result=refine_pair(cloud(np.zeros((20,3))),cloud(np.ones((20,3))),cloud(np.zeros((20,3))),cloud(np.ones((20,3))),np.eye(4),roi=True)
    assert not result['accepted'] and result['reason']=='insufficient_roi_support'
