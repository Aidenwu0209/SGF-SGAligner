from dataclasses import replace
import numpy as np
import pytest
from pose_pipeline.depth_loop_witness import (DepthWitnessConfig,projective_depth_stats,direction_pass,
                                             witness_ordinals,inconsistent_triangles)
from pose_pipeline.pose_graph import PoseGraphEdge

def scene():
    return np.full((80,100),2.),(90.,90.,49.5,39.5),replace(DepthWitnessConfig(),minimum_points=100)

def test_correct_depth_hypothesis_improves_displaced_baseline():
    d,k,c=scene();t=np.eye(4);t[2,3]=.3
    initial=projective_depth_stats(d,d,k,k,t,c);correct=projective_depth_stats(d,d,k,k,np.eye(4),c)
    assert direction_pass(correct,initial,c)

@pytest.mark.parametrize('shift',[-.35,.35,3.])
def test_wrong_transform_cannot_hide_depth_violations(shift):
    d,k,c=scene();t=np.eye(4);t[2,3]=shift
    good=projective_depth_stats(d,d,k,k,np.eye(4),c);bad=projective_depth_stats(d,d,k,k,t,c)
    assert not direction_pass(bad,good,c)

def test_invalid_depth_has_no_support():
    d,k,c=scene();blank=np.zeros_like(d)
    stats=projective_depth_stats(blank,d,k,k,np.eye(4),c)
    assert stats['projected_points']==0 and not direction_pass(stats,stats,c)

def test_witness_frames_are_excluded_from_fit_and_anchors():
    excluded=set(range(80,121))|set(range(280,321))|{124}
    pairs=witness_ordinals(100,300,500,excluded)
    assert pairs==[(68,268),(76,276),(132,332)]
    assert not any(i in excluded or j in excluded for i,j in pairs)

def test_consistent_and_inconsistent_measured_triangles():
    t=np.eye(4);t[0,3]=.2;two=np.eye(4);two[0,3]=.4
    edges=[PoseGraphEdge(0,1,t,kind="local_rgbd"),PoseGraphEdge(1,2,t,kind="local_rgbd"),PoseGraphEdge(0,2,two,kind="depth_verified_loop")]
    bad,report=inconsistent_triangles(edges,{(0,2)})
    assert not bad and report['status']=='checked'
    wrong=two.copy();wrong[2,3]=.5;edges[-1]=PoseGraphEdge(0,2,wrong,kind="depth_verified_loop")
    assert inconsistent_triangles(edges,{(0,2)})[0]=={(0,2)}

def test_no_triangle_is_not_a_consistency_pass():
    _,report=inconsistent_triangles([PoseGraphEdge(0,1,np.eye(4),kind="local_rgbd")],{(0,1)})
    assert report['status']=='no_measured_triangle_available'
