from types import SimpleNamespace
import numpy as np
import pytest
from pose_pipeline import postfit_depth as p


def test_postfit_frames_cannot_overlap_training_views():
    with pytest.raises(ValueError):p.PostfitDepthConfig(offsets=(-8,8))


@pytest.mark.parametrize('corrected,expected',[(True,True),(False,False)])
def test_new_views_verify_actual_pose_improvement(monkeypatch,corrected,expected):
    import reconstruction.rgbd_refusion as fusion
    depth=np.full((160,160),2000,dtype=np.uint16)
    monkeypatch.setattr(fusion,'_read_rgbd',lambda f:(None,depth,(100.,100.,80.,80.)))
    monkeypatch.setattr(p,'sha256_file',lambda path:str(path))
    frames=[SimpleNamespace(frame_id=i,color_path='c'+str(i),depth_path='d'+str(i)) for i in range(100)]
    base=[];candidate=[]
    for i in range(100):
        t=np.eye(4)
        if i>=50:t[2,3]=.3
        base.append(SimpleNamespace(frame_id=i,t_world_camera=t))
        candidate.append(SimpleNamespace(frame_id=i,t_world_camera=np.eye(4) if corrected else t.copy()))
    manifest=SimpleNamespace(frames=frames,depth_scale=1000.)
    edges=[SimpleNamespace(source=0,target=1)]
    result=p.audit_postfit_depth(manifest,base,candidate,[20,75],edges)
    assert result['passes'] is expected
    assert result['loop_count']==1
    used={x['frame_id'] for x in result['rgbd_sha256']}
    assert not used & {a+d for a in [20,75] for d in [-8,-6,-4,-2,0,2,4,6,8]}


def test_no_verified_loop_cannot_establish_improvement():
    m=SimpleNamespace(frames=[])
    assert not p.audit_postfit_depth(m,[],[],[],[])['passes']


def relative_audit(ratios):
    depth=dict(projected_points=500,visible_points=450,projection_fraction=.8,visible_fraction=.9)
    direction=dict(candidate=dict(depth),baseline=dict(depth))
    return dict(gt_consumed=False,passes=False,
        rows=[dict(mean_depth_loss_ratio=r,views=[dict(directions=[direction])]) for r in ratios],
        config=dict(minimum_loop_pass_fraction=.75,maximum_loop_loss_ratio=1.05),
        depth_thresholds=dict(minimum_loss_improvement=.20,minimum_points=200,minimum_projection_fraction=.15,
            minimum_coverage_retention=.8,minimum_visible_fraction=.35))


def test_relative_improvement_does_not_claim_absolute_accuracy():
    result=p.relative_improvement_decision(relative_audit([.5,.6,.7,.9]))
    assert result['passes'] and not result['absolute_alignment_passed']


@pytest.mark.parametrize('ratios',[[1.,1.,1.,1.],[.5,.5,.5,1.2],[]])
def test_unchanged_or_regressed_loop_is_not_an_improvement(ratios):
    assert not p.relative_improvement_decision(relative_audit(ratios))['passes']


def test_losing_projection_support_cannot_fake_an_improvement():
    audit=relative_audit([.2,.2,.2,.2])
    for row in audit['rows']:
        row['views'][0]['directions'][0]['candidate']['projected_points']=0
    assert not p.relative_improvement_decision(audit)['passes']
