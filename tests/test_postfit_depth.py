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
