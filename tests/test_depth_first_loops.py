from types import SimpleNamespace
import numpy as np
import pytest
from pose_pipeline import depth_first_loops as d


def supported(transform,loss=.02):
    return dict(transform=transform.tolist(),depth_witness=dict(accepted=True,
        view_pairs=[dict(directions=[dict(candidate=dict(capped_loss_m=loss))])]))


def test_conflicting_depth_supported_hypotheses_are_not_selected():
    a=np.eye(4);b=np.eye(4);b[0,3]=.3
    chosen,reason=d.choose_depth_verified([supported(a),supported(b)])
    assert chosen is None and reason=='ambiguous_depth_supported_transforms'


def test_selects_lowest_depth_loss_only_within_one_basin():
    a=np.eye(4);b=np.eye(4);b[0,3]=.02
    x=supported(a,.03);y=supported(b,.02)
    assert d.choose_depth_verified([x,y])[0] is y


def test_geometry_recovery_does_not_require_visual(monkeypatch):
    seen=[]
    def geometry(source,target,robust,config,*,visual_evidence):
        assert visual_evidence is None and not config.preconsensus_geometric_icp
        seen.append('geometry');return dict(accepted=True,transform=np.eye(4).tolist())
    def witness(manifest,poses,s,t,transform,excluded,config):
        assert excluded=={100,108,116,120,128,136}
        assert config.offsets==(-6,-2,2,6)
        seen.append('depth');return supported(np.eye(4))['depth_witness']
    monkeypatch.setattr(d,'register_submaps_bidirectional',geometry)
    monkeypatch.setattr(d,'icp_proposal',lambda *a:dict(provider='icp',proposal_usable=False))
    monkeypatch.setattr(d,'verify_depth_loop',witness)
    poses=[SimpleNamespace(t_world_camera=np.eye(4),frame_id=100+i) for i in range(40)]
    clouds=[SimpleNamespace(points=np.ones((200,3)))]*2
    bindings=[dict(frames=[dict(frame_id=f,role='fit') for f in group]+[dict(frame_id=999,role='check')])
        for group in [[100,108,116],[120,128,136]]]
    result=d.recover_pair(None,poses,[8,28],0,1,clouds,bindings)
    assert result['accepted'] and result['pnp_consumed'] is False and seen==['geometry','depth']


def test_single_plane_is_not_a_fully_observable_loop():
    o3d=pytest.importorskip('open3d')
    x,y=np.meshgrid(np.linspace(-2,2,61),np.linspace(-2,2,61))
    points=np.c_[x.ravel(),y.ravel(),np.ones(x.size)]
    c=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    result=d.icp_proposal(c,c,np.eye(4))
    assert not result['proposal_usable'] and result['observability_ratio']<1.e-5


def test_three_surfaces_recover_translation():
    o3d=pytest.importorskip('open3d')
    rng=np.random.default_rng(731)
    p=rng.uniform(-2,2,(6000,3))
    p[:2000,0]=2;p[2000:4000,1]=2;p[4000:,2]=2
    a=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    transform=np.eye(4);transform[:3,3]=[.08,-.06,.04]
    b=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p+transform[:3,3]))
    result=d.icp_proposal(a,b,np.eye(4))
    assert result['proposal_usable']
    assert np.linalg.norm(np.asarray(result['transform'])[:3,3]-transform[:3,3])<.005
