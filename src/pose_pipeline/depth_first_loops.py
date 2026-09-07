"""Depth-first loop proposals with disjoint multi-view acceptance.

PnP is neither an initializer nor a prerequisite. Geometry solvers and
odometry-seeded ICP propose transforms; only held-out depth can release them.
No GT, mesh, semantic object ID, or scene-specific frame ID is consumed.
"""
from dataclasses import asdict, dataclass, replace
import copy
import numpy as np
from .contracts import validate_se3
from .depth_loop_witness import DepthWitnessConfig, verify_depth_loop
from .geometry_backend import GeometryBootstrapConfig, register_submaps_bidirectional, dense_verification
from .lightweight_rgbd import LightweightConfig, anchor_clouds
from .robust_backend import RobustPoseConfig, transform_distance


@dataclass(frozen=True)
class DepthFirstPipelineConfig:
    enabled: bool = False

    def __post_init__(self):
        if not isinstance(self.enabled,bool):raise ValueError('enabled must be boolean')


@dataclass(frozen=True)
class DepthFirstConfig:
    icp_distances_m: tuple[float,...] = (.60,.30,.15,.075)
    icp_voxels_m: tuple[float,...] = (.10,.075,.05,.025)
    minimum_overlap: float = .30
    maximum_trimmed_rmse_m: float = .05
    maximum_cycle_translation_m: float = .05
    maximum_cycle_rotation_deg: float = 3.
    maximum_initial_update_m: float = 1.
    maximum_initial_update_deg: float = 20.
    ambiguity_translation_m: float = .10
    ambiguity_rotation_deg: float = 5.
    minimum_observability_ratio: float = 1.e-5

    def __post_init__(self):
        if len(self.icp_distances_m)!=len(self.icp_voxels_m) or not self.icp_voxels_m:
            raise ValueError('ICP scale mismatch')
        if not 0<self.minimum_overlap<=1:raise ValueError('overlap')
        for k,v in asdict(self).items():
            values=v if isinstance(v,tuple) else (v,)
            if any(not np.isfinite(x) or x<=0 for x in values):raise ValueError(k)


def compact_clouds(manifest,poses,ordinals):
    # Only -8,0,+8 are used for fitting. The lightweight +4 check is not
    # consumed by registration or acceptance in this module.
    return anchor_clouds(manifest,poses,ordinals,LightweightConfig())


def _multiscale_icp(source,target,initial,config):
    import open3d as o3d
    transform=validate_se3(initial)
    for voxel,distance in zip(config.icp_voxels_m,config.icp_distances_m):
        s=source.voxel_down_sample(voxel);t=target.voxel_down_sample(voxel)
        if min(len(s.points),len(t.points))<200:raise ValueError('insufficient scale support')
        t.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=40))
        result=o3d.pipelines.registration.registration_icp(s,t,distance,transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        transform=validate_se3(result.transformation)
    return transform


def icp_proposal(source,target,initial,config=DepthFirstConfig()):
    import open3d as o3d
    from scipy.spatial import cKDTree
    forward=_multiscale_icp(source,target,initial,config)
    reverse=_multiscale_icp(target,source,np.linalg.inv(initial),config)
    rotation,translation=transform_distance(forward,np.linalg.inv(reverse))
    update_r,update_t=transform_distance(initial,forward)
    metrics=dense_verification(np.asarray(source.points),np.asarray(target.points),forward,.075)
    check=copy.deepcopy(target)
    check.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.10,max_nn=40))
    moved=np.asarray(source.points)@forward[:3,:3].T+forward[:3,3]
    distance,index=cKDTree(np.asarray(check.points)).query(moved)
    support=distance<=.075;points=moved[support];normals=np.asarray(check.normals)[index[support]]
    if len(points)>=200:
        centered=points-points.mean(0);scale=max(float(np.sqrt(np.mean(centered**2))),.1)
        jacobian=np.c_[np.cross(centered/scale,normals),normals]
        eigen=np.linalg.eigvalsh(jacobian.T@jacobian/len(points))
        observability=float(max(eigen[0],0)/max(eigen[-1],1.e-12))
    else:observability=0.
    usable=(rotation<=config.maximum_cycle_rotation_deg and translation<=config.maximum_cycle_translation_m
        and update_r<=config.maximum_initial_update_deg and update_t<=config.maximum_initial_update_m
        and metrics['minimum_overlap']>=config.minimum_overlap
        and metrics['trimmed_rmse_m']<=config.maximum_trimmed_rmse_m
        and observability>=config.minimum_observability_ratio)
    return dict(provider='bidirectional_multiscale_depth_icp',proposal_usable=bool(usable),
        transform=forward.tolist(),cycle_rotation_deg=rotation,cycle_translation_m=translation,
        update_rotation_deg=update_r,update_translation_m=update_t,geometry=metrics,
        observability_ratio=observability,gt_consumed=False)


def choose_depth_verified(proposals,config=DepthFirstConfig()):
    passed=[p for p in proposals if p.get('depth_witness',{}).get('accepted')]
    if not passed:return None,'no_heldout_depth_pass'
    for i,a in enumerate(passed):
        for b in passed[i+1:]:
            rotation,translation=transform_distance(a['transform'],b['transform'])
            if rotation>config.ambiguity_rotation_deg or translation>config.ambiguity_translation_m:
                return None,'ambiguous_depth_supported_transforms'
    def loss(p):
        return np.mean([d['candidate']['capped_loss_m'] for r in p['depth_witness']['view_pairs']
            for d in r['directions'] if d['candidate']['capped_loss_m'] is not None])
    return min(passed,key=loss),'heldout_depth_verified'


def recover_pair(manifest,poses,ordinals,source,target,clouds,bindings,
                 robust_config=RobustPoseConfig(),geometry_config=GeometryBootstrapConfig(),
                 config=DepthFirstConfig()):
    s=ordinals[source];t=ordinals[target]
    initial=np.linalg.inv(poses[t].t_world_camera)@poses[s].t_world_camera
    fit_ids={r['frame_id'] for i in (source,target) for r in bindings[i]['frames'] if r['role']=='fit'}
    # These views are outside the compact fitting set and close enough to
    # reduce contamination from long DPV transport intervals. Their poses
    # still use local DPV motion; this correlation is reported explicitly.
    witness_config=DepthWitnessConfig(offsets=(-6,-2,2,6))
    proposals=[]
    try:
        registration=register_submaps_bidirectional(np.asarray(clouds[source].points),np.asarray(clouds[target].points),
            robust_config,replace(geometry_config,preconsensus_geometric_icp=False),visual_evidence=None)
        record=dict(provider='independent_fpfh_geometry',proposal_usable=bool(registration['accepted']),registration=registration)
        if registration['accepted']:record['transform']=registration['transform']
        proposals.append(record)
    except (ValueError,RuntimeError,ImportError) as e:
        proposals.append(dict(provider='independent_fpfh_geometry',proposal_usable=False,error=str(e)))
    try:proposals.append(icp_proposal(clouds[source],clouds[target],initial,config))
    except (ValueError,RuntimeError,ImportError) as e:
        proposals.append(dict(provider='bidirectional_multiscale_depth_icp',proposal_usable=False,error=str(e)))
    for p in proposals:
        if p['proposal_usable']:
            p['depth_witness']=verify_depth_loop(manifest,poses,s,t,np.asarray(p['transform']),fit_ids,witness_config)
    selected,reason=choose_depth_verified(proposals,config)
    return dict(source=source,target=target,source_frame=poses[s].frame_id,target_frame=poses[t].frame_id,
        accepted=selected is not None,reason=reason,proposals=proposals,gt_consumed=False,
        pnp_consumed=False,local_pose_transport='cached_DPV_not_independent_motion_estimator',
        selected_provider=None if selected is None else selected['provider'],
        transform=None if selected is None else selected['transform'],fit_frame_ids=sorted(fit_ids))


def build_depth_first_edges(manifest,poses,ordinals,evidence,original_edges,robust_config,geometry_config):
    """Public-runner integration of the same frozen experimental recipe."""
    from .lightweight_rgbd import refine_pair
    from .pose_graph import PoseGraphEdge
    from .depth_loop_witness import inconsistent_triangles
    clouds,checks,bindings=compact_clouds(manifest,poses,ordinals)
    fixed=list(original_edges);local_rows=[]
    for i in range(len(ordinals)-1):
        j=i+1;initial=np.linalg.inv(poses[ordinals[j]].t_world_camera)@poses[ordinals[i]].t_world_camera
        row=refine_pair(clouds[i],clouds[j],checks[i],checks[j],initial,roi=False,colored=False)
        local_rows.append(dict(source=i,target=j,result=row))
        if row['accepted']:fixed.append(PoseGraphEdge(i,j,np.asarray(row['transform']),kind='local_rgbd',
            information=np.asarray(row['information']),provenance='lightweight_heldout_checked_plain'))
    rows=[];new=[]
    for proposal in evidence:
        if proposal['edge_verified']:continue
        r=recover_pair(manifest,poses,ordinals,proposal['source_anchor_index'],proposal['target_anchor_index'],
            clouds,bindings,robust_config,geometry_config)
        rows.append(r)
        if r['accepted']:new.append(PoseGraphEdge(r['source'],r['target'],np.asarray(r['transform']),kind='depth_first_loop',
            weight=1.,provenance='independent_depth_first_plus_disjoint_multiview'))
    bad,cycles=inconsistent_triangles(fixed+new,{(e.source,e.target) for e in new})
    degree={i:0 for i in range(len(ordinals))}
    for e in fixed:degree[e.source]+=1;degree[e.target]+=1
    retained=[];dropped=[]
    for e in new:
        if (e.source,e.target) in bad:reason='inconsistent_measured_triangle'
        elif max(degree[e.source],degree[e.target])>=4:reason='degree_budget'
        else:retained.append(e);degree[e.source]+=1;degree[e.target]+=1;continue
        dropped.append(dict(source=e.source,target=e.target,reason=reason))
    return fixed+retained,retained,dict(gt_consumed=False,rows=rows,local_rows=local_rows,rgbd_bindings=bindings,
        proposed_depth_edges=len(new),retained_depth_edges=len(retained),dropped=dropped,cycle_check=cycles)
