"""Opt-in held-out projective depth corroboration of frozen geometric loops.

No transform is estimated or changed here. GT-free additional frames may
corroborate a geometry-accepted hypothesis rejected by the sparse PnP witness.
The result is experimental evidence, not an unconditional promotion decision.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .contracts import validate_se3, sha256_file


@dataclass(frozen=True)
class DepthWitnessConfig:
    offsets: tuple[int, ...] = (-32, -24, 24, 32)
    pixel_stride: int = 4
    minimum_points: int = 200
    minimum_view_pairs: int = 2
    minimum_pass_fraction: float = .75
    minimum_projection_fraction: float = .15
    minimum_visible_fraction: float = .35
    minimum_inlier_fraction: float = .55
    minimum_coverage_retention: float = .8
    maximum_median_error_m: float = .05
    inlier_distance_m: float = .10
    occlusion_distance_m: float = .10
    loss_cap_m: float = .20
    minimum_loss_improvement: float = .20

    def __post_init__(self):
        if len(set(self.offsets)) != len(self.offsets) or 0 in self.offsets:
            raise ValueError('witness offsets must be distinct and exclude anchor')
        if min(self.pixel_stride,self.minimum_points,self.minimum_view_pairs)<1:
            raise ValueError('invalid witness sample budgets')
        for name in ('minimum_pass_fraction','minimum_projection_fraction','minimum_visible_fraction',
                     'minimum_inlier_fraction','minimum_coverage_retention'):
            if not 0 < getattr(self,name) <= 1:raise ValueError(name)
        if not 0 <= self.minimum_loss_improvement < 1:raise ValueError('loss improvement')
        for name in ('maximum_median_error_m','inlier_distance_m','occlusion_distance_m','loss_cap_m'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError(name)


def witness_ordinals(source, target, count, excluded, config=DepthWitnessConfig()):
    """Exclude actual source/target submap samples and anchor images."""
    excluded=set(excluded)|{source,target}
    return [(source+d,target+d) for d in config.offsets
            if 0<=source+d<count and 0<=target+d<count
            and source+d not in excluded and target+d not in excluded]


def projective_depth_stats(source_depth,target_depth,source_k,target_k,transform,config=DepthWitnessConfig()):
    transform=validate_se3(transform)
    source_depth=np.asarray(source_depth,dtype=float);target_depth=np.asarray(target_depth,dtype=float)
    if source_depth.ndim!=2 or target_depth.ndim!=2:raise ValueError('depth must be HxW meters')
    yy,xx=np.mgrid[0:source_depth.shape[0]:config.pixel_stride,0:source_depth.shape[1]:config.pixel_stride]
    z=source_depth[::config.pixel_stride,::config.pixel_stride]
    valid=np.isfinite(z)&(z>=.3)&(z<=4.5)
    fx,fy,cx,cy=source_k
    xyz=np.column_stack(((xx[valid]-cx)*z[valid]/fx,(yy[valid]-cy)*z[valid]/fy,z[valid]))
    total=len(xyz)
    xyz=xyz@transform[:3,:3].T+transform[:3,3]
    xyz=xyz[np.isfinite(xyz).all(1)&(xyz[:,2]>=.3)&(xyz[:,2]<=4.5)]
    fx,fy,cx,cy=target_k
    u=np.rint(fx*xyz[:,0]/xyz[:,2]+cx).astype(int);v=np.rint(fy*xyz[:,1]/xyz[:,2]+cy).astype(int)
    inside=(u>=0)&(u<target_depth.shape[1])&(v>=0)&(v<target_depth.shape[0])
    measured=target_depth[v[inside],u[inside]];predicted=xyz[inside,2]
    valid=np.isfinite(measured)&(measured>=.3)&(measured<=4.5)
    error=predicted[valid]-measured[valid]
    n=len(error);visible=error<=config.occlusion_distance_m
    # Large positive residuals may be occluded, but they still contribute to
    # the capped loss and reduce visible/inlier fractions. Hiding points cannot
    # turn a bad transform into a perfect score.
    residual=np.abs(error[visible])
    return {'sampled_points':total,'projected_points':n,'visible_points':int(visible.sum()),
            'projection_fraction':n/max(total,1),'visible_fraction':float(visible.sum()/max(n,1)),
            'inlier_fraction':float(np.count_nonzero(np.abs(error)<=config.inlier_distance_m)/max(n,1)),
            'free_space_fraction':float(np.count_nonzero(error < -config.inlier_distance_m)/max(n,1)),
            'visible_median_m':float(np.median(residual)) if len(residual) else None,
            'capped_loss_m':float(np.mean(np.minimum(np.abs(error),config.loss_cap_m))) if n else None}


def direction_pass(candidate,baseline,config=DepthWitnessConfig()):
    return bool(candidate['projected_points']>=config.minimum_points
        and candidate['visible_points']>=config.minimum_points
        and candidate['projection_fraction']>=config.minimum_projection_fraction
        and candidate['projection_fraction']>=config.minimum_coverage_retention*baseline['projection_fraction']
        and candidate['visible_fraction']>=config.minimum_visible_fraction
        and candidate['inlier_fraction']>=config.minimum_inlier_fraction
        and candidate['visible_median_m'] is not None
        and candidate['visible_median_m']<=config.maximum_median_error_m
        and baseline['capped_loss_m'] is not None
        and candidate['capped_loss_m'] <= (1-config.minimum_loss_improvement)*baseline['capped_loss_m'])


def verify_depth_loop(manifest,poses,source,target,transform,excluded_frame_ids,config=DepthWitnessConfig()):
    from reconstruction.rgbd_refusion import _read_rgbd
    transform=validate_se3(transform)
    index={p.frame_id:i for i,p in enumerate(poses)}
    excluded={index[f] for f in excluded_frame_ids}
    pairs=witness_ordinals(source,target,len(poses),excluded,config)
    frames={f.frame_id:f for f in manifest.frames}
    rows=[];bindings=[]
    for s,t in pairs:
        sf=frames[poses[s].frame_id];tf=frames[poses[t].frame_id]
        _,sd,sk=_read_rgbd(sf);_,td,tk=_read_rgbd(tf)
        sd=sd.astype(float)/manifest.depth_scale;td=td.astype(float)/manifest.depth_scale
        initial=np.linalg.inv(poses[t].t_world_camera)@poses[s].t_world_camera
        measured=(np.linalg.inv(poses[t].t_world_camera)@poses[target].t_world_camera@transform
                  @np.linalg.inv(poses[source].t_world_camera)@poses[s].t_world_camera)
        directions=[]
        for a,b,ak,bk,init,hyp in [(sd,td,sk,tk,initial,measured),(td,sd,tk,sk,np.linalg.inv(initial),np.linalg.inv(measured))]:
            bs=projective_depth_stats(a,b,ak,bk,init,config);cs=projective_depth_stats(a,b,ak,bk,hyp,config)
            directions.append({'baseline':bs,'candidate':cs,'passes':direction_pass(cs,bs,config)})
        rows.append({'source_frame':sf.frame_id,'target_frame':tf.frame_id,'directions':directions,
                     'passes':all(d['passes'] for d in directions)})
        for f in (sf,tf):bindings.append({'frame_id':f.frame_id,'depth_sha256':sha256_file(f.depth_path),
                                        'color_sha256':sha256_file(f.color_path)})
    passed=sum(r['passes'] for r in rows)
    accepted=len(rows)>=config.minimum_view_pairs and passed>=int(np.ceil(config.minimum_pass_fraction*len(rows)))
    return {'schema':'heldout_projective_depth_loop.v1','gt_consumed':False,'diagnostic_only':True,
            'accepted':accepted,'reason':'multi_view_bidirectional_depth_pass' if accepted else 'insufficient_multi_view_depth_support',
            'config':asdict(config),'view_pairs':rows,'passed_view_pairs':passed,'view_pair_count':len(rows),
            'excluded_frame_ids':sorted(excluded_frame_ids),'rgbd_sha256':bindings,
            'source_frame':poses[source].frame_id,'target_frame':poses[target].frame_id}


def inconsistent_triangles(edges,new_pairs,translation_limit=.10,rotation_limit_deg=5.):
    """Check only triangles of measured edges, never fabricate DPV loop votes."""
    directed={}
    for edge in edges:
        directed[edge.source,edge.target]=edge.source_to_target
        directed[edge.target,edge.source]=np.linalg.inv(edge.source_to_target)
    nodes=sorted({i for pair in directed for i in pair});bad=set();rows=[]
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes[i+1:],i+1):
            for c in nodes[j+1:]:
                if not all(p in directed for p in [(a,b),(b,c),(c,a)]):continue
                pairs={(a,b),(b,c),(a,c)}
                affected=pairs & new_pairs
                if not affected:continue
                cycle=directed[c,a]@directed[b,c]@directed[a,b]
                translation=float(np.linalg.norm(cycle[:3,3]))
                rotation=float(np.degrees(np.arccos(np.clip((np.trace(cycle[:3,:3])-1)/2,-1,1))))
                passed=translation<=translation_limit and rotation<=rotation_limit_deg
                rows.append({'nodes':[a,b,c],'translation_m':translation,'rotation_deg':rotation,'passes':passed})
                if not passed:bad.update(affected)
    return bad,{'assessed_triangles':rows,'status':'checked' if rows else 'no_measured_triangle_available'}
