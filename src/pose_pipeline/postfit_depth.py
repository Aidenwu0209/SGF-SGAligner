"""Audit optimized poses on depth frames unused by fitting and loop admission."""
from dataclasses import asdict,dataclass
import numpy as np
from .contracts import sha256_file
from .depth_loop_witness import DepthWitnessConfig,projective_depth_stats,direction_pass,witness_ordinals


@dataclass(frozen=True)
class PostfitDepthConfig:
    offsets: tuple[int,...] = (-14,-10,10,14)
    minimum_loop_pass_fraction: float = .75
    maximum_loop_loss_ratio: float = 1.05

    def __post_init__(self):
        if not self.offsets or len(set(self.offsets))!=len(self.offsets):raise ValueError('offsets')
        if set(self.offsets)&{-8,-6,-4,-2,0,2,4,6,8}:raise ValueError('postfit view overlaps fitting/admission')
        if not 0<self.minimum_loop_pass_fraction<=1:raise ValueError('pass fraction')
        if not np.isfinite(self.maximum_loop_loss_ratio) or self.maximum_loop_loss_ratio<1:raise ValueError('loss ratio')


def audit_postfit_depth(manifest,baseline,candidate,ordinals,depth_edges,config=PostfitDepthConfig()):
    from reconstruction.rgbd_refusion import _read_rgbd
    if [p.frame_id for p in baseline]!=[p.frame_id for p in candidate]:raise ValueError('trajectory identity changed')
    if [f.frame_id for f in manifest.frames]!=[p.frame_id for p in baseline]:raise ValueError('manifest identity')
    # Retain the admission residual/coverage thresholds, but use different
    # RGB-D frames and evaluate final poses, not proposed loop transforms.
    checks=DepthWitnessConfig(offsets=config.offsets,pixel_stride=8)
    frames={f.frame_id:f for f in manifest.frames};cache={};bindings=[];rows=[]
    for edge in depth_edges:
        a=ordinals[edge.source];b=ordinals[edge.target]
        excluded={x+d for x in [a,b] for d in [-8,-6,-4,-2,0,2,4,6,8] if 0<=x+d<len(baseline)}
        pairs=witness_ordinals(a,b,len(baseline),excluded,checks);views=[]
        for s,t in pairs:
            for i in [s,t]:
                if i not in cache:
                    f=frames[baseline[i].frame_id];_,depth,k=_read_rgbd(f);cache[i]=(depth.astype(float)/manifest.depth_scale,k)
                    bindings.append(dict(frame_id=f.frame_id,color_sha256=sha256_file(f.color_path),depth_sha256=sha256_file(f.depth_path)))
            sd,sk=cache[s];td,tk=cache[t]
            old=np.linalg.inv(baseline[t].t_world_camera)@baseline[s].t_world_camera
            new=np.linalg.inv(candidate[t].t_world_camera)@candidate[s].t_world_camera
            directions=[]
            for d,e,dk,ek,x,y in [(sd,td,sk,tk,old,new),(td,sd,tk,sk,np.linalg.inv(old),np.linalg.inv(new))]:
                before=projective_depth_stats(d,e,dk,ek,x,checks);after=projective_depth_stats(d,e,dk,ek,y,checks)
                directions.append(dict(baseline=before,candidate=after,passes=direction_pass(after,before,checks)))
            views.append(dict(source_frame=baseline[s].frame_id,target_frame=baseline[t].frame_id,
                directions=directions,passes=all(d['passes'] for d in directions)))
        losses=[(d['baseline']['capped_loss_m'],d['candidate']['capped_loss_m']) for v in views for d in v['directions']
            if d['baseline']['capped_loss_m'] is not None and d['candidate']['capped_loss_m'] is not None]
        ratio=float(np.mean([y for x,y in losses])/max(np.mean([x for x,y in losses]),1.e-12)) if losses else None
        passed=sum(v['passes'] for v in views)
        rows.append(dict(source=edge.source,target=edge.target,views=views,passed_views=passed,
            passes=len(views)>=checks.minimum_view_pairs and passed>=int(np.ceil(checks.minimum_pass_fraction*len(views))),
            mean_depth_loss_ratio=ratio))
    passed=sum(r['passes'] for r in rows)
    no_regression=bool(rows) and all(r['mean_depth_loss_ratio'] is not None and r['mean_depth_loss_ratio']<=config.maximum_loop_loss_ratio for r in rows)
    accepted=bool(rows) and passed>=int(np.ceil(config.minimum_loop_pass_fraction*len(rows))) and no_regression
    return dict(schema='postfit_depth_audit.v1',gt_consumed=False,passes=accepted,loop_count=len(rows),passed_loops=passed,
        no_loop_mean_depth_regression=no_regression,config=asdict(config),depth_thresholds=asdict(checks),
        rows=rows,rgbd_sha256=bindings,scope='new RGBD views; geometry correspondence and local DPV errors may remain correlated')


def relative_improvement_decision(audit):
    """Decide relative improvement, separately from the absolute 5 cm audit.

    A final trajectory need not perfectly satisfy every loop to improve the
    reconstruction. Require new-view depth loss reduction, retained observable
    support, and no loop with a material mean-depth regression. Callers must
    additionally enforce full-map safety and trajectory bounds.
    """
    if audit.get('gt_consumed') is not False:raise ValueError('GT-free audit required')
    rows=audit['rows'];checks=audit['depth_thresholds'];config=audit['config']
    ratios=[r['mean_depth_loss_ratio'] for r in rows]
    improved=sum(v is not None and v<=1-checks['minimum_loss_improvement'] for v in ratios)
    coverage=[]
    for row in rows:
        for view in row['views']:
            for direction in view['directions']:
                a=direction['candidate'];b=direction['baseline']
                coverage.append(a['projected_points']>=checks['minimum_points']
                    and a['visible_points']>=checks['minimum_points']
                    and a['projection_fraction']>=checks['minimum_projection_fraction']
                    and a['projection_fraction']>=checks['minimum_coverage_retention']*b['projection_fraction']
                    and a['visible_fraction']>=checks['minimum_visible_fraction'])
    support=bool(coverage) and np.mean(coverage)>=config['minimum_loop_pass_fraction']
    gains=bool(rows) and improved>=int(np.ceil(config['minimum_loop_pass_fraction']*len(rows)))
    nonregression=bool(rows) and all(v is not None and np.isfinite(v) and v<=config['maximum_loop_loss_ratio'] for v in ratios)
    return dict(schema='postfit_relative_improvement.v1',gt_consumed=False,
        passes=bool(support and gains and nonregression),loop_count=len(rows),improved_loops=improved,
        adequate_projection_fraction=float(np.mean(coverage)) if coverage else 0.,
        support_retained=bool(support),sufficient_relative_gain=bool(gains),no_loop_loss_regression=bool(nonregression),
        absolute_alignment_passed=audit['passes'],mean_loss_ratios=ratios)
