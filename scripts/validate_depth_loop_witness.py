"""Controlled replay: keep local ICP and frozen geometric hypotheses unchanged."""
import argparse,importlib.util,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from pose_pipeline.contracts import sha256_file,stable_json_sha256,write_trajectory,validate_se3
from pose_pipeline.pose_graph import PoseGraphEdge,LoopWeightConfig,loop_edge_weight,audit_corrected_trajectory
from pose_pipeline.bounded_backend import optimize_bounded_trajectory
from pose_pipeline.depth_loop_witness import verify_depth_loop,inconsistent_triangles
spec=importlib.util.spec_from_file_location('light',Path(__file__).with_name('validate_lightweight_rgbd.py'))
light=importlib.util.module_from_spec(spec);spec.loader.exec_module(light)
verified=light.verified;legacy=light.legacy

def edge_from_row(r):
    return PoseGraphEdge(source=r['source'],target=r['target'],source_to_target=np.asarray(r['transform']),
        kind=r['kind'],weight=r['weight'],information=None if r['information'] is None else np.asarray(r['information']),
        provenance=r['provenance'],confidence=r.get('confidence',1.))

def experiment(args):
    from pose_pipeline.geometry_metrics import ply_geometry_metrics,compare_no_gt_geometry_v2
    from reconstruction.rgbd_refusion import FullRefusionRequest,run_full_rgbd_refusion
    started=time.perf_counter()
    scene,paths,manifest,baseline,data,hashes,configs,ordinals,original,receipt,parameters=light.load_inputs(args)
    bank=json.loads(args.edge_bank.read_text());legacy.assert_gt_free(bank)
    assert sha256_file(args.edge_bank)==args.edge_bank_sha256
    verified.same_value(bank['source_sha256'],hashes,'local bank source')
    verified.same_value(bank['anchor_ordinals'],ordinals,'local bank anchors')
    assert bank['scene_id']==args.scene and bank['input_spec_sha256']==sha256_file(args.inputs)
    frames={f.frame_id:f for f in manifest.frames}
    for anchor in bank['rgbd_bindings']:
        for r in anchor['frames']:
            f=frames[r['frame_id']]
            assert sha256_file(f.depth_path)==r['depth_sha256'] and sha256_file(f.color_path)==r['color_sha256']
    local=[]
    for pair in bank['pairs']:
        r=pair['variants']['plain']
        if r['accepted']:
            local.append(PoseGraphEdge(source=pair['source'],target=pair['target'],source_to_target=np.asarray(r['transform']),
                kind='local_rgbd',weight=1.,information=np.asarray(r['information']),provenance='lightweight_heldout_checked_plain'))
    fixed=original+local;evidence=data['loop_evidence.json'];new=[];rows=[]
    for proposal in evidence['evidence']:
        reg=proposal['registration'];s=proposal['source_anchor_index'];t=proposal['target_anchor_index']
        if proposal['edge_verified'] or not reg.get('accepted'):continue
        if not reg.get('decision',{}).get('usable_for_reconstruction'):continue
        excluded=set(evidence['anchors'][s]['source_frame_ids'])|set(evidence['anchors'][t]['source_frame_ids'])
        transform=validate_se3(np.asarray(reg['transform']))
        witness=verify_depth_loop(manifest,baseline,ordinals[s],ordinals[t],transform,excluded)
        row={'source':s,'target':t,'source_frame':baseline[ordinals[s]].frame_id,'target_frame':baseline[ordinals[t]].frame_id,
             'registration_sha256':stable_json_sha256(reg),'original_visual_reason':(proposal.get('visual_verification') or {}).get('reason'),
             'depth_witness':witness,'admitted':False}
        if witness['accepted']:
            weight=loop_edge_weight(reg['forward']['verification']['minimum_overlap'],s,t,len(ordinals),
                                    LoopWeightConfig(**evidence['loop_weight_config']))
            new.append(PoseGraphEdge(source=s,target=t,source_to_target=transform,kind='depth_verified_loop',weight=weight,
                information=np.asarray(reg['information_matrix']),confidence=reg['edge_confidence'],provenance='frozen_geometry_plus_heldout_projective_depth'))
        rows.append(row)
    bad,cycles=inconsistent_triangles(fixed+new,{(e.source,e.target) for e in new})
    degree={i:0 for i in range(len(ordinals))}
    for e in fixed:degree[e.source]+=1;degree[e.target]+=1
    retained=[]
    for e in new:
        r=next(r for r in rows if (r['source'],r['target'])==(e.source,e.target))
        if (e.source,e.target) in bad:r['rejection']='inconsistent_measured_triangle';continue
        if max(degree[e.source],degree[e.target])>=4:r['rejection']='fixed_degree_budget';continue
        retained.append(e);degree[e.source]+=1;degree[e.target]+=1;r['admitted']=True
    args.output.mkdir(parents=True,exist_ok=False)
    legacy.write_json(args.output/'depth_witness.json',{'gt_consumed':False,'diagnostic_only':True,'config_frozen_before_gt':True,
        'eligible_hypothesis_count':len(rows),'recovered_edge_count':len(retained),'rows':rows,'cycle_check':cycles,
        'source_sha256':hashes,'bank_sha256':args.edge_bank_sha256,'scope':'geometry-accepted, sparse-PnP-rejected frozen hypotheses only'})
    directory=args.output/args.scene;directory.mkdir()
    for src,dst in [(paths['manifest'],directory/'tracked_manifest.json'),(paths['trajectory'],directory/'baseline/trajectory.json'),
                    (paths['baseline_cloud'],directory/'baseline/refused.ply')]:legacy.copy_exact(src,dst)
    _,bounded,correction,optimizer,_=configs
    bounded=replace(bounded,maximum_loop_degree=4);optimizer=replace(optimizer,huber_information_policy='scalar')
    arms={}
    # Both arms are recomputed in the same runtime with the same source snapshot.
    for arm,edges in [('control',fixed),('depth_witness',fixed+retained)]:
        start=time.perf_counter();out=directory/arm;out.mkdir()
        corrected,report=optimize_bounded_trajectory(baseline,ordinals,edges,config=bounded,correction_config=correction,optimization_config=optimizer)
        verified.assert_replay_kept_edges(report,edges)
        legacy.validate_exact_coverage(manifest,corrected)
        candidate=out/'candidate_trajectory.json'
        write_trajectory(candidate,corrected,sequence_id=args.scene,arm=arm,metadata={'gt_consumed':False,'diagnostic_only':True,
            'input_source_sha256':hashes,'depth_witness_sha256':sha256_file(args.output/'depth_witness.json')})
        legacy.write_json(out/'bounded_backend.json',report)
        legacy.write_json(out/'edges.json',{'gt_consumed':False,'edges':[verified.edge_fingerprint(e) for e in edges]})
        fusion=run_full_rgbd_refusion(FullRefusionRequest(manifest=directory/'tracked_manifest.json',trajectory=candidate,
            output_dir=out/'candidate_refusion',**parameters))
        legacy.validate_baseline_receipt(fusion,{'manifest':directory/'tracked_manifest.json','trajectory':candidate,
                                               'baseline_cloud':Path(fusion['cloud'])},len(baseline))
        bm=ply_geometry_metrics(paths['baseline_cloud']);cm=ply_geometry_metrics(Path(fusion['cloud']))
        geometry=compare_no_gt_geometry_v2(bm,cm,admitted_frame_sha256=stable_json_sha256([p.frame_id for p in baseline]))
        guard=audit_corrected_trajectory(baseline,corrected,correction)
        changed=any(not np.allclose(a.t_world_camera,b.t_world_camera,atol=1e-12,rtol=0) for a,b in zip(baseline,corrected))
        accepted=bool(changed and report['success'] and guard['passes'] and geometry['passes_scene_safety'] and geometry['passes_scene_improvement'])
        committed=out/'committed_trajectory.json';legacy.copy_exact(candidate if accepted else paths['trajectory'],committed)
        result={'gt_consumed':False,'diagnostic_only':True,'promotion_eligible':False,'arm':arm,'pose_count':len(baseline),
            'complete_frame_coverage':True,'identity_fallback_used':False,'backend_success':report['success'],
            'committed_candidate':accepted,'dpv_rollback_byte_identical':not accepted and sha256_file(committed)==sha256_file(paths['trajectory']),
            'candidate_trajectory_sha256':sha256_file(candidate),'committed_trajectory_sha256':sha256_file(committed),
            'candidate_refusion':fusion,'no_gt_geometry':geometry,'strict_correction_guard':guard,
            'recovered_edges':len(retained) if arm=='depth_witness' else 0,'runtime_s':time.perf_counter()-start}
        legacy.write_json(out/'result.json',result);arms[arm]=result
    for row in rows:
        for b in row['depth_witness']['rgbd_sha256']:
            f=frames[b['frame_id']];assert sha256_file(f.depth_path)==b['depth_sha256'] and sha256_file(f.color_path)==b['color_sha256']
    legacy.resolve_bound_inputs(scene,args.inputs.resolve().parent)
    assert all(sha256_file(args.source_root/name)==digest for name,digest in hashes.items())
    assert sha256_file(args.edge_bank)==args.edge_bank_sha256
    legacy.write_json(args.output/'summary.json',{'schema':'unified_pose_replay.v1','gt_consumed':False,'diagnostic_only':True,
        'promotion_eligible':False,'scenes':[{'scene_id':args.scene,'inputs':scene,'arms':arms}],
        'source_inputs_unchanged':True,'runtime_s':time.perf_counter()-started})

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    q=sub.add_parser('infer')
    for flag in ['inputs','source-root','source-config','edge-bank','output']:q.add_argument('--'+flag,type=Path,required=True)
    q.add_argument('--scene',required=True);q.add_argument('--edge-bank-sha256',required=True);q.set_defaults(handler=experiment)
    q=sub.add_parser('evaluate')
    q.add_argument('--run-root',type=Path,required=True);q.add_argument('--references',type=Path,required=True);q.set_defaults(handler=legacy.evaluate)
    args=p.parse_args();args.handler(args)
if __name__=='__main__':main()
