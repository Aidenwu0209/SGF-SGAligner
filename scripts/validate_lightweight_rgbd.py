#!/usr/bin/env python3
"""Sealed CPU ablation: overlap ROI, multiscale color ICP, local information.

prepare reads only RGB-D and sealed prediction evidence. infer consumes the
prepared local edge bank; evaluate uses the existing separate GT process.
No source registration is silently upgraded to a new independently verified
loop. Local edges are explicitly tagged and evaluated as an experiment.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from pose_pipeline.contracts import load_manifest, load_trajectory, sha256_file, stable_json_sha256, write_trajectory
from pose_pipeline.pose_graph import PoseGraphEdge, audit_corrected_trajectory
from pose_pipeline.bounded_backend import optimize_bounded_trajectory
from pose_pipeline.lightweight_rgbd import LightweightConfig, anchor_clouds, refine_pair

spec=importlib.util.spec_from_file_location('verified_replay',Path(__file__).with_name('replay_verified_unified_backend.py'))
verified=importlib.util.module_from_spec(spec);spec.loader.exec_module(verified)
legacy=verified.legacy
ARMS={'control':(None,False),'local_base':('plain',False),'roi':('roi',False),
      'color':('color',False),'information':('plain',True),'combined':('roi_color',True)}


def load_inputs(args):
    spec=json.loads(args.inputs.read_text())
    if spec.get('schema')!='unified_pose_replay_inputs.v1':raise ValueError('input schema')
    matches=[s for s in spec['scenes'] if s['scene_id']==args.scene]
    if len(matches)!=1:raise ValueError('unique scene required')
    scene=matches[0];paths=legacy.resolve_bound_inputs(scene,args.inputs.resolve().parent)
    manifest=load_manifest(paths['manifest']);baseline,payload=load_trajectory(paths['trajectory'])
    legacy.assert_gt_free(payload,'baseline');verified.assert_no_fallback(payload)
    legacy.validate_exact_coverage(manifest,baseline)
    settings=json.loads(args.source_config.read_text())
    data,files,hashes,configs,ordinals,edges=verified.load_source(args.source_root,scene,paths,manifest,baseline,settings)
    receipt=json.loads(paths['baseline_refusion_receipt'].read_text())
    parameters=legacy.validate_baseline_receipt(receipt,paths,len(baseline))
    return scene,paths,manifest,baseline,data,hashes,configs,ordinals,edges,receipt,parameters


def prepare(args):
    started=time.perf_counter()
    scene,paths,manifest,baseline,data,hashes,configs,ordinals,edges,receipt,parameters=load_inputs(args)
    config=LightweightConfig()
    fit,checks,bindings=anchor_clouds(manifest,baseline,ordinals,config)
    # Adjacent anchor factors complement the unchanged DPV odometry prior.
    # Non-neighbor loops are retained from the independently verified source.
    rows=[]
    for i in range(len(ordinals)-1):
        j=i+1
        initial=np.linalg.inv(baseline[ordinals[j]].t_world_camera) @ baseline[ordinals[i]].t_world_camera
        outputs={}
        for label,roi,color in (('plain',False,False),('roi',True,False),('color',False,True),('roi_color',True,True)):
            outputs[label]=refine_pair(fit[i],fit[j],checks[i],checks[j],initial,roi=roi,colored=color,config=config)
        rows.append({'source':i,'target':j,'initial':initial.tolist(),'variants':outputs})
        print(json.dumps({'pair':[i,j],'accepted':{k:v['accepted'] for k,v in outputs.items()}}),flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    bank={'schema':'lightweight_local_edges.v1','scene_id':args.scene,'config':asdict(config),
          'gt_consumed':False,'diagnostic_only':True,'semantic_guidance_tested':False,
          'source_sha256':hashes,'input_spec_sha256':sha256_file(args.inputs),
          'anchor_ordinals':ordinals,'rgbd_bindings':bindings,'pairs':rows,
          'runtime_s':time.perf_counter()-started}
    # Revalidate consumed pixels before sealing the cache.
    frame_map={f.frame_id:f for f in manifest.frames}
    for anchor in bindings:
        for b in anchor['frames']:
            f=frame_map[b['frame_id']]
            assert sha256_file(f.color_path)==b['color_sha256'] and sha256_file(f.depth_path)==b['depth_sha256']
    legacy.write_json(args.output/'edge_bank.json',bank)


def infer(args):
    from pose_pipeline.geometry_metrics import ply_geometry_metrics, compare_no_gt_geometry_v2
    from reconstruction.rgbd_refusion import FullRefusionRequest,run_full_rgbd_refusion
    started=time.perf_counter()
    scene,paths,manifest,baseline,data,hashes,configs,ordinals,edges,receipt,parameters=load_inputs(args)
    bank=json.loads(args.edge_bank.read_text());legacy.assert_gt_free(bank,'bank')
    if args.edge_bank_sha256!=sha256_file(args.edge_bank):raise ValueError('edge bank changed')
    verified.same_value(hashes,bank['source_sha256'],'edge bank source')
    verified.same_value(ordinals,bank['anchor_ordinals'],'edge bank anchors')
    if bank['input_spec_sha256']!=sha256_file(args.inputs) or bank['scene_id']!=args.scene:raise ValueError('bank input mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    legacy.copy_exact(args.inputs,args.output/'inputs.json')
    arm=args.arm;variant,information=ARMS[arm]
    destination=args.output/args.scene;destination.mkdir()
    for path,target in ((paths['manifest'],destination/'tracked_manifest.json'),
                        (paths['trajectory'],destination/'baseline/trajectory.json'),
                        (paths['baseline_cloud'],destination/'baseline/refused.ply')):
        legacy.copy_exact(path,target)
    legacy.write_json(destination/'baseline/reuse_receipt.json',{'verified_receipt':receipt,'gt_consumed':False})
    legacy.write_json(args.output/'configuration.json',{'arm':arm,'variant':variant,'local_information':information,'bank_sha256':args.edge_bank_sha256})
    added=[]
    if variant:
        for pair in bank['pairs']:
            row=pair['variants'][variant]
            if row['accepted']:
                added.append(PoseGraphEdge(source=pair['source'],target=pair['target'],
                    source_to_target=np.asarray(row['transform']),kind='local_rgbd',weight=1.0,
                    provenance='lightweight_heldout_checked_'+variant,information=np.asarray(row['information'])))
    original,bounded,correction,optimizer,geometry_config=configs
    bounded=replace(bounded,maximum_loop_degree=4)
    optimizer=replace(optimizer,huber_information_policy='local_normalized' if information else 'scalar')
    legacy.write_json(destination/'edge_input.json',{'gt_consumed':False,'original_verified_edges':[verified.edge_fingerprint(e) for e in edges],
        'added_local_edges':[verified.edge_fingerprint(e) for e in added],'bank_sha256':args.edge_bank_sha256})
    arms={};noop=None
    if not edges and not added:
        legacy.copy_exact(paths['trajectory'],destination/'committed_trajectory.json')
        noop={'reason':'no_verified_or_local_edges','candidate_generated':False,'dpv_rollback_byte_identical':True,
              'final_cloud_sha256':sha256_file(paths['baseline_cloud']),'pose_count':len(baseline)}
    else:
        arm_dir=destination/arm;arm_dir.mkdir()
        corrected,report=optimize_bounded_trajectory(baseline,ordinals,edges+added,config=bounded,
            correction_config=correction,optimization_config=optimizer)
        verified.assert_replay_kept_edges(report,edges+added)
        legacy.validate_exact_coverage(manifest,corrected)
        candidate=arm_dir/'candidate_trajectory.json'
        write_trajectory(candidate,corrected,sequence_id=args.scene,arm=arm,metadata={
            'gt_consumed':False,'diagnostic_only':True,'source_sha256':hashes,'edge_bank_sha256':args.edge_bank_sha256})
        legacy.write_json(arm_dir/'bounded_backend.json',report)
        refusion=run_full_rgbd_refusion(FullRefusionRequest(manifest=destination/'tracked_manifest.json',
            trajectory=candidate,output_dir=arm_dir/'candidate_refusion',**parameters))
        legacy.validate_baseline_receipt(refusion,{'manifest':destination/'tracked_manifest.json',
            'trajectory':candidate,'baseline_cloud':Path(refusion['cloud'])},len(baseline))
        bm=ply_geometry_metrics(paths['baseline_cloud']);cm=ply_geometry_metrics(Path(refusion['cloud']))
        geometry=compare_no_gt_geometry_v2(bm,cm,admitted_frame_sha256=stable_json_sha256([r.frame_id for r in baseline]))
        guard=audit_corrected_trajectory(baseline,corrected,correction)
        changed=any(not np.allclose(a.t_world_camera,b.t_world_camera,atol=1e-12,rtol=0) for a,b in zip(baseline,corrected))
        accepted=bool(changed and report['success'] and guard['passes'] and geometry['passes_scene_safety'] and geometry['passes_scene_improvement'])
        committed=arm_dir/'committed_trajectory.json';legacy.copy_exact(candidate if accepted else paths['trajectory'],committed)
        for name,value in (('baseline_geometry.json',bm),('candidate_geometry.json',cm),('geometry_comparison.json',geometry),('strict_correction_guard.json',guard)):
            legacy.write_json(arm_dir/name,value)
        result={'arm':arm,'diagnostic_only':True,'promotion_eligible':False,'gt_consumed':False,'complete_frame_coverage':True,
                'identity_fallback_used':False,'pose_count':len(baseline),'backend_success':report['success'],
                'original_verified_loop_count':len(edges),'added_local_edge_count':len(added),
                'candidate_trajectory_sha256':sha256_file(candidate),'committed_trajectory_sha256':sha256_file(committed),
                'committed_candidate':accepted,'dpv_rollback_byte_identical':not accepted and sha256_file(committed)==sha256_file(paths['trajectory']),
                'no_gt_geometry':geometry,'strict_correction_guard':guard,'candidate_refusion':refusion,
                'bounded_config':asdict(bounded),'optimization_config':asdict(optimizer),'runtime_s':time.perf_counter()-started}
        legacy.write_json(arm_dir/'result.json',result);arms[arm]=result
    legacy.resolve_bound_inputs(scene,args.inputs.resolve().parent)
    current_hashes={name:sha256_file(args.source_root/name) for name in hashes}
    verified.same_value(hashes,current_hashes,'source changed during inference')
    if sha256_file(args.edge_bank)!=args.edge_bank_sha256:raise ValueError('bank changed during inference')
    row={'scene_id':args.scene,'inputs':scene,'arms':arms,'no_candidate':noop,'gt_consumed':False}
    legacy.write_json(destination/'result.json',row)
    legacy.write_json(args.output/'summary.json',{'schema':'unified_pose_replay.v1','diagnostic_only':True,
        'promotion_eligible':False,'gt_consumed':False,'arms':[arm],'scenes':[row],
        'registration_gate_recomputed':False,'new_local_constraints_heldout_depth_checked':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    for name in ('prepare','infer'):
        q=sub.add_parser(name)
        for flag in ('inputs','source-root','source-config','output'):q.add_argument('--'+flag,type=Path,required=True)
        q.add_argument('--scene',required=True)
        if name=='infer':
            q.add_argument('--edge-bank',type=Path,required=True);q.add_argument('--edge-bank-sha256',required=True)
            q.add_argument('--arm',choices=list(ARMS),required=True)
        q.set_defaults(handler=prepare if name=='prepare' else infer)
    q=sub.add_parser('evaluate');q.add_argument('--run-root',type=Path,required=True);q.add_argument('--references',type=Path,required=True);q.set_defaults(handler=legacy.evaluate)
    args=p.parse_args();args.handler(args)


if __name__=='__main__':main()
