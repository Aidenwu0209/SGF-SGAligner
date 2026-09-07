"""Depth-first recovery ablation: identical inputs, frozen inference before GT."""
import argparse,importlib.util,json,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from pose_pipeline.depth_first_loops import compact_clouds,recover_pair
from pose_pipeline.depth_loop_witness import inconsistent_triangles
from pose_pipeline.contracts import sha256_file,stable_json_sha256,write_trajectory
from pose_pipeline.geometry_backend import GeometryBootstrapConfig
from pose_pipeline.robust_backend import RobustPoseConfig
from pose_pipeline.pose_graph import PoseGraphEdge,audit_corrected_trajectory
from pose_pipeline.bounded_backend import optimize_bounded_trajectory
spec=importlib.util.spec_from_file_location('experiment',Path(__file__).with_name('validate_depth_loop_witness.py'))
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
legacy=exp.legacy


def prepare(args):
    start=time.perf_counter()
    scene,paths,manifest,baseline,data,hashes,configs,ordinals,original,receipt,parameters=exp.light.load_inputs(args)
    clouds,_,bindings=compact_clouds(manifest,baseline,ordinals)
    e=data['loop_evidence.json'];rows=[]
    args.output.mkdir(parents=True,exist_ok=False)
    for p in e['evidence']:
        if p['edge_verified']:continue
        row=recover_pair(manifest,baseline,ordinals,p['source_anchor_index'],p['target_anchor_index'],clouds,bindings,
            RobustPoseConfig(**e['robust_config']),GeometryBootstrapConfig(**e['geometry_config']))
        row['old_rejection']=p['registration'].get('reason');rows.append(row)
        legacy.write_json(args.output/f"pair_{row['source']}_{row['target']}.json",row)
        print(json.dumps(dict(pair=[row['source_frame'],row['target_frame']],accepted=row['accepted'],
            providers=[dict(provider=x['provider'],usable=x['proposal_usable'],depth=x.get('depth_witness',{}).get('accepted')) for x in row['proposals']])),flush=True)
    bank=dict(scene_id=args.scene,gt_consumed=False,rows=rows,source_sha256=hashes,
        input_spec_sha256=sha256_file(args.inputs),anchor_ordinals=ordinals,rgbd_bindings=bindings,runtime_s=time.perf_counter()-start)
    frames={f.frame_id:f for f in manifest.frames}
    for b in bindings:
        for r in b['frames']:
            f=frames[r['frame_id']];assert sha256_file(f.depth_path)==r['depth_sha256'] and sha256_file(f.color_path)==r['color_sha256']
    legacy.write_json(args.output/'depth_bank.json',bank)


def infer(args):
    from reconstruction.rgbd_refusion import FullRefusionRequest,run_full_rgbd_refusion
    from pose_pipeline.geometry_metrics import ply_geometry_metrics,compare_no_gt_geometry_v2
    scene,paths,manifest,baseline,data,hashes,configs,ordinals,original,receipt,parameters=exp.light.load_inputs(args)
    local=json.loads(args.local_bank.read_text());depth=json.loads(args.depth_bank.read_text())
    for bank in [local,depth]:
        legacy.assert_gt_free(bank);assert bank['scene_id']==args.scene
        assert bank['source_sha256']==hashes and bank['anchor_ordinals']==ordinals
        assert bank['input_spec_sha256']==sha256_file(args.inputs)
    fixed=list(original)
    for p in local['pairs']:
        r=p['variants']['plain']
        if r['accepted']:fixed.append(PoseGraphEdge(p['source'],p['target'],np.asarray(r['transform']),kind='local_rgbd',
            information=np.asarray(r['information']),provenance='lightweight_heldout_checked_plain'))
    new=[PoseGraphEdge(r['source'],r['target'],np.asarray(r['transform']),kind='depth_first_loop',weight=1.,
        provenance='independent_depth_first_plus_disjoint_multiview') for r in depth['rows'] if r['accepted']]
    bad,cycles=inconsistent_triangles(fixed+new,{(e.source,e.target) for e in new})
    retained=[];degree={i:0 for i in range(len(ordinals))};dropped=[]
    for e in fixed:degree[e.source]+=1;degree[e.target]+=1
    for e in new:
        if (e.source,e.target) in bad:reason='inconsistent_measured_triangle'
        elif max(degree[e.source],degree[e.target])>=4:reason='degree_budget'
        else:retained.append(e);degree[e.source]+=1;degree[e.target]+=1;continue
        dropped.append(dict(source=e.source,target=e.target,reason=reason))
    args.output.mkdir(parents=True,exist_ok=False);directory=args.output/args.scene;directory.mkdir()
    for a,b in [(paths['manifest'],directory/'tracked_manifest.json'),(paths['trajectory'],directory/'baseline/trajectory.json'),
                (paths['baseline_cloud'],directory/'baseline/refused.ply')]:legacy.copy_exact(a,b)
    legacy.write_json(args.output/'edge_audit.json',dict(proposed=len(new),retained=len(retained),dropped=dropped,cycle_check=cycles))
    _,bounded,correction,optimizer,_=configs
    bounded=replace(bounded,maximum_loop_degree=4);optimizer=replace(optimizer,huber_information_policy='scalar')
    arms={}
    for arm,edges,cap,angle in [('control',fixed,.25,5.),('depth25',fixed+retained,.25,5.),('depth75',fixed+retained,.75,15.)]:
        start=time.perf_counter();out=directory/arm;out.mkdir()
        guardconfig=replace(correction,maximum_absolute_correction_translation_m=cap,maximum_absolute_correction_rotation_deg=angle)
        poses,report=optimize_bounded_trajectory(baseline,ordinals,edges,config=bounded,correction_config=guardconfig,optimization_config=optimizer)
        exp.verified.assert_replay_kept_edges(report,edges);legacy.validate_exact_coverage(manifest,poses)
        candidate=out/'candidate_trajectory.json';write_trajectory(candidate,poses,sequence_id=args.scene,arm=arm,
            metadata=dict(gt_consumed=False,diagnostic_only=True,depth_bank_sha256=sha256_file(args.depth_bank)))
        legacy.write_json(out/'bounded_backend.json',report)
        legacy.write_json(out/'edges.json',dict(edges=[exp.verified.edge_fingerprint(e) for e in edges]))
        fusion=run_full_rgbd_refusion(FullRefusionRequest(manifest=directory/'tracked_manifest.json',trajectory=candidate,
            output_dir=out/'candidate_refusion',**parameters))
        legacy.validate_baseline_receipt(fusion,dict(manifest=directory/'tracked_manifest.json',trajectory=candidate,baseline_cloud=Path(fusion['cloud'])),len(baseline))
        geometry=compare_no_gt_geometry_v2(ply_geometry_metrics(paths['baseline_cloud']),ply_geometry_metrics(Path(fusion['cloud'])),
            admitted_frame_sha256=stable_json_sha256([p.frame_id for p in baseline]))
        guard=audit_corrected_trajectory(baseline,poses,guardconfig)
        committed=out/'committed_trajectory.json';legacy.copy_exact(paths['trajectory'],committed)
        result=dict(arm=arm,gt_consumed=False,diagnostic_only=True,promotion_eligible=False,committed_candidate=False,
            dpv_rollback_byte_identical=sha256_file(committed)==sha256_file(paths['trajectory']),pose_count=len(poses),
            candidate_trajectory_sha256=sha256_file(candidate),committed_trajectory_sha256=sha256_file(committed),candidate_refusion=fusion,
            no_gt_geometry=geometry,strict_correction_guard=guard,added_depth_edges=len(retained) if arm!='control' else 0,
            inference_gates_pass=bool(report['success'] and guard['passes'] and geometry['passes_scene_safety'] and geometry['passes_scene_improvement']),
            runtime_s=time.perf_counter()-start)
        legacy.write_json(out/'result.json',result);arms[arm]=result
    legacy.write_json(args.output/'summary.json',dict(schema='unified_pose_replay.v1',gt_consumed=False,diagnostic_only=True,
        scenes=[dict(scene_id=args.scene,inputs=scene,arms=arms)]))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='phase',required=True)
    for phase in ['prepare','infer']:
        q=sub.add_parser(phase)
        for name in ['inputs','source-root','source-config','output']:q.add_argument('--'+name,type=Path,required=True)
        q.add_argument('--scene',required=True)
        if phase=='infer':
            q.add_argument('--local-bank',type=Path,required=True);q.add_argument('--depth-bank',type=Path,required=True)
        q.set_defaults(handler=prepare if phase=='prepare' else infer)
    q=sub.add_parser('evaluate');q.add_argument('--run-root',type=Path,required=True);q.add_argument('--references',type=Path,required=True)
    q.set_defaults(handler=legacy.evaluate)
    args=p.parse_args();args.handler(args)
if __name__=='__main__':main()
