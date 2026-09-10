"""Continuous native SGF over a frozen trajectory; no GT or pose feedback."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .contracts import load_manifest, load_trajectory, bind_manifest_trajectory, sha256_file, validate_se3
from .semantic_mapping import sgf_pose, write_json, load_submap, export_map
from reconstruction.rgbd_refusion import _read_rgbd


def save_snapshot(engine, out, transform, frame_ids, final):
    out.mkdir()
    cloud = engine.native.snapshot_inseg(min_segment_points=1)
    graph = engine.native.snapshot_graph(min_segment_points=1)
    if graph.get('prediction_enabled') is not True:
        raise RuntimeError('SGF prediction was disabled')
    if cloud['coordinate_unit'] != 'metre':
        raise ValueError('expected native snapshot in metres')
    xyz = (cloud['xyz'] - transform[:3, 3]) @ transform[:3, :3]
    normals = cloud['normals'] @ transform[:3, :3]
    if not np.isfinite(xyz).all() or not np.isfinite(normals).all():
        raise ValueError('nonfinite native geometry')
    # The native renderer converts surfel radii from millimetres to metres.
    # Retain their footprints: surfels represent discs, not isolated centres.
    radius = np.asarray(cloud['radius'], dtype=float) * .001
    if not np.isfinite(radius).all() or np.any(radius < 0):
        raise ValueError('invalid native surfel radii')
    np.savez_compressed(out/'inseg_cloud.npz', xyz=xyz, normals=normals,
                        colors=cloud['colors'], labels=cloud['labels'],
                        radius_m=radius, quality=cloud['quality'])
    write_json(out/'graph.json', {k:v for k,v in graph.items()
               if k not in ('node_features', 'node_feature_valid')})
    write_json(out/'replay.json', {'frame_ids':frame_ids, 'processed_frames':len(frame_ids),
        'final':final, 'continuous_engine':True, 'snapshot_scope':'cumulative, not independent evidence',
        'T_model_estimated':transform, 'cloud_coordinate_frame':'original estimated world',
        'graph_geometry_coordinate_frame':'model world', 'ground_truth_consumed':False,
        'radius_unit':'metre', 'source_surfel_count':cloud['source_surfel_count'],
        'filtered_surfel_count':cloud['filtered_surfel_count']})


def export_native(snapshot, classes):
    from plyfile import PlyData, PlyElement
    s=load_submap(snapshot);c=s['cloud'];n=len(c['xyz'])
    dtype=[(k,'<f4') for k in ('x','y','z','nx','ny','nz')]
    dtype += [(k,'u1') for k in ('red','green','blue')]
    dtype += [('segment_id','<i4'),('semantic_id','<i4'),('instance_id','<i4'),('semantic_confidence','<f4')]
    v=np.empty(n,dtype=dtype)
    for j,k in enumerate(('x','y','z')):v[k]=c['xyz'][:,j]
    for j,k in enumerate(('nx','ny','nz')):v[k]=c['normals'][:,j]
    for j,k in enumerate(('red','green','blue')):v[k]=c['colors'][:,j]
    v['segment_id']=c['labels']
    v['semantic_id']=[classes.get(s['nodes'].get(int(l),{}).get('label'),0) for l in c['labels']]
    v['semantic_confidence']=[s['nodes'].get(int(l),{}).get('confidence',0) for l in c['labels']]
    v['instance_id']=[max(0,s['nodes'].get(int(l),{}).get('native_instance_id',0)) for l in c['labels']]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(snapshot/'native_labeled.ply'))
    for key in ('semantic_id','instance_id','segment_id'):
        q=v.copy();ids=q[key].astype(np.uint64)
        for ch,mul in zip(('red','green','blue'),(73,151,199)):
            q[ch]=np.where(ids>0,50+(ids*mul)%206,90).astype('u1')
        PlyData([PlyElement.describe(q,'vertex')],text=False).write(str(snapshot/f'native_{key}.ply'))
    return s


def run(args):
    from sgf_runtime import CameraIntrinsics, FramePacket, SceneGraphFusion
    import sgf_native
    manifest=load_manifest(args.manifest)
    if args.profile == 'scannet-historical' and manifest.dataset != 'scannet':
        raise ValueError('ScanNet historical profile requires a ScanNet manifest')
    poses,_=load_trajectory(args.trajectory)
    all_bound=bind_manifest_trajectory(manifest,poses)
    bound=all_bound if args.stream=='all' else [x for i,x in enumerate(all_bound) if i%2==int(args.stream=='odd')]
    H=np.eye(4)
    if args.frame_control:
        control=json.loads(args.frame_control.read_text())
        if control.get('uses_gt_for_transform') is not False:raise ValueError('GT-free transform declaration required')
        H[:3,:3]=control['R_model_estimated']
    validate_se3(H)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    model_hash={p.name:sha256_file(p) for p in args.model.iterdir() if p.is_file()}
    source_paths=[args.manifest,args.trajectory]+([args.baseline] if args.baseline else [])
    sources={str(p):sha256_file(p) for p in source_paths}
    status={'status':'running','stream':args.stream,'expected_frames':len(bound),
            'available_frames':len(all_bound),'processed_frames':0,'engine_initializations':1,
            'pose_feedback':False,'ground_truth_consumed':False,'source_sha256':sha256_file(Path(__file__)),
            'native_sha256':sha256_file(Path(sgf_native.__file__)),'model_sha256':model_hash,
            'input_sha256':sources,'settings':{'sample_points':args.sample_points,'filter':args.segment_filter,
            'pyr':args.min_pyr_level,'edge':args.depth_edge_threshold,'sample_with_replacement':True,
            'seed':42,'final_min_segment_points':args.final_min_points,'T_model_estimated':H}}
    write_json(out/'status.json',status)
    rgb,d,k=_read_rgbd(bound[0][0]);h,w=d.shape
    engine=SceneGraphFusion(CameraIntrinsics(w,h,*k),model_path=args.model,
        enable_prediction=True,use_thread=False,segment_filter=args.segment_filter,
        min_pyr_level=args.min_pyr_level,depth_edge_threshold=args.depth_edge_threshold,
        sampling_seed=42,num_sample_points=args.sample_points,sample_with_replacement=True)
    started=time.monotonic();seen=[]
    try:
        for f,p in bound:
            rgb,d,ki=_read_rgbd(f)
            if d.shape!=(h,w) or not np.allclose(ki,k):raise ValueError('changing intrinsics or dimensions')
            mm=np.rint(d.astype(float)*1000/manifest.depth_scale);mm[(mm>4500)|(mm<0)]=0
            engine.process_frame(FramePacket(f.frame_id,np.ascontiguousarray(rgb[:,:,::-1]),
                np.ascontiguousarray(mm,dtype=np.uint16),sgf_pose(H@p.t_world_camera),
                f.timestamp_us*1000,time.monotonic_ns()))
            seen.append(f.frame_id)
            if len(seen)%100==0:
                status.update(processed_frames=len(seen),last_frame_id=f.frame_id,seconds=time.monotonic()-started)
                write_json(out/'status.json',status);print(json.dumps({'frames':len(seen),'total':len(bound)}),flush=True)
            if len(seen)%600==0:save_snapshot(engine,out/f'checkpoint_{len(seen):04}',H,list(seen),False)
        engine.run_full_prediction(min_segment_points=args.final_min_points)
        save_snapshot(engine,out/'final',H,seen,True)
    except BaseException as e:
        status.update(status='failed',processed_frames=len(seen),error=f'{type(e).__name__}: {e}')
        write_json(out/'status.json',status);raise
    finally:engine.stop()
    if seen != [f.frame_id for f,_ in bound]:raise RuntimeError('frame coverage mismatch')
    classes={x:i+1 for i,x in enumerate(args.model.joinpath('classes.txt').read_text().splitlines()) if x}
    scene=export_native(out/'final',classes)
    ids={l:n['native_instance_id'] if n['native_instance_id']>0 else l for l,n in scene['nodes'].items() if l>0}
    metrics={'native_point_count':len(scene['cloud']['xyz']), 'baseline_projection_executed':False}
    if args.baseline:
        (out/'map').mkdir();metrics.update(export_map(args.baseline,out/'map',[scene],[ids],classes))
        write_json(out/'map/global_ids.json',ids)
        from .semantic_instances import final_scene_graph
        objects=json.loads((out/'map/objects.json').read_text())
        write_json(out/'map/scene_graph.json',final_scene_graph([scene],[ids],{x['instance_id'] for x in objects},manifest.sequence_id))
        metrics['baseline_projection_executed']=True
    if sources!={str(p):sha256_file(p) for p in source_paths}:raise RuntimeError('frozen input changed')
    status.update(status='completed',processed_frames=len(seen),seconds=time.monotonic()-started,
        complete_selected_stream=True,complete_full_sequence=args.stream=='all',quality_accepted=False,
        sga_inference_executed=False,**metrics)
    write_json(out/'result.json',status);write_json(out/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('manifest','trajectory','model','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--frame-control',type=Path)
    p.add_argument('--sample-points',type=int,default=512)
    p.add_argument('--profile',choices=('legacy','scannet-historical'),default='legacy')
    p.add_argument('--segment-filter',type=int)
    p.add_argument('--min-pyr-level',type=int)
    p.add_argument('--depth-edge-threshold',type=float)
    p.add_argument('--final-min-points',type=int,default=50)
    p.add_argument('--stream',choices=('all','even','odd'),default='all')
    args=p.parse_args()
    defaults=(128,3,.90) if args.profile=='scannet-historical' else (96,2,.98)
    for key,value in zip(('segment_filter','min_pyr_level','depth_edge_threshold'),defaults):
        if getattr(args,key) is None:setattr(args,key,value)
    run(args)

if __name__=='__main__':main()
