"""Preserve full-stream SGF semantics while attaching SGA-associated identities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import sha256_file
from .semantic_mapping import load_submap, export_map, write_json
from .semantic_instances import final_scene_graph
from .semantic_fusion import transfer_consensus, transfer_surfel_completion


def attach_instances(semantic, confidence, candidate_semantic, candidate_instance):
    """Instance ambiguity never clears a supported full-stream semantic label."""
    instance=np.where((semantic>0)&(semantic==candidate_semantic),candidate_instance,0).astype(np.int32)
    return semantic.copy(),instance,confidence.copy()


def run(args):
    paths=[args.full,args.first,args.second]
    replay=[json.loads((p/'replay.json').read_text()) for p in paths]
    frames=[r['frame_ids'] for r in replay]
    if any(not r.get('final') or r.get('ground_truth_consumed') is not False for r in replay):
        raise ValueError('final, GT-free SGF snapshots required')
    if any(len(f)!=len(set(f)) for f in frames) or set(frames[1])&set(frames[2]) or set(frames[0])!=set(frames[1]+frames[2]):
        raise ValueError('SGA streams must partition the complete semantic stream')
    if not all(np.allclose(r['T_model_estimated'],replay[0]['T_model_estimated']) for r in replay):
        raise ValueError('SGF model frames differ')
    association=json.loads(args.association.read_text())
    if association.get('inference_executed') is not True or association.get('status')!='completed':
        raise ValueError('completed real SGA inference required')
    if not np.allclose(association['T_model_world'],replay[0]['T_model_estimated']):
        raise ValueError('SGA feature frame must match SGF model frame')
    evidence=[args.baseline,args.association,args.global_ids,args.classes]
    evidence += [p/f for p in paths for f in ('inseg_cloud.npz','graph.json','replay.json')]
    before={str(p.resolve()):sha256_file(p) for p in evidence}
    scenes=[load_submap(p) for p in paths]
    ids=[{int(k):int(v) for k,v in m.items()} for m in json.loads(args.global_ids.read_text())]
    if len(ids)!=2 or any(set(m)!=set(s['nodes']) for m,s in zip(ids,scenes[1:])):
        raise ValueError('global ID maps do not match SGF node IDs')
    for candidate in association['candidates']:
        if candidate['accepted'] and ids[0][int(candidate['source_segment'])]!=ids[1][int(candidate['target_segment'])]:
            raise ValueError('accepted SGA match missing from global IDs')
    classes={v:int(k) for k,v in json.loads(args.classes.read_text()).items() if int(k)>0}
    full_ids={k:i+1 for i,k in enumerate(sorted(scenes[0]['nodes']))}
    full_transfer=transfer_surfel_completion if args.use_surfel_footprints else transfer_consensus
    def transfer(points,normals,submaps,mappings,class_ids):
        semantic,_,confidence,_=full_transfer(points,normals,[scenes[0]],[full_ids],class_ids)
        pair_semantic,pair_instance,_,ambiguous=transfer_consensus(points,normals,submaps,mappings,class_ids)
        semantic,instance,confidence=attach_instances(semantic,confidence,pair_semantic,pair_instance)
        return semantic,instance,confidence,ambiguous
    args.output.mkdir(parents=True,exist_ok=False)
    metrics=export_map(args.baseline,args.output,scenes[1:],ids,classes,label_transfer=transfer)
    objects=json.loads((args.output/'objects.json').read_text())
    confirmed={ids[0][int(c['source_segment'])] for c in association['candidates'] if c['accepted']}
    for obj in objects:
        obj['association_support']='SGA cross-stream match' if obj['instance_id'] in confirmed else 'SGF stream-local evidence; no accepted cross-stream match'
    write_json(args.output/'objects.json',objects)
    write_json(args.output/'scene_graph.json',final_scene_graph(scenes[1:],ids,{o['instance_id'] for o in objects},args.sequence))
    if before!={str(p.resolve()):sha256_file(p) for p in evidence}:raise RuntimeError('source evidence changed')
    write_json(args.output/'result.json',{'status':'completed','sequence':args.sequence,'input_sha256':before,
        'source_sha256':sha256_file(Path(__file__)),'semantic_source':'full continuous SGF',
        'instance_source':'SGF regions and same-part relations, SGA correspondences, geometric validation',
        'sga_inference_executed':True,'sga_accepted_associations':association['accepted_count'],
        'semantic_labels_independent_of_instance_ambiguity':True,'pose_feedback':False,
        'ground_truth_consumed':False,'oneformer_consumed':False,'processed_full_frames':len(frames[0]),
        'surfel_footprints_used':args.use_surfel_footprints,'quality_accepted':False,**metrics})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full','first','second','association','global-ids','classes','baseline','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sequence',required=True)
    p.add_argument('--use-surfel-footprints',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
