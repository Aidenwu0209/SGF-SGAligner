"""Measured SAM-object associations with an optional real SGAligner proposal.

SGF contributes only its existing predicted relationships. Unavailable semantic
attributes are disabled; no GT or fabricated learned relation is supplied.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def compatible(a, b):
    return a == b or (a == 33 and b in (8,17)) or (b == 33 and a in (8,17)) or (a == 34 and b in (7,14)) or (b == 34 and a in (7,14))


def select_tracks(tracks, limit=128):
    # Context chosen without GT. Small/one-view objects remain in exports, but
    # cannot displace all stable context in a bounded learned descriptor pass.
    eligible=[t for t in tracks if len(t['points']) >= 50]
    return sorted(eligible,key=lambda t:(-len(t['frames']),-len(t['points']),t['track_id']))[:limit]


def build_scene_contract(tracks, xyz, sgf_instances, sgf_relations, relation_vocab, H):
    from adapters.sgf.object_adapter import adapt_objects
    from adapters.sgf.graph_adapter import adapt_graph
    from adapters.sgf.relation_mapper import RelationMapper
    selected=select_tracks(tracks)
    segments={t['track_id']:xyz[t['points']] @ H[:3,:3].T + H[:3,3] for t in selected}
    objects=adapt_objects(segments)
    anchor={};records=[]
    for t in selected:
        ids=np.asarray(t['points'],int);observed=sgf_instances[ids];observed=observed[observed>0]
        if len(observed)<50:continue
        values,counts=np.unique(observed,return_counts=True);best=counts.argmax()
        if counts[best]/len(observed)>=.6:
            anchor[t['track_id']]=int(values[best]);records.append({'track':t['track_id'],'sgf_instance':int(values[best]),'anchor_points':int(counts[best])})
    inverse={}
    for t,i in anchor.items():inverse.setdefault(i,[]).append(t)
    triples=set()
    for r in sgf_relations:
        if r['confidence']<.5 or r['relation'] in ('none','same part'):continue
        for a in inverse.get(r['source_instance'],[]):
          for b in inverse.get(r['target_instance'],[]):
            if a!=b:triples.add((a,b,r['relation']))
    triples=sorted(triples)
    contract=adapt_graph(objects,mode='sgf_predicted',directed_pairs=sorted(set((a,b) for a,b,_ in triples)),
        relation_triples=triples,relation_mapper=RelationMapper(Path(relation_vocab)))
    return contract,selected,{'anchors':records,'transferred_predicted_relations':len(triples),'attribute_available':False,'relation_source':'existing SGF predicted scene graph; missing edges not invented'}


def measure_candidates(candidates, tracks_a, tracks_b, xyz):
    from .semantic_mapping import measured_association
    ta={t['track_id']:t for t in tracks_a};tb={t['track_id']:t for t in tracks_b}
    records=[]
    for a,b,score in candidates:
        x,y=ta[a],tb[b]
        r={'source_track':a,'target_track':b,'rank_score':float(score),'accepted_geometry':False}
        if not compatible(x['category'],y['category']):r['reason']='incompatible_category'
        else:
            p=xyz[x['points']];q=xyz[y['points']]
            if np.any(p.min(axis=0)>q.max(axis=0)+.05) or np.any(q.min(axis=0)>p.max(axis=0)+.05):r['reason']='disjoint_bounds'
            else:
                evidence=measured_association(p,q)
                r.update(evidence);r['accepted_geometry']=bool(evidence['accepted'])
        records.append(r)
    return records


def greedy_pairs(records, method):
    valid=[r for r in records if r['accepted_geometry']]
    if method=='geometry':valid.sort(key=lambda r:(-r['coverage_5cm'],r['rmse_m'],r['source_track'],r['target_track']))
    else:valid.sort(key=lambda r:(r['rank_score'],r['source_track'],r['target_track']))
    used_a=set();used_b=set();accepted=[]
    for r in valid:
        a,b=r['source_track'],r['target_track']
        if a in used_a or b in used_b:continue
        used_a.add(a);used_b.add(b);accepted.append((a,b))
    return accepted


def compare_associations(streams, xyz, sgf_instances, relations, relation_vocab, checkpoint, H, device='cuda'):
    from adapters.sgf.graph_adapter import merge_pair_contracts
    import inference.sgf_official.inference as official
    from .contracts import sha256_file
    import torch
    np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    expected='b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e'
    if sha256_file(Path(checkpoint))!=expected:raise ValueError('SGA checkpoint mismatch')
    chosen=[select_tracks(t) for t in streams]
    if min(map(len,chosen))<2:
        return {'geometry':[],'sga':[]},{'inference_executed':False,'status':'insufficient_objects','counts':list(map(len,chosen))}
    builds=[build_scene_contract(t,xyz,sgf_instances,relations,relation_vocab,H) for t in streams]
    contracts=[x[0] for x in builds];selected=[x[1] for x in builds]
    center=xyz.mean(axis=0) @ H[:3,:3].T + H[:3,3]
    data=merge_pair_contracts(*contracts,center)
    official.OFFICIAL_SNAPSHOT=str(checkpoint)
    embedding,epoch=official.official_forward(data,'official_sgf_predicted',device=device)
    if not np.isfinite(embedding).all() or np.any(np.linalg.norm(embedding,axis=1)==0):raise ValueError('invalid SGA embeddings')
    n=len(contracts[0].obj_ids);embedding=embedding/np.linalg.norm(embedding,axis=1,keepdims=True)
    distance=1-embedding[:n]@embedding[n:].T
    legacy,_,_=official.official_matching(embedding,n)
    all_candidates=[];sga_candidate_ids=set()
    for i,a in enumerate(contracts[0].obj_ids):
        top=set(np.argsort(distance[i],kind='stable')[:3].tolist())
        for j,b in enumerate(contracts[1].obj_ids):
            if compatible(next(t['category'] for t in selected[0] if t['track_id']==a),next(t['category'] for t in selected[1] if t['track_id']==b)):
                all_candidates.append((int(a),int(b),float(distance[i,j])))
            if j in top:sga_candidate_ids.add((int(a),int(b)))
    measured=measure_candidates(all_candidates,*selected,xyz)
    sga_records=[r for r in measured if (r['source_track'],r['target_track']) in sga_candidate_ids]
    pairs={'geometry':greedy_pairs(measured,'geometry'),'sga':greedy_pairs(sga_records,'sga')}
    return pairs,{'status':'completed','inference_executed':True,'checkpoint_sha256':expected,'checkpoint_epoch':epoch,
       'modules':['pct','gat','rel'],'attribute_available':False,'legacy_official_mixed_stream_top3_count':len(legacy),
       'candidate_policy':'top3 cross-stream SGA embeddings; then same measured geometry gate and one-to-one matching',
       'stream_node_counts':[len(s) for s in selected],'contracts':[x[2] for x in builds],
       'geometry_candidates':measured,'sga_candidates':sga_records,'accepted_pairs':pairs,'pose_feedback':False,
       'T_model_world':H.tolist(),'device':torch.cuda.get_device_name(0)}


def object_consensus(streams, pairs, base, xyz):
    """Propagate within measured mask unions; preserve B's known fine labels."""
    keys=[(s,t['track_id']) for s,rows in enumerate(streams) for t in rows]
    tracks={(s,t['track_id']):t for s,rows in enumerate(streams) for t in rows}
    parent={k:k for k in keys}
    def root(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for a,b in pairs:
        ka,kb=(0,a),(1,b)
        if not compatible(tracks[ka]['category'],tracks[kb]['category']):raise ValueError('incompatible merge')
        parent[root(kb)]=root(ka)
    groups={}
    for k in keys:groups.setdefault(root(k),[]).append(tracks[k])
    sem=base['semantic'].copy();conf=base['confidence'].copy();instance=np.zeros(len(sem),np.int32)
    owner=np.zeros(len(sem),np.int32);claims=np.zeros(len(sem),np.int32);ambiguous=np.zeros(len(sem),bool);objects=[]
    fixed=(sem>0)&(sem<33)
    for gid,(groupkey,parts) in enumerate(sorted(groups.items()),1):
        frames=set(f for t in parts for f in t['frames']);ids=np.unique(np.concatenate([np.asarray(t['points'],int) for t in parts]))
        cats={t['category'] for t in parts};fine=cats-{33,34}
        if len(fine)>1:raise ValueError('contradictory fine labels in one group')
        category=next(iter(fine)) if fine else min(cats)
        score=float(np.average([t['mean_point_score'] for t in parts],weights=[len(t['points']) for t in parts]))
        if len(frames)<2 or score<.8 or len(ids)<50:continue
        ids=ids[(~fixed[ids])|(sem[ids]==category)]
        ambiguous[ids]|=(owner[ids]>0)&(owner[ids]!=gid)
        owner[ids]=gid;claims[ids]=category
        objects.append({'instance_id':gid,'semantic_id':category,'source_tracks':[[s,t['track_id']] for s,rows in enumerate(streams) for t in rows if root((s,t['track_id']))==groupkey],
                        'supporting_frames':sorted(frames),'mean_observed_mask_score':score})
    usable=(owner>0)&~ambiguous
    change=usable&~fixed
    sem[change]=claims[change]
    for ob in objects:
        ids=(owner==ob['instance_id'])&usable&(sem==ob['semantic_id']);instance[ids]=ob['instance_id']
        conf[ids&change]=ob['mean_observed_mask_score'];pts=xyz[ids]
        ob['point_count']=len(pts)
        if len(pts):ob.update(center=pts.mean(axis=0).tolist(),min=pts.min(axis=0).tolist(),max=pts.max(axis=0).tolist())
    assert np.array_equal(sem[fixed],base['semantic'][fixed])
    return {'semantic':sem,'instance':instance,'confidence':conf},[o for o in objects if o['point_count']>0],{
      'groups_before':len(keys),'groups_after':len(groups),'retained_groups':sum(o['point_count']>0 for o in objects),
      'ambiguous_ownership_points':int(ambiguous.sum()),'semantic_added_points':int(np.sum((sem>0)&(base['semantic']==0))),
      'coarse_refined_points':int(np.sum((base['semantic']>=33)&(sem<33)&(sem>0))),
      'b_fine_labels_preserved':True,'only_observed_mask_interior_points':True}
