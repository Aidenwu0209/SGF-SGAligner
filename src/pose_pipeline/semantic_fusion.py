"""Fuse independent SGF streams; preserve semantics when instance IDs disagree.

Only final snapshots from disjoint frame sets are independent observations.
Cumulative checkpoints must not be passed here as separate streams.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def transfer_consensus(points, normals, submaps, ids, class_ids, max_distance=.04,
                       *, use_surfel_footprints=False):
    if len(submaps)!=len(ids):raise ValueError('one ID mapping per stream is required')
    if not class_ids or min(class_ids.values())<1:raise ValueError('positive class IDs are required')
    n=len(points);c=max(class_ids.values())+1
    score=np.zeros((n,c),np.float32);weight=np.zeros(n,np.float32)
    claims=[]
    pn=np.linalg.norm(normals,axis=1)
    for scene,mapping in zip(submaps,ids):
        cloud=scene['cloud'];g=scene['graph']
        if len(cloud['xyz'])==0:continue
        probability={}
        for label,name,conf,probs in zip(g['node_labels'],g['semantic_labels'],g['semantic_confidences'],g['semantic_probabilities']):
            v=np.zeros(c,np.float32)
            for key,value in probs.items():
                if key in class_ids and np.isfinite(value) and value>0:v[class_ids[key]]=value
            if v.sum()>0:v/=v.sum()
            probability[int(label)]=v
        unique,rev=np.unique(cloud['labels'],return_inverse=True)
        pr=np.array([probability.get(int(x),np.zeros(c,np.float32)) for x in unique])[rev]
        gi=np.array([mapping.get(int(x),0) for x in unique],np.int32)[rev]
        if use_surfel_footprints:
            radius=np.asarray(cloud['radius_m'])
            if radius.shape!=(len(cloud['xyz']),) or not np.isfinite(radius).all() or np.any(radius<0):
                raise ValueError('finite nonnegative radius_m required for every surfel')
        ds,ix=cKDTree(cloud['xyz']).query(points,k=min(8 if use_surfel_footprints else 3,len(cloud['xyz'])),workers=1)
        if ds.ndim==1:ds,ix=ds[:,None],ix[:,None]
        cn=np.linalg.norm(cloud['normals'],axis=1)
        best=np.full(n,np.inf);selected=np.zeros(n,np.int64)
        for k in range(ds.shape[1]):
            j=ix[:,k];co=np.abs(np.einsum('ij,ij->i',normals,cloud['normals'][j]))/np.maximum(pn*cn[j],1e-12)
            distance=ds[:,k]
            if use_surfel_footprints:
                delta=points-cloud['xyz'][j]
                perpendicular=np.abs(np.einsum('ij,ij->i',delta,cloud['normals'][j]))/np.maximum(cn[j],1e-12)
                tangent=np.sqrt(np.maximum(distance**2-perpendicular**2,0))
                distance=np.hypot(perpendicular,np.maximum(tangent-radius[j],0))
            valid=(distance<=max_distance)&(co>=.7071)&(pr[j].sum(axis=1)>0)
            win=valid&(distance<best);best[win]=distance[win];selected[win]=j[win]
        valid=np.isfinite(best);w=np.where(valid,1/(1+best/max_distance),0).astype(np.float32)
        score+=pr[selected]*w[:,None];weight+=w
        claims.append((valid,gi[selected],pr[selected].argmax(axis=1)))
    avg=score/np.maximum(weight[:,None],1e-12)
    order=np.argsort(avg,axis=1);sem=order[:,-1].astype(np.int32)
    conf=avg[np.arange(n),sem];runner=avg[np.arange(n),order[:,-2]]
    known=(weight>0)&(conf>=.5)&((conf-runner)>=.15)&(sem>0)
    sem[~known]=0;conf[~known]=0
    instance=np.zeros(n,np.int32);ambiguous=np.zeros(n,bool)
    for valid,claimed,category in claims:
        useful=valid&known&(category==sem)&(claimed>0)
        ambiguous|=useful&(instance>0)&(instance!=claimed)
        take=useful&(instance==0);instance[take]=claimed[take]
    instance[ambiguous]=0
    return sem,instance,conf.astype(np.float32),int(ambiguous.sum())


def transfer_surfel_consensus(points,normals,submaps,ids,class_ids,max_distance=.04):
    """Distance to measured native surfel discs, retaining the 4cm tolerance."""
    return transfer_consensus(points,normals,submaps,ids,class_ids,max_distance,
                              use_surfel_footprints=True)


def transfer_surfel_completion(points,normals,submaps,ids,class_ids,max_distance=.04):
    """Fill previously unknown points from measured discs; preserve known labels."""
    center=transfer_consensus(points,normals,submaps,ids,class_ids,max_distance)
    disc=transfer_surfel_consensus(points,normals,submaps,ids,class_ids,max_distance)
    known=center[0]>0
    semantic,instance,confidence=[np.where(known,a,b) for a,b in zip(center[:3],disc[:3])]
    return semantic,instance,confidence,int(np.sum((semantic>0)&(instance==0)))
