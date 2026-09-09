"""Retain SGF same-part evidence when consolidating SGA-linked instances."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .contracts import sha256_file


def consolidate(submaps, mappings):
    parent = {int(v):int(v) for m in mappings for v in m.values()}
    def root(i):
        while parent[i]!=i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    accepted, rejected = [],Counter()
    for index,(s,mapping) in enumerate(zip(submaps,mappings)):
        labels = s['cloud']['labels']
        order = np.argsort(labels,kind='stable')
        keys,starts = np.unique(labels[order],return_index=True)
        parts = np.split(order,starts[1:])
        points = {int(k):s['cloud']['xyz'][ix] for k,ix in zip(keys,parts)}
        trees = {}
        seen = set()
        g = s['graph']
        for (a,b),relation,conf in zip(g['relation_edges'],g['relation_labels'],g['relation_confidences']):
            a,b = int(a),int(b)
            if relation!='same part' or a==b or a not in mapping or b not in mapping:
                continue
            pair = tuple(sorted((a,b)))
            if pair in seen:continue
            # Mark only after confidence, so a stronger reverse edge is usable.
            if conf<.7:
                rejected['low_relation_confidence'] += 1;continue
            seen.add(pair)
            na,nb = s['nodes'][a],s['nodes'][b]
            if (na['native_instance_id']<=0 or na['native_instance_id']!=nb['native_instance_id']
                    or na['label']!=nb['label'] or min(na['confidence'],nb['confidence'])<.5):
                rejected['semantic_or_native_instance_conflict'] += 1;continue
            if root(mapping[a])==root(mapping[b]):continue
            if min(len(points[a]),len(points[b]))<50:
                rejected['insufficient_surface'] += 1;continue
            if b not in trees:trees[b] = cKDTree(points[b])
            distances = trees[b].query(points[a],workers=1)[0]
            contact = int(np.sum(distances<=.05))
            if contact<3:
                rejected['no_surface_contact'] += 1;continue
            old_a,old_b = root(mapping[a]),root(mapping[b])
            parent[max(old_a,old_b)] = min(old_a,old_b)
            accepted.append({'submap':index,'segments':[a,b],
                'merged_global_ids':[old_a,old_b],'relation_confidence':conf,
                'contact_points_5cm':contact,'semantic_label':na['label']})
    result = [{int(k):root(int(v)) for k,v in m.items()} for m in mappings]
    return result,{'policy':'SGF native same-part + matching confident category + surface contact',
        'same_part_min_confidence':.7,'contact_distance_m':.05,'min_contact_points':3,
        'accepted_merges':accepted,'rejected_counts':dict(rejected),
        'before_global_ids':len(parent),'after_global_ids':len({root(i) for i in parent}),
        'pose_feedback':False}


def final_scene_graph(submaps,mappings,exported_ids,sequence):
    relations = []
    for index,(scene,m) in enumerate(zip(submaps,mappings)):
        g = scene['graph']
        for (a,b),name,conf in zip(g['relation_edges'],g['relation_labels'],g['relation_confidences']):
            if (a in m and b in m and m[a]!=m[b] and name!='none' and conf>=.5
                    and m[a] in exported_ids and m[b] in exported_ids):
                relations.append({'source_instance':m[a],'target_instance':m[b],
                                  'relation':name,'confidence':conf,'submap':index})
    return {'instance_scope':sequence,'relations':relations,'node_file':'objects.json',
            'submap_instance_ids':mappings}


def main():
    from .semantic_mapping import load_submap, export_map, write_json
    p = argparse.ArgumentParser(description='Refine only labels over completed frozen SGF/SGA outputs')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    source = a.source.resolve()
    record = json.loads((source/'result.json').read_text())
    if record['status']!='completed':raise ValueError('source must be completed')
    baseline = next(Path(x) for x in record['input_sha256'] if x.endswith('/refused.ply'))
    if sha256_file(baseline)!=record['input_sha256'][str(baseline)]:
        raise ValueError('baseline changed')
    paths = sorted(source.glob('submap_*'))
    evidence = [source/'result.json',source/'classes.json']
    evidence += [d/name for d in paths for name in ('graph.json','inseg_cloud.npz','global_ids.json')]
    before = {str(x):sha256_file(x) for x in evidence}
    scenes = [load_submap(d) for d in paths]
    ids = [{int(k):int(v) for k,v in json.loads((d/'global_ids.json').read_text()).items()} for d in paths]
    merged,report = consolidate(scenes,ids)
    classes = {v:int(k) for k,v in json.loads((source/'classes.json').read_text()).items() if int(k)>0}
    a.output.mkdir(parents=True,exist_ok=False)
    metrics = export_map(baseline,a.output,scenes,merged,classes)
    visible = {o['instance_id'] for o in json.loads((a.output/'objects.json').read_text())}
    write_json(a.output/'scene_graph.json',final_scene_graph(scenes,merged,visible,record['sequence']))
    write_json(a.output/'instance_grouping.json',report)
    if before!={str(x):sha256_file(x) for x in evidence}:raise RuntimeError('source outputs changed')
    record.update(**metrics,instance_grouping_merges=len(report['accepted_merges']),
        source_sgf_sga_run=str(source),source_sgf_sga_sha256=before,
        refinement_source_sha256=sha256_file(Path(__file__)),
        export_source_sha256=sha256_file(Path(__file__).with_name('semantic_mapping.py')),
        validation_mode='fixed SGF/SGA predictions, instance consolidation and label export only',
        quality_accepted=False)
    write_json(a.output/'result.json',record)


if __name__=='__main__':main()
