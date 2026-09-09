"""Read-back verification of labeled PLY, graph IDs and full frame coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData
from pose_pipeline.contracts import sha256_file


def verify(root):
    record = json.loads((root/'result.json').read_text())
    source = Path(record.get('source_sgf_sga_run',str(root)))
    inputs = {Path(k):v for k,v in record['input_sha256'].items()}
    for path,digest in inputs.items():
        if sha256_file(path)!=digest:raise ValueError(f'input changed: {path}')
    baseline = next(p for p in inputs if p.suffix=='.ply')
    manifest_path = None
    for p in inputs:
        if p.suffix=='.json':
            value = json.loads(p.read_text())
            if value.get('schema')=='rgbd_sequence_manifest.v1':
                manifest_path = p;manifest = value
    if manifest_path is None:raise ValueError('no bound manifest')
    expected = {int(x['frame_id']) for x in manifest['frames']}
    seen = set()
    graphs = []
    for p in sorted(source.glob('submap_*/replay.json')):
        replay = json.loads(p.read_text())
        if replay['processed_frames']!=len(replay['frame_ids']):raise ValueError('replay count')
        seen.update(replay['frame_ids'])
        graph = json.loads((p.parent/'graph.json').read_text())
        if graph.get('prediction_enabled') is not True:raise ValueError('SGF prediction disabled')
        graphs.append(p.parent)
    if seen!=expected:raise ValueError('semantic replay does not cover all raw frames')
    old = PlyData.read(baseline)['vertex'].data
    new = PlyData.read(root/'map_labeled.ply')['vertex'].data
    if len(old)!=len(new):raise ValueError('point count changed')
    for name in old.dtype.names:
        if not np.array_equal(old[name],new[name]):raise ValueError(f'changed original field {name}')
    for field in ('semantic_id','instance_id'):
        if new[field].dtype.kind not in 'iu':raise ValueError('label fields must be integer')
        if np.any(new[field]<0):raise ValueError('negative label')
    classes = {int(k) for k in json.loads((root/'classes.json').read_text())}
    if not set(np.unique(new['semantic_id'])).issubset(classes):raise ValueError('unknown class ID')
    objects = json.loads((root/'objects.json').read_text())
    ids = {x['instance_id'] for x in objects}
    if ids != set(np.unique(new['instance_id']))-{0}:raise ValueError('object table IDs mismatch')
    for obj in objects:
        mask = new['instance_id']==obj['instance_id']
        if mask.sum()!=obj['point_count']:raise ValueError('object point count mismatch')
        if not np.all(new['semantic_id'][mask]==obj['semantic_id']):raise ValueError('mixed instance categories')
    for e in json.loads((root/'scene_graph.json').read_text())['relations']:
        if e['source_instance'] not in ids or e['target_instance'] not in ids:
            raise ValueError('dangling scene graph edge')
    return {'status':'passed','sequence':record['sequence'],'full_frame_count':len(seen),
        'submaps':len(graphs),'point_count':len(new),'original_fields_identical':list(old.dtype.names),
        'label_coverage':float(np.mean(new['instance_id']>0)),'instances':len(ids),
        'ply_sha256':sha256_file(root/'map_labeled.ply'),'quality_accepted':False,
        'scope':'artifact integrity and execution coverage, not semantic accuracy'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result',type=Path)
    a = p.parse_args()
    report = verify(a.result)
    with (a.result/'verification.json').open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
