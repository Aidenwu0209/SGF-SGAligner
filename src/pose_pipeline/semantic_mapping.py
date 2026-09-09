"""Experimental SGF/SGA labels over a frozen, complete RGB-D reconstruction.

No trajectory or baseline vertex is optimized here. Independent overlapping
SGF submaps provide predictions; SGA proposes associations, verified in the
shared estimated world frame before creating scene-scoped instance IDs.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .contracts import (bind_manifest_trajectory, load_manifest, load_trajectory,
                        sha256_file, validate_se3)
from reconstruction.rgbd_refusion import _read_rgbd
from .rgbd_mapping import _run_stage


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                    allow_nan=False) + "\n")


def sgf_pose(t_world_camera):
    result = np.linalg.inv(validate_se3(t_world_camera)).copy()
    result[:3, 3] *= 1000.0
    return np.ascontiguousarray(result, dtype=np.float32)


def windows(count, size, overlap):
    if count < 1 or size < 2 or not 0 <= overlap < size:
        raise ValueError("invalid submap frame window")
    start = 0
    while True:
        end = min(count, start + size)
        yield start, end
        if end == count:
            break
        start = end - overlap


def replay_submap(bound, depth_scale, model, output):
    from sgf_runtime import CameraIntrinsics, FramePacket, SceneGraphFusion
    output.mkdir()
    first_rgb, first_depth, k = _read_rgbd(bound[0][0])
    h, w = first_depth.shape
    engine = SceneGraphFusion(
        CameraIntrinsics(w, h, *k), model_path=model, enable_prediction=True,
        use_thread=False, segment_filter=96, min_pyr_level=2,
        depth_edge_threshold=0.98, sampling_seed=42, num_sample_points=128,
        sample_with_replacement=True,
    )
    started = time.monotonic()
    try:
        for frame, pose in bound:
            rgb, depth, intrinsics = _read_rgbd(frame)
            if depth.shape != (h, w) or not np.allclose(k, intrinsics):
                raise ValueError("SGF submap requires constant calibrated intrinsics")
            millimetres = np.rint(depth.astype(np.float64) * 1000 / depth_scale)
            # Match the fixed geometry depth truncation; invalid depth stays 0.
            millimetres[(millimetres > 4500) | (millimetres < 0)] = 0
            packet = FramePacket(
                frame.frame_id, np.ascontiguousarray(rgb[:, :, ::-1]),
                np.ascontiguousarray(millimetres, dtype=np.uint16),
                sgf_pose(pose.t_world_camera), frame.timestamp_us * 1000,
                time.monotonic_ns(),
            )
            engine.process_frame(packet)
        engine.run_full_prediction(min_segment_points=50)
        cloud = engine.native.snapshot_inseg(min_segment_points=1)
        graph = engine.native.snapshot_graph(min_segment_points=1)
        if graph.get('prediction_enabled') is not True:
            raise RuntimeError('SGF silently disabled prediction; semantic mapping failed')
        # Feature tensors can be recomputed by SGA from these observed objects.
        graph = {k: v for k, v in graph.items()
                 if k not in {"node_features", "node_feature_valid"}}
        if cloud["coordinate_unit"] != "metre":
            raise ValueError("SGF snapshot is not in metres")
        np.savez_compressed(output / "inseg_cloud.npz", **{
            k: cloud[k] for k in ("xyz", "normals", "colors", "labels")})
        write_json(output / "graph.json", graph)
        write_json(output / "replay.json", {
            "frame_ids": [f.frame_id for f, _ in bound],
            "processed_frames": len(bound), "seconds": time.monotonic()-started,
            "pose_source": "frozen estimated T_world_camera, inverse in millimetres",
            "prediction_enabled": True,
        })
    finally:
        engine.stop()
    return load_submap(output)


def load_submap(path):
    with np.load(path / "inseg_cloud.npz", allow_pickle=False) as data:
        cloud = {k: data[k] for k in data.files}
    graph = json.loads((path / "graph.json").read_text())
    nodes = {}
    for i, label in enumerate(graph["node_labels"]):
        nodes[int(label)] = {
            "label": graph["semantic_labels"][i],
            "confidence": float(graph["semantic_confidences"][i]),
            "native_instance_id": int(graph["instance_labels"][i]),
        }
    return {"cloud": cloud, "graph": graph, "nodes": nodes}


def measured_association(a, b):
    """Shared-frame overlap plus an ICP measurement; never apply its pose."""
    import open3d as o3d
    da = cKDTree(b).query(a, workers=1)[0]
    db = cKDTree(a).query(b, workers=1)[0]
    coverage = min(float(np.mean(da < .05)), float(np.mean(db < .05)))
    if coverage < .3:
        return {"accepted": False, "coverage_5cm": coverage, "reason": "overlap"}
    def pc(x):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(x)
        return p.voxel_down_sample(.02)
    fit = o3d.pipelines.registration.registration_icp(
        pc(a), pc(b), .05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20))
    transform = fit.transformation
    angle = float(np.degrees(np.arccos(np.clip((np.trace(transform[:3,:3])-1)/2, -1, 1))))
    translation = float(np.linalg.norm(transform[:3, 3]))
    accepted = bool(fit.fitness >= .3 and fit.inlier_rmse <= .03
                    and translation <= .10 and angle <= 5)
    return {"accepted": accepted, "coverage_5cm": coverage,
            "fitness": fit.fitness, "rmse_m": fit.inlier_rmse,
            "translation_m": translation, "rotation_deg": angle,
            "T_ref_src_measured_only": transform, "applied_to_trajectory": False}


def associate(previous, current, relation_vocab, device):
    from adapters.sgf.object_adapter import adapt_objects
    from adapters.sgf.graph_adapter import adapt_graph, merge_pair_contracts
    from adapters.sgf.relation_mapper import RelationMapper
    from inference.sgf_official.inference import official_forward, official_matching
    contracts = []
    for scene in (previous, current):
        cloud = scene["cloud"]
        segments = {oid: cloud["xyz"][cloud["labels"] == oid]
                    for oid in scene["nodes"] if oid > 0}
        if sum(len(x) >= 50 for x in segments.values()) < 2:
            return [], {"status": "insufficient_objects", "inference_executed": False}
        objects = adapt_objects(segments)
        g = scene["graph"]
        triples = [(int(a), int(b), label) for (a,b),label,conf in zip(
            g["relation_edges"], g["relation_labels"], g["relation_confidences"])
            if a in objects.object_id2idx and b in objects.object_id2idx
            and label != "none" and conf >= .5]
        contracts.append(adapt_graph(objects, mode="sgf_predicted",
            directed_pairs=[(a,b) for a,b,_ in triples], relation_triples=triples,
            relation_mapper=RelationMapper(relation_vocab)))
    center = previous["cloud"]["xyz"].mean(axis=0)
    data = merge_pair_contracts(*contracts, center)
    embedding, epoch = official_forward(data, "official_sgf_predicted", device=device)
    if not np.isfinite(embedding).all() or np.any(np.linalg.norm(embedding,axis=1) == 0):
        raise ValueError("SGA returned invalid embeddings")
    count = len(contracts[0].obj_ids)
    candidates, _, distance = official_matching(embedding, count)
    candidates = sorted(candidates, key=lambda pair: float(distance[pair]))
    used_a, used_b, accepted, records = set(), set(), [], []
    for i, j in candidates:
        a, b = int(contracts[0].obj_ids[i]), int(contracts[1].obj_ids[j-count])
        na, nb = previous["nodes"][a], current["nodes"][b]
        row = {"source_segment": a, "target_segment": b,
               "embedding_distance": float(distance[i,j]), "accepted": False}
        if a in used_a or b in used_b:
            row["reason"] = "one_to_one_conflict"
        elif (na["confidence"] < .5 or nb["confidence"] < .5
              or na["label"] != nb["label"] or na["label"] in {"", "__unknown__"}):
            row["reason"] = "semantic_uncertainty_or_conflict"
        else:
            row.update(measured_association(
                contracts[0].registration_pts[i], contracts[1].registration_pts[j-count]))
            if row["accepted"]:
                used_a.add(a); used_b.add(b); accepted.append((a,b))
        records.append(row)
    return accepted, {"status": "completed", "inference_executed": True,
                      "checkpoint_epoch": epoch, "candidates": records,
                      "accepted_count": len(accepted), "pose_feedback": False}


def transfer_labels(points, normals, submaps, ids, class_ids, max_distance=.04):
    """Conservative nearest-surface evidence; disagreeing close labels abstain."""
    n = len(points)
    best = np.full(n, np.inf)
    runner = np.full(n, np.inf)
    instance = np.zeros(n, np.int32)
    semantic = np.zeros(n, np.int32)
    confidence = np.zeros(n, np.float32)
    point_normal_norm = np.linalg.norm(normals,axis=1)
    for scene, mapping in zip(submaps, ids):
        cloud = scene["cloud"]
        if not len(cloud["xyz"]):
            continue
        unique, inverse = np.unique(cloud['labels'],return_inverse=True)
        global_id = np.array([mapping.get(int(x),0) for x in unique],np.int32)[inverse]
        class_id = np.array([class_ids.get(scene['nodes'].get(int(x),{}).get('label'),0)
                             for x in unique],np.int32)[inverse]
        class_confidence = np.array([scene['nodes'].get(int(x),{}).get('confidence',0)
                                    for x in unique],np.float32)[inverse]
        native_normal_norm = np.linalg.norm(cloud['normals'],axis=1)
        dist, ix = cKDTree(cloud["xyz"]).query(points, k=min(3,len(cloud["xyz"])),workers=1)
        if dist.ndim == 1:
            dist, ix = dist[:,None], ix[:,None]
        for col in range(dist.shape[1]):
            ii = global_id[ix[:,col]]
            ss = class_id[ix[:,col]]
            cc = class_confidence[ix[:,col]]
            sn = cloud["normals"][ix[:,col]]
            # Normal sign is irrelevant; opposite surface direction is equivalent.
            cosine = np.abs(np.einsum('ij,ij->i', normals, sn)) / np.maximum(
                point_normal_norm*native_normal_norm[ix[:,col]],1e-12)
            dd = dist[:,col].copy()
            dd[(ii==0)|(ss==0)|(cc<.5)|(cosine<.7071)|(dd>max_distance)] = np.inf
            # Keep the two closest DISTINCT instance IDs, including repeated
            # observations of a former runner-up in another submap.
            same = ii == instance
            wins = (dd < best) & np.isfinite(dd)
            promote = wins & ~same
            other = ~same & ~promote & (dd < runner) & np.isfinite(dd)
            runner[other] = dd[other]
            runner[promote] = best[promote]
            instance[wins], semantic[wins], confidence[wins] = ii[wins], ss[wins], cc[wins]
            best[wins] = dd[wins]
    # Within 5mm, competing instance claims are unresolved; keep unknown.
    known = np.isfinite(best)
    ambiguous = known & (runner <= best + .005)
    instance[ambiguous], semantic[ambiguous], confidence[ambiguous] = 0, 0, 0
    return semantic, instance, confidence, int(ambiguous.sum())


def export_map(baseline, output, submaps, ids, classes):
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(baseline)
    vertex = ply['vertex'].data
    points = np.column_stack([vertex[x] for x in ('x','y','z')])
    normals = np.column_stack([vertex[x] for x in ('nx','ny','nz')])
    semantic, instance, confidence, ambiguous = transfer_labels(
        points, normals, submaps, ids, classes)
    fields = [('semantic_id','<i4'),('instance_id','<i4'),('semantic_confidence','<f4')]
    if any(name in vertex.dtype.names for name,_ in fields):
        raise ValueError("baseline already contains label fields")
    tagged = np.empty(len(vertex), dtype=vertex.dtype.descr+fields)
    for name in vertex.dtype.names:
        tagged[name] = vertex[name]
    tagged['semantic_id'], tagged['instance_id'], tagged['semantic_confidence'] = semantic, instance, confidence
    def save(name, data):
        PlyData([PlyElement.describe(data,'vertex')],text=False).write(str(output/name))
    save('map_labeled.ply',tagged)
    for key, filename in [('semantic_id','map_semantic.ply'),('instance_id','map_instance.ply')]:
        colored = tagged.copy()
        values = tagged[key].astype(np.uint64)
        for channel, factor in zip(('red','green','blue'),(73,151,199)):
            colored[channel] = np.where(values>0, 50+(values*factor)%206, 90).astype(np.uint8)
        save(filename,colored)
    check = PlyData.read(output/'map_labeled.ply')['vertex'].data
    if not all(np.array_equal(vertex[name],check[name]) for name in vertex.dtype.names):
        raise RuntimeError("baseline geometry or RGB changed during label export")
    objects = []
    for iid in np.unique(instance):
        if iid == 0:
            continue
        mask = instance == iid
        semantic_values, counts = np.unique(semantic[mask],return_counts=True)
        objects.append({"instance_id": int(iid), "semantic_id": int(semantic_values[np.argmax(counts)]),
            "point_count": int(mask.sum()), "centroid_m": points[mask].mean(axis=0),
            "bbox_min_m": points[mask].min(axis=0), "bbox_max_m": points[mask].max(axis=0)})
    write_json(output/'objects.json',objects)
    write_json(output/'classes.json',{'0':'unknown', **{str(v):k for k,v in classes.items()}})
    return {"point_count":len(vertex),"labeled_points":int((instance>0).sum()),
        "label_coverage":float(np.mean(instance>0)),"ambiguous_points":ambiguous,
        "exported_instances":len(objects),"baseline_vertex_fields_identical":True}


def run(args):
    manifest_path, trajectory_path, baseline = args.manifest, args.trajectory, args.baseline
    manifest = load_manifest(manifest_path)
    trajectory, _ = load_trajectory(trajectory_path)
    bound = bind_manifest_trajectory(manifest, trajectory)
    source_paths = [manifest_path, trajectory_path, baseline]
    before = {str(p.resolve()):sha256_file(p) for p in source_paths}
    receipt_path = baseline.parent/'refusion_result.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('trajectory_sha256') != sha256_file(trajectory_path)
            or receipt.get('manifest_sha256') != sha256_file(manifest_path)
            or receipt.get('cloud_sha256') != sha256_file(baseline)):
        raise ValueError('baseline receipt does not bind this manifest/trajectory/cloud')
    classes = {name.strip():i+1 for i,name in enumerate((args.model/'classes.txt').read_text().splitlines())
               if name.strip()}
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    status = {'status':'running','dataset':manifest.dataset,'sequence':manifest.sequence_id,
        'geometry_baseline_commit':'1cf90f7', 'input_sha256':before,
        'raw_frame_count':len(bound), 'submap_frames':args.submap_frames,'overlap':args.overlap,
        'pose_feedback':False,'ground_truth_consumed':False,'unknown_label':0}
    write_json(out/'status.json',status)
    previous_sigterm = None
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGTERM)==signal.SIG_DFL:
        def terminated(signum,_frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous_sigterm = signal.signal(signal.SIGTERM,terminated)
    try:
        import sgf_native
        from inference.sgf_official.inference import OFFICIAL_SNAPSHOT
        status['native_sha256'] = sha256_file(Path(sgf_native.__file__))
        status['sga_checkpoint_sha256'] = sha256_file(Path(OFFICIAL_SNAPSHOT))
        status['sgf_model_sha256'] = {str(p.relative_to(args.model)):sha256_file(p)
                                    for p in sorted(args.model.rglob('*')) if p.is_file()}
        status['source_sha256'] = sha256_file(Path(__file__))
        src = Path(__file__).resolve().parents[1]
        components = ('pose_pipeline/semantic_mapping.py','pose_pipeline/semantic_instances.py',
            'pose_pipeline/semantic_sga_worker.py','pose_pipeline/contracts.py',
            'reconstruction/rgbd_refusion.py','adapters/sgf/object_adapter.py',
            'adapters/sgf/graph_adapter.py','adapters/sgf/relation_mapper.py',
            'inference/sgf_official/inference.py')
        status['source_component_sha256'] = {name:sha256_file(src/name) for name in components}
        submaps, ids, pair_reports = [], [], []
        next_id = 1
        for index,(start,end) in enumerate(windows(len(bound),args.submap_frames,args.overlap)):
            print(json.dumps({'submap':index,'start':start,'end':end}),flush=True)
            current = replay_submap(bound[start:end],manifest.depth_scale,args.model,out/f'submap_{index:03}')
            mapping = {}
            if submaps:
                pair_file = out/f'sga_{index-1:03}_{index:03}.json'
                command = [str(args.sga_python), '-m', 'pose_pipeline.semantic_sga_worker',
                    '--source',str(out/f'submap_{index-1:03}'),
                    '--target',str(out/f'submap_{index:03}'),
                    '--relation-vocab',str(args.relation_vocab),
                    '--device',args.device,'--output',str(pair_file)]
                import os
                execution = _run_stage(command,out/f'sga_{index-1:03}_{index:03}.log',300,dict(os.environ))
                if execution['returncode']!=0:
                    raise RuntimeError(f'SGA stage failed: {execution}')
                report = json.loads(pair_file.read_text())
                accepted = report['accepted_pairs']
                for a,b in accepted:
                    mapping[b] = ids[-1][a]
                pair_reports.append(report)
            for label in sorted(current['nodes']):
                if label <= 0:
                    continue
                if label not in mapping:
                    mapping[label] = next_id
                    next_id += 1
            ids.append(mapping); submaps.append(current)
            write_json(out/f'submap_{index:03}'/'global_ids.json',mapping)
            status.update(completed_submaps=len(submaps),processed_unique_frames=end)
            write_json(out/'status.json',status)
        from .semantic_instances import consolidate, final_scene_graph
        ids, grouping = consolidate(submaps,ids)
        write_json(out/'instance_grouping.json',grouping)
        metrics = export_map(baseline,out,submaps,ids,classes)
        exported_ids = {o['instance_id'] for o in json.loads((out/'objects.json').read_text())}
        write_json(out/'scene_graph.json',final_scene_graph(submaps,ids,exported_ids,manifest.sequence_id))
        after = {str(p.resolve()):sha256_file(p) for p in source_paths}
        if before != after:
            raise RuntimeError('frozen input changed')
        if status['source_component_sha256']!={name:sha256_file(src/name) for name in components}:
            raise RuntimeError('semantic mapping source changed during execution')
        status.update(status='completed',**metrics,
            sga_pairs_executed=sum(r['inference_executed'] for r in pair_reports),
            sga_accepted_associations=sum(r.get('accepted_count',0) for r in pair_reports),
            instance_grouping_merges=len(grouping['accepted_merges']),
            frozen_inputs_unchanged=True,quality_accepted=False)
        write_json(out/'result.json',status)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM,previous_sigterm)
        write_json(out/'status.json',status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','trajectory','baseline','model','relation-vocab','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--submap-frames',type=int,default=120)
    parser.add_argument('--overlap',type=int,default=30)
    parser.add_argument('--sga-python',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
