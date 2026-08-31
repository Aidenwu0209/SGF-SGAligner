#!/usr/bin/env python3
"""One isolated official ColorPCR forward pass for one sentinel transform.

Run this file only with the separately audited jojo2026 interpreter.  It emits
correspondences, scores, and estimated transform only; GT-node outputs created
inside the upstream model are deliberately discarded.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from pathlib import Path

# The sentinel launches this worker with ``-I -S`` so ambient ``.pth`` files
# and editable checkouts cannot participate.  Add only the site-packages owned
# by the hash-bound interpreter; the parent audit seals every file actually
# consumed from this directory.
_PYTHON_PREFIX = Path(sys.executable).resolve().parent.parent
_PYTHON_SITE = (_PYTHON_PREFIX / "lib" /
                f"python{sys.version_info.major}.{sys.version_info.minor}" /
                "site-packages")
if not _PYTHON_SITE.is_dir():
    raise SystemExit("sealed jojo site-packages missing")
sys.path.insert(0, str(_PYTHON_SITE))

import numpy as np

INPUT_VOXEL_M = 0.10
COARSEST_CAP = 512
FROZEN_NEIGHBOR_LIMITS = [38,36,36,38]

def fhash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def write_npz_create_only(path: Path, **arrays) -> None:
    """Publish one worker artifact without an overwrite/rename window.

    The active fixed4 parent grants a fresh pathname for every attempt.  A
    pre-existing path therefore means that the authorization is stale or the
    attempt is being replayed; fail closed instead of truncating it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())

def python_tree_hash(repo: Path) -> str:
    digest=hashlib.sha256()
    files=[]
    for root in (repo/"geotransformer",repo/"experiments/ColorPCR"):
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in sorted(files,key=lambda item:item.relative_to(repo).as_posix()):
        relative=path.relative_to(repo).as_posix().encode()
        digest.update(relative);digest.update(b"\0");digest.update(path.read_bytes());digest.update(b"\0")
    return digest.hexdigest()

def sentinel(name: str) -> np.ndarray:
    if name == "identity": return np.eye(4,dtype=np.float32)
    if name != "proper_nonzero": raise ValueError("unknown sentinel")
    angle=math.radians(90.0);c,s=math.cos(angle),math.sin(angle)
    t=np.eye(4,dtype=np.float32);t[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
    t[:3,3]=[0.123,-0.071,0.049]
    return t

def move(value, torch, device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,list):return [move(x,torch,device) for x in value]
    if isinstance(value,dict):return {k:move(v,torch,device) for k,v in value.items()}
    return value

def proportional_targets(lengths, max_points=COARSEST_CAP):
    """Allocate exactly max_points across scans by deterministic largest remainder."""
    values=np.asarray(lengths,dtype=np.int64)
    total=int(values.sum())
    if total<=max_points:return values.copy()
    if len(values)>max_points or np.any(values<=0):raise ValueError("invalid stacked lengths")
    quotas=values.astype(np.float64)*(float(max_points)/float(total))
    targets=np.maximum(1,np.floor(quotas).astype(np.int64))
    while int(targets.sum())<max_points:
        candidates=[i for i in range(len(values)) if targets[i]<values[i]]
        index=max(candidates,key=lambda i:(quotas[i]-targets[i],-i))
        targets[index]+=1
    while int(targets.sum())>max_points:
        candidates=[i for i in range(len(values)) if targets[i]>1]
        index=min(candidates,key=lambda i:(quotas[i]-targets[i],i))
        targets[index]-=1
    return targets

def cap_coarsest_points_fps(points,lengths,hsv,torch,max_points=COARSEST_CAP):
    values=[int(x) for x in lengths.detach().cpu().tolist()]
    targets=proportional_targets(values,max_points)
    if sum(values)<=max_points:return points,lengths,hsv,False
    out_points=[];out_hsv=[];start=0
    for length,target in zip(values,targets.tolist()):
        segment=points[start:start+length];segment_hsv=hsv[start:start+length]
        chosen=torch.zeros(target,dtype=torch.long,device=segment.device)
        distance=torch.full((length,),float("inf"),dtype=segment.dtype,device=segment.device)
        farthest=0
        for index in range(target):
            chosen[index]=farthest
            delta=segment-segment[farthest]
            distance=torch.minimum(distance,torch.sum(delta*delta,dim=1))
            farthest=int(torch.argmax(distance).item())
        out_points.append(segment[chosen]);out_hsv.append(segment_hsv[chosen]);start+=length
    result_lengths=torch.as_tensor(targets,dtype=lengths.dtype,device=lengths.device)
    return torch.cat(out_points),result_lengths,torch.cat(out_hsv),True

def precompute_capped(points,lengths,num_stages,voxel_size,radius,neighbor_limits,hsv,torch,
                      grid_subsample_dps,radius_search):
    if num_stages!=len(neighbor_limits):raise ValueError("neighbor limit/stage mismatch")
    points_list=[];lengths_list=[];hsv_list=[];cap_applied=False
    for stage in range(num_stages):
        if stage>0:
            points,hsv,lengths=grid_subsample_dps(points,hsv,lengths,voxel_size=voxel_size)
        if stage==num_stages-1:
            points,lengths,hsv,cap_applied=cap_coarsest_points_fps(
                points,lengths,hsv,torch,max_points=COARSEST_CAP)
        points_list.append(points);lengths_list.append(lengths);hsv_list.append(hsv);voxel_size*=2
    neighbors=[];subsampling=[];upsampling=[]
    for stage in range(num_stages):
        cur_points,cur_lengths=points_list[stage],lengths_list[stage]
        neighbors.append(radius_search(cur_points,cur_points,cur_lengths,cur_lengths,
                                       radius,neighbor_limits[stage]))
        if stage<num_stages-1:
            sub_points,sub_lengths=points_list[stage+1],lengths_list[stage+1]
            subsampling.append(radius_search(sub_points,cur_points,sub_lengths,cur_lengths,
                                             radius,neighbor_limits[stage]))
            upsampling.append(radius_search(cur_points,sub_points,cur_lengths,sub_lengths,
                                             radius*2,neighbor_limits[stage+1]))
        radius*=2
    return {"points":points_list,"lengths":lengths_list,"hsv":hsv_list,
            "neighbors":neighbors,"subsampling":subsampling,"upsampling":upsampling},cap_applied

def registration_collate_capped(raw,cfg,limits,torch,grid_subsample_dps,radius_search):
    ref_points=torch.from_numpy(raw["ref_points"]);src_points=torch.from_numpy(raw["src_points"])
    ref_hsv=torch.from_numpy(raw["ref_hsv"]);src_hsv=torch.from_numpy(raw["src_hsv"])
    points=torch.cat([ref_points,src_points]);hsv=torch.cat([ref_hsv,src_hsv])
    lengths=torch.LongTensor([len(ref_points),len(src_points)])
    stack,cap_applied=precompute_capped(points,lengths,cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,cfg.backbone.init_radius,limits,hsv,torch,
        grid_subsample_dps,radius_search)
    data={"features":torch.ones((len(points),1),dtype=torch.float32),
          "transform":torch.from_numpy(raw["transform"]),"batch_size":1,**stack}
    return data,cap_applied

def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--expected-commit",required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--expected-weight-sha256",required=True);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--expected-python-tree-sha256",required=True)
    p.add_argument("--expected-tracked-diff-sha256",required=True)
    p.add_argument("--extension",type=Path,required=True);p.add_argument("--expected-extension-sha256",required=True)
    p.add_argument("--arm",choices=("sgf_selected_union","fullscan"),required=True)
    p.add_argument("--direction",choices=("forward","reverse"),required=True)
    p.add_argument("--sentinel",choices=("identity","proper_nonzero"),required=True)
    p.add_argument("--neighbor-limits",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sampling",choices=("voxel10",),default="voxel10")
    p.add_argument("--device",default="cuda:0");args=p.parse_args()
    repo=args.repo.resolve();weights=args.weights.resolve()
    # Git identity and the complete repository closure are already verified by
    # the signed parent input manifest.  Re-running git here consumed mutable
    # .git/worktree state after authorization and made the runtime less sealed.
    # Retain the signed values in the output metadata and independently recheck
    # the immutable Python tree, weight, and native extension bytes below.
    commit=args.expected_commit
    tracked_diff=args.expected_tracked_diff_sha256
    extension=args.extension.resolve()
    if len(commit)!=40 or len(tracked_diff)!=64 \
            or fhash(weights) != args.expected_weight_sha256 \
            or python_tree_hash(repo)!=args.expected_python_tree_sha256 \
            or fhash(extension)!=args.expected_extension_sha256:
        raise SystemExit("ColorPCR source/weight identity mismatch")
    limits=[int(x) for x in args.neighbor_limits.split(",")]
    if limits!=FROZEN_NEIGHBOR_LIMITS:
        raise SystemExit("neighbor limits must be frozen [38,36,36,38]")
    sys.path[:0]=[str(repo),str(repo/"experiments/ColorPCR")]
    import torch
    from skimage.color import rgb2hsv
    # Official ColorPCR's config module creates training output directories at
    # import time.  Inference does not consume them, so suppress only that
    # side effect while retaining the official configuration values.
    import geotransformer.utils.common as geot_common
    original_ensure_dir=geot_common.ensure_dir
    geot_common.ensure_dir=lambda _path: None
    try:
        from config import make_cfg
    finally:
        geot_common.ensure_dir=original_ensure_dir
    from model import create_model
    from geotransformer.modules.ops import grid_subsample_dps, radius_search
    random.seed(7351);np.random.seed(7351);torch.manual_seed(7351);torch.cuda.manual_seed_all(7351)
    torch.use_deterministic_algorithms(True,warn_only=False)
    with np.load(args.input,allow_pickle=False) as d:
        def get(side,key):return np.asarray(d[f"{args.arm}_{side}_voxel10_{key}"])
        src_xyz,ref_xyz=get("source","xyz"),get("reference","xyz")
        src_rgb,ref_rgb=get("source","colors_mean_0_255"),get("reference","colors_mean_0_255")
    if args.direction=="reverse":
        src_xyz,ref_xyz=ref_xyz,src_xyz;src_rgb,ref_rgb=ref_rgb,src_rgb
    src_hsv=rgb2hsv(src_rgb.astype(np.float32)/255.0).astype(np.float32)
    ref_hsv=rgb2hsv(ref_rgb.astype(np.float32)/255.0).astype(np.float32)
    raw={"ref_points":ref_xyz.astype(np.float32),"src_points":src_xyz.astype(np.float32),
         "ref_feats":np.ones((len(ref_xyz),1),np.float32),"src_feats":np.ones((len(src_xyz),1),np.float32),
         "ref_hsv":ref_hsv,"src_hsv":src_hsv,"transform":sentinel(args.sentinel)}
    cfg=make_cfg();data,cap_applied=registration_collate_capped(raw,cfg,limits,torch,
        grid_subsample_dps,radius_search)
    stage_lengths=[[int(x) for x in stage.detach().cpu().tolist()] for stage in data["lengths"]]
    if sum(stage_lengths[-1])>COARSEST_CAP:raise SystemExit("coarsest cap failed")
    snapshot=torch.load(weights,map_location="cpu");model=create_model(cfg)
    model.load_state_dict(snapshot["model"],strict=True);model.eval().to(args.device)
    with torch.no_grad():out=model(move(data,torch,args.device))
    release=lambda x:x.detach().cpu().numpy()
    arrays={"ref_corr_points":release(out["ref_corr_points"]).astype(np.float32),
            "src_corr_points":release(out["src_corr_points"]).astype(np.float32),
            "corr_scores":release(out["corr_scores"]).astype(np.float32),
            "estimated_transform":release(out["estimated_transform"]).astype(np.float64)}
    meta={"schema":"v13-colorpcr-sentinel-worker-v2","sentinel":args.sentinel,
          "sentinel_transform":sentinel(args.sentinel).tolist(),"direction":args.direction,
          "arm":args.arm,"input_sha256":fhash(args.input),"repo_commit":commit,
          "weight_sha256":fhash(weights),"neighbor_limits":limits,
          "python_tree_sha256":python_tree_hash(repo),
          "tracked_diff_sha256":tracked_diff,
          "extension_sha256":fhash(extension),
          "input_voxel_m":INPUT_VOXEL_M,"sampling":args.sampling,
          "coarsest_cap":COARSEST_CAP,"coarsest_cap_applied":cap_applied,
          "stage_lengths":stage_lengths,
          "torch_version":torch.__version__,"device":args.device,
          "forbidden_outputs_omitted":["gt_node_corr_indices","gt_node_corr_overlaps","transform"]}
    write_npz_create_only(
        args.output, **arrays,
        meta_json=np.asarray(json.dumps(meta, sort_keys=True)))

if __name__=="__main__":main()
