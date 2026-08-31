#!/usr/bin/env python3
"""Launch two isolated ColorPCR sentinels and seal invariant corr cache."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np

ARRAYS=("ref_corr_points","src_corr_points","corr_scores","estimated_transform")
ATOL={"ref_corr_points":1e-6,"src_corr_points":1e-6,"corr_scores":1e-7,"estimated_transform":1e-6}

def fhash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def compare_sentinels(identity: Path, nonzero: Path) -> dict:
    result={}
    with np.load(identity,allow_pickle=False) as a,np.load(nonzero,allow_pickle=False) as b:
        for key in ARRAYS:
            x,y=np.asarray(a[key]),np.asarray(b[key])
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise RuntimeError(f"sentinel shape/finite mismatch: {key}")
            delta=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))) if x.size else 0.0
            result[key]={"max_abs_diff":delta,"atol":ATOL[key],"invariant":delta<=ATOL[key]}
            if delta>ATOL[key]:raise RuntimeError(f"transform sentinel influenced {key}: {delta}")
    return result

def read_meta(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as data:
        return json.loads(str(data["meta_json"].item()))

def write_npz_create_only(path: Path, arrays: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument("--python",type=Path,required=True)
    p.add_argument("--worker",type=Path,required=True);p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--expected-commit",required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--expected-weight-sha256",required=True);p.add_argument("--input",type=Path,required=True)
    p.add_argument("--expected-python-tree-sha256",required=True)
    p.add_argument("--expected-tracked-diff-sha256",required=True)
    p.add_argument("--extension",type=Path,required=True);p.add_argument("--expected-extension-sha256",required=True)
    p.add_argument("--arm",required=True);p.add_argument("--direction",required=True)
    p.add_argument("--neighbor-limits",required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--evidence-dir",type=Path,
                   help="persistent directory for both raw sentinel artifacts")
    p.add_argument("--sampling",choices=("voxel10",),default="voxel10")
    p.add_argument("--device",default="cuda:0");args=p.parse_args()
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    evidence_dir=(args.evidence_dir or
                  (args.output.parent/"sentinel_artifacts")).resolve()
    evidence_dir.mkdir(parents=True,exist_ok=True)
    rows={}
    for name in ("identity","proper_nonzero"):
        final=evidence_dir/f"{args.output.stem}.{name}.npz"
        command=[str(args.python),"-I","-S","-B","-X",
            "pycache_prefix=/proc/v16-b716-fixed4-no-pyc",str(args.worker),
            "--repo",str(args.repo),
            "--expected-commit",args.expected_commit,"--weights",str(args.weights),
            "--expected-weight-sha256",args.expected_weight_sha256,"--input",str(args.input),
            "--expected-python-tree-sha256",args.expected_python_tree_sha256,
            "--expected-tracked-diff-sha256",args.expected_tracked_diff_sha256,
            "--extension",str(args.extension),"--expected-extension-sha256",args.expected_extension_sha256,
            "--arm",args.arm,"--direction",args.direction,"--sentinel",name,
            "--neighbor-limits",args.neighbor_limits,"--sampling",args.sampling,
            "--output",str(final),"--device",args.device]
        child_env=dict(os.environ);child_env.pop("PYTHONPATH",None);child_env.pop("PYTHONHOME",None)
        child_env["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","0")
        child_env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        subprocess.run(command,check=True,env=child_env)
        rows[name]={"sha256":fhash(final),"path":str(final)}
    comparison=compare_sentinels(Path(rows["identity"]["path"]),Path(rows["proper_nonzero"]["path"]))
    identity_meta=read_meta(Path(rows["identity"]["path"]));nonzero_meta=read_meta(Path(rows["proper_nonzero"]["path"]))
    comparable=("input_sha256","repo_commit","weight_sha256","neighbor_limits","sampling",
                "python_tree_sha256","tracked_diff_sha256","extension_sha256",
                "input_voxel_m","coarsest_cap","coarsest_cap_applied","stage_lengths","direction","arm")
    if any(identity_meta.get(key)!=nonzero_meta.get(key) for key in comparable):
        raise RuntimeError("sentinel worker contract metadata mismatch")
    with np.load(rows["identity"]["path"],allow_pickle=False) as d:
        arrays={key:np.asarray(d[key]) for key in ARRAYS}
    meta={"schema":"v13-colorpcr-corr-cache-v2","sentinel_invariant":True,
          "comparison":comparison,"sentinel_artifact_sha256":{k:v["sha256"] for k,v in rows.items()},
          "sentinel_artifact_path":{k:v["path"] for k,v in rows.items()},
          "worker_contract":{key:identity_meta.get(key) for key in comparable},
          "input_sha256":fhash(args.input),"worker_sha256":fhash(args.worker),
          "eligible_downstream_solvers":["pointdsc","pygcransac"],
          "gt_consumed":False,"identity_fallback":False}
    arrays["meta_json"]=np.asarray(json.dumps(meta,sort_keys=True))
    write_npz_create_only(args.output, arrays)
    print(json.dumps({"status":"COLORPCR_SENTINEL_INVARIANT","output":str(args.output),
                      "sha256":fhash(args.output)},sort_keys=True))

if __name__=="__main__":main()
