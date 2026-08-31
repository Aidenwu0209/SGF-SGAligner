#!/usr/bin/env python3
"""Rebuild the pinned official ColorPCR extension in a clean worktree and seal receipts."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile
from pathlib import Path

def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name);stream.write(data)
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--python",type=Path,required=True);parser.add_argument("--expected-commit",required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    repo=args.repo.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    commit=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],check=True,text=True,
                          capture_output=True).stdout.strip()
    tracked_before=subprocess.run(["git","-C",str(repo),"diff","--binary"],check=True,
                                  capture_output=True).stdout
    if commit!=args.expected_commit or tracked_before:
        raise SystemExit("clean official pin or tracked source mismatch")
    command=[str(args.python),"setup.py","build_ext","--inplace","--force"]
    run=subprocess.run(command,cwd=repo,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=("$ "+" ".join(command)+"\n"+run.stdout).encode();atomic(output/"build.log",log)
    tracked_after=subprocess.run(["git","-C",str(repo),"diff","--binary"],check=True,
                                 capture_output=True).stdout
    extension=repo/"geotransformer/ext.cpython-310-x86_64-linux-gnu.so"
    if run.returncode or tracked_after or not extension.is_file():
        raise SystemExit("clean official build failed or changed tracked source")
    probe=subprocess.run([str(args.python),"-c","import torch; import geotransformer.ext; "
                          "print(torch.__version__); print(geotransformer.ext.__file__)"],cwd=repo,
                         text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    atomic(output/"import_probe.log",probe.stdout.encode())
    if probe.returncode:raise SystemExit("clean extension import failed")
    receipt={"schema":"v13-colorpcr-clean-official-build-v1","repo":str(repo),
        "commit":commit,"tracked_diff_sha256":hashlib.sha256(tracked_after).hexdigest(),
        "build_command":command,"build_returncode":run.returncode,"build_log_sha256":sha(output/"build.log"),
        "extension_path":str(extension),"extension_sha256":sha(extension),
        "import_probe_returncode":probe.returncode,"import_probe_sha256":sha(output/"import_probe.log"),
        "official_tracked_source_modified":False,"nvcc_required_from_path":False}
    receipt_path=output/"clean_build_receipt.json"
    atomic(receipt_path,(json.dumps(receipt,indent=2,sort_keys=True)+"\n").encode())
    artifacts=[output/"build.log",output/"import_probe.log",receipt_path]
    manifest={"schema":"v13-colorpcr-clean-build-artifact-manifest-v1",
              "files":[{"path":path.name,"bytes":path.stat().st_size,"sha256":sha(path)}
                       for path in artifacts]}
    atomic(output/"artifact_manifest.json",(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode())
    print(json.dumps(receipt,sort_keys=True))

if __name__=="__main__":main()
