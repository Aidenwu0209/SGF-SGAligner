#!/usr/bin/env python3
"""CPU-only V13 fixed4 builder; never imports ColorPCR/PointDSC."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
from adapters.sgf.data_sources import _source_inseg_cloud  # noqa: E402
from safety.v13_colorpcr_pointdsc_shadow import (  # noqa: E402
    ARMS, FORBIDDEN_INPUTS, PILOT_POSITIONS, Q4, SCHEMA, SOLVERS,
    build_color_pair, sha256_file, stable_json_sha256, worker_plan)

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v113-sealed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sealed, output = args.v113_sealed_root.resolve(), args.output.resolve()
    summary_path, input_path = sealed / "pilot_summary.json", sealed / "input_manifest.json"
    summary = json.loads(summary_path.read_text())
    pair_ids = list(summary["pair_ids"])
    if len(pair_ids) != 4:
        raise SystemExit("sealed V11.3 fixed4 mismatch")
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair_id in pair_ids:
        source, reference = pair_id.split("_to_")
        pairs.append(build_color_pair(pair_id, _source_inseg_cloud(source),
            _source_inseg_cloud(reference), sealed / "shadow_inputs" / f"{pair_id}.npz",
            output / "prepared" / f"{pair_id}.npz"))
    protocol = ROOT / "docs/V13_COLORPCR_POINTDSC_SHADOW_PROTOCOL.md"
    prereg = ROOT / "manifests/v13_colorpcr_pointdsc_fixed4_preregister.json"
    result = {"schema": SCHEMA, "stage": "cpu_preflight_resource_safe_fixed4_not_run",
        "pair_ids": pair_ids, "pilot_positions": list(PILOT_POSITIONS),
        "arms": list(ARMS), "solvers": list(SOLVERS), "repeats": Q4.repeats,
        "quorum": Q4.quorum, "worker_count": len(worker_plan(pair_ids)), "pairs": pairs,
        "identical_arm_pairs": sum(bool(row["arms_identical_after_filter_and_voxel"])
                                   for row in pairs),
        "forbidden_inputs": list(FORBIDDEN_INPUTS),
        "v113_input_manifest_sha256": sha256_file(input_path),
        "v113_summary_sha256": sha256_file(summary_path),
        "protocol_sha256": sha256_file(protocol), "preregister_sha256": sha256_file(prereg),
        "dependency_status": "PHASE1_INPUT_AND_RUNTIME_CONTRACT_READY_FIXED4_NOT_RUN",
        "gpu_authorized": False,
        "gpu_authorized_for": "isolated_fixed4_shadow_only_after_manifest_and_sentinel_gate"}
    result["payload_sha256"] = stable_json_sha256(result)
    manifest_path = output / "preflight_manifest.json"
    atomic_json(manifest_path, result)
    material = [manifest_path, *sorted((output / "prepared").glob("*.npz"))]
    closure = {"schema": "v13-colorpcr-preflight-artifact-manifest-v1",
        "files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                   "sha256": sha256_file(path)} for path in material],
        "protocol_sha256": result["protocol_sha256"],
        "preregister_sha256": result["preregister_sha256"],
        "payload_sha256": result["payload_sha256"]}
    atomic_json(output / "artifact_manifest.json", closure)
    print(json.dumps({"status": result["dependency_status"], "pairs": len(pairs),
                      "workers_planned": result["worker_count"],
                      "manifest": str(manifest_path),
                      "artifact_manifest": str(output / "artifact_manifest.json")}, sort_keys=True))

if __name__ == "__main__":
    main()
