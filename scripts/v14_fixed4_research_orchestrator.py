#!/usr/bin/env python3
"""Bounded CPU-only fixed4 orchestrator for the V14 research shadow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from safety.v13_dual_solver_runtime import atomic_json, sha256_file, stable_json_sha256
from safety.v14_rigid_multihypothesis import (
    aggregate_fixed4_research, load_candidate_contract,
    select_unique_safe_candidate, verify_candidate_set_contract,
)
from scripts.v14_candidate_strict_runner import verify_v14_authorization
from scripts.v14_fixed4_input_builder import (
    verify_conversion_lineage, verify_v13_fixed4_root,
)
from scripts.v14_formal_source_manifest import (
    formal_source_sha256, verify_reviewed_source_authorization,
)


def load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def load_input_manifest(
    path: Path, preregister: dict, *, preregister_path: Path,
    v13_preregister_path: Path, preflight_manifest_path: Path, repo: Path,
) -> dict:
    """Load an exact, fully source-bound fixed4 input manifest."""
    path = Path(path).resolve()
    preregister_path = Path(preregister_path).resolve()
    v13_preregister_path = Path(v13_preregister_path).resolve()
    preflight_manifest_path = Path(preflight_manifest_path).resolve()
    value = load_json(path)
    payload = value.get("payload_sha256")
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    if (value.get("schema") != "v14-fixed4-candidate-inputs-v1"
            or payload != stable_json_sha256(unsigned)):
        raise RuntimeError("V14 fixed4 input manifest closure mismatch")
    expected_bindings = {
        "v14_preregister_path": str(preregister_path),
        "v14_preregister_sha256": sha256_file(preregister_path),
        "v13_preregister_path": str(v13_preregister_path),
        "v13_preregister_sha256": sha256_file(v13_preregister_path),
        "preflight_manifest_path": str(preflight_manifest_path),
        "preflight_manifest_sha256": sha256_file(preflight_manifest_path),
    }
    if any(value.get(key) != expected
           for key, expected in expected_bindings.items()):
        raise RuntimeError("V14 fixed4 input source binding mismatch")
    if (preregister.get("v13_preregister_sha256")
            != expected_bindings["v13_preregister_sha256"]
            or preregister.get("preflight_manifest_sha256")
            != expected_bindings["preflight_manifest_sha256"]):
        raise RuntimeError("V14 preregistration source binding mismatch")
    expected_sources = formal_source_sha256(repo)
    if (value.get("formal_source_sha256") != expected_sources
            or preregister.get("reviewed_formal_source_sha256")
            != expected_sources):
        raise RuntimeError("V14 fixed4 formal source mismatch")
    expected = [(pair_id, arm)
                for pair_id in preregister["fixed_pair_order"]
                for arm in (preregister["primary_arm"],
                            preregister["control_arm"])]
    rows = value.get("rows")
    if (not isinstance(rows, list)
            or [(row.get("pair_id"), row.get("arm")) for row in rows]
            != expected):
        raise RuntimeError("V14 fixed4 input rows are not exact ordered 4x2")
    preflight_value = load_json(preflight_manifest_path)
    preflight_by_pair = {row["pair_id"]: row
                         for row in preflight_value.get("pairs", ())}
    v13_binding = verify_v13_fixed4_root(
        Path(value.get("v13_fixed4_binding", {}).get("root", "")),
        repo=repo, v13_preregister=v13_preregister_path,
        preflight=preflight_manifest_path,
        pairs=list(preregister["fixed_pair_order"]),
        arms=(preregister["primary_arm"], preregister["control_arm"]))
    if value.get("v13_fixed4_binding") != {
            key: item for key, item in v13_binding.items()
            if key != "pair_receipts"}:
        raise RuntimeError("V14 input V13 fixed4 root binding mismatch")
    for row in rows:
        for prefix in ("candidate_set", "prepared_input"):
            artifact = Path(str(row.get(f"{prefix}_path", ""))).resolve()
            if (not artifact.is_file()
                    or sha256_file(artifact) != row.get(f"{prefix}_sha256")):
                raise RuntimeError(f"V14 input artifact closure mismatch: {prefix}")
        verified_set = verify_candidate_set_contract(
            Path(row["candidate_set_path"]))
        candidate_set = verified_set["value"]
        if (candidate_set.get("pair_id") != row["pair_id"]
                or candidate_set.get("arm") != row["arm"]
                or candidate_set.get("candidate_count")
                != len(candidate_set.get("candidates", ()))
                or not 0 <= int(candidate_set.get("candidate_count", -1)) <= 8
                or candidate_set.get("preregister_sha256")
                != expected_bindings["v14_preregister_sha256"]):
            raise RuntimeError("V14 candidate-set identity/bound mismatch")
        prepared_record = preflight_by_pair.get(row["pair_id"])
        if prepared_record is None:
            raise RuntimeError("V14 input pair missing from preflight")
        lineage = {}
        for direction in ("forward", "reverse"):
            record = row.get("direction_lineage", {}).get(direction, {})
            lineage[direction] = verify_conversion_lineage(
                repo=repo, cache_path=Path(record.get("cache_path", "")),
                receipt_path=Path(record.get("conversion_receipt_path", "")),
                prepared_path=Path(row["prepared_input_path"]),
                prepared_record=prepared_record, pair_id=row["pair_id"],
                arm=row["arm"], direction=direction)
            direction_manifest = verified_set["direction_manifests"][direction]
            if (direction_manifest.get("source_cache_path")
                    != lineage[direction]["cache_path"]
                    or direction_manifest.get("source_cache_sha256")
                    != lineage[direction]["cache_sha256"]):
                raise RuntimeError("candidate source differs from V13 lineage")
        if lineage != row.get("direction_lineage"):
            raise RuntimeError("V14 direction lineage receipt mismatch")
        if row.get("v13_pair_receipt") != v13_binding["pair_receipts"][
                (row["pair_id"], row["arm"] )]:
            raise RuntimeError("V14 V13 pair receipt binding mismatch")
    return value


def artifact_manifest(root: Path) -> dict:
    files = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()
                       and value.name not in ("artifact_manifest.json", "closure.json")):
        files[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"schema": "v14-fixed4-artifact-manifest-v1",
            "file_count": len(files), "files": files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--v14-preregister", required=True, type=Path)
    parser.add_argument("--v13-preregister", required=True, type=Path)
    parser.add_argument("--preflight-manifest", required=True, type=Path)
    parser.add_argument("--pointdsc-root", required=True, type=Path)
    parser.add_argument("--pointdsc-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("V14 fixed4 output must be a fresh empty directory")
    output.mkdir(parents=True, exist_ok=True)
    preregister = load_json(args.v14_preregister)
    # Authorization is checked again inside every candidate runner.  The
    # orchestrator cannot bypass the explicit future authorization commit.
    if preregister.get("allow_real_pilot") is not True:
        raise RuntimeError("V14 real CPU pilot is not explicitly authorized")
    verify_reviewed_source_authorization(
        Path(__file__).resolve().parents[1], preregister)
    inputs = load_input_manifest(
        args.input_manifest, preregister,
        preregister_path=args.v14_preregister,
        v13_preregister_path=args.v13_preregister,
        preflight_manifest_path=args.preflight_manifest,
        repo=Path(__file__).resolve().parents[1])
    formal_sources = formal_source_sha256(Path(__file__).resolve().parents[1])
    runner = Path(__file__).resolve().with_name("v14_candidate_strict_runner.py")
    row_outputs, commands, resources = [], [], []
    for input_row in inputs["rows"]:
        candidate_set_path = Path(input_row["candidate_set_path"])
        candidate_set = load_json(candidate_set_path)
        evidence = []
        for index in range(int(candidate_set.get("candidate_count", -1))):
            contract = load_candidate_contract(candidate_set_path, index)
            # Fails until the separately reviewed authorization commit binds
            # the candidate set to that exact preregistration SHA.
            verify_v14_authorization(args.v14_preregister, contract)
            candidate_output = (output / "pairs" / input_row["pair_id"]
                                / input_row["arm"] / f"candidate_{index:02d}")
            argv = [
                args.python, str(runner), "--candidate-set", str(candidate_set_path),
                "--candidate-index", str(index), "--pair-id", input_row["pair_id"],
                "--arm", input_row["arm"], "--prepared-input",
                input_row["prepared_input_path"], "--v13-preregister",
                str(args.v13_preregister), "--v14-preregister",
                str(args.v14_preregister), "--preflight-manifest",
                str(args.preflight_manifest), "--pointdsc-root",
                str(args.pointdsc_root), "--pointdsc-checkpoint",
                str(args.pointdsc_checkpoint), "--output", str(candidate_output),
                "--device", "cpu",
            ]
            started = time.monotonic()
            completed = subprocess.run(argv, cwd=Path(__file__).resolve().parents[1])
            elapsed = time.monotonic() - started
            summary_path = candidate_output / "summary.json"
            if completed.returncode not in (0, 2) or not summary_path.is_file():
                raise RuntimeError("candidate strict runner failed without evidence")
            strict = load_json(summary_path)
            candidate = contract["candidate"]
            expected_strict_identity = {
                "candidate_sha256": candidate["candidate_sha256"],
                "candidate_index": index,
                "candidate_set_path": contract["candidate_set_path"],
                "candidate_set_sha256": contract["candidate_set_sha256"],
                "pair_id": input_row["pair_id"],
                "arm": input_row["arm"],
                "cache_sha256": contract["cache_sha256"],
                "candidate_cache_path": {
                    direction: candidate[f"{direction}_candidate_cache_path"]
                    for direction in ("forward", "reverse")},
                "candidate_receipt_sha256": contract[
                    "candidate_receipt_sha256"],
                "candidate_receipt_path": {
                    direction: candidate[f"{direction}_candidate_receipt_path"]
                    for direction in ("forward", "reverse")},
            }
            if any(strict.get(key) != value
                   for key, value in expected_strict_identity.items()):
                raise RuntimeError("candidate strict summary binding mismatch")
            if (strict.get("candidate_set_sha256")
                    != contract["candidate_set_sha256"]
                    or strict.get("v14_preregister_sha256")
                    != sha256_file(args.v14_preregister)
                    or strict.get("v14_preregister_path")
                    != str(args.v14_preregister.resolve())):
                raise RuntimeError("candidate strict source binding mismatch")
            if strict.get("formal_source_sha256") != formal_sources:
                raise RuntimeError("candidate strict formal source mismatch")
            evidence.append((contract, strict))
            commands.append({"pair_id": input_row["pair_id"],
                             "arm": input_row["arm"], "candidate_index": index,
                             "argv": argv, "returncode": completed.returncode})
            resources.append({"pair_id": input_row["pair_id"],
                              "arm": input_row["arm"], "candidate_index": index,
                              "wall_seconds": elapsed})
        known_bad = input_row["pair_id"] == preregister["known_bad_pair_id"]
        decision = select_unique_safe_candidate(evidence, known_bad=known_bad)
        row_outputs.append({"pair_id": input_row["pair_id"],
                            "arm": input_row["arm"], "decision": decision,
                            "candidate_count": len(evidence)})
    summary = aggregate_fixed4_research(row_outputs, preregister)
    summary.update({
        "input_manifest_sha256": sha256_file(args.input_manifest),
        "v14_preregister_sha256": sha256_file(args.v14_preregister),
        "v13_preregister_sha256": sha256_file(args.v13_preregister),
        "preflight_manifest_sha256": sha256_file(args.preflight_manifest),
        "orchestrator_sha256": sha256_file(Path(__file__)),
        "formal_source_sha256": formal_sources,
    })
    atomic_json(output / "rows.json", {"schema": "v14-fixed4-rows-v1",
                                        "rows": row_outputs})
    atomic_json(output / "commands.json", {"schema": "v14-fixed4-commands-v1",
                                            "commands": commands})
    atomic_json(output / "resource_usage.json",
                {"schema": "v14-fixed4-resource-v1", "candidate_runs": resources,
                 "maximum_candidate_runs": 64,
                 "pointdsc_device": "cpu", "pygcransac_device": "cpu"})
    atomic_json(output / "environment.json",
                {"schema": "v14-fixed4-environment-v1",
                 "python": sys.executable, "python_version": sys.version,
                 "platform": platform.uname()._asdict(),
                 "CUDA_used_by_orchestrator": False})
    atomic_json(output / "summary.json", summary)
    manifest = artifact_manifest(output)
    atomic_json(output / "artifact_manifest.json", manifest)
    closure = {"schema": "v14-fixed4-closure-v1",
               "summary_sha256": sha256_file(output / "summary.json"),
               "artifact_manifest_sha256": sha256_file(
                   output / "artifact_manifest.json")}
    atomic_json(output / "closure.json", closure)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
