#!/usr/bin/env python3
"""Authenticate V10/V11 provenance and build disabled V16 ColorPCR inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src", ROOT / "scripts",
              ROOT / "src/inference/sgf_official"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from safety.v16_matched_region_colorpcr import (  # noqa: E402
    PAIR_SCHEMA, SCHEMA, V16ContractError, atomic_json,
    build_hypothesis_artifact, canonical_provenance_binding, load_raw_inseg,
    reject_forbidden_fields, resolve_unique_inseg_path, sha256_file,
    stable_json_sha256, verify_file,
)


DEFAULT_V10 = Path("/home/aidenwu/Documents/sgaligner-sgf-v10-crossgraph-candidates/outputs/v10_crossgraph_gtfree_20260830/v10_manifest.json")
DEFAULT_V11 = Path("/home/aidenwu/Documents/sgaligner-sgf-v11-matched-region-multiobject/outputs/v11_matched_region_pilot_20260830/structural_manifest.json")
RAW_ROOTS = (
    Path("/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/delivery_stage1_20260823/training_dataset/cache"),
    Path("/home/aidenwu/Documents/sgaligner-sgf-official/outputs/official_sgaligner_migration_20260825_235139/supplementary_scan_cache"),
)


def repo_root_from_output_manifest(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        index = path.parts.index("outputs")
    except ValueError as exc:
        raise V16ContractError("source manifest is not under a repository outputs directory") from exc
    return Path(*path.parts[:index])


def closure(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def checked_json(path: Path, expected_sha: str) -> dict:
    verify_file(path, expected_sha)
    value = json.loads(Path(path).read_text())
    reject_forbidden_fields({k: v for k, v in value.items()
                             if k not in ("forbidden_inputs", "forbidden_fields")})
    if sha256_file(path) != expected_sha:
        raise V16ContractError("JSON source changed while reading")
    if "payload_sha256" in value:
        payload = {key: item for key, item in value.items()
                   if key != "payload_sha256"}
        if stable_json_sha256(payload) != value["payload_sha256"]:
            raise V16ContractError("JSON internal payload SHA mismatch")
    return value


def verify_source_hash_closure(source_cache_path: Path,
                               source_cache: dict) -> list[dict[str, str]]:
    provenance = source_cache.get("provenance")
    source_hashes = provenance.get("source_hashes") if isinstance(provenance, dict) else None
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise V16ContractError("V10 source-cache source-hash closure is missing")
    root = repo_root_from_output_manifest(source_cache_path)
    rows = []
    for rel, expected in sorted(source_hashes.items()):
        path = (root / rel).resolve()
        verify_file(path, expected)
        rows.append({"path": rel, "sha256": expected})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10-manifest", type=Path, default=DEFAULT_V10)
    parser.add_argument("--v11-manifest", type=Path, default=DEFAULT_V11)
    parser.add_argument("--pair-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    prereg_path = ROOT / "manifests/v16_matched_region_colorpcr_preregister.json"
    prereg = json.loads(prereg_path.read_text())
    if prereg.get("disabled") is not True or prereg.get("frozen") is not True:
        raise V16ContractError("V16 builder stage is not frozen-disabled")
    frozen = prereg["frozen_inputs"]
    rank_checkpoint = frozen["rank_source_checkpoint"]
    release_checkpoint = frozen["downstream_official_release_checkpoint"]
    if (rank_checkpoint["sha256"] == release_checkpoint["sha256"]
            or prereg.get("checkpoint_domains_match") is not False
            or prereg.get("colorpcr_consumption_allowed") is not False):
        raise V16ContractError("V16 checkpoint-domain mismatch must remain fail-closed")
    v10 = checked_json(args.v10_manifest, frozen["v10_manifest"]["sha256"])
    v11 = checked_json(args.v11_manifest, frozen["v11_structural_manifest"]["sha256"])
    if len(v10.get("pair_caches", [])) != 89 or len(v11.get("pairs", [])) != 89:
        raise V16ContractError("frozen manifests must each contain 89 pairs")
    v10_lines = ["{} {} {} {}".format(x["pair_id"], x["cache_sha256"],
                 x["candidate_fingerprint"], x["structural_sha256"])
                 for x in sorted(v10["pair_caches"], key=lambda x: x["pair_id"])]
    v11_lines = ["{} {} {}".format(x["pair_id"], x["plan_sha256"], x["hypothesis_count"])
                 for x in sorted(v11["pairs"], key=lambda x: x["pair_id"])]
    if closure(v10_lines) != frozen["v10_manifest"]["candidate_cache_closure_sha256"]:
        raise V16ContractError("V10 candidate-cache closure mismatch")
    if closure(v11_lines) != frozen["v11_structural_manifest"]["structural_plan_closure_sha256"]:
        raise V16ContractError("V11 structural-plan closure mismatch")
    for rel, expected in {
        release_checkpoint["path"]: release_checkpoint["sha256"],
        frozen["canonical_builder"]["path"]: frozen["canonical_builder"]["sha256"],
        **frozen["v13_preprocessing_sources"],
    }.items():
        verify_file(ROOT / rel, expected)

    v10_root, v11_root = repo_root_from_output_manifest(args.v10_manifest), repo_root_from_output_manifest(args.v11_manifest)
    by_v10 = {row["pair_id"]: row for row in v10["pair_caches"]}
    by_v11 = {row["pair_id"]: row for row in v11["pairs"]}
    if len(by_v10) != 89 or len(by_v11) != 89:
        raise V16ContractError("pair ids are duplicated")
    rows = []
    for pair_id in args.pair_id:
        if pair_id not in by_v10 or pair_id not in by_v11:
            raise V16ContractError(f"pair missing from frozen manifests: {pair_id}")
        cache_entry, plan_entry = by_v10[pair_id], by_v11[pair_id]
        cache_path = (v10_root / cache_entry["cache_path"]).resolve()
        plan_path = (v11_root / plan_entry["plan_path"]).resolve()
        verify_file(cache_path, cache_entry["cache_sha256"])
        verify_file(plan_path, plan_entry["plan_sha256"])
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        plan = json.loads(plan_path.read_text())
        if (sha256_file(cache_path) != cache_entry["cache_sha256"]
                or sha256_file(plan_path) != plan_entry["plan_sha256"]):
            raise V16ContractError("V10 cache or V11 plan changed while reading")
        if (cache.get("pair_id") != pair_id or plan.get("pair_id") != pair_id
                or cache.get("candidate_fingerprint") != plan.get("candidate_fingerprint")
                or plan.get("v10_cache_sha256") != cache_entry["cache_sha256"]):
            raise V16ContractError("V10/V11 pair provenance does not join")
        if len(plan.get("hypotheses", [])) != int(plan_entry["hypothesis_count"]):
            raise V16ContractError("V11 hypothesis count differs from manifest")
        if (cache.get("checkpoint_id") != rank_checkpoint["checkpoint_id"]
                or cache.get("checkpoint_sha256") != rank_checkpoint["sha256"]):
            raise V16ContractError("V10 rank-source checkpoint differs from preregistration")
        reject_forbidden_fields({"candidate_rank_records": cache["candidate_rank_records"],
                                 "hypotheses": plan["hypotheses"]})
        raw_source_cache_path = cache.get("source_cache_path")
        if (not isinstance(raw_source_cache_path, str)
                or not Path(raw_source_cache_path).is_absolute()):
            raise V16ContractError("V10 source_cache_path must be an absolute frozen path")
        source_cache_path = Path(raw_source_cache_path).resolve()
        source_cache_sha = verify_file(
            source_cache_path, cache.get("source_cache_sha256", ""))
        source_cache = torch.load(
            source_cache_path, map_location="cpu", weights_only=False)
        if sha256_file(source_cache_path) != source_cache_sha:
            raise V16ContractError("V10 source cache changed while reading")
        source_hash_rows = verify_source_hash_closure(
            source_cache_path, source_cache)
        src_scan, ref_scan = pair_id.split("_to_")
        src_path = resolve_unique_inseg_path(src_scan, RAW_ROOTS)
        ref_path = resolve_unique_inseg_path(ref_scan, RAW_ROOTS)
        source = load_raw_inseg(src_path, scan_id=src_scan, side="source")
        reference = load_raw_inseg(ref_path, scan_id=ref_scan, side="reference")
        data, labels = build_canonical_pair(pair_id, with_labels=False)
        if labels:
            raise V16ContractError("canonical builder unexpectedly returned labels")
        canonical_binding = canonical_provenance_binding(
            data, cache, source_cache, plan,
            v10_cache_sha256=cache_entry["cache_sha256"],
            source_cache_sha256=source_cache_sha)
        if (source_cache["provenance"]["source_hashes"].get(
                "scripts/canonical_inputs.py")
                != frozen["canonical_builder"]["sha256"]):
            raise V16ContractError(
                "V10 source cache binds a different canonical builder")
        provenance = {
            "preregister_path": str(prereg_path), "preregister_sha256": sha256_file(prereg_path),
            "v10_manifest_path": str(args.v10_manifest.resolve()), "v10_manifest_sha256": sha256_file(args.v10_manifest),
            "v10_cache_path": str(cache_path), "v10_cache_sha256": sha256_file(cache_path),
            "v10_candidate_fingerprint": cache["candidate_fingerprint"],
            "v10_source_cache_path": str(source_cache_path),
            "v10_source_cache_sha256": source_cache_sha,
            "v10_source_hashes": source_hash_rows,
            "v10_source_hash_closure_sha256": stable_json_sha256(source_hash_rows),
            "v11_manifest_path": str(args.v11_manifest.resolve()), "v11_manifest_sha256": sha256_file(args.v11_manifest),
            "v11_plan_path": str(plan_path), "v11_plan_sha256": sha256_file(plan_path),
            "v11_canonical_input_sha256": plan["canonical_input_sha256"],
            "canonical_provenance_binding": canonical_binding,
            "canonical_builder_path": str((ROOT / frozen["canonical_builder"]["path"]).resolve()),
            "canonical_builder_sha256": frozen["canonical_builder"]["sha256"],
            "rank_source_checkpoint": dict(rank_checkpoint),
            "downstream_official_release_checkpoint": dict(release_checkpoint),
            "checkpoint_domains_match": False,
            "rank_hypotheses_authoritative_for_release": False,
            "builder_source_sha256": sha256_file(ROOT / "src/safety/v16_matched_region_colorpcr.py"),
            "cli_source_sha256": sha256_file(Path(__file__)),
            "source_raw_inseg_sha256": source.file_sha256,
            "reference_raw_inseg_sha256": reference.file_sha256,
            "source_instance_membership_field_sha256": source.field_sha256["raw_instance_membership_key"],
            "reference_instance_membership_field_sha256": reference.field_sha256["raw_instance_membership_key"],
        }
        out_dir = args.output_root / "pairs" / pair_id
        artifacts = [build_hypothesis_artifact(
            pair_id, hypothesis, cache["candidate_rank_records"], data, source,
            reference, out_dir, provenance) for hypothesis in plan["hypotheses"]]
        pair_manifest = {
            "schema": PAIR_SCHEMA, "pair_id": pair_id, "disabled": True,
            "independent_evidence": False, "registration_executed": False,
            "colorpcr_consumption_allowed": False,
            "diagnostic_only_due_checkpoint_domain_mismatch": True,
            "hypothesis_count": len(artifacts), "hypotheses": artifacts,
            "provenance_closure_sha256": stable_json_sha256(provenance),
            "forbidden_inputs": ["semantic/GT/selection/posthoc labels", "GT transforms", "official92", "fallback"],
        }
        pair_manifest["payload_sha256"] = stable_json_sha256(pair_manifest)
        pair_path = out_dir / "pair_manifest.json"
        atomic_json(pair_path, pair_manifest)
        rows.append({"pair_id": pair_id, "hypothesis_count": len(artifacts),
                     "pair_manifest_path": str(pair_path),
                     "pair_manifest_sha256": sha256_file(pair_path)})
    result = {
        "schema": SCHEMA, "disabled": True, "independent_evidence": False,
        "registration_executed": False, "official92_executed": False,
        "colorpcr_consumption_allowed": False,
        "diagnostic_only_due_checkpoint_domain_mismatch": True,
        "rank_source_checkpoint": dict(rank_checkpoint),
        "downstream_official_release_checkpoint": dict(release_checkpoint),
        "pair_count": len(rows), "hypothesis_count": sum(x["hypothesis_count"] for x in rows),
        "pairs": rows, "preregister_path": str(prereg_path),
        "preregister_sha256": sha256_file(prereg_path),
        "forbidden_inputs": ["semantic/GT/selection/posthoc labels", "GT transforms", "official92", "fallback"],
    }
    result["payload_sha256"] = stable_json_sha256(result)
    atomic_json(args.output_root / "builder_manifest.json", result)
    artifact_rows = []
    for path in sorted(p for p in args.output_root.rglob("*")
                       if p.is_file() and p.name != "artifact_manifest.json"):
        artifact_rows.append({
            "path": str(path.relative_to(args.output_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    artifact_manifest = {
        "schema": "v16-matched-region-colorpcr-artifact-manifest-v1",
        "disabled": True,
        "registration_executed": False,
        "file_count": len(artifact_rows),
        "files": artifact_rows,
        "closure_sha256": stable_json_sha256(artifact_rows),
    }
    atomic_json(args.output_root / "artifact_manifest.json", artifact_manifest)
    result["artifact_manifest_path"] = str(args.output_root / "artifact_manifest.json")
    result["artifact_manifest_sha256"] = sha256_file(
        args.output_root / "artifact_manifest.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
