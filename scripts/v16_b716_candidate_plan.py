#!/usr/bin/env python3
"""Build frozen fixed4/selection89 candidate and matched-region plans.

This stage executes zero new GeoTransformer jobs.  Existing pair-cache entries
are converted to immutable evidence; absent candidate keys remain an explicit,
non-authorized deficit ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src", ROOT / "scripts",
              ROOT / "src/inference/sgf_official"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from adapters.sgf.data_sources import PREDICTION_CACHE  # noqa: E402
from canonical_inputs import build_canonical_pair  # noqa: E402
from inference import GEOT_SNAPSHOT, official_matching  # noqa: E402
from safety.cross_graph_candidates import (  # noqa: E402
    CrossGraphCandidateConfig, candidate_fingerprint, cross_graph_candidates,
)
from safety.matched_region_hypotheses import (  # noqa: E402
    MatchedRegionConfig, generate_matched_region_hypotheses,
)
from safety.v16_b716_candidate_plan import (  # noqa: E402
    B716PlanError, OFFICIAL_RELEASE_SHA256, array_sha256, atomic_json,
    canonical_boundary, file_evidence, freeze_existing_geot,
    load_input_tensors, load_joint_model, safe_pair_metadata, sha256_file,
    stable_json_sha256, validate_pair_metadata, write_deterministic_npz,
)
from safety.v16_matched_region_colorpcr import (  # noqa: E402
    array_sha256 as v16_array_sha256,
    canonical_surface_from_rows, load_raw_inseg, node_object_id,
    resolve_unique_inseg_path, verify_canonical_surface,
)


CACHE_ROOT = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
    "official_sgaligner_v3_pct_parity_baseline_20260827/"
    "final_inference_cache/selection89"
)
RAW_ROOTS = (
    Path("/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
         "delivery_stage1_20260823/training_dataset/cache"),
    Path("/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
         "official_sgaligner_migration_20260825_235139/"
         "supplementary_scan_cache"),
)
FIXED4 = ("09582205_1883", "68bae76c_5364",
          "f38169cf_56fe", "6a36052f_c2b5")
EXPECTED = {
    "candidate_counts": [48, 48, 48, 47],
    "hypothesis_counts": [12, 8, 2, 12],
    "existing_counts": [46, 27, 27, 19],
    "missing_counts": [2, 21, 21, 28],
}
INPUT_FIELDS = (
    "tot_obj_pts", "tot_rel_pose", "tot_bow_vec_object_edge_feats",
    "edges", "obj_ids",
)


def _raw_bindings(data: Mapping[str, Any], pair_id: str) -> tuple[list[dict], list[dict]]:
    src_scan, ref_scan = pair_id.split("_to_")
    src_path = resolve_unique_inseg_path(src_scan, RAW_ROOTS)
    ref_path = resolve_unique_inseg_path(ref_scan, RAW_ROOTS)
    raw = {
        "source": load_raw_inseg(src_path, scan_id=src_scan, side="source"),
        "reference": load_raw_inseg(ref_path, scan_id=ref_scan, side="reference"),
    }
    src_count = int(data["src_count"])
    rows = []
    for node in range(len(data["obj_ids"])):
        side = "source" if node < src_count else "reference"
        oid = node_object_id(data, node, side=side)
        indices, reconstructed = canonical_surface_from_rows(raw[side], oid)
        surface_sha = verify_canonical_surface(data, node, reconstructed)
        rows.append({
            "node_index": node, "side": side, "scan_id": raw[side].scan_id,
            "object_id": oid, "raw_inseg_path": str(raw[side].path),
            "raw_inseg_sha256": raw[side].file_sha256,
            "raw_row_count": len(indices),
            "raw_row_indices_sha256": v16_array_sha256(indices),
            "canonical_registration_surface_sha256": surface_sha,
            "canonical_registration_points": len(reconstructed),
        })
    return rows, [file_evidence(src_path, "raw_inseg_source"),
                  file_evidence(ref_path, "raw_inseg_reference")]


def _input_sha(tensors: Mapping[str, np.ndarray]) -> str:
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(tensors[name]).tobytes() for name in INPUT_FIELDS
    )).hexdigest()


def _source_files(pair_id: str, pair_dir: Path,
                  raw_rows: list[dict]) -> list[dict]:
    src_scan, ref_scan = pair_id.split("_to_")
    rows = [
        file_evidence(pair_dir / "pair_cache.json", "official_pair_metadata"),
        file_evidence(pair_dir / "input_tensors.npz", "official_input_tensors"),
        file_evidence(pair_dir / "embeddings.npz", "official_embeddings"),
        file_evidence(pair_dir / "geot_corrs.npz", "official_geot_cache"),
        *raw_rows,
        file_evidence(ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar",
                      "official_release_checkpoint"),
        file_evidence(Path(GEOT_SNAPSHOT), "official_geotransformer_checkpoint"),
        file_evidence(ROOT / "manifests/v16_b716_candidate_plan_preregister.json",
                      "frozen_preregistration"),
    ]
    for scan, role in ((src_scan, "predicted_relations_source"),
                       (ref_scan, "predicted_relations_reference")):
        path = Path(PREDICTION_CACHE) / f"{scan}.npz"
        if path.is_file():
            rows.append(file_evidence(path, role))
    for rel in (
        "scripts/v16_b716_candidate_plan.py",
        "src/safety/v16_b716_candidate_plan.py",
        "scripts/canonical_inputs.py",
        "src/inference/sgf_official/inference.py",
        "src/safety/cross_graph_candidates.py",
        "src/safety/matched_region_hypotheses.py",
        "src/adapters/sgf/data_sources.py",
        "src/adapters/sgf/graph_adapter.py",
        "src/adapters/sgf/object_adapter.py",
    ):
        rows.append(file_evidence(ROOT / rel, f"source:{rel}"))
    return sorted(rows, key=lambda row: (row["path"], row["role"]))


def _validate_preregister() -> None:
    path = ROOT / "manifests/v16_b716_candidate_plan_preregister.json"
    value = json.loads(path.read_text())
    expected_config = {
        "cross_graph_k": 5, "require_mutual": True,
        "max_candidates_per_pair": 48,
    }
    if (value.get("schema") != "v16-b716-candidate-plan-preregister-v1"
            or value.get("frozen") is not True or value.get("disabled") is not True
            or value.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or value.get("candidate_config") != expected_config
            or value.get("fixed4_short_ids") != list(FIXED4)
            or value.get("expected_candidate_counts")
            != EXPECTED["candidate_counts"]
            or value.get("expected_hypothesis_counts")
            != EXPECTED["hypothesis_counts"]
            or value.get("expected_existing_geot_counts")
            != EXPECTED["existing_counts"]
            or value.get("expected_missing_geot_counts")
            != EXPECTED["missing_counts"]
            or value.get("new_geot_execution_allowed") is not False
            or value.get("official92_allowed") is not False
            or value.get("downstream_colorpcr_authorized") is not False):
        raise B716PlanError("frozen preregistration contract mismatch")


def build_pair(short_id: str, cache_root: Path, output_root: Path) -> dict[str, Any]:
    pair_dir = cache_root / short_id
    pair_json = pair_dir / "pair_cache.json"
    meta = safe_pair_metadata(pair_json)
    validate_pair_metadata(meta)
    pair_id = str(meta["pair_id"])
    checkpoint = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
    if sha256_file(checkpoint) != OFFICIAL_RELEASE_SHA256:
        raise B716PlanError("release checkpoint bytes are not b716")
    tensors = load_input_tensors(pair_dir / "input_tensors.npz")
    if _input_sha(tensors) != meta["cache_key"].get("input_tensor_sha256"):
        raise B716PlanError("pair cache input_tensor_sha256 mismatch")
    data, labels = build_canonical_pair(pair_id, with_labels=False)
    if labels:
        raise B716PlanError("canonical builder unexpectedly returned labels")
    boundary = canonical_boundary(data, tensors)
    surface_bindings, raw_files = _raw_bindings(data, pair_id)
    joint = load_joint_model(pair_dir / "embeddings.npz", boundary["total_objects"])
    official_corrs, rank_list, _distance = official_matching(
        joint, boundary["src_count"])
    config = CrossGraphCandidateConfig()
    candidates = cross_graph_candidates(
        rank_list.tolist(), boundary["src_count"], config)
    candidate_sha = candidate_fingerprint(candidates, config)
    region_config = MatchedRegionConfig()
    hypotheses = generate_matched_region_hypotheses(
        candidates, data["registration_pts"], boundary["src_count"],
        explicit_edges=data["edges_explicit"], config=region_config)
    geot_rows, geot_arrays, geot_counts = freeze_existing_geot(
        candidates, meta["geot_node_pairs"], pair_dir / "geot_corrs.npz", data)
    for row in geot_rows:
        if row["origin"] == "official_pair_cache":
            source_meta = row["source_metadata"]
            if (int(source_meta.get("src_object_id", -1)) != row["object_pair"][0]
                    or int(source_meta.get("ref_object_id", -1)) != row["object_pair"][1]):
                raise B716PlanError("existing GeoT object-id binding mismatch")
    out_dir = output_root / "pairs" / short_id
    geot_out = out_dir / "immutable_geot_entries.npz"
    write_deterministic_npz(geot_out, geot_arrays)
    sources = _source_files(pair_id, pair_dir, raw_files)
    source_closure = stable_json_sha256(sources)
    plan = {
        "schema": "v16-b716-candidate-structural-plan-v1",
        "evidence_class": "GT-free official-release fixed candidate plan",
        "short_id": short_id, "pair_id": pair_id,
        "disabled": True, "new_geot_execution_allowed": False,
        "new_geot_executed": 0, "official92_executed": False,
        "downstream_colorpcr_authorized": False,
        "authorization_blocker": "missing official GeoTransformer entries",
        "domain": {
            "rank_source": "official cache embeddings.npz:joint_model",
            "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "checkpoint_id": "official_release",
            "matched": True,
            "legacy_B_ep20_or_89ed_consumed": False,
        },
        "pair_cache_code_head": meta["code_head"],
        "model_config": meta["model_config"],
        "src_count_evidence": boundary,
        "canonical_surface_bindings": surface_bindings,
        "joint_model": {"shape": list(joint.shape), "dtype": str(joint.dtype),
                        "sha256": array_sha256(joint)},
        "official_matching": {
            "semantics": "global REG_K=3 then same-graph filter",
            "node_corrs": [[int(a), int(b)] for a, b in official_corrs],
            "rank_list": rank_list.tolist(),
            "rank_list_sha256": array_sha256(rank_list),
        },
        "candidate_config": {name: getattr(config, name)
                             for name in config.__dataclass_fields__},
        "candidate_fingerprint": candidate_sha,
        "candidate_count": len(candidates),
        "candidate_rank_records": candidates,
        "matched_region_config": {name: getattr(region_config, name)
                                  for name in region_config.__dataclass_fields__},
        "hypothesis_count": len(hypotheses), "hypotheses": hypotheses,
        "geot_counts": geot_counts, "geot_entries": geot_rows,
        "immutable_geot_npz": {
            "path": str(Path("pairs") / short_id / geot_out.name),
            "bytes": int(geot_out.stat().st_size),
            "sha256": sha256_file(geot_out),
        },
        "source_closure": sources,
        "recursive_source_closure_sha256": source_closure,
        "forbidden_inputs": [
            "GT transforms", "selection/evaluation labels", "combos/node_metrics",
            "posthoc", "official92", "fallbacks",
        ],
    }
    plan["payload_sha256"] = stable_json_sha256(plan)
    plan_path = out_dir / "candidate_structural_plan.json"
    atomic_json(plan_path, plan)
    return {
        "short_id": short_id, "pair_id": pair_id,
        "candidate_count": len(candidates),
        "hypothesis_count": len(hypotheses), **geot_counts,
        "plan_path": str(Path("pairs") / short_id / plan_path.name),
        "plan_bytes": int(plan_path.stat().st_size),
        "plan_sha256": sha256_file(plan_path),
        "geot_npz_path": str(Path("pairs") / short_id / geot_out.name),
        "geot_npz_bytes": int(geot_out.stat().st_size),
        "geot_npz_sha256": sha256_file(geot_out),
        "source_closure_sha256": source_closure,
    }


def _selection_dirs(root: Path) -> list[str]:
    rows = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_dir() and all((path / name).is_file() for name in (
                "pair_cache.json", "input_tensors.npz", "embeddings.npz",
                "geot_corrs.npz")):
            rows.append(path.name)
    if len(rows) != 89:
        raise B716PlanError("selection89 cache directory count is not 89")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("fixed4", "selection89"),
                        default="fixed4")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _validate_preregister()
    if not args.cache_root.is_dir():
        raise B716PlanError("frozen official cache root is absent")
    short_ids = list(FIXED4) if args.scope == "fixed4" else _selection_dirs(args.cache_root)
    rows = []
    for index, short_id in enumerate(short_ids, 1):
        print(f"[{index}/{len(short_ids)}] {short_id}", flush=True)
        rows.append(build_pair(short_id, args.cache_root, args.output_root))
    if args.scope == "fixed4":
        observed = {
            "candidate_counts": [row["candidate_count"] for row in rows],
            "hypothesis_counts": [row["hypothesis_count"] for row in rows],
            "existing_counts": [row["existing_reused"] for row in rows],
            "missing_counts": [row["missing_disabled"] for row in rows],
        }
        if observed != EXPECTED:
            raise B716PlanError(f"fixed4 frozen count contract changed: {observed}")
    artifacts = sorted([
        {"path": row["plan_path"], "bytes": row["plan_bytes"],
         "sha256": row["plan_sha256"], "role": "candidate_structural_plan"}
        for row in rows
    ] + [
        {"path": row["geot_npz_path"], "bytes": row["geot_npz_bytes"],
         "sha256": row["geot_npz_sha256"], "role": "immutable_geot_entries"}
        for row in rows
    ], key=lambda row: row["path"])
    manifest = {
        "schema": "v16-b716-candidate-plan-manifest-v1",
        "scope": args.scope, "frozen": args.scope == "fixed4",
        "disabled": True, "new_geot_execution_allowed": False,
        "new_geot_executed": 0, "official92_executed": False,
        "official_release_domain_matched": True,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "legacy_B_ep20_or_89ed_consumed": False,
        "pair_count": len(rows),
        "candidate_count": sum(row["candidate_count"] for row in rows),
        "hypothesis_count": sum(row["hypothesis_count"] for row in rows),
        "geot_existing_reused": sum(row["existing_reused"] for row in rows),
        "geot_existing_ok": sum(row["existing_ok"] for row in rows),
        "geot_existing_failed": sum(row["existing_failed"] for row in rows),
        "geot_missing_disabled": sum(row["missing_disabled"] for row in rows),
        "downstream_colorpcr_authorized": False,
        "authorization_blockers": [
            "72 candidate GeoTransformer entries remain missing/disabled",
            "this builder stage has no execution authorization",
        ] if args.scope == "fixed4" else [
            "selection89 is development-only and not an authorization stage"],
        "pairs": rows, "artifact_closure": artifacts,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifacts),
        "recursive_source_closure_sha256": stable_json_sha256(sorted(
            {row["source_closure_sha256"] for row in rows})),
        "forbidden_inputs": [
            "GT transforms", "selection/evaluation labels", "combos/node_metrics",
            "posthoc", "official92", "fallbacks",
        ],
    }
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    path = args.output_root / "fixed4_manifest.json"
    if args.scope == "selection89":
        path = args.output_root / "selection89_manifest.json"
    atomic_json(path, manifest)
    print(json.dumps({
        "manifest": str(path), "manifest_sha256": sha256_file(path),
        "candidate_count": manifest["candidate_count"],
        "hypothesis_count": manifest["hypothesis_count"],
        "geot_existing_reused": manifest["geot_existing_reused"],
        "geot_missing_disabled": manifest["geot_missing_disabled"],
        "downstream_colorpcr_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
