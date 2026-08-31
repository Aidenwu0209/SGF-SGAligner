#!/usr/bin/env python3
"""Offline, pre-registered V11.3 whole-scene GeoTransformer pilot.

The expensive official GeoTransformer outputs already exist in immutable V11
diagnostic caches.  This script validates them against freshly rebuilt
canonical surfaces and runs only the frozen RANSAC/ICP/Rule-B/q4 pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import tempfile
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src", ROOT / "scripts",
             ROOT / "src/inference/sgf_official"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))
os.environ["SGALIGNER_CODE_ROOT"] = str(ROOT)

from canonical_inputs import build_canonical_pair  # noqa: E402
from adapters.sgf.data_sources import _source_inseg_cloud  # noqa: E402
from safety import decision_features as dfx  # noqa: E402
from safety.matched_region_hypotheses import full_scene_union  # noqa: E402
from safety.v8_stage_order_consensus import (  # noqa: E402
    V8Config, evaluate_stage_order, transform_distance,
)
from v11_matched_region_runner import (  # noqa: E402
    CACHE_SCHEMA, EXPECTED_RULE_B, PILOT_POSITIONS,
)
from v3b_cache_runner import ransac_from_pooled  # noqa: E402
from v7_registration_pilot import (  # noqa: E402
    rule_b_features, segment_icp_with_trace,
)
from v9_nodepair_multihypothesis import (  # noqa: E402
    KNOWN_BAD, array_sha256, atomic_create_json, jsonable, sha256_file,
    stable_json_hash,
)

SCHEMA = "v11.3-whole-scene-offline-pilot-v1"
WORKER_SCHEMA = "v11.3-whole-scene-worker-v1"
PROTOCOL = ROOT / "docs/V11_3_WHOLE_SCENE_GEOT_PILOT_PROTOCOL.md"
V11_PROTOCOL = ROOT / "docs/V11_MATCHED_REGION_MULTI_OBJECT_PROTOCOL.md"
OFFICIAL_SGA_CKPT = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
OFFICIAL_GEOT_CKPT = ROOT / "checkpoints/geotransformer/geotransformer-3dmatch.pth.tar"
Q4 = V8Config(repeats=5, quorum=4,
              max_rotation_deg=5.0, max_translation_m=0.10)
FORBIDDEN = ("GT transforms", "selection labels", "calibration labels",
             "posthoc outcomes", "official92")


class EvidenceError(RuntimeError):
    pass


def fhash(path: Path) -> str:
    return sha256_file(path)


def arrhash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def payload_hash(value: Mapping[str, Any]) -> str:
    return stable_json_hash({key: item for key, item in value.items()
                             if key not in ("payload_sha256", "_file_sha256")})


def write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value)
    data["payload_sha256"] = payload_hash(data)
    atomic_create_json(path, data)
    data["_file_sha256"] = fhash(path)
    return data


def pilot_ids(v11_manifest: Mapping[str, Any]) -> list[str]:
    rows = v11_manifest.get("pairs")
    if not isinstance(rows, list) or len(rows) != 4:
        raise EvidenceError("V11 pilot manifest must contain fixed four pairs")
    expected = list(v11_manifest.get("pilot_pair_ids", []))
    observed = [str(row.get("pair_id")) for row in rows]
    if expected != observed or KNOWN_BAD not in observed:
        raise EvidenceError("V11 pilot ids/order or known-bad mismatch")
    if v11_manifest.get("pilot_positions") != list(PILOT_POSITIONS):
        raise EvidenceError("V11 pilot positions mismatch")
    return observed


def load_cache(cache_path: Path, row: Mapping[str, Any], pair_id: str,
               expected_checkpoint: str | None,
               historical_protocol_sha: str) -> tuple[dict[str, Any], str]:
    before = fhash(cache_path)
    if before != row["whole_scene_diagnostic"]["artifact_sha256"]:
        raise EvidenceError("diagnostic cache file SHA mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("pair_id") != pair_id:
        raise EvidenceError("diagnostic cache schema/pair mismatch")
    if cache.get("arm") != "whole_scene_diagnostic_only" \
            or cache.get("selector_eligible") is not False:
        raise EvidenceError("historical diagnostic cache arm mismatch")
    if cache.get("payload_sha256") != payload_hash(cache):
        raise EvidenceError("diagnostic cache payload hash mismatch")
    if cache.get("protocol_sha256") != historical_protocol_sha:
        raise EvidenceError("historical diagnostic protocol SHA mismatch")
    if tuple(cache.get("forbidden_inputs", ())) != (
            "selection labels", "GT transforms", "posthoc", "official92"):
        raise EvidenceError("historical cache forbidden-input contract mismatch")
    checkpoint = str(cache.get("checkpoint_sha256", ""))
    if not checkpoint or (expected_checkpoint and checkpoint != expected_checkpoint):
        raise EvidenceError("checkpoint mismatch across diagnostic caches")
    if fhash(cache_path) != before:
        raise EvidenceError("diagnostic cache changed while read")
    return cache, checkpoint


def canonical_surfaces(pair_id: str, cache: Mapping[str, Any]):
    data, labels = build_canonical_pair(pair_id, with_labels=False)
    if labels:
        raise EvidenceError("label-free canonical builder returned labels")
    if data.get("sampling_mode") != "official_mt19937" \
            or int(data.get("scan_seed", -1)) != 0:
        raise EvidenceError("canonical sampling contract mismatch")
    source, reference = full_scene_union(
        data["registration_pts"], int(data["src_count"]))
    expected_counts = cache.get("surface_points", {})
    expected_hashes = cache.get("surface_sha256", {})
    if (len(source) != expected_counts.get("source")
            or len(reference) != expected_counts.get("reference")
            or arrhash(source) != expected_hashes.get("source")
            or arrhash(reference) != expected_hashes.get("reference")):
        raise EvidenceError("canonical whole-scene surfaces differ from cache")
    src_indices = range(int(data["src_count"]))
    ref_indices = range(int(data["src_count"]), len(data["registration_pts"]))
    src_centres = np.asarray([
        np.asarray(data["registration_pts"][index]).mean(axis=0)
        for index in src_indices])
    ref_centres = np.asarray([
        np.asarray(data["registration_pts"][index]).mean(axis=0)
        for index in ref_indices])
    return (np.asarray(source), np.asarray(reference), src_centres,
            ref_centres, data)


def atomic_npz(path: Path, **arrays: Any) -> str:
    """Create one immutable, self-contained downstream shadow input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                               dir=path.parent)
    os.close(fd)
    temporary = Path(raw)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EvidenceError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return fhash(path)


def write_shadow_input(pair_id: str, cache: Mapping[str, Any], out: Path,
                       source: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Expose points/correspondences without inventing unavailable RGB."""
    f_src, f_ref, f_score = geot_arrays(cache, "forward")
    r_src, r_ref, r_score = geot_arrays(cache, "reverse")
    path = out / "shadow_inputs" / f"{pair_id}.npz"
    digest = atomic_npz(
        path,
        source_points=np.asarray(source, dtype=np.float32),
        reference_points=np.asarray(reference, dtype=np.float32),
        forward_src_corr=np.asarray(f_src, dtype=np.float32),
        forward_ref_corr=np.asarray(f_ref, dtype=np.float32),
        forward_scores=np.asarray(f_score, dtype=np.float32),
        reverse_src_corr=np.asarray(r_src, dtype=np.float32),
        reverse_ref_corr=np.asarray(r_ref, dtype=np.float32),
        reverse_scores=np.asarray(r_score, dtype=np.float32),
    )
    raw_sources = []
    for scan_id in pair_id.split("_to_"):
        raw_path = _source_inseg_cloud(scan_id)
        with np.load(raw_path) as raw:
            required = {"xyz", "labels", "colors"}
            if not required.issubset(raw.files):
                raise EvidenceError("raw InSeg color provenance keys missing")
            xyz, labels, colors = raw["xyz"], raw["labels"], raw["colors"]
            valid = bool(
                xyz.ndim == 2 and xyz.shape[1] == 3
                and labels.shape == (len(xyz),)
                and colors.shape == (len(xyz), 4)
                and colors.dtype == np.uint8)
            if not valid:
                raise EvidenceError("raw InSeg color provenance shape mismatch")
        raw_sources.append({
            "scan_id": scan_id, "path": str(raw_path),
            "sha256": fhash(raw_path), "rows": int(len(xyz)),
            "xyz_shape": list(xyz.shape), "colors_shape": list(colors.shape),
            "colors_dtype": str(colors.dtype),
        })
    return {
        "schema": "v11.3-shadow-solver-input-v1",
        "path": str(path.relative_to(out)), "sha256": digest,
        "point_contract": "whole_scene_balanced_object_union",
        "source_point_count": int(len(source)),
        "reference_point_count": int(len(reference)),
        "color_available": False,
        "color_unavailable_reason": (
            "canonical SGF registration_pts contract contains XYZ only; "
            "no RGB is fabricated"),
        "rgb_available_in_raw_source": True,
        "raw_color_sources": raw_sources,
        "raw_color_usage": "provenance_only_not_consumed_by_current_pilot",
        "correspondence_arms": ["forward", "reverse"],
        "eligible_for_current_pilot": False,
        "intended_shadow_solvers": ["ColorPCR_if_RGB_is_separately_sealed",
                                     "PointDSC", "pyGCRANSAC"],
    }


def geot_arrays(cache: Mapping[str, Any], direction: str):
    item = cache.get(direction, {})
    if item.get("status") != "ok":
        raise EvidenceError(f"cached GeoTransformer {direction} not ok")
    src = np.asarray(item.get("src_corr"), dtype=np.float64)
    ref = np.asarray(item.get("ref_corr"), dtype=np.float64)
    scores = np.asarray(item.get("scores"), dtype=np.float64)
    if (src.ndim != 2 or src.shape != ref.shape or src.shape[1:] != (3,)
            or scores.shape != (len(src),) or len(src) == 0
            or not np.isfinite(src).all() or not np.isfinite(ref).all()
            or not np.isfinite(scores).all()):
        raise EvidenceError("malformed cached GeoTransformer arrays")
    digest = hashlib.sha256(
        np.asarray(src, dtype=np.float32).tobytes()
        + np.asarray(ref, dtype=np.float32).tobytes()
        + np.asarray(scores, dtype=np.float32).tobytes()).hexdigest()
    if digest != item.get("correspondence_sha256"):
        raise EvidenceError("GeoTransformer correspondence SHA mismatch")
    keep = np.argsort(-scores, kind="stable")[:1000]
    return src[keep], ref[keep], scores[keep]


def permutation(count: int, pair_id: str, direction: str, replicate: int,
                protocol_sha: str):
    contract = {"schema": WORKER_SCHEMA, "pair_id": pair_id,
                "direction": direction, "replicate": int(replicate),
                "protocol_sha256": protocol_sha}
    signature = stable_json_hash(contract)
    order = np.random.default_rng(int(signature[:16], 16)).permutation(count)
    return order, stable_json_hash({
        "contract_sha256": signature,
        "permutation_sha256": hashlib.sha256(
            np.asarray(order, dtype=np.int64).tobytes()).hexdigest(),
        "count": int(count),
    })


def failed_worker(pair_id: str, direction: str, replicate: int,
                  signature: str, reason: str) -> dict[str, Any]:
    identity = np.eye(4)
    value = {
        "schema": WORKER_SCHEMA, "pair_id": pair_id,
        "direction": direction, "replicate": replicate, "status": "failed",
        "permutation_provenance_sha256": signature,
        "raw_transform": identity, "final_transform": identity,
        "rule_b_features": {}, "rule_b_accepted": False,
        "decision": {"rejection_reasons": [reason]}, "icp": {"trace": []},
    }
    value["evidence_sha256"] = stable_json_hash(value)
    return value


def run_worker(pair_id: str, cache: Mapping[str, Any], direction: str,
               replicate: int, protocol_sha: str, source: np.ndarray,
               reference: np.ndarray, src_centres: np.ndarray,
               ref_centres: np.ndarray) -> dict[str, Any]:
    try:
        src, ref, _scores = geot_arrays(cache, direction)
        order, signature = permutation(
            len(src), pair_id, direction, replicate, protocol_sha)
        src, ref = src[order], ref[order]
        raw, inliers = ransac_from_pooled(src, ref)
        if direction == "forward":
            from_surface, to_surface, centres = source, reference, src_centres
        else:
            from_surface, to_surface, centres = reference, source, ref_centres
        icp = segment_icp_with_trace(
            from_surface, to_surface, raw,
            seed=42 if direction == "forward" else 43)
        features, decision = rule_b_features(
            from_surface, to_surface, centres, raw, inliers, len(src),
            1, 0, icp, direction=direction)
        value = {
            "schema": WORKER_SCHEMA, "pair_id": pair_id,
            "direction": direction, "replicate": replicate, "status": "ok",
            "permutation_provenance_sha256": signature,
            "correspondence_count": int(len(src)),
            "solver": "official_pygcransac_composition",
            "correspondence_sha256": hashlib.sha256(
                np.asarray(src, dtype=np.float32).tobytes()
                + np.asarray(ref, dtype=np.float32).tobytes()).hexdigest(),
            "raw_transform": raw, "raw_transform_sha256": array_sha256(raw),
            "final_transform": icp["transform"],
            "final_transform_sha256": array_sha256(icp["transform"]),
            "ransac": {"inliers_10cm": int(inliers)}, "icp": icp,
            "rule_b_features": features, "decision": decision,
            "rule_b_accepted": bool(decision["usable_for_reconstruction"]),
        }
        value["evidence_sha256"] = stable_json_hash(value)
        return value
    except Exception as exc:
        signature = stable_json_hash({
            "pair_id": pair_id, "direction": direction,
            "replicate": replicate, "protocol_sha256": protocol_sha})
        return failed_worker(pair_id, direction, replicate, signature,
                             f"{type(exc).__name__}:{exc}")


def pair_gate(pair_id: str, cache: Mapping[str, Any], out: Path,
              protocol_sha: str) -> dict[str, Any]:
    source, reference, src_centres, ref_centres, data = canonical_surfaces(
        pair_id, cache)
    shadow = write_shadow_input(pair_id, cache, out, source, reference)
    workers = []
    for direction in ("forward", "reverse"):
        for rep in range(5):
            worker = run_worker(
                pair_id, cache, direction, rep, protocol_sha, source,
                reference, src_centres, ref_centres)
            worker_path = out / "workers" / pair_id / f"{direction}_{rep}.json"
            workers.append(write_json(worker_path, jsonable(worker)))
    gate = evaluate_stage_order(
        workers, Q4, dfx.evaluate_rule_b, require_fixed_trace=True)
    known_bad = pair_id == KNOWN_BAD
    eligible = bool(gate["fresh_v8_qualified"] and not known_bad)
    result = {
        "schema": SCHEMA, "pair_id": pair_id, "protocol_sha256": protocol_sha,
        "known_bad": known_bad, "known_bad_veto": bool(known_bad),
        "eligible": eligible,
        "gate": gate,
        "worker_failures": sum(row["status"] != "ok" for row in workers),
        "canonical": {
            "src_count": int(data["src_count"]),
            "source_points": len(source), "reference_points": len(reference),
            "source_sha256": arrhash(source),
            "reference_sha256": arrhash(reference),
        },
        "shadow_solver_interface": shadow,
        "worker_evidence_sha256": [row["evidence_sha256"] for row in workers],
        "forbidden_inputs": list(FORBIDDEN),
    }
    return write_json(out / "gates" / f"{pair_id}.json", jsonable(result))


def artifact_manifest(out: Path, exclusions: set[Path]) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        if path in exclusions:
            continue
        rows.append({"path": str(path.relative_to(out)),
                     "bytes": path.stat().st_size, "sha256": fhash(path)})
    return {"schema": SCHEMA, "artifact_count": len(rows), "artifacts": rows}


def write_run_metadata(out: Path, args: argparse.Namespace,
                       protocol_sha: str) -> None:
    command = {
        "schema": SCHEMA, "stage": "command",
        "argv": list(sys.argv), "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "protocol_sha256": protocol_sha,
    }
    write_json(out / "command.json", command)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_head = "unavailable"
    environment = {
        "schema": SCHEMA, "stage": "environment",
        "platform": platform.platform(), "python": sys.version,
        "numpy": np.__version__, "torch": torch.__version__,
        "git_head": git_head,
        "env": {key: os.environ.get(key) for key in (
            "PYTHONPATH", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        "run_id": args.run_id,
    }
    write_json(out / "environment.json", environment)


def run(args: argparse.Namespace) -> int:
    started = time.time()
    protocol_sha = fhash(PROTOCOL)
    historical_protocol_sha = fhash(V11_PROTOCOL)
    official_sga_sha = fhash(OFFICIAL_SGA_CKPT)
    official_geot_sha = fhash(OFFICIAL_GEOT_CKPT)
    write_run_metadata(args.out_root, args, protocol_sha)
    if dict(dfx.RULE_THRESHOLDS) != EXPECTED_RULE_B:
        raise EvidenceError("Rule-B thresholds differ from pre-registration")
    manifest_path = args.cache_root / "pilot_manifest.json"
    manifest_sha = fhash(manifest_path)
    v11 = json.loads(manifest_path.read_text())
    ids = pilot_ids(v11)
    by_id = {row["pair_id"]: row for row in v11["pairs"]}
    rows, cache_inputs, checkpoint = [], [], None
    for index, pair_id in enumerate(ids, 1):
        cache_path = args.cache_root / "diagnostic" / f"{pair_id}.pt"
        cache, checkpoint = load_cache(
            cache_path, by_id[pair_id], pair_id, checkpoint,
            historical_protocol_sha)
        cache_inputs.append({
            "pair_id": pair_id,
            "path": str(cache_path),
            "file_sha256": fhash(cache_path),
            "payload_sha256": cache["payload_sha256"],
            "surface_sha256": cache["surface_sha256"],
            "forward_correspondence_sha256": cache["forward"].get(
                "correspondence_sha256"),
            "reverse_correspondence_sha256": cache["reverse"].get(
                "correspondence_sha256"),
        })
        row = pair_gate(pair_id, cache, args.out_root, protocol_sha)
        rows.append(row)
        print(f"[{index}/{len(ids)}] {pair_id} eligible={row['eligible']} "
              f"fresh={row['gate']['fresh_v8_qualified']}", flush=True)
    write_json(args.out_root / "input_manifest.json", {
        "schema": SCHEMA, "stage": "input_manifest",
        "protocol_sha256": protocol_sha,
        "historical_v11_manifest_sha256": manifest_sha,
        "official_sgaligner_checkpoint_sha256": official_sga_sha,
        "official_geotransformer_checkpoint_sha256": official_geot_sha,
        "historical_matcher_checkpoint_sha256": checkpoint,
        "caches": cache_inputs,
    })
    non_bad = [row for row in rows if not row["known_bad"]]
    known = [row for row in rows if row["known_bad"]]
    pilot_pass = bool(
        len(non_bad) == 3 and all(row["eligible"] for row in non_bad)
        and len(known) == 1 and not known[0]["eligible"]
        and all(row["worker_failures"] == 0 for row in rows))
    summary = {
        "schema": SCHEMA, "stage": "primary" if args.run_id == "primary"
        else "independent_replay", "status": "PILOT_PASS" if pilot_pass
        else "PILOT_FAILED", "pilot_pass": pilot_pass,
        "protocol_path": str(PROTOCOL), "protocol_sha256": protocol_sha,
        "source_sha256": fhash(Path(__file__)),
        "historical_v11_manifest": str(manifest_path),
        "historical_v11_manifest_sha256": manifest_sha,
        "historical_v11_protocol_sha256": historical_protocol_sha,
        "historical_matcher_checkpoint_sha256": checkpoint,
        "official_sgaligner_checkpoint_sha256": official_sga_sha,
        "official_geotransformer_checkpoint_sha256": official_geot_sha,
        "checkpoint_usage": {
            "official_sgaligner": "provenance_only_not_consumed_by_offline_arm",
            "historical_matcher": "cache_provenance_only_not_consumed_by_offline_arm",
            "official_geotransformer": "correspondence_generator_validated_on_disk",
        },
        "source_contract_sha256": {
            "canonical_inputs": fhash(ROOT / "scripts/canonical_inputs.py"),
            "ransac": fhash(ROOT / "scripts/v3b_cache_runner.py"),
            "icp_rule_b": fhash(ROOT / "scripts/v7_registration_pilot.py"),
            "q4": fhash(ROOT / "src/safety/v8_stage_order_consensus.py"),
        },
        "pair_ids": ids,
        "non_known_bad_eligible": sum(row["eligible"] for row in non_bad),
        "known_bad_veto": bool(len(known) == 1 and not known[0]["eligible"]),
        "worker_failures": sum(row["worker_failures"] for row in rows),
        "wall_seconds": time.time() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "pairs": [{"pair_id": row["pair_id"], "known_bad": row["known_bad"],
                   "eligible": row["eligible"],
                   "fresh_v8_qualified": row["gate"]["fresh_v8_qualified"],
                   "gate_sha256": row["_file_sha256"]} for row in rows],
        "forbidden_inputs": list(FORBIDDEN),
        "authorization": {"selection89": pilot_pass,
                          "calibration90": False, "fixed12": False,
                          "official92": False, "reconstruction": False,
                          "checkpoint_promotion": False},
    }
    summary_path = args.out_root / "pilot_summary.json"
    saved = write_json(summary_path, summary)
    manifest_out = args.out_root / "artifact_manifest.json"
    write_json(manifest_out, artifact_manifest(
        args.out_root, {manifest_out}))
    print(json.dumps(jsonable({key: value for key, value in saved.items()
                               if key != "pairs"}), indent=2))
    return 0 if pilot_pass else 2


def compare(args: argparse.Namespace) -> int:
    first = json.loads((args.primary / "pilot_summary.json").read_text())
    second = json.loads((args.replay / "pilot_summary.json").read_text())
    for label, root, summary in (("primary", args.primary, first),
                                 ("replay", args.replay, second)):
        if summary.get("payload_sha256") != payload_hash(summary):
            raise EvidenceError(f"{label} summary payload hash mismatch")
        receipt = json.loads((root / "verification_receipt.json").read_text())
        if receipt.get("verification_pass") is not True:
            raise EvidenceError(f"{label} verification receipt is not passing")
    if (first.get("pair_ids") != second.get("pair_ids")
            or first.get("protocol_sha256") != second.get("protocol_sha256")
            or first.get("source_sha256") != second.get("source_sha256")
            or first.get("official_geotransformer_checkpoint_sha256")
            != second.get("official_geotransformer_checkpoint_sha256")):
        raise EvidenceError("primary/replay frozen contract mismatch")
    gates_a = {row["pair_id"]: json.loads(
        (args.primary / "gates" / f"{row['pair_id']}.json").read_text())
        for row in first["pairs"]}
    gates_b = {row["pair_id"]: json.loads(
        (args.replay / "gates" / f"{row['pair_id']}.json").read_text())
        for row in second["pairs"]}
    comparisons, all_ok = [], True
    for pair_id in first["pair_ids"]:
        a, b = gates_a[pair_id], gates_b[pair_id]
        ta = a["gate"].get("selected_observed_forward_medoid")
        tb = b["gate"].get("selected_observed_forward_medoid")
        if ta and tb:
            dr, dt = transform_distance(ta["final_transform"],
                                        tb["final_transform"])
        else:
            dr = dt = None
        verdict_same = (a["eligible"] == b["eligible"] and
                        a["gate"]["fresh_v8_qualified"] ==
                        b["gate"]["fresh_v8_qualified"])
        within = bool((dr is None and dt is None) or
                      (dr <= Q4.max_rotation_deg
                       and dt <= Q4.max_translation_m))
        ok = verdict_same and within
        all_ok &= ok
        comparisons.append({"pair_id": pair_id, "verdict_same": verdict_same,
                            "rotation_deg": dr, "translation_m": dt,
                            "within_frozen_bound": within, "ok": ok})
    result = {"schema": SCHEMA, "stage": "replay_comparison",
              "replay_pass": bool(all_ok), "pairs": comparisons,
              "primary_summary_sha256": fhash(args.primary / "pilot_summary.json"),
              "replay_summary_sha256": fhash(args.replay / "pilot_summary.json")}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if all_ok else 2


def verify(args: argparse.Namespace) -> int:
    manifest_path = args.root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("payload_sha256") != payload_hash(manifest):
        raise EvidenceError("artifact manifest payload hash mismatch")
    mismatches = []
    for row in manifest.get("artifacts", []):
        path = args.root / row["path"]
        if (not path.is_file() or path.stat().st_size != row["bytes"]
                or fhash(path) != row["sha256"]):
            mismatches.append(row["path"])
    receipt = {
        "schema": SCHEMA, "stage": "independent_verification",
        "artifact_manifest_sha256": fhash(manifest_path),
        "declared_artifact_count": int(manifest.get("artifact_count", -1)),
        "verified_artifact_count": len(manifest.get("artifacts", [])),
        "mismatches": mismatches,
        "verification_pass": bool(
            not mismatches and manifest.get("artifact_count")
            == len(manifest.get("artifacts", []))),
    }
    saved = write_json(args.root / "verification_receipt.json", receipt)
    print(json.dumps(saved, indent=2))
    return 0 if receipt["verification_pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--cache-root", type=Path, required=True)
    runner.add_argument("--out-root", type=Path, required=True)
    runner.add_argument("--run-id", choices=("primary", "replay"), required=True)
    comparator = sub.add_parser("compare")
    comparator.add_argument("--primary", type=Path, required=True)
    comparator.add_argument("--replay", type=Path, required=True)
    comparator.add_argument("--output", type=Path, required=True)
    verifier = sub.add_parser("verify")
    verifier.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run(args)
    if args.command == "verify":
        return verify(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
