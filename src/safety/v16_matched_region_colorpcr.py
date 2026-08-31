"""Fail-closed V16 matched-region ColorPCR input builder.

This module authenticates frozen V10 rank records and V11 structural plans,
maps their node indices to same-scan raw InSeg instance row sets, and prepares
the frozen V13 filter-before-voxel 0.10 m inputs.  The official V13 worker owns
the later multi-level grid subsampling and coarsest-level cap512.  This module
never executes a registration solver and its outputs are not independent
evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np

from safety.v13_colorpcr_pointdsc_shadow import (
    COLORPCR_INPUT_VOXEL_M,
    color_preserving_voxel_aggregate,
)


SCHEMA = "v16-matched-region-colorpcr-builder-v1"
PAIR_SCHEMA = "v16-matched-region-colorpcr-pair-v1"
HYPOTHESIS_SCHEMA = "v16-matched-region-colorpcr-hypothesis-v1"
FORBIDDEN_KEY_PARTS = (
    "gt_transform", "ground_truth", "selection_label", "semantic_label",
    "evaluation_label", "oracle", "official92", "posthoc", "fallback",
)
DECLARATION_KEYS = {"forbidden_inputs", "forbidden_fields", "stop_conditions"}


class V16ContractError(RuntimeError):
    """An authenticated input or deterministic mapping failed closed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode()).hexdigest()


def raw_float32_sha256(value: Any) -> str:
    """Reproduce V10/V11 registration-surface hashes byte for byte."""
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(value, dtype=np.float32)).tobytes()).hexdigest()


def canonical_provenance_binding(
    data: Mapping[str, Any], v10_cache: Mapping[str, Any],
    source_cache: Mapping[str, Any], plan: Mapping[str, Any], *,
    v10_cache_sha256: str, source_cache_sha256: str,
) -> dict[str, Any]:
    """Bind current canonical inputs to the complete frozen V10/V11 chain.

    The V11 hash did not include object IDs, so checking only
    ``canonical_input_sha256`` is insufficient: a node index could silently
    bind to a different object after an input-order drift.  This validator
    separately authenticates obj_ids, src_count, every registration surface,
    the V10 source cache, and all checkpoint identities.
    """
    required_cache = {
        "pair_id", "checkpoint_id", "checkpoint_sha256", "input_sha256",
        "embedding_sha256", "rank_list", "candidate_fingerprint",
        "source_cache_path", "source_cache_sha256", "provenance",
    }
    required_source = {
        "cache_schema", "pair_id", "checkpoint_id", "checkpoint_sha256",
        "input_sha256", "embedding_sha256", "rank_list", "provenance",
    }
    if not required_cache.issubset(v10_cache):
        raise V16ContractError("V10 cache provenance fields are incomplete")
    if not required_source.issubset(source_cache):
        raise V16ContractError("V10 source cache provenance fields are incomplete")
    if source_cache_sha256 != v10_cache["source_cache_sha256"]:
        raise V16ContractError("V10 source-cache SHA mismatch")
    if plan.get("v10_cache_sha256") != v10_cache_sha256:
        raise V16ContractError("V11 plan does not bind the loaded V10 cache SHA")
    if plan.get("candidate_fingerprint") != v10_cache["candidate_fingerprint"]:
        raise V16ContractError("V11/V10 candidate fingerprint mismatch")
    checkpoint = v10_cache["checkpoint_sha256"]
    if (source_cache["checkpoint_id"] != v10_cache["checkpoint_id"]
            or source_cache["checkpoint_sha256"] != checkpoint
            or plan.get("checkpoint_sha256") != checkpoint):
        raise V16ContractError("V10 source/cache/V11 checkpoint chain mismatch")
    for field in ("pair_id", "input_sha256", "embedding_sha256", "rank_list"):
        if source_cache[field] != v10_cache[field]:
            raise V16ContractError(f"V10 source/cache field mismatch: {field}")

    source_provenance = source_cache["provenance"]
    if not isinstance(source_provenance, Mapping):
        raise V16ContractError("V10 source-cache inner provenance is malformed")
    candidate_provenance = dict(v10_cache["provenance"])
    contract = candidate_provenance.pop("v10_candidate_contract", None)
    if not isinstance(contract, Mapping) or candidate_provenance != source_provenance:
        raise V16ContractError("V10 candidate provenance is not a copy of source cache")
    if (source_provenance.get("cache_key") != source_cache["input_sha256"]
            or source_provenance.get("pair_id") != source_cache["pair_id"]
            or source_provenance.get("checkpoint_id") != source_cache["checkpoint_id"]
            or source_provenance.get("checkpoint_sha256") != checkpoint
            or source_provenance.get("unit") != "metres"):
        raise V16ContractError("V10 source-cache inner provenance mismatch")

    obj_ids = np.asarray(data.get("obj_ids"))
    if obj_ids.ndim != 1 or not np.issubdtype(obj_ids.dtype, np.integer):
        raise V16ContractError("current canonical object table is malformed")
    current_ids = [int(value) for value in obj_ids.tolist()]
    current_src_count = int(data.get("src_count", 0))
    if (current_ids != source_provenance.get("object_ids_order")
            or current_src_count != int(source_provenance.get("src_count", -1))):
        raise V16ContractError("current canonical obj_ids/src_count differ from V10 provenance")

    surfaces = data.get("registration_pts")
    if not isinstance(surfaces, Mapping):
        raise V16ContractError("current canonical registration surfaces are missing")
    current_surfaces = [{
        "index": int(index),
        "points": int(len(np.asarray(points))),
        "sha256": raw_float32_sha256(points),
    } for index, points in sorted(surfaces.items(), key=lambda row: int(row[0]))]
    expected_surfaces = source_provenance.get("registration_surfaces")
    if current_surfaces != expected_surfaces:
        raise V16ContractError("current canonical registration surfaces differ from V10 provenance")

    edges = np.asarray(data.get("edges_explicit"))
    if edges.ndim != 2:
        raise V16ContractError("current canonical explicit-edge table is malformed")
    v11_input = {
        "src_count": current_src_count,
        "surface_hashes": {str(row["index"]): row["sha256"]
                           for row in current_surfaces},
        "explicit_edges": edges.tolist(),
    }
    observed_v11_sha = stable_json_sha256(v11_input)
    if observed_v11_sha != plan.get("canonical_input_sha256"):
        raise V16ContractError("V11 canonical_input_sha256 recomputation mismatch")
    return {
        "pair_id": source_cache["pair_id"],
        "rank_source_checkpoint_id": source_cache["checkpoint_id"],
        "rank_source_checkpoint_sha256": checkpoint,
        "v10_cache_sha256": v10_cache_sha256,
        "v10_source_cache_sha256": source_cache_sha256,
        "canonical_input_sha256": observed_v11_sha,
        "canonical_obj_ids_sha256": stable_json_sha256(current_ids),
        "canonical_obj_ids": current_ids,
        "canonical_src_count": current_src_count,
        "registration_surfaces": current_surfaces,
        "binding_sha256": stable_json_sha256({
            "obj_ids": current_ids,
            "src_count": current_src_count,
            "registration_surfaces": current_surfaces,
            "v11_canonical_input_sha256": observed_v11_sha,
            "v10_cache_sha256": v10_cache_sha256,
            "v10_source_cache_sha256": source_cache_sha256,
            "rank_source_checkpoint_sha256": checkpoint,
        }),
    }


def verify_file(path: Path, expected_sha256: str) -> str:
    path = Path(path).resolve()
    before = sha256_file(path)
    if before != expected_sha256:
        raise V16ContractError(f"source SHA mismatch: {path}")
    if sha256_file(path) != before:
        raise V16ContractError(f"source changed while reading: {path}")
    return before


def reject_forbidden_fields(value: Any, path: str = "$") -> None:
    """Reject consumed authority fields, but allow explicit deny-list metadata."""
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key not in DECLARATION_KEYS and any(
                    part in key for part in FORBIDDEN_KEY_PARTS):
                raise V16ContractError(f"forbidden field {path}.{raw_key}")
            reject_forbidden_fields(item, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_forbidden_fields(item, f"{path}[{index}]")


def hypothesis_payload(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "members": hypothesis["members"],
        "member_rank_records": hypothesis["member_rank_records"],
        "member_count": hypothesis["member_count"],
    }


def validate_hypothesis(
    hypothesis: Mapping[str, Any], candidate_rank_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, int]]:
    required = {"hypothesis_index", "hypothesis_sha256", "member_count",
                "members", "member_rank_records"}
    if not required.issubset(hypothesis):
        raise V16ContractError("hypothesis contract missing fields")
    records = [dict(row) for row in hypothesis["member_rank_records"]]
    members = [[int(a), int(b)] for a, b in hypothesis["members"]]
    if len(records) != int(hypothesis["member_count"]) or len(records) != len(members):
        raise V16ContractError("hypothesis member count mismatch")
    expected_hash = stable_json_sha256(hypothesis_payload(hypothesis))
    if hypothesis["hypothesis_sha256"] != expected_hash:
        raise V16ContractError("hypothesis rank/member order or payload SHA mismatch")
    cache_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in candidate_rank_records:
        row = dict(raw)
        key = (int(row["source_index"]), int(row["reference_index"]))
        if key in cache_by_pair:
            raise V16ContractError("candidate rank list has duplicate node pair")
        cache_by_pair[key] = row
    for member, record in zip(members, records):
        key = tuple(member)
        if key not in cache_by_pair or cache_by_pair[key] != record:
            raise V16ContractError("V11 rank record differs from frozen V10 cache")
        if [int(record["source_index"]), int(record["reference_index"])] != member:
            raise V16ContractError("member and rank-record node indices differ")
    reject_forbidden_fields({"hypothesis": hypothesis_payload(hypothesis)})
    return records


def validate_metres(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz)
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,)
            or not np.issubdtype(xyz.dtype, np.floating)
            or not np.isfinite(xyz).all() or len(xyz) == 0):
        raise V16ContractError("raw XYZ must be finite non-empty Nx3")
    span = np.ptp(np.asarray(xyz, np.float64), axis=0)
    if float(np.max(span)) > 100.0 or float(np.max(np.abs(xyz))) > 100.0:
        raise V16ContractError("raw InSeg XYZ is not in the frozen metre scale")
    return np.ascontiguousarray(xyz, np.float32)


@dataclass(frozen=True)
class RawInseg:
    scan_id: str
    side: str
    path: Path
    file_sha256: str
    xyz: np.ndarray
    instance_ids: np.ndarray
    colors: np.ndarray
    field_sha256: Mapping[str, str]


def resolve_unique_inseg_path(scan_id: str, roots: Sequence[Path]) -> Path:
    matches = [Path(root) / scan_id / "inseg_cloud.npz" for root in roots]
    matches = [path.resolve() for path in matches if path.is_file()]
    unique = sorted({str(path): path for path in matches}.values(), key=str)
    if len(unique) != 1:
        raise V16ContractError(
            f"raw InSeg mapping for scan {scan_id} is not unique: {len(unique)}")
    return unique[0]


def load_raw_inseg(path: Path, *, scan_id: str, side: str) -> RawInseg:
    if not scan_id or side not in ("source", "reference"):
        raise V16ContractError("raw InSeg membership requires scan_id and side")
    path = Path(path).resolve()
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"xyz", "labels", "colors"}
        if not required.issubset(data.files):
            raise V16ContractError("raw InSeg lacks xyz/labels/colors")
        xyz = validate_metres(np.asarray(data["xyz"]))
        instance_ids = np.asarray(data["labels"])
        colors = np.asarray(data["colors"])
    if (instance_ids.ndim != 1 or len(instance_ids) != len(xyz)
            or not np.issubdtype(instance_ids.dtype, np.integer)):
        raise V16ContractError("raw instance-id membership column is invalid")
    if (colors.ndim != 2 or colors.shape[0] != len(xyz)
            or colors.shape[1] not in (3, 4)
            or colors.dtype != np.uint8):
        raise V16ContractError("raw RGB must be row-aligned uint8 RGB/RGBA")
    # Equal 1 mm coordinates carrying conflicting instance IDs make membership
    # ambiguous and fail closed. Multiple rows for one object are normal.
    keys = np.rint(np.asarray(xyz, np.float64) / 0.001).astype(np.int64)
    order = np.lexsort((instance_ids, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_o, labels_o = keys[order], instance_ids[order]
    same = np.all(keys_o[1:] == keys_o[:-1], axis=1)
    if np.any(same & (labels_o[1:] != labels_o[:-1])):
        raise V16ContractError("same raw coordinate has conflicting instance membership")
    if sha256_file(path) != before:
        raise V16ContractError("raw InSeg changed while reading")
    fields = {
        "xyz": array_sha256(xyz),
        "raw_instance_membership_key": array_sha256(instance_ids),
        "colors": array_sha256(colors),
    }
    return RawInseg(scan_id, side, path, before, xyz,
                    np.ascontiguousarray(instance_ids, np.int64),
                    np.ascontiguousarray(colors, np.uint8), fields)


def node_object_id(data: Mapping[str, Any], node_index: int, *, side: str) -> int:
    if side not in ("source", "reference"):
        raise V16ContractError("node mapping requires explicit side")
    src_count = int(data["src_count"])
    obj_ids = np.asarray(data["obj_ids"])
    if obj_ids.ndim != 1 or not 0 < src_count < len(obj_ids):
        raise V16ContractError("canonical obj_ids/src_count contract is malformed")
    if len(set(int(x) for x in obj_ids[:src_count])) != src_count:
        raise V16ContractError("source node-to-object mapping is not unique")
    if len(set(int(x) for x in obj_ids[src_count:])) != len(obj_ids) - src_count:
        raise V16ContractError("reference node-to-object mapping is not unique")
    node_index = int(node_index)
    if not 0 <= node_index < len(obj_ids):
        raise V16ContractError("node index missing from canonical object table")
    if side == "source" and not node_index < src_count:
        raise V16ContractError("source node index belongs to reference side")
    if side == "reference" and not node_index >= src_count:
        raise V16ContractError("reference node index belongs to source side")
    return int(obj_ids[node_index])


def canonical_surface_from_rows(raw: RawInseg, object_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero(raw.instance_ids == int(object_id)).astype(np.int64)
    if len(rows) < 50:
        raise V16ContractError("object_id has no admissible raw InSeg row set")
    surface = np.unique(np.round(
        np.asarray(raw.xyz[rows], np.float64), 3), axis=0)
    return rows, np.ascontiguousarray(surface, np.float64)


def verify_canonical_surface(
    data: Mapping[str, Any], node_index: int, reconstructed: np.ndarray,
) -> str:
    surfaces = data.get("registration_pts", {})
    if int(node_index) not in surfaces:
        raise V16ContractError("canonical registration surface is missing")
    expected = np.ascontiguousarray(np.asarray(surfaces[int(node_index)], np.float64))
    if expected.shape != reconstructed.shape or not np.array_equal(expected, reconstructed):
        raise V16ContractError("canonical surface hash mismatch from raw row set")
    return array_sha256(expected)


def canonical_member_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row["source_index"]), int(row["reference_index"]))


def build_side_union(
    records: Sequence[Mapping[str, Any]], data: Mapping[str, Any], raw: RawInseg,
    *, side: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if raw.side != side:
        raise V16ContractError("raw InSeg side does not match union side")
    node_key = "source_index" if side == "source" else "reference_index"
    # Canonical node order makes output independent of caller iteration order.
    nodes = sorted({int(row[node_key]) for row in records})
    arrays, members, offset = [], [], 0
    for node in nodes:
        oid = node_object_id(data, node, side=side)
        rows, reconstructed = canonical_surface_from_rows(raw, oid)
        surface_hash = verify_canonical_surface(data, node, reconstructed)
        xyz = np.ascontiguousarray(raw.xyz[rows], np.float32)
        colors = np.ascontiguousarray(raw.colors[rows], np.uint8)
        end = offset + len(rows)
        arrays.append((xyz, colors, rows, np.full(len(rows), oid, np.int64)))
        members.append({
            "scan_id": raw.scan_id,
            "side": side,
            "node_index": node,
            "object_id": oid,
            "raw_inseg_path": str(raw.path),
            "raw_inseg_sha256": raw.file_sha256,
            "raw_instance_membership_field_sha256":
                raw.field_sha256["raw_instance_membership_key"],
            "surface_offset": [offset, end],
            "raw_row_count": len(rows),
            "raw_row_indices_sha256": array_sha256(rows),
            "raw_xyz_sha256": array_sha256(xyz),
            "raw_rgb_sha256": array_sha256(colors),
            "canonical_surface_sha256": surface_hash,
        })
        offset = end
    if not arrays:
        raise V16ContractError("hypothesis side has no members")
    union = {
        "xyz": np.ascontiguousarray(np.concatenate([x[0] for x in arrays])),
        "colors": np.ascontiguousarray(np.concatenate([x[1] for x in arrays])),
        "source_row_indices": np.ascontiguousarray(np.concatenate([x[2] for x in arrays])),
        "membership_object_ids": np.ascontiguousarray(np.concatenate([x[3] for x in arrays])),
        "member_offsets": np.asarray([m["surface_offset"][0] for m in members]
                                     + [members[-1]["surface_offset"][1]], np.int64),
    }
    return union, members


def _deterministic_npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        tmp = Path(temporary.name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _deterministic_npy_bytes(arrays[key]))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        tmp = Path(stream.name)
    os.replace(tmp, path)


def build_hypothesis_artifact(
    pair_id: str, hypothesis: Mapping[str, Any], candidate_records: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any], source: RawInseg, reference: RawInseg,
    output_dir: Path, provenance: Mapping[str, Any],
) -> dict[str, Any]:
    records = validate_hypothesis(hypothesis, candidate_records)
    src_union, src_members = build_side_union(records, data, source, side="source")
    ref_union, ref_members = build_side_union(records, data, reference, side="reference")
    src_voxel = color_preserving_voxel_aggregate(src_union, COLORPCR_INPUT_VOXEL_M)
    ref_voxel = color_preserving_voxel_aggregate(ref_union, COLORPCR_INPUT_VOXEL_M)
    arrays: dict[str, np.ndarray] = {}
    for side, union, voxel in (
        ("source", src_union, src_voxel),
        ("reference", ref_union, ref_voxel),
    ):
        for key, value in union.items():
            arrays[f"{side}_{key}"] = value
        for key, value in voxel.items():
            arrays[f"{side}_voxel10_{key}"] = value
    stem = str(hypothesis["hypothesis_sha256"])
    npz_path = Path(output_dir) / f"{stem}.npz"
    write_deterministic_npz(npz_path, arrays)
    evidence = {
        "schema": HYPOTHESIS_SCHEMA,
        "pair_id": pair_id,
        "hypothesis_index": int(hypothesis["hypothesis_index"]),
        "hypothesis_sha256": stem,
        "disabled": True,
        "independent_evidence": False,
        "registration_executed": False,
        "colorpcr_consumption_allowed": False,
        "diagnostic_only_due_checkpoint_domain_mismatch": True,
        "forward_reverse_colorpcr": (
            "not authorized; after release-domain reranking, each direction must "
            "be executed independently in a later preregistered stage"),
        "member_rank_records": records,
        "members": {"source": src_members, "reference": ref_members},
        "raw_instance_membership_contract": (
            "same-scan (scan_id,side,object_id)->rows only; not cross-scan identity, "
            "semantic/ranking/solver input"),
        "preprocessing": {
            "filter_before_voxel": True,
            "voxel_size_m": COLORPCR_INPUT_VOXEL_M,
            "voxel_origin": [0.0, 0.0, 0.0],
            "builder_output_stage": "raw_union_and_voxel10_prepared_only",
            "builder_cap512_applied": False,
            "official_worker_cap512_stage": (
                "after official repeated grid_subsample at the final coarsest level"),
            "official_worker_owns_cap512": True,
        },
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                          "sha256": array_sha256(value)}
                   for key, value in sorted(arrays.items())},
        "npz_path": str(npz_path),
        "npz_sha256": sha256_file(npz_path),
        "provenance_closure": dict(provenance),
        "forbidden_inputs": ["semantic/GT/selection/posthoc labels", "GT transforms",
                             "posthoc", "official92", "fallback"],
    }
    evidence["payload_sha256"] = stable_json_sha256(evidence)
    json_path = Path(output_dir) / f"{stem}.json"
    atomic_json(json_path, evidence)
    return {"hypothesis_sha256": stem, "npz_path": str(npz_path),
            "npz_sha256": sha256_file(npz_path), "evidence_path": str(json_path),
            "evidence_sha256": sha256_file(json_path)}
