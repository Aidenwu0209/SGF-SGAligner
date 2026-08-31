"""GT-free, fail-closed ColorPCR/PointDSC shadow contracts for V13."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from safety.v8_stage_order_consensus import V8Config, cluster_direction, transform_distance

SCHEMA = "v13-colorpcr-pointdsc-shadow-v2"
PAIR_SCHEMA = "v13-color-preserving-pair-v2"
PILOT_POSITIONS = (0, 44, 88)
ARMS = ("sgf_selected_union", "fullscan")
SOLVERS = ("pointdsc", "pygcransac")
Q4 = V8Config(repeats=5, quorum=4, max_rotation_deg=5.0, max_translation_m=0.10)
FORBIDDEN_INPUTS = ("GT transforms", "pair identity fallback", "selection labels",
                    "calibration labels", "posthoc outcomes", "official92")
UNIT_CONTRACT = "metres_from_snapshot_inseg_no_rescale"
SHADOW_GRID_M = 0.001
COLORPCR_INPUT_VOXEL_M = 0.10
COLORPCR_COARSEST_CAP = 512


class V13ContractError(RuntimeError):
    pass


class DependencyNotAudited(V13ContractError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class RawInsegCloud:
    xyz: np.ndarray
    labels: np.ndarray
    colors: np.ndarray
    source_path: str
    source_sha256: str
    kept_row_indices: np.ndarray
    duplicate_rows_removed: int


def _validate(xyz: np.ndarray, labels: np.ndarray, colors: np.ndarray) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise V13ContractError("xyz must be non-empty N x 3")
    if labels.shape != (len(xyz),):
        raise V13ContractError("labels must stay row-aligned with xyz")
    if colors.ndim != 2 or colors.shape[0] != len(xyz) or colors.shape[1] not in (3, 4):
        raise V13ContractError("colors must be row-aligned uint8 RGB/RGBA")
    if not np.issubdtype(labels.dtype, np.integer) or colors.dtype != np.uint8:
        raise V13ContractError("labels must be integer and colors source uint8")
    if not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all():
        raise V13ContractError("xyz must be finite floating-point metres")


def _dedup(xyz: np.ndarray, labels: np.ndarray, colors: np.ndarray):
    seen: dict[bytes, tuple[bytes, bytes]] = {}
    keep: list[int] = []
    for index in range(len(xyz)):
        key = np.ascontiguousarray(xyz[index]).tobytes()
        payload = (np.ascontiguousarray(labels[index]).tobytes(),
                   np.ascontiguousarray(colors[index]).tobytes())
        if key not in seen:
            seen[key] = payload
            keep.append(index)
        elif seen[key] != payload:
            raise V13ContractError("duplicate xyz has conflicting label/color")
    indices = np.asarray(keep, np.int64)
    return xyz[indices], labels[indices], colors[indices], indices


def load_raw_inseg(path: Path) -> RawInsegCloud:
    path = Path(path).resolve()
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if not {"xyz", "labels", "colors"}.issubset(data.files):
            raise V13ContractError("raw InSeg requires xyz/labels/colors")
        xyz, labels, colors = (np.asarray(data[k]) for k in ("xyz", "labels", "colors"))
    original = len(xyz)
    _validate(xyz, labels, colors)
    xyz, labels, colors, indices = _dedup(xyz.astype(np.float32), labels, colors)
    if sha256_file(path) != before:
        raise V13ContractError("raw source changed while reading")
    return RawInsegCloud(xyz, labels, colors, str(path), before, indices,
                         original - len(indices))


def _shadow_membership_indices(cloud: RawInsegCloud, shadow_points: np.ndarray) -> tuple[np.ndarray, dict]:
    raw_q = np.rint(np.asarray(cloud.xyz, np.float64) / SHADOW_GRID_M).astype(np.int64)
    shadow_q = np.rint(np.asarray(shadow_points, np.float64) / SHADOW_GRID_M).astype(np.int64)
    raw_index: dict[tuple[int, int, int], int] = {}
    ambiguous: set[tuple[int, int, int]] = set()
    for index, row in enumerate(raw_q):
        key = tuple(int(x) for x in row)
        if key in raw_index:
            ambiguous.add(key)
        else:
            raw_index[key] = index
    required = sorted({tuple(int(x) for x in row) for row in shadow_q})
    missing = [key for key in required if key not in raw_index]
    collisions = [key for key in required if key in ambiguous]
    if missing or collisions:
        raise V13ContractError(
            f"sgf selected-union membership is not unique: missing={len(missing)} ambiguous={len(collisions)}"
        )
    indices = np.asarray(sorted((raw_index[key] for key in required),
                                key=lambda i: int(cloud.kept_row_indices[i])), np.int64)
    if len(indices) < 3:
        raise V13ContractError("sgf selected-union has fewer than three uniquely mapped rows")
    return indices, {"shadow_unique_grid_points": len(required),
                     "mapped_unique_rows": len(indices), "missing": 0, "ambiguous": 0,
                     "mapping_coverage": 1.0, "grid_m": SHADOW_GRID_M}


def arm_arrays(cloud: RawInsegCloud, arm: str,
               shadow_points: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if arm not in ARMS:
        raise V13ContractError("unknown arm")
    if arm == "fullscan":
        indices = np.arange(len(cloud.xyz), dtype=np.int64)
        membership = {"mode": "all_raw_rows", "mapped_unique_rows": len(indices)}
    else:
        if shadow_points is None:
            raise V13ContractError("sgf_selected_union requires sealed V11.3 shadow points")
        indices, membership = _shadow_membership_indices(cloud, shadow_points)
        membership["mode"] = "sealed_v113_shadow_1mm_unique_membership"
    return {"xyz": np.ascontiguousarray(cloud.xyz[indices], np.float32),
            "labels": np.ascontiguousarray(cloud.labels[indices]),
            "colors": np.ascontiguousarray(cloud.colors[indices], np.uint8),
            "source_row_indices": np.ascontiguousarray(cloud.kept_row_indices[indices], np.int64),
            "membership_json": np.asarray(json.dumps(membership, sort_keys=True))}


def color_preserving_voxel_aggregate(values: Mapping[str, np.ndarray],
                                     voxel_size_m: float = COLORPCR_INPUT_VOXEL_M) -> dict[str, np.ndarray]:
    """Deterministically aggregate one already-filtered arm without label voting.

    The grid is anchored at the world origin and uses floor(xyz / voxel_size).
    Voxels are lexicographically ordered.  XYZ and RGB are float64 means cast to
    float32.  Every output voxel retains the exact source-row membership through
    CSR-style offsets; labels are intentionally not aggregated or exposed.
    """
    if voxel_size_m != COLORPCR_INPUT_VOXEL_M:
        raise V13ContractError("voxel size is pre-registered at exactly 0.10 m")
    xyz = np.asarray(values["xyz"], np.float64)
    colors = np.asarray(values["colors"][:, :3], np.float64)
    rows = np.asarray(values["source_row_indices"], np.int64)
    if not len(xyz) or len(colors) != len(xyz) or len(rows) != len(xyz):
        raise V13ContractError("voxel input must be non-empty and row aligned")
    keys = np.floor(xyz / voxel_size_m).astype(np.int64)
    order = np.lexsort((rows, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys, xyz, colors, rows = keys[order], xyz[order], colors[order], rows[order]
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    offsets = np.r_[starts, len(keys)].astype(np.int64)
    counts = np.diff(offsets).astype(np.float64)
    xyz_sum = np.add.reduceat(xyz, starts, axis=0)
    color_sum = np.add.reduceat(colors, starts, axis=0)
    return {
        "xyz": np.ascontiguousarray(xyz_sum / counts[:, None], np.float32),
        "colors_mean_0_255": np.ascontiguousarray(color_sum / counts[:, None], np.float32),
        "keys": np.ascontiguousarray(keys[starts], np.int64),
        "source_row_indices_flat": np.ascontiguousarray(rows, np.int64),
        "source_offsets": np.ascontiguousarray(offsets, np.int64),
    }


def _grid_keys(points: np.ndarray) -> set[tuple[int, int, int]]:
    q = np.rint(np.asarray(points, np.float64) / SHADOW_GRID_M).astype(np.int64)
    return {tuple(row) for row in q}


def validate_v113_shadow(path: Path, pair_id: str,
                         source: RawInsegCloud, reference: RawInsegCloud) -> dict:
    path = Path(path).resolve()
    before = sha256_file(path)
    required = {"source_points", "reference_points", "forward_src_corr",
                "forward_ref_corr", "forward_scores", "reverse_src_corr",
                "reverse_ref_corr", "reverse_scores"}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise V13ContractError("V11.3 shadow NPZ schema mismatch")
        arrays = {key: np.asarray(data[key]) for key in sorted(required)}
    for key, value in arrays.items():
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise V13ContractError(f"invalid shadow array {key}")
    for raw, key in ((source, "source_points"), (reference, "reference_points")):
        if not _grid_keys(arrays[key]).issubset(_grid_keys(raw.xyz)):
            raise V13ContractError("shadow XYZ not contained in raw InSeg on 1mm grid")
    if sha256_file(path) != before:
        raise V13ContractError("shadow changed while reading")
    return {"pair_id": pair_id, "path": str(path), "sha256": before,
            "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
            "compatibility": "shadow_xyz_subset_on_frozen_1mm_grid"}


def _cloud_manifest(cloud: RawInsegCloud) -> dict:
    return {"path": cloud.source_path, "sha256": cloud.source_sha256,
            "rows_after_dedup": len(cloud.xyz),
            "duplicates_removed": cloud.duplicate_rows_removed,
            "xyz_sha256": array_sha256(cloud.xyz),
            "labels_sha256": array_sha256(cloud.labels),
            "colors_sha256": array_sha256(cloud.colors),
            "kept_row_indices_sha256": array_sha256(cloud.kept_row_indices)}


def build_color_pair(pair_id: str, source_path: Path, reference_path: Path,
                     shadow_path: Path, output_path: Path) -> dict:
    if len(pair_id.split("_to_")) != 2:
        raise V13ContractError("pair id must be src_to_ref")
    source, reference = load_raw_inseg(source_path), load_raw_inseg(reference_path)
    shadow = validate_v113_shadow(shadow_path, pair_id, source, reference)
    with np.load(shadow_path, allow_pickle=False) as frozen:
        shadow_points = {"source": np.asarray(frozen["source_points"]),
                         "reference": np.asarray(frozen["reference_points"])}
    if sha256_file(shadow_path) != shadow["sha256"]:
        raise V13ContractError("shadow changed between validation and membership mapping")
    payload: dict[str, np.ndarray] = {}
    arms: dict[str, Any] = {}
    for arm in ARMS:
        arms[arm] = {}
        for side, cloud in (("source", source), ("reference", reference)):
            values = arm_arrays(cloud, arm, shadow_points[side] if arm == "sgf_selected_union" else None)
            membership = json.loads(str(values.pop("membership_json").item()))
            voxel = color_preserving_voxel_aggregate(values)
            arms[arm][side] = {"points_before_voxel": len(values["xyz"]),
                "points_after_voxel10": len(voxel["xyz"]),
                **{f"{key}_sha256": array_sha256(value) for key, value in values.items()},
                **{f"voxel10_{key}_sha256": array_sha256(value) for key, value in voxel.items()},
                "label_aggregation": "forbidden_filter_only", "selection_membership": membership}
            payload.update({f"{arm}_{side}_{key}": value for key, value in values.items()})
            payload.update({f"{arm}_{side}_voxel10_{key}": value for key, value in voxel.items()})
    arms_identical = all(
        arms["sgf_selected_union"][side]["voxel10_xyz_sha256"] ==
        arms["fullscan"][side]["voxel10_xyz_sha256"] and
        arms["sgf_selected_union"][side]["voxel10_colors_mean_0_255_sha256"] ==
        arms["fullscan"][side]["voxel10_colors_mean_0_255_sha256"]
        for side in ("source", "reference")
    )
    manifest = {"schema": PAIR_SCHEMA, "pair_id": pair_id,
                "unit_contract": UNIT_CONTRACT, "rescale_applied": False,
                "source": _cloud_manifest(source), "reference": _cloud_manifest(reference),
                "v113_shadow": shadow, "arms": arms,
                "arms_identical_after_filter_and_voxel": arms_identical,
                "arms_are_not_independent_evidence": True,
                "colorpcr_input_voxel": {
                    "size_m": COLORPCR_INPUT_VOXEL_M,
                    "origin": [0.0, 0.0, 0.0],
                    "key": "floor(xyz/0.10)",
                    "xyz": "float64_mean_then_float32",
                    "rgb": "float64_mean_then_float32_0_255",
                    "labels": "filter_before_voxel_no_majority_or_solver_input",
                    "source_provenance": "csr_flat_rows_plus_offsets",
                },
                "colorpcr_coarsest_cap": {
                    "max_total_points": COLORPCR_COARSEST_CAP,
                    "method": "deterministic_per_scan_proportional_fps_start_index_0",
                },
                "color_normalization_for_solver_only": "float32_rgb_mean/255.0",
                "forbidden_inputs": list(FORBIDDEN_INPUTS)}
    manifest["payload_sha256"] = stable_json_sha256(manifest)
    payload["manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    manifest["prepared_npz_path"] = str(output_path.resolve())
    manifest["prepared_npz_sha256"] = sha256_file(output_path)
    return manifest


def worker_plan(pair_ids: Sequence[str]) -> list[dict]:
    return [{"pair_id": pair, "arm": arm, "solver": solver, "direction": direction,
             "repeat": repeat} for pair in pair_ids for arm in ARMS for solver in SOLVERS
            for direction in ("forward", "reverse") for repeat in range(Q4.repeats)]


def validate_solver_worker(row: Mapping[str, Any]) -> dict:
    solver = row.get("solver")
    if solver not in SOLVERS:
        raise V13ContractError("solver missing")
    if row.get("dependency_audited") is not True or not row.get("implementation_sha256") \
            or not row.get("checkpoint_sha256"):
        raise DependencyNotAudited(f"{solver} source/weights not sealed")
    if row.get("executed") is not True or row.get("fallback_used") is not False:
        raise V13ContractError("executed non-fallback solver required")
    if not row.get("correspondence_sha256") or not row.get("evidence_sha256"):
        raise V13ContractError("evidence binding missing")
    transform = np.asarray(row.get("transform"), np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() \
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8) \
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) \
            or np.linalg.det(transform[:3, :3]) < 0.999:
        raise V13ContractError("proper finite SE3 required")
    result = dict(row)
    result.update(transform=transform, status="ok")
    return result


def independent_solver_q4(rows: Sequence[Mapping[str, Any]]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        item = validate_solver_worker(row)
        grouped.setdefault((str(row["solver"]), str(row["direction"])), []).append(item)
    expected = {(s, d) for s in SOLVERS for d in ("forward", "reverse")}
    if set(grouped) != expected:
        raise V13ContractError("both solvers/directions mandatory")
    medoids, gates = {}, {}
    for key in sorted(expected):
        group = sorted(grouped[key], key=lambda x: int(x["repeat"]))
        if [int(x["repeat"]) for x in group] != list(range(5)):
            raise V13ContractError("repeats must be 0..4")
        gate = cluster_direction(group, Q4)
        gates["/".join(key)] = gate
        if not gate["usable"]:
            return {"safe": False, "reason": "solver_q4_failed", "gates": gates}
        medoids[key] = group[gate["medoid_original_index"]]["transform"]
    canonical = {}
    for solver in SOLVERS:
        forward, reverse = medoids[(solver, "forward")], np.linalg.inv(medoids[(solver, "reverse")])
        if any(a > b for a, b in zip(transform_distance(forward, reverse),
                                      (Q4.max_rotation_deg, Q4.max_translation_m))):
            return {"safe": False, "reason": "cross_direction_failed", "gates": gates}
        canonical[solver] = forward
    if any(a > b for a, b in zip(transform_distance(canonical[SOLVERS[0]], canonical[SOLVERS[1]]),
                                  (Q4.max_rotation_deg, Q4.max_translation_m))):
        return {"safe": False, "reason": "independent_solvers_disagree", "gates": gates}
    if any(row.get("rule_b_safe") is not True for row in rows):
        return {"safe": False, "reason": "rule_b_not_safe", "gates": gates}
    if any(row.get("known_bad") is True for row in rows):
        return {"safe": False, "reason": "known_bad_veto", "gates": gates}
    return {"safe": True, "reason": "unique_independent_q4_mode", "gates": gates}
