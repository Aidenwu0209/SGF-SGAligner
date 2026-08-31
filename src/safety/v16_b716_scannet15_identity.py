"""Hash-bound identity contract for the independent ScanNet15 bridge.

This module extends *identity validation only*.  It does not authorize or run
ColorPCR, a rigid solver, ICP, reconstruction, refusion, or official92.  The
historical V13/V14 fixed-four schemas remain separate and unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


PREREGISTER_SCHEMA = "v16-b716-scannet15-identity-preregister-v1"
PREFLIGHT_SCHEMA = "v16-b716-scannet15-identity-preflight-v1"
PREPARED_SCHEMA = "v16-b716-scannet15-official-colorpcr-input-v1"
PAIR_COUNT = 15
PREPARED_MANIFEST_KEYS = {
    "schema", "scene_id", "pair_role", "arm", "unit",
    "attribute_available", "source_raw_ply_sha256",
    "reference_raw_ply_sha256", "raw_pair_inventory_sha256",
    "raw_pair_receipt_sha256", "sgf_prediction_sha256",
    "official_repo_head", "official_checkpoint_sha256",
    "geotransformer_checkpoint_sha256",
    "bridge_source_sha256", "colorpcr_schema_source_sha256",
    "official_sgaligner_tensor_contract_compatible",
    "v13_colorpcr_worker_schema_compatible", "source_pretransformed",
    "transform_present", "registration_executed", "gt_consumed",
    "worker_execution_authorized", "formal_execution_authorized",
    "payload_sha256",
}
OFFICIAL_KEYS = {
    "official_edges", "official_graph_per_edge_count",
    "official_graph_per_obj_count", "official_obj_ids",
    "official_pcl_center", "official_registration_node_indices",
    "official_registration_object_ids", "official_registration_offsets",
    "official_registration_xyz", "official_src_count",
    "official_tot_bow_vec_object_edge_feats", "official_tot_obj_pts",
    "official_tot_rel_pose",
}
SIDE_SUFFIXES = {
    "colors", "labels", "member_offsets", "membership_object_ids",
    "source_row_indices", "voxel10_colors_mean_0_255", "voxel10_keys",
    "voxel10_source_offsets", "voxel10_source_row_indices_flat",
    "voxel10_xyz", "xyz",
}
PREPARED_NPZ_KEYS = {"manifest_json", *OFFICIAL_KEYS} | {
    f"sgf_selected_union_{side}_{suffix}"
    for side in ("source", "reference") for suffix in SIDE_SUFFIXES
}
PAIR_BINDING_SHA_FIELDS = {
    "prepared_npz_sha256", "prepared_manifest_sha256",
    "prepared_manifest_payload_sha256", "raw_inventory_sha256",
    "raw_pair_receipt_sha256", "source_raw_ply_sha256",
    "reference_raw_ply_sha256", "sgf_prediction_sha256",
}
PAIR_BINDING_PATH_FIELDS = {
    "prepared_npz_path", "prepared_manifest_path", "raw_inventory_path",
    "raw_pair_receipt_path", "source_raw_ply_path",
    "reference_raw_ply_path", "sgf_prediction_path",
}
POLICY_FALSE_FIELDS = {
    "execution_authorized", "gpu_authorized", "gt_allowed",
    "identity_fallback_allowed", "threshold_change_allowed",
    "result_selection_allowed", "reconstruction_authorized",
    "refusion_allowed", "official92_allowed",
}


class ScanNet15IdentityError(RuntimeError):
    """A ScanNet15 identity or evidence closure is malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def _sha(value: Any, name: str) -> str:
    value = str(value)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ScanNet15IdentityError(f"{name} is not lowercase SHA-256")
    return value


def _sealed(value: Mapping[str, Any], name: str) -> None:
    observed = value.get("payload_sha256")
    unsigned = {key: item for key, item in value.items()
                if key != "payload_sha256"}
    if observed != stable_json_sha256(unsigned):
        raise ScanNet15IdentityError(f"{name} payload SHA mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ScanNet15IdentityError(f"JSON is not an exact regular file: {path}")
    before = sha256_file(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise ScanNet15IdentityError(f"JSON changed while reading: {path}")
    return value


def pair_id_for_scene(scene_id: str) -> str:
    if (not isinstance(scene_id, str)
            or re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", scene_id) is None):
        raise ScanNet15IdentityError("scene_id is malformed")
    return f"{scene_id}_source_to_reference"


def validate_prepared_manifest(
    manifest: Mapping[str, Any], *, pair_id: str, prepared_sha256: str,
    preregister: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every field in the independent prepared manifest.

    A new-schema manifest is accepted only when its exact pair is also present
    in the supplied hash-bound preregistration.  This is deliberately stricter
    than merely checking the schema string.
    """
    if set(manifest) != PREPARED_MANIFEST_KEYS:
        raise ScanNet15IdentityError("prepared manifest fields are not exact")
    if manifest.get("schema") != PREPARED_SCHEMA:
        raise ScanNet15IdentityError("prepared manifest schema mismatch")
    _sealed(manifest, "prepared manifest")
    scene_id = str(manifest.get("scene_id", ""))
    if pair_id_for_scene(scene_id) != pair_id:
        raise ScanNet15IdentityError("prepared pair/scene identity mismatch")
    if (manifest.get("pair_role")
            != "same-terminal-surface-spatial-partition"
            or manifest.get("arm") != "sgf_selected_union"
            or manifest.get("unit") != "metre"
            or manifest.get("attribute_available") is not False
            or manifest.get("official_sgaligner_tensor_contract_compatible")
            is not True
            or manifest.get("v13_colorpcr_worker_schema_compatible") is not True
            or manifest.get("source_pretransformed") is not False
            or manifest.get("transform_present") is not False
            or manifest.get("registration_executed") is not False
            or manifest.get("gt_consumed") is not False
            or manifest.get("worker_execution_authorized") is not False
            or manifest.get("formal_execution_authorized") is not False):
        raise ScanNet15IdentityError("prepared manifest policy/contract mismatch")
    for field in PREPARED_MANIFEST_KEYS:
        if field.endswith("_sha256"):
            _sha(manifest[field], f"prepared manifest {field}")
    if (not isinstance(manifest.get("official_repo_head"), str)
            or len(str(manifest["official_repo_head"])) != 40):
        raise ScanNet15IdentityError("official repo HEAD is malformed")
    prepared_sha256 = _sha(prepared_sha256, "prepared NPZ")
    if preregister is None:
        raise ScanNet15IdentityError(
            "ScanNet15 prepared schema requires preregistered identity")
    validate_preregister(preregister)
    row = pair_row(preregister, pair_id)
    expected = {
        "scene_id": scene_id,
        "prepared_npz_sha256": prepared_sha256,
        "prepared_manifest_payload_sha256": manifest["payload_sha256"],
        "raw_inventory_sha256": manifest["raw_pair_inventory_sha256"],
        "raw_pair_receipt_sha256": manifest["raw_pair_receipt_sha256"],
        "source_raw_ply_sha256": manifest["source_raw_ply_sha256"],
        "reference_raw_ply_sha256": manifest["reference_raw_ply_sha256"],
        "sgf_prediction_sha256": manifest["sgf_prediction_sha256"],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ScanNet15IdentityError(
                f"prepared manifest preregistration mismatch: {field}")
    global_expected = {
        "official_repo_head": manifest["official_repo_head"],
        "official_checkpoint_sha256": manifest["official_checkpoint_sha256"],
        "geotransformer_checkpoint_sha256":
            manifest["geotransformer_checkpoint_sha256"],
        "bridge_source_sha256": manifest["bridge_source_sha256"],
        "colorpcr_schema_source_sha256":
            manifest["colorpcr_schema_source_sha256"],
    }
    for field, value in global_expected.items():
        if preregister.get(field) != value:
            raise ScanNet15IdentityError(
                f"prepared global preregistration mismatch: {field}")
    return dict(row)


def _exact_dtype(array: np.ndarray, dtype: np.dtype[Any], name: str) -> None:
    if np.asarray(array).dtype != np.dtype(dtype):
        raise ScanNet15IdentityError(f"prepared array dtype mismatch: {name}")


def _finite_shape(array: np.ndarray, shape_tail: tuple[int, ...],
                  name: str) -> None:
    array = np.asarray(array)
    if (array.ndim != len(shape_tail) + 1
            or tuple(array.shape[1:]) != shape_tail
            or not np.isfinite(array).all()):
        raise ScanNet15IdentityError(f"prepared array shape/value mismatch: {name}")


def validate_prepared_npz(
    prepared_path: Path, *, pair_id: str, preregister: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact 36-key prepared tensor/provenance contract."""
    prepared_path = Path(prepared_path).resolve()
    if not prepared_path.is_file() or prepared_path.is_symlink():
        raise ScanNet15IdentityError("prepared NPZ is not a regular file")
    before = sha256_file(prepared_path)
    with np.load(prepared_path, allow_pickle=False) as data:
        if set(data.files) != PREPARED_NPZ_KEYS:
            raise ScanNet15IdentityError("prepared NPZ keys are not exact36")
        manifest_raw = np.asarray(data["manifest_json"])
        if manifest_raw.shape != () or manifest_raw.dtype.kind != "U":
            raise ScanNet15IdentityError(
                "prepared manifest_json must be scalar unicode")
        manifest = json.loads(str(manifest_raw.item()))
        arrays = {key: np.asarray(data[key]) for key in data.files
                  if key != "manifest_json"}
    if sha256_file(prepared_path) != before:
        raise ScanNet15IdentityError("prepared NPZ changed while reading")
    row = validate_prepared_manifest(
        manifest, pair_id=pair_id, prepared_sha256=before,
        preregister=preregister)

    integer = (
        "official_edges", "official_graph_per_edge_count",
        "official_graph_per_obj_count", "official_obj_ids",
        "official_registration_node_indices",
        "official_registration_object_ids", "official_registration_offsets",
        "official_src_count",
    )
    for key in integer:
        _exact_dtype(arrays[key], np.int64, key)
    n_obj = len(arrays["official_obj_ids"])
    src_count = int(np.asarray(arrays["official_src_count"]).item())
    if (arrays["official_edges"].ndim != 2
            or arrays["official_edges"].shape[1:] != (2,)
            or arrays["official_graph_per_edge_count"].shape != (2,)
            or arrays["official_graph_per_obj_count"].shape != (2,)
            or arrays["official_obj_ids"].shape != (n_obj,)
            or arrays["official_registration_node_indices"].shape != (n_obj,)
            or arrays["official_registration_object_ids"].shape != (n_obj,)
            or arrays["official_registration_offsets"].shape != (n_obj + 1,)
            or arrays["official_src_count"].shape != ()
            or not 0 < src_count < n_obj):
        raise ScanNet15IdentityError("official integer tensor contract mismatch")
    for key, tail, dtype in (
        ("official_pcl_center", (), np.float64),
        ("official_registration_xyz", (3,), np.float32),
        ("official_tot_bow_vec_object_edge_feats", (41,), np.float32),
        ("official_tot_obj_pts", (512, 3), np.float32),
        ("official_tot_rel_pose", (3,), np.float32),
    ):
        array = arrays[key]
        _exact_dtype(array, dtype, key)
        if key == "official_pcl_center":
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ScanNet15IdentityError("official pcl center malformed")
        else:
            _finite_shape(array, tail, key)
    if (len(arrays["official_tot_bow_vec_object_edge_feats"]) != n_obj
            or len(arrays["official_tot_obj_pts"]) != n_obj
            or len(arrays["official_tot_rel_pose"]) != n_obj
            or arrays["official_registration_offsets"][0] != 0
            or arrays["official_registration_offsets"][-1]
            != len(arrays["official_registration_xyz"])
            or (np.diff(arrays["official_registration_offsets"]) <= 0).any()
            or not np.array_equal(arrays["official_obj_ids"],
                                  arrays["official_registration_object_ids"])):
        raise ScanNet15IdentityError("official tensor rows/provenance mismatch")

    side_receipts = {}
    for side in ("source", "reference"):
        prefix = f"sgf_selected_union_{side}_"
        xyz, colors = arrays[prefix + "xyz"], arrays[prefix + "colors"]
        labels = arrays[prefix + "labels"]
        membership = arrays[prefix + "membership_object_ids"]
        rows = arrays[prefix + "source_row_indices"]
        member_offsets = arrays[prefix + "member_offsets"]
        voxel_xyz = arrays[prefix + "voxel10_xyz"]
        voxel_colors = arrays[prefix + "voxel10_colors_mean_0_255"]
        voxel_keys = arrays[prefix + "voxel10_keys"]
        voxel_offsets = arrays[prefix + "voxel10_source_offsets"]
        voxel_flat = arrays[prefix + "voxel10_source_row_indices_flat"]
        for key in ("xyz", "voxel10_xyz", "voxel10_colors_mean_0_255"):
            _exact_dtype(arrays[prefix + key], np.float32, prefix + key)
            _finite_shape(arrays[prefix + key], (3,), prefix + key)
        _exact_dtype(colors, np.uint8, prefix + "colors")
        if colors.ndim != 2 or colors.shape[1:] != (3,):
            raise ScanNet15IdentityError(f"{side} RGB shape mismatch")
        for key in ("labels", "membership_object_ids", "source_row_indices",
                    "member_offsets", "voxel10_keys",
                    "voxel10_source_offsets",
                    "voxel10_source_row_indices_flat"):
            _exact_dtype(arrays[prefix + key], np.int64, prefix + key)
        count, voxel_count = len(xyz), len(voxel_xyz)
        if (count < 40 or colors.shape != (count, 3)
                or labels.shape != (count,) or membership.shape != (count,)
                or rows.shape != (count,) or not np.array_equal(labels, membership)
                or (rows < 0).any() or len(np.unique(rows)) != count
                or member_offsets.ndim != 1 or member_offsets[0] != 0
                or member_offsets[-1] != count
                or (np.diff(member_offsets) <= 0).any()):
            raise ScanNet15IdentityError(f"{side} raw surface contract mismatch")
        for start, stop in zip(member_offsets[:-1], member_offsets[1:]):
            if np.any(labels[start:stop] != labels[start]):
                raise ScanNet15IdentityError(
                    f"{side} object membership is not contiguous")
        if (voxel_count < 40 or voxel_colors.shape != (voxel_count, 3)
                or voxel_keys.shape != (voxel_count, 3)
                or voxel_offsets.shape != (voxel_count + 1,)
                or voxel_offsets[0] != 0 or voxel_offsets[-1] != count
                or (np.diff(voxel_offsets) <= 0).any()
                or voxel_flat.shape != (count,)
                or len(np.unique(voxel_flat)) != count
                or set(map(int, voxel_flat)) != set(map(int, rows))
                or (voxel_colors < 0).any() or (voxel_colors > 255).any()):
            raise ScanNet15IdentityError(f"{side} voxel CSR contract mismatch")
        key_tuples = [tuple(map(int, value)) for value in voxel_keys]
        if key_tuples != sorted(set(key_tuples)):
            raise ScanNet15IdentityError(f"{side} voxel keys are not canonical")
        # A float32 voxel mean can round onto the open upper boundary (for
        # example, original z<0.9 becoming mean z==0.9).  Re-flooring the mean
        # would then spuriously advance the key.  Validate the actual CSR
        # members against the recorded key and recompute both means instead.
        local_by_origin = {int(origin): index
                           for index, origin in enumerate(rows)}
        for voxel_index, (start, stop) in enumerate(zip(
                voxel_offsets[:-1], voxel_offsets[1:])):
            try:
                local = np.asarray([
                    local_by_origin[int(origin)]
                    for origin in voxel_flat[start:stop]], np.int64)
            except KeyError as exc:
                raise ScanNet15IdentityError(
                    f"{side} voxel CSR references an unknown row") from exc
            member_xyz = xyz[local]
            if not np.all(np.floor(
                    member_xyz.astype(np.float64) / 0.10).astype(np.int64)
                    == voxel_keys[voxel_index]):
                raise ScanNet15IdentityError(
                    f"{side} voxel member/key mismatch")
            if (not np.allclose(
                    voxel_xyz[voxel_index],
                    member_xyz.astype(np.float64).mean(axis=0),
                                rtol=0.0, atol=2e-6)
                    or not np.allclose(
                        voxel_colors[voxel_index],
                        colors[local].astype(np.float64).mean(axis=0),
                        rtol=0.0, atol=2e-5)):
                raise ScanNet15IdentityError(
                    f"{side} voxel aggregate mismatch")
        side_receipts[side] = {
            "raw_point_count": count, "voxel10_point_count": voxel_count,
        }
    return row, {"prepared_npz_sha256": before, "sides": side_receipts}


def validate_preregister(value: Mapping[str, Any]) -> None:
    if value.get("schema") != PREREGISTER_SCHEMA:
        raise ScanNet15IdentityError("ScanNet15 preregister schema mismatch")
    _sealed(value, "ScanNet15 preregister")
    if value.get("pair_count") != PAIR_COUNT:
        raise ScanNet15IdentityError("ScanNet15 preregister is not exact15")
    pair_ids = value.get("pair_ids")
    rows = value.get("pairs")
    if (not isinstance(pair_ids, list) or not isinstance(rows, list)
            or len(pair_ids) != PAIR_COUNT or len(set(pair_ids)) != PAIR_COUNT
            or [row.get("pair_id") for row in rows
                if isinstance(row, dict)] != pair_ids):
        raise ScanNet15IdentityError("ScanNet15 pair identity/order mismatch")
    if any(pair_id.count("_to_") != 1 for pair_id in pair_ids):
        raise ScanNet15IdentityError("ScanNet15 pair ID is malformed")
    for row in rows:
        scene_id = str(row.get("scene_id", ""))
        if row.get("pair_id") != pair_id_for_scene(scene_id):
            raise ScanNet15IdentityError("ScanNet15 row scene/pair mismatch")
        for field in PAIR_BINDING_PATH_FIELDS:
            path = row.get(field)
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ScanNet15IdentityError(f"pair path is not absolute: {field}")
        for field in PAIR_BINDING_SHA_FIELDS:
            _sha(row.get(field), f"pair {field}")
        _sha(row.get("identity_payload_sha256"), "pair identity payload")
        unsigned = {key: item for key, item in row.items()
                    if key != "identity_payload_sha256"}
        if row["identity_payload_sha256"] != stable_json_sha256(unsigned):
            raise ScanNet15IdentityError("pair identity payload SHA mismatch")
    for field in POLICY_FALSE_FIELDS:
        if value.get(field) is not False:
            raise ScanNet15IdentityError(f"preregister policy is not false: {field}")
    if (value.get("allow_real_pilot") is not False
            or value.get("allow_gpu_pilot") is not False
            or value.get("posthoc_allowed") is not False
            or value.get("algorithm_or_threshold_change") is not False):
        raise ScanNet15IdentityError("preregister execution policy mismatch")
    for field in (
        "raw_inventory_sha256", "prepared_bridge_summary_sha256",
        "official_checkpoint_sha256", "geotransformer_checkpoint_sha256",
        "sgf_model_closure_sha256", "bridge_source_sha256",
        "colorpcr_schema_source_sha256",
    ):
        _sha(value.get(field), f"preregister {field}")
    sources = value.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ScanNet15IdentityError("preregister source closure missing")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ScanNet15IdentityError("preregister source name missing")
        _sha(digest, f"source {name}")


def pair_row(preregister: Mapping[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [row for row in preregister.get("pairs", ())
               if isinstance(row, dict) and row.get("pair_id") == pair_id]
    if len(matches) != 1:
        raise ScanNet15IdentityError("ScanNet15 pair is not uniquely preregistered")
    return dict(matches[0])


def verify_source_closure(preregister: Mapping[str, Any]) -> None:
    paths = preregister.get("source_paths")
    hashes = preregister.get("source_sha256")
    if not isinstance(paths, dict) or set(paths) != set(hashes):
        raise ScanNet15IdentityError("source path/hash keys differ")
    for name, raw_path in paths.items():
        path = Path(str(raw_path)).resolve()
        if not path.is_file() or path.is_symlink() \
                or sha256_file(path) != hashes[name]:
            raise ScanNet15IdentityError(f"source closure mismatch: {name}")


def verify_preflight_closure(
    *, preregister_path: Path, preflight_path: Path, prepared_path: Path,
    pair_id: str,
) -> dict[str, Any]:
    """Verify one pair against the exact15 preregister and preflight closure."""
    preregister_path = Path(preregister_path).resolve()
    preflight_path = Path(preflight_path).resolve()
    prepared_path = Path(prepared_path).resolve()
    preregister = _load_json(preregister_path)
    preflight = _load_json(preflight_path)
    validate_preregister(preregister)
    verify_source_closure(preregister)
    if preflight.get("schema") != PREFLIGHT_SCHEMA:
        raise ScanNet15IdentityError("ScanNet15 preflight schema mismatch")
    _sealed(preflight, "ScanNet15 preflight")
    if (preflight.get("preregister_path") != str(preregister_path)
            or preflight.get("preregister_sha256")
            != sha256_file(preregister_path)
            or preflight.get("pair_ids") != preregister.get("pair_ids")
            or preflight.get("pair_count") != PAIR_COUNT):
        raise ScanNet15IdentityError("preflight/preregister binding mismatch")
    for field in POLICY_FALSE_FIELDS:
        if preflight.get(field) is not False:
            raise ScanNet15IdentityError(f"preflight policy is not false: {field}")
    prereg_row = pair_row(preregister, pair_id)
    matches = [row for row in preflight.get("pairs", ())
               if isinstance(row, dict) and row.get("pair_id") == pair_id]
    if len(matches) != 1 or matches[0] != prereg_row:
        raise ScanNet15IdentityError("preflight pair binding differs from preregister")
    if (Path(prereg_row["prepared_npz_path"]).resolve() != prepared_path
            or not prepared_path.is_file() or prepared_path.is_symlink()
            or sha256_file(prepared_path) != prereg_row["prepared_npz_sha256"]):
        raise ScanNet15IdentityError("prepared NPZ closure mismatch")
    for path_field, sha_field in (
        ("prepared_manifest_path", "prepared_manifest_sha256"),
        ("raw_inventory_path", "raw_inventory_sha256"),
        ("raw_pair_receipt_path", "raw_pair_receipt_sha256"),
        ("source_raw_ply_path", "source_raw_ply_sha256"),
        ("reference_raw_ply_path", "reference_raw_ply_sha256"),
        ("sgf_prediction_path", "sgf_prediction_sha256"),
    ):
        path = Path(prereg_row[path_field]).resolve()
        if not path.is_file() or path.is_symlink() \
                or sha256_file(path) != prereg_row[sha_field]:
            raise ScanNet15IdentityError(f"pair file closure mismatch: {path_field}")
    with np.load(prepared_path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise ScanNet15IdentityError("prepared NPZ manifest_json missing")
        manifest = json.loads(str(data["manifest_json"].item()))
    validate_prepared_manifest(
        manifest, pair_id=pair_id,
        prepared_sha256=prereg_row["prepared_npz_sha256"],
        preregister=preregister)
    return {
        "schema": "v16-b716-scannet15-verified-pair-identity-v1",
        "pair_id": pair_id,
        "preregister_path": str(preregister_path),
        "preregister_sha256": sha256_file(preregister_path),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "prepared_input": str(prepared_path),
        "prepared_input_sha256": prereg_row["prepared_npz_sha256"],
        "identity_payload_sha256": prereg_row["identity_payload_sha256"],
        "execution_authorized": False,
        "gt_consumed": False,
    }
