"""Fail-closed b716 V10/V11 candidate-plan helpers.

Pair ``combos`` are skipped lexically: their evaluation/node-metric fields are
never decoded or consumed.  Only official-release metadata, ``joint_model``,
canonical predicted inputs and existing GeoTransformer entries are admitted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_RELEASE_SHA256 = (
    "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e"
)
OFFICIAL_CHECKPOINT_EPOCH = 6
OFFICIAL_CODE_HEAD = "98df603c53849da4028c4bc86d22b34194c31961"
OFFICIAL_MODEL_CONFIG_SHA256 = (
    "614d386b71e038911f87c85fde6951003e514b18f910f2a010678c8e9fd570d3"
)
SAFE_PAIR_FIELDS = frozenset({
    "pair_id", "mode", "sampling_mode", "scan_seed", "checkpoint_sha256",
    "checkpoint_epoch", "code_head", "model_config", "cache_key", "status",
    "joint_online_offline_consistent", "geot_node_pairs",
    "pair_record_sha256", "requested_structured_completed",
})


class B716PlanError(RuntimeError):
    """Frozen provenance or structural evidence is invalid."""


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def file_evidence(path: Path, role: str) -> dict[str, Any]:
    path = Path(path).resolve()
    before = sha256_file(path)
    row = {"path": str(path), "bytes": int(path.stat().st_size),
           "sha256": before, "role": role}
    if sha256_file(path) != before:
        raise B716PlanError(f"source changed while hashing: {path}")
    return row


def _ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _skip_value(text: str, index: int) -> int:
    """Skip one JSON value without constructing it."""
    index = _ws(text, index)
    if index >= len(text):
        raise B716PlanError("truncated JSON")
    first = text[index]
    if first == '"':
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index + 1
            index += 1
        raise B716PlanError("unterminated JSON string")
    if first in "[{":
        stack = [first]
        index += 1
        in_string = escaped = False
        while index < len(text):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                expected = "[" if char == "]" else "{"
                if not stack or stack.pop() != expected:
                    raise B716PlanError("malformed nested JSON")
                if not stack:
                    return index + 1
            index += 1
        raise B716PlanError("unterminated nested JSON")
    end = index
    while end < len(text) and text[end] not in ",}":
        end += 1
    token = text[index:end].strip()
    if token not in ("true", "false", "null"):
        try:
            float(token)
        except ValueError as exc:
            raise B716PlanError("invalid JSON scalar") from exc
    return end


def safe_pair_metadata(path: Path) -> dict[str, Any]:
    """Decode only safe top-level fields; skip ``combos`` lexically."""
    path = Path(path)
    before = sha256_file(path)
    text = path.read_text()
    decoder = json.JSONDecoder()
    index = _ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise B716PlanError("pair metadata must be an object")
    index += 1
    output: dict[str, Any] = {}
    while True:
        index = _ws(text, index)
        if index < len(text) and text[index] == "}":
            index += 1
            break
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            raise B716PlanError("malformed top-level JSON key") from exc
        index = _ws(text, index)
        if not isinstance(key, str) or index >= len(text) or text[index] != ":":
            raise B716PlanError("malformed top-level JSON member")
        start = _ws(text, index + 1)
        if key in SAFE_PAIR_FIELDS:
            try:
                output[key], index = decoder.raw_decode(text, start)
            except json.JSONDecodeError as exc:
                raise B716PlanError(f"malformed safe field {key}") from exc
        else:
            index = _skip_value(text, start)
        index = _ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            index += 1
            break
        raise B716PlanError("malformed JSON separator")
    if _ws(text, index) != len(text) or sha256_file(path) != before:
        raise B716PlanError("pair metadata trailing bytes or concurrent change")
    return output


def validate_pair_metadata(meta: Mapping[str, Any]) -> None:
    required = SAFE_PAIR_FIELDS - {
        "pair_record_sha256", "requested_structured_completed",
    }
    if not required.issubset(meta):
        raise B716PlanError("safe pair metadata fields missing")
    if meta["checkpoint_sha256"] != OFFICIAL_RELEASE_SHA256:
        raise B716PlanError("rank source is not official release b716")
    if (meta["mode"] != "official_sgf_predicted"
            or meta["sampling_mode"] != "official_mt19937"
            or int(meta["scan_seed"]) != 0 or meta["status"] != "ok"
            or int(meta["checkpoint_epoch"]) != OFFICIAL_CHECKPOINT_EPOCH
            or meta["code_head"] != OFFICIAL_CODE_HEAD
            or meta["joint_online_offline_consistent"] is not True
            or not isinstance(meta["geot_node_pairs"], Mapping)):
        raise B716PlanError("official predicted cache contract mismatch")
    cache_key = meta["cache_key"]
    expected_cache_fields = {
        "pair_id", "input_tensor_sha256", "checkpoint_sha256",
        "sampling_mode", "model_config_sha256", "code_head",
    }
    if (not isinstance(cache_key, Mapping)
            or set(cache_key) != expected_cache_fields
            or cache_key.get("pair_id") != meta["pair_id"]
            or cache_key.get("checkpoint_sha256") != OFFICIAL_RELEASE_SHA256
            or cache_key.get("sampling_mode") != "official_mt19937"
            or cache_key.get("model_config_sha256")
            != OFFICIAL_MODEL_CONFIG_SHA256
            or cache_key.get("code_head") != OFFICIAL_CODE_HEAD
            or not isinstance(cache_key.get("input_tensor_sha256"), str)
            or len(cache_key["input_tensor_sha256"]) != 64
            or any(char not in "0123456789abcdef"
                   for char in cache_key["input_tensor_sha256"])):
        raise B716PlanError("official inner cache_key contract mismatch")


def load_input_tensors(path: Path) -> dict[str, np.ndarray]:
    fields = (
        "tot_obj_pts", "tot_rel_pose", "tot_bow_vec_object_edge_feats",
        "edges", "obj_ids",
    )
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(fields):
            raise B716PlanError("input_tensors.npz fields changed")
        result = {name: np.ascontiguousarray(archive[name]) for name in fields}
    if sha256_file(path) != before:
        raise B716PlanError("input tensors changed while reading")
    return result


def input_tensor_sha256(tensors: Mapping[str, np.ndarray]) -> str:
    fields = (
        "tot_obj_pts", "tot_rel_pose", "tot_bow_vec_object_edge_feats",
        "edges", "obj_ids",
    )
    if set(tensors) != set(fields):
        raise B716PlanError("input tensor semantic-hash fields changed")
    return hashlib.sha256(b"".join(
        np.ascontiguousarray(tensors[name]).tobytes() for name in fields
    )).hexdigest()


def load_joint_model(path: Path, nodes: int) -> np.ndarray:
    before = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        if "joint_model" not in archive.files:
            raise B716PlanError("embeddings lack joint_model")
        joint = np.ascontiguousarray(archive["joint_model"], dtype=np.float32)
    if (joint.shape != (nodes, 300) or not np.isfinite(joint).all()
            or np.any(np.linalg.norm(joint, axis=1) == 0)):
        raise B716PlanError("joint_model shape/value contract mismatch")
    if sha256_file(path) != before:
        raise B716PlanError("embeddings changed while reading")
    return joint


def canonical_boundary(data: Mapping[str, Any],
                       cached: Mapping[str, np.ndarray]) -> dict[str, Any]:
    obj_ids = np.asarray(data.get("obj_ids"))
    counts = np.asarray(data.get("graph_per_obj_count"))
    src_count = int(data.get("src_count", -1))
    if (obj_ids.ndim != 1 or counts.shape != (2,) or np.any(counts <= 0)
            or int(counts.sum()) != len(obj_ids) or src_count != int(counts[0])):
        raise B716PlanError("graph/object boundary is malformed")
    src_map = {int(oid): i for i, oid in enumerate(obj_ids[:src_count])}
    ref_map = {int(oid): i for i, oid in enumerate(obj_ids[src_count:])}
    if data.get("src_object_id2idx") != src_map or data.get("ref_object_id2idx") != ref_map:
        raise B716PlanError("src_count lacks per-side object-table proof")
    surfaces = data.get("registration_pts")
    oid_map = data.get("registration_id2oid")
    expected_oid_map = {i: int(obj_ids[i]) for i in range(len(obj_ids))}
    if (not isinstance(surfaces, Mapping)
            or set(map(int, surfaces)) != set(range(len(obj_ids)))
            or {int(k): int(v) for k, v in oid_map.items()} != expected_oid_map):
        raise B716PlanError("registration surfaces/object rows are not bijective")
    if set(cached) != {"tot_obj_pts", "tot_rel_pose",
                       "tot_bow_vec_object_edge_feats", "edges", "obj_ids"}:
        raise B716PlanError("cached input field set mismatch")
    for name, observed in cached.items():
        expected = np.asarray(data[name])
        if (expected.shape != observed.shape or expected.dtype != observed.dtype
                or not np.array_equal(expected, observed)):
            raise B716PlanError(f"canonical tensor mismatch: {name}")
    return {
        "src_count": src_count, "ref_count": int(counts[1]),
        "total_objects": len(obj_ids),
        "source_object_ids": [int(x) for x in obj_ids[:src_count]],
        "reference_object_ids": [int(x) for x in obj_ids[src_count:]],
        "graph_per_obj_count_sha256": array_sha256(counts),
        "object_ids_sha256": array_sha256(obj_ids),
        "authority": (
            "graph_per_obj_count + src/ref object_id2idx + registration_id2oid; "
            "never inferred from tensor shape"),
    }


def freeze_existing_geot(
    candidates: Sequence[Mapping[str, Any]], meta: Mapping[str, Any],
    geot_path: Path, data: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, int]]:
    """Convert existing cache keys to immutable entries; leave absent keys disabled."""
    before = sha256_file(geot_path)
    with np.load(geot_path, allow_pickle=False) as archive:
        raw_arrays = {name: np.ascontiguousarray(archive[name])
                      for name in archive.files}
    if sha256_file(geot_path) != before:
        raise B716PlanError("GeoT NPZ changed while reading")
    obj_ids = np.asarray(data["obj_ids"])
    rows, arrays = [], {}
    counts = {"candidate_count": len(candidates), "existing_reused": 0,
              "existing_ok": 0, "existing_failed": 0,
              "missing_disabled": 0, "new_geot_executed": 0}
    used_cache_rows: set[int] = set()
    for index, candidate in enumerate(candidates):
        node_pair = (int(candidate["source_index"]),
                     int(candidate["reference_index"]))
        key = f"{node_pair[0]}_{node_pair[1]}"
        if key not in meta:
            counts["missing_disabled"] += 1
            entry = {
                "candidate_index": index, "node_pair": list(node_pair),
                "object_pair": [int(obj_ids[node_pair[0]]),
                                int(obj_ids[node_pair[1]])],
                "origin": "missing_execution_disabled", "immutable": False,
                "status": "disabled_missing_geotransformer",
            }
        else:
            raw = meta[key]
            if not isinstance(raw, Mapping) or "status" not in raw or "cache_row" not in raw:
                raise B716PlanError(f"malformed existing GeoT entry {key}")
            cache_row = int(raw["cache_row"])
            if cache_row in used_cache_rows:
                raise B716PlanError("GeoT cache_row reused by candidate keys")
            used_cache_rows.add(cache_row)
            counts["existing_reused"] += 1
            ok = raw["status"] == "ok"
            counts["existing_ok" if ok else "existing_failed"] += 1
            entry = {
                "candidate_index": index, "node_pair": list(node_pair),
                "object_pair": [int(obj_ids[node_pair[0]]),
                                int(obj_ids[node_pair[1]])],
                "origin": "official_pair_cache", "immutable": True,
                "status": str(raw["status"]), "source_cache_row": cache_row,
                "source_metadata": jsonable(raw),
            }
            names = {field: f"{field}_{cache_row}"
                     for field in ("src_corr", "ref_corr", "scores")}
            present = {field: name in raw_arrays for field, name in names.items()}
            if ok and not all(present.values()):
                raise B716PlanError(f"ok GeoT arrays missing for {key}")
            if not ok and any(present.values()):
                raise B716PlanError(f"failed GeoT entry has arrays for {key}")
            if ok:
                src = np.ascontiguousarray(raw_arrays[names["src_corr"]], dtype=np.float32)
                ref = np.ascontiguousarray(raw_arrays[names["ref_corr"]], dtype=np.float32)
                scores = np.ascontiguousarray(raw_arrays[names["scores"]], dtype=np.float32)
                if (src.ndim != 2 or src.shape[1:] != (3,) or ref.shape != src.shape
                        or scores.shape != (len(src),) or len(src) == 0
                        or not np.isfinite(src).all() or not np.isfinite(ref).all()
                        or not np.isfinite(scores).all()):
                    raise B716PlanError(f"malformed GeoT arrays for {key}")
                for field, value in (("src_corr", src), ("ref_corr", ref),
                                     ("scores", scores)):
                    arrays[f"{field}_{index}"] = value
                    entry[f"{field}_sha256"] = array_sha256(value)
                    entry[f"{field}_shape"] = list(value.shape)
        entry["entry_sha256"] = stable_json_sha256(entry)
        rows.append(entry)
    if counts["existing_reused"] + counts["missing_disabled"] != len(candidates):
        raise B716PlanError("candidate/GeoT accounting mismatch")
    return rows, arrays, counts


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    import io
    import zipfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for key in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{key}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise B716PlanError(
                    f"refusing to overwrite different immutable NPZ {path}")
            return
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise B716PlanError(f"artifact appeared concurrently: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise B716PlanError(f"refusing to overwrite different artifact {path}")
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise B716PlanError(f"artifact appeared concurrently: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
