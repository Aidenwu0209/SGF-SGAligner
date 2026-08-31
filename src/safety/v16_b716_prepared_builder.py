"""Sealed CPU-only b716 matched-region prepared-input builder.

The exact191 manifest is an integrity/coverage authority only.  Candidate
status, GeoTransformer outcomes and scores never select or remove a frozen
hypothesis.  Surface membership comes exclusively from the four b716
candidate structural plans and same-scan raw InSeg instance rows.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np

from safety.v16_b716_candidate_plan import (
    array_sha256 as execution_array_sha256,
)
from safety.v13_colorpcr_pointdsc_shadow import (
    COLORPCR_INPUT_VOXEL_M,
    color_preserving_voxel_aggregate,
)
from safety.v16_matched_region_colorpcr import (
    RawInseg,
    V16ContractError,
    array_sha256,
    build_side_union,
    canonical_surface_from_rows,
    node_object_id,
    sha256_file,
    stable_json_sha256,
    validate_hypothesis,
    verify_file,
)


OFFICIAL_RELEASE_SHA256 = (
    "b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e"
)
EXACT191_SCHEMA = "v16-b716-exact191-merged-manifest-v1"
EXACT191_PAIR_SCHEMA = "v16-b716-exact191-pair-v1"
ALLOWLIST_SCHEMA = "v16-b716-frozen-hypothesis-allowlist-v1"
CANDIDATE_MANIFEST_SCHEMA = "v16-b716-candidate-plan-manifest-v1"
CANDIDATE_PLAN_SCHEMA = "v16-b716-candidate-structural-plan-v1"
PREPARED_SCHEMA = "v16-b716-matched-region-prepared-input-v2"
PAIR_MANIFEST_SCHEMA = "v16-b716-matched-region-prepared-pair-v2"
BUILDER_MANIFEST_SCHEMA = "v16-b716-matched-region-prepared-builder-v2"
ARTIFACT_MANIFEST_SCHEMA = "v16-b716-prepared-artifact-manifest-v2"
PREFLIGHT_SCHEMA = "v16-b716-geot-backfill-preflight-v1"
TASK_SCHEMA = "v16-b716-geot-backfill-task-v1"
AUTH_SCHEMA = "v16-b716-geot-backfill-authorization-v1"
CLEAN_SCHEMA = "v16-b716-clean-service-receipt-v1"
AUTHORIZED_TASK_SCHEMA = "v16-b716-geot-authorized-task-view-v1"
ATTEMPT_SCHEMA = "v16-b716-geot-attempt-receipt-v1"
RESULT_SCHEMA = "v16-b716-geot-backfill-result-v1"
BATCH_SCHEMA = "v16-b716-geot-backfill-batch-result-v1"
ARM = "sgf_selected_union"

FIXED_PAIR_ORDER = (
    "09582205-e2c2-2de1-9475-1cdac7639e60_to_"
    "0958220d-e2c2-2de1-9710-c37018da1883",
    "68bae76c-3567-2f7c-827d-373035a2d942_to_"
    "68bae76e-3567-2f7c-82bd-a09641695364",
    "f38169cf-378c-2a65-855f-05d491a3f26e_to_"
    "f38169c7-378c-2a65-8543-3c7481e856fe",
    "6a36052f-fa53-2915-9400-831b60c63077_to_"
    "6a36052d-fa53-2915-9764-30d81b2cc2b5",
)
EXPECTED_CANDIDATE_COUNTS = (48, 48, 48, 47)
EXPECTED_EXISTING_COUNTS = (46, 27, 27, 19)
EXPECTED_NEW_COUNTS = (2, 21, 21, 28)
EXPECTED_HYPOTHESIS_COUNTS = (12, 8, 2, 12)
EXPECTED_EXISTING_TYPED_FAILURE_COUNTS = (9, 2, 4, 1)
EXPECTED_NEW_TYPED_FAILURE_COUNTS = (0, 4, 5, 3)
EXPECTED_TYPED_FAILURE_COUNTS = tuple(
    existing + new for existing, new in zip(
        EXPECTED_EXISTING_TYPED_FAILURE_COUNTS,
        EXPECTED_NEW_TYPED_FAILURE_COUNTS))
EXPECTED_EXISTING_TYPED_FAILURE_TOTAL = 16
EXPECTED_NEW_TYPED_FAILURE_TOTAL = 12
EXPECTED_TYPED_FAILURE_TOTAL = 28
EXPECTED_EXISTING_TYPED_HYPOTHESES = 8
EXPECTED_ALL_TYPED_HYPOTHESES = 10
ALLOWED_NEW_TYPED_FAILURES = frozenset({"insufficient_post_voxel_points"})
EXPECTED_BY_PAIR = dict(zip(FIXED_PAIR_ORDER, EXPECTED_HYPOTHESIS_COUNTS))
LEGACY_STRING_TOKENS = (
    "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200",
    "b_ep20",
    "sgaligner-sgf-v10-crossgraph-candidates",
    "sgaligner-sgf-v11-matched-region-multiobject",
)
DECLARATION_KEYS = {"forbidden_inputs", "stop_conditions", "forbidden_fields"}
ArtifactEventHook = Callable[[str, Path], None]
EXECUTION_BINDING_SHA_FIELDS = {
    "authorization_sha256", "preregister_sha256",
    "preflight_manifest_sha256", "preflight_payload_sha256",
    "recursive_source_closure_sha256",
    "recursive_artifact_closure_sha256", "task_closure_sha256",
    "immutable_runtime_source_bundle_sha256",
    "runtime_module_entrypoint_closure_sha256",
}


def _deterministic_npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _create_only_publish(
    path: Path, writer: Callable[[Path], None], *,
    event_hook: ArtifactEventHook | None = None,
) -> str:
    """Publish one fsynced artifact atomically without replacing a target.

    A hard-link from a same-directory temporary inode is the POSIX create-only
    primitive: the link either creates the final name or fails with EEXIST.
    This intentionally does not use ``os.replace`` and therefore cannot hide a
    mid-run collision.  The final inode is re-hashed after publication so a
    concurrent modification also fails closed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=".v16-create-only-", dir=path.parent)
    os.close(fd)
    tmp = Path(raw_tmp)
    published = False
    try:
        writer(tmp)
        file_fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        expected_sha256 = sha256_file(tmp)
        if event_hook is not None:
            event_hook("before_publish", path)
        try:
            os.link(tmp, path)
            published = True
        except FileExistsError as exc:
            raise V16ContractError(
                f"create-only artifact collision: {path}") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if event_hook is not None:
            event_hook("after_publish", path)
        if (not path.is_file() or sha256_file(path) != expected_sha256):
            raise V16ContractError(
                f"create-only artifact changed during publication: {path}")
        return expected_sha256
    finally:
        tmp.unlink(missing_ok=True)
        if not published and path.exists() and path.is_symlink():
            # Never clean a foreign collision; this branch only documents that
            # a symlink target is deliberately left untouched for audit.
            pass


def create_only_deterministic_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *,
    event_hook: ArtifactEventHook | None = None,
) -> str:
    def writer(tmp: Path) -> None:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as archive:
            for key in sorted(arrays):
                info = zipfile.ZipInfo(
                    f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _deterministic_npy_bytes(arrays[key]))
    return _create_only_publish(path, writer, event_hook=event_hook)


def create_only_json(
    path: Path, payload: Mapping[str, Any], *,
    event_hook: ArtifactEventHook | None = None,
) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()

    def writer(tmp: Path) -> None:
        with tmp.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return _create_only_publish(path, writer, event_hook=event_hook)


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise V16ContractError(f"invalid {name}")
    return value


def _reject_legacy_string_values(value: Any, path: str = "$") -> None:
    """Reject actual legacy checkpoint/path values without rejecting deny keys."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in DECLARATION_KEYS:
                continue
            _reject_legacy_string_values(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_legacy_string_values(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in LEGACY_STRING_TOKENS):
            raise V16ContractError(f"legacy B/89ed source consumed at {path}")


def _payload_sha(value: Mapping[str, Any], name: str) -> None:
    expected = value.get("payload_sha256")
    if expected is None:
        raise V16ContractError(f"{name} payload SHA missing")
    _sha(expected, f"{name} payload SHA")
    payload = {key: item for key, item in value.items()
               if key != "payload_sha256"}
    if stable_json_sha256(payload) != expected:
        raise V16ContractError(f"{name} payload SHA mismatch")


def load_bound_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    verify_file(path, _sha(expected_sha256, f"{name} file SHA"))
    try:
        value = json.loads(Path(path).read_text())
    except Exception as exc:
        raise V16ContractError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise V16ContractError(f"{name} must be a JSON object")
    _payload_sha(value, name)
    if sha256_file(path) != expected_sha256:
        raise V16ContractError(f"{name} changed while reading")
    return value


def load_sha_json(
    path: Path, expected_sha256: str, name: str, *,
    require_payload: bool = False,
) -> dict[str, Any]:
    verify_file(path, _sha(expected_sha256, f"{name} file SHA"))
    before = sha256_file(path)
    try:
        value = json.loads(Path(path).read_text())
    except Exception as exc:
        raise V16ContractError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict) or sha256_file(path) != before:
        raise V16ContractError(f"{name} changed while reading")
    if require_payload:
        _payload_sha(value, name)
    return value


def _nonzero_sha(value: Any, name: str) -> str:
    digest = _sha(value, name)
    if digest == "0" * 64:
        raise V16ContractError(f"zero placeholder SHA forbidden: {name}")
    return digest


def _validate_execution_binding(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or set(value) != EXECUTION_BINDING_SHA_FIELDS | {"cuda_device_uuid"}
            or not isinstance(value.get("cuda_device_uuid"), str)
            or not value["cuda_device_uuid"]):
        raise V16ContractError("exact191 execution binding malformed")
    for field in EXECUTION_BINDING_SHA_FIELDS:
        _nonzero_sha(value.get(field), f"execution binding {field}")
    return dict(value)


def _verify_closure_rows(
    rows: Any, expected_sha256: Any, name: str, *,
    root: Path | None = None, require_files: bool = True,
    allow_empty_files: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise V16ContractError(f"{name} is absent")
    if stable_json_sha256(rows) != _nonzero_sha(
            expected_sha256, f"{name} closure SHA"):
        raise V16ContractError(f"{name} closure SHA mismatch")
    output = []
    identities = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise V16ContractError(f"{name} row malformed")
        raw = row.get("path")
        if not isinstance(raw, str) or not raw:
            raise V16ContractError(f"{name} row path missing")
        path = Path(raw)
        if not path.is_absolute():
            if root is None:
                raise V16ContractError(f"{name} relative path lacks root")
            path = Path(root) / path
        path = path.resolve()
        digest = _nonzero_sha(row.get("sha256"), f"{name} row SHA")
        size = row.get("bytes")
        minimum_size = 0 if allow_empty_files else 1
        if (type(size) is not int or size < minimum_size
                or (require_files and (not path.is_file()
                    or path.stat().st_size != size or sha256_file(path) != digest))):
            raise V16ContractError(f"{name} file bytes/SHA mismatch")
        identity = (str(path), str(row.get("role", "")))
        if identity in identities:
            raise V16ContractError(f"{name} duplicate row")
        identities.add(identity)
        output.append({**dict(row), "_resolved_path": str(path)})
    return output


def _bound_path(root: Path, row: Mapping[str, Any], prefix: str) -> Path:
    raw = row.get(f"{prefix}_path")
    if not isinstance(raw, str) or not raw:
        raise V16ContractError(f"exact191 {prefix} path missing")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    expected_bytes = row.get(f"{prefix}_bytes")
    if not isinstance(expected_bytes, int) or expected_bytes < 1:
        raise V16ContractError(f"exact191 {prefix} byte count invalid")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise V16ContractError(f"exact191 {prefix} bytes mismatch")
    verify_file(path, _sha(row.get(f"{prefix}_sha256"),
                           f"exact191 {prefix} SHA"))
    return path


def _candidate_index_map(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index, row in enumerate(records):
        key = (int(row["source_index"]), int(row["reference_index"]))
        if key in result:
            raise V16ContractError("candidate plan has duplicate node pair")
        result[key] = index
    return result


def _validate_raw_surface_bindings(
    plan: Mapping[str, Any], data: Mapping[str, Any],
    source: RawInseg, reference: RawInseg,
) -> None:
    rows = plan.get("canonical_surface_bindings")
    obj_ids = np.asarray(data.get("obj_ids"))
    if not isinstance(rows, list) or len(rows) != len(obj_ids):
        raise V16ContractError("candidate-plan raw surface bindings missing")
    by_node = {row.get("node_index"): row for row in rows}
    if len(by_node) != len(rows) or set(by_node) != set(range(len(obj_ids))):
        raise V16ContractError("candidate-plan raw surface node identities mismatch")
    src_count = int(data.get("src_count", 0))
    for node in range(len(obj_ids)):
        side = "source" if node < src_count else "reference"
        raw = source if side == "source" else reference
        row = by_node[node]
        object_id = node_object_id(data, node, side=side)
        indices, reconstructed = canonical_surface_from_rows(raw, object_id)
        expected_surface = np.ascontiguousarray(
            np.asarray(data["registration_pts"][node], np.float64))
        observed = {
            "side": side,
            "scan_id": raw.scan_id,
            "object_id": object_id,
            "raw_inseg_path": str(raw.path),
            "raw_inseg_sha256": raw.file_sha256,
            "raw_row_count": len(indices),
            "raw_row_indices_sha256": array_sha256(indices),
            "canonical_registration_surface_sha256": array_sha256(expected_surface),
            "canonical_registration_points": len(reconstructed),
        }
        if any(row.get(key) != value for key, value in observed.items()):
            raise V16ContractError("candidate-plan raw surface binding mismatch")
        if (expected_surface.shape != reconstructed.shape
                or not np.array_equal(expected_surface, reconstructed)):
            raise V16ContractError("candidate-plan canonical surface changed")


def _validate_allowlist(
    value: Mapping[str, Any], *, pair_id: str,
    hypotheses: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    exact_entries: Sequence[Mapping[str, Any]],
) -> None:
    if (value.get("schema") != ALLOWLIST_SCHEMA
            or value.get("pair_id") != pair_id
            or value.get("all_hypotheses_must_be_replayed") is not True
            or value.get("typed_failure_members_visible_and_never_filtered")
            is not True):
        raise V16ContractError("exact191 hypothesis allowlist contract mismatch")
    rows = value.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != len(hypotheses):
        raise V16ContractError("exact191 hypothesis allowlist count mismatch")
    candidate_indices = _candidate_index_map(candidate_records)
    expected = []
    for hypothesis in hypotheses:
        records = validate_hypothesis(hypothesis, candidate_records)
        indices = [candidate_indices[(int(row["source_index"]),
                                      int(row["reference_index"]))]
                   for row in records]
        expected.append({
            "hypothesis_index": int(hypothesis["hypothesis_index"]),
            "hypothesis_sha256": str(hypothesis["hypothesis_sha256"]),
            "member_candidate_indices": indices,
        })
    observed = [{
        "hypothesis_index": row.get("hypothesis_index"),
        "hypothesis_sha256": row.get("hypothesis_sha256"),
        "member_candidate_indices": row.get("member_candidate_indices"),
    } for row in rows]
    if observed != expected:
        raise V16ContractError("exact191 allowlist differs from frozen hypotheses")
    for row in rows:
        unsigned = dict(row); digest = unsigned.pop("allowlist_entry_sha256", None)
        typed = row.get("typed_failure_member_candidate_indices")
        existing_typed = row.get(
            "existing_typed_failure_member_candidate_indices")
        new_typed = row.get("new_typed_failure_member_candidate_indices")
        expected_existing = [
            index for index in row["member_candidate_indices"]
            if exact_entries[index].get("origin", {}).get("kind")
            == "frozen_existing"
            and exact_entries[index].get("status") != "ok"]
        expected_new = [
            index for index in row["member_candidate_indices"]
            if exact_entries[index].get("origin", {}).get("kind")
            == "authorized_backfill"
            and exact_entries[index].get("status") != "ok"]
        expected_typed = expected_existing + expected_new
        if (not isinstance(typed, list)
                or not isinstance(existing_typed, list)
                or not isinstance(new_typed, list)
                or typed != expected_typed
                or existing_typed != expected_existing
                or new_typed != expected_new
                or row.get("all_members_ok") != (not expected_typed)
                or row.get("contains_typed_failure_members") != bool(typed)
                or digest != stable_json_sha256(unsigned)):
            raise V16ContractError("exact191 typed-failure allowlist mismatch")
    if value.get("hypotheses_with_typed_failure_members") != sum(
            bool(row["typed_failure_member_candidate_indices"]) for row in rows):
        raise V16ContractError("exact191 typed-failure hypothesis count mismatch")


def _validate_exact_pair_entries(
    value: Mapping[str, Any], *, pair_id: str, candidate_count: int,
) -> None:
    rows = value.get("entries")
    if (value.get("schema") != EXACT191_PAIR_SCHEMA
            or value.get("pair_id") != pair_id
            or value.get("candidate_count") != candidate_count
            or not isinstance(rows, list) or len(rows) != candidate_count):
        raise V16ContractError("exact191 pair-entry contract mismatch")
    if [row.get("candidate_index") for row in rows] != list(range(candidate_count)):
        raise V16ContractError("exact191 candidate order is not exact")
    node_pairs = [tuple(row.get("node_pair", ())) for row in rows]
    if len(set(node_pairs)) != candidate_count:
        raise V16ContractError("exact191 candidate node-pair identity is duplicated")


def _input_by_role(
    rows: Sequence[Mapping[str, Any]], role: str,
) -> tuple[Path, str]:
    matches = [row for row in rows if row.get("role") == role]
    if len(matches) != 1:
        raise V16ContractError(f"exact191 input role is not unique: {role}")
    return Path(str(matches[0]["_resolved_path"])), str(matches[0]["sha256"])


def _validate_result_npz(path: Path, row: Mapping[str, Any]) -> None:
    if (row.get("path") != "correspondences.npz"
            or type(row.get("bytes")) is not int
            or not path.is_file() or path.stat().st_size != row["bytes"]
            or sha256_file(path) != _nonzero_sha(
                row.get("sha256"), "result NPZ SHA")):
        raise V16ContractError("exact191 result NPZ lineage mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"src_corr", "ref_corr", "scores"}:
                raise V16ContractError("exact191 result NPZ schema mismatch")
            arrays = {name: np.ascontiguousarray(archive[name])
                      for name in archive.files}
    except (OSError, ValueError) as exc:
        raise V16ContractError("exact191 result NPZ cannot be decoded") from exc
    if (arrays["src_corr"].dtype != np.float32
            or arrays["src_corr"].ndim != 2
            or arrays["src_corr"].shape[1:] != (3,)
            or arrays["ref_corr"].dtype != np.float32
            or arrays["ref_corr"].shape != arrays["src_corr"].shape
            or arrays["scores"].dtype != np.float32
            or arrays["scores"].shape != (len(arrays["src_corr"]),)
            or len(arrays["src_corr"]) == 0
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise V16ContractError("exact191 result NPZ array contract mismatch")
    metadata = row.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(arrays):
        raise V16ContractError("exact191 result NPZ metadata missing")
    for name, value in arrays.items():
        item = metadata[name]
        if (not isinstance(item, Mapping)
                or set(item) != {"shape", "dtype", "sha256"}
                or item.get("shape") != list(value.shape)
                or item.get("dtype") != str(value.dtype)
                or item.get("sha256") != execution_array_sha256(value)):
            raise V16ContractError("exact191 result NPZ array SHA mismatch")


def _validate_hardened_execution_closure(
    exact: Mapping[str, Any], *, candidate_manifest_sha256: str,
) -> dict[str, Any]:
    """Revalidate the sealed exact72 execution lineage without selecting it."""
    binding = _validate_execution_binding(exact.get("execution_binding"))
    input_rows = _verify_closure_rows(
        exact.get("input_closure"), exact.get("recursive_input_closure_sha256"),
        "exact191 input closure")
    expected_roles = {
        "frozen_candidate_manifest", "authorized_preflight_manifest",
        "authorized_preregistration", "execution_authorization",
        "exact72_batch_result",
    }
    if {row.get("role") for row in input_rows} != expected_roles:
        raise V16ContractError("exact191 input closure role set mismatch")
    candidate_path, candidate_sha = _input_by_role(
        input_rows, "frozen_candidate_manifest")
    prereg_path, prereg_sha = _input_by_role(
        input_rows, "authorized_preregistration")
    preflight_path, preflight_sha = _input_by_role(
        input_rows, "authorized_preflight_manifest")
    authorization_path, authorization_sha = _input_by_role(
        input_rows, "execution_authorization")
    batch_path, batch_sha = _input_by_role(input_rows, "exact72_batch_result")
    if (candidate_sha != candidate_manifest_sha256
            or candidate_sha != exact.get("candidate_manifest_sha256")
            or prereg_sha != exact.get("preregister_sha256")
            or preflight_sha != exact.get("preflight_manifest_sha256")
            or authorization_sha != exact.get("authorization_sha256")
            or batch_sha != exact.get("batch_result_sha256")
            or binding["authorization_sha256"] != authorization_sha
            or binding["preregister_sha256"] != prereg_sha
            or binding["preflight_manifest_sha256"] != preflight_sha):
        raise V16ContractError("exact191 input/execution SHA binding mismatch")

    # The candidate is already validated through the caller's supplied path and
    # SHA; this second path equality prevents an alternate same-SHA role.
    if sha256_file(candidate_path) != candidate_manifest_sha256:
        raise V16ContractError("exact191 input candidate changed")
    prereg = load_sha_json(prereg_path, prereg_sha, "exact191 preregistration")
    preflight = load_sha_json(
        preflight_path, preflight_sha, "exact191 preflight", require_payload=True)
    authorization = load_sha_json(
        authorization_path, authorization_sha, "exact191 authorization")
    batch = load_sha_json(
        batch_path, batch_sha, "exact191 batch", require_payload=True)
    if (preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("frozen") is not True
            or preflight.get("disabled") is not False
            or preflight.get("exact_batch_only") is not True
            or preflight.get("key_selection_allowed") is not False
            or preflight.get("result_based_selection_allowed") is not False
            or preflight.get("official92_executed") is not False
            or preflight.get("task_count") != 72
            or preflight.get("missing_key_count") != 72
            or prereg.get("disabled") is not False
            or prereg.get("execution_contract", {}).get(
                "real_execution_allowed") is not True):
        raise V16ContractError("exact191 authorized preflight contract mismatch")
    for field in (
        "preflight_payload_sha256", "recursive_source_closure_sha256",
        "recursive_artifact_closure_sha256", "task_closure_sha256",
        "immutable_runtime_source_bundle_sha256",
        "runtime_module_entrypoint_closure_sha256",
    ):
        preflight_field = "payload_sha256" if field == \
            "preflight_payload_sha256" else field
        if binding[field] != preflight.get(preflight_field):
            raise V16ContractError(f"exact191 preflight binding mismatch: {field}")

    source_rows = _verify_closure_rows(
        preflight.get("source_closure"),
        preflight.get("recursive_source_closure_sha256"),
        "exact191 preflight source closure", allow_empty_files=True)
    runtime_rows = sorted([
        {key: value for key, value in row.items() if key != "_resolved_path"}
        for row in source_rows if str(row.get("role", "")).startswith(
            "immutable_runtime_source_bundle:")
    ], key=lambda row: (row["path"], row["role"]))
    if (not runtime_rows or preflight.get("runtime_source_bundle") != runtime_rows
            or stable_json_sha256(runtime_rows)
            != binding["immutable_runtime_source_bundle_sha256"]):
        raise V16ContractError("exact191 immutable runtime source closure mismatch")
    modules = preflight.get("runtime_module_entrypoints")
    runtime_identities = {(row["path"], row["bytes"], row["sha256"])
                          for row in runtime_rows}
    if (not isinstance(modules, list) or not modules
            or stable_json_sha256(modules)
            != binding["runtime_module_entrypoint_closure_sha256"]
            or len({row.get("module") for row in modules}) != len(modules)
            or any((row.get("path"), row.get("bytes"), row.get("sha256"))
                   not in runtime_identities for row in modules)):
        raise V16ContractError("exact191 runtime module entrypoint closure mismatch")

    task_rows = preflight.get("tasks")
    if (not isinstance(task_rows, list) or len(task_rows) != 72
            or stable_json_sha256(task_rows) != binding["task_closure_sha256"]):
        raise V16ContractError("exact191 task closure mismatch")
    artifact_rows = _verify_closure_rows(
        preflight.get("artifact_closure"),
        preflight.get("recursive_artifact_closure_sha256"),
        "exact191 preflight artifact closure", root=preflight_path.parent)
    task_artifacts = sorted((row["path"], row["bytes"], row["sha256"])
                            for row in task_rows)
    observed_artifacts = sorted((row["path"], row["bytes"], row["sha256"])
                                for row in artifact_rows)
    if task_artifacts != observed_artifacts:
        raise V16ContractError("exact191 task/artifact closure mismatch")

    if (authorization.get("schema") != AUTH_SCHEMA
            or authorization.get("authorized") is not True
            or authorization.get("candidate_manifest_sha256")
            != candidate_manifest_sha256
            or authorization.get("exact_batch_count") != 72
            or authorization.get("key_selection_allowed") is not False
            or authorization.get("result_selection_allowed") is not False
            or authorization.get("gt_allowed") is not False
            or authorization.get("official92_allowed") is not False
            or any(authorization.get(field) != value
                   for field, value in binding.items()
                   if field != "authorization_sha256")):
        raise V16ContractError("exact191 execution authorization mismatch")
    clean_path = Path(str(authorization.get("clean_service_receipt_path", "")))
    clean_sha = _nonzero_sha(
        authorization.get("clean_service_receipt_sha256"),
        "clean-service receipt SHA")
    clean = load_sha_json(
        clean_path, clean_sha, "clean-service receipt", require_payload=True)
    cuda_gate = prereg.get("cuda_hard_gate")
    if not isinstance(cuda_gate, Mapping):
        raise V16ContractError("exact191 CUDA gate is absent")
    baseline_service = cuda_gate.get("baseline_service_identity")
    if (clean.get("schema") != CLEAN_SCHEMA
            or clean.get("clean") is not True
            or clean.get("cuda_device_uuid") != binding["cuda_device_uuid"]
            or clean.get("services_checked")
            != cuda_gate.get("required_services_checked")
            or clean.get("compute_process_count")
            != int(cuda_gate.get("baseline_service_process_count", 0))
            or clean.get("baseline_service_identity") != baseline_service
            or clean.get("baseline_service_identity_sha256")
            != (baseline_service or {}).get("identity_sha256")
            or not isinstance(clean.get("cuda_snapshot"), Mapping)
            or clean.get("cuda_snapshot_sha256")
            != stable_json_sha256(clean["cuda_snapshot"])):
        raise V16ContractError("clean-service/CUDA lineage mismatch")

    if (batch.get("schema") != BATCH_SCHEMA
            or batch.get("exact_batch_count") != 72
            or batch.get("selector_eligible") is not False
            or batch.get("result_based_selection_allowed") is not False
            or batch.get("execution_binding") != binding):
        raise V16ContractError("exact191 batch execution binding mismatch")
    batch_rows = batch.get("results")
    if not isinstance(batch_rows, list) or len(batch_rows) != 72:
        raise V16ContractError("exact191 batch rows are not exact72")
    attempt_closure = [{
        "task_id": row.get("task_id"),
        "attempt_receipt_sha256": row.get("attempt_receipt_sha256"),
    } for row in batch_rows]
    if stable_json_sha256(attempt_closure) != _nonzero_sha(
            batch.get("attempt_receipt_closure_sha256"),
            "attempt receipt closure SHA"):
        raise V16ContractError("exact191 attempt receipt closure mismatch")
    result_closure = exact.get("new_result_closure")
    if (not isinstance(result_closure, list) or len(result_closure) != 72
            or stable_json_sha256(result_closure) != _nonzero_sha(
                exact.get("recursive_new_result_closure_sha256"),
                "new result closure SHA")):
        raise V16ContractError("exact191 new result closure mismatch")
    closure_by_task = {}
    for row in result_closure:
        task_id = (f"{row.get('short_id')}__{row.get('node_pair', [None, None])[0]}_"
                   f"{row.get('node_pair', [None, None])[1]}")
        if task_id in closure_by_task:
            raise V16ContractError("exact191 duplicate new-result task")
        closure_by_task[task_id] = row

    expected_ids = []
    new_ok_count = 0
    new_typed_failure_count = 0
    for task_row, batch_row in zip(task_rows, batch_rows):
        if set(batch_row) != {
                "task_id", "status", "resumed", "attempt_receipt_sha256",
                "result_sha256"} or type(batch_row.get("resumed")) is not bool:
            raise V16ContractError("exact191 batch row field set mismatch")
        task_path = (preflight_path.parent / str(task_row.get("path", ""))).resolve()
        task = load_sha_json(
            task_path, _nonzero_sha(task_row.get("sha256"), "task file SHA"),
            "exact191 task")
        task_payload = {key: value for key, value in task.items()
                        if key != "task_sha256"}
        task_id = (f"{task.get('short_id')}__{task.get('node_pair', [None, None])[0]}_"
                   f"{task.get('node_pair', [None, None])[1]}")
        expected_ids.append(task_id)
        if (task.get("schema") != TASK_SCHEMA
                or task.get("state") != "planned_disabled"
                or task.get("execution_authorized") is not False
                or task.get("task_sha256") != stable_json_sha256(task_payload)
                or task.get("task_sha256") != task_row.get("task_sha256")
                or task_id != task_row.get("task_id")
                or task_id != batch_row.get("task_id")):
            raise V16ContractError("exact191 task identity/payload mismatch")
        closure = closure_by_task.get(task_id)
        if not isinstance(closure, Mapping):
            raise V16ContractError("exact191 task absent from result closure")
        task_dir = task_path.parent
        view_path = task_dir / "authorized_task_view.json"
        attempt_path = task_dir / "attempt_receipt.json"
        result_path = task_dir / "result.json"
        view_sha = _nonzero_sha(
            closure.get("authorized_task_view_sha256"), "task-view SHA")
        attempt_sha = _nonzero_sha(
            batch_row.get("attempt_receipt_sha256"), "attempt SHA")
        result_sha = _nonzero_sha(batch_row.get("result_sha256"), "result SHA")
        view = load_sha_json(
            view_path, view_sha, "authorized task view", require_payload=True)
        attempt = load_sha_json(
            attempt_path, attempt_sha, "attempt receipt", require_payload=True)
        result = load_sha_json(
            result_path, result_sha, "result receipt", require_payload=True)
        snapshot = attempt.get("cuda_snapshot")
        status = result.get("status")
        if (view.get("schema") != AUTHORIZED_TASK_SCHEMA
                or view.get("state") != "authorized_pending"
                or view.get("execution_authorized") is not True
                or view.get("planned_task_immutable") is not True
                or view.get("planned_task_sha256") != task["task_sha256"]
                or view.get("execution_binding") != binding
                or attempt.get("schema") != ATTEMPT_SCHEMA
                or attempt.get("task_sha256") != task["task_sha256"]
                or attempt.get("authorized_task_view_sha256") != view_sha
                or any(attempt.get(field) != value
                       for field, value in binding.items())
                or not isinstance(snapshot, Mapping)
                or snapshot.get("uuid") != binding["cuda_device_uuid"]
                or attempt.get("cuda_snapshot_sha256")
                != stable_json_sha256(snapshot)
                or result.get("schema") != RESULT_SCHEMA
                or result.get("task_sha256") != task["task_sha256"]
                or status not in ({"ok"} | ALLOWED_NEW_TYPED_FAILURES)
                or result.get("selector_eligible") is not False
                or result.get("attempt_receipt_sha256") != attempt_sha
                or result.get("authorized_task_view_sha256") != view_sha
                or any(result.get(field) != value
                       for field, value in binding.items())):
            raise V16ContractError("exact191 task/attempt/result execution mismatch")
        if status == "ok":
            corr = result.get("correspondences")
            if not isinstance(corr, Mapping) or "failure" in result:
                raise V16ContractError(
                    "exact191 result correspondence lineage missing")
            _validate_result_npz(task_dir / "correspondences.npz", corr)
            correspondence_sha256 = corr.get("sha256")
            new_ok_count += 1
        else:
            failure = result.get("failure")
            if (not isinstance(failure, Mapping)
                    or set(failure) != {"status", "detail"}
                    or failure.get("status") != status
                    or not isinstance(failure.get("detail"), Mapping)
                    or "correspondences" in result
                    or (task_dir / "correspondences.npz").exists()):
                raise V16ContractError(
                    "exact191 typed new failure evidence is malformed")
            correspondence_sha256 = None
            new_typed_failure_count += 1
        if (batch_row.get("status") != status
                or closure.get("task_sha256") != task["task_sha256"]
                or closure.get("authorized_task_view_sha256") != view_sha
                or closure.get("attempt_sha256") != attempt_sha
                or closure.get("result_sha256") != result_sha
                or closure.get("correspondence_sha256")
                != correspondence_sha256
                or closure.get("authorization_sha256") != authorization_sha):
            raise V16ContractError("exact191 new-result lineage mismatch")
    if [row.get("task_id") for row in batch_rows] != expected_ids:
        raise V16ContractError("exact191 exact72 task order mismatch")
    if (new_ok_count != 60
            or new_typed_failure_count != EXPECTED_NEW_TYPED_FAILURE_TOTAL):
        raise V16ContractError("exact191 typed new-result accounting mismatch")
    return {
        "execution_binding": binding,
        "preflight_payload_sha256": preflight["payload_sha256"],
        "attempt_receipt_closure_sha256":
            batch["attempt_receipt_closure_sha256"],
        "clean_service_receipt_sha256": clean_sha,
        "new_result_count": 72,
        "new_ok_count": new_ok_count,
        "new_typed_failure_count": new_typed_failure_count,
    }


def validate_candidate_and_exact191(
    *, candidate_manifest_path: Path, candidate_manifest_sha256: str,
    exact191_manifest_path: Path, exact191_manifest_sha256: str,
    allow_test_fixture: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the complete b716 plan/exact191 closure without selecting outcomes."""
    candidate = load_bound_json(
        candidate_manifest_path, candidate_manifest_sha256, "candidate manifest")
    exact = load_bound_json(
        exact191_manifest_path, exact191_manifest_sha256, "exact191 manifest")
    _reject_legacy_string_values(candidate)
    _reject_legacy_string_values(exact)
    if (candidate.get("schema") != CANDIDATE_MANIFEST_SCHEMA
            or candidate.get("official_release_domain_matched") is not True
            or candidate.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or candidate.get("legacy_B_ep20_or_89ed_consumed") is not False
            or candidate.get("candidate_count") != 191
            or candidate.get("hypothesis_count") != 34
            or candidate.get("pair_count") != 4
            or candidate.get("official92_executed") is not False):
        raise V16ContractError("candidate manifest is not frozen b716 fixed4")
    fixture = exact.get("synthetic_test_fixture") is True
    if fixture and not allow_test_fixture:
        raise V16ContractError("synthetic exact191 fixture requires explicit test opt-in")
    if (exact.get("schema") != EXACT191_SCHEMA
            or exact.get("sealed") is not True
            or exact.get("candidate_count") != 191
            or exact.get("existing_count") != 119
            or exact.get("new_authorized_count") != 72
            or exact.get("new_authorized_ok_count") != 60
            or exact.get("new_authorized_typed_failure_count")
            != EXPECTED_NEW_TYPED_FAILURE_TOTAL
            or exact.get("typed_failure_existing_count") != 16
            or exact.get("typed_failure_total_count")
            != EXPECTED_TYPED_FAILURE_TOTAL
            or exact.get("hypothesis_count") != 34
            or exact.get("hypotheses_with_typed_failure_members")
            != EXPECTED_ALL_TYPED_HYPOTHESES
            or exact.get("hypotheses_with_existing_typed_failure_members")
            != EXPECTED_EXISTING_TYPED_HYPOTHESES
            or exact.get("typed_failures_visible_and_never_filtered") is not True
            or exact.get("consumer_scope")
            != "only_the_34_frozen_hypotheses_across_fixed4"
            or exact.get("candidate_selection_allowed") is not False
            or exact.get("result_based_selection_allowed") is not False
            or exact.get("hypothesis_selection_allowed") is not False
            or exact.get("gt_allowed") is not False
            or exact.get("official92_allowed") is not False
            or exact.get("new_geot_execution_performed_by_merger") is not False
            or exact.get("b716_domain_only") is not True
            or exact.get("legacy_B_ep20_or_89ed_consumed") is not False
            or exact.get("official_release_checkpoint_sha256")
            != OFFICIAL_RELEASE_SHA256
            or exact.get("candidate_manifest_sha256") != candidate_manifest_sha256
            or exact.get("fixed_hypothesis_distribution")
            != list(EXPECTED_HYPOTHESIS_COUNTS)):
        raise V16ContractError("exact191 sealed manifest contract mismatch")
    candidate_rows = candidate.get("pairs")
    exact_rows = exact.get("pairs")
    if (not isinstance(candidate_rows, list) or not isinstance(exact_rows, list)
            or len(candidate_rows) != 4 or len(exact_rows) != 4):
        raise V16ContractError("fixed4 pair rows missing")
    if [row.get("pair_id") for row in candidate_rows] != list(FIXED_PAIR_ORDER) \
            or [row.get("pair_id") for row in exact_rows] != list(FIXED_PAIR_ORDER):
        raise V16ContractError("fixed4 pair order mismatch")

    candidate_root = Path(candidate_manifest_path).resolve().parent
    exact_root = Path(exact191_manifest_path).resolve().parent
    hardened_execution = _validate_hardened_execution_closure(
        exact, candidate_manifest_sha256=candidate_manifest_sha256)
    artifact_rows = _verify_closure_rows(
        exact.get("artifact_closure"),
        exact.get("recursive_artifact_closure_sha256"),
        "exact191 artifact closure", root=exact_root)
    existing_rows = exact.get("existing_entry_closure")
    if (not isinstance(existing_rows, list) or len(existing_rows) != 119
            or stable_json_sha256(existing_rows) != _nonzero_sha(
                exact.get("recursive_existing_entry_closure_sha256"),
                "existing entry closure SHA")):
        raise V16ContractError("exact191 existing-entry closure mismatch")
    expected_artifacts = sorted([
        (str(row[f"{prefix}_path"]), int(row[f"{prefix}_bytes"]),
         str(row[f"{prefix}_sha256"]))
        for row in exact_rows
        for prefix in ("entries", "correspondences", "allowlist")
    ])
    observed_artifacts = sorted([
        (str(row["path"]), int(row["bytes"]), str(row["sha256"]))
        for row in artifact_rows
    ])
    if expected_artifacts != observed_artifacts:
        raise V16ContractError("exact191 pair/artifact closure mismatch")
    bindings: dict[str, dict[str, Any]] = {}
    ordered_keys, existing_keys, backfill_keys = [], [], []
    observed_existing_typed_failures = 0
    observed_new_typed_failures = 0
    observed_typed_hypotheses = 0
    observed_existing_typed_hypotheses = 0
    for ordinal, (candidate_row, exact_row) in enumerate(zip(candidate_rows, exact_rows)):
        pair_id = FIXED_PAIR_ORDER[ordinal]
        expected = {
            "candidate_count": EXPECTED_CANDIDATE_COUNTS[ordinal],
            "existing_count": EXPECTED_EXISTING_COUNTS[ordinal],
            "new_count": EXPECTED_NEW_COUNTS[ordinal],
            "hypothesis_count": EXPECTED_HYPOTHESIS_COUNTS[ordinal],
        }
        for key, value in expected.items():
            row_key = key if key in exact_row else (
                "hypothesis_count" if key == "hypothesis_count" else key)
            if exact_row.get(row_key) != value:
                raise V16ContractError(f"exact191 {pair_id} {key} mismatch")
        if (candidate_row.get("candidate_count") != expected["candidate_count"]
                or candidate_row.get("hypothesis_count") != expected["hypothesis_count"]):
            raise V16ContractError("candidate-plan pair count mismatch")
        plan_path = Path(str(candidate_row.get("plan_path", "")))
        if not plan_path.is_absolute():
            plan_path = candidate_root / plan_path
        plan = load_bound_json(
            plan_path.resolve(), candidate_row.get("plan_sha256"),
            f"candidate plan {pair_id}")
        _reject_legacy_string_values(plan)
        if (plan.get("schema") != CANDIDATE_PLAN_SCHEMA
                or plan.get("pair_id") != pair_id
                or plan.get("short_id") != candidate_row.get("short_id")
                or plan.get("domain", {}).get("checkpoint_sha256")
                != OFFICIAL_RELEASE_SHA256
                or plan.get("domain", {}).get("matched") is not True
                or plan.get("domain", {}).get("legacy_B_ep20_or_89ed_consumed") is not False
                or plan.get("candidate_count") != expected["candidate_count"]
                or plan.get("hypothesis_count") != expected["hypothesis_count"]):
            raise V16ContractError("candidate structural plan b716 contract mismatch")
        hypotheses = plan.get("hypotheses")
        records = plan.get("candidate_rank_records")
        if (not isinstance(hypotheses, list) or not isinstance(records, list)
                or [row.get("hypothesis_index") for row in hypotheses]
                != list(range(expected["hypothesis_count"]))):
            raise V16ContractError("candidate hypothesis order/count mismatch")
        for hypothesis in hypotheses:
            validate_hypothesis(hypothesis, records)
        geot_entries = plan.get("geot_entries")
        if not isinstance(geot_entries, list) or len(geot_entries) != len(records):
            raise V16ContractError("candidate ordered-key ledger missing")
        for index, entry in enumerate(geot_entries):
            key = {"short_id": plan["short_id"], "candidate_index": index,
                   "node_pair": entry.get("node_pair")}
            if entry.get("candidate_index") != index:
                raise V16ContractError("candidate ordered-key ledger changed")
            ordered_keys.append(key)
            if entry.get("origin") == "official_pair_cache":
                existing_keys.append(key)
            elif entry.get("origin") == "missing_execution_disabled":
                backfill_keys.append(key)
            else:
                raise V16ContractError("candidate ordered-key origin changed")
        entries_path = _bound_path(exact_root, exact_row, "entries")
        correspondences_path = _bound_path(exact_root, exact_row, "correspondences")
        allowlist_path = _bound_path(exact_root, exact_row, "allowlist")
        entries = load_bound_json(
            entries_path, exact_row["entries_sha256"], f"exact191 entries {pair_id}")
        allowlist = load_bound_json(
            allowlist_path, exact_row["allowlist_sha256"], f"exact191 allowlist {pair_id}")
        _validate_exact_pair_entries(
            entries, pair_id=pair_id, candidate_count=expected["candidate_count"])
        pair_existing_typed_failures = sum(
            row.get("origin", {}).get("kind") == "frozen_existing"
            and row.get("status") != "ok" for row in entries["entries"])
        pair_new_typed_failures = sum(
            row.get("origin", {}).get("kind") == "authorized_backfill"
            and row.get("status") != "ok" for row in entries["entries"])
        pair_typed_failures = (
            pair_existing_typed_failures + pair_new_typed_failures)
        observed_existing_typed_failures += pair_existing_typed_failures
        observed_new_typed_failures += pair_new_typed_failures
        _validate_allowlist(
            allowlist, pair_id=pair_id, hypotheses=hypotheses,
            candidate_records=records, exact_entries=entries["entries"])
        pair_typed_hypotheses = int(
            allowlist["hypotheses_with_typed_failure_members"])
        pair_existing_typed_hypotheses = sum(
            bool(row["existing_typed_failure_member_candidate_indices"])
            for row in allowlist["hypotheses"])
        observed_typed_hypotheses += pair_typed_hypotheses
        observed_existing_typed_hypotheses += (
            pair_existing_typed_hypotheses)
        if (pair_existing_typed_failures
                != exact_row.get("existing_failed_count")
                or pair_new_typed_failures
                != exact_row.get("new_typed_failure_count")
                or pair_new_typed_failures
                != EXPECTED_NEW_TYPED_FAILURE_COUNTS[ordinal]
                or pair_existing_typed_failures
                != EXPECTED_EXISTING_TYPED_FAILURE_COUNTS[ordinal]
                or pair_typed_failures
                != EXPECTED_TYPED_FAILURE_COUNTS[ordinal]
                or pair_typed_hypotheses
                != exact_row.get("hypotheses_with_typed_failure_members")
                or pair_existing_typed_hypotheses
                != exact_row.get(
                    "hypotheses_with_existing_typed_failure_members")):
            raise V16ContractError("exact191 typed-failure pair accounting mismatch")
        # The correspondences archive is SHA/byte-bound but never opened here:
        # GeoT outcomes cannot select matched-region surface membership.
        bindings[pair_id] = {
            "candidate_row": dict(candidate_row),
            "exact_row": dict(exact_row),
            "plan": plan,
            "plan_path": str(plan_path.resolve()),
            "entries_path": str(entries_path),
            "correspondences_path": str(correspondences_path),
            "allowlist_path": str(allowlist_path),
            "allowlist": allowlist,
            "hardened_execution": hardened_execution,
        }
    for field, rows in (
        ("ordered_candidate_key_closure_sha256", ordered_keys),
        ("existing_candidate_key_closure_sha256", existing_keys),
        ("backfill_candidate_key_closure_sha256", backfill_keys),
    ):
        if exact.get(field) != stable_json_sha256(rows):
            raise V16ContractError("exact191 ordered-key closure mismatch")
    if (observed_existing_typed_failures
            != EXPECTED_EXISTING_TYPED_FAILURE_TOTAL
            or observed_new_typed_failures
            != EXPECTED_NEW_TYPED_FAILURE_TOTAL
            or observed_typed_hypotheses != EXPECTED_ALL_TYPED_HYPOTHESES
            or observed_existing_typed_hypotheses
            != EXPECTED_EXISTING_TYPED_HYPOTHESES):
        raise V16ContractError("exact191 typed-failure global accounting mismatch")
    closure_by_candidate = {
        (str(row.get("short_id")), int(row.get("candidate_index", -1))): row
        for row in exact["new_result_closure"]
    }
    consumed = set()
    for pair_id, binding in bindings.items():
        entries = load_bound_json(
            Path(binding["entries_path"]),
            binding["exact_row"]["entries_sha256"],
            f"exact191 entry-origin audit {pair_id}")
        short_id = str(binding["plan"]["short_id"])
        for row in entries["entries"]:
            origin = row.get("origin")
            if not isinstance(origin, Mapping) or origin.get("kind") \
                    != "authorized_backfill":
                continue
            key = (short_id, int(row["candidate_index"]))
            closure = closure_by_candidate.get(key)
            if (not isinstance(closure, Mapping)
                    or origin.get("status") != row.get("status")
                    or closure.get("status") != row.get("status")
                    or any(origin.get(field) != closure.get(closure_field)
                           for field, closure_field in (
                               ("task_sha256", "task_sha256"),
                               ("authorized_task_view_sha256",
                                "authorized_task_view_sha256"),
                               ("attempt_sha256", "attempt_sha256"),
                               ("result_sha256", "result_sha256"),
                               ("correspondence_sha256",
                                "correspondence_sha256"),
                               ("authorization_sha256",
                                "authorization_sha256")))):
                raise V16ContractError(
                    "exact191 authorized-backfill entry lineage mismatch")
            if row.get("status") == "ok":
                if origin.get("failure") is not None:
                    raise V16ContractError(
                        "exact191 ok backfill carries failure evidence")
            elif (row.get("status") not in ALLOWED_NEW_TYPED_FAILURES
                    or origin.get("correspondence_sha256") is not None
                    or not isinstance(origin.get("failure"), Mapping)
                    or origin["failure"].get("status") != row.get("status")):
                raise V16ContractError(
                    "exact191 typed backfill lineage mismatch")
            consumed.add(key)
    if consumed != set(closure_by_candidate) or len(consumed) != 72:
        raise V16ContractError(
            "exact191 authorized-backfill entry closure is not exact72")
    return candidate, exact, bindings


def build_prepared_hypothesis(
    *, pair_id: str, short_id: str, hypothesis: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]], data: Mapping[str, Any],
    source: RawInseg, reference: RawInseg, output_root: Path,
    provenance: Mapping[str, Any],
    typed_failure_metadata: Mapping[str, Any],
    artifact_event_hook: ArtifactEventHook | None = None,
) -> dict[str, Any]:
    """Build one deterministic V13-worker-compatible prepared NPZ."""
    records = validate_hypothesis(hypothesis, candidate_records)
    source_union, source_members = build_side_union(
        records, data, source, side="source")
    reference_union, reference_members = build_side_union(
        records, data, reference, side="reference")
    source_voxel = color_preserving_voxel_aggregate(
        source_union, COLORPCR_INPUT_VOXEL_M)
    reference_voxel = color_preserving_voxel_aggregate(
        reference_union, COLORPCR_INPUT_VOXEL_M)
    stem = str(hypothesis["hypothesis_sha256"])
    relative_base = Path("pairs") / short_id / "hypotheses" / (
        f"h{int(hypothesis['hypothesis_index']):02d}-{stem[:16]}")
    relative_npz = relative_base.with_suffix(".npz")
    relative_evidence = relative_base.with_suffix(".json")
    arrays: dict[str, np.ndarray] = {}
    for side, union, voxel in (
        ("source", source_union, source_voxel),
        ("reference", reference_union, reference_voxel),
    ):
        prefix = f"{ARM}_{side}"
        arrays[f"{prefix}_xyz"] = union["xyz"]
        arrays[f"{prefix}_colors"] = union["colors"]
        arrays[f"{prefix}_labels"] = union["membership_object_ids"]
        arrays[f"{prefix}_source_row_indices"] = union["source_row_indices"]
        arrays[f"{prefix}_membership_object_ids"] = union["membership_object_ids"]
        arrays[f"{prefix}_member_offsets"] = union["member_offsets"]
        for key, value in voxel.items():
            arrays[f"{prefix}_voxel10_{key}"] = value
    embedded_manifest = {
        "schema": "v13-color-preserving-pair-v2",
        "pair_id": pair_id,
        "arm": ARM,
        "source": "v16-b716-matched-region-prepared-input",
        "hypothesis_index": int(hypothesis["hypothesis_index"]),
        "hypothesis_sha256": stem,
        "checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "exact191_manifest_sha256": provenance["exact191_manifest_sha256"],
        "exact191_entries_sha256": provenance["exact191_entries_sha256"],
        "exact191_allowlist_sha256": provenance["exact191_allowlist_sha256"],
        "typed_failure_members_visible_and_never_filtered": True,
        "contains_typed_failure_members": bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "typed_failure_member_candidate_indices": list(
            typed_failure_metadata["typed_failure_member_candidate_indices"]),
        "existing_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "existing_typed_failure_member_candidate_indices"]),
        "new_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "new_typed_failure_member_candidate_indices"]),
        "selector_eligible": False,
        "safe_pose_vote_eligible": not bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "unit": "metre",
        "gt_consumed": False,
        "fallback_used": False,
    }
    embedded_manifest["payload_sha256"] = stable_json_sha256(embedded_manifest)
    arrays["manifest_json"] = np.asarray(json.dumps(
        embedded_manifest, sort_keys=True, separators=(",", ":")))
    npz_path = Path(output_root) / relative_npz
    prepared_sha256 = create_only_deterministic_npz(
        npz_path, arrays, event_hook=artifact_event_hook)
    evidence = {
        "schema": PREPARED_SCHEMA,
        "pair_id": pair_id,
        "short_id": short_id,
        "hypothesis_index": int(hypothesis["hypothesis_index"]),
        "hypothesis_sha256": stem,
        "prepared_input_path": relative_npz.as_posix(),
        "prepared_input_sha256": prepared_sha256,
        "prepared_manifest_payload_sha256":
            embedded_manifest["payload_sha256"],
        "v13_worker_arm": ARM,
        "v13_worker_schema_compatible": True,
        "worker_execution_authorized": False,
        "registration_executed": False,
        "geot_result_filtering_used": False,
        "typed_failure_members_visible_and_never_filtered": True,
        "contains_typed_failure_members": bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "typed_failure_member_candidate_indices": list(
            typed_failure_metadata["typed_failure_member_candidate_indices"]),
        "existing_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "existing_typed_failure_member_candidate_indices"]),
        "new_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "new_typed_failure_member_candidate_indices"]),
        "selector_eligible": False,
        "safe_pose_vote_eligible": not bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "member_rank_records": records,
        "members": {"source": source_members, "reference": reference_members},
        "preprocessing": {
            "filter_before_voxel": True,
            "voxel_size_m": COLORPCR_INPUT_VOXEL_M,
            "builder_cap512_applied": False,
            "official_worker_owns_cap512": True,
        },
        "arrays": {key: {"shape": list(np.asarray(value).shape),
                         "dtype": str(np.asarray(value).dtype),
                         "sha256": array_sha256(value)}
                   for key, value in sorted(arrays.items())},
        "provenance": dict(provenance),
        "forbidden_inputs": [
            "legacy B/89ed paths", "GT transforms", "selection/evaluation labels",
            "posthoc", "official92", "fallbacks", "GeoT result filtering",
        ],
    }
    evidence["payload_sha256"] = stable_json_sha256(evidence)
    evidence_path = Path(output_root) / relative_evidence
    evidence_sha256 = create_only_json(
        evidence_path, evidence, event_hook=artifact_event_hook)
    return {
        "hypothesis_index": int(hypothesis["hypothesis_index"]),
        "hypothesis_sha256": stem,
        "prepared_input_path": relative_npz.as_posix(),
        "prepared_input_sha256": prepared_sha256,
        "prepared_manifest_payload_sha256":
            embedded_manifest["payload_sha256"],
        "exact191_manifest_sha256":
            provenance["exact191_manifest_sha256"],
        "exact191_entries_sha256":
            provenance["exact191_entries_sha256"],
        "exact191_allowlist_sha256":
            provenance["exact191_allowlist_sha256"],
        "typed_failure_members_visible_and_never_filtered": True,
        "contains_typed_failure_members": bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "typed_failure_member_candidate_indices": list(
            typed_failure_metadata["typed_failure_member_candidate_indices"]),
        "existing_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "existing_typed_failure_member_candidate_indices"]),
        "new_typed_failure_member_candidate_indices": list(
            typed_failure_metadata[
                "new_typed_failure_member_candidate_indices"]),
        "selector_eligible": False,
        "safe_pose_vote_eligible": not bool(
            typed_failure_metadata["contains_typed_failure_members"]),
        "evidence_path": relative_evidence.as_posix(),
        "evidence_sha256": evidence_sha256,
    }


def build_fixed4_prepared_inputs(
    *, candidate_manifest_path: Path, candidate_manifest_sha256: str,
    exact191_manifest_path: Path, exact191_manifest_sha256: str,
    output_root: Path, raw_roots: Sequence[Path],
    canonical_builder: Callable[[str], tuple[Mapping[str, Any], Any]],
    raw_loader: Callable[[Path, str, str], RawInseg],
    raw_resolver: Callable[[str, Sequence[Path]], Path],
    source_hashes: Mapping[str, str], allow_test_fixture: bool = False,
    artifact_event_hook: ArtifactEventHook | None = None,
) -> dict[str, Any]:
    """Build all 34 prepared inputs; no worker or solver is imported/launched."""
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise V16ContractError("output root must be empty to prevent path collision")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate, exact, bindings = validate_candidate_and_exact191(
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_sha256=candidate_manifest_sha256,
        exact191_manifest_path=exact191_manifest_path,
        exact191_manifest_sha256=exact191_manifest_sha256,
        allow_test_fixture=allow_test_fixture)
    pair_rows, all_paths = [], set()
    for pair_id in FIXED_PAIR_ORDER:
        binding = bindings[pair_id]
        plan = binding["plan"]
        src_scan, ref_scan = pair_id.split("_to_")
        src_path = raw_resolver(src_scan, raw_roots)
        ref_path = raw_resolver(ref_scan, raw_roots)
        source = raw_loader(src_path, src_scan, "source")
        reference = raw_loader(ref_path, ref_scan, "reference")
        data, labels = canonical_builder(pair_id)
        if labels:
            raise V16ContractError("canonical builder unexpectedly returned labels")
        _validate_raw_surface_bindings(plan, data, source, reference)
        provenance = {
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "exact191_manifest_sha256": exact191_manifest_sha256,
            "candidate_plan_path": binding["plan_path"],
            "candidate_plan_sha256": binding["candidate_row"]["plan_sha256"],
            "exact191_entries_sha256": binding["exact_row"]["entries_sha256"],
            "exact191_correspondences_sha256":
                binding["exact_row"]["correspondences_sha256"],
            "exact191_allowlist_sha256": binding["exact_row"]["allowlist_sha256"],
            "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
            "raw_source_path": str(Path(src_path).resolve()),
            "raw_source_sha256": source.file_sha256,
            "raw_reference_path": str(Path(ref_path).resolve()),
            "raw_reference_sha256": reference.file_sha256,
            "formal_source_sha256": dict(sorted(source_hashes.items())),
            "synthetic_test_fixture": exact.get("synthetic_test_fixture") is True,
        }
        artifacts = []
        allowlist_by_index = {
            int(row["hypothesis_index"]): row
            for row in binding["allowlist"]["hypotheses"]
        }
        for hypothesis in plan["hypotheses"]:
            typed_failure_metadata = allowlist_by_index[
                int(hypothesis["hypothesis_index"])]
            artifact = build_prepared_hypothesis(
                pair_id=pair_id, short_id=plan["short_id"],
                hypothesis=hypothesis,
                candidate_records=plan["candidate_rank_records"], data=data,
                source=source, reference=reference, output_root=output_root,
                provenance=provenance,
                typed_failure_metadata=typed_failure_metadata,
                artifact_event_hook=artifact_event_hook)
            for key in ("prepared_input_path", "evidence_path"):
                if artifact[key] in all_paths:
                    raise V16ContractError("prepared output path collision")
                all_paths.add(artifact[key])
            artifacts.append(artifact)
        pair_manifest = {
            "schema": PAIR_MANIFEST_SCHEMA,
            "pair_id": pair_id,
            "short_id": plan["short_id"],
            "expected_hypothesis_count": EXPECTED_BY_PAIR[pair_id],
            "hypothesis_count": len(artifacts),
            "hypotheses": artifacts,
            "all_hypotheses_replayed": True,
            "typed_failure_members_visible_and_never_filtered": True,
            "hypotheses_with_typed_failure_members": sum(
                row["contains_typed_failure_members"] for row in artifacts),
            "hypotheses_with_existing_typed_failure_members": sum(
                bool(row[
                    "existing_typed_failure_member_candidate_indices"])
                for row in artifacts),
            "hypotheses_with_new_typed_failure_members": sum(
                bool(row["new_typed_failure_member_candidate_indices"])
                for row in artifacts),
            "existing_typed_failure_count":
                binding["exact_row"]["existing_failed_count"],
            "new_typed_failure_count":
                binding["exact_row"]["new_typed_failure_count"],
            "typed_failure_total_count": (
                binding["exact_row"]["existing_failed_count"]
                + binding["exact_row"]["new_typed_failure_count"]),
            "geot_result_filtering_used": False,
            "worker_execution_authorized": False,
        }
        if len(artifacts) != EXPECTED_BY_PAIR[pair_id]:
            raise V16ContractError("prepared hypothesis count mismatch")
        pair_manifest["payload_sha256"] = stable_json_sha256(pair_manifest)
        pair_relative = Path("pairs") / plan["short_id"] / "pair_manifest.json"
        pair_path = output_root / pair_relative
        pair_manifest_sha256 = create_only_json(
            pair_path, pair_manifest, event_hook=artifact_event_hook)
        pair_rows.append({
            "pair_id": pair_id,
            "short_id": plan["short_id"],
            "hypothesis_count": len(artifacts),
            "pair_manifest_path": pair_relative.as_posix(),
            "pair_manifest_sha256": pair_manifest_sha256,
        })
    if [row["hypothesis_count"] for row in pair_rows] \
            != list(EXPECTED_HYPOTHESIS_COUNTS) or len(all_paths) != 68:
        raise V16ContractError("fixed4 prepared closure is not exact 34")
    result = {
        "schema": BUILDER_MANIFEST_SCHEMA,
        "sealed": True,
        "cpu_only": True,
        "worker_execution_authorized": False,
        "registration_executed": False,
        "official92_executed": False,
        "gt_consumed": False,
        "fallback_used": False,
        "geot_result_filtering_used": False,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "exact191_manifest_sha256": exact191_manifest_sha256,
        "official_release_checkpoint_sha256": OFFICIAL_RELEASE_SHA256,
        "legacy_B_ep20_or_89ed_consumed": False,
        "synthetic_test_fixture": exact.get("synthetic_test_fixture") is True,
        "pair_count": 4,
        "hypothesis_count": 34,
        "hypothesis_distribution": list(EXPECTED_HYPOTHESIS_COUNTS),
        "existing_typed_failure_count":
            EXPECTED_EXISTING_TYPED_FAILURE_TOTAL,
        "new_typed_failure_count": EXPECTED_NEW_TYPED_FAILURE_TOTAL,
        "typed_failure_total_count": EXPECTED_TYPED_FAILURE_TOTAL,
        "hypotheses_with_existing_typed_failure_members":
            EXPECTED_EXISTING_TYPED_HYPOTHESES,
        "hypotheses_with_typed_failure_members":
            EXPECTED_ALL_TYPED_HYPOTHESES,
        "typed_failures_visible_and_never_filtered": True,
        "pairs": pair_rows,
        "formal_source_sha256": dict(sorted(source_hashes.items())),
    }
    result["payload_sha256"] = stable_json_sha256(result)
    builder_manifest_sha256 = create_only_json(
        output_root / "builder_manifest.json", result,
        event_hook=artifact_event_hook)
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append({
                "path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    artifact_manifest = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "file_count": len(artifacts),
        "files": artifacts,
        "recursive_artifact_closure_sha256": stable_json_sha256(artifacts),
    }
    artifact_manifest["payload_sha256"] = stable_json_sha256(artifact_manifest)
    artifact_manifest_sha256 = create_only_json(
        output_root / "artifact_manifest.json", artifact_manifest,
        event_hook=artifact_event_hook)
    for row in artifacts:
        path = output_root / row["path"]
        if (not path.is_file() or path.stat().st_size != row["bytes"]
                or sha256_file(path) != row["sha256"]):
            raise V16ContractError(
                f"builder artifact changed before closure: {row['path']}")
    return {
        **result,
        "builder_manifest_sha256": builder_manifest_sha256,
        "artifact_manifest_sha256": artifact_manifest_sha256,
    }
