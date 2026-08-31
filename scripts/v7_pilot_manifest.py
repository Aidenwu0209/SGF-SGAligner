"""Seal the GT-free, pre-registered 12-pair V7 pilot manifest.

The selector consumes only a materialized whitelist projection.  It never
opens the original audit rows, label sidecars or post-hoc result trees.  Pair
identifiers are opaque strings used only for cache lookup and deterministic
tie breaking.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECTION_SCHEMA = "v7-pilot-whitelist-projection-v1"
MANIFEST_SCHEMA = "v7-registration-veto-batch-manifest-v1"
MANIFEST_STATUS = "FROZEN"
CHECKPOINT_ID = "B"
CHECKPOINT_SHA256 = (
    "89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200"
)
PROTOCOL_SHA256 = (
    "399ec014689f1bb5e0128b77f65c461c07e548f7ffe0cc7d0fd77f8debfaf477"
)
KNOWN_ERROR_PAIR = (
    "6a36052f-fa53-2915-9400-831b60c63077_to_"
    "6a36052d-fa53-2915-9764-30d81b2cc2b5"
)

EXPECTED_PAIR_IDS = (
    KNOWN_ERROR_PAIR,
    "09582207-e2c2-2de1-972c-225d968c2ab4_to_0958220d-e2c2-2de1-9710-c37018da1883",
    "8f0f1455-55de-28ce-832d-d58f1c6c398d_to_8f0f144b-55de-28ce-8053-2828b87a0cc9",
    "73315a33-185c-2c8a-87a3-7915ecadfa45_to_73315a2d-185c-2c8a-87e9-d8dfe07ae3cb",
    "bcb0fe2b-4f39-2c70-9d6b-5e92d634ac35_to_bcb0fe2d-4f39-2c70-9d1a-49a4a6868d7d",
    "8f0f144d-55de-28ce-8075-69a0a3b631b5_to_8f0f144b-55de-28ce-8053-2828b87a0cc9",
    "a644cb91-0ee5-2f66-9da1-5edabca2f13d_to_a644cb97-0ee5-2f66-9cc7-3ecaa29c19df",
    "8f0f1447-55de-28ce-83c5-092887498eea_to_8f0f144b-55de-28ce-8053-2828b87a0cc9",
    "09582209-e2c2-2de1-9610-08baed932919_to_0958220d-e2c2-2de1-9710-c37018da1883",
    "a644cb93-0ee5-2f66-9efb-b16adfb14eff_to_a644cb97-0ee5-2f66-9cc7-3ecaa29c19df",
    "c12890cf-d3df-2d0d-876d-f774cb9d9861_to_c12890cc-d3df-2d0d-85cd-eebc3e1c4b62",
    "634d11d9-6833-255d-8fa2-ce325873192d_to_3b7b33a9-1b11-283e-9b02-e8f35f6ba24c",
)

EXPECTED_ROLES = (
    "known_error_exception",
    "decision_flip",
    "decision_flip",
    "decision_flip",
    "decision_flip",
    "stable_pass_near_boundary",
    "stable_pass_near_boundary",
    "stable_pass_near_boundary",
    "stable_pass_high_margin",
    "stable_pass_high_margin",
    "stable_pass_high_margin",
    "stable_reject_max_dispersion",
)

FORBIDDEN_PATH_COMPONENTS = frozenset({
    "node_evidence", "calibration", "fixed12", "official92",
})
FORBIDDEN_SYMBOLS = frozenset({"load_gt_transform", "load_anchor_ids"})
FORBIDDEN_PROJECTION_KEYS = frozenset({
    "paths", "anchors", "rre", "rte", "strict", "relaxed",
    "accepted_correct", "accepted_error", "error", "scene_id",
    "source_scene", "reference_scene", "gt_transform",
})

TOP_KEYS = frozenset({
    "schema", "status", "checkpoint_id", "checkpoint_sha256",
    "protocol_sha256", "source_files", "pairs",
})
SOURCE_KEYS = frozenset({"repeat", "sha256", "bytes"})
PAIR_KEYS = frozenset({
    "pair_id", "cache_sha256", "cache_bytes", "stable_signature", "repeats",
})
REPEAT_KEYS = frozenset({
    "repeat", "node_corr_count", "registration_valid", "decision_usable",
    "stable_signature", "execution_signature", "transform", "rule_b_features",
})
RULE_B_KEYS = frozenset({
    "ransac_inliers", "ransac_inlier_ratio", "spatial_extent_m",
    "spatial_second_axis_m", "icp_update_translation_m",
    "icp_update_rotation_deg", "bidirectional_rotation_deg",
    "bidirectional_translation_m", "overlap_ratio", "icp_converged",
    "overlap_10cm", "overlap_5cm", "symmetric_trimmed_chamfer_m",
    "median_residual_m", "p90_residual_m", "icp_fitness", "icp_rmse_m",
    "node_pair_success_ratio", "successful_node_pairs", "failed_node_pairs",
    "bidirectional_available",
})

LOWER_RULES = {
    "ransac_inliers": 6.0,
    "spatial_extent_m": 1.0,
    "icp_fitness": 0.30,
    "overlap_10cm": 0.10,
}
UPPER_RULES = {
    "median_residual_m": 0.10,
    "symmetric_trimmed_chamfer_m": 0.10,
    "icp_update_translation_m": 0.20,
    "icp_update_rotation_deg": 10.0,
    "bidirectional_rotation_deg": 5.0,
    "bidirectional_translation_m": 0.20,
}
BOOLEAN_RULES = ("icp_converged", "bidirectional_available")


class ManifestSealError(RuntimeError):
    """Projection, provenance or frozen selection failed validation."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def guard_path(path: Path, purpose: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    lowered = tuple(part.casefold() for part in resolved.parts)
    blocked = sorted(
        token for token in FORBIDDEN_PATH_COMPONENTS
        if any(token in component for component in lowered)
    )
    if blocked:
        raise ManifestSealError(
            f"{purpose} path crosses forbidden evidence component: {blocked}"
        )
    return resolved


def safe_read_bytes(path: Path, purpose: str) -> bytes:
    checked = guard_path(path, purpose)
    if not checked.is_file():
        raise ManifestSealError(f"missing {purpose}: {checked}")
    with checked.open("rb") as handle:
        return handle.read()


def safe_read_json(path: Path, purpose: str) -> Any:
    try:
        return json.loads(safe_read_bytes(path, purpose))
    except json.JSONDecodeError as exc:
        raise ManifestSealError(f"invalid JSON in {purpose}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    checked = guard_path(path, "manifest output")
    checked.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checked.name}.", suffix=".tmp", dir=checked.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, checked)
        except FileExistsError as exc:
            raise ManifestSealError(f"refusing to overwrite frozen manifest: {checked}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def require_exact_keys(value: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise ManifestSealError(f"{where} must be an object")
    observed = frozenset(str(key) for key in value)
    if observed != allowed:
        raise ManifestSealError(
            f"{where} schema mismatch: missing={sorted(allowed-observed)}, "
            f"unknown={sorted(observed-allowed)}"
        )
    forbidden = observed & FORBIDDEN_PROJECTION_KEYS
    if forbidden:
        raise ManifestSealError(f"{where} contains forbidden keys: {sorted(forbidden)}")


def require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestSealError(f"{where} must be a lowercase SHA-256")
    return value


def require_finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestSealError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestSealError(f"{where} must be finite")
    return result


def require_opaque_pair_id(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ManifestSealError("pair_id must be a non-empty opaque string")
    if any(token in value for token in ("/", "\\", "..", "\x00")):
        raise ManifestSealError("pair_id is unsafe for opaque cache lookup")
    return value


def validate_transform(value: Any, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ManifestSealError(f"{where} must be a finite 4x4 transform")
    if not np.allclose(array[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ManifestSealError(f"{where} has an invalid homogeneous row")
    return array


def validate_projection(document: Any) -> list[dict[str, Any]]:
    require_exact_keys(document, TOP_KEYS, "projection")
    if document["schema"] != PROJECTION_SCHEMA or document["status"] != "FROZEN":
        raise ManifestSealError("projection schema/status mismatch")
    if document["checkpoint_id"] != CHECKPOINT_ID:
        raise ManifestSealError("projection checkpoint id mismatch")
    if document["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ManifestSealError("projection checkpoint SHA mismatch")
    if document["protocol_sha256"] != PROTOCOL_SHA256:
        raise ManifestSealError("projection protocol SHA mismatch")
    sources = document["source_files"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ManifestSealError("projection must seal exactly three source repeats")
    for expected_repeat, source in enumerate(sources):
        require_exact_keys(source, SOURCE_KEYS, f"source_files[{expected_repeat}]")
        if source["repeat"] != expected_repeat:
            raise ManifestSealError("source repeat order mismatch")
        require_sha(source["sha256"], "source sha256")
        if not isinstance(source["bytes"], int) or source["bytes"] <= 0:
            raise ManifestSealError("source bytes must be positive")
    pairs = document["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ManifestSealError("projection pairs must be a non-empty list")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        require_exact_keys(pair, PAIR_KEYS, f"pairs[{pair_index}]")
        pair_id = require_opaque_pair_id(pair["pair_id"])
        if pair_id in seen:
            raise ManifestSealError(f"duplicate pair_id {pair_id}")
        seen.add(pair_id)
        require_sha(pair["cache_sha256"], f"{pair_id}.cache_sha256")
        require_sha(pair["stable_signature"], f"{pair_id}.stable_signature")
        if not isinstance(pair["cache_bytes"], int) or pair["cache_bytes"] <= 0:
            raise ManifestSealError(f"{pair_id}.cache_bytes must be positive")
        repeats = pair["repeats"]
        if not isinstance(repeats, list) or len(repeats) != 3:
            raise ManifestSealError(f"{pair_id} must have exactly three repeats")
        for expected_repeat, repeat in enumerate(repeats):
            require_exact_keys(
                repeat, REPEAT_KEYS, f"{pair_id}.repeats[{expected_repeat}]"
            )
            if repeat["repeat"] != expected_repeat:
                raise ManifestSealError(f"{pair_id} repeat order mismatch")
            if not isinstance(repeat["node_corr_count"], int) or repeat["node_corr_count"] <= 0:
                raise ManifestSealError(f"{pair_id} node_corr_count invalid")
            for flag in ("registration_valid", "decision_usable"):
                if not isinstance(repeat[flag], bool):
                    raise ManifestSealError(f"{pair_id}.{flag} must be bool")
            for signature in ("stable_signature", "execution_signature"):
                require_sha(repeat[signature], f"{pair_id}.{signature}")
            if repeat["stable_signature"] != pair["stable_signature"]:
                raise ManifestSealError(f"{pair_id} stable signature drift")
            validate_transform(repeat["transform"], f"{pair_id}.transform")
            features = repeat["rule_b_features"]
            require_exact_keys(features, RULE_B_KEYS, f"{pair_id}.rule_b_features")
            for key, feature in features.items():
                if key in BOOLEAN_RULES:
                    if not isinstance(feature, bool):
                        raise ManifestSealError(f"{pair_id}.{key} must be bool")
                else:
                    require_finite_number(feature, f"{pair_id}.{key}")
        validated.append(dict(pair))
    return validated


def rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def transform_dispersion(repeats: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    transforms = [validate_transform(row["transform"], "transform") for row in repeats]
    rotations, translations = [], []
    for left_index in range(len(transforms)):
        for right_index in range(left_index + 1, len(transforms)):
            left, right = transforms[left_index], transforms[right_index]
            rotations.append(rotation_distance_deg(left, right))
            translations.append(float(np.linalg.norm(left[:3, 3] - right[:3, 3])))
    max_rotation = max(rotations)
    max_translation = max(translations)
    return max(max_rotation / 5.0, max_translation / 0.20), max_rotation, max_translation


def rule_b_margins(repeats: Sequence[Mapping[str, Any]]) -> list[float]:
    margins: list[float] = []
    for repeat in repeats:
        features = repeat["rule_b_features"]
        for key, threshold in LOWER_RULES.items():
            margins.append((float(features[key]) - threshold) / threshold)
        for key, threshold in UPPER_RULES.items():
            margins.append((threshold - float(features[key])) / threshold)
        for key in BOOLEAN_RULES:
            margins.append(1.0 if features[key] else -1.0)
    return margins


def pair_metrics(pair: Mapping[str, Any]) -> dict[str, Any]:
    repeats = pair["repeats"]
    pattern = "".join("P" if row["decision_usable"] else "R" for row in repeats)
    margins = rule_b_margins(repeats)
    dispersion, max_rotation, max_translation = transform_dispersion(repeats)
    node_counts = {int(row["node_corr_count"]) for row in repeats}
    if len(node_counts) != 1:
        raise ManifestSealError(f"{pair['pair_id']} node count changed across repeats")
    return {
        "pair_id": pair["pair_id"],
        "cache_sha256": pair["cache_sha256"],
        "cache_bytes": pair["cache_bytes"],
        "pattern": pattern,
        "pass_count": pattern.count("P"),
        "node_corr_count": next(iter(node_counts)),
        "worst_rule_b_margin": min(margins),
        "closest_rule_b_margin": min(abs(value) for value in margins),
        "max_rotation_dispersion_deg": max_rotation,
        "max_translation_dispersion_m": max_translation,
        "normalized_dispersion": dispersion,
        "stable_signature": pair["stable_signature"],
    }


def select_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metrics = [pair_metrics(pair) for pair in pairs]
    by_id = {row["pair_id"]: row for row in metrics}
    if KNOWN_ERROR_PAIR not in by_id:
        raise ManifestSealError("predeclared exception pair is absent")
    selected: list[tuple[dict[str, Any], str, str]] = []
    known = by_id[KNOWN_ERROR_PAIR]
    selected.append((
        known,
        "known_error_exception",
        "predeclared opaque pair id; no label metric participates in ranking",
    ))
    flips = sorted(
        (row for row in metrics
         if row["pair_id"] != KNOWN_ERROR_PAIR and row["pass_count"] in (1, 2)),
        key=lambda row: (-row["normalized_dispersion"], row["pair_id"]),
    )
    for row in flips:
        selected.append((
            row, "decision_flip",
            "all non-exception three-repeat decision flips, dispersion descending",
        ))
    stable_pass = [row for row in metrics if row["pass_count"] == 3]
    near = sorted(
        stable_pass,
        key=lambda row: (row["closest_rule_b_margin"], row["pair_id"]),
    )[:3]
    for row in near:
        selected.append((
            row, "stable_pass_near_boundary",
            "stable pass nearest an unchanged Rule-B boundary",
        ))
    near_ids = {row["pair_id"] for row in near}
    high = sorted(
        (row for row in stable_pass if row["pair_id"] not in near_ids),
        key=lambda row: (
            -row["worst_rule_b_margin"],
            row["normalized_dispersion"],
            row["pair_id"],
        ),
    )[:3]
    for row in high:
        selected.append((
            row, "stable_pass_high_margin",
            "stable pass high-margin control after boundary stratum removal",
        ))
    rejects = sorted(
        (row for row in metrics if row["pass_count"] == 0),
        key=lambda row: (-row["normalized_dispersion"], row["pair_id"]),
    )
    if not rejects:
        raise ManifestSealError("no stable-reject stress control")
    selected.append((
        rejects[0], "stable_reject_max_dispersion",
        "stable reject with maximum three-repeat transform dispersion",
    ))
    pair_ids = tuple(row[0]["pair_id"] for row in selected)
    roles = tuple(row[1] for row in selected)
    if pair_ids != EXPECTED_PAIR_IDS or roles != EXPECTED_ROLES:
        raise ManifestSealError(
            "mechanical selection differs from the pre-registered frozen 12-pair list"
        )
    return [
        {**row, "role": role, "selection_reason": reason}
        for row, role, reason in selected
    ]


def pair_ids_sha256(pair_ids: Iterable[str]) -> str:
    payload = "".join(f"{pair_id}\n" for pair_id in pair_ids).encode()
    return sha256_bytes(payload)


def assert_gt_free_ast(path: Path) -> None:
    source = safe_read_bytes(path, "selector source").decode("utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
        if FORBIDDEN_SYMBOLS.intersection(names):
            raise ManifestSealError("selector AST references a forbidden GT loader")


def verify_cache_files(cache_root: Path, selected: Sequence[Mapping[str, Any]]) -> None:
    root = guard_path(cache_root, "cache root")
    if not root.is_dir():
        raise ManifestSealError(f"cache root is missing: {root}")
    for row in selected:
        pair_id = require_opaque_pair_id(row["pair_id"])
        path = guard_path(root / f"{pair_id}.pt", "cache file")
        payload = safe_read_bytes(path, "cache file")
        if len(payload) != row["cache_bytes"]:
            raise ManifestSealError(f"cache byte count mismatch for {pair_id}")
        if sha256_bytes(payload) != row["cache_sha256"]:
            raise ManifestSealError(f"cache SHA mismatch for {pair_id}")


def build_manifest(
    projection: Mapping[str, Any],
    projection_bytes: bytes,
    selected: Sequence[Mapping[str, Any]],
    selector_sha256: str,
) -> dict[str, Any]:
    pair_ids = [row["pair_id"] for row in selected]
    pattern_counts: dict[str, int] = {}
    for pair in validate_projection(projection):
        pattern = pair_metrics(pair)["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    return {
        "schema": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "pair_count": 12,
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "cache_sha256": row["cache_sha256"],
                "role": row["role"],
            }
            for row in selected
        ],
        "pair_ids_sha256": pair_ids_sha256(pair_ids),
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "projection_schema": PROJECTION_SCHEMA,
        "projection_sha256": sha256_bytes(projection_bytes),
        "selector_sha256": selector_sha256,
        "source_files": projection["source_files"],
        "population_decision_patterns": dict(sorted(pattern_counts.items())),
        "selection_receipt": [
            {
                key: row[key]
                for key in (
                    "pair_id", "role", "selection_reason", "pattern", "pass_count",
                    "node_corr_count", "worst_rule_b_margin",
                    "closest_rule_b_margin", "max_rotation_dispersion_deg",
                    "max_translation_dispersion_m", "normalized_dispersion",
                    "stable_signature",
                )
            }
            for row in selected
        ],
        "gt_free_contract": {
            "pair_id_semantics": "opaque lookup and deterministic tie-break only",
            "known_pair_exception": "predeclared opaque id insertion only",
            "cache_access": "raw bytes for SHA-256 and byte count only",
            "unknown_projection_fields": "fail_closed",
            "official92_authorized": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing", action="store_true",
        help="require the existing output to equal the recomputed manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assert_gt_free_ast(Path(__file__))
    projection_path = guard_path(args.projection, "whitelist projection")
    projection_bytes = safe_read_bytes(projection_path, "whitelist projection")
    try:
        projection = json.loads(projection_bytes)
    except json.JSONDecodeError as exc:
        raise ManifestSealError(f"invalid projection JSON: {exc}") from exc
    pairs = validate_projection(projection)
    selected = select_pairs(pairs)
    verify_cache_files(args.cache_root, selected)
    selector_sha = sha256_bytes(safe_read_bytes(Path(__file__), "selector source"))
    manifest = build_manifest(projection, projection_bytes, selected, selector_sha)
    if args.verify_existing:
        existing = safe_read_json(args.output, "frozen manifest")
        if existing != manifest:
            raise ManifestSealError("frozen manifest differs from mechanical recomputation")
    else:
        atomic_write_json(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
