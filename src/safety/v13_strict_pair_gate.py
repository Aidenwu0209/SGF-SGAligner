"""Single authoritative V13 pair gate after ColorPCR and dual solvers.

This layer applies the frozen surface ICP trace and unchanged Rule-B to the
observed dual-solver medoids.  It is the only object that the fixed4 aggregate
is allowed to consume.
"""
from __future__ import annotations

import json
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from safety import decision_features as dfx
from safety.v8_stage_order_consensus import fixed_trace_gate
from safety.v13_dual_solver_runtime import (
    DIRECTIONS, INLIER_THRESHOLD_M, MAX_ROTATION_DEG, MAX_TRANSLATION_M,
    SOLVERS, load_frozen_correspondences, sha256_file, transform_distance,
    validate_se3, array_sha256,
)


SCHEMA = "v13-strict-pair-gate-v1"
AUTHORITY = "fixed_trace_icp_plus_unchanged_rule_b_plus_dual_solver_q4"


class StrictPairGateError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    """Convert sealed numerical evidence to JSON-native containers.

    Formal evidence is written with ``allow_nan=False`` downstream.  Keeping
    this conversion next to the gate prevents an ndarray (notably the ICP
    transform and trace) from being silently omitted merely because it is not
    JSON serializable.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _centres(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    labels = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),):
        raise StrictPairGateError("surface points/segment ids are not row aligned")
    values = [points[labels == label].mean(axis=0) for label in np.unique(labels)
              if np.any(labels == label)]
    if not values:
        raise StrictPairGateError("surface has no segment centres")
    return np.asarray(values, np.float64)


def _load_surfaces(prepared_path: Path, pair_id: str, arm: str) -> dict[str, Any]:
    prepared_path = Path(prepared_path).resolve()
    before = sha256_file(prepared_path)
    with np.load(prepared_path, allow_pickle=False) as data:
        if "manifest_json" not in data.files:
            raise StrictPairGateError("prepared manifest missing")
        manifest = json.loads(str(data["manifest_json"].item()))
        required = [f"{arm}_{side}_{key}" for side in ("source", "reference")
                    for key in ("xyz", "labels")]
        if any(key not in data.files for key in required):
            raise StrictPairGateError("frozen surface arrays missing")
        arrays = {key: np.asarray(data[key]) for key in required}
    if sha256_file(prepared_path) != before:
        raise StrictPairGateError("prepared input changed while reading")
    if manifest.get("schema") != "v13-color-preserving-pair-v2" \
            or manifest.get("pair_id") != pair_id:
        raise StrictPairGateError("prepared pair identity mismatch")
    return {"path": str(prepared_path), "sha256": before, "manifest": manifest,
            "source": np.asarray(arrays[f"{arm}_source_xyz"], np.float64),
            "reference": np.asarray(arrays[f"{arm}_reference_xyz"], np.float64),
            "source_centres": _centres(arrays[f"{arm}_source_xyz"],
                                       arrays[f"{arm}_source_labels"]),
            "reference_centres": _centres(arrays[f"{arm}_reference_xyz"],
                                          arrays[f"{arm}_reference_labels"])}


def strict_pair_gate(
    *, pair_id: str, arm: str, prepared_path: Path,
    forward_cache_path: Path, reverse_cache_path: Path,
    dual_summary: Mapping[str, Any], preregistration: Mapping[str, Any],
    icp_fn: Callable[..., Mapping[str, Any]],
    rule_features_fn: Callable[..., tuple[dict, dict]],
    test_injection: bool = False,
) -> dict[str, Any]:
    if arm not in ("sgf_selected_union", "fullscan"):
        raise StrictPairGateError("unknown arm")
    known_bad_id = str(preregistration.get("known_bad_pair_id", ""))
    allowed = set(preregistration.get("normal_pair_ids", ())) | {known_bad_id}
    if not known_bad_id or pair_id not in allowed:
        raise StrictPairGateError("pair is not in frozen fixed4 identity set")
    if dual_summary.get("schema") != "v13-dual-solver-summary-v1":
        raise StrictPairGateError("dual summary schema mismatch")
    runtime_pins = preregistration.get("strict_gate_runtime_pins", {})
    runtime_files = {
        "v7_registration_pilot.py": Path(inspect.getsourcefile(icp_fn) or ""),
        "decision_features.py": Path(dfx.__file__),
        "v8_stage_order_consensus.py": Path(inspect.getsourcefile(fixed_trace_gate) or ""),
    }
    # icp_fn and rule_features_fn are sealed in the same v7 module.  Formal
    # execution rejects injected callables; unit tests must opt into a visibly
    # non-production test path.
    rule_source = Path(inspect.getsourcefile(rule_features_fn) or "")
    runtime_receipt = {}
    if test_injection:
        runtime_receipt = {"mode": "TEST_ONLY_INJECTION_UNSEALED"}
    else:
        if rule_source != runtime_files["v7_registration_pilot.py"]:
            raise StrictPairGateError("ICP and Rule-B feature functions are not co-sealed")
        for name, path in runtime_files.items():
            expected = runtime_pins.get(name)
            if not path.is_file() or not isinstance(expected, str) \
                    or sha256_file(path) != expected:
                raise StrictPairGateError(f"strict runtime source SHA mismatch: {name}")
        runtime_receipt = {"mode": "SEALED_FORMAL_RUNTIME",
                           "source_sha256": {name: sha256_file(path)
                                             for name, path in runtime_files.items()}}
    surfaces = _load_surfaces(prepared_path, pair_id, arm)
    caches = {"forward": load_frozen_correspondences(forward_cache_path),
              "reverse": load_frozen_correspondences(reverse_cache_path)}
    if dual_summary.get("cache_sha256") != {
            direction: cache.cache_sha256 for direction, cache in caches.items()}:
        raise StrictPairGateError("dual summary cache binding mismatch")

    rows: dict[str, Any] = {}
    finals: dict[tuple[str, str], np.ndarray] = {}
    for solver in SOLVERS:
        for direction in DIRECTIONS:
            name = f"{solver}/{direction}"
            if direction == "forward":
                source, reference = surfaces["source"], surfaces["reference"]
                centres = surfaces["source_centres"]
            else:
                source, reference = surfaces["reference"], surfaces["source"]
                centres = surfaces["reference_centres"]
            surface_evidence = {
                "surface_source_point_count": int(len(source)),
                "surface_reference_point_count": int(len(reference)),
                "surface_source_sha256": array_sha256(source),
                "surface_reference_sha256": array_sha256(reference),
            }
            gate = dual_summary.get("gates", {}).get(name, {})
            if gate.get("usable") is not True or gate.get("medoid_transform") is None:
                rows[name] = {
                    **surface_evidence,
                    "usable": False, "reason": "dual_medoid_missing",
                    "rule_b_features": None,
                    "recorded_rule_b_decision": None,
                    "icp": None,
                }
                continue
            raw = validate_se3(gate["medoid_transform"])
            cache = caches[direction]
            residuals = np.linalg.norm(cache.src @ raw[:3, :3].T + raw[:3, 3]
                                       - cache.ref, axis=1)
            inliers = int((residuals <= INLIER_THRESHOLD_M).sum())
            icp = dict(icp_fn(source, reference, raw,
                              seed=42 if direction == "forward" else 43))
            features, recorded = rule_features_fn(
                source, reference, centres, raw, inliers, len(cache.src),
                0, 0, icp, direction=direction)
            violations = list(dfx.evaluate_rule_b(dict(features)))
            recorded_reasons = list(recorded.get("rejection_reasons", ()))
            trace = fixed_trace_gate({"icp": icp})
            recorded_consistent = (violations == recorded_reasons
                                   and bool(not violations)
                                   is bool(recorded.get("usable_for_reconstruction")))
            usable = not violations and trace["usable"] and recorded_consistent
            finals[(solver, direction)] = validate_se3(icp["transform"])
            rows[name] = {**surface_evidence,
                          "usable": usable, "rule_b_rejection_reasons": violations,
                          "recorded_rule_b_consistent": recorded_consistent,
                          "rule_b_features": _json_safe(features),
                          "recorded_rule_b_decision": _json_safe(recorded),
                          "icp": _json_safe(icp),
                          "fixed_trace": trace, "raw_transform": raw.tolist(),
                          "final_transform": finals[(solver, direction)].tolist(),
                          "inliers_10cm": inliers,
                          "correspondence_count": len(cache.src),
                          "segment_centre_count": len(centres),
                          "segment_centres_sha256": array_sha256(centres),
                          "successful_node_pairs": 0, "failed_node_pairs": 0,
                          "node_pair_success_ratio": 0.0,
                          "rule_c_claimed": False}

    final_checks: dict[str, Any] = {}
    for solver in SOLVERS:
        if (solver, "forward") in finals and (solver, "reverse") in finals:
            dr, dt = transform_distance(finals[(solver, "forward")],
                                        np.linalg.inv(finals[(solver, "reverse")]))
            final_checks[f"{solver}/bidirectional"] = {
                "rotation_deg": dr, "translation_m": dt,
                "usable": dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M}
    if all((solver, "forward") in finals for solver in SOLVERS):
        dr, dt = transform_distance(finals[(SOLVERS[0], "forward")],
                                    finals[(SOLVERS[1], "forward")])
        final_checks["cross_solver/forward"] = {
            "rotation_deg": dr, "translation_m": dt,
                "usable": dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M}
    if all((solver, "reverse") in finals for solver in SOLVERS):
        dr, dt = transform_distance(finals[(SOLVERS[0], "reverse")],
                                    finals[(SOLVERS[1], "reverse")])
        final_checks["cross_solver/reverse"] = {
            "rotation_deg": dr, "translation_m": dt,
            "usable": dr <= MAX_ROTATION_DEG and dt <= MAX_TRANSLATION_M}
    strict_geometry_safe = (dual_summary.get("safe") is True
                            and len(rows) == 4
                            and all(row.get("usable") is True for row in rows.values())
                            and len(final_checks) == 4
                            and all(row["usable"] for row in final_checks.values()))
    known_bad = pair_id == known_bad_id
    safe = strict_geometry_safe and not known_bad
    reason = ("known_bad_veto" if known_bad else
              "strict_pair_gate_pass" if safe else "strict_pair_gate_failed")
    output_schema, authority = SCHEMA, AUTHORITY
    if test_injection:
        output_schema, authority = "v13-strict-pair-gate-test-only-v1", "TEST_ONLY"
        safe, reason = False, "test_only_injection"
    return {"schema": output_schema, "pair_id": pair_id, "arm": arm, "safe": safe,
            "reason": reason, "gate_authority": authority,
            "known_bad_veto": known_bad, "bound_known_bad_pair_id": known_bad_id,
            "strict_geometry_safe_before_veto": strict_geometry_safe,
            "prepared_input_sha256": surfaces["sha256"],
            "cache_sha256": {direction: cache.cache_sha256
                             for direction, cache in caches.items()},
            "medoid_safety": rows, "final_consistency": final_checks,
            "runtime_receipt": runtime_receipt,
            "rule_b_evaluator": "evaluate_rule_b", "rule_c_claimed": False,
            "gt_consumed": False, "fallback_used": False}
