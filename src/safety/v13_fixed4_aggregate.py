"""Fail-closed top-level aggregate for the pre-registered V13 fixed4 pilot."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


SCHEMA = "v13-fixed4-aggregate-v1"
PAIR_SCHEMA = "v13-strict-pair-gate-v1"
AUTHORITY = "fixed_trace_icp_plus_unchanged_rule_b_plus_dual_solver_q4"
PRIMARY_ARM = "sgf_selected_union"
CONTROL_ARM = "fullscan"
MEDOID_KEYS = tuple(f"{solver}/{direction}"
                    for solver in ("pointdsc", "pygcransac")
                    for direction in ("forward", "reverse"))
EVIDENCE_FIELDS = {
    "rule_b_features", "recorded_rule_b_decision", "icp",
    "surface_source_point_count", "surface_reference_point_count",
    "surface_source_sha256", "surface_reference_sha256",
}
RULE_B_FEATURE_FIELDS = {
    "overlap_10cm", "median_residual_m", "symmetric_trimmed_chamfer_m",
    "icp_converged", "icp_update_translation_m", "icp_update_rotation_deg",
    "icp_fitness", "ransac_inliers", "spatial_extent_m",
    "bidirectional_available", "bidirectional_rotation_deg",
    "bidirectional_translation_m",
}
RECORDED_DECISION_FIELDS = {
    "rejection_reasons", "usable_for_reconstruction", "rule", "thresholds",
}
ICP_FIELDS = {
    "transform", "converged", "fitness", "rmse_m", "update_rotation_deg",
    "update_translation_m", "trace",
}


class Fixed4AggregateError(RuntimeError):
    pass


def aggregate_fixed4(rows: Sequence[Mapping[str, Any]],
                     preregistration: Mapping[str, Any]) -> dict[str, Any]:
    normal_ids = tuple(str(value) for value in preregistration.get("normal_pair_ids", ()))
    known_bad_id = str(preregistration.get("known_bad_pair_id", ""))
    if len(normal_ids) != 3 or len(set(normal_ids)) != 3 or not known_bad_id \
            or known_bad_id in normal_ids:
        raise Fixed4AggregateError("pre-registration must bind exactly 3 normal plus 1 known-bad pair")
    if preregistration.get("primary_arm") != PRIMARY_ARM \
            or preregistration.get("control_arm") != CONTROL_ARM \
            or preregistration.get("control_can_rescue") is not False:
        raise Fixed4AggregateError("primary/control policy differs from frozen contract")
    expected_pairs = set(normal_ids) | {known_bad_id}
    keys = [(str(row.get("pair_id", "")), str(row.get("arm", ""))) for row in rows]
    expected_keys = {(pair_id, arm) for pair_id in expected_pairs
                     for arm in (PRIMARY_ARM, CONTROL_ARM)}
    if len(rows) != len(expected_keys) or Counter(keys) != Counter(expected_keys):
        raise Fixed4AggregateError("fixed4 evidence must contain exact pair x arm matrix")
    by_key = {key: dict(row) for key, row in zip(keys, rows)}
    for key, row in by_key.items():
        if row.get("schema") != PAIR_SCHEMA or row.get("gate_authority") != AUTHORITY:
            raise Fixed4AggregateError(f"strict gate authority missing for {key}")
        expected_pins = preregistration.get("strict_gate_runtime_pins", {})
        receipt = row.get("runtime_receipt", {})
        if receipt.get("mode") != "SEALED_FORMAL_RUNTIME" \
                or receipt.get("source_sha256") != expected_pins \
                or row.get("rule_b_evaluator") != "evaluate_rule_b" \
                or row.get("rule_c_claimed") is not False:
            raise Fixed4AggregateError(f"unsealed or test-only strict gate evidence for {key}")
        if not isinstance(row.get("safe"), bool):
            raise Fixed4AggregateError(f"boolean safe verdict missing for {key}")
        medoids = row.get("medoid_safety")
        if not isinstance(medoids, Mapping) or set(medoids) != set(MEDOID_KEYS):
            raise Fixed4AggregateError(f"complete medoid evidence missing for {key}")
        for medoid_key, evidence in medoids.items():
            if not isinstance(evidence, Mapping) or not EVIDENCE_FIELDS.issubset(evidence):
                raise Fixed4AggregateError(
                    f"auditable strict-gate fields missing for {key}/{medoid_key}")
            for field in ("surface_source_sha256", "surface_reference_sha256"):
                if not isinstance(evidence.get(field), str) or len(evidence[field]) != 64:
                    raise Fixed4AggregateError(
                        f"surface hash missing for {key}/{medoid_key}/{field}")
            for field in ("surface_source_point_count", "surface_reference_point_count"):
                if not isinstance(evidence.get(field), int) or evidence[field] <= 0:
                    raise Fixed4AggregateError(
                        f"surface count missing for {key}/{medoid_key}/{field}")
            if evidence.get("usable") is True:
                features = evidence.get("rule_b_features")
                decision = evidence.get("recorded_rule_b_decision")
                icp = evidence.get("icp")
                if not isinstance(features, Mapping) \
                        or not RULE_B_FEATURE_FIELDS.issubset(features) \
                        or not isinstance(decision, Mapping) \
                        or not RECORDED_DECISION_FIELDS.issubset(decision) \
                        or not isinstance(icp, Mapping) \
                        or not ICP_FIELDS.issubset(icp) \
                        or not isinstance(icp.get("trace"), list) \
                        or not icp["trace"]:
                    raise Fixed4AggregateError(
                        f"usable medoid lacks Rule-B/ICP evidence for {key}/{medoid_key}")

    primary = {pair_id: by_key[(pair_id, PRIMARY_ARM)] for pair_id in expected_pairs}
    control = {pair_id: by_key[(pair_id, CONTROL_ARM)] for pair_id in expected_pairs}
    normal_failures = [pair_id for pair_id in normal_ids if primary[pair_id]["safe"] is not True]
    def exact_known_bad_veto(row: Mapping[str, Any]) -> bool:
        return (row["safe"] is False
                and row.get("known_bad_veto") is True
                and row.get("reason") == "known_bad_veto"
                and row.get("bound_known_bad_pair_id") == known_bad_id)

    # fullscan cannot rescue a normal primary, but the pre-registered known-bad
    # safety probe must be vetoed in *both* arms.  Otherwise one arm would
    # demonstrate an unsafe acceptance that the aggregate silently hides.
    known_bad_veto_by_arm = {
        PRIMARY_ARM: exact_known_bad_veto(primary[known_bad_id]),
        CONTROL_ARM: exact_known_bad_veto(control[known_bad_id]),
    }
    known_bad_veto = all(known_bad_veto_by_arm.values())
    control_rescue_candidates = [pair_id for pair_id in normal_failures
                                 if control[pair_id]["safe"] is True]
    safe = not normal_failures and known_bad_veto
    reason = ("fixed4_primary_pass" if safe else
              "known_bad_veto_failed" if not known_bad_veto else
              "normal_primary_failed")
    return {
        "schema": SCHEMA, "safe": safe, "reason": reason,
        "primary_arm": PRIMARY_ARM, "control_arm": CONTROL_ARM,
        "control_can_rescue": False,
        "normal_pair_ids": list(normal_ids), "known_bad_pair_id": known_bad_id,
        "normal_primary_safe": {pair_id: primary[pair_id]["safe"] for pair_id in normal_ids},
        "normal_primary_failures": normal_failures,
        "known_bad_veto": known_bad_veto,
        "known_bad_veto_by_arm": known_bad_veto_by_arm,
        "control_safe_diagnostic": {pair_id: control[pair_id]["safe"]
                                    for pair_id in sorted(expected_pairs)},
        "control_rescue_candidates_not_used": control_rescue_candidates,
        "gate_authority": AUTHORITY,
    }
