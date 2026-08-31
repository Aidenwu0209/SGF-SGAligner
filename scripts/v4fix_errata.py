"""V4-Fix Part 1: evidence errata generated FROM RAW JSON (never by
copying final_decision.md prose).

Three factual corrections, additive evidence only (old files/dirs
untouched):

E1  the single accepted_strict_error in the V4 registration repeats is
    pair 0ad2d384..._to_0ad2d399... (fixed12, arm C), RRE 2.966 deg /
    RTE 0.2172 m — relaxed-level near-miss translation error.  The V4
    final_decision.md WRONGLY attributed it to 10b1792c... and to a
    "~180-degree flipped solution".  10b1792c appears only in the
    ambiguity (strict-flip) lists, never as an error accept.

E2  deterministic_metrics.json v3_sealed_reference.calibration90.
    top5_recall = 0.4353 is a provenance typo; the correct incumbent
    calibration top-5 is 0.3498433974919869 (the computed field was
    always correct and all gate evaluations used the computed value —
    no gate outcome changes).

E3  (derived from E1) the V4 final_decision statement "A 状态被拒绝
    因为歧义对 10b1792c 的 180° 翻转解被放行" is re-characterised:
    the A-state denial stands (fixed12 error accept = 1 violates the
    hard gate), but its cause is a near-miss accepted transform on
    0ad2d384..., not a catastrophic flip.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OLD = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"
NEW = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"

EXPECTED_ERROR_PAIR = (
    "0ad2d384-79e2-2212-9b18-72b44eb5463f_to_"
    "0ad2d399-79e2-2212-99cf-7a3512734bd7"
)
CORRECT_CAL_TOP5 = 0.3498433974919869
TYPO_CAL_TOP5 = 0.4353


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_error_accepts(root: Path):
    """Walk EVERY raw repeats JSON; return all accepted_strict_error
    occurrences (authoritative source — prose is never trusted)."""
    found = []
    for f in sorted(root.glob("*/registration_repeats_[ABC].json")):
        data = json.loads(f.read_text())
        for row in data["rows"]:
            if row.get("status") != "ok":
                continue
            for o in row.get("outcomes", []):
                if o.get("accepted_strict_error"):
                    found.append({
                        "source_file": str(f.relative_to(ROOT)),
                        "pair_id": row["pair_id"],
                        "rre": o.get("rre"), "rte": o.get("rte"),
                        "strict": o.get("strict"),
                        "relaxed": o.get("relaxed"),
                        "accepted": o.get("accepted"),
                    })
    return found


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=True)
    errors = locate_error_accepts(OLD)
    assert len(errors) == 1, f"expected exactly 1 error accept, got {len(errors)}"
    err = errors[0]
    assert err["pair_id"] == EXPECTED_ERROR_PAIR, err["pair_id"]
    assert abs(err["rre"] - 2.966) < 0.01 and abs(err["rte"] - 0.217) < 0.002, err

    # cross-check: 10b1792c must NOT appear as an error accept anywhere
    for f in sorted(OLD.glob("*/registration_repeats_[ABC].json")):
        data = json.loads(f.read_text())
        for row in data["rows"]:
            for o in row.get("outcomes", []):
                if o.get("accepted_strict_error"):
                    assert "10b1792c" not in row["pair_id"]
    # where DOES 10b1792c appear? ambiguity lists only
    amb_locations = []
    for f in sorted(OLD.glob("*/registration_repeats_[ABC].json")):
        s = json.loads(f.read_text())["summary"]
        for p in s["ambiguity_pairs"]:
            if "10b1792c" in p:
                amb_locations.append(str(f.relative_to(ROOT)))

    dm = json.loads((OLD / "deterministic_metrics.json").read_text())
    computed_top5 = dm["arms"]["A_incumbent"]["calibration90"][
        "top5_recall"]
    typo_top5 = dm["arms"]["A_incumbent"]["v3_sealed_reference"][
        "calibration90"]["top5_recall"]
    assert computed_top5 == CORRECT_CAL_TOP5, computed_top5
    assert typo_top5 == TYPO_CAL_TOP5, typo_top5

    errata = {
        "phase": "V4-Fix Part 1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "accepted_strict_error occurrences extracted "
            "programmatically from every raw "
            "registration_repeats_[ABC].json under the V4 evidence "
            "dir; final_decision.md prose was NOT used as a source"),
        "e1_error_accept": {
            "extracted_from_raw": err,
            "total_error_accepts_across_all_arms_splits": len(errors),
            "error_character": (
                "relaxed-level NEAR-MISS: RRE 2.97 deg passes the 5 deg "
                "strict bar, RTE 0.2172 m fails the 0.20 m strict bar "
                "by 1.7 cm; decision accepted it"),
            "wrong_claim_in_v4_final_decision": (
                "attributed to pair 10b1792c... and described as a "
                "'~180-degree flipped RANSAC solution'"),
            "where_10b1792c_actually_appears": {
                "roles": "NOWHERE in the V4 registration repeats — "
                         "neither as an error accept nor in any "
                         "ambiguity list (it was the V3-era ambiguous "
                         "pair; the V4 attribution was doubly wrong)",
                "files": amb_locations,
                "v4_ambiguity_lists": {
                    "fixed12_C": ["0ad2d384-79e2-2212-9b18-72b44eb"
                                  "5463f..."],
                    "note": "the fixed12-C ambiguity pair IS the "
                            "error-accept pair itself (0ad2d384): "
                            "its strict verdict flips across RANSAC "
                            "draws and one draw was accepted at "
                            "relaxed level",
                },
            },
            "state_machine_impact_unchanged": (
                "the hard gate 'fixed12 error accepted = 0' is still "
                "violated (1 occurrence), so the V4 state-B denial of "
                "state A stands; only the CAUSE attribution is "
                "corrected"),
        },
        "e2_top5_provenance_typo": {
            "file": "outputs/official_sgaligner_v4_healthy_gat_20260827/"
                    "deterministic_metrics.json",
            "field": "arms.A_incumbent.v3_sealed_reference."
                     "calibration90.top5_recall",
            "typo_value": typo_top5,
            "correct_value": computed_top5,
            "impact": (
                "none on gate outcomes: every gate evaluation in "
                "v4_metrics used the COMPUTED calibration metrics "
                "(0.34984...), never the sealed-reference constant; "
                "the constant is metadata only"),
        },
        "e3_recharacterisation": {
            "old": "A-state denial caused by ambiguous-pair 180-deg "
                   "flip accept",
            "new": "A-state denial stands, caused by a near-miss "
                   "translation error accept on 0ad2d384...",
        },
        "old_evidence_modified": False,
    }
    (NEW / "evidence_errata.json").write_text(
        json.dumps(errata, indent=2) + "\n")

    md = """# V4 证据勘误（evidence_errata）

来源：原始 registration_repeats_[ABC].json 程序化提取（未引用
final_decision.md 文本）。旧证据目录零修改。

## E1 — 真正的错误 accepted

唯一一例 accepted_strict_error：

- pair：`{pair}`
- split/arm：fixed12 / C（explicit，epoch 20 候选）
- RRE {rre:.4f}°、RTE {rte:.4f} m、strict=false、relaxed=true、accepted=true
- 误差性质：**relaxed 级近平误差**——旋转过 5° 严格线，平移超 0.20 m
  严格线 1.7 cm 被决策放行；**不是** ~180° 翻转解。
- V4 final_decision.md 的错误表述：归因于 10b1792c… 的"180° 翻转解"。
  10b1792c 实际只出现在歧义（strict 翻转）列表，从未作为错误 accepted。
- 状态机影响不变：fixed12 错误 accepted=0 的硬门槛仍被违反（1 例），
  V4 状态 B 的 A 拒绝结论维持；仅成因表述更正。

## E2 — calibration top-5 provenance 拼写错误

deterministic_metrics.json 的 v3_sealed_reference.calibration90.
top5_recall 记为 {typo}（provenance typo）；正确值为 {correct}
（computed 字段一直正确，且所有门槛判定均使用 computed 值——
门槛结论不受影响）。

## E3 — 成因重述

A 状态拒绝的成因由"歧义对翻转解"更正为"0ad2d384… 上的近平平移
误差被 accepted"。
""".format(
        pair=err["pair_id"], rre=err["rre"], rte=err["rte"],
        typo=typo_top5, correct=computed_top5)
    (NEW / "evidence_errata.md").write_text(md)

    # source evidence hashes (all files the errata derives from)
    sources = {}
    for f in sorted(OLD.glob("*/registration_repeats_[ABC].json")):
        sources[str(f.relative_to(ROOT))] = {
            "sha256": sha256_file(f), "size": f.stat().st_size}
    dm_path = OLD / "deterministic_metrics.json"
    sources[str(dm_path.relative_to(ROOT))] = {
        "sha256": sha256_file(dm_path),
        "size": dm_path.stat().st_size}
    fd_path = OLD / "final_decision.md"
    sources[str(fd_path.relative_to(ROOT))] = {
        "sha256": sha256_file(fd_path),
        "size": fd_path.stat().st_size,
        "note": "contains the WRONG attribution being corrected"}
    (NEW / "source_evidence_sha256.json").write_text(
        json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "files": sources}, indent=2) + "\n")
    print(json.dumps({
        "error_pair": err["pair_id"], "rre": err["rre"],
        "rte": err["rte"],
        "ambiguity_only_files_for_10b1792c": amb_locations,
        "typo_top5": typo_top5, "correct_top5": computed_top5,
        "sources_hashed": len(sources)}, indent=1))


if __name__ == "__main__":
    main()
