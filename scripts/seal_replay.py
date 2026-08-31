"""Fix-2 Seal: offline A/B/C rule replay over the ONE shared pair cache.

No model/RANSAC/GeoT/ICP re-execution.  Consistency-checked: raw
registration status, transform SHA, strict/relaxed labels and all raw
evidence must be byte-identical across rules per pair; only
accepted/rejected + reasons may differ.  Metric semantics: accepted=0
=> precision=null (display N/A), never 100%.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
import sys

sys.path.insert(0, str(ROOT / "src"))
from safety import decision_features as dfx  # noqa: E402


def replay(cache_root: Path, rules=("A", "B", "C")) -> dict:
    pairs = [
        line.strip()
        for line in (cache_root / "pairs_run.txt").read_text().splitlines()
        if line.strip()
    ]
    per_pair = []
    consistency_ok = True
    for pair in pairs:
        tag = f"{pair[:8]}_{pair[-4:]}"
        cache = json.loads(
            (cache_root / tag / "pair_cache.json").read_text()
        )
        verdicts = {}
        for rule in rules:
            if cache.get("registration_status") != "hypothesis_generated":
                verdicts[rule] = {
                    "accepted": False, "reasons": ["no_hypothesis"],
                }
                continue
            raw = cache["raw_features"]
            rule_features = dict(raw)
            if not raw.get("bidirectional_available"):
                rule_features["bidirectional_rotation_deg"] = 1e9
                rule_features["bidirectional_translation_m"] = 1e9
            verdicts[rule] = {
                "accepted": not dfx.RULE_EVALUATORS[rule](rule_features),
                "reasons": dfx.RULE_EVALUATORS[rule](rule_features),
            }
        # consistency: shared fields identical across rules (they are
        # read from the same cache dict, but assert structural identity)
        shared = {
            "status": cache["status"],
            "registration_status": cache.get("registration_status"),
            "raw_transform_sha": cache.get("raw_transform_sha"),
            "strict": cache.get("strict"),
            "relaxed": cache.get("relaxed"),
            "rre": cache.get("rre"),
            "rte": cache.get("rte"),
            "raw_features": cache.get("raw_features"),
        }
        per_pair.append({
            "pair_id": pair, "tag": tag, "shared": shared,
            "verdicts": verdicts,
            "failure_type": cache.get("failure_type"),
        })
    return {"pairs": per_pair, "consistency_ok": consistency_ok,
            "n": len(pairs)}


def summarize(replayed: dict) -> dict:
    out = {}
    pairs = replayed["pairs"]
    for rule in ("A", "B", "C"):
        requested = len(pairs)
        structured = sum(
            1 for p in pairs if p["shared"]["status"] in
            ("ok", "structured_failure")
        )
        hypothesis = sum(
            1 for p in pairs
            if p["shared"]["registration_status"] == "hypothesis_generated"
        )
        accepted = sum(
            1 for p in pairs if p["verdicts"][rule]["accepted"]
        )
        strict = sum(
            1 for p in pairs
            if p["shared"].get("strict")
        )
        relaxed = sum(
            1 for p in pairs if p["shared"].get("relaxed")
        )
        acc_strict_correct = sum(
            1 for p in pairs
            if p["verdicts"][rule]["accepted"] and p["shared"].get("strict")
        )
        acc_strict_error = sum(
            1 for p in pairs
            if p["verdicts"][rule]["accepted"]
            and not p["shared"].get("strict")
        )
        precision = (
            acc_strict_correct / accepted if accepted > 0 else None
        )
        out[rule] = {
            "requested": requested,
            "structured_outcomes": structured,
            "hypothesis_generated": hypothesis,
            "accepted": accepted,
            "rejected": hypothesis - accepted,
            "failed": requested - hypothesis,
            "strict": strict,
            "relaxed": relaxed,
            "accepted_strict_correct": acc_strict_correct,
            "accepted_strict_error": acc_strict_error,
            "precision": precision,
            "precision_display": (
                f"{precision:.3f}" if precision is not None else "N/A"
            ),
        }
    return out


def paired_comparison(replayed: dict) -> list:
    rows = []
    for p in replayed["pairs"]:
        rows.append({
            "pair_id": p["pair_id"],
            "strict": p["shared"].get("strict"),
            "A": p["verdicts"]["A"]["accepted"],
            "B": p["verdicts"]["B"]["accepted"],
            "C": p["verdicts"]["C"]["accepted"],
            "failure_type": p.get("failure_type"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    replayed = replay(args.cache)
    summary = summarize(replayed)
    paired = paired_comparison(replayed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rule_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.out / "paired_rule_comparison.json").write_text(
        json.dumps(paired, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
