"""Fix-1 evidence collector + manifest; fail-closed on manifest drift."""
import hashlib, json, sys
from pathlib import Path
from collections import Counter

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / sys.argv[1]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

manifest_pairs = [
    l.strip() for l in (OUT / "pair_manifest.txt").read_text().splitlines()
    if l.strip()
]
assert len(manifest_pairs) == 12, len(manifest_pairs)

def arm(pattern):
    rows = []
    for pair in manifest_pairs:
        tag = f"{pattern}_{pair[:8]}_{pair[-4:]}"
        status = OUT / tag / "status.json"
        if not status.exists():
            raise FileNotFoundError(f"missing result for {tag}")
        rows.append((pair, json.loads(status.read_text())))
    return rows

summary = {}
file_hashes = {}
for name, pattern in (
    ("official_oracle", "official_oracle"),
    ("official_sgf_predicted", "official_sgf_predicted"),
):
    rows = arm(pattern)
    ok = [r for _p, r in rows if r["status"] == "ok"]
    summary[name] = {
        "n": len(rows),
        "pair_ids": [p for p, _r in rows],
        "ok": len(ok),
        "strict": sum(1 for r in ok if r.get("strict")),
        "relaxed": sum(1 for r in ok if r.get("relaxed")),
        "accepted": sum(1 for r in ok if r.get("accepted")),
        "accepted_strict_errors": sum(
            1 for r in ok if r.get("accepted") and not r.get("strict")
        ),
        "mean_node_f1": round(
            sum(r.get("node_f1") or 0 for _p, r in rows) / len(rows), 4
        ),
        "statuses": dict(Counter(r["status"] for _p, r in rows)),
    }
    for pair, _r in rows:
        tag = f"{pattern}_{pair[:8]}_{pair[-4:]}"
        file_hashes[f"{tag}/status.json"] = sha256(OUT / tag / "status.json")

legacy_root = OUT / "legacy_arm"
rows = []
for pair in manifest_pairs:
    tag = f"legacy_{pair[:8]}_{pair[-4:]}"
    status = legacy_root / tag / "status.json"
    if not status.exists():
        raise FileNotFoundError(f"missing legacy result {tag}")
    rows.append((pair, json.loads(status.read_text())))
    file_hashes[f"legacy_arm/{tag}/status.json"] = sha256(status)
ok = [r for _p, r in rows if r["status"] == "ok"]
summary["legacy_geometry_baseline"] = {
    "n": len(rows),
    "pair_ids": [p for p, _r in rows],
    "ok": len(ok),
    "strict": sum(1 for r in ok if r.get("strict")),
    "relaxed": sum(1 for r in ok if r.get("relaxed")),
    "accepted": sum(1 for r in ok if r.get("accepted")),
    "accepted_strict_errors": sum(
        1 for r in ok if r.get("accepted") and not r.get("strict")
    ),
    "statuses": dict(Counter(r["status"] for _p, r in rows)),
}

# predicted failure taxonomy from per-pair failure.json + reg results
taxonomy = Counter()
for pair in manifest_pairs:
    tag = f"official_sgf_predicted_{pair[:8]}_{pair[-4:]}"
    fail = OUT / tag / "failure.json"
    reg = OUT / tag / "registration_result.json"
    if fail.exists():
        d = json.loads(fail.read_text())
        for k, v in (d.get("failure_stage_counts") or {}).items():
            taxonomy[f"pair:{k}"] += v
        if not d.get("failure_stage_counts"):
            taxonomy[f"pair:{d.get('stage', '?')}"] += 1
    if reg.exists():
        d = json.loads(reg.read_text())
        for nf in d.get("node_pair_failures", []):
            taxonomy[f"nodepair:{nf['stage']}"] += 1
summary["predicted_failure_taxonomy"] = dict(taxonomy)

summary["pair_manifest_sha256"] = sha256(OUT / "pair_manifest.txt")
summary["result_file_sha256"] = file_hashes
summary["differential"] = json.loads(
    (OUT / "differential_validation.json").read_text()
) if (OUT / "differential_validation.json").exists() else None

(OUT / "fix1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
slim = {
    arm: {
        k: v for k, v in data.items()
        if k not in ("pair_ids",)
    } for arm, data in summary.items()
    if isinstance(data, dict)
}
print(json.dumps(slim, indent=1))
