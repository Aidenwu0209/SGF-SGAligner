"""V4 checkpoint selection: pre-registered deterministic lexicographic
key over selection89 eval history (macro F1, top-1, top-5, margin,
earlier epoch). Verifies PCT/rel frozen-hash invariance between the
initialization audit and the chosen checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"


def sha16(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("complete", "explicit"),
                        required=True)
    args = parser.parse_args()
    arm_dir = OUT / "training" / args.arm
    history = [
        json.loads(line) for line in
        (arm_dir / "history.jsonl").read_text().splitlines() if line.strip()
    ]
    evals = [h for h in history if h.get("kind") == "eval"]
    assert evals, "no eval records"

    def key(rec):
        m = rec["metrics"]
        return (m["macro_node_f1"], m["top1_precision"],
                m["top5_recall"], m["margin"])

    best = max(evals, key=lambda r: (key(r), -r["epoch"]))
    # verify uniqueness of the winning key
    winners = [r for r in evals if key(r) == key(best)]
    assert len(winners) == 1 or all(
        r["epoch"] == best["epoch"] for r in winners), \
        "lexicographic key tie across epochs"

    ckpt_path = arm_dir / f"epoch_{best['epoch']:05d}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    init_audit = json.loads(
        (OUT / f"initialization_audit_{args.arm}.json").read_text())
    state = ckpt["model"]
    frozen_now = {}
    for name, tensor in state.items():
        if name.startswith((
                "object_encoder", "object_embedding", "meta_embedding_rel")):
            frozen_now[name] = sha16(tensor)
    frozen_before = init_audit["frozen_tensor_hashes"]
    mismatched = [
        k for k, v in frozen_before.items()
        if not k.startswith("fusion.row") and frozen_now.get(k) != v]
    # fusion rows 0/2 frozen values
    fusion_w = state["fusion.weight"]
    if frozen_before.get("fusion.row0") != sha16(fusion_w[0]) \
            or frozen_before.get("fusion.row2") != sha16(fusion_w[2]):
        mismatched.append("fusion.rows(0,2)")

    gat_params = torch.cat([
        state[k].flatten() for k in sorted(state)
        if k.startswith("structure_encoder")])
    subnormal = float((
        (gat_params.abs() > 0)
        & (gat_params.abs() < torch.finfo(torch.float32).tiny)
        ).float().mean())

    result = {
        "arm": args.arm,
        "selected_checkpoint": str(ckpt_path.relative_to(ROOT)),
        "selected_epoch": best["epoch"],
        "selection_metrics": best["metrics"],
        "all_evals": [
            {"epoch": r["epoch"], **r["metrics"]} for r in evals],
        "lexicographic_key": ["macro_node_f1", "top1_precision",
                              "top5_recall", "margin", "earlier_epoch"],
        "pct_frozen_hashes_match": len(mismatched) == 0,
        "frozen_mismatches": mismatched,
        "gat_subnormal_fraction_at_selected": subnormal,
        "model_naming": (
            "official-architecture SGF-predicted healthy-GAT research "
            "candidate"),
    }
    sel_dir = OUT / "checkpoint_selection"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / f"{args.arm}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "arm": args.arm, "epoch": best["epoch"],
        "metrics": best["metrics"],
        "frozen_ok": result["pct_frozen_hashes_match"],
        "subnormal": subnormal,
    }, indent=1))


if __name__ == "__main__":
    main()
