"""V5 wrap-up: paired baseline comparison + checkpoint inventory."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / "outputs/official_sgaligner_v5_relation_gat_20260828"


def main() -> None:
    reg = json.loads(
        (OUT / "selection_registration_metrics.json").read_text())
    node = json.loads(
        (OUT / "selection_node_metrics.json").read_text())

    base_rows = reg["runs"]["A_baseline_Cep25"]["rows"]
    base_by_pair = {r["pair_id"]: r for r in base_rows}

    paired = {"selection89": {}}
    for label in ("B_ep10", "B_ep5", "C_ep5"):
        rows = reg["runs"][label]["rows"]
        strict_flip_pos = strict_flip_neg = 0
        acc_pos = acc_neg = 0
        for r in rows:
            b = base_by_pair[r["pair_id"]]
            if r.get("strict") and not b.get("strict"):
                strict_flip_pos += 1
            if b.get("strict") and not r.get("strict"):
                strict_flip_neg += 1
            if r.get("outcome") == "accepted_strict_correct" \
                    and b.get("outcome") != \
                    "accepted_strict_correct":
                acc_pos += 1
            if b.get("outcome") == "accepted_strict_correct" \
                    and r.get("outcome") != \
                    "accepted_strict_correct":
                acc_neg += 1
        paired["selection89"][label] = {
            "vs_A_baseline": {
                "strict_new_wins": strict_flip_pos,
                "strict_losses": strict_flip_neg,
                "accepted_correct_new_wins": acc_pos,
                "accepted_correct_losses": acc_neg,
            }}
    (OUT / "paired_baseline_comparison.json").write_text(
        json.dumps(paired, indent=2) + "\n")

    inventory = {"checkpoints": []}
    for arm in ("B", "C"):
        d = OUT / "training" / arm
        for f in sorted(d.glob("epoch_*.pt")):
            rank_row = next(
                (r for r in node["arms"][arm]
                 if r["epoch"] == int(f.stem.split("_")[1])), None)
            inventory["checkpoints"].append({
                "arm": arm,
                "epoch": int(f.stem.split("_")[1]),
                "path": str(f.relative_to(ROOT)),
                "sha256": hashlib.sha256(
                    f.read_bytes()).hexdigest(),
                "macro_node_f1": (
                    rank_row["metrics"]["macro_node_f1"]
                    if rank_row else None),
                "node_rank": (
                    rank_row["rank"] if rank_row else None),
            })
    inventory["total"] = len(inventory["checkpoints"])
    (OUT / "checkpoint_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n")

    # frozen tensor audit (after-training hash check vs init audit)
    import torch

    for arm in ("B", "C"):
        init = json.loads(
            (OUT / f"frozen_tensor_audit_{arm}.json").read_text())
        best = torch.load(
            OUT / "training" / arm / "epoch_00005.pt",
            map_location="cpu", weights_only=False)
        after = {}
        state = best["model"]
        for name, tensor in state.items():
            if name.startswith(
                    ("object_encoder", "object_embedding",
                     "structure_encoder", "structure_embedding")):
                after[name] = hashlib.sha256(
                    tensor.numpy().tobytes()).hexdigest()[:16]
        # fusion frozen rows per arm
        frozen_rows = [0] + ([1] if arm == "B" else [])
        for i in frozen_rows:
            after[f"fusion.row{i}"] = hashlib.sha256(
                state["fusion.weight"][i].numpy().tobytes()
            ).hexdigest()[:16]
        mismatch = [
            k for k, v in init["frozen_tensor_hashes"].items()
            if after.get(k) != v]
        audit = json.loads(
            (OUT / f"frozen_tensor_audit_{arm}.json").read_text())
        audit["after_training_hashes"] = after
        audit["frozen_unchanged_through_training"] = (
            len(mismatch) == 0)
        audit["mismatches"] = mismatch
        (OUT / f"frozen_tensor_audit_{arm}.json").write_text(
            json.dumps(audit, indent=2) + "\n")
        print(arm, "frozen unchanged:", len(mismatch) == 0)
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
