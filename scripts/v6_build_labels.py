"""V6 label audit runner: builds SGF cross-scan node labels for
train437 with the pre-registered thresholds and writes the audit
(evidence dir label_audit/)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
for p in (str(ROOT), str(ROOT / "src"),
          str(ROOT / "src/inference/sgf_official"),
          str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sgf_node_labels import label_pair, audit  # noqa: E402
from adapters.sgf.data_sources import (  # noqa: E402
    PredictedGraphSource, load_pair_record,
)

OUT = ROOT / ("outputs/official_sgaligner_v6_sgf_domain_matcher_"
              "20260829")


def main() -> None:
    predicted = PredictedGraphSource()
    man = json.loads(Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/"
        "dataset_three_way.json").read_text())
    pair_root = Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/pairs")
    pairs = [pair_root / Path(r).parent.name
             for r in man["train_pairs"]]

    audit_dir = OUT / "label_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        "pairs_processed": 0, "pairs_no_positive": [],
        "examples": [], "per_pair": []}
    agg = {"positive": 0, "negative": 0, "ambiguous": 0,
           "hard_negative": 0, "split_sources": 0,
           "merged_refs": 0, "one_to_one": 0}
    for index, pair_dir in enumerate(pairs):
        pair_id = pair_dir.name
        try:
            src_pred = predicted.load(pair_id.split("_to_")[0])
            ref_pred = predicted.load(pair_id.split("_to_")[1])
            payload = load_pair_record(pair_id)
            gt = np.asarray(
                payload["gt_transform"], dtype=np.float64
            ).reshape(4, 4)
            stats = label_pair(
                src_pred.segments, ref_pred.segments, gt)
            a = audit(stats)
            a["pair_id"] = pair_id
            totals["per_pair"].append(a)
            for k in ("positive", "negative", "ambiguous",
                      "hard_negative", "split_sources",
                      "merged_refs"):
                agg[k] += a[k]
            if a["positive"] == 0:
                totals["pairs_no_positive"].append(pair_id)
            if len(totals["examples"]) < 20 and a["positive"] >= 2:
                totals["examples"].append({
                    "pair_id": pair_id, "summary": a,
                    "top_positive_pairs": [
                        {"src": s.src, "ref": s.ref,
                         "bidir10": round(s.bidir_10, 3),
                         "unidir10": round(s.unidir_10, 3),
                         "iou": round(s.voxel_iou, 3),
                         "centroid_res": round(
                             s.centroid_residual, 3),
                         "extent_ratio": round(s.extent_ratio, 3),
                         "semantic": s.semantic_conf}
                        for s in sorted(
                            stats, key=lambda x: -x.bidir_10)[:4]]})
            totals["pairs_processed"] += 1
        except Exception as exc:  # noqa: BLE001 — recorded
            totals.setdefault("errors", []).append(
                {"pair_id": pair_id, "error": repr(exc)[:150]})
        if (index + 1) % 50 == 0:
            print(f"labeled {index+1}/{len(pairs)}", flush=True)
    (audit_dir / "train437_label_audit.json").write_text(
        json.dumps({"aggregate": agg, **totals}, indent=2) + "\n")
    print(json.dumps(agg, indent=1))
    print("no-positive pairs:", len(totals["pairs_no_positive"]))


if __name__ == "__main__":
    main()
