"""Fix-1 arm C: legacy_geometry_baseline fresh run on the exact 12 pairs.

Runs the legacy frozen path (epoch-55 checkpoint + frozen D params +
segment ICP + RegistrationDecision) in the legacy environment via a
subprocess boundary, per pair, writing status.json next to its outputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
OUT = ROOT / sys.argv[1]
MANIFEST = OUT / "pair_manifest.txt"
LEGACY_ENV_PYTHON = "/home/aidenwu/miniconda3/envs/torch113/bin/python"

PAIR_ROOTS = [
    Path("/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
         "delivery_stage1_20260823/training_dataset/pairs"),
    Path("/home/aidenwu/Documents/inseg-sgaligner-v2/outputs/"
         "3rscan_native_graph_dataset_full/pairs"),
    Path("/home/aidenwu/Documents/inseg-sgaligner-v2/outputs/"
         "3rscan_native_graph_dataset_pilot/pairs"),
]

CKPT = ("/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/phase6_registration_aware_closure/"
        "training/epoch_00055.pt")

SCRIPT = r"""
import json
import numpy as np
import torch
from inseg_sgaligner.data import load_graph_pair
from inseg_sgaligner.matching import select_geometrically_consistent_matches
from inseg_sgaligner.registration import (
    ransac_registration, segment_aware_icp_refine, transform_errors,
)
from inseg_sgaligner.registration_decision import (
    compute_decision_features, evaluate_registration_decision,
)

import pathlib as _pl
pair_path = _pl.Path({pair_path!r})
out_dir = {out_dir!r}
ckpt = {ckpt!r}

from inseg_sgaligner.inference import graph_tensors, load_trained_model

pair = load_graph_pair(pair_path, require_anchors=False)
gt = pair.record.gt_transform
model, state, _ = load_trained_model(ckpt, "cpu")
with torch.no_grad():
    src_out, ref_out = model(
        graph_tensors(pair.src, "cpu"), graph_tensors(pair.ref, "cpu")
    )
src_emb = torch.nn.functional.normalize(src_out["graph"], dim=1)
ref_emb = torch.nn.functional.normalize(ref_out["graph"], dim=1)
sim = (src_emb @ ref_emb.T).cpu().numpy()
matches = select_geometrically_consistent_matches(
    sim, pair.src.node_centroids.astype(np.float64),
    pair.ref.node_centroids.astype(np.float64),
    max_matches_per_node=2, score_margin=0.08,
    minimum_similarity=0.60, distance_tolerance=0.05,
    relative_distance_tolerance=0.10, minimum_geometric_support=8,
)
status = {{"pair_id": pair_path.parent.name,
          "mode": "legacy_geometry_baseline"}}
if len(matches) < 3:
    status.update({{"status": "failed", "failed_stage": "matching",
                    "strict": False, "relaxed": False}})
else:
    src_idx = np.asarray([m[0] for m in matches])
    ref_idx = np.asarray([m[1] for m in matches])
    try:
        transform, inliers = ransac_registration(
            pair.src.node_centroids[src_idx].astype(np.float64),
            pair.ref.node_centroids[ref_idx].astype(np.float64),
            iterations=3000, inlier_threshold=0.12, seed=42,
        )
    except ValueError as exc:
        status.update({{"status": "failed", "failed_stage": "ransac",
                        "error": str(exc), "strict": False,
                        "relaxed": False}})
    else:
        src_all = pair.src.node_points_world.reshape(-1, 3).astype(np.float64)
        ref_all = pair.ref.node_points_world.reshape(-1, 3).astype(np.float64)
        src_labels = np.repeat(pair.src.node_labels,
                               pair.src.node_points_world.shape[1])
        ref_labels = np.repeat(pair.ref.node_labels,
                               pair.ref.node_points_world.shape[1])
        label_pairs = np.array([
            [pair.src.node_labels[s], pair.ref.node_labels[r]]
            for s, r in zip(src_idx, ref_idx)
        ])
        icp_conv = False
        try:
            transform, icp = segment_aware_icp_refine(
                src_all, ref_all, transform, label_pairs,
                src_labels=src_labels, ref_labels=ref_labels,
                distance_threshold=0.20, seed=42,
            )
            icp_conv = bool(icp.get("segment_icp_converged", False))
        except Exception:
            pass
        err = transform_errors(transform, gt)
        rre = float(err["rotation_error_degrees"])
        rte = float(err["translation_error_m"])
        strict = rre <= 5.0 and rte <= 0.20
        relaxed = rre <= 10.0 and rte <= 0.30
        match_dicts = [
            {{"src_label": int(pair.src.node_labels[s]),
              "ref_label": int(pair.ref.node_labels[r]),
              "ransac_inlier": bool(keep)}}
            for (s, r), keep in zip(zip(src_idx, ref_idx), inliers)
        ]
        try:
            feats = compute_decision_features(
                pair, match_dicts,
                {{"ransac_inliers": int(inliers.sum()),
                  "ransac_matches": len(matches)}},
                transform, ransac_threshold=0.12,
            )
            feats["icp_converged"] = icp_conv
            decision = evaluate_registration_decision(feats)
            accepted = bool(decision.get(
                "usable_for_reconstruction", False))
            reasons = decision.get("rejection_reasons", [])
        except Exception as exc:
            accepted, reasons = False, [f"decision_unavailable: {{exc}}"]
        anchor_set = {{(int(s), int(r))
                      for s, r in pair.anchor_indices.tolist()}}
        tp = sum(1 for s, r in zip(src_idx, ref_idx) if (s, r) in anchor_set)
        precision = tp / len(matches)
        recall = tp / max(len(anchor_set), 1)
        status.update({{
            "status": "ok", "strict": strict, "relaxed": relaxed,
            "rre": rre, "rte": rte, "inliers": int(inliers.sum()),
            "matches": len(matches),
            "node_precision": precision, "node_recall": recall,
            "node_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "accepted": accepted, "rejection_reasons": reasons,
            "icp_converged": icp_conv,
        }})
import pathlib
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(out_dir, "status.json").write_text(json.dumps(status, indent=2))
print("LEGACY_OK" if status["status"] == "ok" else "LEGACY_FAILED")
"""


def find_pair(pair_id: str) -> Path:
    for root in PAIR_ROOTS:
        path = root / pair_id / "pair.json"
        if path.exists():
            return path
    raise FileNotFoundError(pair_id)


def main() -> None:
    pairs = [
        line.strip()
        for line in MANIFEST.read_text().splitlines() if line.strip()
    ]
    assert len(pairs) == 12, len(pairs)
    arm_root = OUT / "legacy_arm"
    arm_root.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        tag = f"legacy_{pair[:8]}_{pair[-4:]}"
        out_dir = arm_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        script = SCRIPT.format(
            pair_path=str(find_pair(pair)),
            out_dir=str(out_dir),
            ckpt=CKPT,
        )
        result = subprocess.run(
            [LEGACY_ENV_PYTHON, "-c", script],
            capture_output=True, text=True,
            cwd="/home/aidenwu/Documents/inseg-sgaligner-sgf-context-v1",
            env={"PATH": "/home/aidenwu/miniconda3/envs/torch113/bin:/usr/bin:/bin",
                 "OMP_NUM_THREADS": "1"},
            timeout=1800,
        )
        marker = "LEGACY_OK" if result.returncode == 0 else "LEGACY_ERROR"
        if result.returncode != 0:
            (out_dir / "stderr.txt").write_text(result.stderr[-2000:])
            (out_dir / "status.json").write_text(json.dumps({
                "pair_id": pair, "mode": "legacy_geometry_baseline",
                "status": "failed", "failed_stage": "legacy_subprocess",
                "error": result.stderr.strip().splitlines()[-1][:200]
                if result.stderr.strip() else "unknown",
            }, indent=2))
        print(marker, tag, flush=True)


if __name__ == "__main__":
    main()
